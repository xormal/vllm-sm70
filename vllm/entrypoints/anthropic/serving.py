# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/vllm/vllm/entrypoints/openai/serving_chat.py

"""Anthropic Messages API serving handler"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicDelta,
    AnthropicError,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    StreamOptions,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels

logger = logging.getLogger(__name__)


def wrap_data_with_event(data: str, event: str):
    return f"event: {event}\ndata: {data}\n\n"


def _missing_required_claude_tool_args(
    tool_name: str,
    parsed_args: dict[str, Any],
) -> list[str]:
    required_by_tool = {
        "Bash": ("command",),
        "Write": ("file_path", "content"),
        "Edit": ("file_path", "old_string", "new_string"),
        "MultiEdit": ("file_path", "edits"),
        "Read": ("file_path",),
        "Glob": ("pattern",),
        "Grep": ("pattern",),
        "LS": ("path",),
        "NotebookRead": ("notebook_path",),
        "NotebookEdit": ("notebook_path", "cell_id", "new_source"),
        "WebFetch": ("url", "prompt"),
        "TodoWrite": ("todos",),
    }
    required = required_by_tool.get(tool_name, ())
    missing = [
        key
        for key in required
        if key not in parsed_args or parsed_args[key] is None
    ]
    command = parsed_args.get("command")
    if tool_name == "Bash" and (
        not isinstance(command, str) or not command.strip()
    ):
        if "command" not in missing:
            missing.append("command")
    return missing


class AnthropicServingMessages(OpenAIServingChat):
    """Handler for Anthropic Messages API requests"""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            response_role=response_role,
            request_logger=request_logger,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            reasoning_parser=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            tool_parser=tool_parser,
            enable_prompt_tokens_details=enable_prompt_tokens_details,
            enable_force_include_usage=enable_force_include_usage,
        )
        self.stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }

    def _convert_anthropic_to_openai_request(
        self, anthropic_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """Convert Anthropic message format to OpenAI format"""
        openai_messages = []
        valid_tool_call_ids: set[str] = set()
        dropped_tool_call_ids: set[str] = set()

        # Add system message if provided
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                openai_messages.append(
                    {"role": "system", "content": anthropic_request.system}
                )
            else:
                system_prompt = ""
                for block in anthropic_request.system:
                    if block.type == "text" and block.text:
                        system_prompt += block.text
                openai_messages.append({"role": "system", "content": system_prompt})

        for msg in anthropic_request.messages:
            openai_msg: dict[str, Any] = {"role": msg.role}  # type: ignore
            if isinstance(msg.content, str):
                openai_msg["content"] = msg.content
            else:
                # Handle complex content blocks
                content_parts: list[dict[str, Any]] = []
                tool_calls: list[dict[str, Any]] = []

                for block in msg.content:
                    if block.type == "text" and block.text:
                        content_parts.append({"type": "text", "text": block.text})
                    elif block.type == "image" and block.source:
                        image_url = block.source.get("url") or block.source.get("data", "")
                        if block.source.get("type") == "base64" and image_url:
                            media_type = block.source.get("media_type", "image/png")
                            image_url = f"data:{media_type};base64,{image_url}"
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            }
                        )
                    elif block.type == "tool_use":
                        # Convert tool use to function call format
                        tool_name = block.name or ""
                        tool_args = block.input or {}
                        if not isinstance(tool_args, dict):
                            if block.id:
                                dropped_tool_call_ids.add(block.id)
                            logger.warning(
                                "Dropping historical tool_use with non-object input "
                                "(id=%s name=%s type=%s)",
                                block.id,
                                tool_name,
                                type(tool_args).__name__,
                            )
                            continue
                        missing_args = _missing_required_claude_tool_args(
                            tool_name,
                            tool_args,
                        )
                        if missing_args:
                            if block.id:
                                dropped_tool_call_ids.add(block.id)
                            logger.warning(
                                "Dropping historical incomplete tool_use "
                                "(id=%s name=%s missing=%s)",
                                block.id,
                                tool_name,
                                missing_args,
                            )
                            continue
                        if block.id:
                            valid_tool_call_ids.add(block.id)
                        tool_call = {
                            "id": block.id or f"call_{int(time.time())}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args),
                            },
                        }
                        tool_calls.append(tool_call)
                    elif block.type == "tool_result":
                        if msg.role == "user":
                            tool_call_id = block.tool_use_id or block.id or ""
                            if (
                                tool_call_id in dropped_tool_call_ids
                                or tool_call_id not in valid_tool_call_ids
                            ):
                                logger.warning(
                                    "Dropping historical orphaned tool_result "
                                    "(tool_use_id=%s)",
                                    tool_call_id,
                                )
                                continue
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": str(block.content)
                                    if block.content
                                    else "",
                                }
                            )
                        else:
                            # Assistant tool result becomes regular text
                            tool_result_text = (
                                str(block.content) if block.content else ""
                            )
                            content_parts.append(
                                {
                                    "type": "text",
                                    "text": f"Tool result: {tool_result_text}",
                                }
                            )

                # Add tool calls to the message if any
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls  # type: ignore

                # Add content parts if any
                if content_parts:
                    if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                        openai_msg["content"] = content_parts[0]["text"]
                    else:
                        openai_msg["content"] = content_parts  # type: ignore
                elif not tool_calls:
                    continue

            openai_messages.append(openai_msg)

        req = ChatCompletionRequest(
            model=anthropic_request.model,
            messages=openai_messages,
            max_tokens=anthropic_request.max_tokens,
            max_completion_tokens=anthropic_request.max_tokens,
            stop=anthropic_request.stop_sequences,
            temperature=anthropic_request.temperature,
            top_p=anthropic_request.top_p,
            top_k=anthropic_request.top_k,
            min_p=anthropic_request.min_p,
            min_tokens=anthropic_request.min_tokens or 0,
            ignore_eos=bool(anthropic_request.ignore_eos),
            presence_penalty=anthropic_request.presence_penalty,
            repetition_penalty=anthropic_request.repetition_penalty,
        )

        if anthropic_request.stream:
            req.stream = anthropic_request.stream
            req.stream_options = StreamOptions.validate(
                {"include_usage": True, "continuous_usage_stats": True}
            )

        if anthropic_request.tool_choice is None:
            req.tool_choice = None
        elif anthropic_request.tool_choice.type == "auto":
            req.tool_choice = "auto"
        elif anthropic_request.tool_choice.type == "any":
            req.tool_choice = "required"
        elif anthropic_request.tool_choice.type == "tool":
            req.tool_choice = ChatCompletionNamedToolChoiceParam.model_validate(
                {
                    "type": "function",
                    "function": {"name": anthropic_request.tool_choice.name},
                }
            )

        tools = []
        if anthropic_request.tools is None:
            return req
        for tool in anthropic_request.tools:
            tools.append(
                ChatCompletionToolsParam.model_validate(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
                    }
                )
            )
        if req.tool_choice is None:
            req.tool_choice = "auto"
        req.tools = tools
        return req

    async def create_messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | AnthropicMessagesResponse | ErrorResponse:
        """
        Messages API similar to Anthropic's API.

        See https://docs.anthropic.com/en/api/messages
        for the API specification. This API mimics the Anthropic messages API.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Received messages request %s", request.model_dump_json())
        chat_req = self._convert_anthropic_to_openai_request(request)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Convert to OpenAI request %s", chat_req.model_dump_json())
        generator = await self.create_chat_completion(chat_req, raw_request)

        if isinstance(generator, ErrorResponse):
            return generator

        elif isinstance(generator, ChatCompletionResponse):
            return self.messages_full_converter(generator)

        return self.message_stream_converter(generator)

    def messages_full_converter(
        self,
        generator: ChatCompletionResponse,
    ) -> AnthropicMessagesResponse:
        def _missing_required_tool_args(
            tool_name: str,
            parsed_args: dict[str, Any],
        ) -> list[str]:
            required_by_tool = {
                "Bash": ("command",),
                "Write": ("file_path", "content"),
                "Edit": ("file_path", "old_string", "new_string"),
                "MultiEdit": ("file_path", "edits"),
                "Read": ("file_path",),
                "Glob": ("pattern",),
                "Grep": ("pattern",),
                "LS": ("path",),
                "NotebookRead": ("notebook_path",),
                "NotebookEdit": ("notebook_path", "cell_id", "new_source"),
                "WebFetch": ("url", "prompt"),
                "TodoWrite": ("todos",),
            }
            required = required_by_tool.get(tool_name, ())
            missing = [
                key
                for key in required
                if key not in parsed_args or parsed_args[key] is None
            ]
            command = parsed_args.get("command")
            if tool_name == "Bash" and (
                not isinstance(command, str) or not command.strip()
            ):
                if "command" not in missing:
                    missing.append("command")
            return missing

        result = AnthropicMessagesResponse(
            id=generator.id,
            content=[],
            model=generator.model,
            usage=AnthropicUsage(
                input_tokens=generator.usage.prompt_tokens,
                output_tokens=generator.usage.completion_tokens,
            ),
        )
        if generator.choices[0].finish_reason == "stop":
            result.stop_reason = "end_turn"
        elif generator.choices[0].finish_reason == "length":
            result.stop_reason = "max_tokens"
        elif generator.choices[0].finish_reason == "tool_calls":
            result.stop_reason = "tool_use"

        message = generator.choices[0].message
        has_tool_calls = bool(message.tool_calls)
        message_text = message.content or ""
        content: list[AnthropicContentBlock] = []

        # Tool-call parsers return any text before <tool_call> as message.content.
        # Qwen reasoning can leak there even with thinking disabled; Anthropic
        # coding clients expect a clean tool_use block they can replay as-is.
        if message_text and not has_tool_calls:
            content.append(AnthropicContentBlock(type="text", text=message_text))

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name if tool_call.function else None
            if not tool_call.id or not tool_name:
                logger.warning(
                    "Dropping full-response tool_call without id/name "
                    "(id=%s name=%s)",
                    tool_call.id,
                    tool_name,
                )
                continue
            try:
                parsed_args = json.loads(
                    tool_call.function.arguments or "{}",
                    strict=False,
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning(
                    "Dropping invalid full-response tool_call args "
                    "(id=%s name=%s len=%d preview=%r tail=%r)",
                    tool_call.id,
                    tool_name,
                    len(tool_call.function.arguments or ""),
                    (tool_call.function.arguments or "")[:240],
                    (tool_call.function.arguments or "")[-240:],
                )
                continue
            if not isinstance(parsed_args, dict):
                logger.warning(
                    "Dropping non-object full-response tool_call args "
                    "(id=%s name=%s type=%s)",
                    tool_call.id,
                    tool_name,
                    type(parsed_args).__name__,
                )
                continue
            missing_args = _missing_required_tool_args(tool_name, parsed_args)
            if missing_args:
                logger.warning(
                    "Dropping incomplete full-response tool_call args "
                    "(id=%s name=%s missing=%s len=%d preview=%r tail=%r)",
                    tool_call.id,
                    tool_name,
                    missing_args,
                    len(tool_call.function.arguments or ""),
                    (tool_call.function.arguments or "")[:240],
                    (tool_call.function.arguments or "")[-240:],
                )
                continue
            anthropic_tool_call = AnthropicContentBlock(
                type="tool_use",
                id=tool_call.id,
                name=tool_name,
                input=parsed_args,
            )
            content += [anthropic_tool_call]

        emitted_tool_calls = any(block.type == "tool_use" for block in content)
        if emitted_tool_calls:
            result.stop_reason = "tool_use"
        elif has_tool_calls:
            result.stop_reason = "end_turn"

        result.content = content

        return result

    async def message_stream_converter(
        self,
        generator: AsyncGenerator[str, None],
    ) -> AsyncGenerator[str, None]:
        def _find_json_string_start(raw: str, key: str) -> int | None:
            key_pos = raw.find(json.dumps(key))
            if key_pos < 0:
                return None
            colon_pos = raw.find(":", key_pos + len(key) + 2)
            if colon_pos < 0:
                return None
            quote_pos = raw.find('"', colon_pos + 1)
            if quote_pos < 0:
                return None
            return quote_pos + 1

        def _read_json_string_value(raw: str, key: str) -> str | None:
            start = _find_json_string_start(raw, key)
            if start is None:
                return None

            escaped = False
            chars: list[str] = []
            for char in raw[start:]:
                if escaped:
                    chars.append("\\" + char)
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    try:
                        return json.loads('"' + "".join(chars) + '"', strict=False)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        return "".join(chars)
                chars.append(char)
            return None

        def _decode_json_string_escapes(value: str) -> str:
            """Decode JSON escapes in a possibly malformed string fragment.

            `json.loads()` cannot be used for recovered Write.content when the
            model inserted unescaped quotes into the file body. Decode only JSON
            escape sequences and preserve all other characters verbatim.
            """
            decoded: list[str] = []
            index = 0
            while index < len(value):
                char = value[index]
                if char != "\\" or index + 1 >= len(value):
                    decoded.append(char)
                    index += 1
                    continue

                escaped = value[index + 1]
                if escaped == "n":
                    decoded.append("\n")
                    index += 2
                elif escaped == "r":
                    decoded.append("\r")
                    index += 2
                elif escaped == "t":
                    decoded.append("\t")
                    index += 2
                elif escaped == "b":
                    decoded.append("\b")
                    index += 2
                elif escaped == "f":
                    decoded.append("\f")
                    index += 2
                elif escaped in {'"', "\\", "/"}:
                    decoded.append(escaped)
                    index += 2
                elif escaped == "u" and index + 5 < len(value):
                    hex_value = value[index + 2:index + 6]
                    try:
                        decoded.append(chr(int(hex_value, 16)))
                        index += 6
                    except ValueError:
                        decoded.append("\\" + escaped)
                        index += 2
                else:
                    decoded.append("\\" + escaped)
                    index += 2
            return "".join(decoded)

        def _recover_unclosed_write_args(raw: str) -> dict[str, Any] | None:
            """Recover Qwen Write calls where only the content JSON string is unclosed.

            Qwen-family models sometimes emit the Write tool as valid structure up
            to `"content":"...`, then include raw file text with unescaped quotes or
            newlines. The Anthropic client can still execute the tool safely if we
            preserve the intended file_path and treat the remaining payload as
            content.
            """
            file_path = _read_json_string_value(raw, "file_path")
            content_start = _find_json_string_start(raw, "content")
            if not file_path or content_start is None:
                return None

            content = raw[content_start:]
            end_marker = "</tool_call>"
            if end_marker in content:
                content = content.split(end_marker, 1)[0]
            content = content.rstrip()
            for suffix in ('"}', '"}', '"'):
                if content.endswith(suffix):
                    content = content[: -len(suffix)]
                    break
            content = _decode_json_string_escapes(content)
            return {"file_path": file_path, "content": content}

        def _parse_tool_arguments(
            tool_name: str,
            raw_args: str,
        ) -> dict[str, Any] | None:
            raw_args = raw_args or "{}"
            try:
                parsed = json.loads(raw_args, strict=False)
            except (json.JSONDecodeError, TypeError, ValueError):
                if tool_name == "Write":
                    return _recover_unclosed_write_args(raw_args)
                return None

            if isinstance(parsed, dict):
                # Some llama.cpp-style generations leak the whole wrapper into
                # function.arguments. Anthropic expects only the input object.
                arguments = parsed.get("arguments")
                if isinstance(arguments, dict) and "name" in parsed:
                    return arguments
                return parsed
            return None

        def _missing_required_tool_args(
            tool_name: str,
            parsed_args: dict[str, Any],
        ) -> list[str]:
            """Reject partial Claude Code tool calls before they reach the client.

            Claude Code starts tool execution as soon as it sees a tool_use block
            in this local fcc path. If the model/parser produced an early empty
            object, the executor fails validation and cancels sibling tool calls.
            """
            required_by_tool = {
                "Bash": ("command",),
                "Write": ("file_path", "content"),
                "Edit": ("file_path", "old_string", "new_string"),
                "MultiEdit": ("file_path", "edits"),
                "Read": ("file_path",),
                "Glob": ("pattern",),
                "Grep": ("pattern",),
                "LS": ("path",),
                "NotebookRead": ("notebook_path",),
                "NotebookEdit": ("notebook_path", "cell_id", "new_source"),
                "WebFetch": ("url", "prompt"),
                "TodoWrite": ("todos",),
            }
            required = required_by_tool.get(tool_name, ())
            missing = [
                key
                for key in required
                if key not in parsed_args or parsed_args[key] is None
            ]
            command = parsed_args.get("command")
            if tool_name == "Bash" and (
                not isinstance(command, str) or not command.strip()
            ):
                if "command" not in missing:
                    missing.append("command")
            return missing

        def _emit_buffered_tool_call(
            tool_state: dict[str, Any],
        ) -> list[AnthropicStreamEvent]:
            """Return a complete tool_use block only when arguments are valid JSON."""
            if tool_state["started"]:
                return []
            if not tool_state["id"] or not tool_state["name"]:
                return []

            parsed_args = _parse_tool_arguments(
                tool_state["name"],
                tool_state["args"] or "{}",
            )
            if parsed_args is None:
                logger.warning(
                    "Dropping invalid streamed tool_call args "
                    "(id=%s name=%s len=%d preview=%r tail=%r)",
                    tool_state["id"],
                    tool_state["name"],
                    len(tool_state["args"] or ""),
                    (tool_state["args"] or "")[:240],
                    (tool_state["args"] or "")[-240:],
                )
                tool_state["dropped"] = True
                return []
            missing_args = _missing_required_tool_args(
                tool_state["name"],
                parsed_args,
            )
            if missing_args:
                logger.warning(
                    "Dropping incomplete streamed tool_call args "
                    "(id=%s name=%s missing=%s len=%d preview=%r tail=%r)",
                    tool_state["id"],
                    tool_state["name"],
                    missing_args,
                    len(tool_state["args"] or ""),
                    (tool_state["args"] or "")[:240],
                    (tool_state["args"] or "")[-240:],
                )
                tool_state["dropped"] = True
                return []
            if tool_state["name"] == "Write" and tool_state["args"]:
                try:
                    json.loads(tool_state["args"], strict=False)
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.warning(
                        "Recovered malformed streamed Write args "
                        "(id=%s len=%d)",
                        tool_state["id"],
                        len(tool_state["args"] or ""),
                    )

            tool_state["started"] = True
            return [
                AnthropicStreamEvent(
                    index=tool_state["block_index"],
                    type="content_block_start",
                    content_block=AnthropicContentBlock(
                        type="tool_use",
                        id=tool_state["id"],
                        name=tool_state["name"],
                        input=parsed_args,
                    ),
                ),
                AnthropicStreamEvent(
                    index=tool_state["block_index"],
                    type="content_block_stop",
                ),
            ]

        try:
            first_item = True
            finish_reason = None
            content_block_index = 0
            content_block_started = False
            tool_use_started = False
            tool_streams: dict[int, dict[str, Any]] = {}
            text_buffer: list[str] = []

            async for item in generator:
                if item.startswith("data:"):
                    data_str = item[5:].strip().rstrip("\n")
                    if data_str == "[DONE]":
                        stop_message = AnthropicStreamEvent(
                            type="message_stop",
                        )
                        data = stop_message.model_dump_json(
                            exclude_unset=True, exclude_none=True
                        )
                        yield wrap_data_with_event(data, "message_stop")
                        yield "data: [DONE]\n\n"
                    else:
                        origin_chunk = ChatCompletionStreamResponse.model_validate_json(
                            data_str
                        )

                        if first_item:
                            chunk = AnthropicStreamEvent(
                                type="message_start",
                                message=AnthropicMessagesResponse(
                                    id=origin_chunk.id,
                                    content=[],
                                    model=origin_chunk.model,
                                    usage=AnthropicUsage(
                                        input_tokens=origin_chunk.usage.prompt_tokens
                                        if origin_chunk.usage
                                        else 0,
                                        output_tokens=0,
                                    ),
                                ),
                            )
                            first_item = False
                            data = chunk.model_dump_json(exclude_unset=True)
                            yield wrap_data_with_event(data, "message_start")
                            continue

                        # last chunk including usage info
                        if len(origin_chunk.choices) == 0:
                            for tool_index in sorted(tool_streams):
                                tool_state = tool_streams[tool_index]
                                if tool_state.get("dropped"):
                                    continue
                                if tool_state["block_index"] is None:
                                    tool_state["block_index"] = content_block_index
                                for chunk in _emit_buffered_tool_call(tool_state):
                                    data = chunk.model_dump_json(
                                        exclude_unset=True, exclude_none=True
                                    )
                                    yield wrap_data_with_event(data, chunk.type)
                                if tool_state["started"]:
                                    tool_use_started = True
                                    content_block_index += 1

                            invalid_tool_dropped = any(
                                state.get("dropped")
                                for state in tool_streams.values()
                            )
                            if invalid_tool_dropped and not tool_use_started:
                                logger.warning(
                                    "All streamed tool calls were dropped because "
                                    "their arguments were invalid JSON; returning "
                                    "a diagnostic text block instead of an empty "
                                    "assistant turn."
                                )
                                chunk = AnthropicStreamEvent(
                                    index=content_block_index,
                                    type="content_block_start",
                                    content_block=AnthropicContentBlock(
                                        type="text", text=""
                                    ),
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_start")
                                chunk = AnthropicStreamEvent(
                                    index=content_block_index,
                                    type="content_block_delta",
                                    delta=AnthropicDelta(
                                        type="text_delta",
                                        text=(
                                            "Invalid tool call was not executed: "
                                            "arguments were not complete valid JSON. "
                                            "Retry the tool call with one complete "
                                            "JSON object."
                                        ),
                                    ),
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_delta")
                                chunk = AnthropicStreamEvent(
                                    index=content_block_index,
                                    type="content_block_stop",
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_stop")
                                content_block_index += 1

                            if text_buffer and not tool_use_started:
                                chunk = AnthropicStreamEvent(
                                    index=content_block_index,
                                    type="content_block_start",
                                    content_block=AnthropicContentBlock(
                                        type="text", text=""
                                    ),
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_start")
                                for text_part in text_buffer:
                                    chunk = AnthropicStreamEvent(
                                        index=content_block_index,
                                        type="content_block_delta",
                                        delta=AnthropicDelta(
                                            type="text_delta",
                                            text=text_part,
                                        ),
                                    )
                                    data = chunk.model_dump_json(exclude_unset=True)
                                    yield wrap_data_with_event(
                                        data, "content_block_delta"
                                    )
                                stop_chunk = AnthropicStreamEvent(
                                    index=content_block_index,
                                    type="content_block_stop",
                                )
                                data = stop_chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(
                                    data, "content_block_stop"
                                )
                                text_buffer.clear()
                            if content_block_started:
                                stop_chunk = AnthropicStreamEvent(
                                    index=content_block_index,
                                    type="content_block_stop",
                                )
                                data = stop_chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_stop")
                            stop_reason = "tool_use" if tool_use_started else (
                                "end_turn" if finish_reason == "tool_calls"
                                else self.stop_reason_map.get(finish_reason or "stop")
                            )
                            chunk = AnthropicStreamEvent(
                                type="message_delta",
                                delta=AnthropicDelta(stop_reason=stop_reason),
                                usage=AnthropicUsage(
                                    input_tokens=origin_chunk.usage.prompt_tokens
                                    if origin_chunk.usage
                                    else 0,
                                    output_tokens=origin_chunk.usage.completion_tokens
                                    if origin_chunk.usage
                                    else 0,
                                ),
                            )
                            data = chunk.model_dump_json(exclude_unset=True)
                            yield wrap_data_with_event(data, "message_delta")
                            continue

                        if origin_chunk.choices[0].finish_reason is not None:
                            finish_reason = origin_chunk.choices[0].finish_reason
                            continue

                        # content
                        if origin_chunk.choices[0].delta.content is not None:
                            if origin_chunk.choices[0].delta.content:
                                text_buffer.append(
                                    origin_chunk.choices[0].delta.content
                                )
                            continue

                        # tool calls
                        elif len(origin_chunk.choices[0].delta.tool_calls) > 0:
                            # Tool calls must be clean for Claude Code. Qwen can emit
                            # reasoning/text before a tool call even with thinking
                            # disabled. Buffer tool arguments until they are complete
                            # valid JSON, otherwise Claude Code turns malformed JSON
                            # into an empty tool input and fails tool validation.
                            text_buffer.clear()
                            tool_call = origin_chunk.choices[0].delta.tool_calls[0]
                            tool_index = tool_call.index
                            if tool_index is None:
                                tool_index = 0
                            tool_state = tool_streams.setdefault(
                                tool_index,
                                {
                                    "id": None,
                                    "name": None,
                                    "args": "",
                                    "started": False,
                                    "block_index": None,
                                    "dropped": False,
                                },
                            )
                            if tool_call.id is not None:
                                tool_state["id"] = tool_call.id
                            arguments = None
                            if tool_call.function:
                                if tool_call.function.name:
                                    tool_state["name"] = tool_call.function.name
                                if tool_call.function.arguments:
                                    arguments = tool_call.function.arguments

                            if arguments:
                                tool_state["args"] += arguments
                            continue
                else:
                    error_response = AnthropicStreamEvent(
                        type="error",
                        error=AnthropicError(
                            type="internal_error",
                            message="Invalid data format received",
                        ),
                    )
                    data = error_response.model_dump_json(exclude_unset=True)
                    yield wrap_data_with_event(data, "error")
                    yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("Error in message stream converter.")
            error_response = AnthropicStreamEvent(
                type="error",
                error=AnthropicError(type="internal_error", message=str(e)),
            )
            data = error_response.model_dump_json(exclude_unset=True)
            yield wrap_data_with_event(data, "error")
            yield "data: [DONE]\n\n"
