# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
# [FA2/SM70 24.08] В нашем форке базовый предлагатель живёт в eagle.py, а не в отдельном
# llm_base_proposer.py (upstream вынес его позже). Класс тот же.
from vllm.v1.spec_decode.eagle import EagleProposer, SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import (
    copy_and_expand_dflash_inputs_kernel,
    next_power_of_2,
)

logger = init_logger(__name__)


# НАСЛЕДУЕМСЯ ОТ EagleProposer, А НЕ ОТ БАЗОВОГО. Причина не в удобстве: движок в НЕСКОЛЬКИХ
# местах проверяет `isinstance(self.drafter, EagleProposer)` (диспетчер графов, ветка propose,
# подготовка метаданных). Менять эти проверки -- значит править чужие файлы в стольких местах,
# сколько их найдётся, и каждый раз заново при обновлении форка. Наследование закрывает их ВСЕ
# сразу и оставляет движок нетронутым. DFlash по устройству и есть eagle-подобный черновик:
# работает на скрытых состояниях цели, отличается лишь тем, что выдаёт блок за один проход.
class DFlashProposer(EagleProposer):
    # ==========================================================================================
    # [FA2/SM70 24.08] СЛОЙ СОВМЕСТИМОСТИ ЖИВЁТ ЗДЕСЬ, А НЕ В ДВИЖКЕ.
    # У upstream предлагатель зовёт `_determine_batch_execution_and_padding(num_query_tokens,
    # use_cudagraphs=...)` и ждёт тройку (режим графа, паддированное число токенов, разбивка по DP).
    # В нашем форке одноимённый метод принадлежит РАННЕРУ и имеет совсем другую сигнатуру
    # (num_tokens, num_reqs, num_scheduled_tokens_np, ...). Подгонять встроенный метод под чужой
    # вызов -- плодить правки в чужих файлах; вместо этого весь стык собран в ОДНОМ месте: здесь.
    # Так порт остаётся обратимым, а движок -- нетронутым.
    # ==========================================================================================
    # Имена полей у upstream и у нас разошлись -- сводим их ЗДЕСЬ же, одним списком алиасов,
    # чтобы стык был в одном месте и снимался одной правкой, когда форк догонит upstream.
    @property
    def _draft_attn_layer_names(self):
        return self.attn_layer_names

    def _determine_batch_execution_and_padding(
        self, num_query_tokens: int, use_cudagraphs: bool = True
    ):
        from vllm.config.compilation import CUDAGraphMode

        if not use_cudagraphs:
            return CUDAGraphMode.NONE, num_query_tokens, None
        # Черновик идёт PIECEWISE -- ровно как eagle в нашем форке ("Eagle currently only
        # supports PIECEWISE cudagraphs"), поэтому режим берём тот же, а паддинг -- по списку
        # захваченных размеров, если он есть.
        размеры = getattr(
            self.vllm_config.compilation_config, "cudagraph_capture_sizes", None
        )
        n = num_query_tokens
        if размеры:
            подходящие = [x for x in sorted(размеры) if x >= num_query_tokens]
            n = подходящие[0] if подходящие else num_query_tokens
        return CUDAGraphMode.PIECEWISE, n, None

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        # EagleProposer сам задаёт `pass_hidden_states_to_model` при вызове базового класса,
        # поэтому здесь его НЕ передаём -- иначе TypeError на лишнем аргументе.
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            runner=runner,
        )

        # [FA2/SM70 24.08] В нашем форке базовый предлагатель не заводит `max_batch_size`
        # (upstream добавил позже). Берём то же число из планировщика -- это предельное число
        # цепочек в батче, ровно тот смысл, что и у upstream.
        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        self.max_padded_query_tokens = max(
            self.max_query_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_padded_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_padded_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_padded_query_tokens,
            dtype=torch.int64,
            device=device,
        )

        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # For DFlash we use the input embeddings to embed the mask token
        self.parallel_drafting_hidden_state_tensor = None

        from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

        self.dflash_causal = not dflash_has_any_non_causal(
            self.draft_model_config.hf_config
        )

    @override
    def load_model(self, target_model: torch.nn.Module) -> None:
        from vllm.model_executor.models.qwen3_dflash import (
            dflash_target_rope_is_neox_style,
        )

        draft_model_config = self.speculative_config.draft_model_config
        assert draft_model_config is not None
        # The drafter must rotate Q/K the way its target does. Take that from the
        # built target before super() constructs the draft.
        is_neox_style = dflash_target_rope_is_neox_style(target_model)
        if is_neox_style is not None:
            draft_model_config.hf_config.is_neox_style = is_neox_style
        super().load_model(target_model)

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        # The draft model is text-only — clear the target's multimodal
        # flag so flash_attn is not rejected for mm_prefix support.
        arch = base.model_config.model_arch_config
        if arch.is_mm_prefix_lm:
            base.model_config.model_arch_config = replace(arch, is_mm_prefix_lm=False)
        return replace(
            base,
            attention_config=replace(
                base.attention_config,
                use_non_causal=not self.dflash_causal,
            ),
        )

    @override
    def _warn_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        pass

    @override
    def propose(self, *args, **kwargs):
        """[FA2/SM70 24.08] Перехватываем скрытые состояния цели ЗДЕСЬ.

        В нашем форке `target_hidden_states` -- параметр `propose`, а в `set_inputs_first_pass`
        он не передаётся (upstream передаёт его туда аргументом). Меняем не движок и не базовый
        класс, а СВОЙ путь: кладём состояния в поле и подставляем их ниже.
        """
        сост = kwargs.get("target_hidden_states")
        if сост is None and len(args) >= 3:
            сост = args[2]
        self._df_target_hidden = сост
        return super().propose(*args, **kwargs)

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        last_token_indices: torch.Tensor | None = None,
        cad: CommonAttentionMetadata = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        target_hidden_states: torch.Tensor | None = None,
        token_indices_to_sample: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        # СИГНАТУРА ПРИВЕДЕНА К НАШЕЙ (last_token_indices вместо token_indices_to_sample,
        # без обязательных состояний), а недостающее берётся из поля, заполненного в propose.
        if target_hidden_states is None:
            target_hidden_states = self._df_target_hidden
        if token_indices_to_sample is None:
            token_indices_to_sample = last_token_indices
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        # Store for build_model_inputs_first_pass to use
        self._dflash_num_context = num_context

        # We don't need to copy into a buffer here since the context preprocessing
        # does not run in a CUDA graph
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Launch fused triton kernel for input_ids, positions, slot_mapping,
        # and token_indices_to_sample
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, next_power_of_2(max_tokens_per_req))
        num_blocks = (max_tokens_per_req + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        copy_and_expand_dflash_inputs_kernel[grid](
            # Inputs
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            # Outputs
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # Block table
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            # Metadata
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            # Scalars
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        # In padded mode, cad.seq_lens includes rejected tokens. Subtract
        # them so attention only sees the valid prefix of context states.
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        # Skip num_rejected_tokens (GPU-only); overestimating is fine here.
        new_seq_lens_cpu_upper_bound = (
            cad.seq_lens_cpu_upper_bound + num_query_per_req
            if cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=query_slot_mapping,
            causal=self.dflash_causal,
        )

        return num_query_total, token_indices_to_sample, new_cad

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Key differences to default dummy_run:
        - Only one forward pass due to parallel drafting
        - DFlash uses context states as unpadded metadata, so hidden_states will
        use the unpadded num_tokens instead of num_input_tokens
        - max_query_tokens is quite small, DFlash only sees spec tokens as queries
        - Multimodal inputs are not currently supported
        """
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        # Slot mapping sized to num_input_tokens (query only), matching
        # the K/V tensor size from the model forward.  Context KVs are
        # pre-inserted separately and don't flow through the model.
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # Context and query positions use separate buffers; no copy needed.
        context_positions = self._context_positions_buffer[:num_tokens]
        # Context states will be passed directly to the precomputation without
        # going through the buffer, since no CUDA graph is used for the precomputation.
        # For the dummy run, we use the dummy buffer.
        context_states = self.hidden_states[:num_tokens]

        # Run the KV projection (GEMM + norms + RoPE) for memory profiling,
        self.model.precompute_and_store_context_kv(context_states, context_positions)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        # Context and query positions/slots were written to separate
        # buffers by the kernel — no copy needed.
        num_context = self._dflash_num_context

        # Pre-insert context KVs directly into cache
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,  # Shape is already [num_context, hidden_size]
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(
            cad, draft_index
        )
        if not self.dflash_causal:
            # Require all layers to support non-causal attention when required by DFlash
            for layer_name, attn_metadata in per_layer.items():
                assert getattr(attn_metadata, "causal", None) is False, (
                    f"Attention metadata for layer {layer_name} does not have"
                    " non-causal support, which is required for DFlash."
                    " Consider using a different attention backend, e.g FlashAttention."
                )
        return per_group, per_layer

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        return self.dflash_config.get("use_aux_hidden_state", True)

    @property
    def dflash_config(self):
        return getattr(self.draft_model_config.hf_config, "dflash_config", None) or {}
