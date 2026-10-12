# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ==================================================================================================
# ПОТЕРЯ ВСЕГО ОТВЕТА У ДУМАЮЩИХ СЕТЕЙ, КОГДА РАЗМЫШЛЕНИЕ ВЫКЛЮЧЕНО.
# ==================================================================================================
#
# ЗАМЕРЕНО на Qwen3.8-27B, непотоковый /v1/chat/completions с
# `chat_template_kwargs: {"enable_thinking": false}`:
#     completion_tokens 235,  content = '',  reasoning = None
# То есть 235 токенов посчитаны, оплачены и ВЫБРОШЕНЫ. Сырой путь и потоковый при этом отдают
# верный текст -- значит дело не в сети, не в квантовании и не в наших ядрах.
#
# КОРЕНЬ (vllm/reasoning/basic_parsers.py):
#     def extract_content_ids(self, input_ids):
#         if self.end_token_id not in input_ids[:-1]: return []      # <-- вот здесь
# А непотоковый путь безусловно переопределяет ответ этим списком. При enable_thinking=false шаблон
# закрывает <think> ЕЩЁ В ПРОМПТЕ, поэтому в выводе тегов нет вовсе -- и «нет закрывающего тега»
# читается как «нет содержимого», хотя верно обратное: раз блока нет, СОДЕРЖИМОЕ -- ВЕСЬ вывод.
#
# ПОЧЕМУ ЗАПЛАТКА БЕЗОПАСНА. Меняется РОВНО ветка «закрывающего тега нет»: было [] (потерять всё),
# стало input_ids (отдать всё). Ветка с тегом не тронута, поэтому думающий путь не меняется ни на
# бит. Приёмка гоняет ОБЕ ветки, и отрицательная (думающий запрос) обязана остаться прежней.
#
# ОТКЛЮЧИТЬ: FA2SM70_QWEN3_THINK=0.
import os
import sys

if os.environ.get("FA2SM70_QWEN3_THINK", "1") == "1":
    try:
        from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

        def _содержимое(self, input_ids):
            if self.end_token_id not in input_ids[:-1]:
                return list(input_ids)       # блока размышления НЕТ -> содержимое это ВЕСЬ вывод
            return list(input_ids[input_ids.index(self.end_token_id) + 1:])

        BaseThinkingReasoningParser.extract_content_ids = _содержимое
        print("[fa2_sm70] думающие сети: ответ БЕЗ блока размышления больше не теряется "
              "(extract_content_ids отдавал пустой список)", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[fa2_sm70] Qwen3-think: подмена не встала: {e}", file=sys.stderr, flush=True)
