# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ФАЛЬСИФИКАТОР СНЯТИЕМ ФАЗЫ: сколько в шаге декода стоит рекуррентное ядро GDN.
#
# ЗАЧЕМ. Слитое тело h+o+wu живёт только на префилле: модель диспетчеризует по фазе
# (qwen3_next.py:825 chunk_... на префилле, :698/:843 fused_recurrent_... на декоде), а подменяем
# мы только чанковый путь. Значит 48 слоёв GDN на декоде идут ЧУЖИМ ядром. Прежде чем писать своё,
# надо знать ЦЕНУ ФАЗЫ -- иначе это стройка вслепую.
#
# КАК. Рекуррентное ядро заменяется возвратом НУЛЕЙ нужной формы. Ответ заведомо неверен, читается
# ТОЛЬКО ВРЕМЯ; разность с боевым прогоном и есть доля фазы. Состояние возвращается как есть.
#
# ВКЛЮЧЕНИЕ: FA2SM70_GDN_DEC_NULL=1 (декод), FA2SM70_GDN_PRE_NULL=1 (ПРЕФИЛЛ, чанковый путь).
# Сервер после этого ГОВОРИТ МУСОР -- это прибор, не режим.
#
# ПРЕФИЛЛЬНАЯ ВЕТКА -- ГЛАВНАЯ. Именно там лежит масса: слитое тело h+o+wu уже наше, и вопрос не
# «сколько стоит ядро», а «сколько стоит ФАЗА», то есть сколько вообще можно отыграть слиянием и
# укладкой в тень. Без этого числа любое слияние -- стройка вслепую.
import os
import sys

_ДЕК = os.environ.get("FA2SM70_GDN_DEC_NULL", "0") == "1"
_ПРЕ = os.environ.get("FA2SM70_GDN_PRE_NULL", "0") == "1"

if _ДЕК or _ПРЕ:
    import torch

    С = {"n": 0}

    def _пусто(*a, **k):
        С["n"] += 1
        v = k.get("v")
        if v is None:                       # у слитого гейтинга v тоже именованный, но подстрахуемся
            v = a[2] if len(a) > 2 else a[-1]
        return torch.zeros_like(v), k.get("initial_state")

    # ДВА ИМЕНИ, А НЕ ОДНО -- И ЭТО ИСПРАВЛЕНИЕ, КУПЛЕННОЕ СЛЕПЫМ ПРИБОРОМ.
    # Первая редакция снимала только `fused_recurrent_gated_delta_rule` и отрапортовала «снято в
    # 3 модулях», но ответ остался ВЕРНЫМ ("Париж — это город..."), а время не сдвинулось. Причина:
    # у декода есть БЫСТРЫЙ ПУТЬ (qwen3_next.py:677), и он зовёт ДРУГУЮ функцию --
    # `fused_sigmoid_gating_delta_rule_update`, где гейтинг слит с обновлением состояния.
    # Отсюда правило: у фальсификатора обязан быть ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ (ответ ДОЛЖЕН испортиться);
    # без него «время не изменилось» читается как «фаза бесплатна», хотя фазу просто не сняли.
    ИМЕНА = ()
    if _ДЕК:
        ИМЕНА += ("fused_recurrent_gated_delta_rule", "fused_sigmoid_gating_delta_rule_update")
    if _ПРЕ:
        ИМЕНА += ("chunk_gated_delta_rule",)
    сколько = 0
    for путь in ("vllm.model_executor.models.qwen3_next",
                 "vllm.model_executor.models.qwen3_5",
                 "vllm.model_executor.layers.fla.ops",
                 "vllm.model_executor.layers.fla.ops.fused_recurrent",
                 "vllm.model_executor.layers.fla.ops.fused_sigmoid_gating",
                 "vllm.model_executor.layers.fla.ops.chunk"):
        try:
            мод = __import__(путь, fromlist=["x"])
            for имя in ИМЕНА:
                if getattr(мод, имя, None) is not None:
                    setattr(мод, имя, _пусто)
                    сколько += 1
        except Exception:
            pass
    ф = ("ДЕКОДА" if _ДЕК else "") + ("+" if _ДЕК and _ПРЕ else "") + ("ПРЕФИЛЛА" if _ПРЕ else "")
    print(f"[fa2_sm70] ФАЛЬСИФИКАТОР: ядро GDN-{ф} снято в {сколько} модулях. "
          f"ОТВЕТЫ НЕВЕРНЫ, читается только время.", file=sys.stderr, flush=True)
    if сколько == 0:
        print("[fa2_sm70] ФАЛЬСИФИКАТОР СЛЕП: ни один модуль не подменён", file=sys.stderr, flush=True)
