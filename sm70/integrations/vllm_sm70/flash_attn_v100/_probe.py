# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ПОСЕГМЕНТНЫЙ ЗАМЕР СТЕНЫ ПО ТИПАМ СЛОЁВ. Включается FA2SM70_PROBE=1.
#
# ЗАЧЕМ. Раскладка префилла свелась к тому, что 127 из 222 секунд достались "остатку", и остаток был
# НАЗВАН именем (GEMM) вместо того, чтобы быть измеренным. А модель гибридная: full_attention_interval=4,
# то есть полное внимание лишь в 16 слоях из 64, а 48 -- linear_attention (gated delta-net: причинная
# свёртка + рекуррентный скан). Их в роофлайне проекций не было вовсе. Этот зонд меряет каждый кусок
# отдельно, чтобы остатка не осталось.
#
# ПОЧЕМУ НА СОБЫТИЯХ, А НЕ НА synchronize. Прошлый замер шёл при FA2SM70_PROF_PREFILL=1, который ставит
# три torch.cuda.synchronize() на вызов -- 384 осушения конвейера на заявку, то есть инструмент менял
# измеряемое. Здесь на каждый участок пишется пара событий (микросекунды, без остановки), а elapsed_time
# считается пачкой раз в DRAIN вызовов -- одна синхронизация на пятьсот участков.
#
# ГДЕ ПАТЧИТЬ. Модуль модели делает `from ... import chunk_gated_delta_rule`, то есть привязывает имя в
# СВОЁ пространство. Патчить надо именно там, а не в исходном модуле -- иначе подмена не видна.
import os
import sys

import torch

ENABLED = os.environ.get("FA2SM70_PROBE", "0") == "1"
DRAIN = int(os.environ.get("FA2SM70_PROBE_DRAIN", "512"))

_acc: dict[str, list] = {}          # имя -> [мс, вызовов]
_pending: list = []                 # (имя, событие начала, событие конца)
_installed = False


def _drain(force: bool = False) -> None:
    if not _pending or (len(_pending) < DRAIN and not force):
        return
    torch.cuda.synchronize()        # ОДНА синхронизация на DRAIN участков
    for name, a, b in _pending:
        e = _acc.setdefault(name, [0.0, 0])
        e[0] += a.elapsed_time(b)
        e[1] += 1
    _pending.clear()


def _wrap(name, fn):
    def inner(*a, **k):
        s = torch.cuda.Event(enable_timing=True)
        t = torch.cuda.Event(enable_timing=True)
        s.record()
        r = fn(*a, **k)
        t.record()
        _pending.append((name, s, t))
        _drain()
        return r
    inner.__name__ = getattr(fn, "__name__", name)
    inner.__fa2_probe__ = True
    return inner


def report() -> str:
    _drain(force=True)
    if not _acc:
        return "[fa2_sm70 probe] пусто"
    tot = sum(v[0] for v in _acc.values())
    rows = sorted(_acc.items(), key=lambda kv: -kv[1][0])
    out = [f"[fa2_sm70 probe] сумма замеренных участков {tot/1000:.2f} с"]
    for n, (ms, c) in rows:
        out.append(f"    {n:<28} {ms/1000:8.2f} с  {100*ms/tot:5.1f}%  вызовов {c:6d}  "
                   f"{ms/max(c,1):7.3f} мс/вызов")
    return "\n".join(out)


def install() -> None:
    """Ставится ЛЕНИВО -- на первом обращении из обвязки. К этому моменту модуль модели уже
    импортирован, а на старте нашего пакета его ещё нет."""
    global _installed
    if _installed or not ENABLED:
        return
    _installed = True
    n = 0
    # Линейное внимание (gated delta-net): скан префилла, скан декода и причинная свёртка.
    for mod in ("vllm.model_executor.models.qwen3_next", "vllm.model_executor.models.qwen3_5"):
        m = sys.modules.get(mod)
        if m is None:
            continue
        for fname, label in (("chunk_gated_delta_rule", "линейное: скан (префилл)"),
                             ("fused_recurrent_gated_delta_rule", "линейное: скан (декод)"),
                             ("causal_conv1d_fn", "линейное: свёртка"),
                             ("causal_conv1d_update", "линейное: свёртка (декод)"),
                             ("fused_gdn_gating", "линейное: гейт")):
            f = getattr(m, fname, None)
            if f is not None and not getattr(f, "__fa2_probe__", False):
                setattr(m, fname, _wrap(label, f)); n += 1
    # ОСТАЛЬНЫЕ КРУПНЫЕ ОПЕРАЦИИ. Классы слоёв патчить БЕСПОЛЕЗНО: модель под torch.compile, и
    # подмена метода после трассировки в граф не попадает (проверено -- функции ловятся, классы нет).
    # А вот пользовательские операции граф вызывает обратно в Python, поэтому их видно.
    # КАЖДАЯ ГРУППА В СВОЁМ try. Прошлая редакция ставила _installed=True, вешала три обёртки и
    # ПАДАЛА на импорте следующей -- молча, потому что исключение уходило наверх после того, как
    # флаг уже стоял. Симптом: работают ровно первые обёртки, дальше тишина. Диагностика обязательна.
    fails = []
    def _try(label, thunk):
        nonlocal n
        try:
            n += thunk() or 0
        except Exception as e:
            fails.append(f"{label}: {type(e).__name__} {str(e)[:60]}")

    def _quant():
        import importlib
        k = 0
        for modname, clsname in (("vllm.model_executor.layers.quantization.gptq", "GPTQLinearMethod"),
                                 ("vllm.model_executor.layers.quantization.gptq_marlin", "GPTQMarlinLinearMethod"),
                                 ("vllm.model_executor.layers.quantization.awq", "AWQLinearMethod")):
            try:
                mm = importlib.import_module(modname)
            except Exception:
                continue
            c = getattr(mm, clsname, None)
            if c is not None and not getattr(c.apply, "__fa2_probe__", False):
                c.apply = _wrap(f"умножение {clsname}", c.apply); k += 1
        return k
    _try("quant", _quant)

    def _moe_f():
        import vllm.model_executor.layers.fused_moe.fused_moe as _moe
        if getattr(_moe.fused_experts, "__fa2_probe__", False):
            return 0
        _moe.fused_experts = _wrap("MoE-эксперты", _moe.fused_experts); return 1
    _try("moe", _moe_f)

    def _mixer_inner():
        """РАЗОБРАТЬ МЕШАЛКУ. Слой стоит 56.3 мс, а его собственные ядра (скан+свёртка+гейт) -- 16.5.
        Остальные 40 мс не разложены. Кандидаты видны в исходнике: перестановка mixed_qkv, КОПИЯ
        состояния SSM на входе (ssm_state[idx].contiguous()) и обратная запись, финальная копия
        выхода и гейтованная нормировка. Оборачиваем ядро целиком (_forward_core) -- разность со
        слоем даст стоимость проекций, а внутри ядра станет видно остальное."""
        import importlib
        k = 0
        m = importlib.import_module("vllm.model_executor.models.qwen3_next")
        for cls, meth, label in (("Qwen3NextGatedDeltaNet", "_forward_core", "мешалка: ядро (без проекций)"),
                                 ("Qwen3NextGatedDeltaNet", "rearrange_mixed_qkv", "мешалка: перестановка qkv")):
            c = getattr(m, cls, None)
            f = getattr(c, meth, None) if c else None
            if f is not None and not getattr(f, "__fa2_probe__", False):
                setattr(c, meth, _wrap(label, f)); k += 1
        for mn, cn in (("vllm.model_executor.layers.layernorm", "RMSNormGated"),
                       ("vllm.model_executor.layers.mamba.mamba_mixer2", "Mixer2RMSNormGated")):
            try:
                mm = importlib.import_module(mn)
            except Exception:
                continue
            c = getattr(mm, cn, None)
            if c is not None and not getattr(c.forward, "__fa2_probe__", False):
                c.forward = _wrap("мешалка: гейтованная норма", c.forward); k += 1
        return k
    _try("mixer", _mixer_inner)

    def _shapes():
        """СНЯТЬ НАСТОЯЩИЕ ФОРМЫ, а не выводить их из config.json. Две мои прикидки по FLOP дали
        невозможные 105 и 124 TFLOP/s -- значит модель работы для гибридной архитектуры была неверна.
        Здесь каждая GPTQ-проекция сама сообщает свои K,N и время, и эффективность считается по
        ФАКТИЧЕСКИМ размерам."""
        import importlib
        mm = importlib.import_module("vllm.model_executor.layers.quantization.gptq")
        c = mm.GPTQLinearMethod
        if getattr(c.apply, "__fa2_shapes__", False):
            return 0
        base = c.apply

        def apply(self, layer, x, bias=None):
            st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
            st.record(); r = base(self, layer, x, bias); en.record()
            M = int(x.numel() // x.shape[-1]); K = int(x.shape[-1])
            N = int(r.shape[-1])
            _pending.append((f"GPTQ {K}x{N} M={M}", st, en))
            _drain()
            return r
        apply.__fa2_probe__ = True; apply.__fa2_shapes__ = True
        c.apply = apply
        return 1
    _try("shapes", _shapes)

    def _runner():
        import vllm.v1.worker.gpu_model_runner as _r
        c = _r.GPUModelRunner
        if getattr(c.execute_model, "__fa2_probe__", False):
            return 0
        c.execute_model = _wrap("ШАГ исполнения (раннер)", c.execute_model); return 1
    _try("runner", _runner)

    def _ar():
        k = 0
        for mn, m in list(sys.modules.items()):
            if not mn.startswith("vllm.") or m is None:
                continue
            f = getattr(m, "tensor_model_parallel_all_reduce", None)
            if callable(f) and not getattr(f, "__fa2_probe__", False):
                setattr(m, "tensor_model_parallel_all_reduce", _wrap("обмен (all-reduce)", f)); k += 1
        return k
    _try("all-reduce", _ar)

    # Слой целиком -- чтобы увидеть, сколько в нём НЕ покрыто перечисленным выше.
    for mod, cls, label in (
            ("vllm.model_executor.models.qwen3_next", "Qwen3NextGatedDeltaNet", "СЛОЙ линейного внимания"),
            ("vllm.model_executor.models.qwen3_next", "Qwen3NextAttention", "СЛОЙ полного внимания"),
            ("vllm.model_executor.models.qwen3_next", "Qwen3NextSparseMoeBlock", "СЛОЙ MoE"),
            ("vllm.model_executor.models.qwen3_5", "Qwen3_5GatedDeltaNet", "СЛОЙ линейного внимания"),
            # MLP НЕ БЫЛ ОБЁРНУТ, и это была дыра размером в половину стены: сумма участков сходилась
            # с ней только потому, что строки СЛОЙ пересекаются с GPTQ. Оборачиваем и его, и весь
            # блок декодера -- тогда "СЛОЙ декодера минус (мешалка + MLP)" даёт остаток явно.
            ("vllm.model_executor.models.qwen3_next", "Qwen3NextMLP", "СЛОЙ MLP"),
            ("vllm.model_executor.models.qwen3_next", "Qwen3NextDecoderLayer", "БЛОК декодера"),
            ("vllm.model_executor.models.qwen3_5", "Qwen3_5DecoderLayer", "БЛОК декодера"),
            ("vllm.model_executor.models.qwen3_next", "Qwen3NextModel", "МОДЕЛЬ целиком"),
            # ЭЛЕМЕНТАРНЫЕ ОПЕРАЦИИ. Гипотеза: они идут на ядрах, не считанных под SM70, и стоят
            # много больше своей полосы памяти. SiLU-mul на плитке 15725x8704 должна стоить ~0.8 мс
            # по 841 ГБ/с, а из раскладки MLP на неё приходится порядка 12 мс. Проверяем.
            ("vllm.model_executor.layers.activation", "SiluAndMul", "активация SiLU-mul"),
            ("vllm.model_executor.layers.layernorm", "RMSNorm", "нормировка RMSNorm"),
            ("vllm.model_executor.layers.layernorm", "GemmaRMSNorm", "нормировка RMSNorm"),
            # ВЕРХНИЙ УРОВЕНЬ -- чтобы разделить "вне блоков декодера" на составляющие:
            #   ШАГ исполнения      -- всё, что делает раннер за вызов (включая планирование и сэмплинг)
            #   МОДЕЛЬ целиком      -- стек слоёв плюс эмбеддинг и финальная нормировка
            #   сумма БЛОКОВ        -- уже есть
            # Разность (ШАГ - МОДЕЛЬ) = обвязка раннера; (МОДЕЛЬ - сумма БЛОКОВ) = эмбеддинг,
            # финальная нормировка и ПРОМЕЖУТКИ между слоями. Второе и есть тёмная четверть.
            ("vllm.model_executor.models.qwen3_5", "Qwen3_5Model", "МОДЕЛЬ (стек слоёв)"),
            ("vllm.model_executor.models.qwen3_5", "Qwen3_5ForConditionalGeneration", "МОДЕЛЬ + голова"),
    ):
        m = sys.modules.get(mod)
        c = getattr(m, cls, None) if m else None
        if c is not None and not getattr(c.forward, "__fa2_probe__", False):
            c.forward = _wrap(label, c.forward); n += 1
    print(f"[fa2_sm70 probe] установлено обёрток: {n}"
          + (("; НЕ ВСТАЛО -- " + "; ".join(fails)) if fails else ""), file=sys.stderr, flush=True)
