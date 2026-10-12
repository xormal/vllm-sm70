# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ==================================================================================================
# ЧЕРНОВИК DFlash2 -- НА ТЕНЗОРНОЕ ТЕЛО (int8 + turbomind). ПОЧЕМУ ИМЕННО ТАК.
# ==================================================================================================
#
# ЗАМЕР (опись пути по ядрам, декодный шаг 45.2 мс): тело черновика идёт `cutlass_70_wmma_tensorop`
# и стоит 4.9 мс -- 11 % шага. Пол ЧТЕНИЯ у него 2.05 мс (5 слоёв, 1.6 млрд параметров fp16, на
# ранг 1.6 ГБ при 780 ГБ/с). То есть плотный GEMM при M=k+1=4 идёт в 2.4 раза выше собственного
# пола: при таком M он связан НЕ полосой, а занятостью.
#
# ПОЧЕМУ НЕ int12 (проверено и отвергнуто замером). `FA2SM70_W12_DRAFT=1` дал 43.5 ток/с против
# 44.2 -- ХУЖЕ. Причина -- закон §26: `w12a16` это gemv-тело на СКАЛЯРНОМ конвейере, и при M>1
# оно платит налог по M ровно как `w8a16_gemv`. Сжатие байт (x1.33) съедается ростом счёта.
# Вывод, который стоит помнить: СЖАТИЕ ВЕСОВ ОКУПАЕТСЯ ТОЛЬКО ВМЕСТЕ С ТЕНЗОРНЫМ ТЕЛОМ.
#
# ЧТО ДЕЛАЕТ ЭТОТ МОДУЛЬ. Квантует веса черновика в int8 (симметрично, по группам 128 вдоль K),
# строит раскладку turbomind и заменяет `apply` слоя. Оба выигрыша складываются: байт вдвое меньше
# И тело ПЛОСКОЕ по M (замер: 124/122/123/123 мкс при M=1/2/4/8 против растущего скалярного).
#
# ИСХОДНЫЙ ВЕС ОСВОБОЖДАЕТСЯ. Упаковка int8 -- половина от fp16, значит замена не добавляет памяти,
# а ОСВОБОЖДАЕТ ~0.8 ГБ на ранг. Держать обе раскладки незачем: у черновика нет зоны, где выгодно
# плотное тело (M у него всегда k+1).
#
# ГЕЙТ -- ПРИЁМКА, А НЕ ГЛАЗ. Квантование черновика меняет его предсказания; единственный честный
# критерий -- приёмка (tau) и покрытие top-n. Падение приёмки съедает выигрыш времени, поэтому
# рычаг ОТКЛЮЧАЕМ по умолчанию и включаем только после сверки: FA2SM70_DRAFT_TM8=1.
import os
import sys

import torch

_ВКЛ = os.environ.get("FA2SM70_DRAFT_TM8", "0") == "1"
_G = 128
_С = {"слоёв": 0, "байт": 0}


def _расш():
    import fa2_sm70
    return fa2_sm70._ext.tm8_ext()


def _квантовать(w: torch.Tensor):
    """[N,K] fp16 -> (u8-коды [N,K] int16, масштабы [K/G,N] fp16, нули [K/G,N] fp16).

    Симметрично по группам вдоль K: s = max|w|/127, код = round(w/s) + 128. Нуль ровно 128,
    поэтому смещение точное и не зависит от данных -- условность нуля тут не возникает
    (в отличие от GPTQ-слепка, где она уже трижды давала тихий дефект).
    """
    N, K = w.shape
    # КУСКАМИ ПО СТРОКАМ. Разворот всего веса в fp32 стоит вчетверо от fp16 и на самом крупном
    # слое (5120x34816) просит 1.19 ГиБ разом -- ровно на этом подъём и упал OOM. Кусок держим
    # около 64 МиБ промежуточных, тогда пик не зависит от размера слоя.
    коды = torch.empty(N, K, dtype=torch.int16, device=w.device)
    масштабы = torch.empty(K // _G, N, dtype=torch.float16, device=w.device)
    шаг = max(1, (1 << 24) // K)
    for i0 in range(0, N, шаг):
        i1 = min(i0 + шаг, N)
        wf = w.data[i0:i1].float().view(i1 - i0, K // _G, _G)
        s = wf.abs().amax(-1).clamp_min(1e-8) / 127.0        # [кусок, K/G]
        q = (wf / s[..., None]).round().clamp_(-127, 127)
        коды[i0:i1] = (q.view(i1 - i0, K) + 128.0).to(torch.int16)
        масштабы[:, i0:i1] = s.t().to(torch.float16)
        del wf, s, q
    нули = torch.full_like(масштабы, 128.0)
    return коды, масштабы, нули


def _перевести(model) -> None:
    кл = type(model).__name__
    if "DFlash" not in кл:
        return
    # [ФИЛЬТР ПО КЛАССУ, А НЕ ПО ПРИЗНАКУ. ОШИБКА, ПОЙМАННАЯ ПОДЪЁМОМ]
    # Первая редакция брала всё, у чего есть `quant_method` -- и захватила ТАБЛИЦУ ВЛОЖЕНИЙ и
    # словарную голову (`VocabParallelEmbedding` тоже её имеет). Их вес заквантован и освобождён,
    # а поиск по вложениям на пустом тензоре даёт device-side ассерт ДАЛЕКО от места. Признак
    # «есть quant_method» -- ЛОЖНЫЙ различитель линейного слоя.
    from vllm.model_executor.layers.linear import LinearBase
    ext = _расш()
    for имя, m in model.named_modules():
        w = getattr(m, "weight", None)
        if w is None or not isinstance(w, torch.nn.Parameter) or w.dim() != 2:
            continue
        if not isinstance(m, LinearBase) or getattr(m, "quant_method", None) is None:
            continue                                          # не линейный слой
        N, K = w.shape
        if K % _G or N % 4 or w.dtype not in (torch.float16, torch.bfloat16):
            continue
        коды, масштабы, нули = _квантовать(w.data)
        tm_w, tm_s, meta = ext.prepare(коды, масштабы, нули, _G)
        уп = (tm_w, tm_s, int(meta[0]), int(meta[1]))
        m.fa2sm70_draft_tm8 = уп
        _М = globals().get("_ТМ8Метод")
        if _М is not None:
            m.quant_method = _М(m.quant_method, уп)
        del коды, масштабы, нули
        # Исходный вес больше не нужен: у черновика нет зоны, где плотное тело выгодно.
        m.weight = torch.nn.Parameter(torch.empty(0, K, dtype=w.dtype, device=w.device),
                                      requires_grad=False)
        torch.cuda.empty_cache()
        _С["слоёв"] += 1
        _С["байт"] += N * K
    torch.cuda.empty_cache()
    print(f"[fa2_sm70 DRAFT-TM8] переведено слоёв {_С['слоёв']}, "
          f"{_С['байт'] / (1 << 20):.0f} МиБ весов черновика -> int8+turbomind",
          file=sys.stderr, flush=True)


def установить() -> None:
    if not _ВКЛ:
        return
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.utils.torch_utils import direct_register_custom_op

    def _мм(x: torch.Tensor, tm_w: torch.Tensor, tm_s: torch.Tensor,
            k_ld: int, q_ld: int) -> torch.Tensor:
        m = x.shape[0]
        y = torch.empty((m, tm_s.shape[1]), dtype=x.dtype, device=x.device)
        # Лестница вариантов -- та же, что у цели (замер 25.08): до 16 v6, дальше v4/v12/v2.
        v = 6 if m <= 16 else (4 if m <= 32 else (12 if m <= 96 else 2))
        _расш().gemm(y, x, tm_w, tm_s, _G, k_ld, q_ld, v)
        return y

    def _мм_фейк(x, tm_w, tm_s, k_ld, q_ld):
        return x.new_empty((x.shape[0], tm_s.shape[1]))

    direct_register_custom_op("fa2sm70_draft_tm8_mm", _мм, mutates_args=[],
                              fake_impl=_мм_фейк)

    # [ШОВ -- `apply` У ЭКЗЕМПЛЯРА, А НЕ У КЛАССА. ДВЕ ОШИБКИ ПОДРЯД, ОБЕ ПОЙМАНЫ ЗАМЕРОМ]
    # (1) Подмена `LinearBase.forward` целиком выбрасывала ОБВЯЗКУ TP (all-reduce у row-parallel)
    #     -- падало device-side ассертом далеко от места.
    # (2) Подмена `UnquantizedLinearMethod.apply` У КЛАССА трогает и ЦЕЛЬ: её слои идут через тот
    #     же метод внутри компилируемой области. Замер: сквозняк 45.8 -> 36.3 ток/с при том, что
    #     тело черновика УЛУЧШИЛОСЬ (4.27 -> 3.57 мс). То есть проигрыш пришёл из цели, а не из
    #     черновика -- классовая подмена и есть тот случай, когда «правка не там, где мерил».
    # Отсюда: метод ставится ОТДЕЛЬНОМУ СЛОЮ ЧЕРНОВИКА, и ни один слой цели его не видит.
    class _ТМ8Метод:
        def __init__(self, прежний, уп):
            self._прежний = прежний
            self._уп = уп

        def apply(self, layer, x, bias=None):
            xr = x.reshape(-1, x.shape[-1])
            у = self._уп
            y = torch.ops.vllm.fa2sm70_draft_tm8_mm(xr, у[0], у[1], у[2], у[3])
            if bias is not None:
                y = y + bias
            return y.view(*x.shape[:-1], y.shape[-1])

        def __getattr__(self, имя):            # всё прочее -- прежнему методу
            return getattr(self.__dict__["_прежний"], имя)

    globals()["_ТМ8Метод"] = _ТМ8Метод

    _прежний_load = BaseModelLoader.load_model

    def _load(self, *a, **k):
        model = _прежний_load(self, *a, **k)
        _перевести(model)
        return model

    BaseModelLoader.load_model = _load
    print("[fa2_sm70] ЧЕРНОВИК НА ТЕНЗОРНОМ ТЕЛЕ (int8+turbomind) включён", file=sys.stderr, flush=True)
