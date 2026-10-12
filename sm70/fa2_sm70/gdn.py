# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ЧАНКОВЫЙ СКАН GATED-DELTA-NET НА НАШИХ ЯДРАХ -- ЗАМЕНА fla.chunk_gated_delta_rule.
#
# Подпись и семантика ДОСЛОВНО повторяют
# vllm/model_executor/layers/fla/ops/chunk.py::chunk_gated_delta_rule, потому что подмена делается
# на уровне ИМЕНИ в модуле-потребителе (см. integrations/vllm_sm70/fa2_gdn_patch), а не правкой
# исходников vLLM. Любое расхождение в подписи здесь -- это отказ на боевом пути, а не тест.
#
# ЧТО ОСТАЁТСЯ ОТ TRITON И ПОЧЕМУ. Из шести шагов цепочки заменяются пять. Первый --
# l2norm + chunk_local_cumsum -- НЕ ЗАМЕНЯЕТСЯ: он идёт 0.195 мс на слой, то есть 0.8% скана,
# и на нём нет ни одного умножения матриц, ради которого существуют тензорные ядра. Замена дала бы
# в лучшем случае десятые доли процента, а рисковала бы точностью нормировки, от которой зависит
# ВСЁ остальное. Правило "сперва алгоритм, потом примитивы": оптимизируется цепочка, а не свободные
# слоты в ней.
#
# ГРАНИЦЫ ПРИМЕНИМОСТИ -- ЯВНЫЕ, И ПРИ ВЫХОДЕ ЗА НИХ ВЫЗОВ УХОДИТ К TRITON, А НЕ ПАДАЕТ.
# Ядра инстанцированы под BT=64 и головы 128, потому что это единственная живая форма Qwen3.5.
# Всё, что вне: другой head_dim, batch>1 без cu_seqlens, отсутствие initial_state, fp32-вход.
# Это НЕ заглушка на будущее: без такого отката один неучтённый режим (например, профилировочный
# прогон vLLM с пустыми метаданными) ронял бы сервер на старте.
import os

import torch

from ._ext import gdn_ext

BT = 64                      # размер чанка; на него рассчитаны все пять ядер
_HEAD_DIM = 128


# ------------------------------------------------------------------------------------------------
# Индексы чанков. Форма ДОСЛОВНО как у fla (ops/index.py), но в int32: наши ядра читают их
# скалярной загрузкой на блок, и int64 удвоил бы этот трафик ни за что.
# Кэш -- по (адрес данных, длина, устройство). Ключом НЕЛЬЗЯ брать сам тензор: cu_seqlens
# пересоздаётся каждый шаг, и кэш по объекту не попадал бы ни разу.
# ------------------------------------------------------------------------------------------------
# Кэш ПО ТОЖДЕСТВУ ОБЪЕКТА, а не по содержимому -- ровно как @tensor_cache у fla (utils.py:32,
# сравнение `a is b`). Два соображения, и оба обязательные:
#   * КОРРЕКТНОСТЬ. Ключ по data_ptr НЕЛЬЗЯ: vLLM переиспользует постоянные буферы под
#     query_start_loc, и по одному адресу в следующем шаге лежат ДРУГИЕ границы. Ключ по
#     тождеству + СИЛЬНАЯ ссылка в записи кэша делает переиспользование id невозможным пока
#     запись жива.
#   * СКОРОСТЬ. Построение индексов требует значений на хосте, то есть СИНХРОНИЗАЦИИ с GPU.
#     Скан вызывается 48 раз за проход с ОДНИМ И ТЕМ ЖЕ объектом cu_seqlens -- без кэша это 48
#     синхронизаций на проход. У fla ровно та же конструкция ровно по той же причине.
_idx_cache: list = []
_зн_cache: list = []       # тот же кэш, но по СОДЕРЖИМОМУ cu_seqlens (см. правку 31.08)
_IDX_CACHE_MAX = 8
_С_МЕТА = [0, 0, None, 0]   # попаданий, всего, прошлое содержимое, совпадений содержимого
import os as _os_meta
# Ключ по содержимому. Откат (FA2SM70_GDN_META_CONTENT=0) оставлен для ПАРНОГО замера:
# без него нечем показать, что правка что-то даёт, кроме процента попаданий.
_ПО_СОДЕРЖИМОМУ = _os_meta.environ.get("FA2SM70_GDN_META_CONTENT", "1") == "1"


def _печать_меты():
    н = int(_os_meta.environ.get("FA2SM70_GDN_META_LOG", "0"))
    if н and _С_МЕТА[1] % н == 0:
        import sys as _s
        print(f"[fa2_sm70 GDN мета] кэш индексов: {_С_МЕТА[0]}/{_С_МЕТА[1]} = "
              f"{100 * _С_МЕТА[0] / max(1, _С_МЕТА[1]):.1f} % попаданий; содержимое повторилось "
              f"{_С_МЕТА[3]}/{_С_МЕТА[1]} = {100 * _С_МЕТА[3] / max(1, _С_МЕТА[1]):.1f} %",
              file=_s.stderr, flush=True)


def _chunk_meta(cu_seqlens: torch.Tensor):
    """-> (cu_i32, chunk_indices_i32 [NT,2], chunk_offsets_i32 [N+1], NT)

    Индексы берутся ИЗ fla (prepare_chunk_indices / prepare_chunk_offsets), а не строятся заново.
    Это не лень: любое расхождение в раскладке чанков дало бы ответ, отличающийся от образца по
    причине, не имеющей отношения к ядрам, и искать её пришлось бы в двух местах сразу.
    Наше здесь -- только приведение к int32: ядра читают эти индексы скалярной загрузкой на блок,
    и int64 удвоил бы этот трафик ни за что.
    """
    # [СЧЁТЧИК ПОПАДАНИЙ 31.08] Кэш по ТОЖДЕСТВУ объекта работает, только если движок отдаёт
    # ТОТ ЖЕ тензор. Профиль показал время на строке постройки, то есть на ПРОМАХЕ -- значит
    # надо знать долю промахов, а не гадать. Печать раз в N вызовов, рычаг FA2SM70_GDN_META_LOG.
    _С_МЕТА[1] += 1
    # [КЛЮЧ ПО СОДЕРЖИМОМУ 31.08 -- КЭШ ПО ТОЖДЕСТВУ БЫЛ СЛЕП]
    # Счётчик показал: попаданий 0 из 400, а СОДЕРЖИМОЕ повторяется в 98.5 % вызовов. То есть
    # движок отдаёт КАЖДЫЙ РАЗ НОВЫЙ объект с теми же числами, и кэш по `is` не срабатывает
    # никогда -- индексы строились заново 48 раз за проход, ровно та работа, ради снятия
    # которой кэш и заводился.
    # Порядок проверки: сперва по тождеству (бесплатно), затем по содержимому. Чтение
    # содержимого стоит одной синхронизации, но постройка индексов синхронизируется ВСЁ РАВНО
    # (prepare_chunk_indices читает значения на хосте) -- значит при попадании мы меняем
    # «синхронизация + две постройки + приведение типов» на одну синхронизацию.
    for i, (ref, out) in enumerate(_idx_cache):
        if ref is cu_seqlens:
            if i:                                   # LRU: свежий -- в конец
                _idx_cache.append(_idx_cache.pop(i))
            _С_МЕТА[0] += 1
            _печать_меты()
            return out
    _зн = None
    if _ПО_СОДЕРЖИМОМУ:
        _зн = (int(cu_seqlens.numel()), cu_seqlens.device.type, tuple(cu_seqlens.tolist()))
        if _зн == _С_МЕТА[2]:
            _С_МЕТА[3] += 1
        _С_МЕТА[2] = _зн
        for i, (зн2, out) in enumerate(_зн_cache):
            if зн2 == _зн:
                if i:
                    _зн_cache.append(_зн_cache.pop(i))
                _С_МЕТА[0] += 1
                _печать_меты()
                return out
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    ci = prepare_chunk_indices(cu_seqlens, BT).to(torch.int32).contiguous()
    co = prepare_chunk_offsets(cu_seqlens, BT).to(torch.int32).contiguous()
    cu32 = cu_seqlens.to(torch.int32).contiguous()
    out = (cu32, ci, co, int(ci.shape[0]))
    _idx_cache.append((cu_seqlens, out))
    if _зн is not None:
        _зн_cache.append((_зн, out))
        while len(_зн_cache) > _IDX_CACHE_MAX:
            _зн_cache.pop(0)
    _печать_меты()
    while len(_idx_cache) > _IDX_CACHE_MAX:
        _idx_cache.pop(0)
    return out


_ОТКАЗЫ = {}


def _отказ(причина, q=None, v=None):
    """ОТКАЗ ГЕЙТА -- СОБЫТИЕ, А НЕ ТИШИНА. Молчаливый возврат в чужую реализацию стоил падения
    сервера: тритон на новой форме уходит в АВТОТЮНЕР, а тот выделяет буферы под все конфигурации
    и валит движок по памяти. Причина печатается ОДИН раз на вид, с формой -- чтобы дыру можно было
    закрыть, а не гадать. Счётчик "наших N из N" эту дверь НЕ ВИДИТ: он считает дошедшее до обёртки.
    """
    import logging, os
    к = _ОТКАЗЫ.get(причина, 0)
    _ОТКАЗЫ[причина] = к + 1
    if к == 0 or os.environ.get("FA2SM70_GDN_ОТКАЗЫ") == "все":
        ф = ""
        if q is not None:
            ф = f" q={tuple(q.shape)}/{q.dtype}"
            if v is not None:
                ф += f" v={tuple(v.shape)}/{v.dtype}"
        logging.getLogger("vllm").warning(
            "[fa2_sm70 gdn] ОТКАЗ ГЕЙТА -> ЧУЖОЙ ТРИТОН: %s%s (раз %d)", причина, ф, к + 1)
    return False


def supported(q, k, v, g, beta, initial_state, cu_seqlens) -> bool:
    """Ровно те условия, при которых инстанцированы ядра. Проверяется ДО любой работы."""
    if cu_seqlens is None:
        return _отказ("нет cu_seqlens", q, v)
    if initial_state is None:
        return _отказ("нет начального состояния", q, v)
    # Развёртка бывает и fp16, и bf16 (этот сервер поднимается с --dtype float16). Обе годны:
    # при fp16 конверсия входа не делается ВООБЩЕ, при bf16 -- слитым проходом внутри расширения.
    if q.dtype != k.dtype or v.dtype != q.dtype or beta.dtype != v.dtype:
        return _отказ(f"разные типы q/k/v/beta: {q.dtype}/{k.dtype}/{v.dtype}/{beta.dtype}", q, v)
    if q.dtype not in (torch.bfloat16, torch.float16):
        return _отказ(f"тип входа {q.dtype}", q, v)
    if q.shape[0] != 1:
        return _отказ(f"батч {q.shape[0]} != 1", q, v)
    if q.shape[-1] != _HEAD_DIM or v.shape[-1] != _HEAD_DIM:
        return _отказ(f"head_dim {q.shape[-1]}/{v.shape[-1]} != {_HEAD_DIM}", q, v)
    H, Hg = v.shape[-2], q.shape[-2]
    if H % Hg != 0 or (H // Hg) not in (1, 2, 3, 4, 6, 8):
        return _отказ(f"отношение голов {H}/{Hg}", q, v)
    if g.dtype != torch.float32:
        return _отказ(f"g {g.dtype} != fp32", q, v)
    # Начальное состояние ОБЯЗАНО быть fp32. У vLLM это значение по умолчанию ("GDN recurrent
    # state needs float32"), но mamba_ssm_cache_dtype его переопределяет -- и тогда наше ядро
    # прочло бы bf16-буфер КАК float, то есть выдало бы мусор БЕЗ ЕДИНОЙ ошибки. Проверяем тип,
    # а не полагаемся на умолчание.
    # Состояние: fp32 ИЛИ fp16 (ядро читает/пишет по StateT, конверсии на шве нет).
    # Проверка ТИПА остаётся: прочесть half-буфер КАК float -- это мусор без единой ошибки.
    if initial_state.dtype not in (torch.float32, torch.float16):
        return _отказ(f"состояние {initial_state.dtype} вне fp32/fp16", q, v)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(q.device)[0] != 7:
        return _отказ("не sm_70", q, v)
    return True


def chunk_gated_delta_rule_volta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    out_dtype: torch.dtype,
):
    """Все пять ядер за один вызов расширения. q,k -- УЖЕ fp16 (их выдаёт l2norm), g -- fp32."""
    cu32, ci, co, _NT = _chunk_meta(cu_seqlens)
    o, ht = gdn_ext().gdn_chunk_scan(
        q, k, v.contiguous(), g.contiguous(), beta.contiguous(),
        initial_state.contiguous() if initial_state is not None else None,
        cu32, ci, co, float(scale), bool(out_dtype == torch.bfloat16),
        bool(output_final_state))
    # Контракт образца: при output_final_state=False возвращается None, а не пустой тензор.
    return o, (ht if output_final_state else None)


# ------------------------------------------------------------------------------------------------
# ПОШАГОВАЯ СВЕРКА. Держится РЯДОМ с боевой функцией намеренно: когда цепочка из пяти ядер разойдётся
# с образцом, вопрос "на каком шаге" обязан отвечаться одной командой, а не бисекцией по коду.
# ------------------------------------------------------------------------------------------------
def steps(q, k, v, g, beta, initial_state, cu_seqlens, scale, out_bf16=False):
    """Возвращает словарь всех промежуточных величин цепочки (A, Ai, w, u, h, v_new, ht, o)."""
    ext = gdn_ext()
    cu32, ci, co, NT = _chunk_meta(cu_seqlens)
    H = v.shape[-2]
    v16 = v.to(torch.float16).contiguous()
    b16 = beta.to(torch.float16).contiguous()
    A = ext.gdn_kkt(k, b16, g, cu32, ci, H)
    Ai = ext.gdn_solve_tril(A, cu32, ci)
    w, u = ext.gdn_wu(k, v16, b16, g, Ai, cu32, ci)
    h, v_new, ht = ext.gdn_h(k, w, u, g, initial_state, cu32, co, NT, True)
    o = ext.gdn_o(q, k, v_new, h, g, cu32, ci, float(scale), bool(out_bf16))
    return dict(A=A, Ai=Ai, w=w, u=u, h=h, v_new=v_new, ht=ht, o=o, v16=v16, b16=b16)


def enabled() -> bool:
    """Выключатель для A/B. Гейт врезки проверяется ПОВЕДЕНИЕМ, а не этой строкой."""
    return os.environ.get("FA2SM70_GDN", "1") != "0"
