# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ПЕРЕКРЫТИЕ ОБМЕНА ВЫЧИСЛЕНИЕМ (задача 201): свёртка одной половины токенов едет по проводу,
# пока вторая половина считается на карте.
#
# ЗАЧЕМ. Разложение 21.08 (задача 200): на 128K тензорный префилл 156.9 с, из них ОБМЕН 80.5 с
# (26.7 с после включения P2P). Fable (research/fable/third_way_answer.md) доказал, что уменьшить
# ОБЪЁМ переносов нельзя -- 2h на границу это теорема (cut-rank), -- но там же сказано главное:
# работа провода (17.9-26.7 с) МЕНЬШЕ работы произведений (76.4 с), то есть информационно она
# ЦЕЛИКОМ помещается в тень. Стена не в расстановке, а в отсутствующем примитиве перекрытия:
# torch.ops.symm_mem на sm_70 не поднимается ("Device capability 7.0 not supported"), а
# sequence parallelism без него даёт РОВНО НОЛЬ (157.1 против 156.9, замерено).
#
# ПОЧЕМУ ЭТО РАБОТАЕТ ИМЕННО ТАК. В тензорном разрезе свёртка нужна не везде: qkv и gate_up
# режутся ПО ВЫХОДУ (column-parallel) и свёртки не требуют вовсе -- значит K/V пишутся в пул
# локально, и вторая половина токенов может считать внимание, НЕ ДОЖИДАЯСЬ свёртки первой.
# Свёртка нужна только после o_proj и down_proj (row-parallel), то есть дважды на блок. Их и
# сдвигаем.
#
# ЧТО ЗДЕСЬ ЕСТЬ. Механика отложенной свёртки: `запустить` кладёт работу на ОТДЕЛЬНЫЙ поток и
# отдаёт обещание, `дождаться` его забирает. Сам обмен -- наш i12 (fa2sm70_exch_i12), потому что
# он уже перехватывает all_reduce выше коммуникатора устройства и везёт 12 разрядов вместо 16.
#
# ЧЕГО ЗДЕСЬ НЕТ И ПОЧЕМУ. Перекрытия ВНУТРИ самого обмена нет намеренно: упаковка и распаковка
# вместе стоят ~0.9 мс из 26 мс передачи (посчитано по трафику 294 + 420 МБ при 813 ГБ/с), то
# есть прятать в них нечего. Прячется только сама передача, и только за чужим вычислением.
import os
import sys

import torch

ВКЛ = os.environ.get("FA2SM70_OVERLAP", "0") == "1"
_поток = None
_счёт = {"пусков": 0, "ожиданий": 0, "тень_мс": 0.0}


def _вторичный():
    """Отдельный поток для обмена. ОДИН на процесс: два потока обмена дерутся за провод и
    сериализуются внутри NCCL, а мы платим за это лишними событиями."""
    global _поток
    if _поток is None:
        _поток = torch.cuda.Stream()
    return _поток


class Обещание:
    """Отложенная свёртка. Держит СВОЙ буфер и события; пока не забрали -- значение не готово.

    ВАЖНО ПРО ВРЕМЯ ЖИЗНИ: тензор, отданный на обмен, обязан дожить до `дождаться`, иначе
    аллокатор отдаст его память другому потоку и обмен затрёт чужое. Поэтому ссылка держится
    здесь, а не у вызывающего.
    """

    __slots__ = ("_вход", "_выход", "_готово", "_забрано")

    def __init__(self, вход):
        self._вход = вход
        self._выход = None
        self._готово = torch.cuda.Event()
        self._забрано = False

    def забрать(self):
        if self._забрано:
            raise RuntimeError("[fa2_sm70 перекрытие] обещание забрано дважды")
        self._забрано = True
        # Ждём НА ОСНОВНОМ ПОТОКЕ, а не синхронизацией процесса: синхронизация здесь съела бы
        # ровно ту тень, ради которой всё и затевалось (урок «барьер ДО отсчёта»).
        torch.cuda.current_stream().wait_event(self._готово)
        _счёт["ожиданий"] += 1
        return self._выход


def запустить(тензор, свёртка):
    """Положить свёртку на вторичный поток и вернуть обещание.

    `свёртка` -- вызываемое, делающее собственно обмен (у нас это перехваченный all_reduce).
    """
    об = Обещание(тензор)
    поток = _вторичный()
    # Вторичный поток обязан дождаться, пока основной ДОСЧИТАЕТ вход: иначе обмен уедет с
    # недописанными числами, и это будет тихая порча, а не падение.
    поток.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(поток):
        об._выход = свёртка(тензор)
        об._готово.record(поток)
    # Аллокатор обязан знать, что буфер живёт на ДВУХ потоках, иначе переиспользует его раньше.
    тензор.record_stream(поток)
    _счёт["пусков"] += 1
    if _счёт["пусков"] & (_счёт["пусков"] - 1) == 0:
        print(f"[fa2_sm70 перекрытие] пусков {_счёт['пусков']}, ожиданий {_счёт['ожиданий']}",
              file=sys.stderr, flush=True)
    return об


def годится(тензор) -> bool:
    """Перекрывать имеет смысл только там, где обмен ЗАМЕТЕН на фоне вычисления.

    Порог по числу элементов, а не по форме: при коротком куске обмен вырождается в латентность,
    и вторичный поток добавит больше событий, чем спрячет байт.
    """
    if not ВКЛ or тензор is None:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False          # в графе поток и события чужие -- не лезем
    return тензор.numel() >= int(os.environ.get("FA2SM70_OVERLAP_MIN", str(1 << 21)))


# ---------------------------------------------------------------------------------------------
# ВРЕЗКА В ПУТЬ: слой разрезан на части, две половины токенов идут со сдвигом.
#
# ПОЧЕМУ РЕЗАТЬ МОЖНО БЕЗ ПРАВКИ ЧУЖИХ ТЕЛ. У RowParallelLinear свёртка стоит под флагом
# `reduce_results` (linear.py:1611). Выключив его на o_proj / down_proj / out_proj, получаем
# ЧАСТИЧНЫЙ выход, а свёртку делаем сами -- и тогда её можно отложить на поток. Ничего в чужом
# коде менять не нужно, меняется только флаг на наших модулях.
#
# ПОРЯДОК В СЛОЕ (повторён дословно, иначе тихая порча):
#   вход -> input_layernorm -> внимание(ЧАСТИЧНОЕ) -> [СВЁРТКА] -> масштаб_вним ->
#   post_attention_layernorm -> mlp(ЧАСТИЧНЫЙ) -> [СВЁРТКА] -> масштаб_ffn
# Масштабы применяются ПОСЛЕ свёртки -- проверено по исходнику, до неё они дали бы другой ответ.
_свёртка = None


def _взять_свёртку():
    global _свёртка
    if _свёртка is None:
        from vllm.distributed import tensor_model_parallel_all_reduce as _ar
        _свёртка = _ar
    return _свёртка


def _вним_часть(слой, h, residual, positions):
    if residual is None:
        residual = h
        h = слой.input_layernorm(h)
    else:
        h, residual = слой.input_layernorm(h, residual)
    out = torch.empty_like(h)
    if слой.layer_type == "linear_attention":
        слой.linear_attn(hidden_states=h, output=out)
    else:
        слой.self_attn(hidden_states=h, output=out, positions=positions)
    return out, residual


def _млп_часть(слой, h, residual):
    if слой.layer_scale:
        h = h * (слой._attn_scale_p1 if h.dim() == 2 else слой._attn_scale_p1.unsqueeze(0))
    h, residual = слой.post_attention_layernorm(h, residual)
    return слой.mlp(h), residual


def _масштаб_ffn(слой, h):
    if слой.layer_scale:
        h = h * (слой._ffn_scale_p1 if h.dim() == 2 else слой._ffn_scale_p1.unsqueeze(0))
    return h


def приготовить(модель) -> int:
    """Выключить свёртку у row-parallel модулей: дальше её делаем мы, и делаем отложенно."""
    n = 0
    for имя, м in модель.named_modules():
        if имя.endswith((".o_proj", ".down_proj", ".out_proj")) and getattr(м, "reduce_results", False):
            м.reduce_results = False
            n += 1
    return n


def проход(слои, hidden_states, positions, residual=None):
    """Две половины токенов через все слои со СДВИГОМ: свёртка одной едет, вторая считается.

    Половины режутся ПО ТОКЕНАМ, и это законно: причинная маска позволяет второй половине
    видеть первую, а K/V пишутся в пул ДО свёртки (qkv -- column-parallel, свёртки не требует),
    поэтому вторая половина не ждёт первую.
    """
    ar = _взять_свёртку()
    T = hidden_states.shape[0]
    гр = (T // 2 // 128) * 128                    # граница кратна 128: упаковщик i12 берёт группы
    if гр == 0 or T - гр == 0:
        return None                               # делить нечего -- пусть идёт обычный путь
    A, B = hidden_states[:гр], hidden_states[гр:]
    pA, pB = (positions[:гр], positions[гр:]) if positions is not None else (None, None)
    rA = rB = None
    if residual is not None:
        rA, rB = residual[:гр], residual[гр:]
    for слой in слои:
        aA, rA = _вним_часть(слой, A, rA, pA)
        обA = запустить(aA, ar)                   # свёртка A уехала
        aB, rB = _вним_часть(слой, B, rB, pB)     # B считается, ПОКА едет A
        обB = запустить(aB, ar)
        mA, rA = _млп_часть(слой, обA.забрать(), rA)
        об2A = запустить(mA, ar)
        mB, rB = _млп_часть(слой, обB.забрать(), rB)
        об2B = запустить(mB, ar)
        A = _масштаб_ffn(слой, об2A.забрать())
        B = _масштаб_ffn(слой, об2B.забрать())
    return torch.cat([A, B], dim=0), torch.cat([rA, rB], dim=0)


# ---------------------------------------------------------------------------------------------
# РАЗРЕЗ МЕТАДАННЫХ ВНИМАНИЯ.
#
# ЗАЧЕМ ОТДЕЛЬНО. Метаданные описывают ВЕСЬ кусок из T токенов. Подать половину с чужими
# метаданными -- внимание посчитает не то и НЕ УПАДЁТ: query_start_loc скажет «токенов T»,
# slot_mapping отправит записи не в те страницы. Это ровно тот класс, где ошибка выглядит как
# работающий сервер с чуть другим ответом.
#
# ГРАНИЦА ПРИМЕНИМОСТИ -- ОДИН ЗАПРОС В КУСКЕ. При батче границы запросов не совпадают с
# границей половин, и резать пришлось бы по каждому запросу отдельно; выигрыш там мал (обмен на
# коротком куске вырождается в латентность), а риск велик. Поэтому один запрос -- и отказ иначе.
def резать_метаданные(м, гр, всего):
    """Вернуть (мета_A, мета_B) или None, если разрез неприменим."""
    if getattr(м, "num_reqs", 1) != 1:
        return None
    try:
        import dataclasses
        qsl = м.query_start_loc
        A = dataclasses.replace(
            м,
            num_actual_tokens=гр,
            max_query_len=гр,
            query_start_loc=torch.tensor([0, гр], dtype=qsl.dtype, device=qsl.device),
            query_start_loc_cpu=torch.tensor([0, гр], dtype=torch.int32),
            # ПОСЛЕДНИЙ ТОКЕН ПОЛОВИНЫ A -- НЕ последний токен куска: длина последовательности
            # для A короче на хвост, иначе внимание A увидит ключи, которых ещё нет.
            seq_lens=м.seq_lens - (всего - гр),
            seq_lens_cpu=(м.seq_lens_cpu - (всего - гр)) if м.seq_lens_cpu is not None else None,
            slot_mapping=м.slot_mapping[:гр],
        )
        B = dataclasses.replace(
            м,
            num_actual_tokens=всего - гр,
            max_query_len=всего - гр,
            query_start_loc=torch.tensor([0, всего - гр], dtype=qsl.dtype, device=qsl.device),
            query_start_loc_cpu=torch.tensor([0, всего - гр], dtype=torch.int32),
            seq_lens=м.seq_lens,
            seq_lens_cpu=м.seq_lens_cpu,
            slot_mapping=м.slot_mapping[гр:],
        )
        return A, B
    except Exception as e:                       # НЕ глотать молча: отказ разреза обязан быть виден
        print(f"[fa2_sm70 перекрытие] разрез метаданных НЕ УДАЛСЯ: {e}", file=sys.stderr, flush=True)
        return None


# ---------------------------------------------------------------------------------------------
# ПОДМЕНА МЕТАДАННЫХ НА ПОЛОВИНУ ВНУТРИ УЖЕ УСТАНОВЛЕННОГО КОНТЕКСТА.
#
# Контекст ставится ВЫШЕ (в раннере), и заново его открывать нельзя -- там висит и режим графа,
# и учёт размеров партии. Поэтому подменяем ПОЛЕ, а по выходе кладём назад: контекст один и тот
# же объект, порча его переживёт запрос.
import contextlib


@contextlib.contextmanager
def _половина_метаданных(мета):
    """Подменить метаданные на половину. `мета` -- СЛОВАРЬ (ключ слоя -> свой срез) либо один
    объект; в гибридной сети всегда словарь, потому что типы у слоёв разные."""
    from vllm.forward_context import get_forward_context
    ctx = get_forward_context()
    было = ctx.attn_metadata
    ctx.attn_metadata = мета
    try:
        yield
    finally:
        ctx.attn_metadata = было


def проход_с_метаданными(слои, hidden_states, positions, residual, метаA, метаB, гр,
                         отложенно=True):
    """То же расписание, что и `проход`, но каждая половина считается СО СВОИМИ метаданными.

    `отложенно=False` -- ФАЛЬСИФИКАТОР: тот же ручной цикл, но свёртка делается СРАЗУ, без
    второго потока. Он отделяет цену САМОГО ручного прохода (потеря компиляции и графов, две
    половины вместо целого) от пользы перекрытия. Без этого разделения «стало хуже на 12.5 %»
    остаётся суммой двух неразличимых слагаемых.
    """
    ar = _взять_свёртку()

    class _Сразу:
        __slots__ = ("_v",)

        def __init__(self, v):
            self._v = v

        def забрать(self):
            return self._v

    _пуск = (запустить if отложенно else (lambda t, f: _Сразу(f(t))))
    A, B = hidden_states[:гр], hidden_states[гр:]
    pA, pB = (positions[:гр], positions[гр:]) if positions is not None else (None, None)
    rA = rB = None
    if residual is not None:
        rA, rB = residual[:гр], residual[гр:]
    for слой in слои:
        with _половина_метаданных(метаA):
            aA, rA = _вним_часть(слой, A, rA, pA)
        обA = _пуск(aA, ar)                           # свёртка A уехала по проводу
        with _половина_метаданных(метаB):
            aB, rB = _вним_часть(слой, B, rB, pB)     # B считается, ПОКА едет A
        обB = _пуск(aB, ar)
        mA, rA = _млп_часть(слой, обA.забрать(), rA)  # MLP метаданных не читает
        об2A = _пуск(mA, ar)
        mB, rB = _млп_часть(слой, обB.забрать(), rB)
        об2B = _пуск(mB, ar)
        A = _масштаб_ffn(слой, об2A.забрать())
        B = _масштаб_ffn(слой, об2B.забрать())
    return torch.cat([A, B], dim=0), torch.cat([rA, rB], dim=0)


# ---------------------------------------------------------------------------------------------
# РАЗРЕЗ МЕТАДАННЫХ РЕКУРРЕНТНЫХ СЛОЁВ (GDN).
#
# ПОЧЕМУ ОТДЕЛЬНО ОТ ВНИМАНИЯ. В гибридной сети attn_metadata -- СЛОВАРЬ ИЗ ДВУХ ТИПОВ: у слоёв
# полного внимания FA2SM70Metadata, у рекуррентных GDNAttentionMetadata со своей структурой.
# Первая попытка резала по одному типу и брала первый попавшийся ключ -- попадался GDN, и разрез
# отказывал. Ловил это счётчик, а не выходы: хэши ТРИЖДЫ совпадали с эталоном при неработающей
# ветке.
#
# СУТЬ РАЗРЕЗА, И ОНА НЕ КОСМЕТИЧЕСКАЯ. Половины идут ПО ОЧЕРЕДИ, поэтому для второй половины
# состояние УЖЕ ОБНОВЛЕНО первой: у неё has_initial_state=True (даже если у куска его не было)
# и число посчитанных токенов больше на границу. Забыть это -- значит тихо пересчитать вторую
# половину с нуля: ответ останется правдоподобным, а длинная генерация поедет.
_причины = {}


def _почему(текст):
    """Печать причины отказа по СТЕПЕНЯМ ДВОЙКИ: в горячем пути строка на вызов вешает сервер."""
    n = _причины[текст] = _причины.get(текст, 0) + 1
    if n & (n - 1) == 0:
        print(f"[fa2_sm70 перекрытие] разрез отклонён ({n}): {текст}", file=sys.stderr, flush=True)


def резать_гдн(м, гр, всего, ncomp_извне=None):
    """Вернуть (мета_A, мета_B) для рекуррентных слоёв или None, если разрез неприменим."""
    try:
        import dataclasses
        # ПРИЧИНА ОТКАЗА ПЕЧАТАЕТСЯ. Молчаливый `return None` здесь уже стоил захода: счётчик
        # показывал ноль пусков, а почему -- было не видно. Молчащий отказ неотличим от того,
        # что ветку не звали.
        _пр, _дек = getattr(м, "num_prefills", 0), getattr(м, "num_decodes", 0)
        if _пр != 1 or _дек != 0:
            _почему(f"не чистый префилл: prefills={_пр} decodes={_дек}")
            return None
        if getattr(м, "num_spec_decodes", 0):
            _почему("идёт спекуляция -- два расписания сразу не смешиваем")
            return None
        qsl = м.non_spec_query_start_loc
        уст = qsl.device if qsl is not None else None

        def _qsl(n):
            return torch.tensor([0, n], dtype=qsl.dtype, device=уст) if qsl is not None else None

        ncomp = м.num_computed_tokens_ns
        ncomp_cpu = м.num_computed_tokens_ns_cpu
        if ncomp is None and ncomp_извне is not None:
            # ПОЛЕ ПУСТОЕ -- НО ЧИСЛО ВЫВОДИМО. Сколько токенов запроса уже посчитано, знают и
            # метаданные ВНИМАНИЯ из того же словаря: seq_lens - num_actual_tokens. Отказываться
            # из-за пустого поля значило бы принять отсутствие ИМЕНИ за отсутствие ВЕЛИЧИНЫ.
            ncomp = ncomp_извне
            ncomp_cpu = ncomp_извне.to("cpu") if hasattr(ncomp_извне, "to") else ncomp_извне
        if ncomp is None or ncomp_cpu is None:
            _почему("нет num_computed_tokens_ns и вывести неоткуда")
            return None

        # ИНДЕКСНЫЕ ПОЛЯ НЕ РЕЖУТСЯ, А ПЕРЕСЧИТЫВАЮТСЯ ТЕМИ ЖЕ ФОРМУЛАМИ, что и в билдере
        # (gdn_attn.py:230-237). Первая попытка их не тронула вовсе -- ядро полезло за границы
        # и упало с illegal memory access. Это лучший из плохих исходов: для рекуррентного
        # состояния тихая порча была бы дороже падения.
        from vllm.utils.math_utils import cdiv as _cdiv
        _бл = getattr(м, "mamba_block_size", 0)

        def _блоки(ctx, seq):
            if not _бл:
                return None, None, None
            return (torch.clamp(_cdiv(ctx, _бл) - 1, min=0),
                    _cdiv(ctx + 1, _бл) - 1,
                    torch.clamp(_cdiv(seq, _бл) - 1, min=0))

        # Сколько токенов ВСЕГО у запроса после куска -- нужно для последнего запланированного блока.
        _seq_A = ncomp + гр
        _seq_B = ncomp + всего
        _lcA, _fsA, _lsA = _блоки(ncomp, _seq_A)
        _lcB, _fsB, _lsB = _блоки(ncomp + гр, _seq_B)

        # Метаданные причинной свёртки строятся ИЗ query_start_loc -- значит для половины их надо
        # построить заново, а не унаследовать: иначе свёртка возьмёт чужие смещения.
        try:
            from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata as _ccm
            _цA = _ccm(torch.tensor([0, гр], dtype=torch.int32),
                       device=qsl.device if qsl is not None else None)
            _цB = _ccm(torch.tensor([0, всего - гр], dtype=torch.int32),
                       device=qsl.device if qsl is not None else None)
        except Exception:
            _цA = _цB = (None, None, None)

        A = dataclasses.replace(
            м, num_actual_tokens=гр, num_prefill_tokens=гр,
            non_spec_query_start_loc=_qsl(гр),
            block_idx_last_computed_token=_lcA,
            block_idx_first_scheduled_token=_fsA,
            block_idx_last_scheduled_token=_lsA,
            nums_dict=_цA[0], batch_ptr=_цA[1], token_chunk_offset_ptr=_цA[2],
        )
        B = dataclasses.replace(
            м, num_actual_tokens=всего - гр, num_prefill_tokens=всего - гр,
            non_spec_query_start_loc=_qsl(всего - гр),
            # ВТОРАЯ ПОЛОВИНА ПРОДОЛЖАЕТ ПЕРВУЮ: состояние есть всегда, посчитано на `гр` больше.
            has_initial_state=(torch.ones_like(м.has_initial_state)
                               if м.has_initial_state is not None else None),
            num_computed_tokens_ns=ncomp + гр,
            num_computed_tokens_ns_cpu=ncomp_cpu + гр,
            block_idx_last_computed_token=_lcB,
            block_idx_first_scheduled_token=_fsB,
            block_idx_last_scheduled_token=_lsB,
            nums_dict=_цB[0], batch_ptr=_цB[1], token_chunk_offset_ptr=_цB[2],
        )
        return A, B
    except Exception as e:
        print(f"[fa2_sm70 перекрытие] разрез ГДН НЕ УДАЛСЯ: {e}", file=sys.stderr, flush=True)
        return None


def резать_словарь(мета_словарь, гр, всего):
    """Разрезать ВЕСЬ словарь метаданных послойно: каждому ключу -- свой тип разреза."""
    A, B = {}, {}
    # СНАЧАЛА берём число посчитанных токенов у ВНИМАНИЯ: у рекуррентных метаданных это поле
    # бывает пустым, а величина одна и та же на весь кусок.
    _ncomp = None
    for _м in мета_словарь.values():
        if not type(_м).__name__.startswith("GDN") and getattr(_м, "seq_lens", None) is not None:
            _ncomp = _м.seq_lens - всего
            break
    for ключ, м in мета_словарь.items():
        if type(м).__name__.startswith("GDN"):
            п = резать_гдн(м, гр, всего, _ncomp)
        else:
            п = резать_метаданные(м, гр, всего)
        if п is None:
            return None                      # хоть один не режется -- отказ целиком, без полумер
        A[ключ], B[ключ] = п
    return A, B
