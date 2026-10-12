# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec



# [FA2/SM70 25.08] ФАЗОМЕР СТРОИТЕЛЯ (FA2SM70_GDN_PHASE=N). Замер показал: этот строитель стоит
# 9.9 мс НА ШАГ при спекуляции -- больше, чем всё остальное вне прохода. Надо назвать УЧАСТОК.
import os as _os
import time as _time

_ГФ = {}
_ГС = [0]
# [ПРИБОР НЕ ДОЛЖЕН СТОИТЬ, КОГДА ВЫКЛЮЧЕН, 31.08]
# `_гф` зовётся ВОСЕМЬ раз в каждом `build`, а `build` -- на каждую группу KV каждого шага
# (у нас их десять-двенадцать): около девяноста вызовов за шаг. В каждом стоял
# `int(os.environ.get(...))` -- обращение к словарю окружения плюс разбор строки, и всё это
# при ВЫКЛЮЧЕННОМ приборе. Рычаг читаем один раз при импорте.
_ГФ_ВКЛ = int(_os.environ.get("FA2SM70_GDN_PHASE", "0"))
# [ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ ХОЗЯЙСКОГО ПУТИ -- 08.09]
# Прежде чем переписывать построение метаданных на нативное ядро (пункт 5 цели, ~2.2 мс),
# надо ответить на вопрос, который дороже самой правки: ЛЕЖИТ ЛИ хозяйская работа на
# критическом пути? Если поток хозяина всё равно ждёт карту, снятие питона даст НОЛЬ.
# Ответ даёт положительный контроль: ДОБАВИТЬ хозяйской работы и посмотреть на шаг.
# Ветвь выбирается по файлу (чередование внутри одного экземпляра, §212), задержка -- в
# микросекундах. Пустой путь = ветка мертва.
_ЗАДЕРЖКА_ФАЙЛ = _os.environ.get("FA2SM70_HOST_DELAY_FILE", "")
_ЗАДЕРЖКА_МКС = int(_os.environ.get("FA2SM70_HOST_DELAY_US", "0"))
_ЗАД = {"вкл": False, "счёт": 0}


def _задержка_хозяина():
    if not _ЗАДЕРЖКА_ФАЙЛ or _ЗАДЕРЖКА_МКС <= 0:
        return
    _ЗАД["счёт"] += 1
    if _ЗАД["счёт"] % 64 == 1:
        # Содержимое файла -- ЧИСЛО микросекунд (0 = ветка мертва). Так одна сборка даёт
        # весь свип величин, а не одну; чередование с нулём снимает дрейф стенда.
        try:
            with open(_ЗАДЕРЖКА_ФАЙЛ) as ф:
                _т = ф.read(16).strip()
            _ЗАД["мкс"] = int(_т) if _т.lstrip("-").isdigit() else (
                _ЗАДЕРЖКА_МКС if _т[:1] == "1" else 0
            )
        except (OSError, ValueError):
            _ЗАД["мкс"] = 0
    _м = _ЗАД.get("мкс", 0)
    if _м > 0:
        _к = _time.perf_counter() + _м * 1e-6
        while _time.perf_counter() < _к:
            pass


def _гф(имя, т0):
    if not _ГФ_ВКЛ:
        return
    _ГФ[имя] = _ГФ.get(имя, 0.0) + (_time.perf_counter() - т0) * 1e3


def _гпечать():
    н = _ГФ_ВКЛ
    if not н:
        return
    _ГС[0] += 1
    if _ГС[0] % н == 0:
        c = _ГС[0]
        стр = "  ".join(f"{и}={в / c:.2f}" for и, в in sorted(_ГФ.items(), key=lambda x: -x[1]))
        print(f"[gdn ФАЗЫ СТРОИТЕЛЯ, мс/вызов, {c}] {стр}", flush=True)

class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    # --- РЕЖИМ 'all' (префикс-кэш ДЛЯ РЕКУРРЕНТНЫХ СЛОЁВ), задача 194 ---------------------
    # В режимах 'none'/'align' на запрос приходится ОДНО состояние: индекс одномерный, и
    # восстановить состояние в середине последовательности нечем -- поэтому со спекуляцией
    # 'align' портит выход (проверено гейтом тождественности 19.08, research/23).
    # В режиме 'all' состояние сохраняется НА КАЖДОЙ ГРАНИЦЕ БЛОКА, поэтому:
    #   * non_spec_state_indices_tensor становится ДВУМЕРНЫМ (запрос x блоки);
    #   * block_idx_* -- указатели В ЭТУ таблицу: откуда взять начальное состояние
    #     (последний ПОСЧИТАННЫЙ токен) и куда положить промежуточные и финальное.
    # Механизм и имена -- ровно как у Mamba2 (mamba_attn.py), чтобы два тела не разошлись.
    # ВСЕ ЧЕТЫРЕ -- ПО НЕ-СПЕКУЛЯТИВНЫМ ЗАПРОСАМ ЦЕЛИКОМ (сперва декоды, затем префиллы), а
    # не по одним префиллам, как у Mamba2. Причина в устройстве GDN: когда в батче есть хоть
    # один префилл, чанковый скан обрабатывает ВЕСЬ не-спекулятивный батч разом, и декодные
    # запросы идут тем же вызовом. Срез «только префиллы» дал бы им чужие блоки -- отказа не
    # будет, будет тихая порча.
    # [FA2/SM70 25.08] Длины нужны, чтобы пересобрать индексы состояний под таблицу блоков ДРУГОЙ
    # группы (см. update_block_table): у Mamba2 они в метаданных есть, у GDN их не было.
    seq_lens_для_обновления: torch.Tensor | None = None
    block_idx_last_computed_token: torch.Tensor | None = None
    block_idx_last_scheduled_token: torch.Tensor | None = None
    block_idx_first_scheduled_token: torch.Tensor | None = None
    num_computed_tokens_ns: torch.Tensor | None = None
    mamba_block_size: int = 0
    # [СВЁРТКА У ГРАНИЦЫ, 04.09] Колонки полной таблицы блоков: откуда спекулятивная свёртка
    # берёт состояние и куда его кладёт. Совпадают всюду, кроме шага, где опора сдвинулась.
    spec_conv_блоки: torch.Tensor | None = None
    spec_conv_чт: torch.Tensor | None = None
    spec_conv_зап: torch.Tensor | None = None
    # [ДЕРЕВО] nacc со сдвигом +W для ЧТЕНИЯ SSM при принятой ветви B (свёртке -- нельзя).
    num_accepted_ssm: torch.Tensor | None = None
    num_accepted_conv: torch.Tensor | None = None
    # Копии на стороне процессора -- ТОЛЬКО для префилла: чтобы разрезать вызов скана по
    # границам блоков, нужны сами числа, а не тензоры на карте. Стоят одной синхронизации на
    # шаг префилла (десятки миллисекунд работы) и НЕ трогаются в декоде, где синхронизация
    # ломала бы полный граф.
    num_computed_tokens_ns_cpu: torch.Tensor | None = None
    non_spec_query_start_loc_cpu: torch.Tensor | None = None


# [FA2/SM70 25.08] ОБЩИЙ СЧЁТ ГРУПП. У модели ДЕСЯТЬ отдельных KV-групп GDN (по 4-5 слоёв), и
# строитель метаданных зовётся по разу на группу: замерено 6300 вызовов на 640 шагов = ~10 за шаг,
# 9.9 мс -- больше, чем всё остальное вне прохода. При этом входные метаданные у групп РАЗЛИЧАЮТСЯ
# ТОЛЬКО таблицей блоков и слотами: маски спекуляции, счётчики, индексы токенов и начала запросов
# совпадают по построению. Считаем их ОДИН раз на шаг и раздаём; блоко-зависимые тензоры каждая
# группа по-прежнему считает СВОИ (иначе состояние читалось бы из чужих блоков -- тихая порча).
# Ключ -- тождество объектов входа (у групп это буквально один и тот же тензор).
_ОБЩЕЕ = {"ключ": None, "знач": None}
# [РЫЧАГ -- КОНСТАНТА МОДУЛЯ, А НЕ ЧТЕНИЕ НА КАЖДЫЙ ВЫЗОВ, 31.08]
# `build` исполняется на КАЖДУЮ группу KV каждого шага (у нас их десять-двенадцать), и внутри
# стояли ДВА `os.environ.get` -- то есть двадцать с лишним обращений к словарю окружения с
# разбором строки за шаг. Окружение в рантайме не меняется, поэтому читаем один раз при импорте.
import sys as _sys
_ГРАН_ДИАГ = __import__("os").environ.get("FA2SM70_GDN_GRAN_DIAG","0")=="1"
_СЧЁТ: dict = {}
# Подстановка блока-источника в spec-таблицу (лечение границы блока состояния).
# Держится на ИСТОРИИ: куда прошлый шаг записал состояние. История сверяется с ctx,
# и при любом несовпадении берётся формула без истории -- порчи быть не может.
_ГРАН_ИСТОК = __import__("os").environ.get("FA2SM70_GRAN_SRC","0")=="1"
# [ПОДСТАНОВКА КАЖДЫЙ ШАГ -- 09.09] Окно «граница рядом» считается по ПРОЦЕССОРНЫМ длинам, а
# при асинхронном планировании они отстают на шаг: окно промахивается мимо границы, и обрыв
# возвращается. Комментарий ниже (строка про FA2SM70_GRAN_ALWAYS) обещал этот рычаг, но в коде
# его НЕ БЫЛО -- только на словах. Здесь он заведён: окно снимается, подстановка идёт всегда.
# Цена -- работа на каждом шаге вместо ~16 из 4096; мерить парно.
_ГРАН_ВСЕГДА = __import__("os").environ.get("FA2SM70_GRAN_ALWAYS","0")=="1"
# Подстановка блока-ПРИЁМНИКА: состояние на КОНЕЦ блока обязано лечь в сам блок, иначе
# префикс-кэш переиспользует недосчитанное состояние (спекулятивное ядро кладёт в блок
# состояние ПЕРВОЙ позиции шага, а не последней позиции блока).
_ГРАН_ПРИЁМ = __import__("os").environ.get("FA2SM70_GRAN_DST","0")=="1"
_ГРАН_КОНВ0 = __import__("os").environ.get("FA2SM70_GRAN_CONV0","0")=="1"
# Пара указателей для СВЁРТКИ в спекулятивной ветке: читать из блока опоры прошлого шага,
# писать в блок опоры текущего. Ровно то, что обычный декод делает через initial_state_idx
# и block_idx_last_scheduled_token, а спекулятивный не делал вовсе.
_ГРАН_КОНВ = __import__("os").environ.get("FA2SM70_GRAN_CONV","0")=="1"
# ПАРА УКАЗАТЕЛЕЙ СВЁРТКЕ НА КАЖДОМ ШАГЕ (FA2SM70_GRAN_CONV_ALL=1). Замер прибором
# запаса конца: сама УСЛОВНОСТЬ пары и есть возмущение -- у границы свёртка шла
# парой указателей, вне границы одним слотом, и переключение стоило разброса 8.64
# нат против пола 0.35. Пара на КАЖДОМ шаге: цена перехода +0.03 нат, разброс 0.61
# против пола 0.60 -- граница становится прозрачной. Здесь строится ТОЛЬКО пара,
# без подстановки SSM-столбца и клона истории (их безусловность стоила ~10 % декода).
_ГРАН_КОНВ_ВСЕГДА = __import__("os").environ.get("FA2SM70_GRAN_CONV_ALL","0")=="1"
# Прибор и фальсификатор к разбору падения 06.09 (Xid 13 на боевом): считать выходы за
# границу и уметь СНЯТЬ зажим, чтобы доказать, что падение шло именно оттуда.
_ГРАН_КОНВ_ДИАГ = __import__("os").environ.get("FA2SM70_GRAN_CONV_DIAG","0")=="1"
_ГРАН_КОНВ_БЕЗ_ЗАЖИМА = __import__("os").environ.get("FA2SM70_GRAN_CONV_NOCLAMP","0")=="1"
# Бисекция пары: 1 = читать оттуда же, куда пишем (без истории прошлого шага).
_ГРАН_КОНВ_БЕЗ_ИСТОРИИ = __import__("os").environ.get("FA2SM70_GRAN_CONV_NOHIST","0")=="1"
# Бисекция: отдать ядру ЧУЖУЮ таблицу блоков напрямую, без нашего постоянного буфера.
# Если падение уходит -- виноват буфер (устаревшие строки/столбцы), если остаётся -- сам путь.
_ГРАН_КОНВ_СЫРАЯ = __import__("os").environ.get("FA2SM70_GRAN_CONV_RAW","0")=="1"
_СПЕК_ПО_CTX = __import__("os").environ.get("FA2SM70_GDN_SPEC_CTX","0")=="1"
# [ГОНКА, 07.09] Все копии метаданных сюда идут асинхронно. Если источник --
# ЗАКРЕПЛЁННЫЙ буфер хоста, переиспользуемый на следующем шаге, питон вправе
# переписать его до того, как DMA прошлого шага завершилась: порча метаданных ->
# негодные номера блоков -> обращение за буфер в чужом ядре. Улика в пользу гонки:
# при CUDA_LAUNCH_BLOCKING=1 стенд прошёл 1400 запросов без падения.
# Рычаг делает копии синхронными, чтобы проверить это ЗАМЕРОМ, а не рассуждением.
_НЕБЛОК = _os.environ.get("FA2SM70_GDN_ASYNC_COPY", "1") == "1"
# [ОДНОРОДНАЯ МАСКА -- 10.09, записка 25 §282]
# Выборка GPU-тензора булевой маской (`t[~spec_sequence_masks]`) внутри зовёт `nonzero`, а это
# СИНХРОНИЗАЦИЯ с картой. Трасса со стеками: в `build` 7 `nonzero` и 7 `item` за шаг, и вместе
# с `index` они стоят 1.3 мс из 3.85 -- то есть треть строителя и четверть всей обвязки.
# А в устойчивом декоде маска ОДНОРОДНА: все запросы спекулятивные, значит `~маска` не выбирает
# НИЧЕГО, а `маска` выбирает ВСЁ. Тогда выборка заменяется видом -- бесплатно и без синхронизации.
# Однородность проверяется по ПРОЦЕССОРНОЙ копии маски (она тут же и считается), поэтому сама
# проверка синхронизации не требует. Значения тождественны по построению.
# Умолчание ВЫКЛЮЧЕНО: venv общий с боевым, и новое поведение не должно приезжать
# туда само при первом же перезапуске. Замер: строитель 1.63 -> 1.39 мс (-16 %),
# пять синхронизаций за шаг убраны, приёмка и гейт 391 не тронуты.
_ОДНОРОДН = _os.environ.get("FA2SM70_GDN_UNIFORM_MASK", "0") == "1"


def _выбор(т, маска, преф, брать_спек: bool):
    """Выборка по булевой маске без `nonzero`, когда маска -- ПРЕФИКС из истин.

    `преф` -- число ведущих истин, если маска ровно префиксная, иначе None. Тогда выборка по
    маске это `т[:преф]`, а по инверсии `т[преф:]` -- виды, без выделения и без синхронизации.
    ПОЧЕМУ ПРЕФИКС, А НЕ «ВСЯ ИСТИННА»: маска строится по ДОПОЛНЕННОМУ батчу, у строк-
    заполнителей `num_decode_draft_tokens = -1`, то есть ложь. «Вся истинна» не бывает никогда --
    первая редакция этой правки из-за того и не срабатывала (замерено: строитель не подешевел).
    Настоящие запросы идут первыми, поэтому префиксность -- обычный случай.
    """
    if _ОДНОРОДН and преф is not None:
        return т[:преф] if брать_спек else т[преф:]
    return т[маска] if брать_спек else т[~маска]

_ДЕЛИТЬ_ОБЩЕЕ = _os.environ.get("FA2SM70_GDN_SHARE", "1") == "1"


def _общий_ключ(m, num_accepted_tokens):
    """Ключ шага. ВНИМАНИЕ: id() безопасен ТОЛЬКО пока объект жив.

    Python переиспользует адреса после сборки мусора, поэтому новый объект может получить id
    старого -- и кэш отдаст ЧУЖИЕ тензоры. Отказа не будет, будет тихая порча. Лечение: вместе с
    ключом держим ЖИВЫЕ ССЫЛКИ на те самые объекты (`_ОБЩЕЕ["якорь"]`), тогда их адреса не могут
    быть переиспользованы, пока запись в кэше действительна.
    """
    # РАЗЛИЧИТЕЛЬ ШАГА ОБЯЗАТЕЛЕН. Раннер переиспользует ОДНИ И ТЕ ЖЕ буферы каждый шаг, поэтому
    # id() совпадает между шагами, а `num_actual_tokens`/`num_reqs` в установившемся декоде
    # постоянны -- кэш отдал бы УСТАРЕВШИЕ индексы блоков (тихая порча, не отказ). `max_seq_len`
    # растёт на каждом шаге декода и одинаков у всех групп внутри шага -- это и есть нужный ключ.
    return (id(m.query_start_loc), id(m.seq_lens), m.num_actual_tokens,
            m.num_reqs, int(m.max_seq_len), id(num_accepted_tokens))


# [FA2/SM70 25.08] ОБЩИЕ БУФЕРЫ ГРУПП. У модели ДЕСЯТЬ KV-групп GDN, и все поля метаданных,
# кроме ИНДЕКСОВ СОСТОЯНИЙ, у них совпадают побитово (они зависят от ДЛИН, а не от таблицы
# блоков). Прежде каждая группа держала свои буферы и `update_block_table` копировал в них по
# двенадцать полей -- замер фазомером: 2.43 мс на шаг при девяти обновлениях.
#
# Общий буфер на все группы законен ИМЕННО потому, что общий и при ЗАХВАТЕ графа: каждая группа
# запекает один и тот же адрес, и заполняет его тот, кто строит первым. Прежняя авария (гейт
# «39» вместо «391») была ОБРАТНОЙ: группа отдавала ЧУЖОЙ буфер, которого её граф не видел.
# Здесь адрес один у всех дорог -- и при захвате, и при повторе.
_ОБЩИЕ_БУФЕРЫ: dict = {}


def _общий_буфер(имя, форма, тип, device):
    ключ = (имя, tuple(форма), тип, str(device))
    т = _ОБЩИЕ_БУФЕРЫ.get(ключ)
    if т is None:
        т = _ОБЩИЕ_БУФЕРЫ[ключ] = torch.zeros(форма, dtype=тип, device=device)
    return т


_ОБЩИЕ_ВКЛ = _os.environ.get("FA2SM70_GDN_SHARED_BUF", "1") == "1"
# [ЖИЗНЬ БУФЕРА, 08.09] Растущие буферы отпускать нельзя: их адрес запечён в графе.
_МАСКА_БУФ = int(_os.environ.get("FA2SM70_KEEP_BUFS", "0") or 0)
_ДЕРЖАТЬ_GDN = bool(_МАСКА_БУФ & 9)          # 1=всё, 8=оба буфера gdn
# Разделение на два, чтобы назвать ВИНОВНИКА, а не держать оба:
#   16 = только таблица блоков (_бs), 32 = только буфер индексов состояния (_буф_ssm)
_ДЕРЖАТЬ_БЛОКИ = bool(_МАСКА_БУФ & (9 | 16))
_ДЕРЖАТЬ_SSM = bool(_МАСКА_БУФ & (9 | 32))
_ПЕНСИЯ_GDN: list = []


def _отставить_gdn(*т):
    if True:
        for x in т:
            if x is not None:
                _ПЕНСИЯ_GDN.append(x)


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self._потолок_запросов = int(getattr(getattr(vllm_config, 'scheduler_config', None),
                                             'max_num_seqs', 0) or 0)
        self.decode_cudagraph_max_bs = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        # [СВЁРТКА У ГРАНИЦЫ, 04.09] ПОСТОЯННЫЕ буферы под таблицу и пару указателей.
        # Новые тензоры каждый шаг полный граф не видит: он читает адреса, снятые при
        # захвате, и обращение уходит в освобождённую память -- падало как illegal memory
        # access в чужом ядре. Ширину берём с запасом под таблицу блоков любой группы.
        _шир_бл = cdiv(vllm_config.model_config.max_model_len,
                       kv_cache_spec.block_size) + self.num_spec + 1
        self._conv_блоки_буф = torch.empty(
            (self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1),
             _шир_бл), dtype=torch.int32, device=device)
        # [БУФЕР ЧТЕНИЯ -- ВСЕГДА НУЛИ, 08.09] Колонка чтения спекулятивной свёртки равна
        # нулю ПО ПОСТРОЕНИЮ (таблица выровнена по контексту, состояние лежит в колонке 0):
        # в `build` она считалась как `torch.zeros_like(...)` и копировалась в буфер ДВАЖДЫ
        # за шаг. Заводим буфер нулями один раз -- и не трогаем вовсе: минус шесть запусков
        # ядер на каждом шаге в горячем пути (замер положительным контролем: хозяйская
        # работа стоит 1:1, §218).
        self._conv_чт_буф = torch.zeros(
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1),
            dtype=torch.int32, device=device)
        self._conv_зап_буф = torch.empty_like(self._conv_чт_буф)
        # [ДЕРЕВО, 05.09-3] Буферы off и nacc_ssm рождаются ЗДЕСЬ -- ДО захвата графов.
        # Ленивые буферы появлялись на первом декодном шаге, ПОЗЖЕ захвата: поле
        # num_accepted_ssm при захвате было None, ядро запекалось с адресом обычного nacc,
        # и сдвиг чтения не действовал никогда (дозор: nacc=2, off=2, а слой видел 2).
        if self.use_spec_decode:
            self._nacc_ssm_буф = torch.ones(
                self.decode_cudagraph_max_bs, dtype=torch.int32, device=device)
            # Свой буфер и для СВЁРТКИ: протокол-реплей показал, что захваченный графом
            # nacc-адрес живёт то с лагом на шаг, то нулём после первого принятия ветви --
            # клубок адресов обходится ЯВНЫМ буфером, созданным до захвата.
            self._nacc_conv_буф = torch.ones(
                self.decode_cudagraph_max_bs, dtype=torch.int32, device=device)
            try:
                from vllm.model_executor.models.qwen3_next import ДЕРЕВО_OFF as _ДО0
                if _ДО0.get("буф") is None:
                    _ДО0["буф"] = torch.zeros(
                        vllm_config.scheduler_config.max_num_seqs,
                        dtype=torch.int32, device=device)
            except Exception:
                pass
        self.spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        # РЕЖИМ 'all': на запрос приходится не один слот, а СТОЛБЕЦ блоков, поэтому
        # постоянный буфер под CUDA-граф обязан быть двумерным. Если оставить его
        # одномерным и подсовывать графу свежий тензор, граф запомнит ЧУЖОЙ адрес --
        # отказа не будет, будет тихая порча (тот же класс, что мы уже ловили гейтом).
        self.mamba_all = vllm_config.cache_config.mamba_cache_mode == "all"
        # ШИРИНА БУФЕРА СЧИТАЕТСЯ ТОЙ ЖЕ ФОРМУЛОЙ, ЧТО И ТАБЛИЦА БЛОКОВ В ДВИЖКЕ.
        # ОТКАЗ, КОТОРЫЙ ЭТО ЛЕЧИТ (боевой :8085, 28.08 23:40, аптайм 4.5 ч):
        #   RuntimeError: The size of tensor a (64) must match the size of tensor b (67)
        #   at non-singleton dimension 1  <- gdn_attn.py:723, copy_ в постоянный буфер.
        # Движок (gpu_model_runner._init_..., «mamba_blocks_per_req») даёт таблице ширину
        #   cdiv(max_model_len, block_size) + num_speculative_blocks
        # при включённом префикс-кэше, а здесь стояло только cdiv(...). Разница -- РОВНО
        # число блоков спекуляции (3), поэтому отказ ждал первого шага, где спекулятивных
        # декодов НЕТ (первый шаг после префилла): только там источник копируется целиком.
        # Класс тот же, что в отчёте 27.08 §4.1: «величина, ВЫВЕДЕННАЯ по формуле, вместо
        # взятой у того, кто её задал». Формулу дублируем ДОСЛОВНО, чтобы расхождение не
        # воскресло: одна и та же величина в двух местах обязана считаться одинаково.
        _блоков = cdiv(vllm_config.model_config.max_model_len, kv_cache_spec.block_size)
        _спек = getattr(kv_cache_spec, "num_speculative_blocks", 0)
        self.mamba_max_blocks = (
            max(_блоков,
                (_блоков if vllm_config.cache_config.enable_prefix_caching else 1) + _спек)
            if self.mamba_all
            else 1
        )
        # Общий буфер при FA2SM70_GDN_SHARED_BUF=1, иначе свой на группу (прежнее поведение).
        _вб = (_общий_буфер if _ОБЩИЕ_ВКЛ
               else (lambda имя, ф, т, d: torch.empty(ф, dtype=т, device=d)))
        self.non_spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.mamba_max_blocks)
            if self.mamba_all
            else (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.block_idx_last_computed_token = _вб(
            "blk_last_comp", (self.decode_cudagraph_max_bs,), torch.int32, device)
        self.block_idx_last_scheduled_token = _вб(
            "blk_last_sched", (self.decode_cudagraph_max_bs,), torch.int32, device)
        self.spec_sequence_masks = _вб(
            "spec_masks", (self.decode_cudagraph_max_bs,), torch.bool, device)
        self.spec_token_indx = _вб(
            "spec_tok", (self.decode_cudagraph_max_bs * (self.num_spec + 1),), torch.int32, device)
        self.non_spec_token_indx = _вб(
            "nonspec_tok", (self.decode_cudagraph_max_bs * (self.num_spec + 1),), torch.int32, device)
        self.spec_query_start_loc = _вб(
            "spec_qsl", (self.decode_cudagraph_max_bs + 1,), torch.int32, device)
        self.non_spec_query_start_loc = _вб(
            "nonspec_qsl", (self.decode_cudagraph_max_bs + 1,), torch.int32, device)
        self.num_accepted_tokens = _вб(
            "nacc", (self.decode_cudagraph_max_bs,), torch.int32, device)

    # [FA2/SM70 25.08] ВЫКЛЮЧЕНО ПО ПРОВАЛУ ГЕЙТА. Механизм написан (см. update_block_table ниже) и
    # даёт правильную форму, но гейт «17*23» вернул «39» вместо «391» -- ТИХАЯ ПОРЧА, а не отказ.
    # Значит пересборка индексов состояний под чужую таблицу расходится с `build` в чём-то, что
    # формой не ловится (подозрение: паддинг постоянных буферов и ветка non_spec). Включать только
    # после ПОБИТОВОЙ сверки метаданных обеих дорог на живом шаге, а не «по виду».
    supports_update_block_table: bool = (
        _os.environ.get("FA2SM70_GDN_UPDATE", "0") == "1"
    )  # включается только вместе со сверкой FA2SM70_META_CHECK

    def _индексы_состояний(
        self,
        block_table_tensor,
        aligned_block_table,
        spec_sequence_masks,
        mamba_all: bool,
        только_спекуляция: bool,
    ):
        """ЕДИНОЕ тело для build и update_block_table. Два тела здесь уже стоили подъёма."""
        if spec_sequence_masks is None:
            spec_state = None
            non_spec_state = (
                block_table_tensor if mamba_all else block_table_tensor[:, 0]
            )
            return spec_state, non_spec_state
        _src = aligned_block_table if mamba_all else block_table_tensor
        if только_спекуляция:
            return _src[:, : self.num_spec + 1], None
        return (
            _src[spec_sequence_masks, : self.num_spec + 1],
            block_table_tensor[~spec_sequence_masks]
            if mamba_all
            else block_table_tensor[~spec_sequence_masks, 0],
        )

    def update_block_table(self, metadata, blk_table, slot_mapping):
        """Пересборка ТОЛЬКО блоко-зависимых полей под таблицу другой группы.

        ЗАЧЕМ. У модели ДЕСЯТЬ KV-групп GDN, и без этого движок звал `build` по разу на группу --
        замерено 10 вызовов за шаг, 9.9 мс (наш строитель внимания -- 0.07). Всё, кроме индексов
        состояний, у групп совпадает: маски, счётчики, индексы токенов, начала запросов и индексы
        блоков (они зависят от ДЛИН, а не от таблицы). Механизм штатный -- ровно так делает
        Mamba2 (`mamba_attn.py`), у GDN он просто не был реализован.
        """
        import copy as _copy

        новое = _copy.copy(metadata)
        режим = self.vllm_config.cache_config.mamba_cache_mode
        mamba_all = режим == "all"
        seq_lens = metadata.seq_lens_для_обновления
        block_table_tensor = mamba_get_block_table_tensor(
            blk_table, seq_lens, self.kv_cache_spec, режим
        )
        aligned = (
            mamba_get_block_table_tensor(blk_table, seq_lens, self.kv_cache_spec, "align")
            if mamba_all
            else None
        )
        только_спек = metadata.num_prefills == 0 and metadata.num_decodes == 0
        # [СВЁРТКА У ГРАНИЦЫ] ТАБЛИЦА -- СВОЯ У ГРУППЫ, УКАЗАТЕЛИ -- ОБЩИЕ.
        # Колонки чтения и записи зависят только от ДЛИН, поэтому считаются один раз в
        # `build` и переносятся копией метаданных. А сама таблица блоков у каждой группы
        # своя, и без пересборки девять групп из десяти читали бы состояние по таблице
        # чужой группы -- ровно та тихая порча, о которой предупреждает закон выше
        # (совпадение значений не значит правильности). Замер без пересборки: полных
        # ответов 0 из 8, ответы вырождались до 24 токенов.
        if ((_ГРАН_КОНВ or _ГРАН_КОНВ_ВСЕГДА) and mamba_all
                and getattr(metadata, "spec_conv_чт", None) is not None):
            _пг = block_table_tensor
            if metadata.spec_sequence_masks is not None and not только_спек:
                _мг = metadata.spec_sequence_masks[: _пг.shape[0]]
                _пг = _пг[_мг]
            _нг2 = min(int(metadata.spec_conv_чт.shape[0]), _пг.shape[0],
                       self._conv_блоки_буф.shape[0])
            _вш2 = min(_пг.shape[1], self._conv_блоки_буф.shape[1])
            if _нг2 > 0 and _вш2 > 0:
                self._conv_блоки_буф[:_нг2, :_вш2].copy_(_пг[:_нг2, :_вш2])
                if metadata.spec_state_indices_tensor is not None:
                    _пад2 = (metadata.spec_state_indices_tensor[:_нг2, 0] == PAD_SLOT_ID)
                    self._conv_блоки_буф[:_нг2][_пад2] = PAD_SLOT_ID
                новое.spec_conv_блоки = self._conv_блоки_буф[:_нг2, :_вш2]
            else:
                новое.spec_conv_блоки = None
        spec_state, non_spec_state = self._индексы_состояний(
            block_table_tensor, aligned, metadata.spec_sequence_masks,
            mamba_all, только_спек,
        )
        # ПОСТОЯННЫЕ БУФЕРЫ -- СВОИ У ЭТОЙ ГРУППЫ (её граф захватил именно эти адреса).
        # ЗАКОН, ДОБЫТЫЙ ОШИБКОЙ (25.08): СОВПАДЕНИЕ ЗНАЧЕНИЙ НЕ ЗНАЧИТ ПРАВИЛЬНОСТИ ПРИ ГРАФАХ.
        # Первая редакция копировала только индексы состояний, а остальные поля отдавала ЧУЖИЕ
        # (буферы соседней группы). Побитовая сверка 378 шагов показала НОЛЬ расхождений по
        # значениям -- и всё равно гейт вернул «39» вместо «391»: граф читает АДРЕСА своих
        # буферов, а их никто не заполнил. Поэтому копируем ВСЕ поля, которые заполняет `build`.
        # Условия и добивка PAD_SLOT_ID повторяют ветку `build` ОДИН В ОДИН: разойдись они --
        # получим тихую порчу на паддинге, а не отказ.
        if (
            self.use_full_cuda_graph
            and metadata.num_prefills == 0
            and metadata.num_decodes == 0
            and metadata.num_spec_decodes > 0
            and metadata.num_spec_decodes <= self.decode_cudagraph_max_bs
            and metadata.num_spec_decode_tokens <= self.decode_cudagraph_max_bs
            and spec_state is not None
            and metadata.spec_state_indices_tensor is not None
        ):
            n = metadata.num_spec_decodes
            batch_size = metadata.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[:n].copy_(spec_state[:n], non_blocking=_НЕБЛОК)
            spec_state = self.spec_state_indices_tensor[:batch_size]
            spec_state[n:].fill_(PAD_SLOT_ID)

            # ОСТАЛЬНЫЕ ПОЛЯ -- В СВОИ БУФЕРЫ, порядок и добивка как в `build`.
            # При ОБЩИХ буферах копировать НЕЧЕГО: `новое` -- поверхностная копия метаданных
            # группы-строителя, и её поля УЖЕ указывают на тот самый общий буфер, который
            # запёк граф этой группы. Копия была бы тензора в себя.
            _м = metadata
            if _м.spec_sequence_masks is not None and not _ОБЩИЕ_ВКЛ:
                self.spec_sequence_masks[:n].copy_(_м.spec_sequence_masks[:n], non_blocking=_НЕБЛОК)
                новое.spec_sequence_masks = self.spec_sequence_masks[:batch_size]
                новое.spec_sequence_masks[n:].fill_(False)
            if _м.non_spec_token_indx is not None and not _ОБЩИЕ_ВКЛ:
                _к = _м.non_spec_token_indx.size(0)
                self.non_spec_token_indx[:_к].copy_(_м.non_spec_token_indx, non_blocking=_НЕБЛОК)
                новое.non_spec_token_indx = self.non_spec_token_indx[:_к]
            if _м.spec_token_indx is not None and not _ОБЩИЕ_ВКЛ:
                _к = _м.spec_token_indx.size(0)
                self.spec_token_indx[:_к].copy_(_м.spec_token_indx, non_blocking=_НЕБЛОК)
                новое.spec_token_indx = self.spec_token_indx[:_к]
            if _м.spec_query_start_loc is not None and not _ОБЩИЕ_ВКЛ:
                self.spec_query_start_loc[: n + 1].copy_(
                    _м.spec_query_start_loc[: n + 1], non_blocking=_НЕБЛОК
                )
                _хвост = _м.spec_query_start_loc[n]
                новое.spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
                новое.spec_query_start_loc[n + 1 :].fill_(_хвост)
            if _м.num_accepted_tokens is not None and not _ОБЩИЕ_ВКЛ:
                self.num_accepted_tokens[:n].copy_(_м.num_accepted_tokens[:n], non_blocking=_НЕБЛОК)
                новое.num_accepted_tokens = self.num_accepted_tokens[:batch_size]
                новое.num_accepted_tokens[n:].fill_(1)
                if getattr(_м, "num_accepted_ssm", None) is not None:
                    # ОДИН БУФЕР С build. Второй буфер здесь прибивал граф к адресу,
                    # который build не обновляет: SSM-nacc замерзал на значении захвата,
                    # и чтение состояния шло из колонки якоря при ЛЮБОМ m -- источник
                    # сплошных «!» при включённом OFF.
                    _бs = getattr(self, "_nacc_ssm_буф", None)
                    if _бs is None or _бs.shape[0] < batch_size:
                        _отставить_gdn(_бs) if _ДЕРЖАТЬ_БЛОКИ else None
                        _бs = self._nacc_ssm_буф = torch.ones(
                            max(batch_size, self.decode_cudagraph_max_bs),
                            dtype=_м.num_accepted_ssm.dtype,
                            device=_м.num_accepted_ssm.device)
                    if _м.num_accepted_ssm.data_ptr() != _бs.data_ptr():
                        _бs[:n].copy_(_м.num_accepted_ssm[:n], non_blocking=_НЕБЛОК)
                    _бs[n:batch_size].fill_(1)
                    новое.num_accepted_ssm = _бs[:batch_size]
                if getattr(_м, "num_accepted_conv", None) is not None:
                    _бк2 = getattr(self, "_nacc_conv_буф", None)
                    if _бк2 is not None and _бк2.shape[0] >= batch_size:
                        if _м.num_accepted_conv.data_ptr() != _бк2.data_ptr():
                            _бк2[:n].copy_(_м.num_accepted_conv[:n], non_blocking=_НЕБЛОК)
                        _бк2[n:batch_size].fill_(1)
                        новое.num_accepted_conv = _бк2[:batch_size]
        # ВТОРАЯ ВЕТКА БУФЕРОВ -- ЧИСТЫЙ ДЕКОД (без спекуляции). Её пропуск дал ПУСТЫЕ ОТВЕТЫ в
        # конфигурации боевого: значения совпадали, но граф читал незаполненные буферы группы.
        # Зеркалим `build` один в один, включая добивку.
        if (
            self.use_full_cuda_graph
            and metadata.num_prefills == 0
            and metadata.num_spec_decodes == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
            and non_spec_state is not None
        ):
            _д = metadata.num_decodes
            _bs = (metadata.non_spec_state_indices_tensor.shape[0]
                   if metadata.non_spec_state_indices_tensor is not None else _д)
            self.non_spec_state_indices_tensor[:_д].copy_(non_spec_state[:_д], non_blocking=_НЕБЛОК)
            non_spec_state = self.non_spec_state_indices_tensor[:_bs]
            non_spec_state[_д:].fill_(PAD_SLOT_ID)

            if (mamba_all and not _ОБЩИЕ_ВКЛ
                    and metadata.block_idx_last_computed_token is not None):
                self.block_idx_last_computed_token[:_д].copy_(
                    metadata.block_idx_last_computed_token[:_д], non_blocking=_НЕБЛОК)
                новое.block_idx_last_computed_token = self.block_idx_last_computed_token[:_bs]
                новое.block_idx_last_computed_token[_д:].fill_(0)
                self.block_idx_last_scheduled_token[:_д].copy_(
                    metadata.block_idx_last_scheduled_token[:_д], non_blocking=_НЕБЛОК)
                новое.block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[:_bs]
                новое.block_idx_last_scheduled_token[_д:].fill_(0)

            if metadata.non_spec_query_start_loc is not None and not _ОБЩИЕ_ВКЛ:
                self.non_spec_query_start_loc[: _д + 1].copy_(
                    metadata.non_spec_query_start_loc[: _д + 1], non_blocking=_НЕБЛОК)
                _хв = metadata.non_spec_query_start_loc[_д]
                новое.non_spec_query_start_loc = self.non_spec_query_start_loc[: _bs + 1]
                новое.non_spec_query_start_loc[_д + 1 :].fill_(_хв)

        новое.spec_state_indices_tensor = spec_state
        новое.non_spec_state_indices_tensor = non_spec_state
        return новое

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        _т0 = _time.perf_counter()
        # ОБЩИЙ КЛЮЧ ШАГА: у всех KV-групп GDN эти объекты -- одни и те же (различаются только
        # таблица блоков и слоты). Тождество объектов и есть признак «тот же шаг, та же группа дел».
        _клч0 = (id(m.query_start_loc), id(m.seq_lens), m.num_actual_tokens, m.num_reqs,
                 int(m.max_seq_len))  # см. пояснение в _общий_ключ: без длины ключ не различает ШАГИ
        _делить = _ДЕЛИТЬ_ОБЩЕЕ
        _общ0 = _ОБЩЕЕ.get("знач0") if (_делить and _ОБЩЕЕ.get("ключ0") == _клч0) else None
        if _общ0 is not None:
            context_lens_tensor = _общ0["ctx"]
        else:
            context_lens_tensor = m.compute_num_computed_tokens()
        _гф("0 compute_num_computed_tokens", _т0)
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        mamba_cache_mode = self.vllm_config.cache_config.mamba_cache_mode
        mamba_all = mamba_cache_mode == "all"
        _т1 = _time.perf_counter()
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            mamba_cache_mode,
        )
        _гф("1 таблица блоков (режим)", _т1)
        # РЕЖИМ 'all' (задача 194). Здесь block_table_tensor приходит ПОЛНЫЙ (запрос x блоки),
        # и одномерные потребители (спекуляция, декод) его использовать не могут. Сведённую
        # таблицу берём ТОЙ ЖЕ функцией в режиме 'align': она отдаёт 1+num_spec последних
        # блоков каждого запроса -- ровно то, что нужно и спекуляции, и декоду. Собственного
        # второго выражения для этого НЕ пишем: расхождение двух тел уже стоило нам подъёма.
        aligned_block_table = None
        block_idx_last_computed_token = None
        aligned_block_table = None
        block_idx_last_scheduled_token = None
        block_idx_first_scheduled_token = None
        if mamba_all:
            _т2 = _time.perf_counter()
            aligned_block_table = mamba_get_block_table_tensor(
                m.block_table_tensor, m.seq_lens, self.kv_cache_spec, "align"
            )
            # [31.08 ГРАНИЦА БЛОКА СОСТОЯНИЯ] Таблица выше начинается с блока ПОСЛЕДНЕГО
            # ЗАПЛАНИРОВАННОГО токена: `(seq_lens-1)//block_size`. При спекуляции seq_lens
            # включает k+1 черновых, поэтому на шаге перехода через границу начало
            # перепрыгивает в НОВЫЙ блок, а состояние ещё лежит в старом -- блока со
            # состоянием в колонках просто нет. Замер прибором (FA2SM70_GDN_GRAN_DIAG):
            #     seq_len=4097 ctx=4093 start_po_seq=1 start_po_ctx=0
            #     seq_len=4098 ctx=4094 start_po_seq=1 start_po_ctx=0
            #     seq_len=4099 ctx=4095 start_po_seq=1 start_po_ctx=0
            #     seq_len=4100 ctx=4096 start_po_seq=1 start_po_ctx=0
            # (разность seq-ctx = 4 = k+1, то есть это ровно спекулятивный декод.)
            # Не-спекулятивный путь этого не знает: он берёт read по посчитанным, write по
            # запланированным, и потому границу проходит верно. Спекуляции даём ТУ ЖЕ опору --
            # начало по ПОСЧИТАННЫМ. Тогда колонка 0 снова указывает на блок с состоянием, а
            # колонки 1..num_spec покрывают блоки, куда k+1 токенов могут перейти.
            aligned_spec_table = aligned_block_table
            if _СПЕК_ПО_CTX:
                aligned_spec_table = mamba_get_block_table_tensor(
                    m.block_table_tensor, context_lens_tensor, self.kv_cache_spec, "align"
                )
            # [31.08 ДИАГНОСТИКА ГРАНИЦЫ] Печатает шаг, где начало align-таблицы, взятое по
            # seq_lens (последний ЗАПЛАНИРОВАННЫЙ токен), расходится с началом по числу
            # ПОСЧИТАННЫХ. Ровно в этот шаг спекуляция читает состояние из нового блока.
            if _ГРАН_ДИАГ and num_accepted_tokens is not None:
                # [04.09 СДВИГ ОПОРЫ] Колонки spec-таблицы -- это блоки, отсчитанные от
                # start=(seq_len-1)//B. Ядро пишет состояние позиции j в колонку j, а на
                # следующем шаге читает колонку (num_accepted-1). Значит согласованность
                # держится ТОЛЬКО пока start не сдвинулся между шагами. Прибор считает оба
                # старта: текущий и тот, что был при ЗАПИСИ (seq_prev-1 = ctx - m + k).
                try:
                    _B3 = self.kv_cache_spec.block_size
                    _m3 = num_accepted_tokens[: int(m.num_reqs)].to(torch.int64)
                    _ctx3 = context_lens_tensor[: int(m.num_reqs)].to(torch.int64)
                    _seq3 = m.seq_lens[: int(m.num_reqs)].to(torch.int64)
                    _st_cur = ((_seq3 - 1) // _B3).clamp(min=0)
                    _st_зап = ((_ctx3 - _m3 + self.num_spec) // _B3).clamp(min=0)
                    if _СЧЁТ.get("q", 0) < 3:
                        _СЧЁТ["q"] = _СЧЁТ.get("q", 0) + 1
                        print(f"[Q ШАГА] seq-ctx={int(_seq3[0]) - int(_ctx3[0])} "
                              f"ctx={int(_ctx3[0])} m={int(_m3[0])}",
                              file=_sys.stderr, flush=True)
                    _расх = (_st_cur != _st_зап)
                    if bool(_расх.any()):
                        _j = int(_расх.nonzero()[0][0])
                        _СЧЁТ["сдвиг"] = _СЧЁТ.get("сдвиг", 0) + 1
                        print(f"[СДВИГ ОПОРЫ] ctx={int(_ctx3[_j])} seq={int(_seq3[_j])} "
                              f"m={int(_m3[_j])} start_тек={int(_st_cur[_j])} "
                              f"start_зап={int(_st_зап[_j])} B={_B3} "
                              f"всего={_СЧЁТ['сдвиг']}", file=_sys.stderr, flush=True)
                except Exception as _e3:
                    print(f"[СДВИГ ОПОРЫ] err: {_e3}", file=_sys.stderr, flush=True)
            if _ГРАН_ДИАГ:
                try:
                    _bs2 = self.kv_cache_spec.block_size
                    _st_seq = ((m.seq_lens - 1) // _bs2).clamp(min=0)
                    _st_ctx = ((context_lens_tensor - 1) // _bs2).clamp(min=0)
                    _разн = (_st_seq != _st_ctx)
                    if bool(_разн.any()):
                        _i = int(_разн.nonzero()[0][0])
                        print(f"[ГРАНИЦА] zapros={_i} seq_len={int(m.seq_lens[_i])} "
                              f"ctx={int(context_lens_tensor[_i])} start_po_seq={int(_st_seq[_i])} "
                              f"start_po_ctx={int(_st_ctx[_i])} nreqs={int(m.num_reqs)}",
                              file=_sys.stderr, flush=True)
                except Exception as _e:
                    print(f"[ГРАНИЦА] diag err: {_e}", file=_sys.stderr, flush=True)
            _гф("2 таблица блоков (align)", _т2)
            mamba_block_size = self.kv_cache_spec.block_size
            if _общ0 is not None:
                # Индексы блоков зависят ТОЛЬКО от длин (контекст/последовательность), а они у
                # всех групп общие -- считаем один раз на шаг. Блоко-зависимое ниже своё.
                block_idx_last_computed_token = _общ0["b_last_computed"]
                block_idx_first_scheduled_token = _общ0["b_first_sched"]
                block_idx_last_scheduled_token = _общ0["b_last_sched"]
            else:
                # Индексы блоков -- как у Mamba2 (mamba_attn._compute_prefix_caching_block_indices)
                # [ТРИ ЛЕСЕНКИ В ОДНУ, 08.09] Те же три выражения считаются на СКЛЕЙКЕ
                # трёх рядов: одно деление с округлением вверх и одно вычитание вместо трёх.
                # Зажим остаётся РАЗНЫМ (у первого и третьего он есть, у второго нет) -- это
                # не косметика, а смысл: `first_scheduled` обязан уметь -1.
                _ряды = torch.stack((context_lens_tensor,
                                     context_lens_tensor + 1,
                                     m.seq_lens))
                _ряды = cdiv(_ряды, mamba_block_size) - 1
                block_idx_last_computed_token = _ряды[0].clamp(min=0)
                block_idx_first_scheduled_token = _ряды[1]
                block_idx_last_scheduled_token = _ряды[2].clamp(min=0)
                if _делить:
                    _ОБЩЕЕ["ключ0"] = _клч0
                    _ОБЩЕЕ["якорь0"] = (m.query_start_loc, m.seq_lens)  # см. _общий_ключ
                    _ОБЩЕЕ["знач0"] = {
                        "ctx": context_lens_tensor,
                        "b_last_computed": block_idx_last_computed_token,
                        "b_first_sched": block_idx_first_scheduled_token,
                        "b_last_sched": block_idx_last_scheduled_token,
                    }
                    _общ0 = _ОБЩЕЕ["знач0"]
            _гф("3 индексы блоков", _т2)

        _тА = _time.perf_counter()
        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = spec_sequence_masks_cpu.to(
                    query_start_loc.device, non_blocking=_НЕБЛОК
                )
        # ПРЕФИКСНОСТЬ маски -- по ПРОЦЕССОРНОЙ копии: без обращения к карте и без
        # синхронизации. `_преф_спек` = число ведущих истин, если маска ровно префиксная.
        _преф_спек = None
        if _ОДНОРОДН and spec_sequence_masks_cpu is not None:
            _к = int(spec_sequence_masks_cpu.sum())
            if bool(spec_sequence_masks_cpu[:_к].all()) and not bool(
                    spec_sequence_masks_cpu[_к:].any()):
                _преф_спек = _к

        _гф("A маски", _тА)
        _тБ = _time.perf_counter()
        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            # В 'all' слой получает ВСЮ таблицу блоков (он сам возьмёт нужный столбец по
            # block_idx_*); в 'none'/'align' таблица уже сведена к одному блоку на запрос.
            non_spec_state_indices_tensor = (
                block_table_tensor if mamba_all else block_table_tensor[:, 0]
            )
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # [СНЯТА ВЫБОРКА РАДИ ДЛИНЫ, 08.09] Здесь стояло
            # `non_spec_query_lens = query_lens[~spec_sequence_masks]` -- выборка по маске НА
            # КАРТЕ с выделением памяти, а её ЕДИНСТВЕННЫЙ потребитель -- `.size(0)` строкой
            # ниже. То же число есть на процессоре даром: `non_spec_query_lens_cpu` считается
            # тут же, и его длина по построению равна длине карточной. py-spy назвал эту
            # строку одной из горячих в фазе Б.
            # [FA2/SM70 25.08] СЧЁТ ВЕДЁТСЯ ПО ПРОЦЕССОРНЫМ КОПИЯМ, А НЕ ПО ТЕНЗОРАМ КАРТЫ.
            # Здесь стояли четыре `.item()` по GPU-тензорам, и каждый -- СИНХРОНИЗАЦИЯ с картой.
            # Замер фазомером: `GDNAttentionMetadataBuilder.build` стоил **9.86 мс НА ШАГ** при
            # спекуляции (наш собственный строитель -- 0.07 мс), то есть весь налог на включение
            # спекуляции (+16.7 мс к шагу) сидел здесь, а не в вычислении. Значения БИТ В БИТ те
            # же: `query_lens_cpu` -- это та же разность, посчитанная движком заранее.
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = int((non_spec_query_lens_cpu == 1).sum())
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes
            num_decode_tokens = num_decodes
            num_prefill_tokens = int(non_spec_query_lens_cpu.sum()) - num_decode_tokens
            num_spec_decode_tokens = (
                int(query_lens_cpu.sum()) - num_prefill_tokens - num_decode_tokens
            )

            _клч = _общий_ключ(m, num_accepted_tokens)
            _общ = _ОБЩЕЕ["знач"] if _ОБЩЕЕ["ключ"] == _клч else None
            if _общ is not None and _ДЕЛИТЬ_ОБЩЕЕ:
                (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens,
                 num_spec_decode_tokens, spec_token_indx, non_spec_token_indx,
                 spec_query_start_loc, non_spec_query_start_loc,
                 non_spec_query_start_loc_cpu) = _общ
                # БЛОКО-ЗАВИСИМОЕ -- СВОЁ У КАЖДОЙ ГРУППЫ.
                _spec_src = aligned_spec_table if mamba_all else block_table_tensor
                if num_prefills == 0 and num_decodes == 0:
                    spec_state_indices_tensor = _spec_src[:, : self.num_spec + 1]
                    non_spec_state_indices_tensor = None
                else:
                    spec_state_indices_tensor = (
                        _spec_src[:, : self.num_spec + 1] if (_преф_спек is not None)
                        else _spec_src[spec_sequence_masks, : self.num_spec + 1]
                    )
                    non_spec_state_indices_tensor = (
                        _выбор(block_table_tensor, spec_sequence_masks, _преф_спек, False)
                        if mamba_all
                        else _выбор(block_table_tensor[:, 0], spec_sequence_masks,
                                    _преф_спек, False)
                    )
            elif num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    int(query_start_loc_cpu[-1]),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Спекуляция всегда работает СВЕДЁННОЙ таблицей: её токены идут за
                # последним запланированным, и им нужны 1+num_spec ПОСЛЕДНИХ блоков --
                # ровно то, что отдаёт режим 'align' (в 'all' полная таблица не годится:
                # её первые столбцы -- начало последовательности, а не её конец).
                _spec_src = aligned_spec_table if mamba_all else block_table_tensor
                spec_state_indices_tensor = _spec_src[:, : self.num_spec + 1]
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                _spec_src = aligned_spec_table if mamba_all else block_table_tensor
                spec_state_indices_tensor = (
                    _spec_src[:, : self.num_spec + 1] if (_преф_спек is not None)
                    else _spec_src[spec_sequence_masks, : self.num_spec + 1]
                )
                non_spec_state_indices_tensor = (
                    _выбор(block_table_tensor, spec_sequence_masks, _преф_спек, False)
                    if mamba_all
                    else _выбор(block_table_tensor[:, 0], spec_sequence_masks,
                                _преф_спек, False)
                )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    _выбор(query_lens, spec_sequence_masks, _преф_спек, True),
                    dim=0, out=spec_query_start_loc[1:]
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    _выбор(query_lens, spec_sequence_masks, _преф_спек, False),
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            if _общ is None:
                _ОБЩЕЕ["ключ"] = _клч
                _ОБЩЕЕ["якорь"] = (m.query_start_loc, m.seq_lens, num_accepted_tokens)
                _ОБЩЕЕ["знач"] = (
                    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens,
                    num_spec_decode_tokens, spec_token_indx, non_spec_token_indx,
                    spec_query_start_loc, non_spec_query_start_loc,
                    non_spec_query_start_loc_cpu,
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = _выбор(num_accepted_tokens, spec_sequence_masks,
                                         _преф_спек, True)

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks is not None:
                has_initial_state = _выбор(has_initial_state, spec_sequence_masks,
                                           _преф_спек, False)
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=_НЕБЛОК
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks, non_blocking=_НЕБЛОК
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=_НЕБЛОК
            )
            _хв_нс = min(int(self.non_spec_token_indx.shape[0]),
                         int(self.decode_cudagraph_max_bs) * (self.num_spec + 1))
            if int(non_spec_token_indx.size(0)) < _хв_нс:
                self.non_spec_token_indx[int(non_spec_token_indx.size(0)):_хв_нс].fill_(0)
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=_НЕБЛОК
            )
            # ХВОСТ ИНДЕКСОВ ТОКЕНОВ ОБНУЛЯЕТСЯ, И ЭТО НЕ ПЕДАНТИЗМ.
            # Полный граф захватывает ДЛИНУ этого среза; на воспроизведении настоящих
            # строк меньше, а ядро идёт по захваченной длине и читает хвост -- там лежат
            # индексы ПРОШЛОГО шага, и они могут указывать ЗА нынешнее число токенов.
            # Дальше это `index_select`/gather за пределами -- то есть Xid 13 и смерть
            # воркера. Все соседние спекулятивные буферы паддинг получают
            # (spec_state_indices, spec_sequence_masks, spec_query_start_loc,
            # num_accepted_tokens), а этот -- нет. Ноль всегда годный индекс.
            _хв_сп = min(int(self.spec_token_indx.shape[0]),
                         int(self.decode_cudagraph_max_bs) * (self.num_spec + 1))
            if int(spec_token_indx.size(0)) < _хв_сп:
                self.spec_token_indx[int(spec_token_indx.size(0)):_хв_сп].fill_(0)
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=_НЕБЛОК
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=_НЕБЛОК
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=_НЕБЛОК
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            if mamba_all:
                # Те же указатели блоков -- в постоянные буферы, по той же причине.
                self.block_idx_last_computed_token[:num_decodes].copy_(
                    block_idx_last_computed_token, non_blocking=_НЕБЛОК
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :batch_size
                ]
                block_idx_last_computed_token[num_decodes:].fill_(0)
                self.block_idx_last_scheduled_token[:num_decodes].copy_(
                    block_idx_last_scheduled_token, non_blocking=_НЕБЛОК
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :batch_size
                ]
                block_idx_last_scheduled_token[num_decodes:].fill_(0)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=_НЕБЛОК
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        # РЕЖИМ 'all': индексы блоков приводим к тому же ПОДМНОЖЕСТВУ и тому же ПОРЯДКУ, что
        # и non_spec_state_indices_tensor (сперва декоды, затем префиллы), иначе состояние
        # запроса будет прочитано из блока соседа -- отказа не будет, будет тихая порча.
        _гф("Б ветка спекуляции + копии", _тБ)
        _тВ = _time.perf_counter()
        num_computed_tokens_ns = None
        num_computed_tokens_ns_cpu = None
        if mamba_all:
            if spec_sequence_masks is not None:
                # ДЛИНЫ СОГЛАСУЕМ ЯВНО. При ЗАХВАТЕ ГРАФА метаданные приходят ПАДДИРОВАННЫМИ
                # (движок сам это признаёт: `unpadded()` в backend.py с пометкой «drafter still
                # only uses piecewise cudagraphs ... does not want padded metadata»), поэтому
                # маска бывает ДЛИННЕЕ индексов блоков: замерено отказом 22.08 -- «mask [8] does
                # not match indexed tensor [2]», подъём падал целиком. Паддинг всегда в ХВОСТЕ,
                # реальные запросы идут первыми, поэтому срез по длине индексов берёт ровно
                # реальную часть. Если длины совпали -- поведение прежнее.
                if _общ0 is not None and "b_last_computed_m" in _общ0:
                    block_idx_last_computed_token = _общ0["b_last_computed_m"]
                    block_idx_last_scheduled_token = _общ0["b_last_sched_m"]
                    block_idx_first_scheduled_token = _общ0["b_first_sched_m"]
                    num_computed_tokens_ns = _общ0["ns"]
                else:
                    _n = block_idx_last_computed_token.shape[0]
                    _keep = ~(spec_sequence_masks[:_n] if spec_sequence_masks.shape[0] > _n
                              else spec_sequence_masks)
                    # При однородной маске `_keep` сплошь ложна -- выборка даёт пустое, и
                    # `nonzero` внутри индексации не нужен вовсе (см. _выбор).
                    if _преф_спек is not None:
                        # [МИНА 12.09, боевой лёг 17:01] Префиксная маска НЕ значит «все строки
                        # спекулятивные»: за префиксом идут НЕ-спекулятивные строки -- в том числе
                        # ПРЕФИЛЛ соседнего запроса в том же батче. Прежде здесь брали `[:0]`, и при
                        # num_prefills > 0 цикл по qsl в qwen3_next читал пустой ncomp -> IndexError,
                        # движок умирал (8 клиентов получили 500). Берём строки ПОСЛЕ префикса --
                        # ровно то, что делает _выбор(..., брать_спек=False), без nonzero и синхронизации.
                        block_idx_last_computed_token = block_idx_last_computed_token[_преф_спек:]
                        block_idx_last_scheduled_token = block_idx_last_scheduled_token[_преф_спек:]
                        block_idx_first_scheduled_token = block_idx_first_scheduled_token[_преф_спек:]
                        num_computed_tokens_ns = context_lens_tensor[_преф_спек:]
                    else:
                        block_idx_last_computed_token = block_idx_last_computed_token[_keep]
                        block_idx_last_scheduled_token = block_idx_last_scheduled_token[_keep]
                        block_idx_first_scheduled_token = block_idx_first_scheduled_token[_keep]
                        num_computed_tokens_ns = context_lens_tensor[_keep]
                    if _делить and _общ0 is not None:
                        _общ0["b_last_computed_m"] = block_idx_last_computed_token
                        _общ0["b_last_sched_m"] = block_idx_last_scheduled_token
                        _общ0["b_first_sched_m"] = block_idx_first_scheduled_token
                        _общ0["ns"] = num_computed_tokens_ns
            else:
                num_computed_tokens_ns = context_lens_tensor
            if num_prefills > 0:
                # Синхронизация ТОЛЬКО когда в батче есть префилл (см. поле выше).
                num_computed_tokens_ns_cpu = num_computed_tokens_ns.to("cpu")

        _гф("В индексы блоков (mamba all)", _тВ)
        _тГ = _time.perf_counter()
        # [ГРАНИЦА БЛОКА СОСТОЯНИЯ -- ПОДСТАНОВКА БЛОКА-ИСТОЧНИКА, 04.09] ---------------
        # Колонки spec-таблицы -- это блоки, отсчитанные от опоры A=(seq_len-1)//B: ядро пишет
        # состояние позиции j в колонку j, а на СЛЕДУЮЩЕМ шаге читает колонку num_accepted-1.
        # Согласовано это лишь пока опора не двигалась. Опора же считается от seq_len, а он
        # растёт: на шаге, где k+1 позиций переходят границу блока, A увеличивается на единицу,
        # и чтение уезжает ровно на блок мимо записи. Прибор (FA2SM70_GDN_GRAN_DIAG) печатает
        # это прямо: `ctx=8190 seq=8194 m=4 start_тек=2 start_зап=1`.
        # Лечить это черновиком нельзя: при асинхронном планировании спекулятивный батч
        # ПАДДИРУЕТСЯ до k+1 (движок сам требует padded drafter batch), и снятие черновика
        # число позиций шага не меняет -- проверено рычагом FA2SM70_GRAN_ALL_OFF=1: шаги шли
        # с q=4 при пустом черновике.
        # Поэтому правится АДРЕС: в ту ячейку, откуда ядро возьмёт начальное состояние
        # (колонка num_accepted-1), кладётся блок, где состояние ЛЕЖИТ НА САМОМ ДЕЛЕ --
        # A_зап + num_accepted - 1, где A_зап=(ctx-num_accepted+k)//B -- опора ПРОШЛОГО шага.
        # Подстановка безусловна: когда опора не двигалась, A_зап==A_тек и в ячейку ложится
        # ровно то же значение, что там и было. Ветвлений нет -- значит переживает CUDA-граф.
        # [ДЕРЕВО, 05.09-2] Сдвиг чтения на W колонок нужен ТОЛЬКО SSM (состояния строк
        # ветви B лежат в колонках W+1..2W), и только при m>1: при m=1 принят один бонус,
        # его состояние -- у якоря, колонка 0. Свёртке сдвиг ПРОТИВОПОКАЗАН: её окно после
        # подмены из чернового слота живёт обычной цепной арифметикой (интеграционный тест
        # test_derevo_conv.py: ветви A и B, m=1..3 -- relL2 ~6e-08). Единый сдвиг, который
        # стоял здесь раньше, давал свёртке ряд [a1, b0, b1] -- подпись «слово!» из §128b.
        num_accepted_ssm = None
        num_accepted_conv = None
        if num_accepted_tokens is not None:
            try:
                from vllm.model_executor.models.qwen3_next import ДЕРЕВО_OFF as _ДО
                _б = _ДО.get("буф")
                # [СНЯТА СИНХРОНИЗАЦИЯ РАДИ ПЕЧАТИ, 08.09]
                # Здесь стояло условие `int(_б[0]) != 0` БЕЗ РЫЧАГА. `_б` -- тензор КАРТЫ,
                # и `int(...)` по нему -- полная синхронизация: хозяин ждёт карту на КАЖДОМ
                # построении метаданных, то есть каждый шаг, ради отладочной строки. Две
                # соседние диагностики ([ПРИЁМНИК], [ПОДСТАНОВКА]) уже стояли за `_ГРАН_ДИАГ`;
                # эта осталась открытой. Теперь она за тем же рычагом -- в бою ветка мертва
                # и ни одного обращения к карте не делает.
                if _ГРАН_ДИАГ:
                    _СЧЁТ["ssm_зов"] = _СЧЁТ.get("ssm_зов", 0) + 1
                    if _СЧЁТ["ssm_зов"] <= 6 or (_б is not None and int(_б[0]) != 0):
                        print(f"[BUILD ssm] зов={_СЧЁТ['ssm_зов']} буф={'есть' if _б is not None else 'НЕТ'} "
                              f"nacc0={int(num_accepted_tokens[0])} "
                              f"off0={int(_б[0]) if _б is not None else '-'}",
                              file=_sys.stderr, flush=True)
                if _б is not None:
                    _nб = min(int(num_accepted_tokens.shape[0]), int(_б.shape[0]))
                    # [ТОЧНОЕ ЛЕЧЕНИЕ ВИСЯЧЕГО УКАЗАТЕЛЯ, 08.09]
                    # Бисекция назвала ИМЕННО ЭТОТ буфер: удержание одного его
                    # даёт 3 соака без падения, а удержание таблицы блоков -- падение
                    # на первом. Механизм: буфер создавался лениво (мог попасть в
                    # ЗАХВАТ) и ЗАМЕНЯЛСЯ при нехватке -- старый отпускался, а его
                    # адрес запечён в графе, и повтор писал в чужую память.
                    # Лечение без вечного удержания: выделяем СРАЗУ под потолок
                    # (число запросов планировщика), а редкий больший случай
                    # обслуживаем ВРЕМЕННЫМ тензором, которого граф никогда не видит.
                    _потолок = max(int(self.decode_cudagraph_max_bs),
                                   int(getattr(self, "_потолок_запросов", 0) or 0), 1)
                    _буф_ssm = getattr(self, "_nacc_ssm_буф", None)
                    if _буф_ssm is None:
                        _буф_ssm = self._nacc_ssm_буф = torch.empty(
                            max(_nб, _потолок),
                            dtype=num_accepted_tokens.dtype,
                            device=num_accepted_tokens.device)
                    elif _буф_ssm.shape[0] < _nб:
                        # НЕ заменяем постоянный буфер: он, возможно, уже в графе.
                        _буф_ssm = torch.empty(
                            _nб, dtype=num_accepted_tokens.dtype,
                            device=num_accepted_tokens.device)
                    _буф_ssm[:_nб].copy_(num_accepted_tokens[:_nб])
                    _буф_ssm[:_nб] += _б[:_nб] * (num_accepted_tokens[:_nб] > 1)
                    num_accepted_ssm = _буф_ssm[:_nб]
                    # [ПРОТОКОЛ-РЕПЛЕЙ, 05.09] Сцепление окон показало: conv в графе видит
                    # nacc ПРОШЛОГО шага (подтверждено на трёх переходах). Граф захватил
                    # ПОСТОЯННЫЙ буфер self.num_accepted_tokens (кладёт переупаковщик), а
                    # build отдавал ЛОКАЛЬНЫЙ тензор -- буфер жил с лагом на шаг. Кладём
                    # свежее значение прямо здесь: адрес, который читает граф, обновлён.
                    _бк = getattr(self, "_nacc_conv_буф", None)
                    if _бк is not None:
                        _нб3 = min(_nб, int(_бк.shape[0]))
                        _бк[:_нб3].copy_(num_accepted_tokens[:_нб3])
                        num_accepted_conv = _бк[:_нб3]
            except Exception:
                num_accepted_ssm = None
        # ЦЕНА ПОДСТАНОВКИ ПЛАТИТСЯ ТОЛЬКО У ГРАНИЦЫ. Строитель метаданных зовётся по разу на
        # KV-группу (их десять), поэтому безусловная подстановка стоила 31.8 -> 25.9 ток/с
        # (-18 %). Сама она нужна на ~8 шагах из 4096, и близость границы видна ПО ПРОЦЕССОРНЫМ
        # длинам -- без единого обращения к карте. Метаданные строятся ВНЕ графа, поэтому
        # условность здесь законна: граф исполняет ядра и читает буфер уже исправленным.
        _спец_conv_блоки = _спец_conv_чт = _спец_conv_зап = None
        _гран_рядом_cpu = _ГРАН_ВСЕГДА
        if (_ГРАН_КОНВ_ВСЕГДА and mamba_all and spec_state_indices_tensor is not None
                and num_accepted_tokens is not None):
            try:
                # [ПЕРЕПИСАНО 06.09 ПОСЛЕ РАЗБОРА ПАДЕНИЯ]
                # Спекулятивная таблица -- это ВЫРОВНЕННАЯ таблица, обрезанная до
                # num_spec+1 колонок (`_индексы_состояний`: `_src[маска, :num_spec+1]`,
                # где `_src` -- align-таблица). Её КОЛОНКИ ОТНОСИТЕЛЬНЫЕ: колонка 0 --
                # это блок, с которого таблица выровнена, а не блок номер ноль.
                # Первая редакция подставляла в `block_idx_last_scheduled_token`
                # АБСОЛЮТНЫЙ номер блока (`(seq_len-1)//B`, у нас это бывало 61) и тем
                # самым читала строку таблицы далеко за её пятью колонками -- оттуда
                # приходил мусорный слот, а ядро координату ЗАПИСИ не проверяет
                # (маска строки 932 смотрит только токены и признаки). Наружу это
                # выходило как Xid 13 и смерть воркера. Отсюда правило: колонки для
                # спекулятивной свёртки считаются ОТНОСИТЕЛЬНО начала выравнивания.
                #
                # Выравнивание берётся по seq_lens (см. `aligned_block_table`), то есть
                # колонка 0 -- блок ПОСЛЕДНЕГО ЗАПЛАНИРОВАННОГО токена. Значит:
                #   писать  -> колонка 0;
                #   читать  -> колонка (блок последнего посчитанного) - (блок последнего
                #              запланированного) <= 0, а отрицательных колонок нет.
                # Поэтому пара выражается ТОЛЬКО при выравнивании ПО КОНТЕКСТУ
                # (FA2SM70_GDN_SPEC_CTX=1): там колонка 0 -- блок последнего посчитанного,
                # состояние лежит ровно в ней, а запись идёт в колонку сдвига 0 или 1.
                # Без этого рычага пару выражать нечем, и мы её НЕ СТРОИМ -- прежнее
                # поведение (один слот, колонка 0) остаётся в силе.
                if True:
                    # СВОЯ ТАБЛИЦА У СВЁРТКИ, А НЕ ОБЩИЙ РЫЧАГ. `FA2SM70_GDN_SPEC_CTX`
                    # менял ОБЕ таблицы сразу -- и свёртки, и рекуррента, -- и в одиночку
                    # ронял воркер (+482 Xid за 168 запросов). Здесь контекстное
                    # выравнивание строится ЛОКАЛЬНО и отдаётся ТОЛЬКО свёртке:
                    # колонка 0 -- блок последнего ПОСЧИТАННОГО токена (там и лежит
                    # состояние), колонки 1..num_spec -- блоки, куда шагнут k+1 токенов.
                    # Тогда чтение выражается нулём, а запись -- сдвигом 0..num_spec,
                    # то есть обе координаты заведомо внутри строки.
                    _Bк = self.kv_cache_spec.block_size
                    if spec_sequence_masks is not None and int(num_spec_decodes) != int(m.num_reqs):
                        _мк = spec_sequence_masks[: context_lens_tensor.shape[0]]
                        _ctxк = context_lens_tensor[_мк]
                        _seqк = m.seq_lens[_мк]
                    else:
                        _ctxк, _seqк = context_lens_tensor, m.seq_lens
                    _nк = min(int(spec_state_indices_tensor.shape[0]), int(_ctxк.shape[0]),
                              int(self._conv_чт_буф.shape[0]))
                    if _nк > 0:
                        _шир_сп = int(spec_state_indices_tensor.shape[1])
                        # ОДНА лесенка вместо двух: те же пять операций на СКЛЕЙКЕ двух
                        # рядов вместо пяти на каждый ряд (минус пять запусков ядер).
                        _об = torch.stack((_ctxк[:_nк], _seqк[:_nк])).to(torch.int64)
                        _об = ((_об - 1) // _Bк).clamp(min=0)
                        _ст_ctx, _ст_seq = _об[0], _об[1]
                        _зап_отн = (_ст_seq - _ст_ctx).clamp(min=0, max=_шир_сп - 1)
                        # Копии сюда НЕ пишем: ниже обе строки переписываются ещё раз
                        # (после зажима по ширине таблицы и сверки с выравниванием), то есть
                        # эта пара была мёртвой работой. Буфер чтения нулевой по построению.
                        # ОТКАТ 06.09: редакция «таблица на блок назад + поиск по номеру
                        # блока» давала цену перехода -0.01 нат, приёмку 3.13 и 80 ток/с --
                        # и РАЗРУШАЛА ответ: каждая генерация вырождалась в «!!!» после
                        # первого токена. Высокая приёмка была следствием вырождения, а не
                        # заслугой: черновику легко угадывать повтор одного токена. Закон
                        # «скорость без гейта качества -- ложь» сработал ровно здесь.
                        # Возвращена редакция §150: таблица выровнена ПО КОНТЕКСТУ,
                        # чтение -- колонка 0, запись -- сдвиг 0/1 со сверкой у движка.
                        _таб_к = mamba_get_block_table_tensor(
                            m.block_table_tensor, context_lens_tensor,
                            self.kv_cache_spec, "align")
                        if (spec_sequence_masks is not None
                                and int(num_spec_decodes) != int(m.num_reqs)):
                            _таб_к = _таб_к[spec_sequence_masks[: _таб_к.shape[0]]]
                        _вш_к = min(_шир_сп, int(_таб_к.shape[1]),
                                    int(self._conv_блоки_буф.shape[1]))
                        if _вш_к <= 0 or int(_таб_к.shape[0]) < _nк:
                            raise ValueError("узкая таблица свёртки")
                        self._conv_блоки_буф[:_nк, :_вш_к].copy_(_таб_к[:_nк, :_вш_к])
                        _падк = (spec_state_indices_tensor[:_nк, 0] == PAD_SLOT_ID)
                        self._conv_блоки_буф[:_nк][_падк] = PAD_SLOT_ID
                        if _nк < int(self._conv_блоки_буф.shape[0]):
                            self._conv_блоки_буф[_nк:] = PAD_SLOT_ID
                            self._conv_зап_буф[_nк:] = 0    # буфер чтения нулевой всегда
                        _зап_отн = _зап_отн.clamp(min=0, max=_вш_к - 1)
                        if aligned_block_table is not None:
                            _свер = aligned_block_table
                            if (spec_sequence_masks is not None
                                    and int(num_spec_decodes) != int(m.num_reqs)):
                                _свер = _свер[spec_sequence_masks[: _свер.shape[0]]]
                            if int(_свер.shape[0]) >= _nк and int(_свер.shape[1]) > 0:
                                _цель = _таб_к[:_nк].gather(1, _зап_отн.unsqueeze(1)).squeeze(1)
                                _ок = (_цель == _свер[:_nк, 0])
                                _зап_отн = torch.where(_ок, _зап_отн,
                                                       torch.zeros_like(_зап_отн))
                        self._conv_зап_буф[:_nк].copy_(_зап_отн.to(torch.int32))
                        _спец_conv_блоки = self._conv_блоки_буф[:_nк, :_вш_к]
                        _спец_conv_чт = self._conv_чт_буф[:_nк]
                        _спец_conv_зап = self._conv_зап_буф[:_nк]
            except Exception:
                _спец_conv_блоки = _спец_conv_чт = _спец_conv_зап = None
        if _ГРАН_ИСТОК and mamba_all and spec_state_indices_tensor is not None:
            try:
                _Bc = self.kv_cache_spec.block_size
                _slc = m.seq_lens_cpu[: int(m.num_reqs)]
                _qlc = (query_start_loc_cpu[1 : int(m.num_reqs) + 1]
                        - query_start_loc_cpu[: int(m.num_reqs)])
                _ctxc = (_slc - _qlc).to(torch.int64)
                # ШИРИНА ОКНА -- ЗАМЕРОМ, А НЕ ИЗ ФОРМУЛЫ. Расширение до 16*(k+1)=64
                # ОТВЕРГНУТО: базовый промпт остался 12/12, а сдвинутый упал с 5/12 до
                # 1/12. Значит лишнее срабатывание НЕ безвредно (подстановка вне
                # границы подсовывает блок из истории, а он совпадает с текущим не
                # всегда), и запас окна -- не свободный параметр. Оставлено 2*(k+1) с
                # ДВУСТОРОННЕЙ проверкой (см. ниже): это строго лучше прежнего в обоих
                # случаях (12/12 против 7/12 и 5/12 против 3/12).
                _окно = 2 * (self.num_spec + 1)
                # ОКНО ДВУСТОРОННЕЕ. Первая редакция смотрела ТОЛЬКО ВПЕРЁД
                # (ctx-1 против ctx+2(k+1)) и переставала срабатывать сразу ПОСЛЕ
                # перехода -- а подстановка нужна ещё несколько шагов: состояние
                # принятой позиции продолжает лежать в ПРЕЖНЕМ блоке. Замер
                # (гейт границы, 12 прогонов под нагрузкой, temp0):
                #   окно вперёд   -- 7/12 полных, обрывы на +4..+13 токенов ЗА границей;
                #   подстановка КАЖДЫЙ шаг (FA2SM70_GRAN_ALWAYS=1) -- 12/12 и детерминизм;
                #   без спекуляции -- 12/12 (спекуляция -- необходимое условие).
                # То есть лечение было верным, но не применялось на том шаге, ради
                # которого написано. Симметричное окно стоит столько же: срабатывает
                # на ~16 шагах из 4096 вместо ~8.
                _гран_рядом_cpu = _ГРАН_ВСЕГДА or bool((((_ctxc - _окно - 1) // _Bc)
                                        != ((_ctxc + _окно) // _Bc)).any())
            except Exception:
                _гран_рядом_cpu = True   # не смогли определить -- работаем как раньше
        if (_ГРАН_ИСТОК and _гран_рядом_cpu and mamba_all
                and spec_state_indices_tensor is not None
                and num_accepted_tokens is not None):
            _Bг = self.kv_cache_spec.block_size
            if spec_sequence_masks is not None:
                _мс = spec_sequence_masks[: block_table_tensor.shape[0]]
                _полн = block_table_tensor[_мс]
                _ctxг = context_lens_tensor[_мс]
                _seqг = m.seq_lens[_мс]
            else:
                _полн = block_table_tensor
                _ctxг = context_lens_tensor
                _seqг = m.seq_lens
            _nг = min(spec_state_indices_tensor.shape[0], _полн.shape[0],
                      _ctxг.shape[0], int(num_accepted_tokens.shape[0]))
            if _nг > 0 and _полн.shape[1] > 0:
                _mг = num_accepted_tokens[:_nг].to(torch.int64).clamp(min=1)
                _A_тек = (((_seqг[:_nг].to(torch.int64) - 1) // _Bг)).clamp(min=0)
                # [ОДНО ПРИВЕДЕНИЕ ВМЕСТО ЧЕТЫРЁХ, 08.09] `_ctxг[:_nг].to(int64)` стояло в
                # этом блоке ЧЕТЫРЕ раза (в _A_зап, _A_без, _ctx64г и в снимке истории).
                # Значение одно и то же -- считаем один раз.
                _ctx64 = _ctxг[:_nг].to(torch.int64)
                # `_A_зап` НУЖЕН ТОЛЬКО ДИАГНОСТИКЕ (единственный потребитель -- печать
                # [ПОДСТАНОВКА] под `_ГРАН_ДИАГ`). Считался безусловно: четыре запуска ядер
                # на каждом шаге ради строки, которой в бою нет.
                _A_зап = ((( _ctx64 - _mг + self.num_spec) // _Bг).clamp(min=0)
                          if _ГРАН_ДИАГ else None)
                # ИСТОЧНИК БЕРЁТСЯ ИЗ ИСТОРИИ, А НЕ ИЗ ДОГАДКИ О ПРОШЛОМ ШАГЕ.
                # Формула `A_зап=(ctx-m+k)//B` верна лишь если прошлый шаг был спекулятивным
                # на полную глубину. После префилла это не так: состояние там записано
                # неспекулятивным путём в блок (ctx-1)//B, и догадка давала блок 0 -- гейт
                # «17*23» отвечал мусором. Поэтому опора прошлого шага ЗАПОМИНАЕТСЯ, а её
                # пригодность сверяется по ctx: он обязан лежать в (ctx_пред, ctx_пред+q_пред].
                # Не сошлось -- берём (ctx-1)//B, то есть блок последнего посчитанного.
                _A_без = ((_ctx64 - 1) // _Bг).clamp(min=0)
                # [ДЕРЕВО, 05.09] СТОЛБЕЦ СОСТОЯНИЯ СДВИНУТ ВЕТВЬЮ. Состояние позиции j
                # рекуррент пишет в столбец СТРОКИ, а при ветви B позиция j -- это строка
                # j+W шага. Подстановка у границы брала столбец m-1 (раскладка ветви A) и
                # подсовывала ветви B блок чужой строки; вне границы это не видно, потому
                # порча шла редкими вспышками посреди ответа и запекалась в префикс-кэш.
                # Сдвиг берётся из того же off-буфера, что и num_accepted_ssm.
                _mк = _mг
                if (num_accepted_ssm is not None
                        and int(num_accepted_ssm.shape[0]) >= _nг):
                    _mк = num_accepted_ssm[:_nг].to(torch.int64).clamp(min=1)
                _ист = _A_без + _mк - 1
                # ИСТОРИЯ ХРАНИТ САМУ ТАБЛИЦУ ПРОШЛОГО ШАГА, А НЕ ОПОРУ. Опора описывает
                # адрес записи только пока таблицу никто не правил; но её правит подстановка
                # приёмника (ниже), и тогда формула по опоре указывает мимо. Таблица же
                # хранит ФАКТИЧЕСКИЕ адреса: состояние позиции j лежит там, где стояла
                # колонка j прошлого шага. Пригодность истории сверяется по ctx.
                _ист = _ист.clamp(min=0, max=_полн.shape[1] - 1)
                _знач = _полн[:_nг].gather(1, _ист.unsqueeze(1))
                _кол = (_mк - 1).clamp(min=0,
                                       max=spec_state_indices_tensor.shape[1] - 1)
                _таб_ист = getattr(self, "_гран_таб_пред", None)
                _ctx_ист = getattr(self, "_гран_ctx_пред", None)
                _q_ист = int(getattr(self, "_гран_q_пред", 0) or 0)
                if (_таб_ист is not None and _ctx_ист is not None
                        and _таб_ист.shape[0] == _nг and _ctx_ист.shape[0] == _nг
                        and _таб_ист.shape[1] == spec_state_indices_tensor.shape[1]
                        and _q_ист > 0):
                    _годно = ((_ctx64 > _ctx_ист)
                              & (_ctx64 <= _ctx_ист + _q_ист)).unsqueeze(1)
                    _знач = torch.where(
                        _годно, _таб_ист.gather(1, _кол.unsqueeze(1)), _знач
                    )
                # ОТКАТ 08.09: постоянный буфер здесь дал ХУЖЕ (фаза Б 1.18 -> 1.46 на трёх
                # повторах). Причина не разобрана, но замер однозначен, а правка была ради
                # скорости -- значит она отменяется. Клон возвращён.
                self._гран_ctx_пред = _ctx64.detach().clone()
                self._гран_q_пред = int(self.num_spec) + 1
                # БЕЗ КЛОНА. Клон разрывает связь с ПОСТОЯННЫМ буфером метаданных: полный
                # граф читает адреса своих буферов, и снимок с новым адресом он не видит --
                # генерация вырождалась в мусор (гейт «17*23» отвечал иероглифами). Пишем
                # на месте: буфер и так перезаписывается каждым шагом.
                spec_state_indices_tensor[:_nг].scatter_(
                    1, _кол.unsqueeze(1), _знач.to(spec_state_indices_tensor.dtype)
                )
                # ---- КОЛОНКА 0: СОСТОЯНИЕ СВЁРТКИ ---------------------------------------
                # Спекулятивная ветка свёртки берёт `spec_state_indices_tensor[:, 0]` -- один
                # слот, без пары «читать/писать», которая есть у обычного декода. Колонка 0 --
                # это блок опоры, и она уезжает на границе ровно так же, как колонка чтения
                # SSM. Обрывы лечила правка SSM, а свёртка продолжала терять окно на каждой
                # границе: режим 'all' со спекуляцией давал стабильные 106 токенов и 7 складов
                # из 8, тогда как 'none' и 'all' БЕЗ спекуляции -- 124 токена и 8 из 8.
                # Кладём в колонку 0 тот блок, где свёрточное состояние лежит на самом деле --
                # то есть колонку 0 таблицы прошлого шага. Запись пойдёт туда же, и следующий
                # шаг снова возьмёт её из истории: связка самосогласована.
                # ЗАМЕР ОТВЕРГ подстановку колонки 0: без нагрузки ответы перестали быть
                # одинаковыми (174/188/124/400/400/106/113/108) и полных стало 1 из 8 против
                # 1 из 10 при стабильных 106. Свёртке нужна не подмена одной ячейки, а ПАРА
                # указателей (ниже) -- её ядро это умеет. Рычаг оставлен выключенным.
                if (_ГРАН_КОНВ0 and _таб_ист is not None and _ctx_ист is not None
                        and _таб_ист.shape[0] == _nг
                        and _таб_ист.shape[1] == spec_state_indices_tensor.shape[1]
                        and _q_ист > 0):
                    _ноль = torch.zeros_like(_кол).unsqueeze(1)
                    _тек0 = spec_state_indices_tensor[:_nг].gather(1, _ноль)
                    _ист0 = _таб_ист.gather(1, _ноль).to(_тек0.dtype)
                    spec_state_indices_tensor[:_nг].scatter_(
                        1, _ноль, torch.where(_годно, _ист0, _тек0)
                    )
                # ---- ПРИЁМНИК: состояние на КОНЕЦ блока -- в сам блок --------------------
                # Спекулятивное ядро пишет состояние позиции j в колонку j, а колонка 0 --
                # это блок последнего ЗАПЛАНИРОВАННОГО токена. Значит в настоящий блок
                # ложится состояние ПЕРВОЙ позиции шага, и блок, который на этом шаге
                # закрывается, сохраняет в кэш недосчитанное состояние: при попадании в
                # префикс-кэш ответ уезжает (замер: без кэша 12/12 полных и ответ побайтово
                # один и тот же, с кэшем 10-16 из 20). Кладём в колонку той позиции, что
                # закрывает блок, номер САМОГО блока -- тогда ядро запишет туда состояние
                # ровно на конец блока. Колонку чтения не трогаем: она главнее.
                if _ГРАН_ПРИЁМ:
                    _ctx64п = _ctxг[:_nг].to(torch.int64)
                    _бл = (_ctx64п // _Bг)
                    _jзв = (_бл + 1) * _Bг - 1 - _ctx64п
                    _шир = spec_state_indices_tensor.shape[1]
                    _годно_п = ((_jзв >= 0) & (_jзв < _шир)
                                & (_jзв != (_mг - 1))).unsqueeze(1)
                    _колп = _jзв.clamp(min=0, max=_шир - 1).unsqueeze(1)
                    _значп = _полн[:_nг].gather(
                        1, _бл.clamp(min=0, max=_полн.shape[1] - 1).unsqueeze(1)
                    )
                    _текущ = spec_state_indices_tensor[:_nг].gather(1, _колп)
                    spec_state_indices_tensor[:_nг].scatter_(
                        1, _колп,
                        torch.where(_годно_п, _значп.to(_текущ.dtype), _текущ),
                    )
                    if _ГРАН_ДИАГ:
                        _срп = int(_годно_п.sum())
                        if _срп:
                            _СЧЁТ["приём"] = _СЧЁТ.get("приём", 0) + _срп
                            print(f"[ПРИЁМНИК] строк={_срп} j*={int(_jзв[0])} "
                                  f"блок={int(_бл[0])} всего={_СЧЁТ['приём']}",
                                  file=_sys.stderr, flush=True)
                if _ГРАН_КОНВ:
                    # Опора ПРОШЛОГО шага как колонка полной таблицы. При негодной истории
                    # берём текущую -- тогда поведение в точности прежнее.
                    _A_пред_к = getattr(self, "_гран_Aк_пред", None)
                    _чт = _A_тек
                    if (_A_пред_к is not None and _A_пред_к.shape[0] == _nг
                            and _q_ист > 0 and _ctx_ист is not None
                            and _ctx_ист.shape[0] == _nг):
                        _чт = torch.where(_годно.squeeze(1), _A_пред_к, _A_тек)
                    _вш = min(_полн.shape[1], self._conv_блоки_буф.shape[1])
                    _вб_н = min(_nг, self._conv_чт_буф.shape[0])
                    self._conv_чт_буф[:_вб_н].copy_(_чт[:_вб_н].to(torch.int32))
                    self._conv_зап_буф[:_вб_н].copy_(_A_тек[:_вб_н].to(torch.int32))
                    _спец_conv_чт = self._conv_чт_буф[:_вб_н]
                    _спец_conv_зап = self._conv_зап_буф[:_вб_н]
                    # ПАДДИНГОВЫЕ СТРОКИ ОБЯЗАНЫ ОСТАТЬСЯ ПОМЕЧЕННЫМИ. Прежний вызов брал
                    # колонку spec-таблицы, где фиктивные строки несут PAD_SLOT_ID, и ядро их
                    # пропускало (`USE_PAD_SLOT`). В полной таблице блоков на их месте нули,
                    # и ядро принималось считать несуществующий запрос -- выход за буфер
                    # всплывал как illegal memory access в чужом ядре через сотни строк.
                    self._conv_блоки_буф[:_вб_н, :_вш].copy_(_полн[:_вб_н, :_вш])
                    _пад = (spec_state_indices_tensor[:_вб_н, 0] == PAD_SLOT_ID)
                    self._conv_блоки_буф[:_вб_н][_пад] = PAD_SLOT_ID
                    _спец_conv_блоки = self._conv_блоки_буф[:_вб_н, :_вш]
                    self._гран_Aк_пред = _A_тек.detach().clone()
                # Историю снимаем ПОСЛЕ всех подстановок -- она обязана хранить фактические
                # адреса записи, иначе следующий шаг прочитает не оттуда.
                # [ПОСТОЯННЫЙ БУФЕР ВМЕСТО КЛОНА, 08.09] `clone().to(int64)` выделял НОВЫЙ
                # тензор на каждом шаге -- это и работа распределителя, и лишняя копия в
                # горячем пути. Буфер выделяется один раз под потолок и НЕ заменяется
                # (закон о висячем указателе: заменять то, что могло попасть в граф, нельзя;
                # здесь тензор в графы не попадает, но правило дешевле соблюсти).
                _ист_буф = getattr(self, "_гран_ист_буф", None)
                if (_ист_буф is None or _ист_буф.shape[0] < _nг
                        or _ист_буф.shape[1] != spec_state_indices_tensor.shape[1]):
                    _потг = max(int(self.decode_cudagraph_max_bs),
                                int(getattr(self, "_потолок_запросов", 0) or 0), _nг, 1)
                    _ист_буф = self._гран_ист_буф = torch.empty(
                        (_потг, spec_state_indices_tensor.shape[1]),
                        dtype=torch.int64, device=spec_state_indices_tensor.device)
                _ист_буф[:_nг].copy_(spec_state_indices_tensor[:_nг])
                self._гран_таб_пред = _ист_буф[:_nг]
                if _ГРАН_ДИАГ:
                    _пр = int((_A_тек != _A_зап).sum())
                    if _пр:
                        _СЧЁТ["подстановок"] = _СЧЁТ.get("подстановок", 0) + _пр
                        print(f"[ПОДСТАНОВКА] строк={_пр} A_тек={int(_A_тек[0])} "
                              f"A_зап={int(_A_зап[0])} всего={_СЧЁТ['подстановок']}",
                              file=_sys.stderr, flush=True)

        attn_metadata = GDNAttentionMetadata(
            seq_lens_для_обновления=m.seq_lens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            num_accepted_ssm=num_accepted_ssm,
            num_accepted_conv=num_accepted_conv,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            num_computed_tokens_ns=num_computed_tokens_ns,
            num_computed_tokens_ns_cpu=num_computed_tokens_ns_cpu,
            non_spec_query_start_loc_cpu=(
                non_spec_query_start_loc_cpu if mamba_all and num_prefills > 0 else None
            ),
            mamba_block_size=(self.kv_cache_spec.block_size if mamba_all else 0),
            spec_conv_блоки=_спец_conv_блоки,
            spec_conv_чт=_спец_conv_чт,
            spec_conv_зап=_спец_conv_зап,
        )
        _гф("Г сборка объекта", _тГ)
        _задержка_хозяина()
        _гпечать()
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
