# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA2_SM70 -- attention backend on the HOOLIGAN sm_70 kernel class.

WHY THIS IS A SEPARATE BACKEND AND NOT A PATCH ON SOMEBODY ELSE'S FILE.
The previous integration substituted the `flash_attn_v100` Python package from the outside. That
works, but it forces us to keep another project's signatures, so we can never honestly declare what
we support -- and it produced a real defect: the KV store and the attention backend ended up gated
by DIFFERENT conditions, so a private int8 pool could be read by a foreign reader as e4m3 (on
Gemma-4 that is 40 layers out of 48 reading garbage). A first-class backend declares its own
capabilities in one place, and the engine refuses at start-up instead of failing mid-forward.

SCOPE (the boundary is enforced, not documented):
  fp16 model, paged prefill + paged decode, head_size 64/128/256/512, СКОЛЬЗЯЩЕЕ ОКНО,
  DECODER only; KV-пул fp16 / `fp8_e4m3` / `int8_per_token_head`.
  No CUDA graphs, no KV-коннекторы, no ALiBi / soft-cap / MLA / encoder.
Everything outside that raises at construction time, where the engine can still pick another
backend -- never silently, and never inside forward().

ШАГ 3: БАЙТОВЫЙ СКЛАД KV. 250K контекста на 2xV100 в fp16 не помещается физически, поэтому байт --
требование, а не оптимизация. Поддержаны ДВА байтовых формата, и разница между ними -- НЕ скорость
(оба читаются одним и тем же fp16-ядром внимания после разворота в сборке), а точность и
приватность:

  `fp8_e4m3` -- ШТАТНЫЙ формат пула vLLM: [nb,2,bs,Hkv,d] uint8, масштаб СКАЛЯРНЫЙ на слой
      (layer._k_scale). Никакой приватности: любой чужой читатель прочтёт его правильно.
      Замерено relL2 к fp32 2.68e-2.
  `int8_per_token_head` -- int8 + масштаб fp32 НА (позицию, kv-голову). Замерено relL2 7.0e-3,
      то есть В 3.8 РАЗА ТОЧНЕЕ при том же байте на элемент. Формат приватный, и его
      законность держится на трёх вещах, а не на надежде (см. ОПИСЬ ЧИТАТЕЛЕЙ ниже).

ОПИСЬ ЧИТАТЕЛЕЙ ПУЛА -- ЧЕМ ГАРАНТИРОВАНО, ЧТО ПРИВАТНЫЙ ФОРМАТ ЧИТАЕТ ТОЛЬКО НАШ КОД.
Правило куплено аварией: на Gemma-4 сорок слоёв из сорока восьми прочли бы наш int8 как e4m3,
потому что склад и внимание гейтились РАЗНЫМИ условиями. Обход дерева этого форка даёт:
  * пул аллоцируется как ПЛОСКИЙ int8-буфер (gpu_model_runner.py:11750), dtype и форму задаёт
    БЭКЕНД -- движок сам значения не интерпретирует НИГДЕ;
  * префикс-кэш хеширует ТОКЕНЫ, а не содержимое блоков (kv_cache_utils.py:375-530) -- безопасен;
  * spec-decode (v1/spec_decode/*) в пул не индексирует вовсе -- безопасен;
  * метрики считают только БАЙТЫ на токен (v1/metrics/perf.py:366) -- безопасны;
  * swap/copy блоков (`_custom_ops.py:2733-2784`), CPU-offload (`v1/kv_offload/cpu/*` --
    там принудительный `view(torch.int8)`), выгрузка на диск -- ПОБАЙТОВЫЕ копии;
  * ОПАСНЫ ровно те, кто перекладывает СОДЕРЖИМОЕ: NIXL при неравном TP, `kv_postprocess_*_on_receive`
    (`kv_transfer/kv_connector/utils.py:223+`, permute на уровне элементов), LMCache, HF3FS, MoRIIO.
    ВСЕ ОНИ включаются ТОЛЬКО явным `--kv-transfer-config`, поэтому гарантия делается ОТКАЗОМ:
    `supports_kv_connector() -> False`. Это не осторожность, а единственный способ сказать
    «формат приватный» так, чтобы движок это ПРОВЕРИЛ.
  * KVBlockZeroer (`v1/worker/utils.py:80-175`) пишет нули в блоки, но включается только у
    mamba-моделей -- у нас их нет.
Вторая половина гарантии -- ОТМЕТКА ФОРМАТА НА КЭШЕ (`_POOL`): формат пишет СКЛАД, читатель его
СВЕРЯЕТ, несовпадение = отказ. Именно рассинхрон гейтов, а не сам формат, дал ту аварию.

РАСКЛАДКА int8-ПУЛА И ПОЧЕМУ ОНА ТАКАЯ. `int8_per_token_head` -- строка ШТАТНАЯ: у неё
`KVQuantMode.INT8_PER_TOKEN_HEAD`, и `AttentionSpec.page_size_bytes` (kv_cache_interface.py:152-164)
УЖЕ добавляет `2*block_size*Hkv*sizeof(f32)` байт на страницу под масштабы. Мы этот бюджет и
занимаем -- но иначе, чем TRITON_ATTN. Тот кладёт масштаб ВНУТРЬ строки головы (`head_size+4`), из-за
чего шаг строки перестаёт равняться d; наши ядра требуют `stride(2)==d` (плотная строка) и обходят
таблицу как ПЛОТНУЮ `[nb, bs, Hkv]`. Поэтому буфер режется на ДВЕ ОБЛАСТИ: сперва все данные
(`nb*2*bs*Hkv*d` байт), затем все масштабы (`nb*2*bs*Hkv*4` байт). Байты сходятся ТОЧНО, ядра
работают БЕЗ ЕДИНОЙ ПРАВКИ, и обе таблицы плотные. Цена ровно одна: страница перестаёт быть
непрерывным куском, поэтому побайтовый перенос блока унёс бы данные без масштабов -- ровно те пути,
которые уже закрыты `supports_kv_connector() -> False`.

ГДЕ int8 НЕ ПОДНИМЕТСЯ, И ЭТО ОГРАНИЧЕНИЕ ДВИЖКА, А НЕ ЯДЕР (ЗАМЕРЕНО).
Модель с РАЗНОЙ геометрией слоёв даёт разные `page_size_bytes`, и движок сводит их, УМНОЖАЯ
block_size меньшего слоя на целое отношение (`unify_kv_cache_spec_page_size`,
kv_cache_utils.py:1012-1049). У Gemma-4 это отношение при int8 перестаёт быть целым:
  скользящий слой (Hkv=8, d=256, bs=16): 16*8*(2*256 + 8) = 66560 Б,
  глобальный  слой (Hkv=1, d=512)      : bs*1*(2*512 + 8) = 1032*bs Б,
  66560 / 1032 = 64.50 -- НЕ ЦЕЛОЕ -> NotImplementedError на подъёме.
В fp16 и в e4m3 те же числа дают ровно 64 (131072/2048 и 65536/1024), поэтому обе конфигурации
поднимаются. Отказ ранний и внятный, но он означает: на Gemma-4 байтовый путь -- это e4m3, а int8
работает там, где геометрия слоёв однородна (проверено на Llama-3.2-1B).

ЧТО СТАВИТСЯ УМОЛЧАНИЕМ И ПОЧЕМУ -- ЧИСЛОМ, А НЕ ВКУСОМ. Замерено на обеих моделях:
  цена формата (склад против сырых fp16):  e4m3 2.5-2.9e-2 | int8 7.3-10.2e-3 (в 3.1 раза точнее);
  ошибка ЯДРА (против torch на тех же деквантованных): 1.4-2.6e-4 у ОБОИХ -- шум fp16;
  ёмкость KV: Gemma-4-12B на одной V100-32GB 11282 -> 22565 токенов (x2.00).
int8 точнее, но на боевой модели НЕ ПОДНИМАЕТСЯ (см. выше), а e4m3 вдобавок читается штатными
путями движка -- то есть снимает мину приватного формата целиком. Поэтому байтовое умолчание --
`fp8_e4m3`; int8 остаётся объявленным и рабочим для однородных моделей.

ШАГ 2 (Gemma-4): ОКНО И head_size=512.
  Gemma-4 держит ДВЕ геометрии в одной модели: 40 скользящих слоёв (d=256, 8 KV-голов, окно 1024)
  и 8 глобальных (d=512, ОДНА KV-голова). Движок раскладывает их в 6 kv-групп с РАЗНЫМ block_size
  (16 у скользящих, 64 у глобальных -- страницы выравниваются по наибольшей), поэтому размер блока
  здесь берётся из формы пула на каждый вызов, а не запоминается один раз.

  ГЛАВНЫЙ ФАКТ ПРО ОКНО, И ОН НЕ ПРО МАСКУ: у скользящего слоя страничная таблица ВНЕ ОКНА
  СОДЕРЖИТ МУСОР. Планировщик подменяет вышедшие из окна блоки на null_block
  (single_type_kv_cache_manager.py:468), а воркер дописывает строку только с конца
  (block_table.py:111-128), поэтому в старых колонках остаются номера УЖЕ ОСВОБОЖДЁННЫХ страниц --
  возможно, отданных другому запросу. `seq_lens` при этом ПОЛНАЯ (gpu_model_runner.py:4948). Значит
  окно нельзя "не заметить": прочитать всю длину -- это прочитать чужой KV. Штатные бэкенды живут
  тем же: и Triton (attention_helpers.py:217 `tile_start = max(0, (q_abs-SW+1)//TILE)`), и FA2
  (`n_block_min`) обрезают ДИАПАЗОН, а не полагаются на маску.

  Отсюда обе реализации:
    префилл -- сборка с блока `lo//block_size` + `attn_fwd_cutlass(window=W)`; маска ядра отсекает
      ключи, попавшие в сборку из-за округления до блока, поэтому результат ТОЧЕН до токена;
    декод   -- новый аргумент `kv_start` у пейджированного ядра (int32 [B]): левая граница на
      последовательность. Подмена строки block_table умеет только кратное block_size и тащит до
      block_size-1 лишних ключей; граница внутри ядра точна до токена И снимает трафик (читается W
      ключей вместо всей длины). Цена -- 12 строк в ядре, замер обеих ветвей ниже.

Enable with `VLLM_SM70_FA2=1` (puts FA2_SM70 first in the sm_70 priority list) or explicitly with
`--attention-backend FA2_SM70`.
"""

import math
import os
import time
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec


def kv_cache_uses_per_token_head_scales(kv_cache_dtype: str) -> bool:
    """Портированный помощник: в боевом дереве `vllm.v1.kv_cache_interface` его НЕТ.

    Семантика взята из форка (`get_kv_quant_mode(...).is_per_token_head`) БУКВАЛЬНО:
    режим "по (позиция, голова)" дают ровно две строки формата, остальные -- нет.
    Возвращать True шире, чем там, нельзя: от этого зависит ФОРМА страницы KV
    (head_size + 4 байта под масштабы), и ошибка тут -- молчаливо иная раскладка памяти.
    """
    return kv_cache_dtype in ("int8_per_token_head", "fp8_per_token_head")

logger = init_logger(__name__)

# A tile reads its own end past the last valid row, so a staging buffer sized exactly to the data
# hands the kernel whatever the allocator left there -- and `0 * inf = NaN` poisons the whole output
# row through the second GEMM. This is not theory: it produced a wall of "!" on the FIRST long
# request after every boot until the buffer was zeroed and padded.
_TILE_PAD = 256

_SAID: set[str] = set()
_TRACE = os.environ.get("FA2SM70_TRACE", "0") == "1"
# prefill_win / decode_win считаются ОТДЕЛЬНО и это не украшение: слой с окном, у которого длина
# ещё не переросла окно, идёт по ТОЙ ЖЕ ветке, что и слой без окна. Без отдельного счётчика
# "окно работает" доказывалось бы наличием кода, а не его исполнением -- ровно та подмена, из-за
# которой в этой сессии трижды принимали объявленное за исполненное.
# ФАЗОМЕР ВЕТКИ ПРЕФИЛЛА (гейт FA2SM70_PHASE=1). Профилировщик сервера в этом форке отдаёт 404,
# а вопрос «где 1.3 с при попадании в префикс» без раскладки по фазам не решается: пять гипотез
# подряд снялись замером, и все дешёвые кандидаты кончились. Мерим СВОИМ фальсификатором:
# синхронизация ставится ТОЛЬКО под гейтом, поэтому в обычном режиме цена ноль.
_PHASE = os.environ.get("FA2SM70_PHASE") == "1"
# РАЗВЕДКА ПОД СЛИТЫЙ ШАГ: один раз пройтись по живой модели и доложить, находятся ли ожидаемые
# тензоры. Ничего не подменяет. Умолчание ВЫКЛЮЧЕНО.
_MEGA_CHECK = os.environ.get("FA2SM70_MEGA_CHECK") == "1"
_MEGA_DONE = False


def _mega_probe():
    """Разведка по ОБЪЯВЛЕННОМУ владельцу: модель регистрирует себя при создании."""
    global _MEGA_DONE
    if _MEGA_DONE:
        return
    _MEGA_DONE = True
    try:
        import fa2_sm70.megastep as ms
        m = ms.registered_model()
        if m is None:
            logger.info("[fa2_sm70 мегашаг] владелец НЕ объявлен -- регистрация не сработала")
            return
        logger.info("[fa2_sm70 мегашаг] РАЗВЕДКА: %s", ms.probe_model(m))
        try:
            слои, флаги, мера = ms.prepare_static(m)
            фм = lambda t: "None" if t is None else tuple(t.shape)
            logger.info("[fa2_sm70 мегашаг] ИЗВЛЕЧЕНИЕ: слоёв %d (gdn %d), мера %s",
                        len(слои), sum(флаги), мера)
            мера2 = ms.подготовить(m)
            logger.info("[fa2_sm70 мегашаг] операция: %s", ms.зарегистрировать())
            факт = ms.FACT
            свод = {"Hq": (мера2.get("Hq"), факт["attn"]["Hq"]),
                    "Hkv": (мера2.get("Hkv"), факт["attn"]["Hkv"]),
                    "d": (мера2.get("d"), факт["attn"]["d"]),
                    "rot": (мера2.get("rot"), факт["attn"]["rotary_dim"]),
                    "Hk": (мера2.get("Hk"), факт["gdn"]["Hk"]),
                    "Hv": (мера2.get("Hv"), факт["gdn"]["Hv"])}
            плохо = {k: v for k, v in свод.items() if v[0] != v[1]}
            logger.info("[fa2_sm70 мегашаг] МЕРА из модели: %s | расхождений с факт-листом: %s",
                        {k: v[0] for k, v in свод.items()}, плохо or "НЕТ")
            for имя, i in (("GDN", флаги.index(1)), ("ВНИМАНИЕ", флаги.index(0))):
                logger.info("[fa2_sm70 мегашаг] %s: %d тензоров, позиции %s",
                            имя, len(слои[i]), [фм(t) for t in слои[i]])
        except Exception as e:
            import traceback
            logger.info("[fa2_sm70 мегашаг] извлечение упало: %s: %s | %s",
                        type(e).__name__, e, traceback.format_exc()[-400:])
    except Exception as e:
        import traceback
        logger.info("[fa2_sm70 мегашаг] разведка упала: %s: %s | %s",
                    type(e).__name__, e, traceback.format_exc()[-600:])




# Снятие фазы ВНИМАНИЯ целиком (доля фазы = разность времени). Умолчание ВЫКЛЮЧЕНО.
_FALSE_ATTN = os.environ.get("FA2SM70_FALSE_ATTN") == "1"

# [ХИМЕРА ДЛИНЫ -- записка 20] Ниже порога путь ПОБИТОВО прежний (точное внимание), выше --
# разреженный. Порог не «оптимальная точка», а граница, за которой квадратичная часть съедает
# префилл: по замеру боевого t(L) = 5.95 + 216 мкс*L + 4386 пс*L^2, и доля квадратичной
# 26 % на 32K, 48 % на 64K, 68 % на 128K, 82 % на 250K. До ~130K резать нечего.
# 0 = разреженная ветка ВЫКЛЮЧЕНА (умолчание): включается сознательно, как и всякий размен точности.
_SPARSE_T = int(os.environ.get("FA2SM70_SPARSE_T", "0"))
# Ниже этого числа строк запроса отбор не нужен: это декод, а не префилл.
_SPARSE_MINSQ = int(os.environ.get("FA2SM70_SPARSE_MINSQ", "64"))
# [ПОСЛОЙНЫЙ ОТБОР, 13.09] Разреженно идут только МОДЕЛЬНЫЕ слои в [LAYER_MIN, LAYER_MAX]; остальные слои
# внимания -- плотно. Прибор для лесенки игл (записка 26 §4.8): где накапливается порча различения --
# в первых слоях (дать им плотный путь) или в последних. Умолчание -- все слои разреженно, как было.
_SPARSE_LMIN = int(os.environ.get("FA2SM70_SPARSE_LAYER_MIN", "0"))
_SPARSE_LMAX = int(os.environ.get("FA2SM70_SPARSE_LAYER_MAX", "1000000"))
# [ПРИБОР НЕЛИНЕЙНОСТИ, 13.09] Окно слоёв и послойная масса -- из ФАЙЛА, перечитываемого по mtime на каждом
# форварде: свип «разреженно только слой L» без перезапусков стенда. Формат файла:
#   строка 1: "MIN MAX" (модельные слои, идущие разреженно); строка "mass 3=0.9,7=0.95" -- масса по слоям
#   (>=1.0 = слой плотно); строка "dump <подкаталог>" -- куда класть дампы выхода внимания.
_SPARSE_LAYER_FILE = os.environ.get("FA2SM70_SPARSE_LAYER_FILE", "")
_SLF = {"mtime": -1.0, "lmin": _SPARSE_LMIN, "lmax": _SPARSE_LMAX, "mass": {}, "dump": "run"}
_MASS_ENV0 = os.environ.get("FA2SM70_SPARSE_MASS")   # масса по умолчанию (читает обёртка C++ через getenv)
# [АЛГОРИТМ ПОСЛОЙНОЙ РАЗРЕЖЕННОСТИ, 13.09] Таблица масс по МОДЕЛЬНЫМ слоям, выведенная из замера нелинейности
# (записка 26 §5.2): "3=0.95,7=0.95,...,43=1.0" -- >=1.0 = слой плотно. Статический аналог файла окна слоёв.
_SPARSE_MASS_LAYERS: dict = {}
for _kv in (os.environ.get("FA2SM70_SPARSE_MASS_LAYERS", "") or "").split(","):
    if "=" in _kv:
        _k, _v = _kv.split("="); _SPARSE_MASS_LAYERS[int(_k.strip())] = float(_v.strip())
def _sloi_obnovit() -> None:
    if not _SPARSE_LAYER_FILE:
        return
    try:
        mt = os.path.getmtime(_SPARSE_LAYER_FILE)
    except OSError:
        return
    if mt == _SLF["mtime"]:
        return
    _SLF["mtime"] = mt
    try:
        lines = [l.strip() for l in open(_SPARSE_LAYER_FILE, encoding="utf-8") if l.strip()]
        a = lines[0].split()
        _SLF["lmin"], _SLF["lmax"] = int(a[0]), int(a[1])
        _SLF["mass"] = {}; _SLF["dump"] = "run"
        for l in lines[1:]:
            if l.startswith("mass"):
                for kv in l.split(None, 1)[1].split(","):
                    k, v = kv.split("="); _SLF["mass"][int(k)] = float(v)
            elif l.startswith("dump"):
                _SLF["dump"] = l.split(None, 1)[1].strip()
        logger.info("[fa2_sm70] route: окно слоёв из файла: [%d, %d], масс по слоям %d, дамп '%s'",
                    _SLF["lmin"], _SLF["lmax"], len(_SLF["mass"]), _SLF["dump"])
    except Exception as e:  # noqa: BLE001
        logger.warning("[fa2_sm70] файл окна слоёв не разобран: %s", e)
_DUMP_O = os.environ.get("FA2SM70_DUMP_O", "")          # каталог дампа выхода внимания по слоям (прибор)
_DUMP_O_T = int(os.environ.get("FA2SM70_DUMP_O_T", "90000"))   # дампить чанки с контекстом >= T
_DUMP_O_DONE: set = set()
# [ПРИБОР KVA, 13.09] Дамп K/V (и Q при файле <dir>/Q) слоёв внимания >= SPLIT для чанков префилла (>= 1024 строк),
# пока лежит <dir>/ON: цели для projector'а поздних слоёв (записка 26 §6). Ранг в имени (у рангов разные головы).
_KVA_DIR = os.environ.get("FA2SM70_KVA_DUMP", "")
_KVA_SPLIT = int(os.environ.get("FA2SM70_KVA_SPLIT", "32"))
_KVA_N: dict = {}
def _kva_dump(self, query, key, value, attn_metadata) -> None:
    if not _KVA_DIR or query.shape[0] < 1024 or not os.path.exists(_KVA_DIR + "/ON"):
        return
    num = getattr(self, "_sparse_layer_num", None)
    if num is None or num < _KVA_SPLIT:
        return
    rk = int(torch.cuda.current_device()); n = _KVA_N.get((num, rk), 0) + 1; _KVA_N[(num, rk)] = n
    d = {"k": key.detach().to(torch.float16).cpu(), "v": value.detach().to(torch.float16).cpu()}
    if os.path.exists(_KVA_DIR + "/Q"):
        d["q"] = query.detach().to(torch.float16).cpu()
    sl = getattr(attn_metadata, "seq_lens", None); ql = getattr(attn_metadata, "query_start_loc", None)
    if sl is not None: d["seq_lens"] = sl.detach().cpu()
    if ql is not None: d["qsl"] = ql.detach().cpu()
    torch.save(d, f"{_KVA_DIR}/kv_L{num}_r{rk}_{n:06d}.pt")
def _dump_o(self, out_rows, Tg: int) -> None:
    if not _DUMP_O or int(Tg) < _DUMP_O_T:
        return
    num = getattr(self, "_sparse_layer_num", None)
    if num is None:
        return
    # РАНГ В ИМЕНИ: у TP-рангов разные головы, а путь был общий -- файлы перетирали друг друга (13.09, первый свип).
    rk = int(torch.cuda.current_device())
    key = (_SLF["dump"], num, int(Tg), rk)
    if key in _DUMP_O_DONE:
        return
    _DUMP_O_DONE.add(key)
    d = os.path.join(_DUMP_O, _SLF["dump"]); os.makedirs(d, exist_ok=True)
    torch.save(out_rows.detach().to("cpu").clone(), os.path.join(d, f"o_L{num}_T{int(Tg)}_r{rk}.pt"))
_SPARSE_ALPHA = float(os.environ.get("FA2SM70_SPARSE_ALPHA", "0.01"))
_SPARSE_B = int(os.environ.get("FA2SM70_SPARSE_B", "128"))       # блок ключей у отбора
_SPARSE_TI = int(os.environ.get("FA2SM70_SPARSE_TI", "128"))     # плитка запросов у отбора
_SPARSE_PROBE = int(os.environ.get("FA2SM70_SPARSE_PROBE", "16"))
# [РАЗРЕЖЕННЫЙ ДЕКОД -- ЗАДЕЛ] Таблица средних ключей обновляется ПРИ ЗАПИСИ KV и живёт рядом
# с пулом, адресуясь по ФИЗИЧЕСКОМУ подблоку. Без неё декодный отбор считал бы средние заново
# каждый шаг: 0.41 мс на слой = 6.6 мс на шаг, больше половины приза (замерено).
# С ней обновление одного токена -- 3.8 мкс, то есть 0.06 мс на шаг.
# 0 = выключено (умолчание): пока декодное ядро не умеет обходить индекс, таблица не нужна.
_SPARSE_DEC = os.environ.get("FA2SM70_SPARSE_DEC", "0") == "1"
_SPARSE_BETA = float(os.environ.get("FA2SM70_SPARSE_BETA", "0"))   # поправка на РАЗБРОС ключей блока
_RSTAT = os.environ.get("FA2SM70_SPARSE_RSTAT", "0") == "1"   # разовый зонд статистики разброса
_SPARSE_GAMMA = float(os.environ.get("FA2SM70_SPARSE_GAMMA", "0"))  # поправка ПО ИЗМЕРЕНИЯМ
_SPARSE_POOL = os.environ.get("FA2SM70_SPARSE_POOL", "0") == "1"  # отбор через таблицу пула
_SEL_DUMP = os.environ.get("FA2SM70_SEL_DUMP", "")   # прибор: каталог дампа отобранных блоков (умолчание выкл)
_SEL_DUMP_N = [0]
_TAIL_DENSE = os.environ.get("FA2SM70_SPARSE_TAIL_DENSE", "0") == "1"   # последняя плитка неполного чанка -- плотно
_TAIL_DENSE_CHUNK = int(os.environ.get("FA2SM70_SPARSE_TAIL_CHUNK", "3072"))
# [КОМПЕНСАЦИЯ ОТБРОШЕННОЙ МАССЫ, 12.09] FA2SM70_SPARSE_COMP=1 (умолчание выкл, ПРОТОТИП в питоне).
# Отбор удерживает MASS массы софтмакса; остаток (1-MASS) не игнорируется, а досчитывается по средним
# блоков: для каждого отброшенного блока J оценка его массы E_J = B*exp(scale*q*kbar_J) (средний ключ)
# и вклад E_J*vbar_J (средний V). Итог: o = (e^lse*o_kept + Σ E_J vbar_J) / (e^lse + Σ E_J), где lse --
# натуральный лог суммы exp по удержанным ключам из ядра. Так 27 % блоков читаются, 73 % -- досчитываются.
_SPARSE_COMP = os.environ.get("FA2SM70_SPARSE_COMP", "0") == "1"
# [ДАМП KV ПО СЛОЯМ, 12.09] FA2SM70_KV_DUMP=<каталог>: первая страница пула (16384 ткн) каждого слоя внимания,
# int8 + шкалы на токен -- для офлайн-проверки межслойных дельт K/V (битность остатка). Прибор, выкл по умолчанию.
_KV_DUMP = os.environ.get("FA2SM70_KV_DUMP", "")
_KV_DUMP_DONE: set = set()


def _komp_vbar(pool, bt, Tg: int, B: int, Hkv: int, D: int) -> torch.Tensor:
    """Средние V по блокам B: [Hkv, NB, D] fp16. Страницы пула по таблице блоков, шкалы vs на токен."""
    v, vs = pool["v"], pool["vs"]
    bs = int(v.shape[1]); npg = (Tg + bs - 1) // bs; NB = (Tg + B - 1) // B
    out = torch.zeros(Hkv, NB, D, dtype=torch.float32, device=v.device)
    for pi in range(npg):
        pg = int(bt[pi]); n0 = pi * bs; n1 = min(Tg, n0 + bs); nt = n1 - n0
        vv = v[pg, :nt].float() * vs[pg * bs * Hkv:(pg * bs + nt) * Hkv].view(nt, Hkv, 1)
        nfull = nt // B
        if nfull:
            out[:, n0 // B:n0 // B + nfull] = vv[:nfull * B].view(nfull, B, Hkv, D).mean(1).permute(1, 0, 2)
        if nt - nfull * B:
            out[:, n0 // B + nfull] = vv[nfull * B:].mean(0)
    return out.half()


def _kompensaciya(out_rows: torch.Tensor, lse: torch.Tensor, q_i: torch.Tensor, kbar: torch.Tensor,
                  vbar: torch.Tensor, cnt: torch.Tensor, idx: torch.Tensor, Tg: int, Sq: int,
                  H: int, Hkv: int, D: int, B: int, TI: int, scale: float) -> None:
    """Поправка выхода разреженного внимания на отброшенные блоки, на месте (out_rows [Sq, H*D] fp16)."""
    import math as _m
    GF = H // Hkv; NB = (Tg + B - 1) // B; ntiles = (Sq + TI - 1) // TI
    o3 = out_rows.view(Sq, H, D); logB = _m.log(B)
    for t in range(ntiles):
        r0, r1 = t * TI, min(Sq, (t + 1) * TI)
        NBvis = min(NB, ((Tg - Sq) + r1 - 1) // B + 1)
        for hk in range(Hkv):
            n = int(cnt[t, hk]); kept = torch.zeros(NBvis, dtype=torch.bool, device=idx.device)
            ii = idx[t, hk, :n].long(); kept[ii[ii < NBvis]] = True
            drop = (~kept).nonzero().squeeze(1)
            if drop.numel() == 0:
                continue
            q = q_i[0, r0:r1, hk * GF:(hk + 1) * GF, :].float()               # [rows, GF, D]
            s = torch.einsum("rgd,jd->rgj", q, kbar[hk, drop].float()) * scale   # [rows, GF, nd]
            lse_t = lse[hk * GF:(hk + 1) * GF, r0:r1].transpose(0, 1)          # [rows, GF]
            mx = torch.maximum(lse_t, s.amax(-1) + logB)
            w = torch.exp(s + logB - mx.unsqueeze(-1))                           # веса отброшенных
            wk = torch.exp(lse_t - mx)                                           # вес удержанных
            tot = wk + w.sum(-1)
            corr = torch.einsum("rgj,jd->rgd", w, vbar[hk, drop].float())
            o = o3[r0:r1, hk * GF:(hk + 1) * GF, :].float()
            o3[r0:r1, hk * GF:(hk + 1) * GF, :] = ((o * (wk / tot).unsqueeze(-1) + corr / tot.unsqueeze(-1))).to(o3.dtype)
_SEL_DUMP_POS = [int(x) for x in os.environ.get("FA2SM70_SEL_DUMP_POS", "").split(",") if x.strip()]
_SEL_BETA = [None]   # умеет ли расширение довод beta; проба один раз за процесс
_SEL_GAMMA = [None]  # ... и довод gamma
_SEL_GRP = [None]    # ... и довод grp
_DEC_QGRP = [None]   # умеет ли расширение декода довод qGrp
_HMMA = [int(os.environ.get("FA2SM70_HMMA", "0") or 0)]
_ДАМП_ПРЕФ = [False]   # 1 -- химерное ядро, 2 -- сверка (см. записку 25 §99)
_ГРАНИЦА = [0]       # сколько раз длина выходила за таблицу блоков (см. _decode_uniform)
_QGRP_ON = os.environ.get("FA2SM70_QGRP", "1") == "1"   # 0 = прежний путь копиями (для сверки)
# УМОЛЧАНИЕ 1 (СТРОКА САМА ПО СЕБЕ) -- ПО ЗАМЕРУ, А НЕ ПО ЗАМЫСЛУ.
# Замысел был: свести счёт по k+1 позициям спекуляции и получить то сглаживание, за счёт
# которого работает префилльное правило. ЗАМЕР 28.08 ОПРОВЕРГ: 9/12 против 10/12
# построчного. Позиции одного запроса -- почти одинаковые вектора, независимой информации
# не несут, а знаменатель порога портят. 0 = взять q автоматически (ХУЖЕ, оставлено для
# повторного замера), 1 = построчно.
_SPARSE_DEC_GRP = int(os.environ.get("FA2SM70_SPARSE_DEC_GRP", "1"))
# Своя alpha у декода: кривая там СДВИНУТА ВПРАВО (отбор ведёт ОДНА строка запроса, а не 16
# зондов), и боевые 0.005 оставили бы 73 % блоков вместо 57 %. Замерено:
#   alpha 0.020 -> 26.7 % (префилл 11.1) | 0.010 -> 49.3 (29.9) | 0.005 -> 72.9 (56.9)
_SPARSE_DEC_ALPHA = float(os.environ.get("FA2SM70_SPARSE_DEC_ALPHA", "0.02"))
# [ОТБОР ПО СТРОКЕ В ДЕКОДЕ, 14.09] доля блоков на строку по оценщику (0 = прежнее правило alpha·max со сведением строк).
# Оракул строки (записка 26 §7.7): у строк-искателей игла в верхних 10 % блоков у 98 % строк -> 0.15 с запасом.
_SPARSE_DEC_TOPF = float(os.environ.get("FA2SM70_SPARSE_DEC_TOPF", "0"))
# [ПОЛ ПО ЧИСЛУ БЛОКОВ 18.09] `topf` -- ДОЛЯ, и на коротком контексте доля вырождается: при 8K
# это 128 блоков, 0.15 от них -- 19, то есть ~1.2K токенов из 8K. Владелец поймал это прибором
# границы: 7 потерь из 50 на контекстах <= 11K, причём с НОВОЙ подписью -- код найден, но с
# ошибкой в ПОСЛЕДНЕЙ ЦИФРЕ (взят соседний токен). На 100K и 250K потерь нет.
# Порог ПО ДЛИНЕ сюда поставить нельзя: декод идёт под ЗАХВАЧЕННЫМ графом, и хозяйская ветка
# запеклась бы навсегда. Пол ставится В ЯДРЕ, по числу блоков: при nvis <= minblk отбор
# становится ПЛОТНЫМ сам собой, а на длинном minblk меньше доли и ничего не меняет.
# 256 блоков при B=64 -- это 16384 токена, ровно граница, ниже которой прибор терял.
_SPARSE_DEC_MINBLK = int(os.environ.get("FA2SM70_SPARSE_DEC_MINBLK", "0"))
_SEL_MINBLK = [None]   # умеет ли расширение довод minblk
# [ЗАЦИКЛИВАНИЕ 17.09] СТОК/ОКНО/REC ОТБОРА -- ЧИСЛАМИ В ВЫЗОВЕ, А НЕ РЫЧАГОМ.
# Отбор обязан всегда брать первые SINK блоков и последние WIN блоков независимо от
# счёта. В декоде стояло WIN=8, то есть гарантированы лишь 512 свежих токенов: всё, что
# сеть написала раньше, конкурирует по счёту с сотнями блоков документа. Владелец: «сетка
# с теми параметрами часто зацикливается», и потеря СОБСТВЕННОГО свежего вывода -- ровно
# та форма порчи, которая даёт повтор. Умолчания НЕ МЕНЯЮ (2/8/2 -- как было), рычаги
# нужны, чтобы проверить эту причину замером, а не рассуждением.
# [ПАМЯТЬ 17.09] ПРЕДОХРАНИТЕЛЬ НА СТРОКЕ ПРЕФИЛЛА.
# Замер: после восьми одновременных запросов с картинками карта занята ЦЕЛИКОМ -- живых
# тензоров 30.57 ГиБ, физически свободно 13.5 МиБ, и следующий запрос на 250K падает на
# буфере отбора в 28 МиБ. При этом у распределителя торча в ту же секунду 460.69 МиБ
# «занято, но не роздано»: подрезка держанного зовётся ОДИН РАЗ ПЕРЕД шагом, а нехватка
# случается ВНУТРИ шага, когда держанное успело накопиться заново.
# Здесь отказ памяти перестаёт быть смертью запроса: кэш возвращается драйверу и строка
# считается заново. Повтор безопасен -- запись в пул KV идёт теми же значениями в те же
# слоты, а выход строки перезаписывается целиком.
# Под ЗАХВАТОМ графа предохранитель молчит: там отказ обязан остаться отказом.
# [ПАМЯТЬ 17.09] ПОСТОЯННЫЙ БУФЕР ПОД СПЛОШНОЙ ЗАПРОС ОТБОРА.
# `query` приходит СТОЛБЦОВЫМ срезом слитого QKV (`qkv.split(...)` по последней оси), то есть
# НЕ сплошной, и `_сплошной(query[t0:t1])` копирует его на каждый слой каждого куска:
# 2048 x H x D x 2 Б = ровно 28 МиБ при чанке 2048. Именно этот запрос и не получил памяти
# после штурма картинками (13.5 МиБ свободных). Постоянный буфер занимает те же 28 МиБ, но
# берёт их ОДИН РАЗ, когда память ещё есть, и дальше не просит у распределителя ничего.
_QBUF_ON = os.environ.get("FA2SM70_QBUF", "0") == "1"
_QBUF = {}


def _сплошной(q):
    """Сплошная копия q в постоянном буфере (или обычный .contiguous(), если рычаг снят)."""
    if not _QBUF_ON or q.is_contiguous():
        return q.contiguous()
    ключ = (q.device.index, q.dtype)
    нужно = q.numel()
    б = _QBUF.get(ключ)
    if б is None or б.numel() < нужно:
        _QBUF[ключ] = б = torch.empty(нужно, dtype=q.dtype, device=q.device)
    из = б[:нужно].view(q.shape)
    из.copy_(q)
    return из


_OOM_RETRY = os.environ.get("FA2SM70_OOM_RETRY", "0") == "1"
_OOM_С = {"повторов": 0}


def _с_повтором(fn):
    if not _OOM_RETRY:
        return fn()
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_current_stream_capturing():
            raise
        torch.cuda.empty_cache()
        _OOM_С["повторов"] += 1
        if _OOM_С["повторов"] <= 3 or _OOM_С["повторов"] % 100 == 0:
            logger.warning("[fa2_sm70] нехватка памяти на строке префилла -- кэш возвращён "
                           "драйверу, повтор №%d", _OOM_С["повторов"])
        return fn()


_SEL_SINK = int(os.environ.get("FA2SM70_SEL_SINK", "2"))
_SEL_WIN = int(os.environ.get("FA2SM70_SEL_WIN", "8"))
_SEL_REC = int(os.environ.get("FA2SM70_SEL_REC", "2"))
_SEL_SINK_D = int(os.environ.get("FA2SM70_SEL_SINK_DEC", str(_SEL_SINK)))
_SEL_WIN_D = int(os.environ.get("FA2SM70_SEL_WIN_DEC", str(_SEL_WIN)))
_SEL_REC_D = int(os.environ.get("FA2SM70_SEL_REC_DEC", str(_SEL_REC)))
_SEQLENS_LAZY = os.environ.get("FA2SM70_SEQLENS_LAZY", "0") == "1"   # без неявной синхронизации длин в build() (см. build)
_SEL_TOPF = [None]
_SEL_QGRP = [None]
_HMMA_SEL = [None]   # умеет ли HMMA-ядро декода список блоков отбора (маркер «selIdx»)

_DEC_SEL = [None]
_PAGED_OUT = [None]   # умеет ли расширение писать выход в буфер вызывающего (довод Out)

# [ДЕРЕВО КАНДИДАТОВ -- ОБВЯЗКА, шаг 1] Цепь спекуляции выражается ДЛИНОЙ на строку: позиция j
# видит префикс seq_len-(q-1-j). Дерево так не выразить: у ветви предки лежат НЕ подряд.
# Ядро для этого уже умеет пару (ctxlen, tailmask) -- граница общего контекста и биты хвоста,
# и она ПРОВЕРЕНА ЗАПУСКОМ (три контроля, два побитово совпали с плотным).
# Первый шаг обвязки -- выразить через маску НЫНЕШНЮЮ ЦЕПЬ: строка j берёт биты 0..j-1.
# Выход обязан не измениться; это и есть гейт плумбинга ДО постройки самого дерева.
_TREE_MASK = os.environ.get("FA2SM70_TREE_MASK", "0") == "1"
# ДЕРЕВО ИЗ ДВУХ ВЕТВЕЙ: ширина ветви W (черновых токенов на ветвь). Всего черновых
# 2W, и движок должен идти с num_speculative_tokens = 2W.
_TREE_W = int(os.environ.get("FA2SM70_TREE_W", "0"))
# Держать отставленные буферы (см. _отставить). По умолчанию ВЫКЛ.
# Значение рычага -- МАСКА, чтобы сузить починку до виновника, а не держать всё:
#   1 = всё (как в замере §194);  2 = только поля бэкенда;  4 = только модульные кэши
_КАКИЕ_ДЕРЖАТЬ = int(os.environ.get("FA2SM70_KEEP_BUFS", "0") or 0)
_ДЕРЖАТЬ_БУФЕРЫ = bool(_КАКИЕ_ДЕРЖАТЬ & 3)          # поля объекта
_ДЕРЖАТЬ_МОД = bool(_КАКИЕ_ДЕРЖАТЬ & 5)             # модульные кэши
# Отставленные буферы МОДУЛЬНЫХ кэшей. Тот же довод, что и у полей объекта: адрес
# старого буфера запечён в захваченном графе, и отпускать его нельзя.
_ПЕНСИЯ_МОДУЛЯ: list = []


def _отставить_мод(*тензоры):
    if _ДЕРЖАТЬ_МОД:
        for т in тензоры:
            if т is not None:
                _ПЕНСИЯ_МОДУЛЯ.append(т)
_TREEBUF: dict = {}
_TREEMASKS: dict = {}


def _tree_chain(B: int, q: int, sl_src, device):
    """(ctxlen, tailmask) для ЦЕПИ: та же видимость, что даёт длина на строку.

    Буферы растут и не пересоздаются: адрес обязан быть постоянным для CUDA-графа.
    """
    need = B * q
    t = _TREEBUF.get(str(device))
    if t is None or t[0].numel() < need:
        n2 = max(need, 32)
        _отставить_мод(*(t or ()))
        t = _TREEBUF[str(device)] = (torch.empty(n2, dtype=torch.int32, device=device),
                                     torch.empty(n2, dtype=torch.int32, device=device),
                                     torch.empty(n2, dtype=torch.int32, device=device))
    ctx, tm, sl = t[0][:need], t[1][:need], t[2][:need]
    # ctxlen = seq_len - (q-1): всё, что было ДО токенов этого шага
    base = (sl_src[:B].to(torch.int32) - (q - 1)).view(B, 1)
    ctx.view(B, q).copy_(base.expand(B, q))
    # строка j видит биты 0..j-1 своего хвоста -> маска (1<<j)-1
    маска = ((1 << torch.arange(q, dtype=torch.int32, device=device)) - 1).view(1, q)
    tm.view(B, q).copy_(маска.expand(B, q))
    # ДЛИНЫ НЕ ТРОГАЕМ. Первая редакция ставила всем строкам ПОЛНУЮ длину (раз хвост фильтрует
    # маска). Видимость от этого верна, но `b_len` входит в нарезку split-K (`Lb`), и порядок
    # сложения в онлайн-софтмаксе меняется -- жадный выбор разошёлся на 130-м токене.
    # Ответ при этом НЕ неверен, он ЧИСЛЕННО ДРУГОЙ; но гейт тождества так не пройти, а без
    # него не отличить «численно другой» от «сломан». Оставляем длины на строку: маска тогда
    # избыточна (режет то, что длина уже отрезала) -- ровно то, что и нужно для гейта плумбинга.
    return ctx, tm


# [ДЕРЕВО, ШАГ 2] Маска ДВУХ ВЕТВЕЙ. Строки шага раскладываются так:
#     0        -- якорь (последний принятый токен)
#     1..W     -- ветвь A: a0, a1, ... (продолжают друг друга)
#     W+1..2W  -- ветвь B: b0, b1, ... (продолжают ДРУГ ДРУГА, но НЕ ветвь A)
# Ветвление стоит на позиции 0: b0 -- второй кандидат словаря там же, где a0 -- первый.
# Приз ровно в этом: покрытие позиции 0 у top-1 56.7 %, у top-2 69.3 % (записка 25 §60).
# Длины строк НЕ трогаем по той же причине, что и в цепи: b_len входит в нарезку split-K, и
# смена длин меняет порядок сложения (жадный выбор разошёлся на 130-м токене). Маска режет
# больше, чем длина -- это законно, длина лишь верхняя граница читаемого хвоста.
def _tree_branch(B: int, q: int, W: int, sl_src, device):
    need = B * q
    t = _TREEBUF.get(str(device))
    if t is None or t[0].numel() < need:
        n2 = max(need, 32)
        _отставить_мод(*(t or ()))
        t = _TREEBUF[str(device)] = (torch.empty(n2, dtype=torch.int32, device=device),
                                     torch.empty(n2, dtype=torch.int32, device=device),
                                     torch.empty(n2, dtype=torch.int32, device=device))
    ctx, tm = t[0][:need], t[1][:need]
    base = (sl_src[:B].to(torch.int32) - (q - 1)).view(B, 1)
    ctx.view(B, q).copy_(base.expand(B, q))
    # МАСКА ЗАВИСИТ ТОЛЬКО ОТ (q, W) -- считаем ОДИН РАЗ.
    # Первая редакция строила её питоновскими циклами КАЖДЫЙ шаг, и это стоило 14 % шага:
    # k=6 цепью шло 37.86 мс/ток (как база 37.79), а с деревом 43.2. Цена оказалась не в
    # ветвлении, а в обвязке -- ровно то, о чём §57: питон вне пути это условие.
    _кл = (q, W, str(device))
    m = _TREEMASKS.get(_кл)
    if m is None:
        m = torch.zeros(q, dtype=torch.int32, device=device)
        # НУМЕРАЦИЯ БИТОВ: бит j -- это строка блока j+1, а НЕ j. Якорь (строка 0)
        # лежит в ctx = seq_len-(q-1) и виден ВСЕМ безусловно, поэтому хвостовых
        # битов ровно q-1. Первая редакция считала бит 0 «якорем»: у ветви A это
        # случайно совпало (её бит 0 -- это a0, то есть строка 1, и она же первая
        # видимая), а ветви B дало строку a0 ВМЕСТО САМОЙ СЕБЯ. Замер: строки b
        # выходили ЦЕЛИКОМ NaN (248320 из 248320 логитов) уже на первом шаге дерева,
        # и любой режим, где ветвь B принималась, портил ответ; ветвь A была чиста.
        # ветвь A: строка 1+j видит a0..a(j) (себя включительно) -> биты 0..j
        for j in range(min(W, q - 1)):
            m[1 + j] = (1 << (j + 1)) - 1
        # ветвь B: строка W+1+j видит b0..b(j) (себя включительно) -> биты W..W+j
        for j in range(min(W, q - 1 - W)):
            m[W + 1 + j] = ((1 << (j + 1)) - 1) << W
        m = m.view(1, q)
        _TREEMASKS[_кл] = m
    tm.view(B, q).copy_(m.expand(B, q))
    return ctx, tm


_KBAR: dict = {}


_SELBUF: dict = {}


def _sel_bufs(N: int, Hkv: int, NB: int, device):
    """Буферы отбора декода: ПРЕДВЫДЕЛЕНЫ и только РАСТУТ.

    Постоянный адрес здесь не про скорость (хотя и про неё: без предвыделения замерено
    3.1 мс на шаг при 250K), а про CUDA-ГРАФ, под которым идёт декод: запечённый адрес
    обязан быть тем же при повторе.
    """
    k = str(device)
    t = _SELBUF.get(k)
    if t is None or t[0].shape[0] < N or t[0].shape[1] != Hkv or t[1].shape[2] < NB:
        n2, nb2 = max(N, 8), max(NB, 512)
        t = _SELBUF[k] = (
            torch.zeros(n2, Hkv, dtype=torch.int32, device=device),
            torch.zeros(n2, Hkv, nb2, dtype=torch.int32, device=device),
            torch.zeros(n2, Hkv, nb2, dtype=torch.float32, device=device),
            torch.zeros(n2, Hkv, nb2, dtype=torch.float32, device=device),
        )
    return (t[0][:N], t[1][:N, :, :NB], t[2][:N, :, :NB], t[3][:N, :, :NB])


_SELBUF_H = {}


def _sel_bufs_h(N: int, Hkv: int, NBg: int, device):
    """Буферы m/s отбора декода ПО ГОЛОВАМ: [строк, Hkv, g*NB] fp32, предвыделены, только растут (адрес постоянен под графом)."""
    k = str(device)
    t = _SELBUF_H.get(k)
    if t is None or t[0].shape[0] < N or t[0].shape[1] != Hkv or t[0].shape[2] < NBg:
        n2 = max(N, 8)
        t = _SELBUF_H[k] = (torch.zeros(n2, Hkv, NBg, dtype=torch.float32, device=device),
                            torch.zeros(n2, Hkv, NBg, dtype=torch.float32, device=device))
    return (t[0][:N, :, :NBg], t[1][:N, :, :NBg])


_DEVP = {}   # таблица размахов на пул (см. _dev_pool)

def _kbar_pool(p, B: int):
    """Таблица средних на ПУЛ: [страниц * (bs/B), Hkv, D] fp16. Ключ -- адрес пула."""
    k = p["k"]
    ключ = (k.data_ptr(), int(k.shape[0]), int(k.shape[1]), int(k.shape[2]), int(k.shape[3]), B)
    t = _KBAR.get(ключ)
    if t is None:
        nb, bs, Hkv, D = (int(x) for x in k.shape)
        if bs % B:
            return None
        _отставить_мод(t)
        t = _KBAR[ключ] = torch.zeros(nb * (bs // B), Hkv, D, dtype=torch.float16, device=k.device)
    return t


def _dev_pool(p, B: int):
    """Таблица РАЗМАХОВ на пул -- та же форма, что таблица средних.

    Нужна декодному отбору за тем же, за чем префилльному: средний ключ не представляет
    блок с разбросанными ключами, и поправка `gamma*sum_d |q_d|*dev_d` это чинит (замер
    28.08: порог поднялся вчетверо при возврате качества). Заводится ОТДЕЛЬНЫМ словарём,
    но по тому же ключу, и обновляется ТЕМ ЖЕ ядром при записи KV -- второго прохода нет.
    """
    if _SPARSE_GAMMA <= 0:
        return None
    k = p["k"]
    ключ = (k.data_ptr(), int(k.shape[0]), int(k.shape[1]), int(k.shape[2]), int(k.shape[3]), B)
    t = _DEVP.get(ключ)
    if t is None:
        nb, bs, Hkv, D = (int(x) for x in k.shape)
        if bs % B:
            return None
        t = _DEVP[ключ] = torch.zeros(nb * (bs // B), Hkv, D, dtype=torch.float16, device=k.device)
    return t
_PH = {}

def _ph(name, dt):
    # СЧЁТЧИК СЛОЁВ ЖИВЁТ НА МОДУЛЕ, А НЕ НА `self`. Первая версия считала на слое -- а слой у
    # каждого свой, поэтому каждый доходил до единицы и ждал шестнадцатого СВОЕГО вызова, то есть
    # шестнадцати запросов. Копилка молчала, и это выглядело как «фазы не сработали».
    a = _PH.setdefault(name, [0.0, 0])
    a[0] += dt; a[1] += 1
    if name == "сборка" and a[1] % 16 == 0:
        _ph_dump("префилл, 16 слоёв")

def _ph_dump(tag):
    if not _PHASE or not _PH:
        return
    tot = sum(v[0] for v in _PH.values())
    parts = " | ".join(f"{k}={v[0]*1e3:.1f}мс/{v[1]}" for k, v in sorted(_PH.items(), key=lambda x: -x[1][0]))
    logger.info("[fa2_sm70 ФАЗЫ %s] сумма=%.1fмс :: %s", tag, tot * 1e3, parts)
    _PH.clear()


_COUNTS = {
    "prefill": 0, "decode": 0, "store": 0, "prefill_win": 0, "decode_win": 0,
    # РАЗРЕЖЕННАЯ ВЕТКА -- СВОЙ счётчик. Ключ обязан быть ЗДЕСЬ: _COUNTS -- обычный dict, и
    # незаявленный ключ роняет воркер KeyError'ом на первом же длинном запросе (поймано замером).
    "prefill_sparse": 0, "kbar_upd": 0, "decode_sparse": 0, "tree_mask": 0,
    # ДЕРЕВО КАНДИДАТОВ: ветка однородного декода при FA2SM70_TREE_W>0. Ключа не было,
    # и включение дерева роняло воркер KeyError'ом на первом же шаге -- ровно то, о чём
    # предупреждает заметка выше.
    "tree_branch": 0,
    # БАЙТОВЫЕ ВЕТВИ СЧИТАЮТСЯ ОТДЕЛЬНО ОТ fp16-ВЕТВЕЙ. Иначе «байтовый путь работает» доказывалось
    # бы тем, что сервер поднялся с байтовым пулом, -- а он поднимется и в случае, когда склад пишет
    # байты, а внимание читает их чужим маршрутом. Три счётчика = три места, где формат обязан был
    # совпасть: запись, сборка префилла, декод.
    "store_e4m3": 0, "store_i8": 0,
    "prefill_e4m3": 0, "prefill_i8": 0, "prefill_i8_paged": 0,
    "decode_e4m3": 0, "decode_i8": 0,
    # Ш1: быстрый однородный декод (путь под CUDA-граф). Отдельным счётчиком, а не внутри "decode":
    # иначе «граф захвачен» доказывалось бы тем, что декод вообще шёл, -- а он шёл бы и прежним путём.
    "decode_uniform": 0,
    # ХИМЕРНОЕ ЯДРО -- свой счётчик по тому же закону, что и разреженная ветка выше:
    # незаявленный ключ роняет воркер KeyError'ом на первом же шаге декода (поймано
    # ровно так и здесь, при первом подключении ядра к бэкенду).
    "decode_hmma": 0,
    # Разбивка отказов ветки химеры: без неё «ядро вызывается в 3 %» не диагностируется
    # (записка 25 §120) -- видно только ИТОГО, а не КТО не дошёл.
    "hmma_skip_d128": 0, "hmma_skip_kvs": 0, "hmma_skip_ctx": 0,
    "hmma_skip_si": 0, "hmma_skip_qg": 0, "hmma_skip_prochee": 0, "hmma_net_fn": 0,
}

# ---------------------------------------------------------------- форматы пула
_FP16, _E4M3, _I8 = "fp16", "e4m3", "i8"

# Строка движка -> формат, которым МЫ заполняем пул. Отображение ЯВНОЕ и полное: `else` по умолчанию
# здесь означал бы, что неизвестная строка тихо поедет по fp16-маршруту поверх байтового буфера.
_DTYPE_FMT: dict[str, str] = {
    "auto": _FP16, "float16": _FP16, "fp16": _FP16,
    "fp8": _E4M3, "fp8_e4m3": _E4M3,
    "int8_per_token_head": _I8,
}

# ОТМЕТКА ФОРМАТА ЖИВЁТ НА КЭШЕ, А НЕ В ПЕРЕМЕННОЙ ОКРУЖЕНИЯ.
# Ключ -- (адрес пула, форма): пул живёт от подъёма до остановки движка и не переезжает, а форма
# отличает kv-группы Gemma-4 друг от друга. Запись делает СКЛАД (do_kv_cache_update); читатель,
# который записи не нашёл, ОТКАЗЫВАЕТ вместо чтения. Так рассинхрон гейтов (склад одним условием,
# внимание другим) становится падением на первом же слое, а не связным враньём на сорока слоях.
_POOL: dict[tuple, dict] = {}
_NOSTORE = os.environ.get("FA2SM70_NOSTORE", "0") == "1"
_NOSTORE_K = os.environ.get("FA2SM70_NOSTORE_KERNEL", "0") == "1"


def _pool_key(kv_cache: torch.Tensor) -> tuple:
    return (kv_cache.data_ptr(), tuple(kv_cache.shape))


def _carve(kv_cache: torch.Tensor, fmt: str) -> dict:
    """Разложить сырой буфер пула по нашему формату.

    fp16 / e4m3 -- штатная форма [nb, 2, bs, Hkv, d], `unbind(1)` даёт K и V как есть.
    int8        -- форма [nb, 2, bs, Hkv, d+4] (ровно `page_size_bytes` движка), но режется НЕ по
                   ней: сперва все данные, затем обе таблицы масштабов. См. шапку файла.
    """
    nb, two, bs, hkv, last = (int(x) for x in kv_cache.shape)
    assert two == 2, f"FA2_SM70: ожидалась ось K/V размера 2, получено {kv_cache.shape}"
    if fmt != _I8:
        k, v = kv_cache.unbind(1)
        return {"fmt": fmt, "k": k, "v": v, "d": last, "ks": None, "vs": None}
    if (last - 4) not in (64, 128, 256, 512):
        # БОКОВОЙ int8: страница в e4m3-геометрии (last == d, без +4 на масштабы -- их несут
        # таблицы _i8_side_tabs). Родной int8-склад отличается по last = d+4.
        # Сырой буфер движка -- uint8 (dtype формата fp8_e4m3); БАЙТЫ наши, поэтому виды
        # переинтерпретируются в int8 (view той же ширины) -- ядра требуют kChar.
        k, v = kv_cache.view(torch.int8).unbind(1)
        ks, vs = _i8_side_tabs(kv_cache)
        return {"fmt": fmt, "k": k, "v": v, "d": last, "ks": ks, "vs": vs}
    d = last - 4
    st = kv_cache.untyped_storage()
    dev = kv_cache.device
    page = 2 * bs * hkv * d                      # байт данных на страницу
    i8 = torch.empty(0, dtype=torch.int8, device=dev)
    shp, strd = (nb, bs, hkv, d), (page, hkv * d, d, 1)
    k = i8.new_empty(0).set_(st, 0, shp, strd)
    v = i8.new_empty(0).set_(st, bs * hkv * d, shp, strd)
    f32 = torch.empty(0, dtype=torch.float32, device=dev)
    n = nb * bs * hkv
    base = nb * page // 4                        # смещение в ЭЛЕМЕНТАХ fp32
    ks = f32.new_empty(0).set_(st, base, (n,), (1,))
    vs = f32.new_empty(0).set_(st, base + n, (n,), (1,))
    return {"fmt": fmt, "k": k, "v": v, "d": d, "ks": ks, "vs": vs}


_COUNTS_FILE = os.environ.get("FA2SM70_COUNTS_FILE", "")


def _bump(kind: str) -> None:
    """Count a route and publish it where the acceptance harness can read it.

    Publishing happens ON THE WORK, not at exit: the engine kills worker processes, so an atexit
    hook never runs and the harness would read an empty file -- i.e. it would report "no work" for
    a run that did the work. A counter that only proves itself when the process shuts down cleanly
    is not a behavioural proof.
    """
    _COUNTS[kind] += 1
    if _COUNTS_FILE:
        try:
            import json

            with open(_COUNTS_FILE, "w", encoding="utf-8") as f:
                json.dump(_COUNTS, f)
        except Exception:  # noqa: BLE001 - diagnostics must never break a forward pass
            pass


def _say_once(msg: str, key: str | None = None) -> None:
    """Print a route decision once per process.

    Not decoration. Three times in one session something was 'declared and not executed' -- a server
    came up, answered, and served on somebody else's path. A route line is the cheapest behavioural
    proof that this code is the code that ran.

    КЛЮЧ ОТДЕЛЁН ОТ ТЕКСТА, И ЭТО ИСПРАВЛЕНИЕ ДЕФЕКТА, А НЕ УКРАШЕНИЕ. Строка «метаданные» несёт
    seq_lens и слоты, то есть меняется КАЖДЫЙ ШАГ -- поэтому «однократная» печать шла на каждом шаге
    декода (6 строк на шаг, по одной на kv-группу), а множество _SAID росло без границы: на боевом
    сервере это утечка, линейная по числу выданных токенов. Ключ по СМЫСЛУ строки, а не по её тексту.
    """
    k = key if key is not None else msg
    if k not in _SAID:
        _SAID.add(k)
        logger.info("[fa2_sm70] route: %s", msg)



# --- SP93: отбор блоков ключей на префилле (задача 93, research/21_SP93_OTBOR_BLOKOV.md) --------
# Прибором на дампе боевой сети доказано: (1) скоринг «усреднённые запросы x ТОЧНЫЕ ключи» бьёт
# обратную асимметрию (0.716 против 0.767 потребной плотности при бюджете оракула@0.70);
# (2) межслойный донор МЁРТВ (обе формы сложения хуже); (3) одна маска на ЧАНК и на ВСЕ kv-головы
# не хуже маски на группу 256 per-head -- поэтому ядро внимания не меняется вовсе, применение =
# index_select уже собранных fp16-плит. Маскируются ТОЛЬКО полные 64-блоки СТРОГО ДО начала чанка;
# остаток префикса и сам чанк всегда плотные, нижне-правая причинность ядра сохраняется РОВНО.
# Сквозной гейт: иглы 3/3 на 128K настоящей прозы при 0.72; фаза внимания ранга с отбором -23%.
# По умолчанию ВЫКЛ; включение и точностная цена принимаются только сквозным гейтом.
# [ЗАДАЧА 163] Слитый префилл: ядро внимания читает K/V ПРЯМО из страниц int8-пула
# (attn_fwd_volta_i8_paged) -- без gather и промежуточных плит («привезти много, увезти мало»).
# Гейт ядра: побитово == gather-путь на 4 формах, включая некратный T и перестановленную
# таблицу; 71.40 мс против 71.63 у gather+ядро на T=32K (сборка исчезла целиком).
_PAGED_PREF = os.environ.get("FA2SM70_PAGED_PREFILL", "0") == "1"
# [ЦЕЛЕВОЙ ВИД КОНТУРА] FA2SM70_KV_I8=1: движку пул объявлен fp8_e4m3 (его страница делит
# mamba-страницу гибрида; int8-страница с масштабами +2*bs*Hkv*4 НЕ делит её ни при каком bs:
# множители 5*13 против 2^16*7^2 -- Gemma-класс тупика унификации), а БАЙТЫ внутри наши:
# int8 + масштаб на (позицию, kv-голову) в БОКОВОЙ таблице бэкенда (12.5 МБ на ранг при 6 ГиБ
# пула). Ровно схема боевой прокладки: «приватный формат законен, пока читатели наши» --
# и все читатели (склад/gather/paged-префилл/декод) гейтятся отметкой формата на пуле.
_KV_I8_SIDE = os.environ.get("FA2SM70_KV_I8", "0") == "1"
_I8SIDE: dict = {}


def _i8_side_tabs(kv_cache: torch.Tensor) -> tuple:
    """Боковые таблицы масштабов [nb*bs*Hkv] fp32 (K и V): один адрес на пул -- граф-совместимо."""
    ck = _pool_key(kv_cache)
    t = _I8SIDE.get(ck)
    if t is None:
        nb, _two, bs, hkv, _d = (int(x) for x in kv_cache.shape)
        n = nb * bs * hkv
        t = (torch.zeros(n, dtype=torch.float32, device=kv_cache.device),
             torch.zeros(n, dtype=torch.float32, device=kv_cache.device))
        _I8SIDE[ck] = t
    return t

_SP93 = os.environ.get("FA2SM70_SP93", "0") == "1"
_SP93_DENS = float(os.environ.get("FA2SM70_SP93_DENS", "0.72"))
_SP93_MINLAYER = int(os.environ.get("FA2SM70_SP93_MINLAYER", "32"))   # МОДЕЛЬНЫЙ слой (>=32 = полные 8-15)
_SP93_MINB = int(os.environ.get("FA2SM70_SP93_MINB", "64"))           # минимум блоков префикса
_SP93_ST: dict = {"calls": 0, "kept": 0, "tot": 0, "buf": {}}


def _sp93_buf(n: int, device) -> tuple:
    """Один растущий буфер компактных плит НА КАРТУ (не на слой -- 8 слоёв по 0.5 ГБ никому не
    нужны), меньшие формы -- префиксные виды. Кэшей по точной форме не заводить (утечка exch_i12)."""
    key = int(device.index if device.index is not None else 0)
    b = _SP93_ST["buf"].get(key)
    if b is None or b[0].numel() < n:
        cap = max(n, 2 * (b[0].numel() if b is not None else 0), 1 << 20)
        b = (torch.zeros(cap, dtype=torch.float16, device=device),
             torch.zeros(cap, dtype=torch.float16, device=device))
        _SP93_ST["buf"][key] = b
    return b


def _sp93_prune(q, kb, vb, Tg: int, Sq: int, scale: float):
    """Скоринг + переупаковка fp16-плит [Hkv,Tg,d]. Возвращает (kb', vb', Tg').

    Скоринг: жёсткие центроиды (сегмент 256 позиций x q-голова, среднее) против ТОЧНЫХ ключей
    префикса; ОДНОПРОХОДНЫЙ онлайн-softmax кусками (бегущий максимум, как во флеш-внимании);
    матричная часть fp16 -- на тензорные ядра (fp32-GEMM на Volta в 6 раз медленнее и был виден
    в цене чанка), exp и суммы fp32. Масса в 64-блоки, сумма по центроидам и головам."""
    pre = Tg - Sq
    nb = pre // 64
    keep = max(_SP93_MINB, int(math.ceil(_SP93_DENS * nb)))
    if nb < _SP93_MINB or keep >= nb:
        return kb, vb, Tg
    Hkv, _, d = kb.shape
    H = int(q.shape[1])
    grp = H // Hkv
    q4 = q.to(torch.float32)                                       # [Sq,H,d]
    nseg = (Sq + 255) // 256
    pad = nseg * 256 - Sq
    if pad:
        q4 = torch.cat([q4, q4[-1:].expand(pad, H, d)], 0)
    c = q4.reshape(nseg, 256, H, d).mean(1)                        # [nseg,H,d]
    c = c.permute(1, 0, 2).reshape(Hkv, grp * nseg, d).to(torch.float16)
    nc = grp * nseg
    CH = 32768
    mx = torch.full((Hkv, nc), float("-inf"), device=q.device)
    zs = torch.zeros(Hkv, nc, device=q.device)
    bm = torch.zeros(Hkv, nc, nb, device=q.device)
    for t0 in range(0, nb * 64, CH):
        t1 = min(t0 + CH, nb * 64)
        lg = torch.bmm(c, kb[:, t0:t1].transpose(1, 2)).float() * scale   # [Hkv,nc,t]
        m2 = torch.maximum(mx, lg.amax(2))
        r = torch.exp(mx - m2)
        zs *= r
        bm *= r[:, :, None]
        e = torch.exp(lg - m2[:, :, None])
        zs += e.sum(2)
        bm[:, :, t0 // 64:t1 // 64] += e.reshape(Hkv, nc, -1, 64).sum(3)
        mx = m2
    score = (bm / zs[:, :, None].clamp_min(1e-30)).sum((0, 1))     # [nb], ОДНА маска на чанк
    blk = torch.topk(score, keep).indices.sort().values            # порядок позиций сохраняем
    poz = (blk[:, None] * 64 + torch.arange(64, device=q.device)).reshape(-1)
    poz = torch.cat([poz, torch.arange(nb * 64, Tg, device=q.device)])
    T2 = int(poz.numel())
    b = _sp93_buf(Hkv * T2 * d, q.device)
    kb2 = b[0][:Hkv * T2 * d].view(Hkv, T2, d)
    vb2 = b[1][:Hkv * T2 * d].view(Hkv, T2, d)
    torch.index_select(kb, 1, poz, out=kb2)
    torch.index_select(vb, 1, poz, out=vb2)
    _SP93_ST["calls"] += 1
    _SP93_ST["kept"] += keep
    _SP93_ST["tot"] += nb
    if _SP93_ST["calls"] % 64 == 0:
        logger.info("[fa2_sm70 SP93] calls=%d dens=%.3f (cfg %.2f, minlayer %d)",
                    _SP93_ST["calls"], _SP93_ST["kept"] / _SP93_ST["tot"],
                    _SP93_DENS, _SP93_MINLAYER)
    return kb2, vb2, T2


def _ext():
    """Import the kernel package.

    MUST raise ImportError on any failure: `platforms/cuda.py` catches only ImportError when it walks
    the backend priority list. An OSError or RuntimeError from a JIT build would kill the boot
    instead of letting the engine fall through to the next backend.
    """
    try:
        import fa2_sm70  # noqa: F401

        return fa2_sm70
    except ImportError:
        raise
    except Exception as e:  # noqa: BLE001 - deliberate: everything becomes ImportError
        raise ImportError(f"fa2_sm70 kernels unavailable: {type(e).__name__}: {e}") from e


@dataclass
class FA2SM70Metadata:
    """Per-batch metadata. Deliberately flat: no tensor is allocated here.

    `query_start_loc_cpu` and `seq_lens_cpu` are carried so that forward() never needs
    `int(tensor[i])` on a device tensor. The old shim did exactly that and paid one
    device-to-host sync PER LAYER PER SEQUENCE on every decode step.
    """

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_reqs: int
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    # ПРИЧИННОСТЬ -- ПАРАМЕТР, А НЕ КОНСТАНТА. У ядра она параметром была всегда
    # (`custom_mask_type = causal ? CausalFromBottomRight : NoCustomMask`), а бэкенд зашивал
    # `True` в каждый вызов. Блочному черновику (DFlash2) нужна ДВУСТОРОННЯЯ маска внутри блока:
    # позиции блока считаются параллельно и обязаны видеть друг друга -- так он и обучен.
    # Умолчание True: обычный путь модели не меняется ни на бит.
    causal: bool = True


class FA2SM70MetadataBuilder(AttentionMetadataBuilder[FA2SM70Metadata]):
    # NEVER, not "not set": step 1 has no persistent buffers, so a captured graph would bake stale
    # pointers. Declaring it explicitly is what makes the engine skip capture instead of crashing.
    #
    # [Ш1, 07.08.2026] Условия появились: рабочая область постоянна, `_kmax` держится нулевым
    # тензором ИМЕННО ради постоянного указателя, ядро пишет прямо в буфер движка, а быстрый путь
    # `_decode_uniform` не делает НИ ОДНОГО выделения и не читает хостом (idx/bt/sl -- виды).
    # Объявление идёт ТОЛЬКО вместе с этим путём (FA2SM70_CG=1) и ТОЛЬКО на однотокенный декод:
    # объявить шире -- значит отдать графу префилл, где сборка выделяет на каждый вызов.
    # Умолчание НЕ меняется: захват доказывается захватом, а не заявкой.
    #
    # [СПЕКУЛЯЦИЯ, 21.08.2026] UNIFORM_BATCH -- НЕ расширение на префилл, а ровно однородный декод
    # на 1+k позициях («decodes are 1 + num_speculative_tokens» в описании самого уровня). Пока мы
    # объявляли уровень на токен, движок при MTP откатывался на PIECEWISE, и шаг стоил 277 мс
    # против 29.5 -- ПРИ ОТЛИЧНЫХ ДОГАДКАХ (acceptance до 4.00 из 4). То есть платили не за
    # спекуляцию, а за то, что её форма не покрыта объявлением.
    # Путь для неё -- тот же `_decode_uniform`: k+1 позиций разворачиваются в ВИРТУАЛЬНЫЙ БАТЧ,
    # ядро и его расщепление по ключам не меняются. Уровень поднимается ТОЛЬКО вместе с этим
    # путём; префилл графу по-прежнему не отдаётся (для него нужен ALWAYS, и его мы не заявляем).
    #
    # [ОТКАТ 21.08, ГЕЙТ ПРОВАЛЕН] Заявка UNIFORM_BATCH дала ТИХУЮ ПОРЧУ: гейт «17*23» вернул
    # «1000*1000 ... 19*100=3000», при этом счётчик быстрого пути остался НУЛЁМ. То есть путь под
    # граф не активировался (форма при захвате паддится, и `query.shape[0] == n*q` не выполняется),
    # а графу достался ОБЫЧНЫЙ путь -- с выделениями и запечёнными указателями. Это ровно то, от
    # чего предостерегает строка ниже: захват доказывается захватом, а не заявкой.
    # Поднимать уровень СНОВА только после того, как счётчик `_decode_uniform` покажет работу ПРИ
    # ЗАХВАТЕ и гейт пройдёт. Механика виртуального батча (`_virt_batch`) остаётся -- она верна и
    # ждёт согласования форм с паддингом графа.
    # Уровень поднят обратно ПОСЛЕ того, как однородный путь доказан на спекуляции: гейт «17*23»
    # прошёл, счётчик пути показал работу на обоих воркерах, декод 11.3 -> 17.1 ток/с (это вклад
    # одного лишь расщепления по ключам, ещё без графа). Тихая порча теперь невозможна: если при
    # ЗАХВАТЕ путь не взят, тело падает громко (см. проверку is_current_stream_capturing).
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
        if os.environ.get("FA2SM70_CG") == "1" else AttentionCGSupport.NEVER)

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    _пробы = [0]

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FA2SM70Metadata:
        c = common_attn_metadata
        # ПРИБОР ФОРМЫ ШАГА. build исполняется КАЖДЫЙ шаг (он вне графа), а forward при повторе
        # графа -- НЕТ. Значит форму шага видно только отсюда, и сверять захват с повтором
        # можно только так. Включается FA2SM70_SHAPE_PROBE=N (первые N шагов).
        _пр = int(os.environ.get("FA2SM70_SHAPE_PROBE", "0"))
        if _пр and self._пробы[0] < _пр:
            self._пробы[0] += 1
            try:
                import sys as _s
                _сл = getattr(self, "layer_names", None) or getattr(self, "_layer_names", None)
                _мет = (f"группа[{len(_сл)} слоёв: {_сл[0].split('.')[-2] if _сл else '?'}"
                        f"..{_сл[-1].split('.')[-2] if _сл else '?'}]" if _сл else "группа[?]")
                print(f"[fa2_sm70 ФОРМА {self._пробы[0]:02d}] {_мет} запросов={int(c.num_reqs)} "
                      f"токенов={int(c.num_actual_tokens)} max_q={int(c.max_query_len)} "
                      f"max_s={int(c.max_seq_len)} захват={torch.cuda.is_current_stream_capturing()} "
                      f"qsl={c.query_start_loc_cpu[:6].tolist() if c.query_start_loc_cpu is not None else '?'}",
                      file=_s.stderr, flush=True)
            except Exception:
                pass
        return FA2SM70Metadata(
            num_actual_tokens=c.num_actual_tokens,
            max_query_len=c.max_query_len,
            query_start_loc=c.query_start_loc,
            max_seq_len=c.max_seq_len,
            seq_lens=c.seq_lens,
            block_table=c.block_table_tensor,
            slot_mapping=c.slot_mapping,
            num_reqs=c.num_reqs,
            query_start_loc_cpu=c.query_start_loc_cpu,
            # [ПИТОН В ПУТИ, 14.09 23:50] `c.seq_lens_cpu` -- устаревшее свойство с НЕЯВНОЙ синхронизацией GPU->CPU, когда
            # CPU-копии нет: путь черновика (`build_for_drafting`) строит метаданные без неё, и наш build() платил
            # синхронизацией на КАЖДЫЙ вызов черновика (py-spy: 4 на шаг, ~3 % выборок воркера). Однородный декод CPU-длины не
            # читает вовсе (m.seq_lens на устройстве). За рычагом FA2SM70_SEQLENS_LAZY=1 берём поле напрямую (может быть None);
            # forward синхронизируется сам, только если неоднородный батч без CPU-копии этого потребует.
            seq_lens_cpu=(c._seq_lens_cpu if _SEQLENS_LAZY else c.seq_lens_cpu),
            causal=bool(getattr(c, "causal", True)),
        )


class FA2SM70Backend(AttentionBackend):
    # fp16 only, and that is a hardware statement rather than a preference: sm_70 has no bf16 in
    # silicon. Accepting bf16 here would mean a silent cast, which belongs to the caller's decision.
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    # ОБЪЯВЛЯЕТСЯ ТОЛЬКО ТО, ЧТО РАБОТАЕТ ПО ОБОИМ ПУТЯМ -- И СКЛАД, И ЧТЕНИЕ.
    # `fp8_e5m2` сюда НЕ входит намеренно: ядро префилла его умеет (attn_fwd_volta_e5m2), а
    # пейджированный декод -- нет (диспетчер знает KVB 16/8/87, e5m2 среди них нет). Объявить его
    # значило бы принять конфигурацию, которая падает на первом же шаге декода.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto", "float16", "fp8", "fp8_e4m3", "int8_per_token_head",
    ]

    # The engine writes KV through do_kv_cache_update() before calling forward(); forward() must not
    # do it again.
    forward_includes_kv_cache_update: bool = False

    # PRAVKA PERENOSA (04.08.2026, boevoe derevo). V forke bazovyy klass dayot
    # accept_output_buffer=True po umolchaniyu, zdes -- False, i togda dvizhok idyot vetkoy
    # BEZ vydelennogo vykhodnogo bufera, gde razdelnaya zapis KV zapreshchena assertom
    # (attention.py:458). Otkaz vyglyadit kak "Data-dependent assertion failed" iz dynamo.
    # Obyavlenie zdes -- NE podgonka: forward() etogo bekenda TREBUET perednnyy bufer
    # (assert output is not None) i pishet imenno v nego (output.copy_, ..._paged_into).
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "FA2_SM70"

    @staticmethod
    def get_impl_cls() -> type["FA2SM70Impl"]:
        return FA2SM70Impl

    @staticmethod
    def get_builder_cls() -> type["FA2SM70MetadataBuilder"]:
        return FA2SM70MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # NHD. `kv_cache.unbind(1)` then hands each of K and V as
        # [num_blocks, block_size, num_kv_heads, head_size] -- exactly the layout the paged gather
        # and the paged decode kernels index, and they read the doubled block stride off stride(0)
        # rather than assuming contiguity.
        #
        # ФОРМА ОБЯЗАНА ПОКРЫВАТЬ `page_size_bytes` РОВНО, И ЭТО НЕ СТИЛЬ, А ЕДИНСТВЕННАЯ ПРОВЕРКА,
        # КОТОРАЯ ЕСТЬ У ДВИЖКА: он аллоцирует ПЛОСКИЙ int8-буфер на `page_size_bytes * nb` байт и
        # делает `raw.view(get_kv_cache_shape(...))` (gpu_model_runner.py:11868). Разойдись числа --
        # и это RuntimeError на подъёме, а не тихая порча. У `int8_per_token_head` спека сама
        # добавляет 2*bs*Hkv*4 байт на страницу под масштабы (kv_cache_interface.py:157), поэтому
        # здесь ровно +4 байта на (позицию, голову) в последней оси. Как эти байты разложены внутри
        # страницы -- дело бэкенда (см. `_carve`), движок в них не смотрит.
        if kv_cache_uses_per_token_head_scales(cache_dtype_str):
            return (num_blocks, 2, block_size, num_kv_heads, head_size + 4)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # РАСКЛАДКА ПУЛА -- ГЛОБАЛЬНАЯ НАСТРОЙКА, А НЕ НАШЕ РЕШЕНИЕ. Захардкоженный NHD означает, что
        # физический порядок осей у нашего пула отличается от того, что предполагают ОСТАЛЬНЫЕ
        # читатели движка. Спрашиваем ту же функцию, что и чужой бэкенд.
        from vllm.v1.attention.backends.utils import get_kv_cache_layout

        layout = get_kv_cache_layout()
        if layout == "NHD":
            return (1, 0, 2, 3, 4) if include_num_layers_dimension else (0, 1, 2, 3, 4)
        if layout == "HND":
            return (1, 4, 0, 2, 3, 5) if include_num_layers_dimension else (0, 1, 3, 2, 4)
        raise ValueError(f"FA2_SM70: неизвестная раскладка пула {layout!r}")

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 512 -- глобальные слои Gemma-4. Оба пути его умеют: префилл через накопитель выхода
        # (attn_fwd_qbshd/attn_fwd_cutlass, d<=512 и d<=1024 соответственно), декод -- явной веткой
        # EPT=16 (split_decode.cu). Список НЕ расширять "на всякий случай": у пейджированного декода
        # диспетчер кончается веткой d=64, поэтому неподдержанный d давал бы мусор молча (проверка
        # на это добавлена в ядро вместе с окном).
        return [64, 128, 256, 512]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in cls.get_supported_head_sizes()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 7 and capability.minor == 0

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        """ОТКАЗ ОТ KV-КОННЕКТОРОВ -- ЭТО ВТОРАЯ ПОЛОВИНА ГАРАНТИИ ПРИВАТНОГО ФОРМАТА.

        Всё, что переносит блоки ПОБАЙТОВО (swap_blocks, CPU-offload, выгрузка на диск), нашему
        формату безразлично. Но у коннекторов есть пути, которые перекладывают СОДЕРЖИМОЕ:
        `kv_postprocess_blksize_on_receive` / `..._layout_on_receive`
        (kv_transfer/kv_connector/utils.py:223+) делают permute/reshape НА УРОВНЕ ЭЛЕМЕНТОВ, NIXL
        при неравном TP режет блок ПО ГОЛОВАМ, а LMCache/HF3FS/MoRIIO трактуют форму страницы
        по-своему. Ни одно из этого не переживёт ни разложение int8-пула на две области, ни
        масштаб-на-позицию.

        Отказывать здесь честнее, чем документировать: движок ПРОВЕРИТ это на подъёме и выберет
        другой бэкенд, вместо того чтобы отдать по сети страницу без её масштабов.
        """
        return False

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @classmethod
    def supports_combination(cls, *args, **kwargs) -> str | None:
        """Return a REASON string when this backend cannot serve the configuration.

        Returning a reason lets the selector move on; raising here would abort the boot.
        """
        try:
            _ext()
        except ImportError as e:
            return f"fa2_sm70 kernels not importable: {e}"
        return None


class FA2SM70Impl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        # REFUSE AT CONSTRUCTION, NOT IN forward(). These features are absent from the selector's
        # config object, so the engine cannot filter on them; construction time is the last moment
        # at which an honest refusal is still cheap.
        unsupported = []
        if alibi_slopes is not None:
            unsupported.append("alibi_slopes")
        if logits_soft_cap:
            unsupported.append("logits_soft_cap")
        if sinks is not None:
            unsupported.append("sinks")
        if attn_type != AttentionType.DECODER:
            unsupported.append(f"attn_type={attn_type}")
        if kv_sharing_target_layer_name is not None:
            unsupported.append("kv_sharing")
        if kv_cache_dtype not in _DTYPE_FMT:
            unsupported.append(f"kv_cache_dtype={kv_cache_dtype}")
        if head_size not in FA2SM70Backend.get_supported_head_sizes():
            unsupported.append(f"head_size={head_size}")
        if unsupported:
            raise NotImplementedError(
                "FA2_SM70 does not support: "
                + ", ".join(unsupported)
                + ". Pick another backend, or extend the backend rather than the caller."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        # ФОРМАТ ВЫЧИСЛЯЕТСЯ ОДИН РАЗ И В ОДНОМ МЕСТЕ. Именно «одно место» -- содержание урока:
        # авария с Gemma-4 случилась не от того, что int8 плох, а от того, что склад и внимание
        # выводили формат КАЖДЫЙ САМ. Здесь его выводит конструктор, а сверяет отметка на кэше.
        self._fmt = _DTYPE_FMT[kv_cache_dtype]
        # [FA2SM70_KV_I8] Движку объявлен e4m3, байты внутри -- int8 с боковыми масштабами.
        # Формат по-прежнему выводится ЗДЕСЬ и только здесь; читатели сверяют отметку пула.
        if _KV_I8_SIDE and self._fmt == _E4M3:
            self._fmt = _I8
        self.attn_type = attn_type
        self.alibi_slopes = None
        # ОКНО ХРАНИТСЯ ЧИСЛОМ ТОКЕНОВ, а пара (left, right) выставляется по той же конвенции, что у
        # остального движка (triton_attn.py:634-639, flash_attn.py:616-621): DECODER -> (W-1, 0), то
        # есть запрос в позиции p видит ключи [p-W+1, p], всего РОВНО W вместе с собой. Ошибка на
        # единицу здесь не падает и не шумит -- она даёт слегка другой текст, который выглядит
        # правдоподобно; поэтому конвенция взята из чужого рабочего бэкенда, а не выведена.
        self.window = int(sliding_window) if sliding_window else 0
        self.sliding_window = (self.window - 1, 0) if self.window else (-1, -1)
        self.logits_soft_cap = 0
        self.kv_sharing_target_layer_name = None
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.fa2 = _ext()
        # Caller-owned scratch, reused across steps: allocating inside a decode step measures the
        # allocator, not the kernel, and churns the caching allocator at 30 us/step scale.
        self._ws_a: torch.Tensor | None = None
        self._ws_b: torch.Tensor | None = None
        self._kbuf: torch.Tensor | None = None
        self._vbuf: torch.Tensor | None = None
        self._kmax: torch.Tensor | None = None
        self._noalibi: torch.Tensor | None = None
        # ВИРТУАЛЬНЫЙ БАТЧ ДЛЯ СПЕКУЛЯЦИИ (k+1 позиций -> строки батча). Держатся по максимуму и
        # не заводятся по форме: словарь по формам здесь стал бы утечкой того же вида, что уже
        # роняла боевой (кэш обмена по точной форме без чистки).
        self._vbt: torch.Tensor | None = None
        self._vsl: torch.Tensor | None = None
        self._voff: torch.Tensor | None = None
        self._voff_q: int = -1
        self._virt_m = None
        self._virt_key = None
        self._said_buf = False
        # ВЕНТИЛИ ЧИТАЮТСЯ ОДИН РАЗ. os.environ.get на горячем пути -- это три поиска в отображении
        # на КАЖДОМ слое КАЖДОГО шага (48 x 3 за шаг у Gemma-4) ради значения, которое не меняется за
        # жизнь процесса. Фальсификаторы от этого не слабеют: они задаются до подъёма движка.
        self._force_ref = os.environ.get("FA2SM70_FORCE_REF") == "1"
        self._check = os.environ.get("FA2SM70_CHECK") == "1"
        # УМОЛЧАНИЕ ДЕРЖИТСЯ В ОДНОМ МЕСТЕ. Здесь стояло "64" СВОИМ литералом, и это вторая копия
        # того же числа: подними умолчание в fa2_sm70._auto_splits -- бэкенд всё равно обрежет своим.
        # Ровно так ручка и оказывалась мёртвой. Берём значение ОТТУДА, а не повторяем его.
        self._max_splits = int(os.environ.get("FA2SM70_MAX_SPLITS",
                                              str(self.fa2.DEFAULT_MAX_SPLITS)))
        # Ш1: быстрый однородный декод (без выделений и синхронизаций).
        #
        # [29.08] ФЛАГ РАСЩЕПЛЁН. Прежде путь включался ТЕМ ЖЕ FA2SM70_CG, что и заявка
        # AttentionCGSupport -- это связывало ДВЕ РАЗНЫЕ вещи: «считать быстрым путём» и
        # «отдать декод CUDA-графу». Когда 29.08 граф пришлось снять по правильности
        # (со спекуляцией он давал НЕ жадный ответ цели), вместе с ним молча ушли:
        #   * сам быстрый однородный путь (счётчик decode_uniform = 0),
        #   * разреженный декод -- он живёт ВНУТРИ этого пути (decode_sparse = 0),
        # и свип разреженности на 252K сравнивал две ОДИНАКОВЫЕ конфигурации, показав
        # «приза нет». Прибор был пуст, а выглядел как замер.
        # Теперь: FA2SM70_UNIFORM управляет ПУТЁМ (умолчание -- по FA2SM70_CG, чтобы
        # прежнее поведение не менялось молча), FA2SM70_CG -- только заявкой графа.
        self._cg_fast = os.environ.get(
            "FA2SM70_UNIFORM", os.environ.get("FA2SM70_CG", "0")) == "1"
        # Потолок однородного пути по длине запроса: спекуляция -- это единицы (1+k), а префилл
        # сюда попадать НЕ ДОЛЖЕН (виртуальный батч дал бы там O(S^2) чтений KV).
        self._uniform_qmax = int(os.environ.get("FA2SM70_UNIFORM_QMAX", "8"))
        # ПОТОЛОК ПО ДЛИНЕ КОНТЕКСТА -- ВЫБОР ПУТИ ЕСТЬ ФУНКЦИЯ ДЛИНЫ, А НЕ СВОЙСТВО ПУТИ.
        # Замер 29.08 одним прибором на одном тексте (карты 0-1, CG=0, спекуляция k=3):
        #     L=13501   общий 29.9  однородный 39.7   <- однородный
        #     L=75001   общий 25.2  однородный 28.8   <- однородный
        #     L=89655   общий 29.5  однородный 33.1   <- однородный
        #     L=134505  общий 21.5  однородный 24.6   <- однородный
        #     L=179355  общий 26.8  однородный 20.8   <- ОБЩИЙ
        #     L=252096  общий 26.3  однородный 18.9   <- ОБЩИЙ
        # Однородный путь монотонно проседает с длиной, общий -- нет; пересечение между
        # 134K и 179K. Прежде путь брался ВСЕГДА при FA2SM70_CG=1, то есть на длине мы
        # платили за него до 39 %. Порог 150000 -- середина замеренной вилки.
        # При ЗАЯВЛЕННОМ графе потолок снимается: захват идёт при max_seq_len = max_model_len,
        # и ограничение по длине выключило бы путь ровно под захватом (а тогда графу достался
        # бы путь с выделениями -- тихая порча, ради которой ниже стоит громкий отказ).
        # [ПОРОГ ПЕРЕМЕРЕН 31.08 -- 150000 УСТАРЕЛ, СТАЛО 65000]
        # Замер, по которому стояло 150000, снят ДО расщепления по ключам, а оно ускоряет
        # ИМЕННО общий путь (там была вырожденная сетка -- 12 блоков на 80 SM, x6.55 на 262K).
        # Значит пересечение обязано было сдвинуться вниз, и оно сдвинулось. Механизм виден на
        # ядре: однородный путь разворачивает k+1 позиций в виртуальный батч, и каждая строка
        # читает префикс ЗАНОВО -- прямой замер декодного ядра даёт q=4 ровно в 3.84 раза
        # дороже q=1 (147K: 1.744 против 0.454 мс, КПД пола чтения KV падает 36.9 % -> 9.6 %).
        # Общий путь читает KV один раз. Сквозной наклонный прибор (стенд, k=3, ABBA):
        #     длина    однородный   общий     кто
        #      15.7K      35.39      43.00    однородный (-18 %)
        #      34.1K      45.19      47.59    однородный (-5 %)
        #      65.0K      49.41      49.52    ничья -- ПЕРЕСЕЧЕНИЕ
        #     126.9K      62.63      51.02    ОБЩИЙ (+19 %)
        # Порог 65000 -- сама точка пересечения.
        self._uniform_maxlen = (
            (1 << 62) if os.environ.get("FA2SM70_CG") == "1"
            else int(os.environ.get("FA2SM70_UNIFORM_MAXLEN", "65000")))
        # Ленивые ссылки на расширения: decode_ext()/prefill_ext() -- это JIT-загрузчики со своим
        # кэшем, но всё же вызов и ветвление на каждом обращении.
        self._dext = None
        self._pext = None
        self._pool_last = None      # память на последний пул (см. _pool)
        self._sm64 = None           # память на int64-вид slot_mapping (см. _store_bytes)
        # SP93: активность решается ОДИН РАЗ по имени слоя (см. forward) -- None = не решено.
        self._sp93_act: bool | None = None
        # РЕЖИМ ОКНА В ДЕКОДЕ -- ВЕНТИЛЬ, ЧТОБЫ ВЕТКИ МОЖНО БЫЛО СРАВНИТЬ ЗАМЕРОМ, А НЕ ВКУСОМ:
        #   exact (по умолчанию) -- левая граница в ядре, точна до токена;
        #   block -- та же граница, округлённая ВНИЗ до block_size: ровно то, что даёт подмена
        #            строки block_table без правки ядра (путь "б" из задания);
        #   full  -- границы нет вовсе (путь "как было"): показывает, что окно вообще необходимо.
        self._winmode = os.environ.get("FA2SM70_DEC_WIN", "exact")
        from vllm.v1.attention.backends.utils import get_kv_cache_layout

        _say_once(
            f"backend=FA2_SM70 heads={num_heads}/{num_kv_heads} d={head_size} "
            f"окно={self.window} kv={kv_cache_dtype} раскладка_пула={get_kv_cache_layout()}"
        )

    # ------------------------------------------------------------ extensions

    def _dec(self):
        if self._dext is None:
            self._dext = self.fa2._ext.decode_ext()
        return self._dext

    def _pre(self):
        if self._pext is None:
            self._pext = self.fa2._ext.prefill_ext()
        return self._pext

    # ---------------------------------------------------------------- scratch

    # [ЖИЗНЬ БУФЕРА КОРОЧЕ ЕГО ИСПОЛЬЗОВАНИЯ, 08.09]
    # Все таблицы индексов проверены изнутри графа и ЧИСТЫ в момент падения (§192),
    # значит порча не адресная, а ВРЕМЕННАЯ: по буферу пишут, когда он уже чужой.
    # Вот механизм: рабочий буфер растёт по требованию, старый ОТПУСКАЕТСЯ, а в
    # захваченном графе его адрес ЗАПЕЧЁН. Дальше любой повтор графа пишет в память,
    # которую распределитель уже отдал кому-то другому. Сходится со всеми тремя
    # условиями: нужен граф (адрес запечён), нужна спекуляция (новые формы вызывают
    # рост), нужен int16-пул (меняет пороги и размеры).
    # Лечение: отставленный буфер НЕ отпускать -- он стоит копейки и живёт до конца
    # процесса, зато ни один запечённый указатель не становится висячим.
    def _отставить(self, т):
        if т is None:
            return
        п = getattr(self, "_пенсия", None)
        if п is None:
            п = self._пенсия = []
        п.append(т)

    def _workspace(self, B: int, H: int, ns: int, d: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        need_a = B * H * ns * d
        if self._ws_a is None or self._ws_a.numel() < need_a:
            if _ДЕРЖАТЬ_БУФЕРЫ:
                self._отставить(self._ws_a); self._отставить(self._ws_b)
            self._ws_a = torch.empty(need_a, dtype=torch.float32, device=device)
            self._ws_b = torch.empty(2 * B * H * ns, dtype=torch.float32, device=device)
        return self._ws_a, self._ws_b

    def _gather_buf(self, Hkv: int, T: int, d: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Staging for the paged->contiguous gather: a FLAT pool, viewed TIGHTLY per call.

        ЗАПАС ОБЯЗАН ЛЕЖАТЬ ЗА ВИДОМ, А НЕ ВНУТРИ НЕГО. Первая версия отдавала тензор
        [Hkv, T + _TILE_PAD, d], и это дало связный, но НЕВЕРНЫЙ ответ модели: ядро сборки считает
        шаг головы как T*d, поэтому головы 1..Hkv-1 легли мимо своих мест, а нули паддинга поехали в
        данные. Диагностика заняла шесть проходов именно потому, что torch-эталон читал ТЕ ЖЕ
        буферы и потому соглашался с ядром -- ошибку такого рода видно только сравнением с ЧУЖИМ
        путём целиком. Плоский пул с запасом ёмкости даёт и плотный вид (шаг = T*d, как ждёт ядро),
        и валидную память за концом данных (плитка читает свой конец за границей; torch.empty
        впритык = NaN на первом длинном запросе).
        """
        need_cap = Hkv * (T + _TILE_PAD) * d
        if self._kbuf is None or self._kbuf.numel() < need_cap:
            # РОСТ УДВОЕНИЕМ УБРАН (07.08.2026, отказ на боевой сети). Здесь стояло
            # cap = max(need_cap, 2 * предыдущий, 1<<20). Удвоение амортизирует число выделений и
            # безобидно ПРИ ЗАПАСЕ, но у потолка оно смертельно: KV-кэш нарезан движком по
            # профилирующему прогону (он этот пик не проходит), и очередной шаг удвоения попросил
            # 522 МиБ там, где свободно 465 -> CUDA OOM -> движок мёртв. Растём РОВНО до нужного,
            # и старый буфер отпускаем ДО выделения нового, иначе пик держит обе плиты разом.
            #
            # ЭТО ОБХОД, А НЕ РЕШЕНИЕ. Правильный ход -- не заводить буфер вовсе: постраничное
            # чтение пула телом внимания (Ш0 в docs/ARCH_FUSED_STEP.md) снимает и этот пик,
            # и два пересечения HBM из четырёх. Буфер нужен ТОЛЬКО значению, пересекающему
            # границу запуска.
            # ФАЛЬСИФИКАТОР 08.08: `torch.cuda.empty_cache()` здесь стоял МОЙ, добавленный сегодня
            # вместе со снятием удвоения. Буфер ПЕР-СЛОЙНЫЙ (у каждого из 16 слоёв свой), значит на
            # первом длинном запросе кэш аллокатора освобождался ШЕСТНАДЦАТЬ раз, и каждая
            # последующая выдача шла к драйверу. Гейт FA2SM70_GBUF_PURGE=1 возвращает прежнее
            # поведение, чтобы доля этой фазы читалась РАЗНОСТЬЮ, а не оценивалась.
            cap = max(need_cap, 1 << 20)
            if os.environ.get("FA2SM70_GBUF_PURGE") == "1":
                self._kbuf = None
                self._vbuf = None
                torch.cuda.empty_cache()
            self._kbuf = torch.zeros(cap, dtype=torch.float16, device=device)
            self._vbuf = torch.zeros(cap, dtype=torch.float16, device=device)
        n = Hkv * T * d
        return self._kbuf[:n].view(Hkv, T, d), self._vbuf[:n].view(Hkv, T, d)

    # ------------------------------------------------------------- kv writing

    def _pool(self, kv_cache: torch.Tensor, who: str) -> dict:
        """Нарезка пула + СВЕРКА ОТМЕТКИ ФОРМАТА. Точка, через которую обязан пройти каждый читатель.

        Отсутствие отметки означает, что склад в этот пул не писал (или писал ЧУЖОЙ), и тогда байты
        значат не то, что мы собираемся прочитать. Молча прочитать -- это и есть тот самый сервер,
        который поднимается и врёт; поэтому здесь падение.
        """
        # [ПАМЯТЬ НА ПОСЛЕДНИЙ ПУЛ -- 31.08] `_pool` зовётся на КАЖДЫЙ слой каждого шага, а
        # `_pool_key` каждый раз строит кортеж (data_ptr + tuple(shape)): по профилю боевого
        # это 0.14 % снимков -- чистые накладные питона, работы там нет. Пул выделяется один
        # раз, поэтому память по ТОЖДЕСТВУ объекта попадает почти всегда; промах просто идёт
        # прежним путём, так что рассинхрона (урок `_общий_ключ`) здесь быть не может.
        # Сверка формата остаётся: она уже прошла для этого объекта при первом вызове, а
        # формат пула по построению не меняется.
        _зп = self._pool_last
        if _зп is not None and _зп[0] is kv_cache:
            return _зп[1]
        p = _POOL.get(_pool_key(kv_cache))
        if p is None:
            raise RuntimeError(
                f"FA2_SM70 ({who}): пул без отметки формата -- склад в него не писал. "
                f"Читать байты как '{self._fmt}' нельзя: формат ставит СКЛАД, а читатель СВЕРЯЕТ.")
        if p["fmt"] != self._fmt:
            raise RuntimeError(
                f"FA2_SM70 ({who}): в пуле формат '{p['fmt']}', а слой читает '{self._fmt}'. "
                f"Склад и внимание обязаны гейтиться ОДНИМ условием.")
        self._pool_last = (kv_cache, p)
        return p

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        if self._fmt != _FP16:
            self._store_bytes(layer, key, value, kv_cache, slot_mapping)
            return
        key_cache, value_cache = kv_cache.unbind(1)
        block_size = key_cache.shape[1]
        # ОТМЕТКА СТАВИТСЯ ДО ВЕТВЛЕНИЯ НА ФАЛЬСИФИКАТОР, И ЭТО ИСПРАВЛЕННЫЙ ДЕФЕКТ, А НЕ ПОРЯДОК
        # РАДИ ПОРЯДКА. Поставь её после -- и `FA2SM70_VENDOR_STORE=1` (чужой склад, заведомо
        # рабочий, тот самый A/B по одному узлу) начал бы валить ЧИТАТЕЛЯ по «пул без отметки»:
        # фальсификатор ломался бы о новый гейт вместо того, чтобы проверять код. Отметка говорит
        # «в пуле лежит fp16», и чужой склад пишет ровно его -- значит она верна и здесь.
        ck = _pool_key(kv_cache)
        if _POOL.get(ck, {}).get("fmt") != _FP16:
            # Отметка ставится и на fp16-пуле, а не только на байтовом: читатель обязан УПАСТЬ, если
            # склад не тот, независимо от формата. Гейт, работающий только в «интересном» случае, --
            # это гейт, который не проверяли. Перезапись при несовпадении -- по той же причине, что
            # и в `_store_bytes`: склад задаёт формат, а адрес пула переиспользуется между движками.
            _POOL[ck] = _carve(kv_cache, _FP16)
        # ОТРИЦАТЕЛЬНЫЙ СЛОТ -- ЭТО НАБИВКА, А НЕ АДРЕС. vLLM ставит -1 добитым позициям (в
        # разогревочном прогоне -- ВСЕМ 256). Наивное деление даёт blk = -1, то есть запись в
        # ПОСЛЕДНИЙ блок пула: сначала портится чужая страница, а на боевом батче -- живые данные
        # соседнего запроса. Чужие ядра склада просто пропускают такие токены (проверка slot < 0
        # внутри ядра). Здесь тот же смысл БЕЗ синхронизации с хостом: набивка уводится в нулевой
        # блок-заглушку, который пул резервирует и никогда не читает (kv_cache_utils.py:1254).
        # [ЧУЖОЙ СКЛАД -- УМОЛЧАНИЕ НА fp16-ПУЛЕ, ЗАМЕР 02.08.2026]
        # Наш склад раскладывается на ВОСЕМЬ ядер (индексная арифметика + две записи по продвинутому
        # индексу), чужой -- ОДНО. Замер в ЖИВОМ движке, парный A/B с чередованием внутри процесса,
        # 9 раундов, бутстрап: -1.3...-2.3 % СКВОЗНОЙ СТЕНЫ префилла. Результат тот же (чужое ядро
        # пишет ровно fp16, отметка пула стоит до ветвления и остаётся верной).
        # NB: на БАЙТОВОМ пуле (e4m3, боевой путь Gemma-4) выигрыш РОВНО НОЛЬ -- там наш склад и так
        # одно ядро. Поэтому переключается только эта ветка, а ветка e4m3 ниже остаётся
        # фальсификатором, как и была.
        # NB2: межпроцессный A/B этого эффекта НЕ ВИДИТ -- он дал «+0.6 % и +2.2 % хуже» при падении
        # времени ядер на 2.0 %. Виден только парным чередованием ВНУТРИ процесса.
        # Откат: FA2SM70_VENDOR_STORE=0 -- он же оставляет наш склад для сверок.
        if os.environ.get("FA2SM70_VENDOR_STORE", "1") != "0":
            from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
                triton_reshape_and_cache_flash,
            )

            one = torch.ones(1, dtype=torch.float32, device=key.device)
            triton_reshape_and_cache_flash(
                key, value, key_cache, value_cache, slot_mapping, "auto", one, one
            )
            _bump("store")
            _say_once("склад fp16 = triton_reshape_and_cache_flash (умолчание, -1.3..-2.3 % стены)")
            return
        valid = slot_mapping >= 0
        slot = torch.where(valid, slot_mapping, torch.zeros_like(slot_mapping))
        blk = torch.div(slot, block_size, rounding_mode="floor")
        off = slot - blk * block_size
        key_cache[blk, off] = key
        value_cache[blk, off] = value
        _bump("store")
        if self._check:
            # РАЗРЫВ СИММЕТРИИ. Сверки префилла и декода обе читают ПУЛ, поэтому ошибка склада им
            # не видна: и ядро, и эталон получат одинаково испорченные данные и сойдутся. Здесь
            # сравниваем содержимое пула с СЫРЫМ key/value, который движок только что передал.
            back_k = key_cache[blk, off]
            back_v = value_cache[blk, off]
            neg = int((slot_mapping < 0).sum().item())
            logger.info(
                "[fa2_sm70] СВЕРКА склад: слотов=%d отрицательных=%d уник=%d "
                "k_равно=%s v_равно=%s key=%s",
                slot_mapping.numel(), neg, int(torch.unique(slot_mapping).numel()),
                bool(torch.equal(back_k, key)), bool(torch.equal(back_v, value)),
                tuple(key.shape))

    def _sm_i64(self, slot_mapping):
        """int64-вид slot_mapping, посчитанный ОДИН раз на шаг, а не на каждый слой.

        [ПОВТОРНАЯ РАБОТА 31.08] `slot_mapping.to(torch.int64)` стоял в `_store_bytes` и звался
        на КАЖДЫЙ слой внимания: шестнадцать одинаковых конвертаций за шаг, каждая с выделением
        и копией. Тензор для всех слоёв ОДИН И ТОТ ЖЕ объект (его строит раннер на шаг), поэтому
        память по тождеству попадает всегда; промах просто считает заново.
        """
        з = self._sm64
        if з is not None and з[0] is slot_mapping:
            return з[1]
        и = slot_mapping.to(torch.int64)
        self._sm64 = (slot_mapping, и)
        return и

    def _store_bytes(self, layer, key, value, kv_cache, slot_mapping) -> None:
        # [ФАЛЬСИФИКАТОР ЦЕНЫ ПИТОНА 31.08] Два рычага, читается ТОЛЬКО ВРЕМЯ (ответ неверен):
        #   FA2SM70_NOSTORE=1        -- снята ВСЯ фаза склада (обвязка + ядро);
        #   FA2SM70_NOSTORE_KERNEL=1 -- снято только ЯДРО, питонная обвязка исполняется.
        # Разность даёт цену обвязки отдельно от цены ядра. Это нужно потому, что py-spy
        # приписывает время, проведённое ВНУТРИ вызванного C++, той питонной строке, с которой
        # вызов сделан: без такого разделения «наш питон 3.7 %» не отличим от «наши ядра 3.7 %».
        if _NOSTORE:
            return
        """БАЙТОВЫЙ СКЛАД: квантование + запись в пул ОДНИМ ядром, без промежуточного тензора.

        Отрицательный слот тут НЕ нужно обходить руками: оба ядра (`reshape_and_cache_e4m3` /
        `_i8`) проверяют `slot < 0` внутри и просто не пишут -- то есть набивка отбрасывается там,
        где ей и место, без синхронизации с хостом и без блока-заглушки.

        ЕДИНИЦЫ МАСШТАБА -- РАЗНЫЕ У ДВУХ ФОРМАТОВ, И ЭТО НЕ ДЕТАЛЬ:
          e4m3 -- скаляр СЛОЯ (`layer._k_scale`), значение в пуле = quant(v / k_scale); он нужен,
                  чтобы вписать активации в узкий диапазон e4m3 (|x| <= 448), и его же обязан
                  применить читатель;
          int8 -- масштаб НА ПОЗИЦИЮ, и он уже несёт истинные единицы fp16 (значение = q*s),
                  поэтому `layer._k_scale` на этом пути НЕ УЧАСТВУЕТ ВОВСЕ. Пропусти это -- и
                  ошибка будет тихой: при k_scale=1.0 (умолчание) всё сойдётся, а на модели с
                  загруженными из чекпойнта масштабами разъедется.
        """
        # Та же память на пул, что в `_pool`: ключ строился на КАЖДЫЙ слой (data_ptr +
        # tuple(shape)), хотя пул один и тот же. Формат сверяется ниже, как и прежде.
        _зп = self._pool_last
        if _зп is not None and _зп[0] is kv_cache:
            ck, p = None, _зп[1]
        else:
            ck = _pool_key(kv_cache)
            p = _POOL.get(ck)
        # ОТМЕТКА ПЕРЕЗАПИСЫВАЕТСЯ, ЕСЛИ ФОРМАТ РАЗОШЁЛСЯ, И ЭТО НЕ ОСЛАБЛЕНИЕ ГЕЙТА.
        # Гейт защищает ЧИТАТЕЛЯ от рассинхрона со складом; сам склад -- источник истины, он прямо
        # сейчас положит в эти байты свой формат. Случай не гипотетический: два движка подряд В ОДНОМ
        # ПРОЦЕССЕ (так устроена приёмка) могут получить от аллокатора ТОТ ЖЕ адрес и ту же форму, и
        # без перезаписи второй подъём падал бы на «в пуле формат X, а слой читает Y» -- отказ верный
        # по форме и ложный по существу.
        if p is not None and p["fmt"] != self._fmt:
            _say_once(f"пул переотмечен: {p['fmt']} -> {self._fmt} (новый движок на том же адресе)",
                      key=f"переотметка {self._fmt}")
            p = None
        if p is None:
            p = _carve(kv_cache, self._fmt)
            if self._fmt == _E4M3:
                # Граница ||k_raw|| на kv-голову: ядро склада сворачивает в неё новые токены тем же
                # проходом. Отложенному декоду она больше не нужна (он перешёл на онлайн-максимум),
                # но аргумент обязателен, а постоянный адрес -- то, что делает её пригодной внутри
                # захваченного графа, если он когда-нибудь появится.
                p["kbound"] = torch.zeros(int(kv_cache.shape[3]), dtype=torch.float32,
                                          device=kv_cache.device)
            if ck is None:
                ck = _pool_key(kv_cache)
            _POOL[ck] = p
            self._pool_last = (kv_cache, p)
            _say_once(f"склад = БАЙТОВЫЙ {self._fmt}: пул {tuple(kv_cache.shape)} "
                      f"d={p['d']} масштабы={'таблица на позицию' if p['ks'] is not None else 'скаляр слоя'}",
                      key=f"склад {self._fmt} d={p['d']}")
        ext = self._dec()
        if self._fmt == _E4M3:
            ks = float(getattr(layer, "_k_scale_float", 1.0) or 1.0)
            vs = float(getattr(layer, "_v_scale_float", 1.0) or 1.0)
            if os.environ.get("FA2SM70_VENDOR_STORE") == "1":
                # A/B РОВНО ПО ОДНОМУ УЗЛУ, И ИМЕННО ЗДЕСЬ ОН ЦЕНЕН БОЛЬШЕ ВСЕГО. Наш склад и наше
                # чтение пишут/читают ОДИН буфер, поэтому общая ошибка кодирования им обоим
                # НЕВИДИМА -- они согласятся, будучи оба неверны. Чужое ядро (`reshape_and_cache_flash`
                # с "fp8_e4m3", тот самый путь, которым на sm_70 идёт TRITON_ATTN) ломает эту
                # симметрию: если наш ответ чинится подменой ТОЛЬКО склада, ошибка в кодировании,
                # а не в чтении. Для int8 такой подмены не существует по построению -- формат наш.
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key, value, p["k"].view(torch.float8_e4m3fn),
                    p["v"].view(torch.float8_e4m3fn), self._sm_i64(slot_mapping),
                    "fp8_e4m3", layer._k_scale, layer._v_scale)
                _say_once("склад = ЧУЖОЙ reshape_and_cache_flash(fp8_e4m3) (A/B)")
            else:
                ext.reshape_and_cache_e4m3(key, value, p["k"], p["v"], slot_mapping,
                                           ks, vs, p["kbound"], 1.15)
            _bump("store_e4m3")
        else:
            if not _NOSTORE_K:
                ext.reshape_and_cache_i8(key, value, p["k"], p["v"], slot_mapping, p["ks"], p["vs"])
            _bump("store_i8")
            if _SPARSE_DEC or _SPARSE_POOL:
                # Средние ключей ТРОНУТЫХ подблоков -- сразу после записи, пока слоты под рукой.
                # [ПРЕФИЛЛ ТОЖЕ] Раньше таблицы велись только под разреженный ДЕКОД, а префилл
                # каждый чанк пересчитывал средние ПО ВСЕМУ контексту заново. Замер 28.08:
                # отбор через готовую таблицу 4.84 мс против 5.79 полного пересчёта, и отбор
                # СОВПАЛ ПОБИТОВО. Обновление тут стоит один тронутый подблок на чанк.
                _kb = _kbar_pool(p, _SPARSE_B)
                if _kb is not None:
                    _dp = _dev_pool(p, _SPARSE_B)
                    if _dp is not None:
                        self._pre().blk_mean_update(
                            p["k"], p["ks"], self._sm_i64(slot_mapping), _kb, _SPARSE_B, _dp)
                    else:
                        self._pre().blk_mean_update(
                            p["k"], p["ks"], self._sm_i64(slot_mapping), _kb, _SPARSE_B)
                    _bump("kbar_upd")
        _bump("store")
        if self._check:
            self._check_store(key, value, p, slot_mapping, layer)

    def _check_store(self, key, value, p, slot_mapping, layer) -> None:
        """ЦЕНА КВАНТОВАНИЯ, ЗАМЕРЕННАЯ НА СКЛАДЕ, -- И ЭТО ДРУГОЕ ЧИСЛО, ЧЕМ ОШИБКА ЯДРА.

        Сверки префилла и декода читают ПУЛ, поэтому сравнивают ядро с torch на УЖЕ КВАНТОВАННЫХ
        значениях -- это метрика ЯДРА. Здесь наоборот: содержимое пула разворачивается обратно и
        сравнивается с СЫРЫМ key/value, который движок только что передал, -- это цена ФОРМАТА.
        Путать их нельзя: первая обязана быть на уровне шума fp16, вторая -- заведомо нет.
        """
        with torch.no_grad():
            # ВАЛИДНЫЕ СЛОТЫ ОТБИРАЮТСЯ ВМЕСТЕ СО СВОИМИ СТРОКАМИ, а не «первые n». Набивка (-1)
            # обычно стоит в хвосте, но полагаться на это -- значит сверять чужие пары (токен, слот)
            # и получить большое relL2 на ровном месте, приняв его за порчу формата.
            pos = (slot_mapping >= 0).nonzero(as_tuple=True)[0]
            sl = slot_mapping[pos]
            if sl.numel() == 0:
                return
            bs = int(p["k"].shape[1])
            blk, off = torch.div(sl, bs, rounding_mode="floor"), sl % bs
            d = p["d"]
            if p["ks"] is not None:
                hkv = int(p["k"].shape[2])
                s_idx = sl.unsqueeze(1) * hkv + torch.arange(hkv, device=sl.device)
                bk = p["k"][blk, off].float() * p["ks"][s_idx].unsqueeze(-1)
                bv = p["v"][blk, off].float() * p["vs"][s_idx].unsqueeze(-1)
            else:
                ks = float(getattr(layer, "_k_scale_float", 1.0) or 1.0)
                vs = float(getattr(layer, "_v_scale_float", 1.0) or 1.0)
                bk = p["k"][blk, off].view(torch.float8_e4m3fn).float() * ks
                bv = p["v"][blk, off].view(torch.float8_e4m3fn).float() * vs
            n = sl.numel()
            rk = key.reshape(-1, key.shape[-2], d)[:n].float()
            rv = value.reshape(-1, value.shape[-2], d)[:n].float()
            logger.info("[fa2_sm70] СВЕРКА склад(%s) ЦЕНА КВАНТОВАНИЯ: relL2 K=%.3e V=%.3e "
                        "токенов=%d d=%d", self._fmt, _relL2(bk, rk), _relL2(bv, rv), n, d)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FA2SM70Metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "FA2_SM70 requires the caller-provided output buffer"
        if getattr(self, "_sparse_layer_num", None) is None:
            # номер МОДЕЛЬНОГО слоя из layer_name ("model.layers.N. ..."); не распарсился -> -1 (как было: разреженно)
            _nm = getattr(layer, "layer_name", "") or ""
            try:
                self._sparse_layer_num = int(_nm.split("layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                self._sparse_layer_num = -1
        if _SPARSE_LAYER_FILE:
            _sloi_obnovit()
            _num = self._sparse_layer_num
            _m = _SLF["mass"].get(_num)
            self._sparse_layer_ok = (_num < 0) or (_SLF["lmin"] <= _num <= _SLF["lmax"] and not (_m is not None and _m >= 1.0))
            self._sparse_mass_layer = _m if (_m is not None and _m < 1.0) else None
        elif getattr(self, "_sparse_layer_ok", None) is None:
            _num = self._sparse_layer_num
            _m = _SPARSE_MASS_LAYERS.get(_num)
            self._sparse_layer_ok = (_num < 0) or (_SPARSE_LMIN <= _num <= _SPARSE_LMAX and not (_m is not None and _m >= 1.0))
            self._sparse_mass_layer = _m if (_m is not None and _m < 1.0) else None
            if not self._sparse_layer_ok:
                _say_once(f"послойный отбор: слой {_num} идёт ПЛОТНО", key=f"sparse-layer-{_num}")
            elif self._sparse_mass_layer is not None:
                _say_once(f"послойный отбор: слой {_num} масса {self._sparse_mass_layer}", key=f"sparse-mass-{_num}")
        _kva_dump(self, query, key, value, attn_metadata)
        if _SP93 and self._sp93_act is None:
            # Номер МОДЕЛЬНОГО слоя из layer_name ("model.layers.N. ...") -- работает и в PP,
            # и в TP без знания о нарезке рангов. Не распарсилось -> отбор ВЫКЛ и сказано вслух.
            nm = getattr(layer, "layer_name", "") or ""
            try:
                num = int(nm.split("layers.")[1].split(".")[0])
                self._sp93_act = num >= _SP93_MINLAYER
            except (IndexError, ValueError):
                self._sp93_act = False
                _say_once(f"SP93: layer_name '{nm}' не распарсился -- отбор выключен на слое",
                          key=f"sp93-noname-{id(self)}")
        if output_block_scale is not None:
            raise NotImplementedError("FA2_SM70: output_block_scale is not supported")
        if attn_metadata is None:
            # Profiling / warm-up run: the KV pool is not populated yet.
            return output.fill_(0)

        if _MEGA_CHECK:
            _mega_probe()
            global _MEGA_TICK
            _MEGA_TICK = globals().get("_MEGA_TICK", 0) + 1
            if _MEGA_TICK in (8, 64, 512):
                try:
                    import fa2_sm70.megastep as _ms2
                    logger.info("[fa2_sm70 мегашаг] СЧЁТЧИК на %d-м входе: %s",
                                _MEGA_TICK, dict(list(_ms2.СЧЁТ.items())[:4]))
                except Exception as _e2:
                    logger.info("[fa2_sm70 мегашаг] счётчик недоступен: %s", _e2)

        if _FALSE_ATTN:
            # ФАЛЬСИФИКАТОР СНЯТИЕМ ФАЗЫ: полное внимание НЕ СЧИТАЕТСЯ вовсе. Склад KV при этом
            # по-прежнему наполняется выше по стеку, поэтому снимается именно фаза ЧТЕНИЯ+СЧЁТА
            # внимания, а не запись пула. Ответ заведомо неверен -- читается только время.
            return output.fill_(0)

        m = attn_metadata
        n = m.num_actual_tokens

        # =====================================================================================
        # Ш1. ОДНОРОДНЫЙ ДЕКОД БЕЗ ЕДИНОГО ВЫДЕЛЕНИЯ И БЕЗ ЧТЕНИЯ ХОСТОМ (путь под CUDA-граф).
        #
        # ЗАЧЕМ. Мы сами объявили `AttentionCGSupport.NEVER`, и из-за этого КАЖДЫЙ из 16 слоёв
        # полного внимания РВЁТ граф: шаг декода рубится на куски, а вся обвязка идёт через хозяина.
        # Замерено, что это стоит: 877 запусков за шаг, фазы чистой выдачи 3.943 мс = 12.6 % шага,
        # плюс 34 % стены вне execute_model. Это самая крупная статья, которую слитый путь снимает.
        #
        # ПОЧЕМУ ЭТО ВООБЩЕ ВОЗМОЖНО СЕЙЧАС. Прежний запрет был верен ДЛЯ ШАГА 1 («нет постоянных
        # буферов -- граф запёк бы протухшие указатели»). С тех пор появилось всё нужное:
        #   * рабочая область A/Bw постоянна (self._workspace);
        #   * `_kmax` уже держат нулевым тензором ИМЕННО ради постоянного указателя в графе;
        #   * ядро пишет ПРЯМО в выходной буфер движка (flash_decode_defer_mqa_paged_into).
        # Оставались три выделения и питоновские списки -- и они исчезают САМИ, если принять контракт
        # UNIFORM_SINGLE_TOKEN_DECODE: там все строки декодные и идут ПОДРЯД, значит
        #   idx == arange(B)  ->  q4 = query[:B].view(...)      ВИД, не копия
        #   bt  == block_table[:B]                              ВИД
        #   sl  == seq_lens[:B]                                 ВИД
        # То есть быстрый путь не «оптимизация», а ровно то же вычисление без обвязки.
        #
        # ГЕЙТ. Умолчание ВЫКЛЮЧЕНО: захват графа надо доказывать захватом, а не заявкой, и до
        # доказательства поведение остаётся прежним. Включается FA2SM70_CG=1.
        # [СПЕКУЛЯЦИЯ] q > 1 -- это НЕ префилл, а однородный декод на k+1 позициях. Раньше такой
        # батч уходил на префилл-путь, и там сетка вырождалась: (Sq/BQ) x H x B = 1 x 24 x 1 = 24
        # блока на 80 SM, то есть машина простаивала на 70 %. Замерено 21.08: шаг 277 мс против
        # 29.5 при acceptance до 4.00 из 4 -- то есть догадки были отличные, а платили за укладку.
        # `n` -- ЧИСЛО СТРОК ДЕКОДА (токенов), а НЕ запросов: при однотокенном декоде они совпадали,
        # и это совпадение читалось как тождество. Замерено прибором 22.08: при спекуляции
        # n=4, num_reqs=1 -- условие `n == num_reqs` ложно по построению, и путь молча отключался.
        # [ТРИ ЦЕЛЫХ -- ОДИН РАЗ НА ШАГ, А НЕ НА КАЖДЫЙ СЛОЙ, 31.08]
        # `max_query_len`, `num_reqs`, `max_seq_len` -- поля ОДНИХ И ТЕХ ЖЕ метаданных для всех
        # слоёв группы, а `int()` над каждым исполнялся на каждом входе в forward (16 слоёв x
        # три преобразования плюс getattr за шаг). Память держим НА ОБЪЕКТЕ метаданных: он
        # строится заново каждый шаг, значит устареть не может -- тот же приём, что
        # `_fa2_idx_pam` в qwen3_next, и, в отличие от кэша по id(), чужое отдать не может.
        _ц = getattr(m, "_fa2_ints", None)
        if _ц is None:
            _ц = (int(m.max_query_len), int(m.num_reqs),
                  int(getattr(m, "max_seq_len", 0) or 0))
            try:
                m._fa2_ints = _ц
            except Exception:
                pass
        _q, _B, _мсл = _ц
        # ВЕРХНЯЯ ГРАНИЦА ОБЯЗАТЕЛЬНА. Виртуальный батч верен и для префилла (позиция j видит
        # seq_len-(q-1-j) -- это и есть причинная маска), но там он был бы катастрофой: q=8192
        # строк, каждая читает СВОЙ префикс целиком, то есть O(S^2) чтений KV вместо тайлинга.
        # Спекуляция -- это q = 1+k, единицы; префилл сюда попадать не должен.
        # ПРИЧИННОСТЬ -- УСЛОВИЕ ВХОДА. Виртуальный батч по построению причинный: строка j видит
        # seq_len-(q-1-j) ключей. У блочного черновика маска ДВУСТОРОННЯЯ, и взять его сюда --
        # значит тихо посчитать не ту задачу. Такие метаданные идут общим путём.
        # Причинность и окно БОЛЬШЕ НЕ ЗАПИРАЮТ этот путь: первая выражается наличием вычитания
        # смещения в виртуальном батче, второе -- левой границей `kv_start` (см. _decode_uniform).
        # Режим окна "block" оставлен на прежнем пути: он округляет границу ВНИЗ до блока, и это
        # ДРУГОЙ ответ, а не то же самое быстрее.
        if (self._cg_fast and _B > 0 and 1 <= _q <= self._uniform_qmax
                and n == _B * _q and query.shape[0] == n
                and _мсл <= self._uniform_maxlen
                and self._winmode != "block"
                and not (self._force_ref or self._check)):
            r = self._decode_uniform(query, output, kv_cache, m, _B, _q)
            if r is not None:
                return r
        # ГРОМКИЙ ОТКАЗ ВМЕСТО ТИХОЙ ПОРЧИ. Если идёт ЗАХВАТ ГРАФА, а однородный путь не взят, то
        # графу достанется обычный путь -- с выделениями и запечёнными указателями, и он даст
        # СВЯЗНЫЙ, НО НЕВЕРНЫЙ текст (поймано 21.08: гейт «17*23» вернул «19*100=3000»). Такое
        # обязано падать на подъёме, а не портить выход молча.
        # ОТКАЗ СУЖЕН ДО СЛУЧАЯ, ГДЕ ПУТЬ ОБЯЗАН РАБОТАТЬ. Первая версия падала при ЛЮБОМ захвате --
        # в том числе на piecewise-захвате ПРЕФИЛЛА, где однородный путь и не должен браться
        # (q там равно длине промпта). Боевой из-за этого не отвечал: execute_model зависал по
        # таймауту RPC. Условие теперь то же, что у входа в путь.
        if (torch.cuda.is_current_stream_capturing() and self._cg_fast
                and 1 <= _q <= self._uniform_qmax and n == _B * _q
                and getattr(m, "causal", True)):
            raise RuntimeError(
                "fa2_sm70: захват графа при query_len=%d, но однородный путь НЕ взят "
                "(строк=%d, запросов=%d, токенов=%d). Граф получил бы путь с выделениями -- "
                "это тихая порча. Либо согласуйте формы, либо снимите объявление "
                "AttentionCGSupport." % (_q, n, _B, query.shape[0]))
        # РАСКЛАДКА БУФЕРОВ -- ПЕЧАТАЕТСЯ, А НЕ ПРЕДПОЛАГАЕТСЯ. У чужого бэкенда в этом форке
        # query = [tokens, H, d], а output = [tokens, H*d] (плоский). Ошибиться здесь -- значит
        # получить связный, но НЕВЕРНЫЙ текст, что ровно и случилось на первом прогоне.
        # Флаг на СЛОЕ, а не ключ по тексту: число токенов в куске меняется каждый шаг, поэтому без
        # него эта строка форматировалась бы (tuple(), f-строка) на каждом слое каждого шага.
        if not self._said_buf:
            self._said_buf = True
            _say_once(f"буферы: query={tuple(query.shape)} output={tuple(output.shape)} "
                      f"kv={tuple(kv_cache.shape)}", key=f"буферы d={self.head_size}")
        out3 = output.view(-1, self.num_heads, self.head_size) if output.dim() == 2 else output
        # ЧИТАТЕЛЬ ИДЁТ ЧЕРЕЗ ОТМЕТКУ, А НЕ ЧЕРЕЗ unbind(1). У байтового пула форма, которую видит
        # движок, и раскладка, по которой работают ядра, -- РАЗНЫЕ вещи (см. `_carve`), поэтому
        # «прочитать пул» и «сверить, чей он» здесь один и тот же вызов.
        pool = self._pool(kv_cache, "forward")
        key_cache, value_cache = pool["k"], pool["v"]
        block_size = key_cache.shape[1]
        Hkv = key_cache.shape[2]
        d = self.head_size

        # ПОДГОТОВКА ШАГА ЖИВЁТ НА МЕТАДАННЫХ, А НЕ НА СЛОЕ. Объект метаданных строится ОДИН РАЗ на
        # kv-группу на шаг и раздаётся всем её слоям (у Gemma-4 это 8 слоёв), а всё, что здесь
        # считается, зависит ТОЛЬКО от него. Пересчёт на каждом слое стоил ЗАМЕРЕННЫЕ 13-16 мс на шаг
        # декода Gemma-4 против 1.3-4.0 мс на сами ядра -- то есть обвязка была в 10 раз дороже
        # работы, ради которой существует. Кэш висит АТРИБУТОМ на самом объекте, а не в словаре по
        # id(): объект умирает вместе с шагом, поэтому просроченным кэш быть не может в принципе.
        cache = getattr(m, "_fa2_step", None)
        if cache is None:
            qsl = m.query_start_loc_cpu.tolist()
            seq_lens = (m.seq_lens_cpu.tolist() if m.seq_lens_cpu is not None
                        else m.seq_lens[:m.num_reqs].cpu().tolist())      # запасная синхронизация: только неоднородный батч без CPU-копии
            dec_rows = []
            pre_rows = []
            for i in range(m.num_reqs):
                qlen = qsl[i + 1] - qsl[i]
                if qlen <= 0:
                    continue
                (dec_rows if qlen == 1 else pre_rows).append(i)
            cache = {"qsl": qsl, "seq_lens": seq_lens, "dec": dec_rows, "pre": pre_rows}
            m._fa2_step = cache
        qsl, seq_lens = cache["qsl"], cache["seq_lens"]
        dec_rows, pre_rows = cache["dec"], cache["pre"]
        if "метаданные" not in _SAID:
            # Форматирование строки (и .tolist() со слотами!) стоит денег на КАЖДОМ слое КАЖДОГО
            # шага, поэтому оно под тем же ключом, что и печать, а не только печать под ним.
            _say_once(
                f"метаданные: num_reqs={m.num_reqs} num_actual={n} qsl={qsl[:6]} "
                f"seq_lens={seq_lens[:6]} max_q={m.max_query_len} max_s={m.max_seq_len} "
                f"bt={tuple(m.block_table.shape)} slots={m.slot_mapping[:8].tolist()}",
                key="метаданные",
            )

        # Скалярные масштабы слоя нужны ТОЛЬКО формату e4m3 и берутся РАЗ НА ВЫЗОВ, а не на строку:
        # при `--calculate-kv-scales` они меняются после первого форварда, поэтому запоминать их
        # в конструкторе нельзя, а читать на каждой строке -- незачем.
        if self._fmt == _E4M3:
            pool["kscale"] = float(getattr(layer, "_k_scale_float", 1.0) or 1.0)
            pool["vscale"] = float(getattr(layer, "_v_scale_float", 1.0) or 1.0)

        for i in pre_rows:
            _с_повтором(lambda i=i: self._prefill_row(
                query, out3, pool, m, i,
                qsl[i], qsl[i + 1], int(seq_lens[i]), block_size, Hkv, d,
            ))

        if dec_rows:
            self._decode_rows(
                query, out3, pool, m, dec_rows, qsl,
                seq_lens, block_size, Hkv, d,
            )
        return output

    def _prefill_row(
        self, query, output, pool, m, row, t0, t1, T, block_size, Hkv, d
    ) -> None:
        """One sequence, whole prefix out of the paged pool.

        The KV pool is written BEFORE attention runs, so even the first chunk reads the pool -- there
        is no dense path to shortcut here, and the gather is not overhead but the only route.

        ОКНО. Запросы куска стоят в абсолютных позициях [T-Sq, T), значит НИ ОДИН из них не смотрит
        левее `lo = T-Sq-W+1`. Сборка начинается с блока, содержащего `lo` -- не потому что так
        быстрее, а потому что левее лежит МУСОР (освобождённые/чужие страницы, см. заголовок файла).
        Именно с этого блока страницы ещё живы: движок обнуляет блоки с индексом строго меньше
        `get_num_skipped_tokens(T-Sq) // block_size`, а это и есть `lo // block_size`. Ключи,
        попавшие в сборку из-за округления до блока, отсекает маска окна ядра, поэтому округление
        НЕ портит ответ -- оно стоит лишь до block_size-1 лишних ключей в сборке.
        """
        # Причинность приезжает МЕТАДАННЫМИ (умолчание True -- обычный путь не меняется).
        # Двусторонняя маска нужна блочному черновику: его позиции обязаны видеть друг друга.
        _прич = bool(getattr(m, "causal", True))
        Sq = t1 - t0
        W = self.window
        # Окно НЕ ДЕЙСТВУЕТ, пока T <= W: у самого правого запроса p = T-1, его левая граница
        # p-W+1 <= T-W <= 0. Тогда идём прежним быстрым путём БЕЗ транспозиции -- а это подавляющее
        # большинство запросов у модели с окном 1024.
        windowed = W > 0 and T > W
        base = 0
        if windowed:
            lo = max(0, T - Sq - W + 1)
            base = (lo // block_size) * block_size
        Tg = T - base
        if (_PAGED_PREF and not windowed and self._fmt == _I8 and d == 256
                and pool["k"].shape[1] % 16 == 0):
            q_i = query[t0:t1].unsqueeze(0).contiguous()
            bt = m.block_table[row].to(torch.int32).contiguous()
            # [ДОЗОР q/K НА ПОСТРАНИЧНОМ ПУТИ -- ЗАДАЧА РАЗРЕЖЕННОСТИ, записка 20]
            # Прежний дозор стоял НИЖЕ этого раннего возврата, то есть на МЁРТВОМ пути: боевой
            # префилл идёт постранично и через него не проходит НИ РАЗУ. Пятый такой случай за
            # проект -- опознавать путь надо ЗАПУСКОМ, а не чтением. Здесь дозор стоит ДО ветвления
            # по-настоящему: этот возврат и есть боевая ветка.
            # Снимок нужен, чтобы посчитать ОФФЛАЙН долю блоков, переживающих правило FlashPrefill V2,
            # и ошибку с поправкой на среднее -- то есть решить осуществимость БЕЗ написания ядра.
            _дамп = os.environ.get("FA2SM70_DUMP_QK", "")
            if _дамп:
                _порог = int(os.environ.get("FA2SM70_DUMP_QK_T", "32768"))
                _n = getattr(FA2SM70Impl, "_qk_счёт", 0)
                if Tg >= _порог and _n < int(os.environ.get("FA2SM70_DUMP_QK_N", "4")):
                    FA2SM70Impl._qk_счёт = _n + 1
                    _kb, _vb = self._gather_buf(Hkv, Tg, d, query.device)
                    self._dec().gather_paged_kv_i8_pool_f16(
                        pool["k"], pool["v"], pool["ks"], pool["vs"], bt, int(Tg), _kb, _vb)
                    torch.cuda.synchronize()
                    _путь = f"{_дамп}.{_n}.r{int(torch.cuda.current_device())}.pt"   # РАНГ В ИМЕНИ: у TP-рангов разные головы, общий путь = гонка (урок 13.09)
                    torch.save({"q": query[t0:t1].detach().cpu(), "k": _kb.detach().cpu(),
                                "v": _vb.detach().cpu(), "scale": float(self.scale),
                                "T": int(Tg), "Sq": int(t1 - t0), "Hkv": int(Hkv), "d": int(d)},
                               _путь)
                    print(f"[fa2_sm70] снят срез q/K НА БОЕВОМ пути: q={tuple(query[t0:t1].shape)} "
                          f"k={tuple(_kb.shape)} T={Tg} -> {_путь}", flush=True)
                    # [14.09] буфер сборки -- НА СЛОЙ (self._kbuf/_vbuf): 16 слоёв x 200 МиБ на 100K = OOM на
                    # третьем слое (стенд, util 0.88). Снимок снят -- буфер отдать сразу.
                    del _kb, _vb; self._kbuf = None; self._vbuf = None; torch.cuda.empty_cache()
            # [РАЗРЕЖЕННАЯ ВЕТКА] Только при ПРИЧИННОСТИ: правило отбора считает, что запросы
            # стоят в конце и смотрят влево. Двусторонний блок черновика идёт плотным путём.
            _sc = _si = None
            # [ОТБОР -- ПРЕФИЛЛУ, НЕ ДЕКОДУ, 01.09]
            # Условие стояло только по длине КОНТЕКСТА (`Tg >= _SPARSE_T`, а порог в бою равен
            # единице), поэтому отбор шёл и на декодных вызовах, где строк запроса всего k+1.
            # Замер: общий путь на 5.9K с отбором 33.88 мс/ток, без отбора 32.09 -- отбор стоил
            # 5 % шага и был ВСЕЙ разницей между общим и скалярным путями (записка 25 §88).
            # Разделять по одному лишь `Tg` нельзя: тот же порог отключил бы отбор и ПРЕФИЛЛУ
            # на 26K, а там он даёт 20 % (замер: 16.94 с с отбором против 20.38 без).
            # Поэтому различаем по ЧИСЛУ ЗАПРОСНЫХ ТОКЕНОВ: настоящий префилл идёт плитками в
            # сотни-тысячи строк, декод -- k+1. Порог `FA2SM70_SPARSE_MINSQ` (умолчание 64).
            if _SPARSE_T and _прич and Tg >= _SPARSE_T and Sq >= _SPARSE_MINSQ and getattr(self, "_sparse_layer_ok", True):
                _ml = getattr(self, "_sparse_mass_layer", None)
                if _ml is not None:
                    os.environ["FA2SM70_SPARSE_MASS"] = str(_ml)          # масса ЭТОГО слоя
                elif _MASS_ENV0 is not None:
                    os.environ["FA2SM70_SPARSE_MASS"] = _MASS_ENV0        # вернуть общую
                else:
                    os.environ.pop("FA2SM70_SPARSE_MASS", None)
                # ПО ИНДЕКСУ, А НЕ РАСПАКОВКОЙ: старое расширение отдаёт три поля, новое --
                # четыре (разбросы блоков). Распаковка `a, b, c =` сломалась бы на любом из
                # двух, стоит их разойтись; срез по индексу совместим с обоими.
                # beta -- только НОВОМУ расширению: старое такого довода не знает.
                if _SEL_BETA[0] is None:
                    _д = getattr(self._pre().sparse_select, "__doc__", "") or ""
                    _SEL_BETA[0] = "beta" in _д
                    _SEL_GAMMA[0] = "gamma" in _д
                    if _SPARSE_GAMMA > 0 and not _SEL_GAMMA[0]:
                        _say_once("расширение БЕЗ поправки по измерениям -- gamma не применяется",
                                  key="без_гаммы")
                    if _SPARSE_BETA > 0 and not _SEL_BETA[0]:
                        _say_once("расширение БЕЗ поправки на разброс -- beta не применяется",
                                  key="без_беты")
                _kbp = _kbar_pool(pool, _SPARSE_B) if _SPARSE_POOL else None
                _dvp = _dev_pool(pool, _SPARSE_B) if _SPARSE_POOL else None
                if _kbp is not None and (_SPARSE_GAMMA <= 0 or _dvp is not None):
                    # ВЕТКА ЧЕРЕЗ ТАБЛИЦУ ПУЛА. Никакого `return` здесь быть не может --
                    # это середина _prefill_row, ниже идёт само внимание.
                    _r = self._pre().sparse_select(
                        _сплошной(query[t0:t1]), pool["k"], pool["ks"], bt, int(Tg),
                        float(self.scale), _SPARSE_ALPHA,
                        _SPARSE_B, _SPARSE_TI, _SPARSE_PROBE, _SEL_SINK, _SEL_WIN, _SEL_REC, _kbp, _SPARSE_BETA,
                        _SPARSE_GAMMA, _dvp)
                    _say_once("префилл РАЗРЕЖЕННЫЙ через ТАБЛИЦУ ПУЛА (средние НЕ "
                              "пересчитываются по всему контексту)", key="разр_пул")
                else:
                  _r = (self._pre().sparse_select(
                            _сплошной(query[t0:t1]), pool["k"], pool["ks"], bt, int(Tg),
                            float(self.scale), _SPARSE_ALPHA,
                            _SPARSE_B, _SPARSE_TI, _SPARSE_PROBE, _SEL_SINK, _SEL_WIN, _SEL_REC, None, _SPARSE_BETA,
                            _SPARSE_GAMMA)
                        if _SEL_GAMMA[0] else
                        self._pre().sparse_select(
                            _сплошной(query[t0:t1]), pool["k"], pool["ks"], bt, int(Tg),
                            float(self.scale), _SPARSE_ALPHA,
                            _SPARSE_B, _SPARSE_TI, _SPARSE_PROBE, _SEL_SINK, _SEL_WIN, _SEL_REC, None, _SPARSE_BETA)
                        if _SEL_BETA[0] else
                        self._pre().sparse_select(
                            _сплошной(query[t0:t1]), pool["k"], pool["ks"], bt, int(Tg),
                            float(self.scale), _SPARSE_ALPHA,
                            _SPARSE_B, _SPARSE_TI, _SPARSE_PROBE, 2, 8, 2))
                _sc, _si = _r[0], _r[1]
                if _KV_DUMP and int(Tg) >= 16384 and id(self) not in _KV_DUMP_DONE:
                    try:
                        _KV_DUMP_DONE.add(id(self)); _pg = int(bt[0]); _bs = int(pool["k"].shape[1]); _hk = int(pool["k"].shape[2])
                        torch.save({"k": pool["k"][_pg].cpu(), "v": pool["v"][_pg].cpu(),
                                    "ks": pool["ks"][_pg * _bs * _hk:(_pg + 1) * _bs * _hk].cpu(),
                                    "vs": pool["vs"][_pg * _bs * _hk:(_pg + 1) * _bs * _hk].cpu(),
                                    "layer": getattr(self, "layer_name", None) or str(id(self))},
                                   f"{_KV_DUMP}/kv_{len(_KV_DUMP_DONE):02d}.pt")
                    except Exception as _e:  # noqa: BLE001
                        _say_once(f"дамп KV не записан: {type(_e).__name__}", key="дамп_kv")
                # [ХВОСТ ПЛОТНО, 12.09] FA2SM70_SPARSE_TAIL_DENSE=1: у НЕПОЛНОГО чанка (последний чанк
                # запроса -- там вопрос) последняя плитка запросов получает ПОЛНЫЙ список блоков, то есть
                # считается плотно тем же ядром (alpha->0 тождественен плотному побайтово). Цена -- одна
                # плитка из ~24 на запрос (~0.1 с на 250K). Мотив: вопрос после кэша префиллится заново
                # хвостом >= MINSQ и отбирал ~15 % блоков; попал ли туда блок спрошенной иглы -- лотерея.
                if _TAIL_DENSE and int(t1 - t0) < _TAIL_DENSE_CHUNK:
                    try:
                        _nbv = min(int(_si.shape[2]), (int(Tg) + _SPARSE_B - 1) // _SPARSE_B)
                        _si[-1, :, :_nbv] = torch.arange(_nbv, dtype=_si.dtype, device=_si.device)
                        _sc[-1, :] = _nbv
                    except Exception as _e:  # noqa: BLE001
                        _say_once(f"хвост плотно не применён: {type(_e).__name__}", key="хвост_плотно")
                # [ДАМП ОТБОРА, 12.09] FA2SM70_SEL_DUMP=<каталог>: счётчики (cnt) для каждого чанка длинного
                # префилла и полный idx для чанков с позициями из FA2SM70_SEL_DUMP_POS (иглы) и последнего.
                # Прибор, а не путь: по умолчанию выключен, в бою не зовётся.
                if _SEL_DUMP and int(Tg) > 100000:
                    try:
                        _SEL_DUMP_N[0] += 1
                        _kv0 = int(Tg) - int(t1 - t0)      # первый ключ этого чанка (запросы в конце)
                        _игла = any(_kv0 <= _p < int(Tg) for _p in _SEL_DUMP_POS)
                        _o = {"cnt": _sc.cpu(), "t0": int(t0), "t1": int(t1), "Tg": int(Tg),
                              "B": int(_SPARSE_B), "TI": int(_SPARSE_TI), "kv0": _kv0}
                        if _игла or _kv0 > 200000:        # хвост префикса и все чанки запроса-вопроса
                            _o["idx"] = _si.cpu()
                        if len(_r) > 5: _o["stat"] = _r[5].cpu()   # масса {удержано, всего}
                        if len(_r) > 7 and "idx" in _o: _o["m"], _o["S"] = _r[6].cpu(), _r[7].cpu()   # сырые оценки блоков
                        torch.save(_o, f"{_SEL_DUMP}/sel_{_SEL_DUMP_N[0]:04d}.pt")
                    except Exception as _e:  # noqa: BLE001
                        _say_once(f"дамп отбора не записан: {type(_e).__name__}", key="дамп_отбора")
                # [ЗОНД КАЛИБРОВКИ beta] Полоса beta не угадывается, а СЧИТАЕТСЯ: поправка
                # правит отклонение r_J, поэтому её масштаб -- std(r_J), а сигнал -- разброс
                # счётов по блокам. Печатается ОДИН раз и только под вентилем: это лишняя
                # свёртка в пути префилла.
                if _RSTAT and len(_r) > 3:
                    _rb = _r[3].float()
                    _m, _sd = float(_rb.mean()), float(_rb.std())
                    _say_once(
                        f"РАЗБРОС БЛОКОВ: r_J сред {_m:.4f} ско {_sd:.4f} "
                        f"({100*_sd/max(_m,1e-9):.2f} % от среднего), блоков {_rb.numel()}",
                        key="rstat")
                _bump("prefill_sparse")
                _say_once(
                    f"префилл РАЗРЕЖЕННЫЙ: порог {_SPARSE_T} ток, alpha={_SPARSE_ALPHA}, "
                    f"блок {_SPARSE_B}, зондов {_SPARSE_PROBE} (записка 20)",
                    key="разреженный",
                )
            # [БЕЗ ВЫДЕЛЕНИЯ И КОПИИ ВЫХОДА -- 01.09]
            # Ядро умеет писать прямо в буфер вызывающего (довод `Out`), как это давно делает
            # скалярный путь. Прежде здесь были torch::empty на каждый слой И копия наружу.
            # Способность проверяется ПО СИГНАТУРЕ, а не предполагается: боевой слепок может
            # быть старше питона (этот класс отказа у нас уже был -- «слепок старше питона»).
            # [ДАМП АРГУМЕНТОВ -- FA2SM70_DUMP_PREF] Ставится ДО ветвления: ветка с `Out`
            # выключена рычагом с 01.09, и дамп внутри неё не сработал ни разу -- ровно тот
            # класс «зонд стоит не там, где проходят все», что уже записан про дозор.
            # Интересует ДЕКОДНАЯ зона (несколько строк на огромный контекст), а не обычный
            # префилл: первый же запрос иначе съедает единственный дамп.
            if (os.environ.get("FA2SM70_DUMP_PREF") and not _ДАМП_ПРЕФ[0]
                    and q_i.shape[1] <= 8 and int(Tg) > 60000):
                _ДАМП_ПРЕФ[0] = True
                torch.save({"q": q_i.cpu(), "k": pool["k"].cpu(), "v": pool["v"].cpu(),
                            "ks": pool["ks"].cpu(), "vs": pool["vs"].cpu(), "bt": bt.cpu(),
                            "T": int(Tg), "scale": float(self.scale), "прич": bool(_прич),
                            "selTI": int(_SPARSE_TI), "selB": int(_SPARSE_B),
                            "sc": None if _sc is None else _sc.cpu(),
                            "si": None if _si is None else _si.cpu(),
                            "kstride": int(pool["k"].stride(0))},
                           os.environ["FA2SM70_DUMP_PREF"])
                logger.warning("[ПРЕФИЛЛ] аргументы сохранены: q=%s T=%d bt=%s ks=%s si=%s",
                               tuple(q_i.shape), int(Tg), tuple(bt.shape),
                               tuple(pool["ks"].shape), None if _si is None else tuple(_si.shape))
            if _PAGED_OUT[0] is None:
                # РЫЧАГ ОТКАТА. Его не было в первой редакции -- правку без отката нельзя
                # ставить в бой, и это едва не стоило разбирательства вслепую.
                _PAGED_OUT[0] = (os.environ.get("FA2SM70_PAGED_OUT", "1") == "1") and "Out" in (
                    getattr(self._pre().attn_fwd_volta_i8_paged, "__doc__", "") or "")
                if not _PAGED_OUT[0]:
                    _say_once("расширение БЕЗ довода Out: выход копируется, как прежде",
                              key="без_out")
            if _PAGED_OUT[0]:
                _вых = output[t0:t1]
                o, _lse = self._pre().attn_fwd_volta_i8_paged(
                    q_i, pool["k"], pool["v"], pool["ks"], pool["vs"], bt, Tg,
                    float(self.scale), _прич, _sc, _si, _SPARSE_TI, _SPARSE_B, _вых)
                _bump("prefill_i8_paged")
                _say_once("префилл = paged-ядро прямо из int8-пула, БЕЗ копии выхода")
                if _SPARSE_COMP and _sc is not None and _r is not None and len(_r) > 2:
                    _vb = _komp_vbar(pool, bt, int(Tg), int(_SPARSE_B), Hkv, d)
                    _kompensaciya(_вых, _lse[0], q_i, _r[2], _vb, _sc, _si, int(Tg), int(Sq), int(q_i.shape[2]), Hkv, d,
                                  int(_SPARSE_B), int(_SPARSE_TI), float(self.scale))
                    _say_once("компенсация отброшенной массы по средним блоков ВКЛЮЧЕНА (прототип)", key="комп")
                _dump_o(self, _вых, int(Tg))
                return
            o, _lse = self._pre().attn_fwd_volta_i8_paged(
                q_i, pool["k"], pool["v"], pool["ks"], pool["vs"], bt, Tg,
                float(self.scale), _прич, _sc, _si, _SPARSE_TI, _SPARSE_B)
            _bump("prefill_i8_paged")
            _say_once("префилл = paged-ядро прямо из int8-пула (задача 163: без gather и плит)")
            output[t0:t1].copy_(o[0])
            if _SPARSE_COMP and _sc is not None and _r is not None and len(_r) > 2:
                _vb = _komp_vbar(pool, bt, int(Tg), int(_SPARSE_B), Hkv, d)
                _kompensaciya(output[t0:t1], _lse[0], q_i, _r[2], _vb, _sc, _si, int(Tg), int(Sq), int(q_i.shape[2]), Hkv, d,
                              int(_SPARSE_B), int(_SPARSE_TI), float(self.scale))
            _dump_o(self, output[t0:t1], int(Tg))
            return
        kb, vb = self._gather_buf(Hkv, Tg, d, query.device)
        # [FA2/SM70 25.08] СНЯТИЕ РЕАЛЬНЫХ q/k -- ФАЛЬСИФИКАТОР РАЗРЕЖЕННОСТИ (задача 24).
        # Ставится ДО ветвления путей: первая попытка стояла в ветке `qbshd`, а боевой префилл идёт
        # оконной веткой -- зонд не сработал ни разу. Дозор обязан стоять там, где ПРОХОДЯТ ВСЕ.
        _дамп = os.environ.get("FA2SM70_DUMP_QK", "")
        if _дамп and Tg >= 8192 and not getattr(FA2SM70Impl, "_qk_снят", False):
            FA2SM70Impl._qk_снят = True
            import threading as _th

            def _снять(_q=query[t0:t1], _kb=kb, _vb=vb, _s=float(self.scale), _p=_дамп):
                torch.save({"q": _q.detach().cpu(), "k": _kb.detach().cpu(),
                            "v": _vb.detach().cpu(), "scale": _s}, _p)
                print(f"[fa2_sm70] снят срез q/k: q={tuple(_q.shape)} k={tuple(_kb.shape)} -> {_p}",
                      flush=True)

            self._снять_после = _снять
        bt = m.block_table[row, base // block_size:].reshape(-1)
        # СБОРКА -- ЕДИНСТВЕННОЕ МЕСТО, ГДЕ БАЙТОВЫЙ ПУЛ ОТЛИЧАЕТСЯ ОТ fp16 НА ПРЕФИЛЛЕ.
        # Дальше идёт ТО ЖЕ fp16-ядро внимания, и это не лень, а необходимость: байтовое ядро
        # префилла (`attn_fwd_volta_i8`) инстанцировано под d=256 БЕЗ окна, а у Gemma-4 все сорок
        # слоёв с d=256 -- как раз оконные, глобальные же имеют d=512. То есть на боевых формах
        # байтовое ядро неприменимо ни одной, и «перенести ядро» дало бы ноль. Разворот в fp16
        # ТОЧЕН (q*s либо e4m3->fp16), поэтому формат один на всём пути, без второго квантования.
        ext = self._dec()
        # ТОЧКА ОТСЧЁТА БЕРЁТСЯ ПОСЛЕ СИНХРОНИЗАЦИИ, А НЕ ДО. Без этой строки первый же
        # `synchronize()` после сборки поглощает ВСЁ, что стояло в очереди раньше -- а между двумя
        # слоями полного внимания стоят ТРИ слоя GDN со своими MLP. Замер тогда показывает не фазу,
        # а «время от предыдущей синхронизации»: сборка вышла 921 мс на вызов против 0.23 мс начисто,
        # то есть в четыре тысячи раз -- величина, невозможная для ядра и типичная для сбитой точки
        # отсчёта. Сумма фаз при этом становится равна почти всему шагу, что само по себе признак.
        if _PHASE:
            torch.cuda.synchronize()
        _t0 = time.perf_counter() if _PHASE else 0.0
        if self._fmt == _I8:
            ext.gather_paged_kv_i8_pool_f16(
                pool["k"], pool["v"], pool["ks"], pool["vs"],
                bt.to(torch.int32), int(Tg), kb, vb)
            _bump("prefill_i8")
        elif self._fmt == _E4M3:
            ext.gather_paged_kv(
                pool["k"], pool["v"], bt.to(torch.int32), int(Tg),
                pool.get("kscale", 1.0), pool.get("vscale", 1.0), kb, vb)
            _bump("prefill_e4m3")
        else:
            ext.gather_paged_kv(
                pool["k"], pool["v"], bt.to(torch.int32), int(Tg), 1.0, 1.0, kb, vb)
        if getattr(self, "_снять_после", None) is not None:
            _ф = self._снять_после
            self._снять_после = None
            _ф()
        if _PHASE:
            torch.cuda.synchronize(); _ph("сборка", time.perf_counter() - _t0)
            _t0 = time.perf_counter()
        if windowed:
            # ОКНО ЕСТЬ ТОЛЬКО У attn_fwd_cutlass, а он берёт Q в BHSD -- транспозиция обязательна.
            # ПОПРАВКА 02.08.2026 ПО ЗАМЕРУ (reports/EV_gather_share.md): здесь стояло "доли процента
            # против самого умножения" -- занижено на порядок. Замерено на боевой геометрии
            # (H=16, Hkv=8, d=256, окно 1024): транспозиция 65.5 мкс против 1749.5 мкс внимания, то
            # есть 3.7 %, а при Sq=8192 она СРАВНИВАЕТСЯ со сборкой и обгоняет её (247.5 против 219.5).
            # Вывод "сперва верно, потом быстро" остаётся в силе, но цена названа неверно: она не
            # доли процента, а сопоставима со сборкой байтового пула. Причинность у
            # ядра выровнена по ПРАВОМУ НИЖНЕМУ углу (causal_diagonal_offset = Sk-Sq), поэтому центр
            # окна для запроса i равен i+Tg-Sq, что в абсолютных координатах ровно T-Sq+i.
            q_i = query[t0:t1].unsqueeze(0).transpose(1, 2).contiguous()   # [1, H, Sq, d]
            if self._noalibi is None:
                self._noalibi = torch.empty(0, dtype=torch.float32, device=query.device)
            o, _lse = self._pre().attn_fwd_cutlass(
                q_i, kb.unsqueeze(0), vb.unsqueeze(0), float(self.scale), _прич,
                int(W), 0.0, self._noalibi, -1,
            )
            o_bshd = o[0].transpose(0, 1)                                  # [Sq, H, d]
            if _PHASE:
                torch.cuda.synchronize(); _ph("внимание+окно", time.perf_counter() - _t0)
        else:
            # BSHD in AND out: the runner hands query as [tokens, heads, d] and wants the output the
            # same way. Passing a transpose here would make the kernel materialise a copy -- 400 MB on
            # a 32K chunk, which is what used to make big prefill chunks OOM next to the weights.
            if _SP93 and self._sp93_act and Tg > Sq:
                kb, vb, Tg = _sp93_prune(query[t0:t1], kb, vb, Tg, Sq, self.scale)
            q_i = query[t0:t1].unsqueeze(0).contiguous()
            o, _lse = self._pre().attn_fwd_qbshd(
                q_i, kb.unsqueeze(0), vb.unsqueeze(0), float(self.scale), _прич,
            )
            o_bshd = o[0]
            if _PHASE:
                torch.cuda.synchronize(); _ph("внимание", time.perf_counter() - _t0)
                _t0 = time.perf_counter()
        if _PHASE:
            # ХВОСТ: копирование выхода и всё, что после ядра. Считается ВСЕГДА, потому что триггер
            # выгрузки висел на оконной ветке -- а у боевой сети окна НЕТ, и он не срабатывал ни разу.
            pass
        if self._force_ref:
            # ФАЛЬСИФИКАТОР: выход ЯДРА заменяется torch-эталоном на тех же входах. Если ответ
            # модели не починился -- ошибка НЕ в ядре, а в обвязке (раскладка, маршрутизация, склад).
            output[t0:t1].copy_(
                self._ref_prefill(query[t0:t1], kb, vb, Tg, windowed).to(output.dtype))
        else:
            output[t0:t1].copy_(o_bshd)
        if _PHASE:
            torch.cuda.synchronize(); _ph("хвост+копия", time.perf_counter() - _t0)
        if self._check:
            self._check_prefill(query[t0:t1], kb, vb, o_bshd, row, Tg, windowed)
        _bump("prefill")
        if windowed:
            _bump("prefill_win")
        _say_once("prefill = paged gather -> attn_fwd_qbshd (fp16)"
                  + (" | окно -> attn_fwd_cutlass(window)" if W else ""))
        if _TRACE:
            logger.info("[fa2_sm70] prefill row=%d q=%d T=%d Tg=%d окно=%s", row, Sq, T, Tg, windowed)

    def _check_prefill(self, q_bshd, kb, vb, o_bshd, row, T, windowed) -> None:
        """Сверка ФАЗЫ ПРЕФИЛЛА с torch на ТЕХ ЖЕ собранных K/V.

        Отделяет ошибку сборки страниц от ошибки ядра: если здесь сходится, а ответ всё равно неверен,
        виноват декод или раскладка выхода, и искать надо там.
        """
        ref = self._ref_prefill(q_bshd, kb, vb, T, windowed)
        logger.info("[fa2_sm70] СВЕРКА префилл row=%d Sq=%d T=%d d=%d окно=%s relL2=%.3e",
                    row, q_bshd.shape[0], T, self.head_size,
                    self.window if windowed else 0, _relL2(o_bshd, ref))

    def _ref_prefill(self, q_bshd, kb, vb, T, windowed):
        """Эталон на СОБРАННЫХ K/V: [Sq,H,d] -> [Sq,H,d].

        Причинность и окно считаются в ТЕХ ЖЕ координатах, что у ядра: последний запрос куска стоит
        у ключа T-1 (выравнивание по правому нижнему углу), поэтому запрос i стоит у ключа i+T-Sq.
        """
        with torch.no_grad():
            q = q_bshd.permute(1, 0, 2).float()                      # [H, Sq, d]
            r = self.num_heads // kb.shape[0]
            k = kb.repeat_interleave(r, dim=0).float()               # [H, T, d]
            v = vb.repeat_interleave(r, dim=0).float()
            s = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            Sq = q.shape[1]
            i = torch.arange(Sq, device=q.device).view(-1, 1) + (T - Sq)
            j = torch.arange(T, device=q.device).view(1, -1)
            s = s.masked_fill(j > i, float("-inf"))
            if windowed:
                s = s.masked_fill(j < i - self.window + 1, float("-inf"))
            return torch.matmul(torch.softmax(s, -1), v).permute(1, 0, 2)

    def _deq_pages(self, pool, pages, off, hi):
        """Страницы пула -> ДЕКВАНТОВАННЫЕ fp32 [T, Hkv, d]. Эталонное чтение, не боевое.

        Это МЕТРИКА ЯДРА, а не цена формата: эталон обязан видеть ровно те значения, которые видит
        ядро, иначе сверка мерила бы квантование, а не вычисление. Цена формата меряется отдельно,
        на складе (`_check_store`).
        """
        d, hkv = pool["d"], int(pool["k"].shape[2])
        k = pool["k"][pages].reshape(-1, hkv, d)[off:hi]
        v = pool["v"][pages].reshape(-1, hkv, d)[off:hi]
        if pool["ks"] is not None:
            n = int(pool["k"].shape[1])
            si = (pages.unsqueeze(1) * n
                  + torch.arange(n, device=pages.device)).reshape(-1)
            ks = pool["ks"].view(-1, hkv)[si][off:hi].unsqueeze(-1)
            vs = pool["vs"].view(-1, hkv)[si][off:hi].unsqueeze(-1)
            return k.float() * ks, v.float() * vs
        if pool["fmt"] == _E4M3:
            return (k.view(torch.float8_e4m3fn).float() * pool.get("kscale", 1.0),
                    v.view(torch.float8_e4m3fn).float() * pool.get("vscale", 1.0))
        return k.float(), v.float()

    def _ref_decode(self, q4, pool, bt, sl, block_size):
        """Эталон декода из ПУЛА тем же путём (block_table -> страницы), с окном.

        Окно здесь ОБЯЗАТЕЛЬНО, и не ради точности сверки: страницы левее окна у скользящего слоя
        уже освобождены и могут принадлежать другому запросу, так что "эталон" без окна читал бы
        чужой KV и был бы неверен САМ.
        """
        with torch.no_grad():
            B, H, _, d = q4.shape
            Hkv = int(pool["k"].shape[2])
            r = H // Hkv
            outs = []
            for b in range(B):
                T = int(sl[b].item())
                lo = max(0, T - self.window) if self.window else 0
                b0 = lo // block_size
                pages = bt[b, b0:(T + block_size - 1) // block_size].to(torch.long)
                off = lo - b0 * block_size
                k, v = self._deq_pages(pool, pages, off, T - b0 * block_size)
                k = k.permute(1, 0, 2).repeat_interleave(r, 0).float()
                v = v.permute(1, 0, 2).repeat_interleave(r, 0).float()
                q = q4[b, :, 0, :].unsqueeze(1).float()
                sc = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, -1)
                outs.append(torch.matmul(sc, v)[:, 0, :])
            return torch.stack(outs, 0)

    def _check_decode(self, q4, pool, bt, sl, block_size, out_rows) -> None:
        """Сверка ФАЗЫ ДЕКОДА с torch, читая пул ТЕМ ЖЕ путём (block_table -> страницы)."""
        ref = self._ref_decode(q4, pool, bt, sl, block_size)
        errs = [_relL2(out_rows[b], ref[b]) for b in range(q4.shape[0])]
        logger.info("[fa2_sm70] СВЕРКА декод B=%d d=%d окно=%d T=%s relL2=%s",
                    q4.shape[0], self.head_size, self.window,
                    [int(x) for x in sl.tolist()], [f"{e:.3e}" for e in errs])

    def _virt_batch(self, B: int, q: int, bt_src, sl_src, m_obj=None, причинно: bool = True):
        """ВИРТУАЛЬНЫЙ БАТЧ ДЛЯ СПЕКУЛЯЦИИ: (запрос, позиция) -> одна строка батча.

        Ядро расщепления адресует запрос как `Q + (b*H + h)*d`, то есть ОДИН вектор на (b, h).
        Поэтому k+1 позиций не требуют нового тела: они разворачиваются в батч B*q, и ядро
        остаётся тем же -- вместе со своим расщеплением по ключам. Это и есть смена единицы:
        не «Sq внутри блока», а «ещё одно измерение батча».

        Причинность внутри окна выражается ОДНИМ числом на строку: позиция j (0..q-1) видит
        префикс длиной seq_len - (q-1-j). KV окна уже лежит в пуле (запись идёт ДО внимания),
        поэтому маска не нужна -- достаточно длины.

        Ни одного выделения: буферы держатся по максимуму, заполнение идёт `copy_` из ВИДОВ
        (unsqueeze/expand не выделяют). В графе это захватывается как обычные копии.
        """
        # ОДИН РАЗ НА ШАГ, А НЕ НА КАЖДЫЙ СЛОЙ. Метаданные у всех 16 слоёв шага -- ОДИН объект,
        # значит и виртуальный батч у них общий; повторное построение было чистой параллельной
        # работой (32 запуска ядер копирования на шаг вместо двух). Сравнение по `is`: другой шаг
        # приносит другой объект метаданных.
        if (self._virt_m is m_obj and self._virt_key == (B, q, причинно)
                and self._vbt is not None and self._vbt.shape[1] == int(bt_src.shape[1])):
            need = B * q
            return self._vbt[:need], self._vsl[:need]
        mb = int(bt_src.shape[1])
        need = B * q
        if (self._vbt is None or self._vbt.shape[0] < need or self._vbt.shape[1] != mb
                or self._vbt.device != bt_src.device):
            _отставить_мод(getattr(self, '_vbt', None))
            self._vbt = torch.empty((need, mb), dtype=torch.int32, device=bt_src.device)
            self._vsl = torch.empty(need, dtype=sl_src.dtype, device=sl_src.device)
            self._voff_q = -1
        if self._voff_q != q:
            self._voff = torch.arange(q - 1, -1, -1, dtype=sl_src.dtype, device=sl_src.device)
            self._voff_q = q
        self._vbt[:need].view(B, q, mb).copy_(bt_src[:B].unsqueeze(1).expand(B, q, mb))
        # [ПРИЧИННОСТЬ -- ЭТО И ЕСТЬ ВЫЧИТАНИЕ СМЕЩЕНИЯ]
        # У цепи позиция j видит префикс `seq_len - (q-1-j)`. У ДВУНАПРАВЛЕННОГО блока (черновик
        # DFlash2, `is_causal: false`) все позиции блока видят ВЕСЬ блок -- значит длина у всех
        # строк ОДНА. Разница между причинным и непричинным блоком в этом пути -- ровно наличие
        # вычитания, и ничего больше.
        if причинно:
            self._vsl[:need].view(B, q).copy_(sl_src[:B].unsqueeze(1) - self._voff.unsqueeze(0))
        else:
            self._vsl[:need].view(B, q).copy_(sl_src[:B].unsqueeze(1).expand(B, q))
        self._virt_m, self._virt_key = m_obj, (B, q, причинно)
        return self._vbt[:need], self._vsl[:need]

    def _decode_uniform(self, query, output, kv_cache, m, B, q: int = 1):
        """Ш1: однородный декод -- ни одного выделения, ни одного чтения хостом.

        Возвращает выходной буфер, если путь применим, иначе None (тогда идёт прежний путь).
        Отказ ВСЕГДА молчаливый и полный: половинчатое исполнение здесь дало бы связный, но неверный
        текст -- тот самый класс, который этот файл уже ловил шесть проходов.

        q > 1 -- спекулятивный декод: те же B запросов, но по k+1 позиций в каждом.
        """
        pool = self._pool(kv_cache, "decode_uniform")
        d, H = self.head_size, self.num_heads
        Hkv = pool["k"].shape[2]
        block_size = int(pool["k"].shape[1])
        out3 = output.view(-1, H, d) if output.dim() == 2 else output
        # ПРЯМАЯ ЗАПИСЬ ОБЯЗАНА ПРОВЕРИТЬ РАСКЛАДКУ САМА (ядро адресует base + b*stride0 + h*d).
        if not (out3.dtype == torch.float16 and out3.stride(1) == d and out3.stride(2) == 1):
            return None
        bt = m.block_table[:B]
        if bt.dtype != torch.int32:            # .to() здесь было бы выделением на каждый слой
            return None
        N = B * q
        # ВИД: строки декода идут подряд по контракту; при q>1 подряд идут (запрос, позиция).
        q4 = query[:N].view(N, H, 1, d)
        _прич = bool(getattr(m, "causal", True))
        # [ВИРТУАЛЬНЫЙ БАТЧ БЕЗ КОПИЙ] Если расширение умеет qGrp, таблицу блоков и длины
        # подаём ИСХОДНЫЕ: ядро само берёт запрос как r/q и длину sl[r/q]-(q-1-r%q). Раньше
        # хозяин раскладывал их двумя копиями (два запуска на шаг) -- это и есть «параллельная
        # работа», которую надо было пересчитать, а не ускорять.
        _qg = 1
        if _DEC_QGRP[0] is None:
            _DEC_QGRP[0] = "qGrp" in (
                getattr(self._dec().flash_decode_defer_mqa_paged_into, "__doc__", "") or "")
        if q > 1:
            if _DEC_QGRP[0] and _QGRP_ON:
                sl = m.seq_lens[:B]; _qg = q
                _say_once("виртуальный батч БЕЗ КОПИЙ (ядро считает запрос и длину само)",
                          key="qgrp")
            else:
                bt, sl = self._virt_batch(B, q, bt, m.seq_lens, m, причинно=_прич)
        else:
            sl = m.seq_lens[:B]
        B = N                                   # дальше всё считается по строкам виртуального батча
        kv_cap = int(bt.shape[1]) * block_size
        # [ГРАНИЦА ТАБЛИЦЫ БЛОКОВ -- ПРИБОР И ЗАМОК, 31.08]
        # Ядро адресует KV по таблице блоков строки; если нужная длина выходит за таблицу,
        # оно читает ЧУЖУЮ память -- ответ остаётся связным, но неверным. Это ровно класс
        # «обрыв на границе блока при спекуляции» из отчёта vllm-granica-ne-ushla.
        # Проверка стоит одно сравнение питоновских чисел: `max_seq_len` уже посчитан движком.
        _ц2 = getattr(m, "_fa2_ints", None)          # уже посчитано в forward -- см. разбор там
        _мсл = _ц2[2] if _ц2 is not None else int(getattr(m, "max_seq_len", 0) or 0)
        if _мсл > kv_cap:
            _к = _ГРАНИЦА
            _к[0] += 1
            if _к[0] <= 8 or (_к[0] & (_к[0] - 1)) == 0:
                import sys as _s
                print(f"[fa2_sm70 ГРАНИЦА] длина {_мсл} ВЫШЛА за таблицу блоков "
                      f"{int(bt.shape[1])}x{block_size}={kv_cap} (q={q}, случай {_к[0]}) "
                      f"-- путь отдан общему", file=_s.stderr, flush=True)
            return None                       # общий путь строит таблицу сам
        # [gf И ns -- ОДИН РАЗ НА ГРУППУ, 31.08] Оба зависят только от (d, H, Hkv, B, kv_cap),
        # то есть внутри одной группы KV одинаковы для ВСЕХ её слоёв, а считались на каждом:
        # `_pick_gf` -- цикл по пяти вариантам, `_auto_splits` -- ещё и `os.environ.get` с
        # разбором строки. Память кладём НА ОБЪЕКТ метаданных (он строится заново каждый шаг),
        # а ключом берём то, от чего они зависят: при смене формы шага память промахнётся и
        # пересчитает. `lru_cache` здесь пробовался и отвергнут -- хеширование кортежа стоит
        # примерно столько же, сколько сами функции.
        _клнс = (d, H, Hkv, B, kv_cap)
        _пнс = getattr(m, "_fa2_ns", None)
        if _пнс is not None and _пнс[0] == _клнс:
            gf, ns = _пнс[1], _пнс[2]
        else:
            gf = self.fa2._pick_gf(d, max(1, H // Hkv))
            ns = min(self.fa2._auto_splits(B, H, kv_cap, gf=gf), self._max_splits)
            try:
                m._fa2_ns = (_клнс, gf, ns)
            except Exception:
                pass
        # БУФЕР ДЕРЖИТСЯ ПО МАКСИМУМУ, А НЕ ПО ТЕКУЩЕЙ ФОРМЕ. При спекуляции B прыгает между
        # 1 (черновик) и k+1 (проверка), и сравнение `.shape != (B, Hkv)` пересоздавало тензор
        # НА КАЖДОМ СЛОЕ КАЖДОГО ШАГА. Указатель должен быть ещё и постоянным для графа.
        if self._kmax is None or self._kmax.shape[0] < B or self._kmax.shape[1] != Hkv:
            _отставить_мод(getattr(self, '_kmax', None))
            self._kmax = torch.zeros((max(B, 8), Hkv), dtype=torch.float32, device=query.device)
        A, Bw = self._workspace(B, H, ns, d, query.device)
        # [ОКНО У БЫСТРОГО ПУТИ -- ЛЕВАЯ ГРАНИЦА, И ТОЛЬКО ОНА]
        # Ядро принимает `kv_start` на строку (это и есть окно, точное ДО ТОКЕНА). Прежде путь
        # ОТКАЗЫВАЛСЯ при любом окне, и слои с окном уходили на префилльный путь с материализацией
        # KV в промежуточный буфер. Для черновика DFlash2 (окно 2048, пять слоёв) это давало
        # 1.16 мс из 7.4 мс предлагателя при поле ЧТЕНИЯ около 0.13 -- девятикратный разрыв.
        # Буфер постоянный и только растёт: указатель обязан быть стабильным для графа.
        kvs = None
        if self.window and self._winmode != "full":
            if getattr(self, "_vkvs", None) is None or self._vkvs.numel() < B:
                self._отставить(getattr(self, '_vkvs', None)) if _ДЕРЖАТЬ_БУФЕРЫ else None
            _отставить_мод(getattr(self, '_vkvs', None))
            self._vkvs = torch.empty(max(B, 16), dtype=torch.int32, device=query.device)
            torch.sub(sl, int(self.window), out=self._vkvs[:B])
            self._vkvs[:B].clamp_(min=0)
            kvs = self._vkvs[:B]
        fp8 = self._fmt == _E4M3
        # [ДЕРЕВО] Видимость через (ctxlen, tailmask) вместо длины на строку. Пока выражается
        # ЦЕПЬ -- выход обязан не измениться; это гейт плумбинга до постройки самого дерева.
        _ctx = _tm = None
        # ТОЛЬКО ПРИ ПРИЧИННОСТИ. `_decode_uniform` обслуживает и ДВУСТОРОННИЙ блок черновика
        # (`causal=False`), где строки видят ДРУГ ДРУГА -- цепная маска их бы порезала.
        # Поймано A/B на ОДНОМ стенде: из двух текстов один расходился, другой нет.
        # СВЕРКА ФОРМЫ, КОТОРОЙ НЕ ХВАТАЛО. Раскладка дерева осмысленна ТОЛЬКО при
        # q == 2W+1 (якорь + W строк ветви A + W строк ветви B). Без этой сверки маска
        # дерева накладывалась на ЛЮБОЙ причинный однородный проход с q>1 -- в том
        # числе туда, где строки идут цепью, и строка 1 переставала видеть строку 0.
        # Поймано пошаговой сверкой токенов: расхождение с базой было на ШАГЕ 0.
        if _TREE_W > 0 and q == 2 * _TREE_W + 1 and _прич:
            _ctx, _tm = _tree_branch(B // q, q, _TREE_W, m.seq_lens, query.device)
            _bump("tree_branch")
        elif _TREE_MASK and q > 1 and _прич:
            _ctx, _tm = _tree_chain(B // q, q, m.seq_lens, query.device)
            _bump("tree_mask")
        # [РАЗРЕЖЕННЫЙ ДЕКОД] Отбор ведётся С ДЛИНОЙ НА СТРОКУ: строки -- виртуальный батч
        # спекуляции, у каждой своя seq_len и своя причинность. Средние ключи берутся из
        # таблицы пула, обновлённой при записи KV.
        _sc = _si = None
        if _SPARSE_DEC and self._fmt == _I8 and _SPARSE_DEC_ALPHA > 0:
            _kb = _kbar_pool(pool, _SPARSE_B)
            # ПОРОГ БЕРЁТСЯ ИЗ ХОЗЯЙСКОГО ЧИСЛА, а не из `int(sl.max())`: последнее -- это
            # синхронизация GPU->CPU в горячем пути (запрещена) и разрыв CUDA-графа.
            _mx = int(getattr(m, "max_seq_len", 0) or 0)
            if _kb is not None and _mx > _SPARSE_B * 16:
                _nb2 = (kv_cap + _SPARSE_B - 1) // _SPARSE_B
                _sc, _si, _mb, _sb = _sel_bufs(B, Hkv, _nb2, query.device)
                if _SPARSE_DEC_TOPF > 0:
                    # [ПО ГОЛОВАМ, 14.09] отбор по строке нормирует каждую q-голову отдельно: буферы m/s шириной g*NB
                    _g = query.shape[1] // Hkv
                    _mb, _sb = _sel_bufs_h(B, Hkv, _nb2 * _g, query.device)
                _dp = _dev_pool(pool, _SPARSE_B)
                # [ГРУППА СТРОК] Позиции спекуляции одного запроса лежат ПОДРЯД (q строк на
                # запрос), и счёт отбора сводится по ним всем -- то самое сглаживание, за счёт
                # которого работает префилльное правило. Вентиль FA2SM70_SPARSE_DEC_GRP:
                # 0 -- взять q автоматически, 1 -- прежнее поведение (строка сама по себе).
                _grp = q if _SPARSE_DEC_GRP == 0 else _SPARSE_DEC_GRP
                if _grp < 1 or (B % _grp):
                    _grp = 1
                if _SEL_GRP[0] is None:
                    _SEL_GRP[0] = "grp" in (
                        getattr(self._pre().sparse_select_rows, "__doc__", "") or "")
                _хв = (_dp, _SPARSE_GAMMA, _grp) if (_dp is not None and _SEL_GRP[0]) \
                      else ((_dp, _SPARSE_GAMMA) if _dp is not None else ())
                if _SEL_TOPF[0] is None:
                    _SEL_TOPF[0] = "topf" in (
                        getattr(self._pre().sparse_select_rows, "__doc__", "") or "")   # проба способности по сигнатуре
                    if _SPARSE_DEC_TOPF > 0 and not _SEL_TOPF[0]:
                        _say_once("FA2SM70_SPARSE_DEC_TOPF задан, но расширение БЕЗ topf -- иду прежним правилом alpha",
                                  key="декод_topf_нет")
                _kw = {"topf": _SPARSE_DEC_TOPF} if (_SPARSE_DEC_TOPF > 0 and _SEL_TOPF[0] and len(_хв) == 3) else {}
                # [ДЕФЕКТ 14.09] при виртуальном батче без копий длины/таблица идут ПО ЗАПРОСУ -- отбор обязан знать qGrp
                if _SEL_QGRP[0] is None:
                    _SEL_QGRP[0] = "qGrp" in (getattr(self._pre().sparse_select_rows, "__doc__", "") or "")
                if _qg > 1:
                    if _SEL_QGRP[0] and len(_хв) == 3:
                        _kw["qGrp"] = int(_qg); _kw["qCausal"] = 1 if _прич else 0
                    else:
                        _say_once("отбор декода: расширение БЕЗ qGrp при виртуальном батче -- отбор ВЫКЛЮЧЕН (иначе чужие длины)",
                                  key="декод_qgrp_нет")
                        _sc = _si = None
                if _SPARSE_DEC_MINBLK > 0 and _kw:
                    if _SEL_MINBLK[0] is None:
                        _SEL_MINBLK[0] = "minblk" in (
                            getattr(self._pre().sparse_select_rows, "__doc__", "") or "")
                        if not _SEL_MINBLK[0]:
                            _say_once("FA2SM70_SPARSE_DEC_MINBLK задан, но расширение БЕЗ minblk "
                                      "(слепок старше питона) -- пол по блокам НЕ применяется",
                                      key="без_minblk")
                    if _SEL_MINBLK[0]:
                        _kw["minblk"] = int(_SPARSE_DEC_MINBLK)
                if _qg > 1 and _sc is None:
                    pass
                else:
                  self._pre().sparse_select_rows(
                    query[:B], _kb, bt, sl.to(torch.int32), kvs,
                    float(self.scale), _SPARSE_DEC_ALPHA, _SPARSE_B,
                    _sc, _si, _mb, _sb, int(block_size), _SEL_SINK_D, _SEL_WIN_D, _SEL_REC_D,
                    *_хв, **_kw)
                  _bump("decode_sparse")
                # ЗОНДА ДОЛИ ЗДЕСЬ БЫТЬ НЕ МОЖЕТ, И Я ЭТО УЖЕ ЗНАЛ.
                # 28.08 поставил сюда `float(_sc.mean())` -- и подъём умер с
                # `cudaErrorStreamCaptureUnsupported`: чтение тензора на хост это
                # синхронизация GPU->CPU, а декод идёт ПОД ЗАХВАТОМ ГРАФА. Тот же класс,
                # что снятый отсюда `int(sl.max())`. Долю отбора мерить только ВНЕ графа
                # (отдельным прогоном по ядру), а порог свипать по мс/ток.
                _say_once(f"декод РАЗРЕЖЕННЫЙ: alpha={_SPARSE_DEC_ALPHA}, блок {_SPARSE_B}, группа {_grp}"
                          + (f", ПО СТРОКЕ верхние {_SPARSE_DEC_TOPF:.2f} блоков" if _kw else ""),
                          key="декод_разреж")
        # ПИТОН НЕ ПРЕДПОЛАГАЕТ РАСШИРЕНИЕ НОВЕЕ СЕБЯ.
        # ОТКАЗ, КОТОРЫЙ ЭТО ЛЕЧИТ (27.08): питон живёт в ОБЩЕМ венве, расширение -- в БОЕВОМ
        # СЛЕПКЕ. Доводы отбора для декода добавлены в питон ПОСЛЕ последнего перевода слепка,
        # и на первом же шаге декода pybind отказал ("incompatible function arguments") ->
        # воркер падал при ПОДЪЁМЕ. Мина ждала ЛЮБОГО перезапуска боевого, а не нашего опыта:
        # работающий сервер держал старый питон в памяти и об этом не знал.
        # Проба -- ОДНОКРАТНАЯ (список, а не global: горячий путь, лишний поиск имени не нужен).
        _dfn = self._dec().flash_decode_defer_mqa_paged_into
        if _DEC_SEL[0] is None:
            _DEC_SEL[0] = "selCnt" in (getattr(_dfn, "__doc__", "") or "")
            if not _DEC_SEL[0]:
                _say_once("расширение БЕЗ доводов отбора (слепок старше питона): разреженный "
                          "декод недоступен, ПЛОТНЫЙ путь цел", key="дек_без_отбора")
        # [СРЕЗЫ И СЛОВАРИ -- ОДИН РАЗ, А НЕ НА КАЖДЫЙ СЛОЙ, 31.08]
        # Ниже идёт вызов ядра с двумя десятками доводов, и он исполняется НА КАЖДЫЙ слой
        # внимания каждого шага. Часть доводов строилась заново каждый раз: `self._kmax[:B]` и
        # `out3[:B]` создают вид (объект), `pool.get(...)` -- два словарных обращения, `float()`
        # и `int()` -- по объекту на довод. Работы в этом нет, это накладные питона в пути
        # (0.44 % снимков профиля на строке вызова). Считаем их один раз на вызов функции.
        _kmaxB = self._kmax[:B]
        _outB = out3[:B]
        _ost = int(out3.stride(0))
        _scl = float(self.scale)
        _nsi, _kvci = int(ns), int(kv_cap)
        _ksc = pool.get("kscale", 1.0) if fp8 else 1.0
        _vsc = pool.get("vscale", 1.0) if fp8 else 1.0
        # [ХИМЕРНОЕ ЯДРО -- FA2SM70_HMMA=1] Плитка K/V в разделяемой памяти обслуживает все
        # строки спекуляции сразу, обе GEMM идут на тензорных ядрах (записка 25 §99).
        # ВЫКЛЮЧЕНО ПО УМОЛЧАНИЮ: в микробенче ядро даёт x2.36 и точность ЛУЧШЕ старого
        # (relL2 2.9e-04 против 7.5e-03), офлайн совпадает со старым на СНЯТЫХ С БОЯ аргументах
        # (relL2 1.4e-04, compute-sanitizer чист), но в сервере даёт неверный текст. Ветка
        # проверена подстановкой: тот же путь со СТАРЫМ ядром отвечает 391, значит аргументы и
        # раскладка выхода верны, а расхождение -- в самом ядре под боевым окружением. Пока не
        # найдено -- рычаг остаётся выключенным (см. §99a).
        if _HMMA[0]:
            _пр = {"fp8": fp8, "d": d, "si": _si is not None, "tm": _tm is not None,
                   "kvs": kvs is not None, "ctx": _ctx is not None,
                   "force_ref": self._force_ref, "check": self._check,
                   "bs_pow2": (block_size & (block_size - 1)) == 0,
                   "qgGF": _qg * (H // Hkv), "B%qg": B % max(1, _qg)}
            _say_once(f"ХИМЕРА предусловия: {_пр}", key=f"хим_пр_{tuple(sorted(_пр.items()))}")
        if _HMMA[0]:
            if d != 256: _bump("hmma_skip_d128")
            elif kvs is not None: _bump("hmma_skip_kvs")
            elif _ctx is not None: _bump("hmma_skip_ctx")
            elif (_si is not None and not _HMMA_SEL[0]) or _tm is not None: _bump("hmma_skip_si")
            elif B % max(1, _qg) or _qg * (H // Hkv) > 32: _bump("hmma_skip_qg")
            elif fp8 or self._force_ref or self._check: _bump("hmma_skip_prochee")
            _пр = (fp8, d, _si is not None, _tm is not None, kvs is not None, _ctx is not None,
                   (block_size & (block_size - 1)) == 0, _qg * (H // Hkv), B % max(1, _qg))
            _say_once(f"ХИМЕРА условия: fp8={_пр[0]} d={_пр[1]} si={_пр[2]} tm={_пр[3]} "
                      f"kvs={_пр[4]} ctx={_пр[5]} bs2={_пр[6]} qgGF={_пр[7]} B%qg={_пр[8]}",
                      key=f"хим_у_{_пр}")
        if _HMMA_SEL[0] is None:
            _HMMA_SEL[0] = "selIdx" in (getattr(getattr(self._dec(), "flash_decode_hmma_paged", None), "__doc__", "") or "")
        if _HMMA[0] and not fp8 and d == 256 and (_si is None or _HMMA_SEL[0]) and _tm is None and kvs is None \
                and _ctx is None and not self._force_ref and not self._check \
                and (block_size & (block_size - 1)) == 0 and _qg * (H // Hkv) <= 32 \
                and B % max(1, _qg) == 0:
            _hf = getattr(self._dec(), "flash_decode_hmma_paged", None)
            if _hf is None:
                _bump("hmma_net_fn")
            if _hf is not None:
                Bw.zero_()
                if _HMMA[0] == 12:
                    _outB.zero_(); A.zero_()
                _hkw = {"selCnt": _sc, "selIdx": _si, "selBlk": _SPARSE_B} if _si is not None else {}
                _hf(q4, pool["k"], pool["v"], bt, block_size, _scl, _nsi, sl,
                    A, Bw, _outB, _ost, pool["ks"], pool["vs"], _qg, 1 if _прич else 0, **_hkw)
                if _HMMA[0] == 12:
                    _say_once(f"ХИМЕРА выход: |out|={float(_outB.float().norm()):.4f} "
                              f"|A|={float(A.norm()):.4f} |B|={float(Bw.norm()):.4f} "
                              f"конечных={int(torch.isfinite(_outB.float()).sum())}/{_outB.numel()}",
                              key="химера_выход")
                _bump("decode_hmma")
                _bump("decode_uniform")
                _say_once("декод = ХИМЕРНОЕ ядро (плитка в разделяемой + тензорные ядра)"
                          + (" СО СПИСКОМ БЛОКОВ ОТБОРА" if _si is not None else ""), key="дек_химера")
                return output
        if _DEC_SEL[0] and _DEC_QGRP[0]:
            _dfn(q4, pool["k"], pool["v"], _kmaxB, bt, block_size, _scl,
                 _nsi, _kvci, sl, fp8, _ksc, _vsc,
                 A, Bw, _outB, _ost,
                 pool["ks"], pool["vs"], kvs, _ctx, _tm, _sc, _si, _SPARSE_B,
                 _qg, 1 if _прич else 0)
        elif _DEC_SEL[0]:
            _dfn(q4, pool["k"], pool["v"], _kmaxB, bt, block_size, _scl,
                 _nsi, _kvci, sl, fp8, _ksc, _vsc,
                 A, Bw, _outB, _ost,
                 pool["ks"], pool["vs"], kvs, _ctx, _tm, _sc, _si, _SPARSE_B)
        else:
            _dfn(q4, pool["k"], pool["v"], _kmaxB, bt, block_size, _scl,
                 _nsi, _kvci, sl, fp8, _ksc, _vsc,
                 A, Bw, _outB, _ost,
                 pool["ks"], pool["vs"], kvs, _ctx, _tm)
        _bump("decode_uniform")
        _say_once("Ш1: однородный декод БЕЗ выделений и синхронизаций (путь под CUDA-граф)")
        return output

    def _decode_rows(
        self, query, output, pool, m, rows, qsl, seq_lens, block_size, Hkv, d
    ) -> None:
        # ВСЁ, ЧТО ЗАВИСИТ ТОЛЬКО ОТ ШАГА, СЧИТАЕТСЯ ОДИН РАЗ НА ГРУППУ (см. forward): idx, срез
        # страничной таблицы и длины одинаковы для всех слоёв группы. Единственное, что зависит от
        # СЛОЯ, -- левая граница окна, и она кэшируется по величине окна.
        step = m._fa2_step
        dc = step.get("dec_t")
        if dc is None:
            idx = torch.tensor([qsl[i] for i in rows], device=query.device, dtype=torch.long)
            bt = m.block_table[torch.tensor(rows, device=m.block_table.device)].to(torch.int32)
            sl = torch.tensor([int(seq_lens[i]) for i in rows],
                              device=query.device, dtype=torch.int32)
            # СПЛОШНЫЕ СТРОКИ -- ЧАСТЫЙ СЛУЧАЙ, А НЕ РЕДКИЙ: в чистом шаге декода строки выхода идут
            # подряд, и тогда ядро пишет ПРЯМО в выходной буфер -- ни временного тензора, ни
            # рассеивающего index_copy_ на каждом слое.
            want = [qsl[i] for i in rows]
            contig = want == list(range(want[0], want[0] + len(want)))
            dc = {"idx": idx, "bt": bt, "sl": sl, "row0": want[0] if contig else -1}
            step["dec_t"] = dc
        idx, bt, sl, row0 = dc["idx"], dc["bt"], dc["sl"], dc["row0"]
        B = len(rows)
        H = self.num_heads
        q4 = query.index_select(0, idx).view(B, H, 1, d).contiguous()
        kv_cap = int(bt.shape[1]) * int(block_size)
        # ЛЕВАЯ ГРАНИЦА КЛЮЧЕЙ. Запрос декода стоит в позиции T-1, окно = [T-W, T-1], то есть окно
        # выражается ОДНИМ числом на последовательность -- началом. Считаем на хосте из уже
        # имеющегося seq_lens_cpu (никакой синхронизации), кладём в постоянный буфер.
        kv_start = None
        if self.window and self._winmode != "full":
            key = (self.window, self._winmode)
            got = step.get(key)
            if got is None:
                st = [max(0, int(seq_lens[i]) - self.window) for i in rows]
                if self._winmode == "block":
                    # Путь "б" из задания: та же граница, округлённая ВНИЗ до блока -- ровно то, что
                    # даёт подмена строки block_table без правки ядра. Оставлен ВЕНТИЛЕМ, чтобы цена
                    # округления была ЗАМЕРЕНА, а не оценена.
                    st = [(s // block_size) * block_size for s in st]
                got = (torch.tensor(st, dtype=torch.int32, device=query.device), max(st) > 0)
                step[key] = got
            kv_start = got[0]
            # Счётчик бьётся НА КАЖДЫЙ ВЫЗОВ СЛОЯ, а не на построение кэша: он доказывает, что окно
            # реально резало, и делать его зависимым от попадания в кэш значило бы занизить в 8 раз.
            # Признак "резало" лежит в кэше рядом с тензором -- читать его с карты нельзя, это синк.
            if got[1]:
                _bump("decode_win")
        gf = self.fa2._pick_gf(d, max(1, H // Hkv))
        ns = min(
            self.fa2._auto_splits(B, H, kv_cap, gf=gf),
            self._max_splits,
        )
        # БУФЕР ДЕРЖИТСЯ ПО МАКСИМУМУ, А НЕ ПО ТЕКУЩЕЙ ФОРМЕ. При спекуляции B прыгает между
        # 1 (черновик) и k+1 (проверка), и сравнение `.shape != (B, Hkv)` пересоздавало тензор
        # НА КАЖДОМ СЛОЕ КАЖДОГО ШАГА. Указатель должен быть ещё и постоянным для графа.
        if self._kmax is None or self._kmax.shape[0] < B or self._kmax.shape[1] != Hkv:
            # Vestigial in the signature: the kernel moved to an online maximum. Kept as a stable
            # zero tensor so a captured graph would bake a constant pointer.
            _отставить_мод(getattr(self, '_kmax', None))
            self._kmax = torch.zeros((max(B, 8), Hkv), dtype=torch.float32, device=query.device)
        A, Bw = self._workspace(B, H, ns, d, query.device)
        # Прямая запись законна только если выходной буфер РОВНО того типа, что пишет ядро, и строки
        # идут подряд; фальсификаторы её выключают, потому что они подменяют/читают out_rows.
        # ЯДРО АДРЕСУЕТ ВЫХОД КАК `base + b*o_row_stride + h*d`, то есть ПРЕДПОЛАГАЕТ, что оси головы
        # и канала уплотнены. Прежний путь (свой тензор + index_copy_) этого не предполагал, поэтому
        # прямая запись обязана проверить раскладку САМА, а не унаследовать чужую удачу: буфер,
        # у которого stride(1) != d, дал бы связный, но неверный текст -- ровно тот класс ошибки,
        # который эта сессия уже ловила шесть проходов.
        direct = (row0 >= 0 and output.dtype == torch.float16
                  and output.stride(1) == d and output.stride(2) == 1
                  and not (self._force_ref or self._check))
        if direct:
            out_rows = output[row0:row0 + B]
            o_stride = int(output.stride(0))
        else:
            out_rows = torch.empty((B, H, d), dtype=torch.float16, device=query.device)
            o_stride = int(out_rows.view(B, H, -1).stride(0))
        # ФОРМАТ ПУЛА ВЫРАЖАЕТСЯ ЗДЕСЬ ДВУМЯ РАЗНЫМИ АРГУМЕНТАМИ, И ЯДРО ОТВЕРГАЕТ ИХ СОЧЕТАНИЕ:
        # int8 опознаётся по НАЛИЧИЮ таблиц масштабов, e4m3 -- по флагу fp8 со скалярами; передать
        # оба сразу = TORCH_CHECK. То есть «два формата одновременно» невозможно физически, а не по
        # договорённости, -- ровно то свойство, которого не хватало прежней связке.
        fp8 = self._fmt == _E4M3
        # [ХИМЕРНОЕ ЯДРО -- второй путь] Однородный путь работает только до
        # FA2SM70_UNIFORM_MAXLEN (65000); выше идёт ЭТОТ путь, то есть ровно та длина, ради
        # которой ядро и писалось. Ветка только в _decode_uniform давала ноль вызовов на 130K --
        # поймано логом предусловий, где печатался лишь черновик (d=128).
        if _HMMA[0]:
            _say_once(f"ХИМЕРА(rows) предусловия: fp8={fp8} d={d} kv_start={kv_start is not None} "
                      f"force_ref={self._force_ref} check={self._check} bs={int(block_size)} "
                      f"H/Hkv={H}/{Hkv} B={B} ns={int(ns)}",
                      key=f"хим_rows_{fp8}_{d}_{kv_start is not None}_{int(block_size)}_{H}_{Hkv}")
        if _HMMA[0] and not fp8 and d == 256 and kv_start is None \
                and not self._force_ref and not self._check \
                and (int(block_size) & (int(block_size) - 1)) == 0 and (H // Hkv) <= 32:
            _hf = getattr(self._dec(), "flash_decode_hmma_paged", None)
            if _hf is None:
                _bump("hmma_net_fn")
            if _hf is not None:
                Bw.zero_()
                _hf(q4, pool["k"], pool["v"], bt, int(block_size), float(self.scale), int(ns),
                    sl, A, Bw, out_rows, o_stride, pool["ks"], pool["vs"], 1, 1)
                _bump("decode_hmma")
                if self._fmt == _I8:
                    _bump("decode_i8")
                if not direct:
                    output.index_copy_(0, idx, out_rows.to(output.dtype))
                _bump("decode")
                _say_once("декод(rows) = ХИМЕРНОЕ ядро", key="дек_химера_rows")
                return
        self._dec().flash_decode_defer_mqa_paged_into(
            q4, pool["k"], pool["v"], self._kmax, bt, int(block_size), float(self.scale),
            int(ns), int(kv_cap), sl, fp8,
            pool.get("kscale", 1.0) if fp8 else 1.0,
            pool.get("vscale", 1.0) if fp8 else 1.0,
            A, Bw, out_rows, o_stride,
            pool["ks"], pool["vs"], kv_start,
        )
        if self._fmt == _I8:
            _bump("decode_i8")
        elif fp8:
            _bump("decode_e4m3")
        if self._force_ref:
            out_rows = self._ref_decode(q4, pool, bt, sl, block_size).half()
        if not direct:
            output.index_copy_(0, idx, out_rows.to(output.dtype))
        if self._check:
            self._check_decode(q4, pool, bt, sl, block_size, out_rows)
        _bump("decode")
        _say_once("decode = paged split-KV deferred (flash_decode_defer_mqa_paged_into)"
                  + (f" | окно={self.window} режим={self._winmode}" if self.window else ""))
        if _TRACE:
            logger.info("[fa2_sm70] decode B=%d ns=%d kv_cap=%d", B, ns, kv_cap)


def _relL2(a, b):
    a, b = a.float(), b.float()
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def get_route_counts() -> dict[str, int]:
    """Behavioural probe for acceptance: proves the paths RAN, not that they exist."""
    return dict(_COUNTS)



