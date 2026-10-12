# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Drop-in `flash_attn_v100` module backed by the fa2_sm70 kernels.
#
# WHY THIS EXISTS
# ---------------
# vLLM (V1) on sm_70 selects `AttentionBackendEnum.FLASH_ATTN_V100`
# (vllm/platforms/cuda.py: `device_capability.major == 7`), and that backend
# (vllm/v1/attention/backends/flash_attn_v100.py) lazily imports a module named
# `flash_attn_v100`, expecting exactly four entry points:
#
#     flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size,
#                     softcap, alibi_slopes, deterministic, return_attn_probs, out)
#     flash_attn_decode_paged(q, k_cache, v_cache, block_table, seq_lens,
#                             softmax_scale, out, kv_cache_dtype, k_scale, v_scale)
#     flash_attn_prefill_paged(... same signature ...)
#     reshape_and_cache_fp8(key, value, key_cache, value_cache, slot_mapping,
#                           kv_cache_dtype, k_scale, v_scale)
#
# So the whole integration is a *module substitution*: put this package ahead of
# the incumbent on sys.path and vLLM runs on our kernels, with no vLLM patch.
#
# MEASURED vs the incumbent (`flash_attn_v100` 26.02), V100 clock-locked 1530,
# Qwen3.5-27B full-attention shapes (H=24, Hkv=4, d=256, causal):
#     prefill  S=1024/2048/4096      -> 1.71x / 1.64x / 1.46x
#     decode   B1c4096/B1c16k/B4c8k/B1c64k -> 1.42x / 1.75x / 2.23x / 1.91x
# both matching the fp32 oracle at relL2 ~2.6e-4 (prefill) / ~3.0e-4 (decode).
#
# THE ONE ARCHITECTURAL REQUIREMENT (`k_absmax`)
# ---------------------------------------------
# Our decode is the *deferred-rescale* flash-decoding kernel: it needs a per
# (sequence, kv-head) bound `max_j ||k_j||`. fa2_sm70.decode_paged can compute it,
# but its fallback walks the whole block table in Python on every step -- that
# alone made paged decode ~45x SLOWER than the incumbent (5.09 ms vs 0.11 ms at
# ctx=4096). A serving engine must maintain it incrementally. We do exactly that:
# a per-BLOCK table `[num_blocks, Hkv]` updated in `reshape_and_cache_fp8` (only
# the blocks touched by the new tokens), then gathered through `block_table` at
# decode time (one vectorised amax, microseconds). The bound only has to be an
# upper bound on the sequence's keys, and a max over its blocks is exactly that.
from __future__ import annotations

import math
import os
import shutil
import sys

import torch

from . import _probe

# --- ВТОРОЙ РУБЕЖ ПОДМЕНЫ СКАНА GDN ---------------------------------------
# Основной рубеж -- sitecustomize.py в этом же каталоге: он встаёт ДО импорта vLLM и потому
# успевает переписать имя раньше, чем qwen3_next его свяжет. Здесь -- страховка на случай, когда
# sitecustomize не отработал (урезанный site, чужой PYTHONPATH, прямой импорт из теста):
# repatch() переписывает globals у уже импортированных потребителей. Двойной подмены не будет --
# признак __fa2sm70__ её ловит. Молча: этот путь не обязан работать, он обязан не мешать.
try:  # noqa: SIM105
    import fa2_gdn_shim as _gdnshim

    _gdnshim.install()
    _gdnshim.repatch()
except Exception:  # noqa: BLE001
    pass


__version__ = "fa2sm70-1.0"

__all__ = [
    "flash_attn_func",
    "flash_attn_decode_paged",
    "flash_attn_prefill_paged",
    "reshape_and_cache_fp8",
]

# --- make the JIT build work regardless of the host env --------------------
# vLLM spawns its TP workers through a multiproc executor whose environment does NOT carry the
# launcher's CC/CXX, so torch's extension loader fell back to conda's gcc (>14) and nvcc refused it
# ("unsupported GNU version"), taking the whole engine down. Pin the toolchain here, inside the module
# that needs it, so the shim is self-sufficient no matter who imports it. Only fills in what's missing.
def _pin_build_toolchain() -> None:
    for var, val in (("CC", "/usr/bin/gcc"), ("CXX", "/usr/bin/g++"), ("CUDAHOSTCXX", "/usr/bin/g++")):
        cur = os.environ.get(var, "")
        if (not cur) or ("conda" in cur) or (not os.path.exists(cur)):
            if os.path.exists(val):
                os.environ[var] = val
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    cuda_home = os.environ.get("CUDA_HOME", "")
    if not cuda_home or not os.path.isdir(cuda_home):
        nvcc = shutil.which("nvcc")
        if nvcc:
            os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
        elif os.path.isdir("/usr/local/cuda"):
            os.environ["CUDA_HOME"] = "/usr/local/cuda"
    # torch's loader shells out to `ninja`, which is not on a bare worker's PATH.
    ninja = shutil.which("ninja")
    extra = [p for p in (os.path.dirname(ninja) if ninja else "",
                         os.path.join(os.environ.get("CUDA_HOME", ""), "bin")) if p and os.path.isdir(p)]
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])


_pin_build_toolchain()

# --- locate fa2_sm70 -------------------------------------------------------
# Allow running from a checkout without installing (FA2SM70_PATH=/path/to/repo).
_p = os.environ.get("FA2SM70_PATH")
if _p and _p not in sys.path:
    sys.path.insert(0, _p)
else:
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))  # .../fa2_sm70_cutlass_grade
    if os.path.isdir(os.path.join(_repo, "fa2_sm70")) and _repo not in sys.path:
        sys.path.insert(0, _repo)

import fa2_sm70  # noqa: E402
from fa2_sm70 import flash_attn_interface as _fa  # noqa: E402

# Optional: the incumbent, used only for paths we do not implement yet (fp8 cache
# stores). Absent is fine -- we then report "unsupported" and vLLM falls back to
# its Triton path, which is correct (just slower).
def _load_incumbent():
    """Load the pre-existing `flash_attn_v100` package by PATH (our module shadows the name).

    Used only for paths we do not implement yet -- today: an fp8 (uint8) paged cache, where the
    engine both stores and reads e4m3. Delegating keeps the server correct and still lets our
    prefill kernel win; raising here would crash vLLM (the backend calls decode without a guard)."""
    import importlib
    import site
    ours = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../integrations/vllm_sm70
    dirs = []
    if hasattr(site, "getsitepackages"):
        dirs += list(site.getsitepackages())
    dirs += [d for d in sys.path if d]
    seen = set()
    for d in dirs:
        d = os.path.abspath(d)
        if d in seen or d == os.path.abspath(ours):
            continue
        seen.add(d)
        if not os.path.isfile(os.path.join(d, "flash_attn_v100", "__init__.py")):
            continue
        # The incumbent's own __init__ does `from flash_attn_v100.flash_attn_interface import ...`
        # -- an ABSOLUTE import that would resolve back to *us* (we are already in sys.modules under
        # that name). So drop our entries for the duration and let it import itself cleanly.
        saved_mods = {k: v for k, v in sys.modules.items() if k == "flash_attn_v100"
                      or k.startswith("flash_attn_v100.")}
        saved_path = list(sys.path)
        try:
            for k in saved_mods:
                del sys.modules[k]
            sys.path = [d] + [p for p in sys.path if os.path.abspath(p) != os.path.abspath(ours)]
            m = importlib.import_module("flash_attn_v100")
            if os.path.abspath(m.__file__) == os.path.abspath(__file__):
                continue                                   # still us -> not a real incumbent
            return m
        except Exception:
            continue
        finally:
            sys.path = saved_path
            for k in [k for k in sys.modules if k == "flash_attn_v100"
                      or k.startswith("flash_attn_v100.")]:
                del sys.modules[k]
            sys.modules.update(saved_mods)                  # restore US as `flash_attn_v100`
    return None


_INC = None
_INC_TRIED = False


def _incumbent():
    global _INC, _INC_TRIED
    if not _INC_TRIED:
        _INC_TRIED = True
        _INC = _load_incumbent()
    return _INC


def _is_fp16_cache(kv_cache_dtype: str, cache: torch.Tensor) -> bool:
    return kv_cache_dtype in ("auto", "fp16", "float16") and cache.dtype in (torch.float16, torch.bfloat16)


def _is_e5m2_cache(kv_cache_dtype: str, cache: torch.Tensor) -> bool:
    """e5m2-пул: единственный однобайтовый формат, который наше ядро внимания ест БЕЗ конверсии.
    Байт e5m2 -- это ровно старший байт fp16 (fp16_bits == byte << 8), поэтому сборка вырождается
    в перекладку страниц, а разворот в ядре -- в одну перестановку байт. Ошибка при этом РОВНО
    ноль: замерено torch.equal против fp16-ядра на тех же числах (см. README, раздел про e5m2)."""
    return kv_cache_dtype == "fp8_e5m2" and cache.element_size() == 1


def _is_e4m3_cache(kv_cache_dtype: str, cache: torch.Tensor) -> bool:
    """vLLM's fp8_e4m3 paged pool (1 byte/element). Our decode reads it in place."""
    return kv_cache_dtype in ("fp8", "fp8_e4m3") and cache.element_size() == 1


# --- ФОРМАТ ПУЛА: int8 + масштаб НА ПОЗИЦИЮ ------------------------------------------------------
# ЧТО МЕНЯЕТСЯ И ПОЧЕМУ РАЗРЕШЕНИЯ vLLM НЕ НУЖНО. В списке форматов KV у vLLM int8 нет вовсе
# (vllm/config/cache.py принимает только fp8_e4m3 и fp8_e5m2), но пул e4m3 -- это uint8, а движок
# считает только КОЛИЧЕСТВО байт и раскладку страниц. Записываем в пул мы (reshape_and_cache_fp8),
# читаем тоже мы (prefill_paged, decode_paged), поэтому смысл байта -- наше внутреннее дело: движку
# остаётся его бухгалтерия (--kv-cache-dtype fp8_e4m3), а в те же байты идёт int8 с масштабом.
#
# ПОЧЕМУ ЭТО ВЫГОДНО (замерено на настоящих K/V этой модели, ошибка на выходе внимания):
#   * e4m3, как сейчас в пуле        2.90e-2 послойная RMS (1.21e-2..5.75e-2);
#   * int8 + масштаб на позицию      6.18e-3  -- ВТРОЕ ТОЧНЕЕ ПРИ ТОМ ЖЕ БАЙТЕ;
#   плюс сборка префилла в 1.82 раза дешевле и ядро ролей быстрее fp16-го при d=256.
#
# ОКУПАЕТСЯ ТОЛЬКО ЕСЛИ ПУЛ ХРАНИТ int8 СРАЗУ. Перекантовка e4m3 -> int8 ПРИ ЧТЕНИИ (маршрут
# FA2SM70_SHIM_I8 ниже) -- это ВТОРОЕ квантование поверх первого, и оно стоит 12.6-17.9x точности:
# e4m3 держит ОТНОСИТЕЛЬНУЮ точность на широком порядке, а int8 с масштабом по строке -- АБСОЛЮТНУЮ.
# Поэтому выигрыш этой ветки нельзя мерить на маршруте с перекантовкой.
#
# ЕДИНИЦЫ. Масштаб хранится в ИСТИННЫХ единицах fp16 (значение = q*s), поэтому k_scale/v_scale движка
# на этом пути НЕ УЧАСТВУЮТ вовсе. Скалярный масштаб на слой существовал только чтобы вписать
# значения в узкий диапазон e4m3; масштаб на позицию делает это точнее и локально.
_KV_I8 = os.environ.get("FA2SM70_KV_I8", "0") == "1"
# КАКИМ ЯДРОМ ЧИТАТЬ int8-пул НА ПРЕФИЛЛЕ -- вопрос ТОЛЬКО СКОРОСТИ, не точности: разворот q*s
# точен, поэтому оба маршрута дают одно и то же с точностью до порядка сложений.
#   1 (по умолчанию): байты как есть -> ядро ролей attn_fwd_volta_i8 (только d=256).
#   0: точный разворот в fp16 -> обычное fp16-ядро attn_fwd_qbshd. Это ЧЕСТНЫЙ способ отделить
#      эффект ФОРМАТА от эффекта ЯДРА: ядро тогда ровно то же, что на e4m3-пути.
_KV_I8_BYTES = os.environ.get("FA2SM70_KV_I8_BYTES", "1") == "1"

# ЯВНЫЙ ПРИЗНАК ФОРМАТА, ПРОВЕРЯЕМЫЙ ВО ВСЕХ ЧЕТЫРЁХ МЕСТАХ (запись / сборка / префилл / декод).
# Рассогласование хранения и чтения даёт сервер, который ПОДНИМАЕТСЯ и выдаёт бессмыслицу -- в этом
# проекте это уже было ровно так (граница ||k|| без нашего store: bound=0, exp(score-0) переполняется,
# на выходе мусор). Поэтому формат не выводится по месту, а ЗАПИСЫВАЕТСЯ store'ом и СВЕРЯЕТСЯ
# читателями; несовпадение -- ОТКАЗ, а не молчаливый расчёт и не тихий откат на другой формат.
_FMT: dict[tuple, str] = {}
# Побочные таблицы масштабов по слоту: fp32 [num_blocks * block_size * Hkv] на каждый из K и V.
# Постоянный адрес (как у _bound8) -- это то, что делает их пригодными внутри захваченного графа.
_SCALES: dict[tuple, tuple] = {}


def _kv_fmt(k_cache: torch.Tensor) -> str:
    """Формат, которым РЕАЛЬНО заполнен этот пул (по записи store'а), иначе -- настроенный."""
    return _FMT.get(_cache_key(k_cache), "i8" if _KV_I8 else "e4m3")


# ВАРИАНТ ЯДРА РОЛЕЙ, ПРОВЕДЁННЫЙ В БОЕВОЙ ВЫЗОВ.
# Набор вариантов (широкая перекладка, широкая загрузка + узкая укладка, расщепление накопителя,
# разведение фаз по строкам, плюс встроенные ФАЛЬСИФИКАТОРЫ фаз) существует в ядре с давних пор,
# но боевые точки вызова звали `attn_fwd_volta_i8` БЕЗ параметра -- то есть всегда базовым nosc=0.
# Рычаги были собраны и не заведены в маршрут. Теперь заведены, и развёртку можно гонять СКВОЗЬ
# СЕРВЕР на боевой форме, а не на стенде с угаданными буферами.
#   nosc = вариант*100 + формат. 0 -- база. 108/408 -- перекладки. 3008 -- расщепление накопителя.
#   2008 -- разведение фаз. 11xx..16xx -- ФАЛЬСИФИКАТОРЫ: ответ НЕВЕРЕН, читается только время.
_WS_NOSC = int(os.environ.get("FA2SM70_WS_NOSC", "0"))

_SAID: set = set()


def _say_once(what: str) -> None:
    """Один раз на процесс сообщить, КАКОЙ маршрут реально пошёл в дело.

    Проверять надо точку входа, а не баннер: на этом проекте уже был A/B, сдвинувший сквозную
    метрику на 0.03%, -- потому что сервер в тот код просто не заходил. Одна строка на маршрут
    стоит ноль и снимает целый класс ложных выводов."""
    if what in _SAID:
        return
    _SAID.add(what)
    print(f"[fa2_sm70] route: {what}", file=sys.stderr, flush=True)


def _scale_tabs(k_cache: torch.Tensor) -> tuple:
    """Таблицы масштабов этого пула: [num_blocks, block_size, Hkv] fp32, выделяются один раз.

    ПРЕЦЕДЕНТ -- ТАБЛИЦА ГРАНИЦЫ ||k|| (_bound8/_absmax_table): та же механика жизненного цикла
    (заводится по первому касанию пула, ЗАПОЛНЯЕТСЯ записью KV, ПОТРЕБЛЯЕТСЯ чтением), тот же
    постоянный адрес ради захваченного графа, тот же ключ по (указатель, блоки, Hkv).

    Стоимость памяти считаемая: 8 байт на слот против d байт данных, то есть d/8 = 3.1% при d=256
    (по 1.6% на K и V). Это цена трёхкратной точности при том же байте на элемент.

    ЧЕМ ЭТА ТАБЛИЦА НАДЁЖНЕЕ ГРАНИЦЫ ||k|| ПРИ ВЫТЕСНЕНИИ. Граница ведётся ПО БЛОКУ и только растёт,
    поэтому переиспользованный блок оставляет её завышенной (законно: нужна лишь верхняя оценка).
    Масштаб же пишется ПО СЛОТУ и ТЕМ ЖЕ вызовом, что и байты этого слота, поэтому вытеснение и
    переиспользование блока не могут развести данные и масштаб во времени -- они всегда одного
    возраста по построению.
    ЧЕГО ОНА НЕ УМЕЕТ. Таблица привязана к ФИЗИЧЕСКОМУ блоку, поэтому разделение блоков между
    запросами (префиксный кэш -- чтение) корректно даром. А вот КОПИРОВАНИЕ блоков движком
    (copy_blocks при ветвлении n>1 в v0, обмен с хостом) скопировало бы байты и НЕ скопировало
    масштабы -- копия читалась бы с масштабом 0, то есть как нули. В vLLM v1 этого пути нет, но если
    он появится, таблицу надо переставлять тем же индексом."""
    ck = _cache_key(k_cache)
    t = _SCALES.get(ck)
    if t is None:
        n = int(k_cache.shape[0]) * int(k_cache.shape[1]) * int(k_cache.shape[2])
        t = (torch.zeros(n, device=k_cache.device, dtype=torch.float32),
             torch.zeros(n, device=k_cache.device, dtype=torch.float32))
        _SCALES[ck] = t
    return t


def _require_i8(k_cache: torch.Tensor, who: str) -> tuple:
    """ГЕЙТ СОГЛАСОВАННОСТИ для читателей int8-пула: формат совпал И таблицы масштабов существуют.

    Второе -- не паранойя, а тот же отказ, что уже случался: таблицы заводит store, поэтому их
    ОТСУТСТВИЕ означает, что запись шла другим путём (например FA2SM70_SHIM_STORE=0), а значит в
    байтах лежит НЕ то, что мы собираемся прочитать. Молча прочитать -- это и есть тот самый сервер,
    который «работает» и врёт; поэтому здесь падение."""
    fmt = _kv_fmt(k_cache)
    if fmt != "i8":
        raise RuntimeError(f"fa2_sm70 shim ({who}): формат пула '{fmt}', а маршрут int8. "
                           f"Хранение, сборка, префилл и декод обязаны читать ОДИН формат "
                           f"(FA2SM70_KV_I8 согласуй со store'ом).")
    ck = _cache_key(k_cache)
    if ck not in _SCALES:
        raise RuntimeError(f"fa2_sm70 shim ({who}): int8-пул выбран, но таблицы масштабов не "
                           f"заведены -- значит store в этот пул не писал (FA2SM70_SHIM_STORE=0?). "
                           f"Читать такие байты как int8 нельзя.")
    return _SCALES[ck]


# Running max ||k_raw|| per kv-head for the e4m3 pool. "raw" = the stored (pre-k_scale) units, which is
# what the kernel sees; ||k_raw|| = ||k_fp16|| / k_scale, so we can fold it in at store time without
# ever reading the pool back. A 1.15 margin keeps it an upper bound despite e4m3 round-to-nearest.
_GLOBAL8: dict[tuple, torch.Tensor] = {}


def _on(name: str) -> bool:
    """Per-entry-point kill switch for A/B against the incumbent without rebuilding:
    FA2SM70_SHIM_PREFILL / _DECODE / _STORE = 0 delegates just that call.

    NB these must stay in sync: our decode needs the ||k|| bound that our store maintains, so turning
    the store off while the decode is on leaves the bound at zero, exp(score-0) overflows and the model
    emits garbage. `_on("DECODE")` therefore also requires the store (see flash_attn_decode_paged)."""
    return os.environ.get(f"FA2SM70_SHIM_{name}", "1") != "0"


_WS: dict[tuple, tuple] = {}


def _workspace(B: int, H: int, ns: int, d: int, device) -> tuple:
    """Cached split workspace (A: partial numerators, B: partial denominators), keyed by shape --
    allocated once per shape, never per call. Stable addresses also make it CUDA-graph friendly."""
    key = (B, H, ns, d, device.index if device.index is not None else -1)
    w = _WS.get(key)
    if w is None:
        # B is 2*B*H*ns: the partial denominators, then the per-split maxima. The decode kernel went
        # online-max (no externally sourced bound), so splits no longer share one exponent shift and the
        # reducer needs each split's max to weight it. Parking the maxima in the tail of this buffer keeps
        # the entry's signature and its zero-allocation property intact.
        w = (torch.empty(B * H * ns * d, device=device, dtype=torch.float32),
             torch.empty(2 * B * H * ns, device=device, dtype=torch.float32))
        _WS[key] = w
    return w


def _bound8(k_cache: torch.Tensor) -> torch.Tensor:
    """Persistent per-cache ||k_raw|| bound [Hkv]: allocated once, then only ever updated IN PLACE by
    the stores. A stable address is what makes it usable from inside a captured CUDA graph."""
    ck = _cache_key(k_cache)
    t = _GLOBAL8.get(ck)
    if t is None:
        t = torch.zeros(k_cache.shape[2], device=k_cache.device, dtype=torch.float32)
        _GLOBAL8[ck] = t
    return t
_E4M3_MARGIN = 1.15


_PROF = {"n": 0, "t": 0.0}


def _prof(dt: float) -> None:
    """FA2SM70_PROF=1: real in-server cost of this entry. Microbenchmarks said 0.05 ms/call while the
    server behaved like ~6 ms/call, so the cost has to be read where it is actually paid."""
    _PROF["n"] += 1
    _PROF["t"] += dt
    if _PROF["n"] % 50 == 0:
        print(f"[fa2_sm70 prof] decode calls={_PROF['n']} avg={_PROF['t'] / _PROF['n'] * 1e3:.3f} ms",
              file=sys.stderr, flush=True)


def _decode_e4m3(q, k_cache, v_cache, block_table, seq_lens, softmax_scale, out, k_scale, v_scale):
    import time as _time
    _prof_on = os.environ.get("FA2SM70_PROF", "0") == "1"
    if _prof_on:
        torch.cuda.synchronize()
        _t0 = _time.perf_counter()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    q4 = q.unsqueeze(2) if q.dim() == 3 else q
    B, H = q4.shape[0], q4.shape[1]
    block_size = k_cache.shape[1]
    sl = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
    bt32 = block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
    # THE BOUND MUST NEVER BE DERIVED HERE. This function runs only while vLLM CAPTURES its decode
    # CUDA graph; afterwards the graph is replayed and our Python never executes again. So anything
    # done on this path is baked in and repeated for every token of every layer. The previous version
    # fell back to scanning the whole pool (`k_cache.view(float8).float().norm(...)`, ~640 MB/layer and
    # 4x that once widened) when the table was empty at capture time -- that single line cost ~65 ms per
    # token and was the entire end-to-end regression vs the incumbent (39 vs 115 tok/s aggregate).
    # Instead: a PERSISTENT zero-initialised bound whose address is stable, so the graph bakes a pointer
    # and each replay reads whatever the stores have written in place. (A closed-form bound is not an
    # option either: e4m3 caps at 448, so ||k_raw|| <= 448*sqrt(d) over-estimates by ~50x, and then
    # exp(score - U) underflows to zero and the softmax degenerates to 0/0.)
    g = _bound8(k_cache)
    kmax = g.view(1, -1).expand(B, g.shape[0]).contiguous()
    # kv_cap is the block table's CAPACITY (max_model_len), not the live length -- reading the true
    # length needs a device sync, which is illegal while the decode graph is being captured. Sizing the
    # split count from capacity asks for ~64 splits even for a 30-token chat; each split costs a slice
    # in the reduce pass and its share of the workspace. The kernel clips every split to the actual
    # per-sequence length, so the count is purely a parallelism knob -- cap it.
    kv_cap = int(bt32.shape[1]) * int(block_size)
    # 64, not 8. The old cap was set to avoid "asking for ~64 splits even for a 30-token chat", but
    # that reasoning was never measured and it was costing 6-8x. This model gives the decode grid
    # (ns, H/GF, B) = (8, 6, 1) = 48 CTAs on 80 SMs -- most of the GPU idle -- and splits are the only
    # axis available to fill it, since H and B are fixed by the request. Measured per call (V100, the
    # server's own shapes: H=12 Hkv=2 d=256 block_size=1568, ms):
    #     ctx      8      16      32      64     128
    #    4096   2.07    1.78    1.78    1.14    1.14
    #   16384   7.96    4.44    2.29    1.16    1.16
    #   65536   9.16    4.61    2.31    1.13    1.13
    #  131072   9.43    4.71    2.36    1.42    1.43
    # Saturates at 64, and SHORT contexts get faster too (1.82x at 4K), so the feared waste is not
    # real: over-split ranges past the sequence end exit immediately. Exactness is unaffected -- the
    # deferred-rescale combine is associative, so the split count is purely a parallelism knob.
    # gf MUST be passed: the paged grid is (splits, H/gf, B), so the fill target is computed for
    # that grid and not for an unfused one (see _auto_splits).
    _gf = fa2_sm70._pick_gf(int(q4.shape[-1]), max(1, H // int(k_cache.shape[2])))
    ns = min(fa2_sm70._auto_splits(B, H, kv_cap, gf=_gf), int(os.environ.get("FA2SM70_MAX_SPLITS", "64")))
    if out is not None and out.dtype == torch.float16:
        # ZERO-ALLOCATION path: caller-owned split workspace + the reduce writes fp16 straight into
        # vLLM's own output view. This mirrors (and then beats) what the incumbent does -- it keeps
        # tmp_out/max_logits/exp_sums cached and writes into `out`. Without it our faster kernel stayed
        # buried under per-call glue: 0.265 ms/call in-server vs 0.020 ms for the kernel itself.
        A, Bw = _workspace(B, H, ns, int(q4.shape[-1]), q4.device)
        o3 = out.view(B, H, -1) if out.dim() == 3 else out.reshape(B, H, -1)
        fa2_sm70._ext.decode_ext().flash_decode_defer_mqa_paged_into(
            q4, k_cache, v_cache, kmax, bt32, int(block_size), float(softmax_scale), int(ns),
            int(kv_cap), sl, True, float(k_scale), float(v_scale), A, Bw, out, int(o3.stride(0)))
        o = out
    else:
        o, _lse = fa2_sm70._ext.decode_ext().flash_decode_defer_mqa_paged_fp8(
            q4, k_cache, v_cache, kmax, bt32, int(block_size), float(softmax_scale), int(ns),
            int(kv_cap), sl, float(k_scale), float(v_scale))
        o = o.to(q.dtype).reshape(q.shape)
    if _prof_on:
        torch.cuda.synchronize()
        _prof(_time.perf_counter() - _t0)
    return o


def _decode_i8(q, k_cache, v_cache, block_table, seq_lens, softmax_scale, out):
    """Декод из НАШЕГО int8-пула: те же байты, но масштаб берётся ПО СЛОТУ из побочной таблицы.

    Отличий от e4m3-пути ровно два, и оба принципиальные: (1) ядро инстанцируется с KVB=8, то есть
    читает байт как знаковое целое, а не как e4m3; (2) k_scale/v_scale движка НЕ УЧАСТВУЮТ -- ни в
    softmax-масштабе, ни на выходе, потому что масштаб на позицию уже несёт истинные единицы.
    Всё остальное -- сетка, число дроблений, площадка без аллокаций -- то же самое, поэтому цена
    декода отличается только на 4 байта масштаба на (позицию, kv-голову), то есть на 1.6% трафика."""
    import time as _time
    _prof_on = os.environ.get("FA2SM70_PROF", "0") == "1"
    if _prof_on:
        torch.cuda.synchronize()
        _t0 = _time.perf_counter()
    ks, vs = _require_i8(k_cache, "decode")
    _say_once("decode_paged=int8 pool + per-position scale tables (KVB=8)")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    q4 = q.unsqueeze(2) if q.dim() == 3 else q
    B, H = q4.shape[0], q4.shape[1]
    block_size = k_cache.shape[1]
    sl = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
    bt32 = block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
    kv_cap = int(bt32.shape[1]) * int(block_size)
    _gf = fa2_sm70._pick_gf(int(q4.shape[-1]), max(1, H // int(k_cache.shape[2])))
    ns = min(fa2_sm70._auto_splits(B, H, kv_cap, gf=_gf), int(os.environ.get("FA2SM70_MAX_SPLITS", "64")))
    # Граница ||k|| здесь ВЕСТИГИАЛЬНА (ядро перешло на онлайновый максимум), но остаётся в подписи --
    # передаём тот же постоянный нулевой тензор, чтобы захваченный граф запёк стабильный указатель.
    kmax = _bound8(k_cache).view(1, -1).expand(B, k_cache.shape[2]).contiguous()
    if out is not None and out.dtype == torch.float16:
        A, Bw = _workspace(B, H, ns, int(q4.shape[-1]), q4.device)
        o3 = out.view(B, H, -1) if out.dim() == 3 else out.reshape(B, H, -1)
        fa2_sm70._ext.decode_ext().flash_decode_defer_mqa_paged_into(
            q4, k_cache, v_cache, kmax, bt32, int(block_size), float(softmax_scale), int(ns),
            int(kv_cap), sl, False, 1.0, 1.0, A, Bw, out, int(o3.stride(0)), ks, vs)
        o = out
    else:
        o, _lse = fa2_sm70._ext.decode_ext().flash_decode_defer_mqa_paged_i8(
            q4, k_cache, ks, v_cache, vs, kmax, bt32, int(block_size), float(softmax_scale),
            int(ns), int(kv_cap), sl)
        o = o.to(q.dtype).reshape(q.shape)
    if _prof_on:
        torch.cuda.synchronize()
        _prof(_time.perf_counter() - _t0)
    return o


# --- incremental per-block ||k|| bound -------------------------------------
# key: (data_ptr, num_blocks, Hkv) of the K cache -> fp32 [num_blocks, Hkv]
_ABSMAX: dict[tuple, torch.Tensor] = {}


_POPULATED: set = set()          # caches whose per-block table we have actually written
# Running max ||k|| over the WHOLE pool, per kv-head: fp32 [Hkv]. This is what decode actually reads.
# A pool-wide max is a valid (slightly loose) upper bound for every sequence, and the deferred-rescale
# identity makes the result EXACTLY independent of the bound -- U only sets the precision margin. Using
# it turns the per-step cost into a view+copy of Hkv floats instead of a block-table gather, which
# measured 0.179 ms of the shim's 0.260 ms (69%) at ctx=4096.
_GLOBAL: dict[tuple, torch.Tensor] = {}


def _absmax_table(k_cache: torch.Tensor) -> torch.Tensor:
    """Per-block max ||k|| table for this K cache, created on first touch."""
    nblocks, hkv = k_cache.shape[0], k_cache.shape[2]
    key = (k_cache.data_ptr(), nblocks, hkv)
    t = _ABSMAX.get(key)
    if t is None:
        t = torch.zeros(nblocks, hkv, device=k_cache.device, dtype=torch.float32)
        _ABSMAX[key] = t
    return t


def _cache_key(k_cache: torch.Tensor) -> tuple:
    return (k_cache.data_ptr(), k_cache.shape[0], k_cache.shape[2])


def _update_absmax(key: torch.Tensor, key_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    """Fold the newly written tokens into the per-block bound. O(new tokens)."""
    tab = _absmax_table(key_cache)
    block_size = key_cache.shape[1]
    blk = (slot_mapping // block_size).long()               # [n] physical block of each new token
    n = key.reshape(key.shape[0], -1, key.shape[-1])        # [n, Hkv, D]
    nrm = n.float().norm(dim=-1)                            # [n, Hkv]
    tab.index_reduce_(0, blk, nrm, "amax", include_self=True)
    ck = _cache_key(key_cache)
    _POPULATED.add(ck)
    g = _GLOBAL.get(ck)                                      # pool-wide running max, O(new tokens)
    hi = nrm.amax(dim=0)
    _GLOBAL[ck] = hi if g is None else torch.maximum(g, hi)


def _seq_absmax(k_cache: torch.Tensor, block_table: torch.Tensor,
                seq_lens: torch.Tensor) -> torch.Tensor:
    """Gather the per-sequence bound [B, Hkv] = max over the blocks it owns."""
    tab = _absmax_table(k_cache)
    bt = block_table.long()
    B, maxblk = bt.shape
    block_size = k_cache.shape[1]
    nblk = ((seq_lens.to(bt.device).long() + block_size - 1) // block_size).clamp(min=0)
    valid = torch.arange(maxblk, device=bt.device).view(1, -1) < nblk.view(-1, 1)
    g = tab[bt.clamp(min=0, max=tab.shape[0] - 1)]          # [B, maxblk, Hkv]
    g = torch.where(valid.unsqueeze(-1), g, torch.zeros((), device=g.device, dtype=g.dtype))
    return g.amax(dim=1).contiguous()                        # [B, Hkv]


def _absmax_fallback(k_cache: torch.Tensor, block_table: torch.Tensor,
                     seq_lens: torch.Tensor) -> torch.Tensor:
    """If the table was never populated (e.g. the engine wrote the cache through a
    path we did not see), derive a *valid* bound from the whole pool once. Correct
    (an over-estimate is still an upper bound) and vectorised."""
    return k_cache.float().norm(dim=-1).amax(dim=1).amax(dim=0).view(1, -1).expand(
        block_table.shape[0], k_cache.shape[2]).contiguous()


_TRACE = os.environ.get("FA2SM70_TRACE", "0") == "1"


def _tr(msg: str) -> None:
    """Entry/exit trace to stderr (FA2SM70_TRACE=1). Worth keeping: vLLM boots by running dummy
    forwards for graph capture, and if one of our entries blocks there the engine hangs with no
    traceback at all -- only the last line printed says which entry it was."""
    if _TRACE:
        print(f"[fa2_sm70] {msg}", file=sys.stderr, flush=True)


# --- ЭТАЛОННОЕ РАЗРЕЖЕННОЕ ВНИМАНИЕ (замер накопления ошибки по глубине) ---------------------
# Отбор блоков ключей по ИСТИННОЙ массе (оракул) внутри работающей модели. Медленно и намеренно:
# это ИЗМЕРИТЕЛЬНЫЙ путь, а не ускорение. Включается FA2SM70_SPARSE_REF=1 плюс файл управления
# FA2SM70_SPARSE_CTL (json {"dens":0.75,"layers":"all"}), который перечитывается по mtime -- так
# один поднятый сервер проходит весь набор плотностей без перезапуска (перезапуск между точками
# добавил бы своё расхождение, а сравнивать надо ТОЛЬКО разреженность).
_SPARSE_REF = os.environ.get("FA2SM70_SPARSE_REF", "0") == "1"
_SPARSE_CTL = os.environ.get("FA2SM70_SPARSE_CTL", "")
_SP: dict = {"dens": 1.0, "layers": "all", "mtime": -1.0, "exact": False,
             "qt": int(os.environ.get("FA2SM70_SPARSE_QT", "256")),
             "mant": 10,
             "kept": 0.0, "tot": 0.0, "calls": 0, "mod": None, "on": False}
# Латч слоя: индекс = ПОЗИЦИЯ В ПРЯМОМ ПРОХОДЕ (внутри прохода слои идут 0..15, и запись KV каждого
# слоя предшествует его вниманию). Само-пересинхронизируется: повторно увиденный указатель кэша
# означает начало нового прохода. Не по значению указателя -- порядок выделения не гарантирован.
_LAY: dict = {"seen": [], "cur": -1, "scale": {}, "warn": []}


def _lay_latch(cache) -> int:
    ptr = cache.data_ptr()
    seen = _LAY["seen"]
    if ptr in seen:
        seen = [ptr]
        _LAY["seen"] = seen
    else:
        seen.append(ptr)
    _LAY["cur"] = len(seen) - 1
    return _LAY["cur"]


def _sp_cfg() -> dict:
    """Перечитать файл управления, если он изменился."""
    if not _SPARSE_CTL:
        return _SP
    try:
        mt = os.path.getmtime(_SPARSE_CTL)
    except OSError:
        return _SP
    if mt != _SP["mtime"]:
        try:
            import json as _json
            with open(_SPARSE_CTL) as f:
                c = _json.load(f)
            _SP["dens"] = float(c.get("dens", 1.0))
            _SP["layers"] = c.get("layers", "all")
            _SP["exact"] = bool(c.get("exact", False))
            _SP["noise"] = float(c.get("noise", 0.0))
            # ПОСЛОЙНАЯ sigma: позволяет воспроизвести профиль ошибки НАСТОЯЩЕГО e4m3 по слоям
            # (замерен отдельно), а не только его среднеквадратичный уровень. Тогда множитель
            # накопления уже оплаченного квантования меряется на его собственном профиле.
            nl = c.get("noise_layers")
            _SP["noise_layers"] = [float(x) for x in nl] if nl else None
            _SP["qt"] = int(c.get("qt", _SP["qt"]))
            _SP["mant"] = int(c.get("mant", 10))
            _SP["mtime"] = mt
            _SP["tag"] = str(c.get("tag", ""))
            _SP["kept"] = _SP["tot"] = 0.0
            _SP["calls"] = 0
            print(f"[fa2_sm70] sparse-ref cfg tag={_SP['tag']} dens={_SP['dens']} "
                  f"layers={_SP['layers']} exact={_SP['exact']} qt={_SP['qt']} "
                  f"mant={_SP['mant']}",
                  file=sys.stderr, flush=True)
        except Exception as e:                                # noqa: BLE001
            print(f"[fa2_sm70] sparse-ref cfg read failed: {e}", file=sys.stderr, flush=True)
    return _SP


def _sp_layer_on(lay: int) -> bool:
    sel = _SP["layers"]
    if sel == "all":
        return True
    if isinstance(sel, list):
        return lay in sel
    if isinstance(sel, str) and sel.startswith("first"):
        return lay < int(sel[5:])
    if isinstance(sel, str) and sel.startswith("last"):
        return lay >= 16 - int(sel[4:])
    if isinstance(sel, str) and sel == "none":
        return False
    return True


def _sp_on(lay: int) -> bool:
    """Путь эталона используется и при dens=1.0 -- это КОНТРОЛЬ, он обязан идти той же арифметикой."""
    if not _SPARSE_REF:
        return False
    c = _sp_cfg()
    if c["layers"] == "none":
        return False
    return lay >= 0


def _sp_mod():
    m = _SP["mod"]
    if m is None:
        from . import sparse_ref as m
        _SP["mod"] = m
    return m


def _sp_run(q4, kb, vb, scale, lay: int, T: int):
    """q4 [1,Sq,H,d]; kb/vb [Hkv,T,d]. Возвращает [1,Sq,H,d]."""
    c = _sp_cfg()
    on = _sp_layer_on(lay)
    dens = c["dens"] if on else 1.0
    m = _sp_mod()
    st = {"kept": 0.0, "tot": 0.0}
    Sq = int(q4.shape[1])
    nl = c.get("noise_layers")
    sig = float(nl[lay]) if (nl and 0 <= lay < len(nl)) else float(c.get("noise", 0.0))
    mant = int(c.get("mant", 10))
    if on and mant < 10:                    # детерминированная ошибка формата (см. coarsen_kv)
        kb, vb = m.coarsen_kv(kb, vb, mant)
    elif on and sig > 0.0:
        kb, vb = m.perturb_kv(kb, vb, sig,
                              1000003 * (lay + 1) + 97 * (int(T) - Sq) + 7)
    o = m.sparse_attn(q4[0], kb, vb, float(scale), float(dens), int(T) - Sq,
                      QT=c["qt"], exact=c["exact"], stats=st)
    _SP["kept"] += st["kept"]
    _SP["tot"] += st["tot"]
    _SP["calls"] += 1
    if _SP["calls"] % 128 == 0 and _SP["tot"]:
        print(f"[fa2_sm70] sparse-ref calls={_SP['calls']} dens_cfg={c['dens']} "
              f"kept={_SP['kept'] / _SP['tot']:.4f}", file=sys.stderr, flush=True)
    return o.unsqueeze(0)


# --- сброс логитов последнего токена (сквозная метрика) --------------------------------------
_LOGIT_DUMP = os.environ.get("FA2SM70_LOGIT_DUMP", "")
_LG: dict = {"installed": False, "n": 0}


def _install_logit_hook() -> None:
    """Сохранить ПОЛНЫЙ вектор логитов позиции, с которой начинается генерация.

    Через API его не достать (logprobs отдаёт только верхушку), а именно он -- сквозная метрика.
    Патчится LogitsProcessor.forward: при выборке одной строки логиты уже собраны по всем рангам
    (use_all_gather=True на CUDA), поэтому пишем с ранга 0."""
    if _LG["installed"] or not _LOGIT_DUMP:
        return
    _LG["installed"] = True
    try:
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
    except Exception as e:                                    # noqa: BLE001
        print(f"[fa2_sm70] logit hook unavailable: {e}", file=sys.stderr, flush=True)
        return
    orig = LogitsProcessor.forward

    def patched(self, lm_head, hidden_states, embedding_bias=None):
        out = orig(self, lm_head, hidden_states, embedding_bias)
        try:
            if out is not None and out.dim() == 2 and out.shape[0] == 1 and _qk_rank() == 0:
                _LG["n"] += 1
                torch.save({"logits": out[0].detach().float().cpu(), "n": _LG["n"]},
                           _LOGIT_DUMP)
        except Exception:                                     # noqa: BLE001
            pass
        return out

    LogitsProcessor.forward = patched
    print(f"[fa2_sm70] logit dump armed -> {_LOGIT_DUMP}", file=sys.stderr, flush=True)


_PROF_PREFILL = os.environ.get("FA2SM70_PROF_PREFILL", "0") == "1"
_PP = {"n": 0, "gather": 0.0, "attn": 0.0, "tok": 0}


def _pp_report():
    """In-server accounting for the prefill path. Standalone kernel timings said attention scales
    exactly quadratically, yet measured TTFT grows FASTER than that above ~65K -- so the excess is
    somewhere else, and guessing was already wrong once (bigger chunks changed nothing). This reports
    what our path actually costs inside the server, so the remainder is attributable by subtraction."""
    if _PP["n"]:
        print(f"[fa2_sm70] prefill_paged calls={_PP['n']} qtok={_PP['tok']} "
              f"gather={_PP['gather']:.2f}s attn={_PP['attn']:.2f}s", file=sys.stderr, flush=True)


_DUMP_DIR = os.environ.get("FA2SM70_DUMP_KV", "")
_DUMP_MAX = int(os.environ.get("FA2SM70_DUMP_MAX", "16384"))
_DUMPED: dict[int, list] = {}


def _dump_kv(key, value, key_cache) -> None:
    """Capture real K/V (fp16, pre-quantisation) per layer, for the KV-compression study.

    Off unless FA2SM70_DUMP_KV names a directory. Any claim about how compressible this model's KV
    is has to be measured on ITS OWN stores -- concentration and spectral tails are properties of
    the trained model, not of attention in general -- and the store is the one place that sees every
    layer's K and V before they are quantised. Keyed by cache pointer, which identifies the layer."""
    if not _DUMP_DIR:
        return
    # A device->host copy is illegal while vLLM captures its graphs, and the tensors there are dummy
    # anyway. Skipping capture keeps this a pure debug hook that cannot take the engine down.
    if key.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    lid = key_cache.data_ptr()
    buf = _DUMPED.setdefault(lid, [])
    have = sum(b[0].shape[0] for b in buf)
    if have >= _DUMP_MAX:
        return
    take = min(key.shape[0], _DUMP_MAX - have)
    buf.append((key[:take].detach().to(torch.float16).cpu(),
                value[:take].detach().to(torch.float16).cpu()))
    if have + take >= _DUMP_MAX:
        import os as _os
        _os.makedirs(_DUMP_DIR, exist_ok=True)
        k = torch.cat([b[0] for b in buf]); v = torch.cat([b[1] for b in buf])
        idx = sorted(_DUMPED).index(lid)
        torch.save({"k": k, "v": v}, _os.path.join(_DUMP_DIR, f"kv_layer{idx:02d}_{lid:x}.pt"))
        print(f"[fa2_sm70] dumped KV layer#{idx} {tuple(k.shape)} -> {_DUMP_DIR}", file=sys.stderr, flush=True)


_DUMPED_Q: dict[int, bool] = {}


def _dump_q(q, key_cache, T: int) -> None:
    """Capture real QUERIES alongside the K/V dump. The block-skip question ('what fraction of the
    store can a certified bound prove is negligible?') is a joint property of q and K, so it cannot
    be answered from the store alone. These are the last chunk's queries attending to a long prefix
    -- the same projection and the same regime a decode query sees."""
    if not _DUMP_DIR:
        return
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    lid = key_cache.data_ptr()
    if _DUMPED_Q.get(lid) or T < 8192:                   # want queries with a long prefix behind them
        return
    _DUMPED_Q[lid] = True
    import os as _os
    _os.makedirs(_DUMP_DIR, exist_ok=True)
    torch.save({"q": q[0, -256:].detach().to(torch.float16).cpu(), "T": T},
               _os.path.join(_DUMP_DIR, f"q_{lid:x}.pt"))
    print(f"[fa2_sm70] dumped Q for cache {lid:x} (T={T})", file=sys.stderr, flush=True)


# --- FULL Q/K/V capture, per layer, whole prompt (block-sparsity oracle) ---------------------
# FA2SM70_DUMP_QK=<dir> captures the COMPLETE Q, K and V of every full-attention layer in fp16, for
# every prompt position -- not the 256-query tail `_dump_q` takes. The block-density question ("can a
# 64x64 block mask keep only 5-8% of the blocks and stay inside the error budget?") is a JOINT
# property of every query against every key, so a tail sample cannot answer it; and the answer has to
# be read on the ATTENTION OUTPUT (relL2), which additionally needs V.
#
# ADDITIVE, OFF BY DEFAULT, AND FED BY TWO ENTRY POINTS -- because vLLM splits a fresh prompt across
# both. The FIRST chunk has no prefix, so the backend takes the dense `flash_attn_func` (which is
# handed k/v directly and carries no cache pointer, hence the layer latch below); every LATER chunk
# has a prefix and goes through `flash_attn_prefill_paged`. K and V are always taken from the store,
# i.e. fp16 BEFORE the e4m3 quantisation, with the layer's k_scale/v_scale recorded so the pool's
# exact quantisation can be reproduced offline.
#
# ARMING. vLLM runs dummy profiling forwards at boot, on garbage tensors, through these same entry
# points; capturing those would silently poison the dump with noise that has no attention structure
# at all -- the exact failure mode the study must avoid. So capture starts only once an `ARM` sentinel
# file exists in the dump directory (create it after the server is up, immediately before sending the
# prompt) and stops for good at the first decode step.
_QK_DIR = os.environ.get("FA2SM70_DUMP_QK", "")
_QK_MAX = int(os.environ.get("FA2SM70_DUMP_QK_MAX", "131072"))
_QK: dict = {"armed": False, "done": False, "rank": None, "dir": None,
             "layer": {}, "cur": -1, "nq": {}, "nk": {}, "part": {}, "warn": []}


def _qk_rank() -> int:
    """Which TP worker we are. Both workers run this code in separate processes and would otherwise
    write over each other; the dump is split into rank<r>/ subdirectories, and the manifest records
    the pid so a rank collision would be visible instead of silent."""
    r = _QK["rank"]
    if r is not None:
        return r
    r = -1
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            r = int(dist.get_rank())
    except Exception:
        r = -1
    if r < 0:
        for v in ("LOCAL_RANK", "RANK", "VLLM_DP_RANK"):
            s = os.environ.get(v, "")
            if s.isdigit():
                r = int(s)
                break
    if r < 0:
        try:
            r = int(torch.cuda.current_device())
        except Exception:
            r = 0
    _QK["rank"] = r
    return r


def _qk_dir() -> str:
    d = _QK["dir"]
    if d is None:
        d = os.path.join(_QK_DIR, f"rank{_qk_rank()}")
        os.makedirs(d, exist_ok=True)
        _QK["dir"] = d
    return d


def _qk_on() -> bool:
    if not _QK_DIR or _QK["done"]:
        return False
    if not _QK["armed"]:
        if not os.path.exists(os.path.join(_QK_DIR, "ARM")):
            return False
        _QK["armed"] = True
    return True


def _qk_layer(cache: torch.Tensor) -> int:
    """Layer index by FIRST-SEEN order, which is model order: within a forward the layers run 0..N-1
    and each layer's store precedes its attention. NOT by pointer value -- allocation order is not
    guaranteed to follow layer order, and the whole per-layer reading of the study depends on this."""
    ptr = cache.data_ptr()
    lay = _QK["layer"].get(ptr)
    if lay is None:
        lay = len(_QK["layer"])
        _QK["layer"][ptr] = lay
    _QK["cur"] = lay
    return lay


def _qk_save(kind: str, lay: int, payload: dict) -> None:
    key = (kind, lay)
    p = _QK["part"].get(key, 0)
    _QK["part"][key] = p + 1
    payload.update({"layer": lay, "rank": _qk_rank(), "part": p})
    torch.save(payload, os.path.join(_qk_dir(), f"{kind}_L{lay:02d}_p{p:03d}.pt"))


def _dump_qk_kv(key, value, key_cache, k_scale, v_scale) -> None:
    """K/V of the new tokens, fp16, pre-quantisation. Chunks land in call order == position order."""
    if not _qk_on():
        return
    if key.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    lay = _qk_layer(key_cache)
    n = int(key.shape[0])
    have = _QK["nk"].get(lay, 0)
    take = min(n, max(0, _QK_MAX - have))
    if take <= 0:
        return
    Hkv, d = int(key_cache.shape[2]), int(key_cache.shape[3])
    _qk_save("kv", lay, {
        "k": key.reshape(n, Hkv, d)[:take].detach().to(torch.float16).cpu(),
        "v": value.reshape(n, Hkv, d)[:take].detach().to(torch.float16).cpu(),
        "start": have, "k_scale": float(k_scale), "v_scale": float(v_scale)})
    _QK["nk"][lay] = have + take


def _dump_qk_q(q, lay: int, seq_len: int) -> None:
    if lay < 0 or not _qk_on():
        return
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    q3 = q.reshape(-1, q.shape[-2], q.shape[-1])          # [B*Sq, H, d]; B==1 on the prefill paths
    n = int(q3.shape[0])
    have = _QK["nq"].get(lay, 0)
    take = min(n, max(0, _QK_MAX - have))
    if take <= 0:
        return
    _qk_save("q", lay, {"q": q3[:take].detach().to(torch.float16).cpu(),
                        "start": have, "seq_len": int(seq_len)})
    _QK["nq"][lay] = have + take


def _qk_finish(why: str) -> None:
    """Close the capture and write the manifest. Called at the first decode step, so the decode
    tokens (which have K but no captured Q) cannot desynchronise the two streams."""
    if not _QK_DIR or _QK["done"] or not _QK["armed"]:
        return
    _QK["done"] = True
    import json
    m = {"rank": _qk_rank(), "pid": os.getpid(), "why": why, "layers": len(_QK["layer"]),
         "nq": {str(k): v for k, v in sorted(_QK["nq"].items())},
         "nk": {str(k): v for k, v in sorted(_QK["nk"].items())},
         "parts": {f"{k[0]}{k[1]:02d}": v for k, v in sorted(_QK["part"].items())},
         "warn": _QK["warn"]}
    with open(os.path.join(_qk_dir(), "manifest.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(f"[fa2_sm70] QK dump closed ({why}): layers={m['layers']} "
          f"nq={sorted(set(_QK['nq'].values()))} nk={sorted(set(_QK['nk'].values()))} "
          f"warn={_QK['warn']} -> {_qk_dir()}", file=sys.stderr, flush=True)


_GATHER: dict[tuple, torch.Tensor] = {}


# ЗАПАС И НУЛИ В ПЛОЩАДКАХ СБОРКИ (30.07). Ядра внимания читают плитками, и хвостовая плитка
# захватывает позиции ЗА пределом Sk; при выделении ВПРИТЫК (cap == need) и через torch.empty за
# концом лежит мусор, который на fp16 может оказаться NaN/Inf. Это ровно та повадка, что наблюдалась:
# ПЕРВЫЙ длинный запрос после свежего подъёма выдавал стену "!" (NaN в логитах уже на первом
# токене -- значит порча в ПРЕФИЛЛЕ), а последующие шли нормально, потому что площадка к тому
# времени вырастала вдвое и за нужным концом оказывались валидные числа прошлого запроса.
# Лечение делает целый класс отказов невозможным, а не маскирует случай: (1) НОЛИ вместо мусора --
# дочитанный ноль даёт exp(-inf)=0, то есть ровно то же, что маска; (2) ЗАПАС на плитку, чтобы
# дочитывание вообще не выходило за выделение. Цена -- одно обнуление на (пере)выделение.
_TILE_PAD = 256          # позиций запаса на голову: перекрывает любую хвостовую плитку наших ядер


def _gather_buf(Hkv: int, T: int, d: int, device):
    """Persistent staging for the paged->contiguous prefill gather, as flat pools viewed per call.

    A flat pool keeps every view contiguous (slicing a [Hkv, cap, d] buffer would not be), and
    reusing it across the 16 attention layers and the chunks of one prefill keeps the allocator out
    of a path where each buffer is hundreds of MB."""
    need = Hkv * (T + _TILE_PAD) * d
    key = (int(device.index if device.index is not None else 0),)
    buf = _GATHER.get(key)
    if buf is None or buf[0].numel() < need:
        cap = max(need, 2 * (buf[0].numel() if buf is not None else 0), 1 << 20)
        buf = (torch.zeros(cap, dtype=torch.float16, device=device),
               torch.zeros(cap, dtype=torch.float16, device=device))
        _GATHER[key] = buf
    need = Hkv * T * d
    return buf[0][:need].view(Hkv, T, d), buf[1][:need].view(Hkv, T, d)


_GATHERB: dict = {}


def _gather_buf_bytes(Hkv: int, T: int, d: int, device):
    """Постоянная площадка под ПОБАЙТОВУЮ перекладку: вдвое меньше fp16-буфера и без конверсии."""
    need = Hkv * (T + _TILE_PAD) * d
    key = (int(device.index if device.index is not None else 0),)
    buf = _GATHERB.get(key)
    if buf is None or buf[0].numel() < need:
        cap = max(need, 2 * (buf[0].numel() if buf is not None else 0), 1 << 20)
        buf = (torch.zeros(cap, dtype=torch.uint8, device=device),
               torch.zeros(cap, dtype=torch.uint8, device=device))
        _GATHERB[key] = buf
    need = Hkv * T * d
    return buf[0][:need].view(Hkv, T, d), buf[1][:need].view(Hkv, T, d)


_GATHER8: dict = {}
# Байтовый префилл: ядро ролей (fa2_sm70/csrc/volta_fwd_ws.cuh) принимает K/V БАЙТАМИ с масштабом на
# позицию и подаёт их на тензорные ядра без разворота в fp16. Пул vLLM и так байтовый (e4m3), поэтому
# разворачивать его ради ядра внимания незачем -- сборщик перекантовывает страницы прямо в int8.
# Включается FA2SM70_SHIM_I8=1; по умолчанию ВЫКЛЮЧЕНО, потому что перекантовка e4m3 -> int8 добавляет
# свою ошибку поверх e4m3, и цена/выигрыш этого решения зависят от развёртывания (см. README).
_SHIM_I8 = os.environ.get("FA2SM70_SHIM_I8", "0") == "1"


def _gather_buf8(Hkv: int, T: int, d: int, device):
    """То же, что _gather_buf, но байтовая плитка плюс масштабы: вдвое меньше памяти и трафика.

    МАСШТАБОВ РОВНО Hkv*T, А НЕ Hkv*T*d. Прежняя версия выделяла площадку масштабов той же ЁМКОСТИ,
    что и данные, но в fp32 -- то есть 4d = 1024 байта на позицию вместо 4. На боевом контексте 250K
    это ~1 ГБ впустую при 256 МБ полезных, ровно в той памяти, из-за которой чанк префилла и упирался
    в OOM рядом с int8-весами. Ёмкости считаются раздельно."""
    need, needs = Hkv * (T + _TILE_PAD) * d, Hkv * (T + _TILE_PAD)
    key = (int(device.index if device.index is not None else 0),)
    buf = _GATHER8.get(key)
    if buf is None or buf[0].numel() < need or buf[2].numel() < needs:
        cap = max(need, 2 * (buf[0].numel() if buf is not None else 0), 1 << 20)
        caps = max(needs, 2 * (buf[2].numel() if buf is not None else 0), 1 << 12)
        buf = (torch.zeros(cap, dtype=torch.int8, device=device),
               torch.zeros(cap, dtype=torch.int8, device=device),
               torch.zeros(caps, dtype=torch.float32, device=device),
               torch.zeros(caps, dtype=torch.float32, device=device))
        _GATHER8[key] = buf
    need, needs = Hkv * T * d, Hkv * T
    return (buf[0][:need].view(Hkv, T, d), buf[2][:needs].view(Hkv, T),
            buf[1][:need].view(Hkv, T, d), buf[3][:needs].view(Hkv, T))


# --- entry points ----------------------------------------------------------
def flash_attn_func(q, k, v, dropout_p: float = 0.0, softmax_scale=None,
                    causal: bool = False, window_size=(-1, -1), softcap: float = 0.0,
                    alibi_slopes=None, deterministic: bool = False,
                    return_attn_probs: bool = False, out=None):
    """Dense prefill, flash-attn layout: q [B,Sq,H,D], k/v [B,Sk,Hkv,D] (GQA native)."""
    _tr(f"flash_attn_func q={tuple(q.shape)} k={tuple(k.shape)} causal={causal}")
    # First chunk of a fresh prompt (no prefix) lands here, not in prefill_paged. No cache pointer is
    # passed on this path, so the layer comes from the latch the store just set (store precedes
    # attention within a layer); prefill_paged cross-checks that latch against the pointer.
    if _QK_DIR:
        _dump_qk_q(q, _QK["cur"], int(k.shape[1]) if k is not None else int(q.shape[1]))
    if _SPARSE_REF and causal and _sp_on(_LAY["cur"]):
        # ПЕРВЫЙ чанк свежего промпта: K/V приходят прямо (до квантования пула), как и в боевом пути.
        sc = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
        B, Sq = int(q.shape[0]), int(q.shape[1])
        Sk = int(k.shape[1])
        o = torch.empty_like(q) if out is None else out
        for b in range(B):
            o[b:b + 1] = _sp_run(q[b:b + 1], k[b].permute(1, 0, 2), v[b].permute(1, 0, 2),
                                 sc, _LAY["cur"], Sk)
        return o
    if not _on("PREFILL"):
        inc = _incumbent()
        if inc is not None:
            return inc.flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size,
                                       softcap, alibi_slopes, deterministic, return_attn_probs, out)
    o = _fa.flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
                            causal=causal, window_size=window_size, softcap=softcap,
                            alibi_slopes=alibi_slopes, return_attn_probs=return_attn_probs)
    if isinstance(o, tuple):
        o, lse = o[0], o[1:]
    else:
        lse = None
    if out is not None:
        out.copy_(o)
        o = out
    return (o, *lse) if lse else o


def flash_attn_decode_paged(q, k_cache, v_cache, block_table, seq_lens,
                            softmax_scale=None, out=None, kv_cache_dtype: str = "auto",
                            k_scale: float = 1.0, v_scale: float = 1.0):
    """Paged decode (Sq==1). q [B,H,D]; caches [num_blocks, block_size, Hkv, D]."""
    _probe_tick()
    _tr(f"decode_paged q={tuple(q.shape)} bt={tuple(block_table.shape)}")
    if _QK_DIR:
        _qk_finish("first decode step")
    if not (_on("DECODE") and _on("STORE")):   # decode without our store => zero bound => garbage
        inc = _incumbent()
        if inc is not None:
            return inc.flash_attn_decode_paged(q, k_cache, v_cache, block_table, seq_lens,
                                               softmax_scale=softmax_scale, out=out,
                                               kv_cache_dtype=kv_cache_dtype, k_scale=k_scale, v_scale=v_scale)
    if _is_e4m3_cache(kv_cache_dtype, k_cache):              # однобайтовый пул -> НАШЕ ядро
        if _kv_fmt(k_cache) == "i8":                         # наш int8 + масштаб на позицию
            return _decode_i8(q, k_cache, v_cache, block_table, seq_lens, softmax_scale, out)
        return _decode_e4m3(q, k_cache, v_cache, block_table, seq_lens, softmax_scale, out,
                            float(k_scale), float(v_scale))
    if not _is_fp16_cache(kv_cache_dtype, k_cache):
        raise NotImplementedError(f"fa2_sm70 shim: unsupported kv_cache_dtype={kv_cache_dtype}")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    q4 = q.unsqueeze(2) if q.dim() == 3 else q               # [B,H,D] -> [B,H,1,D] (view, free)
    block_size = k_cache.shape[1]
    # Skip no-op dtype casts: each `.to()` that actually converts costs a kernel launch on a path
    # whose kernel itself is only ~0.08 ms.
    sl = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
    bt32 = block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
    # HOT PATH: no device->host syncs. `float(kmax.max())`/`int(sl.max())` each cost a full sync
    # (measured: they alone turned a 0.079 ms decode into 0.316 ms). Instead: track "was the table
    # ever written" in Python, and size the split grid from the block-table capacity (an upper bound
    # is fine -- per-sequence extents come from `seqlens`, which the kernel honours).
    ck = _cache_key(k_cache)
    g = _GLOBAL.get(ck)
    if g is None:                                            # cache written behind our back -> derive once
        g = k_cache.float().norm(dim=-1).amax(dim=1).amax(dim=0)
        _GLOBAL[ck] = g
    kmax = g.view(1, -1).expand(q4.shape[0], g.shape[0]).contiguous()
    kv_cap = int(block_table.shape[1]) * int(block_size)      # static, no sync
    o, _lse = fa2_sm70.decode_paged(q4, k_cache, v_cache, bt32,
                                    int(block_size), kv_cap, scale=float(softmax_scale),
                                    k_absmax=kmax, seqlens=sl)
    o = o.to(q.dtype).reshape(q.shape)
    if out is not None:
        out.copy_(o)
        return out
    return o


_PROBE_N = 0


def _probe_tick():
    """Зонд по типам слоёв (FA2SM70_PROBE=1). Ставится ЛЕНИВО: на старте нашего пакета модуль модели
    ещё не импортирован, а к первому префиллу -- уже. Отчёт печатается раз в 128 вызовов, то есть
    примерно раз на чанк (16 слоёв полного внимания * 8 чанков)."""
    global _PROBE_N
    if not _probe.ENABLED:
        return
    _probe.install()
    _PROBE_N += 1
    if _PROBE_N % int(os.environ.get("FA2SM70_PROBE_EVERY", "128")) == 0:
        print(_probe.report(), file=sys.stderr, flush=True)


def flash_attn_prefill_paged(q, k_cache, v_cache, block_table, seq_lens,
                             softmax_scale=None, out=None, kv_cache_dtype: str = "auto",
                             k_scale: float = 1.0, v_scale: float = 1.0):
    """Prefill against the paged cache (prefix / chunked prefill), one sequence.

    q [1, q_len, H, d]; caches [num_blocks, block_size, Hkv, d]. The q_len queries are the LAST
    q_len positions of a seq_len-long sequence, so causal here means bottom-right aligned: query i
    sees cache positions 0..(seq_len - q_len + i). Our forward kernel already implements exactly
    that alignment (verified: 1e-5 against a bottom-right reference, 1.0 against top-left).

    Why we provide this at all: without it vLLM gathers the paged KV itself, and its fast CUDA
    gather is guarded by `dtype != uint8` -- true for an fp8_e4m3 pool -- so 8-bit KV silently gets
    a Python loop with a sync per block. Owning the entry lets us gather with one coalesced kernel
    straight into the BHSD layout the prefill kernel wants."""
    _probe_tick()
    _tr(f"prefill_paged q={tuple(q.shape)} kc={tuple(k_cache.shape)} bt={tuple(block_table.shape)} sl={tuple(seq_lens.shape)}")
    if not _on("PREFILL"):
        inc = _incumbent()
        if inc is not None:
            return inc.flash_attn_prefill_paged(q, k_cache, v_cache, block_table, seq_lens,
                                                softmax_scale=softmax_scale, out=out,
                                                kv_cache_dtype=kv_cache_dtype,
                                                k_scale=k_scale, v_scale=v_scale)
    if not (_is_e4m3_cache(kv_cache_dtype, k_cache) or _is_fp16_cache(kv_cache_dtype, k_cache)
            or _is_e5m2_cache(kv_cache_dtype, k_cache)):
        raise NotImplementedError(f"fa2_sm70 shim: unsupported kv_cache_dtype={kv_cache_dtype}")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    T = int(seq_lens.reshape(-1)[0])                 # one sync per layer per sequence; prefill, not decode
    Hkv, d = k_cache.shape[2], k_cache.shape[3]
    if _qk_on():                                     # every chunk after the first (see _QK_DIR block)
        _prev = _QK["cur"]
        _lay = _qk_layer(k_cache)
        if _prev != _lay and len(_QK["warn"]) < 8:
            _QK["warn"].append(f"layer latch mismatch: store={_prev} cache={_lay}")
        _dump_qk_q(q, _lay, T)

    if _SPARSE_REF and _sp_on(_LAY["cur"]):
        # Собираем и ДЕКВАНТУЕМ пул тем же ядром, что и быстрый путь, -- так в разнице остаётся
        # только разреженность, а не ещё и другое представление KV.
        # ЭТО ПЯТЫЙ ЧИТАТЕЛЬ ПУЛА, И ОН ТОЖЕ ОБЯЗАН ЗНАТЬ ФОРМАТ. Измерительный путь, читающий
        # int8-байты как e4m3, дал бы правдоподобные, но ложные числа -- худший вид отказа в ветке,
        # которая как раз про точность. Поэтому сборщик выбирается по признаку формата.
        kb, vb = _gather_buf(Hkv, T, d, q.device)
        if _kv_fmt(k_cache) == "i8" and k_cache.element_size() == 1:
            ks_tab, vs_tab = _require_i8(k_cache, "sparse-ref")
            fa2_sm70._ext.decode_ext().gather_paged_kv_i8_pool_f16(
                k_cache, v_cache, ks_tab, vs_tab, block_table.reshape(-1), T, kb, vb)
        else:
            fa2_sm70._ext.decode_ext().gather_paged_kv(
                k_cache, v_cache, block_table.reshape(-1), T, float(k_scale), float(v_scale), kb, vb)
        o = _sp_run(q, kb, vb, softmax_scale, _LAY["cur"], T)
        if out is not None:
            out.copy_(o)
            return out
        return o

    # int8-ПУЛ: САМЫЙ КОРОТКИЙ МАРШРУТ, КОТОРЫЙ ЕЩЁ И ТОЧНЕЕ ВСЕХ ОСТАЛЬНЫХ.
    # Пул уже держит ровно тот формат, который ядро ролей (volta_fwd_ws.cuh) подаёт на тензорные ядра:
    # int8 + масштаб на позицию. Поэтому сборка -- ПЕРЕКЛАДКА (байты как есть, масштаб из побочной
    # таблицы в [Hkv,T]), а не конверсия: ни разворота в fp16, ни второго квантования, потерь РОВНО
    # НОЛЬ. Ровно тот же маршрут поверх e4m3-пула стоил 12.6-17.9x точности -- разница вся в том, что
    # квантование теперь ОДНО, на записи. Проверка формата -- ДО всякой работы.
    if _kv_fmt(k_cache) == "i8" and k_cache.element_size() == 1:
        ks_tab, vs_tab = _require_i8(k_cache, "prefill")
        if _PROF_PREFILL:
            import time as _t
            torch.cuda.synchronize(); _t0 = _t.perf_counter()
        if _KV_I8_BYTES and d == 256 and T >= int(q.shape[1]):
            kq, ks, vq, vs = _gather_buf8(Hkv, T, d, q.device)
            fa2_sm70._ext.decode_ext().gather_paged_kv_i8_pool(
                k_cache, v_cache, ks_tab, vs_tab, block_table.reshape(-1), T, kq, ks, vq, vs)
            if _PROF_PREFILL:
                torch.cuda.synchronize(); _t1 = _t.perf_counter()
            qc = q if q.is_contiguous() else q.contiguous()
            o, _lse = fa2_sm70._ext.prefill_ext().attn_fwd_volta_i8(
                qc, kq.unsqueeze(0), ks.unsqueeze(0), vq.unsqueeze(0), vs.unsqueeze(0),
                float(softmax_scale), True, _WS_NOSC)
            _say_once(f"prefill_paged=int8 pool -> byte gather -> role kernel "
                      f"(volta_fwd_ws, nosc={_WS_NOSC})")
        else:
            # Формы, которых ядро ролей не берёт (d != 256), И маршрут FA2SM70_KV_I8_BYTES=0:
            # разворачиваем в fp16 ТОЧНО (q*s) и идём обычным fp16-ядром. Формат по-прежнему ОДИН --
            # никакой тихой подмены, точность та же (разворот точен), меняется только цена.
            kb, vb = _gather_buf(Hkv, T, d, q.device)
            fa2_sm70._ext.decode_ext().gather_paged_kv_i8_pool_f16(
                k_cache, v_cache, ks_tab, vs_tab, block_table.reshape(-1), T, kb, vb)
            if _PROF_PREFILL:
                torch.cuda.synchronize(); _t1 = _t.perf_counter()
            qc = q if q.is_contiguous() else q.contiguous()
            o, _lse = fa2_sm70._ext.prefill_ext().attn_fwd_qbshd(
                qc, kb.unsqueeze(0), vb.unsqueeze(0), float(softmax_scale), True)
            _say_once("prefill_paged=int8 pool -> exact fp16 dequant -> fp16 kernel")
        if _PROF_PREFILL:
            torch.cuda.synchronize(); _t2 = _t.perf_counter()
            _PP["n"] += 1; _PP["gather"] += _t1 - _t0; _PP["attn"] += _t2 - _t1
            _PP["tok"] += int(q.shape[1])
            if _PP["n"] % 64 == 0: _pp_report()
        _tr(f"prefill_paged(i8 pool) done T={T}")
        if out is not None:
            out.copy_(o)
            return out
        return o

    # e5m2: САМЫЙ КОРОТКИЙ ИЗ ВОЗМОЖНЫХ МАРШРУТОВ. Сборка -- побайтовая перекладка страниц (ни
    # конверсии, ни масштабов, ни свёртки), внимание -- ядро ролей с разворотом одной перестановкой
    # байт. Потери точности НЕТ ВОВСЕ, в отличие от int8-маршрута ниже, который поверх e4m3-пула
    # квантует второй раз. Условия те же: d=256 и нижне-правая маска.
    if _is_e5m2_cache(kv_cache_dtype, k_cache) and d == 256 and T >= int(q.shape[1]):
        kb8, vb8 = _gather_buf_bytes(Hkv, T, d, q.device)
        if _PROF_PREFILL:
            import time as _t
            torch.cuda.synchronize(); _t0 = _t.perf_counter()
        fa2_sm70._ext.decode_ext().gather_paged_kv_bytes(k_cache, v_cache, block_table.reshape(-1), T, kb8, vb8)
        if _PROF_PREFILL:
            torch.cuda.synchronize(); _t1 = _t.perf_counter()
        qc = q if q.is_contiguous() else q.contiguous()
        o, _lse = fa2_sm70._ext.prefill_ext().attn_fwd_volta_e5m2(
            qc, kb8.unsqueeze(0), vb8.unsqueeze(0), float(softmax_scale), True)
        if _PROF_PREFILL:
            torch.cuda.synchronize(); _t2 = _t.perf_counter()
            _PP["n"] += 1; _PP["gather"] += _t1 - _t0; _PP["attn"] += _t2 - _t1
            _PP["tok"] += int(q.shape[1])
            if _PP["n"] % 64 == 0: _pp_report()
        _tr(f"prefill_paged(e5m2) done T={T}")
        if out is not None:
            out.copy_(o); return out
        return o

    # БАЙТОВЫЙ МАРШРУТ. Ядро ролей есть только при d=256 и требует Sk >= Sq (нижне-правая маска), то
    # есть ровно случай чанкового префилла. Всё остальное идёт прежним путём -- без тихой подмены.
    if _SHIM_I8 and d == 256 and T >= int(q.shape[1]):
        kq, ks, vq, vs = _gather_buf8(Hkv, T, d, q.device)
        if _PROF_PREFILL:
            import time as _t
            torch.cuda.synchronize(); _t0 = _t.perf_counter()
        fa2_sm70._ext.decode_ext().gather_paged_kv_i8(
            k_cache, v_cache, block_table.reshape(-1), T, float(k_scale), float(v_scale),
            kq, ks, vq, vs)
        if _PROF_PREFILL:
            torch.cuda.synchronize(); _t1 = _t.perf_counter()
        qc = q if q.is_contiguous() else q.contiguous()
        o, _lse = fa2_sm70._ext.prefill_ext().attn_fwd_volta_i8(
            qc, kq.unsqueeze(0), ks.unsqueeze(0), vq.unsqueeze(0), vs.unsqueeze(0),
            float(softmax_scale), True, _WS_NOSC)
        # У ЭТОЙ ВЕТКИ НЕ БЫЛО СТРОКИ МАРШРУТА -- и опись пути по ядрам молчала о префилле, хотя
        # ядро работало. Дыра ровно в проверяемости: путь исполнялся, а доказать это было нечем.
        _say_once(f"prefill_bytes=byte gather (k_scale/v_scale) -> role kernel "
                  f"(volta_fwd_ws, nosc={_WS_NOSC})")
        if _PROF_PREFILL:
            torch.cuda.synchronize(); _t2 = _t.perf_counter()
            _PP["n"] += 1; _PP["gather"] += _t1 - _t0; _PP["attn"] += _t2 - _t1
            _PP["tok"] += int(q.shape[1])
            if _PP["n"] % 64 == 0: _pp_report()
        _tr(f"prefill_paged(i8) done T={T}")
        if out is not None:
            out.copy_(o)
            return out
        return o

    kb, vb = _gather_buf(Hkv, T, d, q.device)
    if _PROF_PREFILL:
        import time as _t
        torch.cuda.synchronize(); _t0 = _t.perf_counter()
    fa2_sm70._ext.decode_ext().gather_paged_kv(
        k_cache, v_cache, block_table.reshape(-1), T, float(k_scale), float(v_scale), kb, vb)
    if _PROF_PREFILL:
        torch.cuda.synchronize(); _t1 = _t.perf_counter()
    # BSHD in AND out: vLLM hands us q as [1, q_len, H, d] and wants the output the same way, so the
    # kernel reads and writes that layout directly via strides. The old path passed q.transpose(1,2),
    # which the kernel's Q.contiguous() then materialised -- 400 MB for a 32K chunk, pure overhead in
    # bandwidth and, worse, in PEAK memory: that copy is what made larger prefill chunks OOM beside
    # 17.7 GB of int8 weights, and chunk size is the knob that controls how often the whole prefix is
    # re-touched (measured: the per-chunk prefix re-touch is what bends prefill above ~65K).
    qc = q if q.is_contiguous() else q.contiguous()
    o, _lse = fa2_sm70._ext.prefill_ext().attn_fwd_qbshd(
        qc, kb.unsqueeze(0), vb.unsqueeze(0), float(softmax_scale), True)
    if _PROF_PREFILL:
        torch.cuda.synchronize(); _t2 = _t.perf_counter()
        _PP["n"] += 1; _PP["gather"] += _t1 - _t0; _PP["attn"] += _t2 - _t1
        _PP["tok"] += int(q.shape[1])
        if _PP["n"] % 64 == 0: _pp_report()
    _dump_q(q, k_cache, T)
    _tr(f"prefill_paged done T={T}")
    if out is not None:
        out.copy_(o)
        return out
    return o


def reshape_and_cache_fp8(key, value, key_cache, value_cache, slot_mapping,
                          kv_cache_dtype: str, k_scale: float = 1.0, v_scale: float = 1.0,
                          allow_i8: bool = True):
    """Write new K/V into the paged cache AND fold them into the incremental ||k|| bound.

    The bound update is what makes our paged decode fast (see the module docstring);
    it is O(new tokens), not O(context)."""
    _tr(f"store key={tuple(key.shape)} kc={tuple(key_cache.shape)} dtype={kv_cache_dtype}")
    _dump_kv(key, value, key_cache)
    if _QK_DIR:
        _dump_qk_kv(key, value, key_cache, k_scale, v_scale)
    if _SPARSE_REF or _LOGIT_DUMP:
        lay = _lay_latch(key_cache)
        # НЕЗАВИСИМАЯ сверка нумерации: калибровочный масштаб KV у каждого слоя свой, поэтому
        # постоянство k_scale при данном индексе доказывает, что латч не съехал.
        prev = _LAY["scale"].get(lay)
        if prev is None:
            _LAY["scale"][lay] = float(k_scale)
        elif abs(prev - float(k_scale)) > 1e-12 and len(_LAY["warn"]) < 8:
            _LAY["warn"].append(f"layer {lay}: k_scale {prev} -> {k_scale}")
            print(f"[fa2_sm70] LAYER LATCH WARNING {_LAY['warn'][-1]}", file=sys.stderr, flush=True)
        _install_logit_hook()
    if _is_fp16_cache(kv_cache_dtype, key_cache):
        _update_absmax(key, key_cache, slot_mapping)
        bs = key_cache.shape[1]
        slot = slot_mapping.long()
        blk, off = slot // bs, slot % bs
        key_cache[blk, off] = key.to(key_cache.dtype).reshape(-1, key_cache.shape[2], key_cache.shape[3])
        value_cache[blk, off] = value.to(value_cache.dtype).reshape(-1, value_cache.shape[2], value_cache.shape[3])
        return
    # OURS end to end: ONE kernel quantises K/V into the engine's (strided) e4m3 pool AND folds the new
    # tokens into the running ||k_raw|| bound. Before this, the store was delegated to the incumbent and
    # the bound cost four torch ops per layer per token; now there is no incumbent anywhere in the path,
    # so what a measurement attributes to us is unambiguously ours.
    if _is_e4m3_cache(kv_cache_dtype, key_cache):
        # ФОРМАТ ВЫБИРАЕТСЯ ЗДЕСЬ И ОТМЕЧАЕТСЯ ЗДЕСЬ. Байты однобайтового пула наши целиком, поэтому
        # при FA2SM70_KV_I8=1 в них идёт int8 с масштабом на позицию, а не e4m3. Отметка формата --
        # то, по чему сборка, префилл и декод потом ОТКАЖУТСЯ читать пул иначе.
        ck = _cache_key(key_cache)
        # allow_i8=False -- слой, чьё ВНИМАНИЕ идёт НЕ по нашему пути (у бэкенда
        # _supports_flash_v100_path() ложно, например из-за скользящего окна). Формат int8 в байтах
        # e4m3 законен РОВНО ПОКА все читатели наши: чужой читатель разберёт те же байты как e4m3 и
        # выдаст мусор. На Gemma4 таких слоёв 40 из 48 (скользящее окно 1024), поэтому здесь стоит
        # вентиль, а не доверие. Пул остаётся СМЕШАННЫМ, и это уже поддержано: формат отмечается
        # НА КАЖДЫЙ кэш (_FMT[ck]), и сборка/префилл/декод читают именно отметку.
        if _KV_I8 and allow_i8:
            ks, vs = _scale_tabs(key_cache)
            fa2_sm70._ext.decode_ext().reshape_and_cache_i8(
                key, value, key_cache, value_cache, slot_mapping, ks, vs)
            _FMT[ck] = "i8"
            _say_once("store=int8+per-position-scale (FA2SM70_KV_I8=1)")
            return
        fa2_sm70._ext.decode_ext().reshape_and_cache_e4m3(
            key, value, key_cache, value_cache, slot_mapping,
            float(k_scale), float(v_scale), _bound8(key_cache), _E4M3_MARGIN)
        _FMT[ck] = "e4m3"
        return
    raise NotImplementedError(f"fa2_sm70 shim: unsupported kv_cache_dtype={kv_cache_dtype} "
                              f"(cache dtype {key_cache.dtype}, {key_cache.element_size()} B/elem)")
