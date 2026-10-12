# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# JIT build/load of the three CUDA extensions (compiled for sm_70 on first use, cached):
#   - prefill:  cutlass FMHA forward   (needs the bundled cutlass headers in ../fa2_src)
#   - backward: cutlass FMHA backward  (same headers; separate pybind module)
#   - decode:   split-KV flash-decoding (self-contained)
# A one-time preflight checks the finicky bits (CUDA toolkit, host-compiler/nvcc compatibility, arch)
# and, on a build failure, prints an ACTIONABLE diagnostic + the known-good recipe rather than a raw
# nvcc dump. Pinned/tested toolchain: CUDA 12.x (nvcc with sm_70), gcc <= 14 (CUDA 12.8 rejects newer),
# torch 2.5-2.7 (+cu12x, built with sm_70). See README / research for the reference environment.
import os
import subprocess
import warnings
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "csrc")
_ROOT = os.path.dirname(_HERE)                       # solutions/fa2_sm70_cutlass_grade
_FA2SRC = os.path.join(_ROOT, "fa2_src")
_INC = [os.path.join(_FA2SRC, "cutlass", "include"),
        os.path.join(_FA2SRC, "cutlass", "tools", "util", "include"),
        os.path.join(_FA2SRC, "fmha_kernel")]
_BUILD = os.environ.get("FA2SM70_BUILD_DIR", os.path.join(_HERE, "_build"))
# Shared nvcc flags. `-DHAS_PYTORCH` enables at::PhiloxCudaState (dropout); the cutlass path also needs
# relaxed constexpr and the half-operator guard. decode is self-contained and takes only the base set.
_CUTLASS_FLAGS = ["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
                  "-D__CUDA_NO_HALF_OPERATORS__", "-DHAS_PYTORCH"]
# Optional extra nvcc flags for experiments (e.g. FA2SM70_EXTRA_NVCC="-DFMHA_BWD_L2_PREFETCH=1").
_CUTLASS_FLAGS += [f for f in os.environ.get("FA2SM70_EXTRA_NVCC", "").split() if f]
_DECODE_FLAGS = ["-O3", "-std=c++17", "--use_fast_math"]

_prefill = None
_decode = None
_mxfp4 = None
_w8a16 = None
_tm8 = None
_w12a16 = None
_backward = None
_gdn = None
_mega = None
_sparse = None
_preflight_done = False

_RECIPE = (
    "Known-good build recipe (V100 / sm_70):\n"
    "  * a CUDA 12.x toolkit whose nvcc supports sm_70 -> set CUDA_HOME to it;\n"
    "  * pin the host compiler to one nvcc accepts (CUDA 12.8 rejects gcc > 14):\n"
    "      export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++\n"
    "  * a CUDA build of torch (torch.version.cuda set) built with sm_70 support;\n"
    "  * export TORCH_CUDA_ARCH_LIST=7.0 (done automatically here).\n"
    "  * override the build dir with FA2SM70_BUILD_DIR if the default is read-only."
)


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return ""


def _preflight():
    """One-time environment check. Raises RuntimeError with an actionable message on a hard blocker;
    warns on a soft mismatch. Cheap (a couple of `--version` calls); results are advisory for the
    build, and are re-printed if a build later fails."""
    global _preflight_done
    if _preflight_done:
        return
    _preflight_done = True
    import torch

    # 1. torch must be a CUDA build with a visible device.
    if torch.version.cuda is None:
        raise RuntimeError("fa2_sm70: this torch has no CUDA (torch.version.cuda is None). Install a "
                           "CUDA (cu12x) build of torch.\n" + _RECIPE)
    if not torch.cuda.is_available():
        raise RuntimeError("fa2_sm70: torch.cuda.is_available() is False (no GPU visible). Check "
                           "CUDA_VISIBLE_DEVICES / driver.\n" + _RECIPE)

    # 2. Device arch: the kernels are tuned for sm_70; newer archs compile but are off-target.
    try:
        cc = torch.cuda.get_device_capability(0)
        if cc != (7, 0):
            warnings.warn(f"fa2_sm70: device is sm_{cc[0]}{cc[1]}, but this package targets sm_70 (V100). "
                          f"It should still build (TORCH_CUDA_ARCH_LIST=7.0), but tuning is for Volta.")
    except Exception:
        pass

    # 3. nvcc present + version (for the gcc-compat hint).
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = os.path.join(cuda_home, "bin", "nvcc") if cuda_home else "nvcc"
    nvcc_ver = _run([nvcc, "--version"])
    if "release" not in nvcc_ver:
        warnings.warn("fa2_sm70: could not run nvcc (set CUDA_HOME to a CUDA 12.x toolkit with sm_70).\n" + _RECIPE)
        return
    cuda_major = 12
    try:
        rel = nvcc_ver.split("release")[1].split(",")[1].strip()   # e.g. "V12.8.61"
        cuda_major = int(rel.lstrip("V").split(".")[0])
    except Exception:
        pass

    # 4. Host-compiler compatibility -- the classic failure: torch picks a too-new default c++ that
    #    nvcc rejects. CUDA 12.8 supports gcc up to 14; earlier 12.x up to 13.
    if not os.environ.get("CC") and not os.environ.get("CXX"):
        cxx_ver = _run(["c++", "--version"]) or _run(["g++", "--version"])
        gmaj = None
        for tok in cxx_ver.replace("(", " ").replace(")", " ").split():
            if tok.count(".") >= 1 and tok.split(".")[0].isdigit():
                gmaj = int(tok.split(".")[0]); break
        max_gcc = 14 if cuda_major >= 12 else 12
        if gmaj is not None and gmaj > max_gcc:
            warnings.warn(f"fa2_sm70: default host compiler is gcc {gmaj}, which nvcc (CUDA {cuda_major}.x) "
                          f"likely rejects (max ~{max_gcc}). Pin it:\n"
                          f"  export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++\n" + _RECIPE)

    # 5. Bundled cutlass headers must be present (needed by prefill/backward).
    if not os.path.isdir(os.path.join(_FA2SRC, "cutlass", "include")):
        raise RuntimeError(f"fa2_sm70: bundled cutlass headers missing at {_FA2SRC}/cutlass/include "
                           f"(the package checkout is incomplete).")


def _common():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")   # Volta / V100
    os.makedirs(_BUILD, exist_ok=True)
    _preflight()


def _build(name, source, flags, inc=None, ldflags=None):
    """load() wrapped with an actionable diagnostic: on failure, surface the likely cause + recipe
    instead of only the raw compiler error.

    Each extension builds in its OWN subdirectory (_BUILD/<name>/). torch's JIT writes a single
    `build.ninja` per load into build_directory; a shared directory makes the three extensions
    (prefill/decode/backward) overwrite each other's build.ninja and .ninja_log, which triggers
    spurious ~8-10min full recompiles whenever a fresh process loads a different extension than the
    last one did. Per-extension dirs make each load hit its own cached .so (no cross-ext thrash)."""
    bdir = os.path.join(_BUILD, name)
    # ВНУТРИ try, А НЕ ДО НЕГО. Создание каталога -- ПЕРВОЕ, что здесь падает, и падало оно голым
    # `PermissionError`, минуя всю подсказку ниже (включая ветку "build dir not writable", ради
    # которой она и писалась). Замерено на себе 02.08.2026: один прогон профилировщика под `sudo`
    # создал `_build/` от root -- и приёмочный харнесс отрапортовал 22 набора FAIL, ни один из
    # которых не имел отношения к правке. Диагноз стоил трёх шагов вместо нуля.
    try:
        os.makedirs(bdir, exist_ok=True)
        # ФАЛЬСИФИКАТОРЫ СНЯТИЕМ ФАЗЫ -- через окружение, а не правкой исходника: снятая фаза
        # даёт заведомо неверный ответ, и путь к ней обязан быть виден в команде запуска.
        доп = os.environ.get("FA2SM70_NVCC_EXTRA", "").split()
        return load(name=name, sources=[os.path.join(_CSRC, source)], build_directory=bdir,
                    extra_include_paths=inc, extra_cuda_cflags=list(flags) + доп,
                    extra_ldflags=ldflags)
    except Exception as e:
        msg = str(e).lower()
        hint = ""
        if "unsupported" in msg and ("gnu" in msg or "gcc" in msg or "compiler" in msg):
            hint = "\n>>> Likely cause: host gcc too new for nvcc. Set CC=/usr/bin/gcc CXX=/usr/bin/g++."
        elif "nvcc" in msg and "not found" in msg:
            hint = "\n>>> Likely cause: nvcc not on PATH. Set CUDA_HOME to a CUDA 12.x toolkit."
        elif "sm_70" in msg or "compute_70" in msg or "architecture" in msg:
            hint = "\n>>> Likely cause: toolchain lacks sm_70. Use a CUDA/torch built with Volta support."
        elif "permission" in msg or "read-only" in msg:
            hint = "\n>>> Likely cause: build dir not writable. Set FA2SM70_BUILD_DIR to a writable path."
        raise RuntimeError(f"fa2_sm70: failed to build '{name}'.{hint}\n{_RECIPE}\n\n--- original error ---\n{e}")


def prefill_ext():
    global _prefill
    if _prefill is None:
        _common()
        # [ЗАНЯТОСТЬ] Цель варпов на SM для forward. Штатные 12 (cutlass для pre-Ampere) дают 3 блока
        # и ~168 регистров; 16 дают 4 блока и ~128 регистров. Перебирается FA2SM70_FWD_WARPS_SM.
        _pf = list(_CUTLASS_FLAGS) + [f"-DFMHA_FWD_WARPS_SM={int(os.environ.get('FA2SM70_FWD_WARPS_SM', 12))}"]
        # [ЗАМЕР ПО ФАЗАМ] FA2SM70_PHASE=1 включает счётчики clock64 в ядре ролей. Это ОТДЕЛЬНАЯ
        # сборка: clock64 и атомарные сложения сами стоят время, поэтому сравнивать её с боевой по
        # секундомеру НЕЛЬЗЯ -- читаются только ПРОПОРЦИИ внутри неё. Класть её надо в свой
        # FA2SM70_BUILD_DIR, иначе она затрёт боевую .so под тем же именем.
        if os.environ.get("FA2SM70_PHASE", "0") == "1":
            _pf = _pf + ["-DFA2_PHASE=1"]
        # [ЧИСЛО ВЕЗУЩИХ ВАРПОВ -- ЭТО ПОТОЛОК РЕГИСТРОВ, А НЕ ТОЛЬКО ПОДАЧА]
        # __launch_bounds__((WSPROD + 8) * 32, 1) задаёт, сколько регистров вправе взять НИТЬ:
        # 65536/384 = 170 при четырёх везущих, 204 при двух, 227 при одном. Ядру с широким
        # накопителем (расщепление по ключам берёт 168 из 170) это и есть стена, а везущие по
        # замеру фаз простаивают 45.1 % времени. Сборка отдельная -- макрос компиляционный.
        if os.environ.get("FA2SM70_WSPROD"):
            _pf = _pf + [f"-DFA2_WSPROD={int(os.environ['FA2SM70_WSPROD'])}"]
        _prefill = _build("fa2sm70_prefill", "attn_fwd_cutlass.cu", _pf, _INC)
    return _prefill


_gdn_head = None


def gdn_head_ext():
    """Голова цепочки GDN одним пуском (записка 25 §108): l2norm(q,k) + cumsum(g)."""
    global _gdn_head
    if _gdn_head is None:
        _gdn_head = _build("fa2sm70_gdn_head", "gdn_head.cu", _DECODE_FLAGS)
    return _gdn_head


def decode_ext():
    global _decode
    if _decode is None:
        _common()
        # FA2SM70_DEC_PF=1 -> cross-iteration KV prefetch in the split decode (A/B knob, see the
        # kernel comment). Off by default until it is measured to pay on this machine.
        _pf = ["-DFA2SM70_DEC_PF=1"] if os.environ.get("FA2SM70_DEC_PF", "0") == "1" else []
        for _v, _d in (("FA2SM70_DEC_UU", None), ("FA2SM70_DEC_VK", None), ("FA2SM70_DEC_QW", None), ("FA2SM70_DEC_BLK", None), ("FA2SM70_DEC_SAH", None)):
            _x = os.environ.get(_v)
            if _x:
                _pf = _pf + [f"-D{_v}={int(_x)}"]
        # [РЕГИСТРЫ ДЕКОДНОГО ЯДРА] Замер 01.09: боевой вариант тела расходует REG:201, то есть
        # 10 варпов на SM из 64 -- занятость 15.6 %. Именно этим объясняется, что ядро идёт в 16
        # раз медленнее предела памяти (записка 25 §80c-80d): латентность нечем скрывать, а любая
        # правка, добавляющая регистры, делает хуже. Рычаг позволяет ограничить их числом и
        # измерить размен «меньше регистров / больше spill» микробенчем за 40 секунд.
        _rr = os.environ.get("FA2SM70_DEC_MAXREG")
        if _rr:
            _pf = _pf + [f"-maxrregcount={int(_rr)}"]
        # Химерное ядро берёт WarpMma из fa2_src/fmha_kernel/gemm -- тот же примитив, что
        # у тел GEMM. Заголовки cutlass сюда по-прежнему не тянутся.
        _decode = _build("fa2sm70_decode", "split_decode.cu", _DECODE_FLAGS + _pf,
                         inc=[os.path.join(_FA2SRC, "fmha_kernel")])
    return _decode


_exch = None


def exch_ext():
    """Упаковка обмена TP в 12 разрядов. Отдельная единица трансляции: ядро крошечное, а собирать
    его вместе с вниманием значило бы ждать четверть часа ради двух ядер."""
    global _exch
    if _exch is None:
        _exch = _build("fa2sm70_exch", "exch_i12.cu", ["-O3", "-std=c++17", "--use_fast_math"])
    return _exch


def mxfp4_ext():
    """MXFP4 (OCP Microscaling) dequantisation. Separate extension: it is a weight-load-time helper,
    not part of the attention path, so nothing else pays its build."""
    global _mxfp4
    if _mxfp4 is None:
        _common()
        # NO --use_fast_math here (unlike the attention extensions): it implies -ftz=true, and MXFP4
        # legitimately produces subnormals -- an E8M0 scale of s=1 with the 0.5 magnitude lands on
        # 2^-127. Flushing it cost exactness against the reference (max|diff| = 5.877e-39 = 2^-127)
        # for no gain: this runs once at weight load, never on the token path.
        _mxfp4 = _build("fa2sm70_mxfp4", "mxfp4_dequant.cu", ["-O3", "-std=c++17"])
    return _mxfp4


_fusedng = None


def fused_norm_gemv_ext():
    """СЛИТОЕ ТЕЛО: RMSNorm + W8A16 GEMV одним запуском (скелет резидентной схемы, Ш1).

    Отдельное расширение НАМЕРЕННО: это проверка СХЕМЫ, и она не должна пересобирать
    ни внимание, ни умножение, чтобы гейт сравнивал одно с одним."""
    global _fusedng
    if _fusedng is None:
        _common()
        _fusedng = _build("fa2sm70_fused_ng", "fused_norm_gemv.cu",
                          ["-O3", "-std=c++17", "--use_fast_math"])
    return _fusedng


def w8a16_ext():
    """Fused GPTQ-int8 W8A16 GEMV. Separate extension: it is a linear-layer helper, not attention."""
    global _w8a16
    if _w8a16 is None:
        _common()
        # путь к рукописному варповому оператору: w8a16_hmma.cuh считает на ТЕНЗОРНЫХ ядрах
        _w8a16 = _build("fa2sm70_w8a16", "w8a16_gemv.cu", ["-O3", "-std=c++17", "--use_fast_math"],
                        inc=[os.path.join(_FA2SRC, "fmha_kernel")])
    return _w8a16


def tm8_ext():
    """Задача 191: наш GPTQ-int8 (v1, g=128) на двигателе turbomind s884 при больших M.
    Заголовки берутся из РАБОЧЕГО дерева форка (vendored turbomind); ни одного нового
    device-тела — только Config_U8_g + прямой пуск по идиому lm_head_top1."""
    global _tm8
    if _tm8 is None:
        _common()
        tm_root = os.environ.get(
            "FA2SM70_TM_ROOT", os.path.join(_ROOT, "third_party", "turbomind"))
        if not os.path.isdir(os.path.join(tm_root, "src", "turbomind")):
            raise RuntimeError(f"fa2_sm70: turbomind headers not found at {tm_root} "
                               f"(set FA2SM70_TM_ROOT to sm70/third_party/turbomind)")
        # torch вкалывает -D__CUDA_NO_HALF_* по умолчанию; turbomind живёт операторами half —
        # снимаем их так же, как это делает сборка самого форка.
        _tm8 = _build("fa2sm70_tm8", "tm8_gemm.cu",
                      ["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                       "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                       "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__"],
                      inc=[tm_root])
    return _tm8


def w12a16_ext():
    """W12A16: вес в 12 разрядах, потребляемый упакованным. Пара к tools/quantize_iN.py."""
    global _w12a16
    if _w12a16 is None:
        _common()
        # путь к рукописному варповому оператору: w12a16_hmma.cuh считает на ТЕНЗОРНЫХ ядрах
        _w12a16 = _build("fa2sm70_w12a16", "w12a16.cu", ["-O3", "-std=c++17", "--use_fast_math"],
                         inc=[os.path.join(_FA2SRC, "fmha_kernel")])
    return _w12a16


def mega_ext():
    """Мегаядро декода (цепь слоёв). ОТДЕЛЬНОЕ расширение -- `decode_chain` живёт в своём исходнике.

    Проверка наличия входа СРАЗУ при загрузке: сегодня выяснилось на боевом шаге, что модуль с
    нужным именем есть, а входа в нём нет (класс "идентичность по владению, а не по имени").
    """
    global _mega
    if _mega is None:
        _common()
        _mega = _build("fa2sm70_mega", "fused_mlp_body.cu",
                       ["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
                        "-D__CUDA_NO_HALF_OPERATORS__"])
        for вход in ("decode_chain", "gdn_layer", "attn_layer", "dense_gemv_norm"):
            if not hasattr(_mega, вход):
                raise RuntimeError(f"mega_ext: собран модуль БЕЗ входа '{вход}' -- не тот исходник")
    return _mega


def gdn_ext():
    """Чанковый скан gated-delta-net (Qwen3.5 linear_attention) на тензорных ядрах Volta.

    ПОЧЕМУ ОТДЕЛЬНАЯ ЕДИНИЦА ТРАНСЛЯЦИИ. Скан -- не внимание: он не нужен ни одной модели без
    линейной мешалки, а его пять шаблонных ядер компилируются заметно дольше, чем весь decode.
    Общая сборка заставила бы каждую модель платить за него временем первого запуска.

    ПОЧЕМУ БЕЗ cutlass-путей во включениях. Ядра скана самодостаточны: им нужен только наш
    volta_warp_mma.h из fa2_src/fmha_kernel/gemm. Тянуть сюда заголовки cutlass значило бы
    прибавить к сборке минуты ради ничего.
    """
    global _gdn
    if _gdn is None:
        _common()
        _gdn = _build("fa2sm70_gdn", "gdn_scan.cu",
                      ["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
                       "-D__CUDA_NO_HALF_OPERATORS__"],
                      inc=[os.path.join(_FA2SRC, "fmha_kernel", "gemm")])
    return _gdn


_conv = None


_i4 = None


def i4_ext():
    """Двухступенчатый аргмакс словаря черновика (записка 25 §238-§239).

    Отдельная единица трансляции по той же причине, что и свёртка: ядро мелкое,
    ничего тяжёлого не тянет и собирается секундами.
    """
    global _i4
    if _i4 is None:
        _common()
        _i4 = _build("fa2sm70_i4", "i4_vocab.cu",
                     ["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
                      "-D__CUDA_NO_HALF_OPERATORS__"])
    return _i4


def conv_ext():
    """СВОЯ свёртка состояния GDN (замена чужому Triton-ядру causal_conv1d_update).

    ОТДЕЛЬНАЯ ЕДИНИЦА ТРАНСЛЯЦИИ намеренно: ядро крошечное и не тянет ни
    cutlass, ни заголовки скана, поэтому собирается секунды, а не минуты, и
    правка в нём не заставляет пересобирать большие тела.
    """
    global _conv
    if _conv is None:
        _common()
        _conv = _build("fa2sm70_conv", "conv_update.cu",
                       ["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
                        "-D__CUDA_NO_HALF_OPERATORS__"])
    return _conv


_clef = None


def clef_ext():
    """Clef внутри боевой сети: поправка весов по строкам запроса + голова классификатора.

    Отдельная единица трансляции: нужна только при FA2SM70_CLEF=1 и не тянет ни cutlass, ни
    наши тела. cuBLAS вызывается напрямую из C++ (хэндл берётся у ATen, поток текущий), поэтому
    линкуемся с тем же libcublas, что уже загрузил torch, -- путь берётся из его пакета nvidia.
    """
    global _clef
    if _clef is None:
        _common()
        ld = []
        try:
            import nvidia.cublas as _nvc
            libdir = os.path.join(list(_nvc.__path__)[0], "lib")
            if os.path.exists(os.path.join(libdir, "libcublas.so.12")):
                ld = [f"-L{libdir}", "-l:libcublas.so.12", f"-Wl,-rpath,{libdir}"]
        except Exception:
            pass
        _clef = _build("fa2sm70_clef", "clef.cu",
                       ["-O3", "-std=c++17", "--use_fast_math", "-D__CUDA_NO_HALF_OPERATORS__"],
                       ldflags=ld or ["-lcublas"])
    return _clef


def backward_ext():
    global _backward
    if _backward is None:
        _common()
        # Backward needs the bundled cutlass headers (same as prefill). Separate module (own pybind).
        # [R7 default] FMHA_BWD_R7 = the store-side atomic dQ-fold (measured's Arrangement B): for d<=128 it removes
        # the 255-reg/spill of the lock-RMW dQ split-reduction (238-254 reg, no spill) and beats/ties torch on all
        # S (0.91-1.05x vs the pre-R7 uniform 0.78-0.85x loss). d-gated on kOutputInRF so d>128 keeps the correct
        # lock-RMW path (all-d exact). Set FA2SM70_BWD_R7=0 to A/B against the pre-R7 kernel.
        _bwd_flags = list(_CUTLASS_FLAGS)
        if os.environ.get("FA2SM70_BWD_R7", "1") != "0" and "-DFMHA_BWD_R7" not in _bwd_flags:
            _bwd_flags.append("-DFMHA_BWD_R7")
        # [COAL_FO default] fragment-order coalesced dQ fold (measured/22 queue-dynamics line): the RED emits
        # in fragment order (addr=warp*kElem*32+idx*32+lane) -> 4 transaction-granules/request instead of the
        # scattered 16 -> 4x less L2 write traffic on the dQ fold, EVERY tile/split. A cheap perm-transpose convert
        # pass reconstructs dQ[query,d]. Gated (p.fo_dq) to the verified fast path: d==128 && kBI==128 && Sq%128==0
        # && BHSD (the fragment map + Sq*d workspace are pinned for that tile; d!=128/partial-Sq fall back to the
        # exact scattered emit). Beats torch's SDPA-efficient backward on the whole d=128 spectrum: 1.16-1.31x
        # isolated (7/7 shapes B*H=8..64, S=512..4096, min-of-N). NB the tiny single-wave B1H8-S512 (~0.14ms) needs
        # adequate warmup (>=40 iters) to measure honestly -- an under-warmed loop reads a false ~0.97x. Exact
        # (3.7e-4). Off-switch FA2SM70_BWD_FO=0.
        if os.environ.get("FA2SM70_BWD_FO", "1") != "0" and "-DFMHA_BWD_COAL_FO" not in _bwd_flags:
            _bwd_flags.append("-DFMHA_BWD_COAL_FO")
        # [ACCUM16 probe] fp16 dK/dV partial accumulators -- halves the gmem round-trip that ncu shows is
        # ~90% of this kernel's DRAM traffic at d>128. OFF by default until the gradient error says it is
        # safe. FA2SM70_BWD_ACCUM16=1 to A/B.
        if os.environ.get("FA2SM70_BWD_ACCUM16", "0") == "1" and "-DFMHA_BWD_ACCUM16" not in _bwd_flags:
            _bwd_flags.append("-DFMHA_BWD_ACCUM16")
        # [RF256 probe] force the register-resident accumulator at d=256 (see kernel_backward.h). This
        # removes the dK/dV gmem round-trip outright rather than shrinking it; capacity says it may fit
        # at d=256 (128 of 255 registers) though never at d=512.
        if os.environ.get("FA2SM70_BWD_RF256", "0") == "1" and "-DFMHA_BWD_RF256" not in _bwd_flags:
            _bwd_flags.append("-DFMHA_BWD_RF256")
        _backward = _build("fa2sm70_backward", "attn_bwd_cutlass.cu", _bwd_flags, _INC)
    return _backward

def sparse_ext():
    """МЕХАНИЗМ АДАПТИВНОЙ РАЗРЕЖЕННОСТИ (attn_fwd_sparse.cu). Отдельная единица трансляции: тянет
    только наш варповый оператор, без cutlass, поэтому собирается секунды, а не минуты -- при
    итерациях по механизму это решает. Внутри и ЗАМОРОЖЕННАЯ копия отгруженного ядра, чтобы A/B
    «механизм против его отсутствия» шёл в одном бинарнике, на одной обвязке, без пересборки."""
    global _sparse
    if _sparse is None:
        _common()
        _sparse = _build("fa2sm70_sparse", "attn_fwd_sparse.cu",
                         ["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
                          "-D__CUDA_NO_HALF_OPERATORS__", "-Xptxas", "-v"],
                         inc=[os.path.join(_FA2SRC, "fmha_kernel")])
    return _sparse
