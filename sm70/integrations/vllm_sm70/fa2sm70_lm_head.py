"""СЛОВАРНАЯ ПРОЕКЦИЯ (`lm_head`) НА НАШЕМ ЯДРЕ ВМЕСТО turbomind.

ЧТО ЭТО ЗАМЕНЯЕТ. Разбор ленты (docs/SPEC_TURBOMIND.md) назвал единственное крупное чужое тело
шага декода: `_C::sm70_f16_gemm_out` -> `turbomind::gemm::gemm_kernel`, K=5120, N=124160 на ранг,
**1463 мкс/токен = 4.79 % ядерного времени**. Оно стоит на **98.9 % замеренной крыши полосы**, то
есть более быстрого ядра в fp16 НЕ СУЩЕСТВУЕТ, и единственный рычаг -- меньше байт: `lm_head` был
ПОСЛЕДНИМ крупным fp16-весом в сети, где всё остальное уже int8.

ЛЕСТНИЦА РАЗРЯДНОСТИ, ЗАМЕРЕНО НА БОЕВОМ ВЕСЕ (benchmarks/bench_lm_head.py, M=1, карта 0):

    путь                relL2 к fp32   top-1   top-8      мс   % крыши   к turbomind
    turbomind fp16        2.07e-04       --      --    1.477    98.1 %      1.000
    torch fp16 плотно     2.07e-04     64/64   0.992   1.939    74.7 %      0.762
    НАШЕ int12            4.87e-04     64/64   1.000   1.301    84.4 %      1.135
    НАШЕ int8             6.20e-03     63/64   0.984   0.790    93.8 %      1.869

int12 ОБОШЁЛ плотный fp16 по порядку (512/512 против 508/512) -- тот же механизм, что замерен на
весах внимания: fp16 тратит экспоненту на каждом весе, хотя диапазон уже вынесен в масштаб группы.

СКВОЗНЯКОМ (256 токенов, 5 раундов, размах ±0.4 %):
    режим 0 turbomind 33.95 ток/с | режим 3 int12 33.89 (= база) | режим 2 34.59 (+1.9 %)
    режим 1 34.94 (+2.9 %)
int12 сквозняком тонет в шуме: тело -- 4.79 % ядерного времени, x1.135 на нём даёт 0.56 %.

РЫЧАГ `FA2SM70_LMH`:
    0 (умолчание) -- выключено, идёт turbomind;
    1 -- наше int8, fp16-вес ОСВОБОЖДАЕТСЯ (минус 620 МБ на ранг). Быстрее всех, но порядок
         отбора отличается от чужого на 0.2-0.7 % токенов;
    2 -- наше int8 + ТОЧНЫЙ ДОВОДЧИК верхушки: порядок отбора СОВПАЛ С ЧУЖИМ В ТОЧНОСТИ
         (см. ниже). fp16-вес остаётся в памяти;
    3 -- наше int12: точность выше плотного fp16, вес 963 МБ, но по времени = чужое.

ГЕЙТ ПРОЙДЕН НА БОЕВЫХ СОСТОЯНИЯХ, А НЕ НА СЛУЧАЙНЫХ. Сверка внутри слоя (FA2SM70_LMH_AUDIT=1) на
1030 скрытых состояниях декода, эталон fp32:

                        сырое int8   + доводчик   turbomind (контроль)
    ранг 0, top-1         0.9981       0.9981           0.9981
    ранг 1, top-1         0.9971       0.9971           0.9971
    истинный первый попал в нашу верхушку 512: 1.0000 (оба ранга)

Остаток 0.2-0.3 % платит И ЧУЖОЕ ТЕЛО -- это округление вывода в fp16, а не квантование. Режим 2
по этой мере НЕОТЛИЧИМ от turbomind.

ПРО ПОБИТОВУЮ СВЕРКУ ВЫВОДА: она на этом сервере НЕПРИМЕНИМА как гейт. Проверено -- два прогона
ОДНОГО сервера при T=0 дают разный текст, потому что наше `w8a16_gemv` сводит k-расщепление
атомарными. Поэтому мерой служит совпадение ПОРЯДКА логитов, а не байты ответа.
"""
from __future__ import annotations

import os
import sys

import torch

_ПУТЬ = os.environ.get("FA2SM70_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
for _p in (_ПУТЬ, os.path.join(_ПУТЬ, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

РЕЖИМ = int(os.environ.get("FA2SM70_LMH", "0"))
ВЕРХ = int(os.environ.get("FA2SM70_LMH_TOP", "512"))     # длина списка для доводчика
ГРУППА = int(os.environ.get("FA2SM70_LMH_G", "128"))
# ПЕРЕЛОМ ЗАМЕРЕН, А НЕ ВЗЯТ ИЗ ГОЛОВЫ (benchmarks/bench_lm_head.py, свод по M на форме словаря):
#   M=1 1.90   M=2 1.41   M=4 1.45   |   M=8 0.91   M=16 0.62   M=32 0.38   M=64 0.20
# то есть наше скалярное тело выигрывает ДО M=4 включительно, а с M=8 уже проигрывает. Стояло 8 --
# это отдавало чужому телу проигрышную для нас точку и забирало выигрышные наоборот.
MMAX = int(os.environ.get("FA2SM70_LMH_MMAX", "4"))
АУДИТ = os.environ.get("FA2SM70_LMH_AUDIT", "0") == "1"   # сверка с плотным fp16 на КАЖДОМ вызове
_расш = [None, None]
_счёт = {"наше": 0, "чужое": 0}
_а = {"всего": 0, "сыр1": 0, "сыр8": 0, "в512": 0, "дов1": 0, "чуж1": 0, "чуж8": 0,
      "плот": None}


def _ext():
    if _расш[0] is None:
        import fa2_sm70
        _расш[0] = fa2_sm70._ext.w8a16_ext()
    return _расш[0]


def _ext12():
    if _расш[1] is None:
        import fa2_sm70
        _расш[1] = fa2_sm70._ext.w12a16_ext()
    return _расш[1]


def _szhat12(w, G, kusok: int = 8192):
    """ПЛОТНЫЙ fp16 -> НАШ int12 (симметричный, [K/8,3,N] + масштабы [K/G,N]).

    Кусками по N: плотный fp32-словарь весит 2.54 ГБ -- вчетверо больше результата, и на карте
    с уже загруженной сетью он не помещается. Третий возврат -- None: у int12 квантование
    СИММЕТРИЧНОЕ, нулевой точки нет вовсе, и эпилог группы это один множитель.
    """
    import torch as _t
    from quantize_iN import квантовать as кв
    N = w.shape[0]
    части, масш = [], []
    for n0 in range(0, N, kusok):
        пк, мс, _, _ = кв(w[n0:min(N, n0 + kusok)].float(), 12, G)
        части.append(пк); масш.append(мс)
    q = _t.cat(части, dim=2).contiguous()
    s = _t.cat(масш, dim=1).contiguous()
    del части, масш
    _t.cuda.empty_cache()
    return q, None, s


def _это_lm_head(layer) -> bool:
    p = getattr(layer, "prefix", "")
    return bool(p) and p.rsplit(".", 1)[-1] == "lm_head"


def установить() -> bool:
    if РЕЖИМ == 0:
        return False
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe

    М = vpe.UnquantizedEmbeddingMethod
    if getattr(М, "_fa2sm70_lmh", False):
        return True
    старый_подг = М.process_weights_after_loading
    старый_прим = М.apply

    def подготовить(self, layer):
        # ЗДЕСЬ И ТОЛЬКО ЗДЕСЬ можно трогать CUDA: снимок свободной памяти движком уже сделан,
        # а компиляция графа ещё нет. Загрузка расширения на ИМПОРТЕ стоила 1.87 ГиБ и отказа
        # подъёма -- [[fa2-sm70-vetka-po-forme-zapekaetsya]], пункт 1.
        старый_подг(self, layer)
        if not _это_lm_head(layer):
            return
        w = getattr(layer, "weight", None)
        if w is None or w.dim() != 2 or w.dtype != torch.float16 or not w.is_cuda:
            return
        N, K = w.shape
        if K % ГРУППА or N % 4 or K % 4:
            print(f"[fa2_sm70 LMH] форма {N}x{K} не ложится (нужны K%{ГРУППА}==0, N%4==0) -- пропуск")
            return
        try:
            if РЕЖИМ == 3:
                qw, qz, sc = _szhat12(w, ГРУППА)
                _ext12()
            else:
                from fa2_sm70.kvant_gptq8 import kvantovat_gptq8
                qw, qz, sc = kvantovat_gptq8(w, ГРУППА)
                _ext()                               # собрать расширение ЗДЕСЬ, не на импорте
        except Exception as e:  # noqa: BLE001
            print(f"[fa2_sm70 LMH] сжатие/сборка не удались ({str(e)[:90]}) -- остаётся чужой путь")
            return
        layer._fa2_qw, layer._fa2_qz, layer._fa2_sc = qw, qz, sc
        if РЕЖИМ >= 2 or АУДИТ:
            layer._fa2_w16 = w.detach()              # доводчику И СВЕРКЕ нужен ПЛОТНЫЙ вес
        else:
            # ОСВОБОЖДЕНИЕ ЗАКОННО ТОЛЬКО ПРИ tie_word_embeddings=False: при связанных словарях
            # этот же тензор служит таблицей вложений и читается `embedding()`.
            св = getattr(getattr(layer, "quant_config", None), "tie_word_embeddings", None)
            связан = bool(getattr(layer, "_fa2_tied", False)) or св is True
            if not связан and os.environ.get("FA2SM70_LMH_FREE", "1") == "1":
                layer.weight.data = torch.empty(0, dtype=w.dtype, device=w.device)
                # И ВТОРУЮ КОПИЮ ТОЖЕ. `sm70_f16_prepare` кладёт ПЕРЕУПАКОВАННЫЙ вес рядом с
                # исходным -- то есть чужой путь держит словарь ДВАЖДЫ, 2.54 ГБ на ранг. В этом
                # режиме чужое тело не зовётся ни разу, значит копия -- чистая потеря.
                if getattr(layer, "_sm70_f16_tm_weight", None) is not None:
                    layer._sm70_f16_tm_weight = None
                    layer._sm70_f16_prepared = False
                layer._fa2_osvobozhdeno = True
        print(f"[fa2_sm70 LMH] словарная проекция {N}x{K} сжата в {'int12' if РЕЖИМ == 3 else 'int8'} "
              f"({w.numel()*2/1e6:.0f} -> {(qw.numel() + (qz.numel() if qz is not None else 0))*4/1e6 + sc.numel()*2/1e6:.0f} МБ), "
              f"режим {РЕЖИМ}"
              + (", плотный вес освобождён" if getattr(layer, "_fa2_osvobozhdeno", False) else ""))

    def применить(self, layer, x, bias=None):
        qw = getattr(layer, "_fa2_qw", None)
        if qw is None:
            return старый_прим(self, layer, x, bias)
        x2 = x.reshape(-1, x.shape[-1])
        M = x2.shape[0]
        if M > MMAX or torch.compiler.is_compiling():
            # Выше порога плотное тело выгоднее (наше связано счётом, а не памятью); внутри
            # компилируемой области не ходим, чтобы не заводить вход-целое в граф.
            # Отвод на чужое тело возможен ТОЛЬКО пока плотный вес жив. Если он освобождён,
            # считаем СВОИМ телом: при большом M оно медленнее, но такой вызов редок (движок
            # берёт логиты лишь для последнего токена последовательности -- замер по ленте:
            # 192 вызова на 192 шага, M=1..4), а цель -- чтобы в цепочке не осталось чужого.
            if not getattr(layer, "_fa2_osvobozhdeno", False):
                _счёт["чужое"] += 1
                return старый_прим(self, layer, x, bias)
            if _счёт["чужое"] == 0:
                print(f"[fa2_sm70 LMH] M={M} выше порога {MMAX}, но плотный вес освобождён -- "
                      f"считаем СВОИМ телом (медленнее, но своё)", flush=True)
            _счёт["чужое"] += 1
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if РЕЖИМ == 3:
            y = _ext12().w12a16_gemv(x2, qw, layer._fa2_sc, ГРУППА)
        else:
            y = _ext().w8a16_gemv(x2, qw, layer._fa2_qz, layer._fa2_sc, ГРУППА, True)
        w16 = getattr(layer, "_fa2_w16", None)
        if АУДИТ and w16 is not None:
            # [СВЕРКА НА БОЕВЫХ СОСТОЯНИЯХ, А НЕ НА СЛУЧАЙНЫХ]
            # Гейт по NLL на закреплённом тексте сюда НЕ ДОХОДИТ: там M=176 > порога, и логиты
            # считает чужое тело. Единственное место, где видно РЕАЛЬНОЕ скрытое состояние
            # декода, -- вот эта точка. Считаем оба ответа и сравниваем ПОРЯДОК, потому что
            # выход слоя идёт в отбор, а не в норму.
            п = (x2.float() @ w16.float().t())
            _а["всего"] += M
            _а["сыр1"] += int((y.float().argmax(-1) == п.argmax(-1)).sum())
            k = min(8, п.shape[-1])
            ai, bi = y.float().topk(k, -1).indices, п.topk(k, -1).indices
            _а["сыр8"] += sum(len(set(ai[i].tolist()) & set(bi[i].tolist())) for i in range(M))
            _а["в512"] += int(sum(
                bool(п[i].argmax().item() in set(y[i].float().topk(min(ВЕРХ, y.shape[-1])).indices.tolist()))
                for i in range(M)))
            # ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ: тот же гейт для ЧУЖОГО тела. Без него «0.998» -- число без
            # шкалы: чужое тоже отдаёт fp16, значит тоже теряет порядок на точных ничьих, и
            # сравнивать надо с ним, а не с недостижимым fp32.
            try:
                ч = старый_прим(self, layer, x, bias).reshape(M, -1)
                _а["чуж1"] += int((ч.float().argmax(-1) == п.argmax(-1)).sum())
                _а["чуж8"] += sum(len(set(ч.float().topk(k, -1).indices[i].tolist()) & set(bi[i].tolist()))
                                  for i in range(M))
            except Exception:  # noqa: BLE001
                pass
            _а["плот"] = п
        if РЕЖИМ >= 2 and w16 is not None and ВЕРХ > 0:
            # ТОЧНЫЙ ДОВОДЧИК. Верхушка списка пересчитывается плотным fp16 -- ровно тем же
            # умножением, что делает чужое тело, поэтому порядок отбора совпадает точно.
            # СВОД В fp32, А НЕ В fp16. Замер на 1030 боевых состояниях: при fp16-своде ранг 1
            # давал 0.9951 -> 0.9990, то есть доводчик снимал НЕ ВСЁ -- остаток был не от
            # квантования, а от ПОРЯДКА СУММИРОВАНИЯ у самого доводчика. В fp32 верхушка
            # становится точной, и цена этого -- 10 МБ временного на вызов.
            idx = y.topk(min(ВЕРХ, y.shape[-1]), dim=-1).indices           # [M, S]
            строки = w16.index_select(0, idx.reshape(-1)).view(M, idx.shape[1], -1)
            точн = torch.einsum("mk,msk->ms", x2.float(), строки.float())
            y = y.scatter(-1, idx, точн.to(y.dtype))
        if АУДИТ and _а.get("плот") is not None:
            п = _а.pop("плот")
            _а["дов1"] += int((y.float().argmax(-1) == п.argmax(-1)).sum())
        _счёт["наше"] += 1
        if _счёт["наше"] == 1 or (_счёт["наше"] & (_счёт["наше"] - 1)) == 0:
            хв = ""
            if АУДИТ and _а["всего"]:
                в = _а["всего"]
                хв = (f" | СВЕРКА на {в} состояниях (эталон fp32): сырое int8 top-1 "
                      f"{_а['сыр1']/в:.4f} top-8 {_а['сыр8']/(в*8):.4f} | ПОСЛЕ ДОВОДЧИКА top-1 "
                      f"{_а['дов1']/в:.4f} | ЧУЖОЕ (контроль) top-1 {_а['чуж1']/в:.4f} top-8 "
                      f"{_а['чуж8']/(в*8):.4f} | истинный первый в верхушке {ВЕРХ}: "
                      f"{_а['в512']/в:.4f}")
            print(f"[fa2_sm70 LMH] НАША словарная проекция сработала, всего {_счёт['наше']} (M={M})"
                  + хв, flush=True)
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])

    М.process_weights_after_loading = подготовить
    М.apply = применить
    М._fa2sm70_lmh = True
    print(f"[fa2_sm70 LMH] словарная проекция переведена на НАШЕ ядро (режим {РЕЖИМ}, "
          f"верхушка {ВЕРХ}, порог M={MMAX})")
    return True
