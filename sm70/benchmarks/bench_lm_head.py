"""ОЧНАЯ СТАВКА НА СЛОВАРНОЙ ПРОЕКЦИИ: НАШЕ int8-ТЕЛО ПРОТИВ turbomind fp16.

ЗАЧЕМ. Разбор ленты (docs/SPEC_TURBOMIND.md) назвал единственное крупное чужое тело шага декода:
`lm_head`, K=5120, N=124160 на ранг, **1463 мкс на токен = 4.79 % ядерного времени**, и оно стоит
на **98.9 % замеренной крыши полосы**. Значит быстрее в fp16 не бывает, и ход один -- меньше байт.

ПОРЯДОК КАК ВСЕГДА: сверка ПЕРВОЙ, потом время. Для логитов «сверка» -- это НЕ relL2: логиты идут
в отбор, поэтому мера -- СОВПАДЕНИЕ ПОРЯДКА (top-1 и множество top-8), а relL2 печатается рядом
как справка.

ВЕС БЕРЁТСЯ ИЗ БОЕВОГО СЛЕПКА, а не случайный: у случайной матрицы логиты почти вырождены, и
любой порядок «совпадает» либо «не совпадает» по шуму.

    python benchmarks/bench_lm_head.py [--rows 124160] [--m 1]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402
from fa2_sm70.kvant_gptq8 import bajt_gptq8, kvantovat_gptq8  # noqa: E402

СЛЕПОК = os.environ["MODEL_DIR"]
ПОЛОСА = 878.0     # ГБ/с, ЗАМЕРЕНО на этой карте


def взять_lm_head(строк):
    import json
    from safetensors import safe_open
    карта = json.load(open(os.path.join(СЛЕПОК, "model.safetensors.index.json")))["weight_map"]
    файл = карта["lm_head.weight"]
    with safe_open(os.path.join(СЛЕПОК, файл), framework="pt") as f:
        w = f.get_slice("lm_head.weight")[0:строк, :]
    return w


def замер(f, n=9):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    т = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); torch.cuda.synchronize()
        т.append(time.perf_counter() - t0)
    т.sort()
    return sum(т[2:-2]) / len(т[2:-2]) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=124160, help="N на ранг (словарь 248320 / TP2)")
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--g", type=int, default=128)
    a = ap.parse_args()
    N, M, G = a.rows, a.m, a.g

    dev = "cuda:0"
    torch.manual_seed(0)
    print(f"=== lm_head: N={N} K=5120 M={M}, группа {G} ===")
    w = взять_lm_head(N).to(dev).half()          # [N, K]
    K = w.shape[1]
    print(f"вес fp16: {w.numel()*2/1e6:.0f} МБ")

    qw, qz, sc = kvantovat_gptq8(w, G)
    наш_байт = bajt_gptq8(N, K, G)
    print(f"вес наш:  {наш_байт/1e6:.0f} МБ  ({наш_байт/(w.numel()*2):.3f} от fp16)")

    ext = fa2_sm70._ext.w8a16_ext()
    x = (torch.randn(M, K, device=dev) * 0.35).half()

    # --- ЭТАЛОН В fp32, ПОКУСОЧНО (плотный fp32-вес на весь словарь не влезет) ---
    y_э = torch.zeros(M, N, device=dev, dtype=torch.float32)
    шаг = 16384
    for n0 in range(0, N, шаг):
        n1 = min(N, n0 + шаг)
        y_э[:, n0:n1] = x.float() @ w[n0:n1].float().t()

    y_н = ext.w8a16_gemv(x, qw, qz, sc, G, True)     # True = условность v2, мы её и записали
    y_т = (x @ w.t())                                 # плотное fp16 -- то, что делает turbomind

    # --- ЧУЖОЕ ТЕЛО, ЕСЛИ ДОСТУПНО ---
    y_tm, tm_w, tm_k = None, None, None
    try:
        from vllm import _custom_ops as vops
        prepared = vops.sm70_f16_prepare(w)
        tm_w, tm_k = prepared[0], int(prepared[1][0].item())
        y_tm = torch.empty((M, N), dtype=torch.float16, device=dev)
        vops.sm70_f16_gemm_out(y_tm, x, tm_w, tm_k, False)
    except Exception as e:  # noqa: BLE001
        print(f"turbomind недоступен ({str(e)[:70]}) -- очной ставки по времени не будет")

    def нрм(u):
        return ((u.float() - y_э).norm() / y_э.norm().clamp_min(1e-30)).item()

    def порядок(u, k=8):
        сов1 = (u.float().argmax(-1) == y_э.argmax(-1)).float().mean().item()
        a = u.float().topk(k, -1).indices
        b = y_э.topk(k, -1).indices
        совk = sum(len(set(a[i].tolist()) & set(b[i].tolist())) for i in range(M)) / (M * k)
        return сов1, совk

    print(f"\n{'путь':<22s} {'relL2 к fp32':>13s} {'top-1':>8s} {'top-8':>8s}")
    for имя, y in (("torch fp16 (плотно)", y_т), ("НАШЕ int8", y_н),
                   *(( ("turbomind fp16", y_tm),) if y_tm is not None else ())):
        с1, с8 = порядок(y)
        print(f"{имя:<22s} {нрм(y):>13.3e} {с1:>8.3f} {с8:>8.3f}")

    # СТАТИСТИКА ПО МНОГИМ ВЕКТОРАМ + ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ.
    # [ЗАЧЕМ КОНТРОЛЬ] На случайном x логиты почти вырождены: разрыв между первым и вторым
    # сравним с округлением fp16, поэтому ТОЧНЫЙ fp16-путь тоже расходится с fp32-эталоном.
    # Без замера этого расхождения любой промах int8 читается как его вина. Известно-верный
    # вариант обязан пройти гейт ПЕРВЫМ -- [[fa2-sm70-polozhitelnyj-kontrol-etalona]].
    ПРОБ = 64
    сч = {"НАШЕ int8": [0, 0], "fp16 (контроль)": [0, 0]}
    for _ in range(ПРОБ):
        xx = (torch.randn(1, K, device=dev) * 0.35).half()
        ye = torch.zeros(1, N, device=dev, dtype=torch.float32)
        for n0 in range(0, N, шаг):
            n1 = min(N, n0 + шаг)
            ye[:, n0:n1] = xx.float() @ w[n0:n1].float().t()
        э8 = set(ye.topk(8, -1).indices[0].tolist())
        for имя, y in (("НАШЕ int8", ext.w8a16_gemv(xx, qw, qz, sc, G, True).float()),
                       ("fp16 (контроль)", (xx @ w.t()).float())):
            сч[имя][0] += int(y.argmax(-1).item() == ye.argmax(-1).item())
            сч[имя][1] += len(set(y.topk(8, -1).indices[0].tolist()) & э8)
    print(f"\nна {ПРОБ} случайных векторах (логиты там почти вырождены -- гейт ЗАВЫШЕННО строгий):")
    for имя, (с1, с8) in сч.items():
        print(f"  {имя:<18s} top-1 {с1}/{ПРОБ}   top-8 {с8}/{ПРОБ*8} = {с8/(ПРОБ*8):.3f}")

    # --- ЛЕСТНИЦА РАЗРЯДНОСТИ: int12 КАК СРЕДНЯЯ СТУПЕНЬ ---
    # Вопрос владельца: «если точности не хватает, можно int10/int12». Ступень между fp16 и int8
    # у нас УЖЕ отгружена (симметричное int12, ядро w12a16_gemv), поэтому она измеряется, а не
    # обсуждается. int10 в лестницу не берём: нечётные ширины не выравнены по байту, а выигрыш
    # между 12 и 8 разрядами меньше, чем цена новой распаковки (см. запись про лестницу).
    q12 = сц12 = None
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
        from quantize_iN import квантовать as кв12
        # КУСКАМИ ПО N: плотный fp32-словарь весит 2.54 ГБ, вчетверо больше результата.
        части, сцч, ошч = [], [], []
        for n0 in range(0, N, 8192):
            n1 = min(N, n0 + 8192)
            пк, мс, ош, _ = кв12(w[n0:n1].float(), 12, G)
            части.append(пк); сцч.append(мс); ошч.append(ош)
            del пк, мс
        q12 = torch.cat(части, dim=2).contiguous(); сц12 = torch.cat(сцч, dim=1).contiguous()
        ош12 = sum(ошч) / len(ошч)
        del части, сцч; torch.cuda.empty_cache()
        ext12 = fa2_sm70._ext.w12_ext() if hasattr(fa2_sm70._ext, "w12_ext") else None
        if ext12 is None:
            from fa2_sm70 import _ext as _e
            ext12 = _e.w12a16_ext()
        y12 = ext12.w12a16_gemv(x, q12, сц12, G)
        байт12 = q12.numel() * 4 + сц12.numel() * 2
        с1, с8 = порядок(y12)
        print(f"{'НАШЕ int12':<22s} {нрм(y12):>13.3e} {с1:>8.3f} {с8:>8.3f}   "
              f"(вес {байт12/1e6:.0f} МБ, ошибка веса {ош12:.2e})")
    except Exception as e:  # noqa: BLE001
        print(f"int12 недоступен ({str(e)[:80]})")
        ext12 = None

    if ext12 is not None:
        сч12 = [0, 0]
        for _ in range(ПРОБ):
            xx = (torch.randn(1, K, device=dev) * 0.35).half()
            ye = torch.zeros(1, N, device=dev, dtype=torch.float32)
            for n0 in range(0, N, шаг):
                n1 = min(N, n0 + шаг)
                ye[:, n0:n1] = xx.float() @ w[n0:n1].float().t()
            y = ext12.w12a16_gemv(xx, q12, сц12, G).float()
            сч12[0] += int(y.argmax(-1).item() == ye.argmax(-1).item())
            сч12[1] += len(set(y.topk(8, -1).indices[0].tolist()) & set(ye.topk(8, -1).indices[0].tolist()))
        print(f"  {'НАШЕ int12':<18s} top-1 {сч12[0]}/{ПРОБ}   "
              f"top-8 {сч12[1]}/{ПРОБ*8} = {сч12[1]/(ПРОБ*8):.3f}")

    # --- ВРЕМЯ ---
    print(f"\n{'путь':<22s} {'мс':>9s} {'ГБ/с':>9s} {'% крыши':>9s} {'к turbomind':>12s}")
    t_н = замер(lambda: ext.w8a16_gemv(x, qw, qz, sc, G, True))
    t_т = замер(lambda: (x @ w.t()))
    t_tm = замер(lambda: vops.sm70_f16_gemm_out(y_tm, x, tm_w, tm_k, False)) if y_tm is not None else None
    строки = [("torch fp16 (плотно)", t_т, w.numel() * 2)]
    if t_tm is not None:
        строки.append(("turbomind fp16", t_tm, w.numel() * 2))
    if ext12 is not None:
        строки.append(("НАШЕ int12", замер(lambda: ext12.w12a16_gemv(x, q12, сц12, G)),
                       q12.numel() * 4 + сц12.numel() * 2))
    строки.append(("НАШЕ int8", t_н, наш_байт))
    for имя, t, б in строки:
        гбс = б / (t / 1e3) / 1e9
        отн = f"{t_tm/t:>12.3f}" if t_tm else f"{t_т/t:>12.3f}*"
        print(f"{имя:<22s} {t:>9.3f} {гбс:>9.1f} {гбс/ПОЛОСА*100:>8.1f}% {отн}")
    if t_tm is None:
        print("  * отношение к torch, не к turbomind")
    return 0


if __name__ == "__main__":
    sys.exit(main())
