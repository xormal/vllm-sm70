# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ЗОНД "ПРОЛОГ" -- СВОБОДНЫЙ ЧЛЕН ЛИНЕЙНОЙ ПОДГОНКИ, БЕЗ ЕДИНОЙ ПРАВКИ ЯДРА.
#
# Ресурс: заполнение конвейера и всё до первой HMMA у ядра ролей (fa2_sm70/csrc/volta_fwd_ws.cuh).
# По плитке ключей ядро линейно: T(n) = a + b*n, где n = число плиток = ceil(Sk/BK), BK = 64.
# Свободный член a -- это ВСЁ, что платится ОДИН раз на CTA (пролог + эпилог) плюс запуск.
#
# ЧТО РАЗВОДИТСЯ ТРЕМЯ ПОДГОНКАМИ (иначе a -- смесь трёх разных вещей):
#   T(n, W) = L + W*(P + E) + W*b*n,   W = число ВОЛН = CTA/80  (один блок на SM: 88.5 КБ)
#   W=1: a1 = L + (P+E),  наклон b1 = b
#   W=2: a2 = L + 2(P+E), наклон b2 = 2b
#   W=3: a3 = L + 3(P+E), наклон b3 = 3b
#   =>  (P+E) = a2 - a1 = a3 - a2   (ДВЕ независимые оценки -- это и есть проверка, а не украшение)
#       L     = 2*a1 - a2
# ВОРОТА ЛИНЕЙНОСТИ: b2/b1 ~ 2 и b3/b1 ~ 3 с точностью 3%. Если не сошлось -- на карте не одна
# волна на 80 SM (или K/V выпали из L2), и весь вердикт по a НЕДЕЙСТВИТЕЛЕН.
#
# ГЕОМЕТРИЯ ПОДОБРАНА ТАК, ЧТОБЫ МЕНЯЛОСЬ ТОЛЬКО n:
#   Sq = 32  -> grid.x = 1 (ровно один блок запросов, ноль дробления по q)
#   H = 16, B = 5*W, Hkv = 1 -> grid = (1, 16, 5W) = 80W CTA, kv_ratio = 16 (степень двойки)
#   causal = False -> kEnd = Sk у ВСЕХ CTA, плитки одинаковой тяжести, хвоста нет
#   Sk <= 1536 -> байты K/V <= 5*1*1536*256*2 = 3.9 МБ, то есть ВСЁ ВРЕМЯ в L2 (6 МБ):
#                 иначе наклон b поедет от промахов, и это прочтётся как кривизна пролога.
import argparse
import os
import statistics
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256
BK = 64
NSM = 80


def cards_state(idx):
    me = str(os.getpid())
    try:
        uu = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", idx],
                            capture_output=True, text=True, timeout=10).stdout.split()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        n = 0
        for line in apps.strip().splitlines():
            if not line.strip():
                continue
            u, p = [s.strip() for s in line.split(",")]
            if u in uu and p != me:
                n += 1
        cl = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw",
                             "--format=csv,noheader,nounits", "-i", idx],
                            capture_output=True, text=True, timeout=10).stdout.strip().replace("\n", " | ")
        return n, cl
    except Exception as e:
        return -1, str(e)


def make(B, H, Hkv, Sq, Sk, dev):
    g = torch.Generator(device=dev).manual_seed(1234)
    q = torch.randn(B, Sq, H, D, generator=g, device=dev, dtype=torch.float16)
    kq = torch.randint(-127, 127, (B, Hkv, Sk, D), generator=g, device=dev, dtype=torch.int8)
    vq = torch.randint(-127, 127, (B, Hkv, Sk, D), generator=g, device=dev, dtype=torch.int8)
    ks = torch.full((B, Hkv, Sk), 0.02, device=dev, dtype=torch.float32)
    vs = torch.full((B, Hkv, Sk), 0.02, device=dev, dtype=torch.float32)
    return q, kq, ks, vq, vs


def burst(fn, reps, rounds, warmup=3):
    """Пачка из reps запусков подряд: латентность ХОСТОВОГО запуска амортизируется, остаётся
    то, что платит УСТРОЙСТВО. Медиана по раундам."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / reps)   # мкс на запуск
    return statistics.median(ts)


def lsq(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    res = max(abs(y - (a + b * x)) for x, y in zip(xs, ys))
    return a, b, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="0")
    ap.add_argument("--nosc", type=int, default=0, help="0 = боевой int8 + таблица; 8 требует float2-таблицу")
    ap.add_argument("--nmax", type=int, default=24, help="плиток максимум (Sk = 64*n)")
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=11)
    a = ap.parse_args()

    os.environ.setdefault("FA2SM70_WS", "1")
    dev = f"cuda:{a.card}"
    torch.cuda.set_device(int(a.card))
    ext = fa2_sm70._ext.load()
    n_before, clk_before = cards_state(a.card)
    print(f"# карта {a.card}: чужих процессов {n_before}, clocks/power: {clk_before}")
    print(f"# FA2SM70_WS={os.environ.get('FA2SM70_WS')} nosc={a.nosc}")

    # n подобраны так, чтобы точек хватило на устойчивый наклон, но Sk остался в L2
    ns = list(range(1, a.nmax + 1))
    fits = {}
    for W in (1, 2, 3):
        B, H, Hkv, Sq = 5 * W, 16, 1, 32
        xs, ys = [], []
        for n in ns:
            Sk = BK * n
            q, kq, ks, vq, vs = make(B, H, Hkv, Sq, Sk, dev)
            sc = 1.0 / (D ** 0.5)
            t = burst(lambda: ext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, False, a.nosc),
                      a.reps, a.rounds)
            xs.append(float(n))
            ys.append(t)
            del q, kq, ks, vq, vs
        a0, b0, res = lsq(xs, ys)
        fits[W] = (a0, b0, res)
        print(f"W={W} (CTA={80*W}): a = {a0:8.3f} мкс   b = {b0:8.4f} мкс/плитка   макс.невязка {res:.4f} мкс")
        print("   точки:", " ".join(f"{n}:{y:.2f}" for n, y in zip(ns, ys)))

    a1, b1, _ = fits[1]
    a2, b2, _ = fits[2]
    a3, b3, _ = fits[3]
    pe_12, pe_23 = a2 - a1, a3 - a2
    L = 2 * a1 - a2
    print()
    print(f"ВОРОТА ЛИНЕЙНОСТИ: b2/b1 = {b2/b1:.3f} (ждём 2.00), b3/b1 = {b3/b1:.3f} (ждём 3.00)")
    print(f"ПРОЛОГ+ЭПИЛОГ на CTA: {pe_12:.3f} мкс (из W1/W2) и {pe_23:.3f} мкс (из W2/W3)")
    print(f"ЗАПУСК (хост+рампа):  {L:.3f} мкс")
    try:
        mhz = float(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits",
                                    "-i", a.card], capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        mhz = 1530.0
    pe = 0.5 * (pe_12 + pe_23)
    print(f"в тактах при {mhz:.0f} МГц: пролог+эпилог = {pe*mhz:.0f} тактов, плитка = {b1*mhz:.0f} тактов")
    print(f"пролог+эпилог = {pe/b1:.2f} ПЛИТКИ")
    print()
    print("ДОЛЯ свободного члена в ядре при разном числе плиток:")
    for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        print(f"   n={n:4d} (Sk={BK*n:6d}):  {100*pe/(pe + b1*n):5.2f} %")
    n_after, clk_after = cards_state(a.card)
    print(f"# после: чужих процессов {n_after}, clocks/power: {clk_after}")
    if n_before or n_after:
        print("# ЗАМЕР НЕДЕЙСТВИТЕЛЕН: на карте были чужие процессы")


if __name__ == "__main__":
    main()
