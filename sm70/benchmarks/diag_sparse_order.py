# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ДИАГНОСТИКА двух аномалий, вылезших в bench_sparse_mech.py, и обе -- «замер против рассуждения»:
#   A. НЕПРЕРЫВНОЕ НАЧАЛО (плитки 0..t далеко от диагонали) при плотности 0.60 дало 0.72x, то есть
#      МЕДЛЕННЕЕ плотного при 40% выброшенной работы. Непрерывный ХВОСТ при том же числе плиток --
#      1.66x. Значит дело не в числе плиток и не в непрерывности, а в ТОМ, КАКИЕ плитки.
#   B. Очередь заданий в порядке «как сетка» дала 183 мс против 126 у самой сетки, хотя перестановка
#      ПОБИТОВО та же. Либо очередь стоит денег сама по себе, либо один из замеров нестабилен.
# Здесь каждая гипотеза получает свой КОНТРОЛЬ, замеры ЧЕРЕДУЮТСЯ (иначе дрейф частот читается как
# эффект), и печатаются СЫРЫЕ времена, а не только медиана.
import os, sys, math, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70
from benchmarks.bench_sparse_mech import build_mask, jobq_from_counts, nkbC_of, BQ, BK, D

B, H, Hkv, Sq, Sk = 1, 12, 2, 8192, 65536
IT = 7


def raw(fn, it=IT, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def main():
    ext = fa2_sm70._ext.sparse_ext()
    sc = 1.0 / math.sqrt(D)
    nqb = (Sq + BQ - 1) // BQ
    torch.manual_seed(0)
    q = (torch.randn(B, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    k = (torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    v = (torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    empty = torch.empty(0, dtype=torch.int32, device="cuda")
    o_ref, _ = ext.fwd_ref(q, k, v, sc, True)
    dn = np.array([nkbC_of(qb, Sq, Sk, True) for qb in range(nqb)] * (B * H))

    print("=== B: ОЧЕРЕДЬ ЗАДАНИЙ. Контроль -- перестановка, ПОБИТОВО равная порядку сетки ===")
    lay = [0.90, 0.85, 0.78, 0.52]
    ph = np.array([[lay[i % 4] for i in range(H)]])
    ml, mc, cnt = build_mask(B, H, Sq, Sk, True, ph, fmt=2, pattern="random")
    p_grid = jobq_from_counts(cnt, B, H, nqb, "grid").cpu().numpy()
    p_inh = jobq_from_counts(cnt, B, H, nqb, "lpt_inhead").cpu().numpy()
    print(f"  перестановки grid и lpt_inhead совпадают: {np.array_equal(p_grid, p_inh)}"
          f"  (различий {int((p_grid != p_inh).sum())} из {len(p_grid)})")
    jq_g = torch.from_numpy(p_grid).cuda()
    jq_i = torch.from_numpy(p_inh).cuda()
    # ГЛАВНЫЙ КОНТРОЛЬ: очередь при плотности 100% и БЕЗ маски -- цена самой очереди, и только её
    ones = np.ones((B, H))
    ml1, mc1, cnt1 = build_mask(B, H, Sq, Sk, True, ones, fmt=2, pattern="random")
    jq_dense = jobq_from_counts(cnt1, B, H, nqb, "grid").cuda() if False else \
        jobq_from_counts(cnt1, B, H, nqb, "grid")
    runs = {
        "сетка 3D, sp=0 (плотно)": lambda: ext.fwd_sparse(q, k, v, sc, True, 0, empty, empty, empty),
        "ОЧЕРЕДЬ grid, sp=0 (плотно)": lambda: ext.fwd_sparse(q, k, v, sc, True, 0, empty, empty, jq_dense),
        "сетка 3D, маска поголовная": lambda: ext.fwd_sparse(q, k, v, sc, True, 2, ml, mc, empty),
        "ОЧЕРЕДЬ grid, маска поголовная": lambda: ext.fwd_sparse(q, k, v, sc, True, 2, ml, mc, jq_g),
        "ОЧЕРЕДЬ lpt_inhead, та же маска": lambda: ext.fwd_sparse(q, k, v, sc, True, 2, ml, mc, jq_i),
    }
    # ЧЕРЕДУЕМ: три круга по всем вариантам, а не три подряд по одному
    got = {n: [] for n in runs}
    for rnd in range(3):
        for n, fn in runs.items():
            got[n] += raw(fn, 3, 1)
    for n in runs:
        ts = sorted(got[n])
        print(f"  {n:<34} медиана {ts[len(ts)//2]:8.3f} мс  сырые {[f'{x:.1f}' for x in got[n]]}")
    # и выход обязан быть один и тот же
    for n, fn in runs.items():
        o, _ = fn()
        if "плотно" in n:
            print(f"  {n:<34} побитово равно плотному: {torch.equal(o, o_ref)}")
        else:
            print(f"  {n:<34} конечно={torch.isfinite(o).all().item()}")
    ogrid, _ = ext.fwd_sparse(q, k, v, sc, True, 2, ml, mc, empty)
    oq, _ = ext.fwd_sparse(q, k, v, sc, True, 2, ml, mc, jq_g)
    print(f"  очередь даёт ТОТ ЖЕ результат, что сетка: {torch.equal(ogrid, oq)}")
    del ml, mc, jq_g, jq_i, jq_dense, ml1, mc1; torch.cuda.empty_cache()

    print("\n=== A: КАКИЕ плитки оставлены. Все режимы -- ровно 0.60 плотности ===")
    pats = ["random", "cluster8", "stride", "recent", "first"]
    res = {}
    tabs = {}
    for pat in pats:
        ml, mc, cnt = build_mask(B, H, Sq, Sk, True, np.full((B, H), 0.60), fmt=2, pattern=pat)
        tabs[pat] = (ml, mc, cnt)
    dense = raw(lambda: ext.fwd_sparse(q, k, v, sc, True, 0, empty, empty, empty), 3, 2)
    got = {p: [] for p in pats}
    for rnd in range(3):
        for pat in pats:
            ml, mc, _ = tabs[pat]
            got[pat] += raw(lambda: ext.fwd_sparse(q, k, v, sc, True, 2, ml, mc, empty), 3, 1)
    td = sorted(dense)[len(dense)//2]
    print(f"  плотное (sp=0): {td:.3f} мс")
    for pat in pats:
        ml, mc, cnt = tabs[pat]
        ts = sorted(got[pat]); md = ts[len(ts)//2]
        # СРЕДНЕЕ РАССТОЯНИЕ между соседними оставленными плитками -- та величина, которой различаются
        # режимы: она задаёт, попадает ли следующая плитка ключей в L2 по следам предыдущих заданий.
        print(f"  {pat:<10} плотность {cnt.sum()/dn.sum():.3f}  {md:8.3f} мс  x{td/md:5.3f}  "
              f"сырые {[f'{x:.1f}' for x in got[pat]]}")
        res[pat] = md
    print("\n  ГИПОТЕЗА ДЛЯ 'first': соседние ЗАДАНИЯ (блоки запросов одной головы) идут по сетке")
    print("  почти синхронно и читают ОДНИ И ТЕ ЖЕ плитки ключей -- отсюда 97.7% L2. Режим 'first'")
    print("  этого не ломает. Значит остаётся вторая ось -- ПРИЧИННАЯ МАСКА: плитки 0..t лежат")
    print("  ЦЕЛИКОМ ниже диагонали, а хвост -- на диагонали. Проверяем это прямо: НЕПРИЧИННОЕ")
    print("  внимание, где маски нет вообще и обе раскладки должны сравняться.")
    for pat in ("recent", "first"):
        ml, mc, cnt = build_mask(B, H, Sq, Sk, False, np.full((B, H), 0.60), fmt=2, pattern=pat)
        tt = raw(lambda: ext.fwd_sparse(q, k, v, sc, False, 2, ml, mc, empty), 5, 2)
        t0 = raw(lambda: ext.fwd_sparse(q, k, v, sc, False, 0, empty, empty, empty), 5, 2)
        a_, b_ = sorted(tt)[2], sorted(t0)[2]
        print(f"  НЕпричинное, {pat:<8} {a_:8.3f} мс против плотного {b_:8.3f} -> x{b_/a_:5.3f}")
        del ml, mc


if __name__ == "__main__":
    main()
