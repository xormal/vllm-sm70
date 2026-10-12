# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# КОРРЕКТНОСТЬ МЕХАНИЗМА РАЗРЕЖЕННОСТИ. Главное требование задания: при плотности 100% ядро обязано
# давать ПОБИТОВО то же, что без механизма. Оно проверяется тремя РАЗНЫМИ путями подачи маски, и
# отдельно проверяется сам отборщик -- что он выдаёт РОВНО заданное число плиток, ОТСОРТИРОВАННЫМ
# списком, и что якоря (нулевая плитка и диагональная) на месте. Диагональная плитка -- не украшение:
# без неё у части строк не остаётся ни одного неотброшенного ключа, и выход строки становится нулём.
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70
from benchmarks.bench_sparse_mech import build_mask, jobq_from_counts, nkbC_of, BQ, BK, D

SHAPES = [(1, 4, 2, 1024, 1024), (1, 12, 2, 2048, 8192), (2, 6, 3, 512, 4096)]
fails = []


def chk(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond: fails.append(msg)


def main():
    ext = fa2_sm70._ext.sparse_ext()
    empty = torch.empty(0, dtype=torch.int32, device="cuda")
    for (B, H, Hkv, Sq, Sk) in SHAPES:
        print(f"=== B={B} H={H} Hkv={Hkv} Sq={Sq} Sk={Sk} ===")
        torch.manual_seed(Sq + Sk + H)
        sc = 1.0 / math.sqrt(D)
        q = (torch.randn(B, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
        k = (torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
        v = (torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
        nqb, nkb = (Sq + BQ - 1) // BQ, (Sk + BK - 1) // BK
        dn = np.array([nkbC_of(qb, Sq, Sk, True) for qb in range(nqb)] * (B * H))

        for causal in (True, False):
            o_ref, l_ref = ext.fwd_ref(q, k, v, sc, causal)
            o0, l0 = ext.fwd_sparse(q, k, v, sc, causal, 0, empty, empty, empty)
            chk(torch.equal(o_ref, o0) and torch.equal(l_ref, l0),
                f"causal={causal}: sp=0 (рефакторинг цикла ключей) ПОБИТОВО == отгруженное ядро")
            for fmt in (1, 3, 2):
                ml, mc, cnt = build_mask(B, H, Sq, Sk, causal, np.ones((B, H)), fmt=fmt)
                o, l = ext.fwd_sparse(q, k, v, sc, causal, fmt, ml, mc, empty)
                chk(torch.equal(o_ref, o) and torch.equal(l_ref, l),
                    f"causal={causal}: плотность 100%, формат {'список' if fmt==1 else ('список+чтение вперёд' if fmt==3 else 'битовая карта')}"
                    f" ПОБИТОВО == отгруженное ядро")
                jq = jobq_from_counts(cnt, B, H, nqb, "grid")
                o, l = ext.fwd_sparse(q, k, v, sc, causal, fmt, ml, mc, jq)
                chk(torch.equal(o_ref, o) and torch.equal(l_ref, l),
                    f"causal={causal}: то же + ОЧЕРЕДЬ ЗАДАНИЙ вместо трёхмерной сетки -- ПОБИТОВО")
                del ml, mc, jq

        # ---- отборщик на устройстве
        kbar, rmax = ext.block_summary(k, 0)
        chk(torch.isfinite(kbar.float()).all().item() and torch.isfinite(rmax).all().item(),
            "сводки плиток конечны")
        # сводка == настоящее среднее плитки (последняя плитка может быть рваной)
        kf = k.float()
        nfull = Sk // BK
        ref_bar = kf[:, :, :nfull * BK].reshape(B, Hkv, nfull, BK, D).mean(3)
        err = (kbar.float()[:, :, :nfull] - ref_bar).abs().max().item()
        chk(err < 2e-3, f"средний ключ совпадает с прямым расчётом (max|diff| = {err:.1e})")

        for dsel in (1.0, 0.6):
            dens = torch.full((B, H), dsel, dtype=torch.float32, device="cuda")
            km, kc = ext.select(q, kbar, rmax, dens, Sk, Hkv, sc, True, 0.0, 1, 1)
            c = kc.cpu().numpy()
            want = np.maximum(np.minimum(2, dn), np.rint(dsel * dn).astype(np.int64))
            want = np.minimum(want, dn)
            chk(np.array_equal(c, want), f"отбор p={dsel}: число плиток РОВНО заданное "
                                         f"(расхождений {int((c != want).sum())})")
            m = km.cpu().numpy()
            asc = all(np.all(np.diff(m[r, :c[r]]) > 0) for r in range(len(c)))
            chk(asc, f"отбор p={dsel}: список СТРОГО возрастает")
            anch = all(m[r, 0] == 0 and m[r, c[r] - 1] == dn[r] - 1 for r in range(len(c)))
            chk(anch, f"отбор p={dsel}: якоря на месте (плитка 0 и диагональная)")
            if dsel == 1.0:
                o_ref, l_ref = ext.fwd_ref(q, k, v, sc, True)
                o, l = ext.fwd_sparse(q, k, v, sc, True, 1, km, kc, empty)
                chk(torch.equal(o_ref, o) and torch.equal(l_ref, l),
                    "отбор p=1.0 на УСТРОЙСТВЕ -> внимание ПОБИТОВО == отгруженное ядро")
            # битовая карта того же отбора обязана нести те же плитки
            km2, kc2 = ext.select(q, kbar, rmax, dens, Sk, Hkv, sc, True, 0.0, 2, 1)
            pc = np.array([bin(x & 0xffffffff).count("1") for x in km2.cpu().numpy().ravel()]).reshape(
                km2.shape).sum(1)
            chk(np.array_equal(pc, c) and np.array_equal(kc2.cpu().numpy(), c),
                f"отбор p={dsel}: битовая карта несёт РОВНО те же плитки, что список")
            o1, _ = ext.fwd_sparse(q, k, v, sc, True, 1, km, kc, empty)
            o2, _ = ext.fwd_sparse(q, k, v, sc, True, 2, km2, kc2, empty)
            chk(torch.equal(o1, o2), f"отбор p={dsel}: список и карта дают ПОБИТОВО один результат")
            # выход конечен и ни одна строка не занулилась (это и проверяют якоря)
            chk(torch.isfinite(o1).all().item() and (o1.float().abs().sum(-1) > 0).all().item(),
                f"отбор p={dsel}: выход конечен и НИ ОДНА строка не занулена")
        del q, k, v, kbar, rmax; torch.cuda.empty_cache()

    print(f"\n{'ВСЁ СОШЛОСЬ' if not fails else 'ПРОВАЛЫ: ' + str(len(fails))}")
    for f in fails: print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
