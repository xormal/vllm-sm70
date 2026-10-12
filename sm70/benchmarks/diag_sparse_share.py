# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ГЛАВНАЯ ОСЬ МЕХАНИЗМА, НАЙДЕННАЯ ЗАМЕРОМ: не сколько плиток выброшено, а СОГЛАСОВАНЫ ли выбросы
# у одновременно идущих заданий.
#
# Замер на 250K: при плотности 0.60 и НЕЗАВИСИМОМ случайном отборе -- 1.28x, при отборе, ОДИНАКОВОМ
# у всех блоков запросов, -- 1.66x (идеал 1.667x). При плотности 0.835 независимый отбор даёт 0.93x,
# то есть УБЫТОК. Число плиток в обоих случаях одно и то же, значит дело не в FLOP.
# Причина считаемая: K/V ОДНОЙ головы на 250K -- 256 МБ, L2 -- 6 МБ. Плотное внимание живёт на 97.7%
# попаданий в L2 только потому, что 80 одновременных блоков запросов идут по плиткам ключей почти
# синхронно. Независимый отбор эту синхронность ломает, и то же число плиток приходится тащить из DRAM.
#
# Здесь мерится ЦЕНА НЕСОГЛАСОВАННОСТИ как функция размера группы блоков запросов, делящих одну маску.
# Это прямо переводится в правило проектирования: на какой ЗЕРНИСТОСТИ ЗАПРОСОВ отбирать.
import os, sys, math, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70
from benchmarks.bench_sparse_mech import (build_mask_shared, nkbC_of, measure, gpu_state, BQ, BK, D)


def run(B, H, Hkv, Sq, Sk, dens, rounds=4):
    ext = fa2_sm70._ext.sparse_ext()
    sc = 1.0 / math.sqrt(D)
    nqb, nkb = (Sq + BQ - 1) // BQ, (Sk + BK - 1) // BK
    torch.manual_seed(0)
    q = (torch.randn(B, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    k = (torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    v = (torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    empty = torch.empty(0, dtype=torch.int32, device="cuda")
    dn = np.array([nkbC_of(qb, Sq, Sk, True) for qb in range(nqb)] * (B * H))
    Gs = sorted({g for g in (1, 2, 4, 8, 16, 64, nqb) if g <= nqb})
    var = {"плотное": lambda: ext.fwd_ref(q, k, v, sc, True)}
    cn = {}
    for G in Gs:
        ml, mc, cnt = build_mask_shared(B, H, Sq, Sk, True, dens, G, fmt=2)
        cn[G] = (cnt, ml, mc)
        var[f"G={G}"] = (lambda m_=ml, c_=mc: ext.fwd_sparse(q, k, v, sc, True, 2, m_, c_, empty))
    r = measure(var, rounds=rounds, reps=2)
    print(f"\n=== B={B} H={H} Hkv={Hkv} Sq={Sq} Sk={Sk}, плотность {dens:.2f} "
          f"(K+V одной головы {2*Sk*D*2/2**20:.0f} МБ против 6 МБ L2) ===")
    print(f'{"группа блоков запросов":>26} {"мс":>9} {"x к плотному":>13} {"идеал":>7} {"добор":>7}')
    for G in Gs:
        t, rat = r[f"G={G}"]
        frac = cn[G][0].sum() / dn.sum()
        print(f'{("G=" + str(G)):>26} {t*1e3:9.3f} {1/rat:12.3f}x {1/frac:6.3f}x {(1/rat)*frac*100:6.1f}%')
    tD, _ = r["плотное"]
    print(f'{"плотное (база)":>26} {tD*1e3:9.3f} {1.0:12.3f}x')
    del q, k, v, cn; torch.cuda.empty_cache()


if __name__ == "__main__":
    print("СОСТОЯНИЕ КАРТ: " + gpu_state())
    run(1, 12, 2, 8192, 65536, 0.60)
    run(1, 12, 2, 16384, 249856, 0.60)
    run(1, 12, 2, 16384, 249856, 0.835)
    print("\nСОСТОЯНИЕ КАРТ: " + gpu_state())
