# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ЗОНД "ПРОЛОГ", ЧАСТЬ B: РАЗЛОЖЕНИЕ СВОБОДНОГО ЧЛЕНА ФАЛЬСИФИКАТОРАМИ.
# Требует правки из scratchpad/prologue_diag.patch (коды nosc 1600/1700/1800/1900/2100).
# Ответ вариантов 16xx-19xx НЕВЕРЕН по построению -- читается ТОЛЬКО время.
# Вариант 2100 (деление -> сдвиг) обязан совпадать ПОБИТОВО: это правка, а не фальсификатор.
#
# Как читать: доля фазы = (t_base - t_вариант) / t_base на КАЖДОМ n. Фаза пролога -- ПОСТОЯННАЯ,
# поэтому её признак -- разность в МКС не зависит от n, а доля падает как 1/n. Если разность
# растёт с n, снята НЕ прологовая фаза (значит гипотеза о её месте неверна).
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402
from probe_prologue_fit import BK, D, cards_state, make  # noqa: E402

VARIANTS = {"base": 0, "noQcopy": 1600, "noQsum": 1700, "noFill": 1800, "noStore": 1900, "shift": 2100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="0")
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=17)
    ap.add_argument("--ns", default="4,8,16,32,64")
    a = ap.parse_args()
    os.environ.setdefault("FA2SM70_WS", "1")
    dev = f"cuda:{a.card}"
    torch.cuda.set_device(int(a.card))
    ext = fa2_sm70._ext.load()
    nb, cb = cards_state(a.card)
    print(f"# карта {a.card}: чужих процессов {nb}, clocks/power: {cb}")

    for n in [int(x) for x in a.ns.split(",")]:
        B, H, Hkv, Sq, Sk = 5, 16, 1, 32, BK * n
        q, kq, ks, vq, vs = make(B, H, Hkv, Sq, Sk, dev)
        sc = 1.0 / (D ** 0.5)
        fns = {k: (lambda v=v: ext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, False, v))
               for k, v in VARIANTS.items()}
        for f in fns.values():
            f()
        torch.cuda.synchronize()
        raw = {k: [] for k in fns}
        for _ in range(a.rounds):                       # ЧЕРЕДОВАНИЕ ВНУТРИ РАУНДА
            for k, f in fns.items():
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                for _ in range(a.reps):
                    f()
                e.record()
                torch.cuda.synchronize()
                raw[k].append(s.elapsed_time(e) * 1e3 / a.reps)
        med = {k: statistics.median(v) for k, v in raw.items()}
        b = med["base"]
        print(f"n={n:4d} Sk={Sk:6d}  база {b:7.3f} мкс")
        for k in VARIANTS:
            if k == "base":
                continue
            d = b - med[k]
            print(f"    {k:9s} {med[k]:7.3f} мкс   снято {d:+7.3f} мкс = {100*d/b:+5.2f} %")
        # ПОБИТОВАЯ сверка правки (не фальсификатора)
        o0 = ext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, False, 0)[0]
        o1 = ext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, False, 2100)[0]
        print(f"    shift побитово равен базе: {torch.equal(o0, o1)}")
        del q, kq, ks, vq, vs, o0, o1
    na, ca = cards_state(a.card)
    print(f"# после: чужих процессов {na}, clocks/power: {ca}")
    if nb or na:
        print("# ЗАМЕР НЕДЕЙСТВИТЕЛЕН: на карте были чужие процессы")


if __name__ == "__main__":
    main()
