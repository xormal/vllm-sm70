# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# РЕШАЮЩИЙ ЗАМЕР: обгоняет ли перекомпонованный int8 неподвижный эталон e5m2.
#
# ДВЕ ЛОВУШКИ, КОТОРЫЕ ЭТОТ ФАЙЛ ЗАКРЫВАЕТ (обе поймали предыдущий замер):
#  1. МЕСТО В РАУНДЕ. При фиксированном порядке вариантов последний систематически получает другое
#     состояние машины, чем первый. Здесь порядок ЦИКЛИЧЕСКИ СДВИГАЕТСЯ каждый раунд, поэтому
#     каждый вариант равное число раз стоит на каждом месте.
#  2. ОТНОШЕНИЕ МЕДИАН != МЕДИАНА ОТНОШЕНИЙ. Дрейф частот двигает ОБА времени раунда согласованно,
#     поэтому честная величина -- ПАРНОЕ отношение ВНУТРИ раунда, и уже по ним медиана.
#     Отношение медиан оставляет дрейф внутри и даёт ложные десятые доли процента.
import math
import os
import statistics
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256


def cards_state(idx="0,1"):
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
                            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return n, cl
    except Exception as e:
        return -1, [str(e)]


def rotated_rounds(fns, warmup=3, rounds=15):
    """Медиана ПАРНЫХ отношений к эталону; порядок вариантов сдвигается каждый раунд."""
    names = list(fns)
    for n in names:
        for _ in range(warmup):
            fns[n]()
    torch.cuda.synchronize()
    per_round = []
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        t = {}
        for n in order:
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fns[n](); e.record(); torch.cuda.synchronize()
            t[n] = s.elapsed_time(e)
        per_round.append(t)
    med = {n: statistics.median([t[n] for t in per_round]) for n in names}
    ratio = {n: statistics.median([t["e5m2"] / t[n] for t in per_round]) for n in names}
    return med, ratio, per_round


def quant_pertoken(x):
    a = x.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
    q = torch.clamp(torch.round(x.float() / a), -127, 127).to(torch.int8).contiguous()
    return q, a.squeeze(-1).contiguous().float()


def to_biased(q):
    return (q.to(torch.int16) + 128).to(torch.uint8).view(torch.int8).contiguous()


def main():
    torch.manual_seed(0)
    pext = fa2_sm70._ext.prefill_ext()
    H, Hkv = int(os.environ.get("H", 16)), 2
    Sq = int(os.environ.get("SQ", 512))
    lens = [int(x) for x in os.environ.get("LENS", "4096,16384,65536,131072").split(",")]
    rounds = int(os.environ.get("ROUNDS", 15))
    sc = 1.0 / math.sqrt(D)

    n0, cl0 = cards_state()
    print(f"# чужих процессов на картах 0-1 в начале: {n0}   частоты/мощность: {cl0}")
    print(f"# H={H} Hkv={Hkv} d={D} Sq={Sq} causal=True  раундов={rounds}  "
          f"карта={os.environ.get('CUDA_VISIBLE_DEVICES','?')}")
    print("# ф6 = смещённый байт; ф8 = смещённый байт + чередующаяся таблица. ОБА побитово == ф0.")
    print()
    hdr = (f"{'Sk':>8} | {'мс баз':>8} {'мс e5m2':>8} {'мс ф6':>8} {'мс ф8':>8} {'мс ф10':>8} | "
           f"{'e5m2/баз':>9} {'e5m2/ф6':>9} {'e5m2/ф8':>9} {'e5m2/ф10':>9} | {'ф8|ф10 побед':>12}")
    print(hdr)
    print("-" * len(hdr))

    for Sk in lens:
        q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
        kf = (torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4).contiguous()
        vf = (torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4).contiguous()
        kb = kf.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        vb = vf.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        kq, ks = quant_pertoken(kf)
        vq, vs = quant_pertoken(vf)
        kqb, vqb = to_biased(kq), to_biased(vq)
        ksv = torch.stack([ks, vs], dim=-1).contiguous()
        # fp16-таблица: те же две величины, но по 2 Б -- байты подаются как float32-тензор,
        # ядро читает их как half2. Точность масштаба 11 бит против 8 бит самих данных.
        ksvh = torch.stack([ks, vs], dim=-1).half().contiguous().view(torch.float32)

        fns = {
            "i8":   lambda: pext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, True),
            "e5m2": lambda: pext.attn_fwd_volta_e5m2(q, kb, vb, sc, True),
            "f6":   lambda: pext.attn_fwd_volta_i8(q, kqb, ks, vqb, vs, sc, True, 6),
            "f8":   lambda: pext.attn_fwd_volta_i8(q, kqb, ksv, vqb, ksv, sc, True, 8),
            "f10":  lambda: pext.attn_fwd_volta_i8(q, kqb, ksv, vqb, ksv, sc, True, 10),
        }
        med, rat, per = rotated_rounds(fns, rounds=rounds)
        wins = sum(1 for t in per if t["f8"] < t["e5m2"])
        wins9 = sum(1 for t in per if t["f10"] < t["e5m2"])
        print(f"{Sk:>8} | {med['i8']:>8.3f} {med['e5m2']:>8.3f} {med['f6']:>8.3f} {med['f8']:>8.3f} "
              f"{med['f10']:>8.3f} | "
              f"{rat['i8']:>9.4f} {rat['f6']:>9.4f} {rat['f8']:>9.4f} {rat['f10']:>9.4f} | "
              f"{wins:>3}/{len(per)} {wins9:>3}/{len(per)}")

        if Sk == lens[0]:
            o0 = fns["i8"]()[0]
            o9 = fns['f10']()[0]
            d9 = (torch.linalg.vector_norm((o9.float() - o0.float()))
                  / torch.linalg.vector_norm(o0.float())).item()
            print(f"         ПОБИТОВО ф0==ф6 {torch.equal(o0, fns['f6']()[0])}   "
                  f"ф0==ф8 {torch.equal(o0, fns['f8']()[0])}   "
                  f"ф0==ф10 {torch.equal(o0, o9)} (откл {d9:.1e})")
        del q, kf, vf, kb, vb, kq, ks, vq, vs, kqb, vqb, ksv, ksvh
        torch.cuda.empty_cache()

    n1, cl1 = cards_state()
    print()
    print(f"# чужих процессов на картах 0-1 в конце: {n1}   частоты/мощность: {cl1}")


if __name__ == "__main__":
    main()
