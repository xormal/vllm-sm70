# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ПОСЫЛКА, КОТОРУЮ НАДО ПРОВЕРИТЬ ПЕРВОЙ: ядро e5m2 быстрее РОВНО ТОГО ЖЕ int8-ядра.
#
# Это единственный чистый A/B, который у нас есть: attn_fwd_volta_e5m2 и attn_fwd_volta_i8 -- ОДНА
# И ТА ЖЕ инстанциация volta_fwd_ws, различающаяся ТОЛЬКО шаблонным KVFMT (0 = int8 с масштабом на
# позицию, 1 = e5m2). Плитка, разделяемая память, число варпов, сетка -- тождественны. Значит любая
# разница по времени объясняется РАСПАКОВКОЙ, а не геометрией и не трафиком (оба однобайтовые).
#
# ДИСЦИПЛИНА ЗАМЕРА (куплена авариями):
#  * частоты залочить нечем (нет sudo), поэтому варианты ЧЕРЕДУЮТСЯ и берётся медиана: дрейф частот
#    бьёт по обоим одинаково, а блочный порядок отдал бы весь дрейф одному из них;
#  * считаем ЧУЖИЕ процессы на картах 0-1 до и после и печатаем это число рядом с результатом;
#  * частоты и мощность снимаются посекундно, а не средним.
import math
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70  # noqa: E402

D = 256


def cards_state(idx="0,1"):
    """(чужих процессов, [частоты SM], [мощности]) -- по картам 0-1. 'Чужой' = любой, кто НЕ мы."""
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


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def interleaved(fns, warmup=3, rounds=9):
    """Чередуем варианты внутри раунда -- дрейф частот достаётся всем поровну."""
    for name, fn in fns.items():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    acc = {k: [] for k in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            acc[name].append(s.elapsed_time(e))
    out = {}
    for k, v in acc.items():
        v.sort()
        out[k] = (v[len(v) // 2], v[0], v[-1])
    return out


def main():
    torch.manual_seed(0)
    pext = fa2_sm70._ext.prefill_ext()
    dext = fa2_sm70._ext.decode_ext()
    H, Hkv = int(os.environ.get("H", 16)), 2
    Sq = int(os.environ.get("SQ", 512))
    lens = [int(x) for x in os.environ.get("LENS", "4096,16384,65536,131072").split(",")]
    sc = 1.0 / math.sqrt(D)

    n0, cl0 = cards_state()
    print(f"# чужих процессов на картах 0-1 в начале: {n0}   частоты/мощность: {cl0}")
    print(f"# H={H} Hkv={Hkv} d={D} Sq={Sq} causal=True")
    print()
    hdr = f"{'Sk':>8} {'fp16 ms':>10} {'int8 ms':>10} {'e5m2 ms':>10} {'e5m2/int8':>10} {'e5m2/fp16':>10} {'KV МБ':>8} {'ГБ/с e5m2':>10}"
    print(hdr); print("-" * len(hdr))

    for Sk in lens:
        q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
        # исходные значения -- одни и те же для всех трёх форматов
        kf = (torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4).contiguous()
        vf = (torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.4).contiguous()
        # e5m2
        kb = kf.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        vb = vf.to(torch.float8_e5m2).view(torch.uint8).contiguous()
        # int8 + масштаб на позицию
        ka = kf.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
        va = vf.float().abs().amax(-1, keepdim=True).clamp_min(6.1e-5) / 127.0
        kq = torch.clamp(torch.round(kf.float() / ka), -127, 127).to(torch.int8).contiguous()
        vq = torch.clamp(torch.round(vf.float() / va), -127, 127).to(torch.int8).contiguous()
        ks = ka.squeeze(-1).contiguous().float()
        vs = va.squeeze(-1).contiguous().float()

        fns = {
            "fp16": lambda: pext.attn_fwd_volta(q, kf, vf, sc, True),
            "int8": lambda: pext.attn_fwd_volta_i8(q, kq, ks, vq, vs, sc, True),
            "e5m2": lambda: pext.attn_fwd_volta_e5m2(q, kb, vb, sc, True),
        }
        r = interleaved(fns)
        kvmb = 2 * Hkv * Sk * D / 1e6          # K+V в однобайтовом формате
        gbs = (2 * Hkv * Sk * D) / (r["e5m2"][0] * 1e-3) / 1e9
        print(f"{Sk:>8} {r['fp16'][0]:>10.3f} {r['int8'][0]:>10.3f} {r['e5m2'][0]:>10.3f} "
              f"{r['e5m2'][0]/r['int8'][0]:>10.4f} {r['e5m2'][0]/r['fp16'][0]:>10.4f} {kvmb:>8.1f} {gbs:>10.1f}")
        print(f"         разброс мин-макс: fp16 {r['fp16'][1]:.3f}-{r['fp16'][2]:.3f}  "
              f"int8 {r['int8'][1]:.3f}-{r['int8'][2]:.3f}  e5m2 {r['e5m2'][1]:.3f}-{r['e5m2'][2]:.3f}")
        del q, kf, vf, kb, vb, kq, vq, ks, vs, ka, va
        torch.cuda.empty_cache()

    n1, cl1 = cards_state()
    print()
    print(f"# чужих процессов на картах 0-1 в конце: {n1}   частоты/мощность: {cl1}")


if __name__ == "__main__":
    main()
