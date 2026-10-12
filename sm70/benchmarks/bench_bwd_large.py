# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Isolated BACKWARD benchmark for large head dims (d=256/512) against torch's SDPA-efficient backward.
# This is the shape family where the hand-written block mainloop (fa2_sm70/csrc/volta_bwd_block.cuh)
# replaces the cutlass one, so the README's "isolated large-head backward" table is reproduced HERE.
#
# The forward is run once outside the timed region; only autograd.grad is timed, so the numbers are the
# backward alone (kernel + dQ down-convert + wrapper), not fwd+bwd. Correctness is checked against the
# same SDPA reference in the same process, so a regression shows up as relL2 rather than as silence.
#
#   python benchmarks/bench_bwd_large.py            # d=256 and d=512
#   FA2SM70_BLOCK_BWD=0 python benchmarks/bench_bwd_large.py   # A/B: force the cutlass path
#
# Lock clocks for stable numbers: sudo nvidia-smi -lgc 1530 (release: sudo nvidia-smi -rgc).
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70

SHAPES = [(1, 1, 2048, 256), (1, 8, 512, 256), (2, 8, 1024, 256), (1, 8, 2048, 256),
          (1, 32, 2048, 256), (1, 8, 1024, 512), (1, 8, 2048, 512)]


def bench(fn, iters=30, reps=6, warmup=15):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):                       # min-of-reps: the first pass builds persistent caches
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e3)
    return best


def main():
    if not torch.cuda.is_available():
        print("CUDA required"); return 1
    print("Isolated backward vs torch SDPA-efficient (causal, fp16, min-of-6)")
    print(f"{'B, H, S, d':<20} {'ours':>10} {'SDPA-eff':>10} {'ours/SDPA':>10} {'relL2':>9}")
    print("-" * 63)
    ok = True
    for B, H, S, d in SHAPES:
        q, k, v = [torch.randn(B, H, S, d, device="cuda", dtype=torch.float16, requires_grad=True)
                   for _ in range(3)]
        gO = torch.randn(B, H, S, d, device="cuda", dtype=torch.float16)
        o = fa2_sm70.flash_attn_func(q, k, v, causal=True)
        ot = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        g = torch.autograd.grad(o, (q, k, v), gO, retain_graph=True)
        gr = torch.autograd.grad(ot, (q, k, v), gO, retain_graph=True)
        rel = max((a - b).float().norm().item() / b.float().norm().item() for a, b in zip(g, gr))
        ours = bench(lambda: torch.autograd.grad(o, (q, k, v), gO, retain_graph=True))
        ref = bench(lambda: torch.autograd.grad(ot, (q, k, v), gO, retain_graph=True))
        bad = rel > 3e-3
        ok &= not bad
        print(f"{f'{B}, {H}, {S}, {d}':<20} {ours:9.3f}ms {ref:9.3f}ms {ref/ours:9.2f}x {rel:9.1e}"
              f"{'  FAIL' if bad else ''}")
    print("OK" if ok else "ACCURACY FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
