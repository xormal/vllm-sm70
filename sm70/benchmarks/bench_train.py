# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Training benchmark: forward vs forward+backward latency for fa2_sm70.flash_attn_func (the exact
# differentiable sm_70 path), compared against torch SDPA (EFFICIENT_ATTENTION backend) fwd+bwd.
# Reports both fwd and fwd+bwd latency and the bwd/fwd ratio. See README.md for
# why the sm_70 backward is register/occupancy-locked at ~13% roofline (ours) / ~19% (torch); see the
# corrected 344-GFLOP roofline analysis there. Run:
#   python benchmarks/bench_train.py
import math, torch
from _common import timed, efficient_backend
import fa2_sm70


def bwd_ours(Q, K, V, dO, scale):
    for x in (Q, K, V):
        x.grad = None
    O = fa2_sm70.flash_attn_func(Q, K, V, causal=True, scale=scale)
    O.backward(dO)


def bwd_sdpa(Q, K, V, dO, scale):
    for x in (Q, K, V):
        x.grad = None
    with efficient_backend():
        O = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)
    O.backward(dO)


torch.manual_seed(0)
print(f"{'B,H,S,d':<16}{'ours fwd':>10}{'ours f+b':>10}{'ours b/f':>9}   {'sdpa f+b':>10}{'ours vs sdpa':>13}")
print("-" * 70)
# Формы d=256/512 добавлены намеренно: именно там стоит рукописный блочный мейнлуп
# (fa2_sm70/csrc/volta_bwd_block.cuh, причинный d=256), и без них его вклад в бенчмарке не виден.
for (B, H, S, d) in [(2, 16, 2048, 128), (1, 32, 4096, 128), (4, 8, 2048, 64), (1, 8, 8192, 128),
                     (1, 8, 2048, 256), (1, 1, 2048, 256), (1, 32, 2048, 256), (2, 8, 1024, 256),
                     (1, 8, 2048, 512)]:
    scale = 1.0 / math.sqrt(d)
    Q = (torch.randn(B, H, S, d, device="cuda", dtype=torch.float16) * 0.5).requires_grad_()
    K = (torch.randn(B, H, S, d, device="cuda", dtype=torch.float16) * 0.5).requires_grad_()
    V = (torch.randn(B, H, S, d, device="cuda", dtype=torch.float16) * 0.5).requires_grad_()
    dO = torch.randn(B, H, S, d, device="cuda", dtype=torch.float16) * 0.5

    def fwd():
        with torch.no_grad():
            fa2_sm70.flash_attn_func(Q, K, V, causal=True, scale=scale)

    f = timed(fwd, it=50, wu=10) * 1e3
    fb = timed(lambda: bwd_ours(Q, K, V, dO, scale), it=30, wu=10) * 1e3
    try:
        sb = timed(lambda: bwd_sdpa(Q, K, V, dO, scale), it=30, wu=10) * 1e3
        sdpa = f"{sb:9.2f}ms"; ratio = f"{sb / fb:6.2f}x"
    except Exception as e:
        sdpa = "   n/a"; ratio = "   n/a"
    print(f"B{B}H{H}S{S}d{d:<4} {f:8.2f}ms {fb:8.2f}ms {fb / f:7.1f}x   {sdpa:>10} {ratio:>13}")
