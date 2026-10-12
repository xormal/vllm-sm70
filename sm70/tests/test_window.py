# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Local-attention window: fa2_sm70.attention(..., window=W [, window_right=R]) vs an fp32 reference.
# Covers the causal past-window (one-sided, shipped) and the two-sided (window_left, window_right)
# bidirectional window across all head dims (incl. the d>128 output-accumulator path where the un-
# normalized-output bug lived), GQA/MQA, and Sq!=Sk. Run:  python tests/test_window.py
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70


def ref(Q, K, V, scale, window, window_right, causal):
    """fp32 reference matching the kernel's window semantics (center = abs_q + (Sk-Sq if causal))."""
    B, H, Sq, d = Q.shape; Hkv, Sk = K.shape[1], K.shape[2]
    Kx = K.repeat_interleave(H // Hkv, 1).float(); Vx = V.repeat_interleave(H // Hkv, 1).float()
    s = (Q.float() @ Kx.transpose(-1, -2)) * scale
    off = (Sk - Sq) if causal else 0
    i = torch.arange(Sq, device=Q.device).view(Sq, 1); j = torch.arange(Sk, device=Q.device).view(1, Sk)
    center = i + off
    wl = (window - 1) if window > 0 else -1
    mask = torch.zeros(Sq, Sk, dtype=torch.bool, device=Q.device)
    if causal:
        mask |= (j > center)
    if wl >= 0:
        mask |= (j < center - wl)
    if window_right >= 0:
        mask |= (j > center + window_right)
    s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, -1) @ Vx


def run(H, Hkv, Sq, Sk, d, window, window_right, causal):
    scale = 1.0 / math.sqrt(d)
    Q = torch.randn(1, H, Sq, d, device="cuda", dtype=torch.float16) * 0.5
    K = torch.randn(1, Hkv, Sk, d, device="cuda", dtype=torch.float16) * 0.5
    V = torch.randn(1, Hkv, Sk, d, device="cuda", dtype=torch.float16) * 0.5
    O, _ = fa2_sm70.attention(Q, K, V, causal=causal, scale=scale, window=window, window_right=window_right)
    Oref = ref(Q, K, V, scale, window, window_right, causal)
    nan = torch.isnan(O).any().item()
    e = float("nan") if nan else (O.float() - Oref).norm().item() / (Oref.norm().item() + 1e-9)
    return (not nan) and e < 3e-3, e, nan


torch.manual_seed(0)
ok = True
print("causal past-window (one-sided):")
for d in [64, 128, 256, 512]:
    for (Sq, Sk, W) in [(512, 512, 1), (512, 512, 64), (512, 512, 256), (256, 512, 64)]:   # incl. Sq!=Sk
        g, e, nan = run(8, 8, Sq, Sk, d, W, -1, True)
        ok &= g; print(f"  d{d} S{Sq}/{Sk} W{W}: relL2={e:.2e} {'OK' if g else ('NAN' if nan else 'FAIL')}")

print("two-sided (bidirectional) window:")
for d in [64, 128, 256, 512]:
    for (wl, wr) in [(0, 0), (8, 8), (32, 32), (64, 16), (16, 64), (128, 128)]:
        g, e, nan = run(8, 8, 384, 384, d, wl + 1, wr, False)   # window=wl+1 -> window_left=wl
        ok &= g; print(f"  d{d} wl{wl} wr{wr}: relL2={e:.2e} {'OK' if g else ('NAN' if nan else 'FAIL')}")

print("two-sided window + GQA/MQA:")
for (H, Hkv) in [(8, 2), (8, 1)]:
    g, e, nan = run(H, Hkv, 320, 320, 128, 33, 32, False)      # wl=32, wr=32
    ok &= g; print(f"  H{H}/{Hkv} d128 wl32 wr32: relL2={e:.2e} {'OK' if g else ('NAN' if nan else 'FAIL')}")

print("WINDOW:", "ALL OK" if ok else "FAILED")
assert ok
