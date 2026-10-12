# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ALiBi per-head linear position bias: fa2_sm70.attention(..., alibi_slopes) vs an fp32 reference that
# adds slope_h*(key - query) to the scaled logits. Covers causal/non-causal, MHA/GQA, and checks that
# alibi_slopes=None is a bit-exact no-op. Run:  python tests/test_alibi.py
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70


def ref(Q, K, V, scale, causal, slopes):
    B, H, Sq, d = Q.shape; Hkv, Sk = K.shape[1], K.shape[2]
    Kx = K.repeat_interleave(H // Hkv, 1).float(); Vx = V.repeat_interleave(H // Hkv, 1).float()
    s = (Q.float() @ Kx.transpose(-1, -2)) * scale
    if slopes is not None:
        i = torch.arange(Sq, device=Q.device).view(1, 1, Sq, 1)
        j = torch.arange(Sk, device=Q.device).view(1, 1, 1, Sk)
        s = s + slopes.view(1, H, 1, 1).float() * (j - i).float()       # slope*(key - query)
    if causal:
        off = Sk - Sq
        qi = torch.arange(Sq, device=Q.device).view(Sq, 1); ki = torch.arange(Sk, device=Q.device).view(1, Sk)
        s = s.masked_fill(ki > qi + off, float("-inf"))
    return torch.softmax(s, -1) @ Vx


torch.manual_seed(0)
ok = True
cases = [
    (8, 8, 256, 256, 64, False),
    (8, 8, 320, 320, 128, True),
    (8, 2, 256, 256, 128, True),    # GQA
    (8, 1, 192, 192, 64, True),     # MQA
    (4, 4, 96, 256, 128, True),     # Sq != Sk (bottom-right)
]
for (H, Hkv, Sq, Sk, d, causal) in cases:
    scale = 1.0 / math.sqrt(d)
    Q = torch.randn(1, H, Sq, d, device="cuda", dtype=torch.float16) * 0.5
    K = torch.randn(1, Hkv, Sk, d, device="cuda", dtype=torch.float16) * 0.5
    V = torch.randn(1, Hkv, Sk, d, device="cuda", dtype=torch.float16) * 0.5
    # standard-ish ALiBi geometric slopes (negative), plus jitter so heads differ
    slopes = -(2.0 ** (-8.0 * (torch.arange(H, device="cuda") + 1) / H)).float()
    O, _ = fa2_sm70.attention(Q, K, V, causal=causal, scale=scale, alibi_slopes=slopes)
    Oref = ref(Q, K, V, scale, causal, slopes)
    e = (O.float() - Oref).norm().item() / (Oref.norm().item() + 1e-9)
    # alibi=None must exactly equal the plain no-alibi path
    Onone, _ = fa2_sm70.attention(Q, K, V, causal=causal, scale=scale)
    Oplain, _ = fa2_sm70.attention(Q, K, V, causal=causal, scale=scale, alibi_slopes=None)
    noop = (Onone.float() - Oplain.float()).abs().max().item()
    good = e < 3e-3 and noop == 0.0; ok &= good
    print(f"  H{H}/{Hkv} S{Sq}/{Sk} d{d} causal={int(causal)}: alibi_relL2={e:.2e} none_noop={noop:.1e} {'OK' if good else 'FAIL'}")
print("ALIBI:", "ALL OK" if ok else "FAILED")
assert ok
