# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Decode-path score modifiers: fa2_sm70.decode(..., softcap=, alibi_slopes=) vs an fp32 reference.
# These ride the ONLINE decode path (running max) -- the deferred path's Cauchy-Schwarz U-bound does
# not bound modified scores, so decode() routes modifiers there. Run:  python tests/test_decode_mod.py
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70


def ref(q, K, V, scale, cap, slopes):
    H, Hkv, Sk = q.shape[1], K.shape[1], K.shape[2]
    Kx = K.repeat_interleave(H // Hkv, 1).float(); Vx = V.repeat_interleave(H // Hkv, 1).float()
    s = (q.float() @ Kx.transpose(-1, -2)) * scale
    if cap > 0:
        s = cap * torch.tanh(s / cap)                              # softcap first (matches the kernel)
    if slopes is not None:
        j = torch.arange(Sk, device=q.device).view(1, 1, 1, Sk)
        s = s + slopes.view(1, H, 1, 1).float() * (j - (Sk - 1)).float()   # ALiBi: query at position Sk-1
    return torch.softmax(s, -1) @ Vx


torch.manual_seed(0)
ok = True
for d in [64, 128, 256, 512]:
    for (H, Hkv, Sk) in [(8, 8, 512), (8, 2, 512), (8, 1, 300)]:
        scale = 1.0 / math.sqrt(d)
        q = torch.randn(1, H, 1, d, device="cuda", dtype=torch.float16) * 0.5
        K = torch.randn(1, Hkv, Sk, d, device="cuda", dtype=torch.float16) * 0.5
        V = torch.randn(1, Hkv, Sk, d, device="cuda", dtype=torch.float16) * 0.5
        slopes = -(2.0 ** (-8.0 * (torch.arange(H, device="cuda") + 1) / H)).float()
        for tag, cap, sl in [("softcap", 30.0, None), ("alibi", 0.0, slopes), ("both", 30.0, slopes)]:
            O, _ = fa2_sm70.decode(q, K, V, scale=scale, softcap=cap, alibi_slopes=sl)
            Oref = ref(q, K, V, scale, cap, sl)
            nan = torch.isnan(O).any().item()
            e = float("nan") if nan else (O.float() - Oref).norm().item() / (Oref.norm().item() + 1e-9)
            good = (not nan) and e < 3e-3; ok &= good
            print(f"  d{d} H{H}/{Hkv} Sk{Sk} {tag}: relL2={e:.2e} {'OK' if good else ('NAN' if nan else 'FAIL')}")

# modifiers default off -> unchanged vs the plain deferred decode
q = torch.randn(1, 8, 1, 128, device="cuda", dtype=torch.float16) * 0.5
K = torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.float16) * 0.5
V = torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.float16) * 0.5
O0, _ = fa2_sm70.decode(q, K, V)
O1, _ = fa2_sm70.decode(q, K, V, softcap=0.0, alibi_slopes=None)
noop = (O0 - O1).abs().max().item(); ok &= (noop == 0.0)
print(f"  modifiers-off no-op: max|dO|={noop:.1e} {'OK' if noop == 0.0 else 'FAIL'}")

print("DECODE-MOD:", "ALL OK" if ok else "FAILED")
assert ok
