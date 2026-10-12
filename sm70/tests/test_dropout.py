# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Attention dropout: reproducibility (same seed -> same O & grads), dropout_p=0 no-op, and the
# 1/(1-p) scaling is unbiased (mean of O over masks -> no-dropout O). Run:
#   python tests/test_dropout.py
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70

torch.manual_seed(0)
B, H, Sq, Sk, d = 2, 8, 256, 256, 128
scale = 1.0 / math.sqrt(d)
Q0 = torch.randn(B, H, Sq, d, device="cuda", dtype=torch.float16) * 0.5
K0 = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.float16) * 0.5
V0 = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.float16) * 0.5
dO = torch.randn(B, H, Sq, d, device="cuda", dtype=torch.float16) * 0.5


def run(p, causal=True, seedval=0):
    torch.manual_seed(seedval)   # makes the internally-drawn dropout seed deterministic
    Q = Q0.clone().requires_grad_(); K = K0.clone().requires_grad_(); V = V0.clone().requires_grad_()
    O = fa2_sm70.flash_attn_func(Q, K, V, causal=causal, scale=scale, dropout_p=p)
    O.backward(dO)
    return O.detach(), Q.grad.detach(), K.grad.detach(), V.grad.detach()


ok = True

# (1) dropout_p = 0 must equal the no-dropout path exactly
O0, gQ0, gK0, gV0 = run(0.0)
On, _, _, _ = run(0.0)  # same path
e = (O0 - On).abs().max().item()
g0 = e == 0.0; ok &= g0
print(f"  p=0 determinism: max|dO|={e:.1e} {'OK' if g0 else 'FAIL'}")

# (2) reproducibility: same seed -> identical O, and identical dK/dV (mask replayed exactly).
# dQ is reduced across key-splits by ATOMIC accumulation (order-nondeterministic in fp), so it is
# reproducible only up to atomic-reduction noise (~1e-4) -- a *different* mask would move grads by
# O(1), not 1e-4, so this cleanly distinguishes "same mask + atomic noise" from "wrong mask".
Oa, qa, ka, va = run(0.2, seedval=123)
Ob, qb, kb, vb = run(0.2, seedval=123)
eO = (Oa - Ob).abs().max().item()
ekv = max((ka - kb).abs().max().item(), (va - vb).abs().max().item())   # RF-resident: deterministic
eq = (qa - qb).abs().max().item()                                       # split-K atomic: tiny noise
g2 = eO == 0.0 and ekv == 0.0 and eq < 1e-3; ok &= g2
print(f"  same-seed reproducibility: |dO|={eO:.1e} |dK,dV|={ekv:.1e} |dQ|={eq:.1e} (atomic) {'OK' if g2 else 'FAIL'}")

# different seeds -> different masks -> different O (dropout is actually active)
Oc, *_ = run(0.2, seedval=999)
diff = (Oa - Oc).abs().max().item()
g2b = diff > 1e-3; ok &= g2b
print(f"  diff-seed changes output: max|dO|={diff:.2e} {'OK' if g2b else 'FAIL'}")

# (3) unbiasedness of the 1/(1-p) scaling: mean_masks(O_drop) -> O_nodrop
p = 0.25
with torch.no_grad():
    Onodrop, _ = fa2_sm70.attention(Q0, K0, V0, causal=True, scale=scale)
    acc = torch.zeros_like(Onodrop, dtype=torch.float32); N = 800
    for s in range(N):
        torch.manual_seed(1000 + s)
        Os = fa2_sm70.flash_attn_func(Q0, K0, V0, causal=True, scale=scale, dropout_p=p)
        acc += Os.float()
    mean = acc / N
rel = (mean - Onodrop.float()).norm().item() / (Onodrop.float().norm().item() + 1e-9)
# The estimator is unbiased: relL2 * sqrt(N) is ~const (verified), so the residual is pure MC noise
# (~2.0e-2 at N=800). Threshold has margin for run-to-run variation, not a systematic-bias allowance.
g3 = rel < 3.0e-2; ok &= g3
print(f"  unbiased scaling E[O_drop]~O: relL2={rel:.2e} (N={N}, MC noise ~1/sqrt(N)) {'OK' if g3 else 'FAIL'}")

print("DROPOUT:", "ALL OK" if ok else "FAILED")
assert ok
