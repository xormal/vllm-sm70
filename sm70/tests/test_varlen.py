# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Variable-length / packed attention: fa2_sm70.attention_varlen vs a per-sequence fp32 reference.
# Independent Q/K lengths per sequence, causal (bottom-right) + non-causal, MHA/GQA/MQA. Run:
#   python tests/test_varlen.py
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70


def ref_one(q, k, v, scale, causal):
    """fp32 exact attention for one sequence: q [Hq,Sq,d], k/v [Hkv,Sk,d]."""
    Hq, Sq, d = q.shape; Hkv, Sk = k.shape[0], k.shape[1]
    kx = k.repeat_interleave(Hq // Hkv, 0).float(); vx = v.repeat_interleave(Hq // Hkv, 0).float()
    s = (q.float() @ kx.transpose(-1, -2)) * scale
    if causal:
        off = Sk - Sq
        qi = torch.arange(Sq, device=q.device).view(Sq, 1); ki = torch.arange(Sk, device=q.device).view(1, Sk)
        s = s.masked_fill(ki > qi + off, float("-inf"))
    return torch.softmax(s, -1) @ vx


torch.manual_seed(0)
ok = True
# (H, Hkv, d, causal, [ (Sq, Sk) per sequence ])
cases = [
    (8, 8, 64,  False, [(10, 10), (37, 37), (128, 128), (5, 5)]),           # MHA non-causal, ragged
    (8, 8, 128, True,  [(64, 64), (200, 200), (17, 17)]),                   # MHA causal
    (8, 2, 128, True,  [(96, 96), (256, 256), (33, 33), (1, 1)]),           # GQA causal
    (8, 1, 64,  True,  [(48, 48), (300, 300)]),                             # MQA causal
    (4, 4, 128, True,  [(16, 64), (50, 120)]),                              # Sq != Sk (cross), bottom-right
    (8, 2, 256, True,  [(64, 64), (200, 200), (33, 33)]),                   # d>128 varlen (Gemma-4 local scale)
]
for (H, Hkv, d, causal, lens) in cases:
    scale = 1.0 / math.sqrt(d)
    sq = [a for a, _ in lens]; sk = [b for _, b in lens]
    Tq, Tk = sum(sq), sum(sk)
    Q = torch.randn(Tq, H, d, device="cuda", dtype=torch.float16) * 0.5
    K = torch.randn(Tk, Hkv, d, device="cuda", dtype=torch.float16) * 0.5
    V = torch.randn(Tk, Hkv, d, device="cuda", dtype=torch.float16) * 0.5
    cu_q = torch.tensor([0] + list(torch.tensor(sq).cumsum(0)), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(sk).cumsum(0)), device="cuda", dtype=torch.int32)
    O, L = fa2_sm70.attention_varlen(Q, K, V, cu_q, cu_k, max(sq), max(sk), causal=causal, scale=scale)
    # per-sequence reference
    err = 0.0
    for i in range(len(lens)):
        q = Q[cu_q[i]:cu_q[i + 1]].transpose(0, 1)   # [H,Sq,d]
        k = K[cu_k[i]:cu_k[i + 1]].transpose(0, 1); v = V[cu_k[i]:cu_k[i + 1]].transpose(0, 1)
        oref = ref_one(q, k, v, scale, causal).transpose(0, 1)   # [Sq,H,d]
        oi = O[cu_q[i]:cu_q[i + 1]].float()
        err = max(err, (oi - oref).norm().item() / (oref.norm().item() + 1e-9))
    good = err < 3e-3; ok &= good
    print(f"  H{H}/{Hkv} d{d} causal={int(causal)} nseq={len(lens)} Tq={Tq} Tk={Tk}: relL2={err:.2e} {'OK' if good else 'FAIL'}")
print("VARLEN FWD:", "ALL OK" if ok else "FAILED")

# ---- backward (differentiable varlen) --------------------------------------------------------
print("varlen backward:")
okb = True
for (H, Hkv, d, causal, lens) in [c for c in cases if c[2] <= 512]:
    scale = 1.0 / math.sqrt(d)
    sq = [a for a, _ in lens]; sk = [b for _, b in lens]
    Tq, Tk = sum(sq), sum(sk)
    Q = (torch.randn(Tq, H, d, device="cuda", dtype=torch.float16) * 0.5).requires_grad_()
    K = (torch.randn(Tk, Hkv, d, device="cuda", dtype=torch.float16) * 0.5).requires_grad_()
    V = (torch.randn(Tk, Hkv, d, device="cuda", dtype=torch.float16) * 0.5).requires_grad_()
    dO = torch.randn(Tq, H, d, device="cuda", dtype=torch.float16) * 0.5
    cu_q = torch.tensor([0] + list(torch.tensor(sq).cumsum(0)), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(sk).cumsum(0)), device="cuda", dtype=torch.int32)
    O = fa2_sm70.flash_attn_varlen_func(Q, K, V, cu_q, cu_k, max(sq), max(sk), causal=causal, scale=scale)
    O.backward(dO)
    dQ, dK, dV = Q.grad.float(), K.grad.float(), V.grad.float()
    # per-sequence fp32 reference grads
    dQr = torch.zeros_like(dQ); dKr = torch.zeros_like(dK); dVr = torch.zeros_like(dV)
    for i in range(len(lens)):
        q = Q[cu_q[i]:cu_q[i + 1]].detach().float().transpose(0, 1).requires_grad_()
        k = K[cu_k[i]:cu_k[i + 1]].detach().float().transpose(0, 1).requires_grad_()
        v = V[cu_k[i]:cu_k[i + 1]].detach().float().transpose(0, 1).requires_grad_()
        o = ref_one(q, k, v, scale, causal).transpose(0, 1)   # [Sq,H,d]
        o.backward(dO[cu_q[i]:cu_q[i + 1]].float())
        dQr[cu_q[i]:cu_q[i + 1]] = q.grad.transpose(0, 1); dKr[cu_k[i]:cu_k[i + 1]] = k.grad.transpose(0, 1)
        dVr[cu_k[i]:cu_k[i + 1]] = v.grad.transpose(0, 1)
    r = lambda a, b: (a - b).norm().item() / (b.norm().item() + 1e-9)
    qe, ke, ve = r(dQ, dQr), r(dK, dKr), r(dV, dVr)
    good = qe < 2e-2 and ke < 2e-2 and ve < 2e-2; okb &= good
    print(f"  H{H}/{Hkv} d{d} causal={int(causal)} nseq={len(lens)}: dQ={qe:.2e} dK={ke:.2e} dV={ve:.2e} {'OK' if good else 'FAIL'}")
print("VARLEN BWD:", "ALL OK" if okb else "FAILED")
assert ok and okb
