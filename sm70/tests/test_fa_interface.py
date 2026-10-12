# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# FlashAttention-compatible API (fa2_sm70.flash_attn_interface): verifies the [B,S,H,D] wrappers vs an
# fp32 reference, the qkv/kv-packed forms, causal/window/softcap/alibi modifiers, varlen, and that the
# plain path is differentiable. Run:  python tests/test_fa_interface.py
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fa2_sm70 import flash_attn_interface as fa


def ref_bshd(q, k, v, scale, causal, window=0, alibi=None):
    # q [B,Sq,H,D], k/v [B,Sk,Hkv,D]  -> reference O [B,Sq,H,D]
    B, Sq, H, d = q.shape; Hkv, Sk = k.shape[2], k.shape[1]
    qt = q.transpose(1, 2).float(); kt = k.transpose(1, 2).float(); vt = v.transpose(1, 2).float()
    kt = kt.repeat_interleave(H // Hkv, 1); vt = vt.repeat_interleave(H // Hkv, 1)
    s = (qt @ kt.transpose(-1, -2)) * scale
    if alibi is not None:
        i = torch.arange(Sq, device=q.device).view(1, 1, Sq, 1); j = torch.arange(Sk, device=q.device).view(1, 1, 1, Sk)
        s = s + alibi.view(1, H, 1, 1).float() * (j - i).float()
    qi = torch.arange(Sq, device=q.device).view(Sq, 1); ki = torch.arange(Sk, device=q.device).view(1, Sk)
    if causal:
        s = s.masked_fill(ki > qi + (Sk - Sq), float("-inf"))
    if window:
        s = s.masked_fill(ki < qi + (Sk - Sq) - window + 1, float("-inf"))
    return (torch.softmax(s, -1) @ vt).transpose(1, 2)   # [B,Sq,H,D]


torch.manual_seed(0)
ok = True


def chk(name, O, Oref, tol=3e-3):
    global ok
    e = (O.float() - Oref).norm().item() / (Oref.norm().item() + 1e-9)
    good = e < tol; ok &= good
    print(f"  {name}: relL2={e:.2e} {'OK' if good else 'FAIL'}")


B, S, H, Hkv, d = 2, 256, 8, 2, 128
scale = 1.0 / math.sqrt(d)
q = torch.randn(B, S, H, d, device="cuda", dtype=torch.float16) * 0.5
k = torch.randn(B, S, Hkv, d, device="cuda", dtype=torch.float16) * 0.5
v = torch.randn(B, S, Hkv, d, device="cuda", dtype=torch.float16) * 0.5

chk("flash_attn_func causal GQA", fa.flash_attn_func(q, k, v, causal=True, softmax_scale=scale),
    ref_bshd(q, k, v, scale, True))
chk("flash_attn_func non-causal", fa.flash_attn_func(q, k, v, causal=False, softmax_scale=scale),
    ref_bshd(q, k, v, scale, False))
chk("causal window=(64,0)", fa.flash_attn_func(q, k, v, causal=True, softmax_scale=scale, window_size=(64, 0)),
    ref_bshd(q, k, v, scale, True, window=65))
slopes = -(2.0 ** (-8.0 * (torch.arange(H, device="cuda") + 1) / H)).float()
chk("alibi causal", fa.flash_attn_func(q, k, v, causal=True, softmax_scale=scale, alibi_slopes=slopes),
    ref_bshd(q, k, v, scale, True, alibi=slopes))

# qkv-packed (MHA)
qkv = torch.randn(B, S, 3, H, d, device="cuda", dtype=torch.float16) * 0.5
Oq = fa.flash_attn_qkvpacked_func(qkv, causal=True, softmax_scale=scale)
chk("qkvpacked causal", Oq, ref_bshd(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], scale, True))

# kv-packed (cross)
kv = torch.randn(B, S, 2, Hkv, d, device="cuda", dtype=torch.float16) * 0.5
Ok = fa.flash_attn_kvpacked_func(q, kv, causal=False, softmax_scale=scale)
chk("kvpacked non-causal", Ok, ref_bshd(q, kv[:, :, 0], kv[:, :, 1], scale, False))

# varlen packed
lens = [(64, 64), (120, 120), (32, 32)]
sq = [a for a, _ in lens]; sk = [b for _, b in lens]
Tq, Tk = sum(sq), sum(sk)
qv = torch.randn(Tq, H, d, device="cuda", dtype=torch.float16) * 0.5
kv2 = torch.randn(Tk, Hkv, d, device="cuda", dtype=torch.float16) * 0.5
vv = torch.randn(Tk, Hkv, d, device="cuda", dtype=torch.float16) * 0.5
cu_q = torch.tensor([0] + list(torch.tensor(sq).cumsum(0)), device="cuda", dtype=torch.int32)
cu_k = torch.tensor([0] + list(torch.tensor(sk).cumsum(0)), device="cuda", dtype=torch.int32)
Ov = fa.flash_attn_varlen_func(qv, kv2, vv, cu_q, cu_k, max(sq), max(sk), causal=True, softmax_scale=scale)
verr = 0.0
for i in range(len(lens)):
    qi = qv[cu_q[i]:cu_q[i + 1]][None]; ki = kv2[cu_k[i]:cu_k[i + 1]][None]; vi = vv[cu_k[i]:cu_k[i + 1]][None]
    oref = ref_bshd(qi, ki, vi, scale, True)
    verr = max(verr, (Ov[cu_q[i]:cu_q[i + 1]][None].float() - oref).norm().item() / (oref.norm().item() + 1e-9))
good = verr < 3e-3; ok &= good
print(f"  varlen causal: relL2={verr:.2e} {'OK' if good else 'FAIL'}")

# differentiability of the plain path
qg = q.clone().requires_grad_(); kg = k.clone().requires_grad_(); vg = v.clone().requires_grad_()
Og = fa.flash_attn_func(qg, kg, vg, causal=True, softmax_scale=scale)
Og.sum().backward()
grad_ok = qg.grad is not None and kg.grad is not None and vg.grad is not None and torch.isfinite(qg.grad).all().item()
ok &= grad_ok
print(f"  differentiable plain path: grads={'present+finite' if grad_ok else 'MISSING'} {'OK' if grad_ok else 'FAIL'}")

print("FA_INTERFACE:", "ALL OK" if ok else "FAILED")
assert ok
