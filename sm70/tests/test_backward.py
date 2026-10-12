# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Backward correctness: fa2_sm70.flash_attn_func gradients vs an fp32 exact-softmax reference
# (independent autograd oracle), across causal/non-causal, Sq!=Sk, MHA/GQA/MQA. Run:
#   python tests/test_backward.py
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70


def ref_attn(Q, K, V, scale, causal):
    """fp32 exact-softmax attention with grouped-KV expand; differentiable oracle."""
    B, H, Sq, d = Q.shape; Hkv, Sk = K.shape[1], K.shape[2]
    Kx = K.repeat_interleave(H // Hkv, 1); Vx = V.repeat_interleave(H // Hkv, 1)
    s = (Q @ Kx.transpose(-1, -2)) * scale
    if causal:
        off = Sk - Sq
        qi = torch.arange(Sq, device=Q.device).view(Sq, 1)
        ki = torch.arange(Sk, device=Q.device).view(1, Sk)
        s = s.masked_fill(ki > qi + off, float("-inf"))
        # [СЛЕПЫЕ СТРОКИ] При Sq > Sk первые Sq-Sk запросов не видят НИ ОДНОГО ключа: softmax по
        # строке из одних -inf не определён (torch SDPA там тоже выдаёт NaN). Обнуляем такие строки,
        # чтобы оракул оставался конечным и дифференцируемым; сравнение по ним всё равно не ведётся.
        blind = (ki > qi + off).all(-1, keepdim=True)
        s = s.masked_fill(blind, 0.0)
        p = torch.softmax(s, -1).masked_fill(blind, 0.0)
        return p @ Vx
    return torch.softmax(s, -1) @ Vx


def rel(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-20)


def rows_defined(x, Sq, Sk, causal):
    """Отбрасывает СЛЕПЫЕ строки запросов (Sq > Sk при причинности): там значение не определено."""
    return x[:, :, Sq - Sk:] if (causal and Sq > Sk) else x


torch.manual_seed(0)
ok = True
cases = [
    # B, H, Hkv, Sq, Sk, d, causal
    (1, 8, 8, 256, 256, 64, False),   # MHA non-causal
    (2, 8, 8, 192, 192, 128, True),   # MHA causal
    (1, 8, 2, 256, 256, 128, True),   # GQA causal
    (1, 8, 1, 320, 320, 64, True),    # MQA causal
    (1, 4, 4, 128, 384, 128, True),   # Sq != Sk (cross-attn / prefix), causal bottom-right
    # [REGRESSION Sq>Sk] Causal bottom-right with Sq > Sk: the leading Sq-Sk queries see NO key at all,
    # so the FIRST query tile is empty. The fragment-order dQ fold captured its permutation map "on tile
    # 0" -- that tile -- so the map stayed zero and dQ came out ALL ZEROS, silently, with dK/dV and the
    # forward output all correct. Found by a sequence-length sweep, fixed by excluding Sq>Sk from
    # fo_elig. These shapes pin it.
    (1, 4, 4, 256, 128, 128, True),   # Sq > Sk, causal: one fully-blind query tile
    (1, 4, 4, 384, 256, 128, True),   # Sq > Sk, causal: two query tiles, one blind
    (1, 4, 4, 512, 256, 64, True),    # same at d=64 (the other fragment-order head dim)
    # REGRESSION GATE. B*H=1 with d=256 and S>=2048 produced dQ/dK wrong by 30-40% while dV stayed
    # correct. The suite missed it because its only d=256 shapes were H8/8 S256 and H8/2 S192 --
    # both with B*H>1 and a short sequence, so neither corner was ever exercised together.
    (1, 1, 1, 2048, 2048, 256, True),
    (1, 4, 4, 256, 128, 256, True),   # same at d=256 (gmem-workspace path, no fold)
    (1, 4, 4, 96, 96, 80, False),     # non-multiple-of-8-ish head (padded), non-causal
    (1, 8, 8, 256, 256, 256, True),   # d>128: gmem output-accum backward (Gemma-4 local scale)
    (1, 8, 2, 192, 192, 256, True),   # d256 GQA
    (1, 4, 1, 256, 256, 512, True),   # d512 MQA (Gemma-4 global scale) -- fine-tuning path
    # --- partial-query-block guards (Sq NOT a multiple of 128): the R7 whole-B dQ transpose mis-computes the
    #     last ~32 queries of a partially-entered high 32-strip; these route to the exact direct GradQ. The
    #     earlier cases only had size-64 partials (strip-aligned) and MISSED this -- keep both strip parities. ---
    (2, 8, 8, 447, 447, 128, True),   # d128 partial high-strip (447%128=63), causal
    (2, 8, 8, 500, 500, 128, False),  # d128 partial high-strip (500%128=116), NON-causal (was the worst)
    (1, 8, 2, 700, 700, 128, True),   # d128 GQA partial (700%128=60)
    (1, 8, 1, 999, 999, 64,  True),   # d64 MQA partial high-strip (999%128=103) -- d<=64 is R7 too
    (1, 4, 4, 163, 384, 128, True),   # Sq!=Sk cross-attn, partial Sq (163%128=35)
]
for (B, H, Hkv, Sq, Sk, d, causal) in cases:
    scale = 1.0 / math.sqrt(d)
    Q = (torch.randn(B, H, Sq, d, device="cuda", dtype=torch.float16) * 0.5).requires_grad_()
    K = (torch.randn(B, Hkv, Sk, d, device="cuda", dtype=torch.float16) * 0.5).requires_grad_()
    V = (torch.randn(B, Hkv, Sk, d, device="cuda", dtype=torch.float16) * 0.5).requires_grad_()
    dO = torch.randn(B, H, Sq, d, device="cuda", dtype=torch.float16) * 0.5

    # ours
    O = fa2_sm70.flash_attn_func(Q, K, V, causal=causal, scale=scale)
    O.backward(dO)
    dQ, dK, dV = Q.grad.detach().float(), K.grad.detach().float(), V.grad.detach().float()
    Oref = ref_attn(Q.detach().float(), K.detach().float(), V.detach().float(), scale, causal)
    oe = rel(rows_defined(O.detach().float(), Sq, Sk, causal), rows_defined(Oref, Sq, Sk, causal))

    # fp32 reference grads
    Qr = Q.detach().float().requires_grad_(); Kr = K.detach().float().requires_grad_(); Vr = V.detach().float().requires_grad_()
    Or = ref_attn(Qr, Kr, Vr, scale, causal)
    Or.backward(dO.float())
    qe = rel(rows_defined(dQ, Sq, Sk, causal), rows_defined(Qr.grad, Sq, Sk, causal))
    ke, ve = rel(dK, Kr.grad), rel(dV, Vr.grad)
    good = oe < 3e-3 and qe < 2e-2 and ke < 2e-2 and ve < 2e-2
    ok &= good
    print(f"  H{H}/{Hkv} S{Sq}/{Sk} d{d} causal={int(causal)}: "
          f"O={oe:.2e} dQ={qe:.2e} dK={ke:.2e} dV={ve:.2e} {'OK' if good else 'FAIL'}")
    Q.grad = K.grad = V.grad = None

print("BACKWARD:", "ALL OK" if ok else "FAILED")
assert ok
