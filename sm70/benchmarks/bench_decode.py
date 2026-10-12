# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Decode benchmark (Sq=1): fa2_sm70.decode variants vs torch SDPA (default) vs SDPA forced onto
# SDPBackend.EFFICIENT_ATTENTION (cutlass FMHA -- prefill-oriented, parallelism-starved at Sq=1).
# Reports latency, speedup vs SDPA-eff, and relL2 accuracy vs an fp32 exact reference.
# Run:  python benchmarks/bench_decode.py
import math, torch
import torch.nn.functional as F
from _common import timed, relL2, expand_kv, efficient_backend
import fa2_sm70


def ref(Q, K, V, scale):
    H, Hkv = Q.shape[1], K.shape[1]
    Kx = K.repeat_interleave(H // Hkv, 1).float(); Vx = V.repeat_interleave(H // Hkv, 1).float()
    s = (Q.float() @ Kx.transpose(-1, -2)) * scale
    return torch.softmax(s, -1) @ Vx


torch.manual_seed(0)
print(f"{'H/Hkv,Sk,d':<16}{'SDPA':>8}{'SDPA-eff':>9}{'def-f16':>9}{'def-i8':>8}  | vs SDPA-eff: f16 / i8   relL2 f16/i8")
print("-" * 92)
for (H, Hkv, Sk, d) in [(32,32,2048,128), (32,32,8192,128), (32,8,8192,128), (32,32,32768,128), (8,8,16384,128), (32,32,4096,64)]:
    B = 1; scale = 1.0 / math.sqrt(d)
    Q = torch.randn(B,H,1,d, device="cuda", dtype=torch.float16)*0.5
    K = torch.randn(B,Hkv,Sk,d, device="cuda", dtype=torch.float16)*0.5
    V = torch.randn(B,Hkv,Sk,d, device="cuda", dtype=torch.float16)*0.5
    Ks, Vs = expand_kv(K, V, H)
    Oref = ref(Q, K, V, scale)
    # a real KV cache maintains these incrementally as keys are appended, so precompute them (NOT
    # part of the per-step decode cost): k_absmax (the U bound) and the int8-quantized cache.
    kabs = K.float().norm(dim=-1).amax(dim=-1)
    c8 = fa2_sm70.quantize_kv(K, V, bits=8)
    ts = timed(lambda: F.scaled_dot_product_attention(Q, Ks, Vs, scale=scale))
    with efficient_backend():
        te = timed(lambda: F.scaled_dot_product_attention(Q, Ks, Vs, scale=scale))
    tdf = timed(lambda: fa2_sm70.decode(Q, K, V, scale=scale, kv_dtype="fp16", deferred=True, k_absmax=kabs)[0])
    tdi = timed(lambda: fa2_sm70.decode_quantized(Q, c8, scale=scale)[0])
    Odf = fa2_sm70.decode(Q, K, V, scale=scale, k_absmax=kabs)[0]
    Odi = fa2_sm70.decode_quantized(Q, c8, scale=scale)[0]
    us = lambda t: f"{t*1e6:>7.0f}"
    print(f"{f'{H}/{Hkv},{Sk},{d}':<16}{us(ts)}{us(te)}{us(tdf)}{us(tdi)}  | {te/tdf:>5.1f}x /{te/tdi:>5.1f}x   {relL2(Odf,Oref):.0e}/{relL2(Odi,Oref):.0e}")
print("\nnote: SDPA-eff (cutlass FMHA) is a PREFILL kernel; at Sq=1 it is parallelism-starved, so the")
print("split-KV decode kernels win by multiples that grow with context length and with fewer heads.")
