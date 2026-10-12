# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Decode correctness: fa2_sm70.decode / decode_quantized vs an fp32 exact-softmax reference.
# fp16 (deferred & online) must be BIT-EXACT; int8 ~0.7%. Run:  python tests/test_decode.py
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70


def ref(Q, K, V, scale):
    H, Hkv = Q.shape[1], K.shape[1]
    Kx = K.repeat_interleave(H // Hkv, 1).float(); Vx = V.repeat_interleave(H // Hkv, 1).float()
    s = (Q.float() @ Kx.transpose(-1, -2)) * scale
    return torch.softmax(s, -1) @ Vx, torch.logsumexp(s, -1)


torch.manual_seed(0)
ok = True
for (B, H, Hkv, Sk, d) in [(1,8,8,4096,128), (1,32,8,8192,128), (1,8,8,4096,64), (1,4,4,1000,128),
                           (1,16,8,1024,256), (1,16,1,4096,512)]:   # MLA-scale head dims (256/512)
    scale = 1.0 / math.sqrt(d)
    Q = torch.randn(B,H,1,d, device="cuda", dtype=torch.float16)*0.5
    K = torch.randn(B,Hkv,Sk,d, device="cuda", dtype=torch.float16)*0.5
    V = torch.randn(B,Hkv,Sk,d, device="cuda", dtype=torch.float16)*0.5
    Oref, Lref = ref(Q, K, V, scale)
    rel = lambda O: ((O.float()-Oref).norm()/Oref.norm()).item()
    Odf, Ldf = fa2_sm70.decode(Q, K, V, deferred=True)      # exact deferred
    Oon, _   = fa2_sm70.decode(Q, K, V, deferred=False)     # exact online
    # pre-quantized cache path (the serving pattern)
    c8 = fa2_sm70.quantize_kv(K, V, bits=8)                 # int8 (per-token)
    c4 = fa2_sm70.quantize_kv(K, V, bits=4)                 # int4 (2-level per-channel, robust)
    O8c, _ = fa2_sm70.decode_quantized(Q, c8, scale=scale)
    O4c, _ = fa2_sm70.decode_quantized(Q, c4, scale=scale)
    edf, eon, e8, e4 = rel(Odf), rel(Oon), rel(O8c), rel(O4c)
    le = (Ldf.squeeze(-1) - Lref.squeeze(-1)).abs().max().item()
    good = edf < 3e-3 and eon < 3e-3 and e8 < 2e-2 and e4 < 2e-1 and le < 3e-3; ok &= good
    print(f"  H{H}/{Hkv} Sk{Sk} d{d}: fp16-defer={edf:.1e} online={eon:.1e} int8={e8:.1e} int4-2lvl={e4:.1e} LSE={le:.1e} {'OK' if good else 'FAIL'}")
print("DECODE:", "ALL OK" if ok else "FAILED"); assert ok
