# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Stateful quantized KV cache: fa2_sm70.QuantizedKVCache (keeps int8/int4 bytes across steps). Verifies
# (a) incremental append+decode is BIT-EXACT vs a one-shot quantize_kv (per-token) + decode_quantized
# over the same prefix, and (b) int8 is ~0.7% vs an fp16 exact reference. Run:
#   python tests/test_quant_cache.py
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70


def ref_fp32(q, K, V, scale):
    H, Hkv = q.shape[1], K.shape[1]
    Kx = K.repeat_interleave(H // Hkv, 1).float(); Vx = V.repeat_interleave(H // Hkv, 1).float()
    s = (q.float() @ Kx.transpose(-1, -2)) * scale
    return torch.softmax(s, -1) @ Vx


torch.manual_seed(0)
ok = True
for (B, H, Hkv, D, bits) in [(1, 8, 8, 128, 8), (1, 8, 2, 128, 8), (2, 8, 1, 64, 8), (1, 8, 8, 128, 4)]:
    Sk, maxlen = 600, 1024
    scale = 1.0 / math.sqrt(D)
    K = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.5
    V = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.5
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16) * 0.5

    # incremental quantized cache: append in ragged chunks
    cache = fa2_sm70.QuantizedKVCache(B, Hkv, maxlen, D, device="cuda", bits=bits)
    for s in range(0, Sk, 128):
        e = min(s + 128, Sk)
        cache.append(K[:, :, s:e], V[:, :, s:e])
    O_cache, _ = cache.decode(q, scale=scale)

    # (a) bit-exact vs one-shot per-token quantize + decode over the SAME prefix
    cref = fa2_sm70.quantize_kv(K, V, bits=bits, per_channel=False)
    O_oneshot, _ = fa2_sm70.decode_quantized(q, cref, scale=scale)
    bitexact = (O_cache - O_oneshot).abs().max().item()

    # (b) vs fp16 exact reference (int8 ~0.7%, int4 coarser)
    Oref = ref_fp32(q, K, V, scale)
    e_ref = (O_cache.float() - Oref).norm().item() / (Oref.norm().item() + 1e-9)
    tol = 1.5e-2 if bits == 8 else 2.0e-1
    good = bitexact == 0.0 and e_ref < tol
    ok &= good
    print(f"  B{B} H{H}/{Hkv} D{D} int{bits}: bitexact_vs_oneshot={bitexact:.1e} vs_fp16={e_ref:.2e} "
          f"(tol {tol}) {'OK' if good else 'FAIL'}")

print("QUANT_CACHE:", "ALL OK" if ok else "FAILED")
assert ok
