# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Paged decode: fa2_sm70.decode_paged (block-table KV pool) vs the contiguous decode over the same
# logical KV (gathered through the block table) and vs an fp32 softmax reference. Covers GQA (fused
# GF>1) and MHA (GF<=1) paths, V==K, and a partial final block. Run:  python tests/test_paged.py
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70


def gather(Kpool, Vpool, block_table, block_size, kv_len):
    """Logical contiguous KV [B, Hkv, kv_len, D] from the block pool + table."""
    B = block_table.shape[0]; Hkv, D = Kpool.shape[2], Kpool.shape[3]
    nblk = (kv_len + block_size - 1) // block_size
    Kc = torch.empty(B, kv_len, Hkv, D, device=Kpool.device, dtype=Kpool.dtype)
    Vc = torch.empty_like(Kc)
    for b in range(B):
        for lb in range(nblk):
            pb = int(block_table[b, lb]); s = lb * block_size; e = min(s + block_size, kv_len)
            Kc[b, s:e] = Kpool[pb, :e - s]; Vc[b, s:e] = Vpool[pb, :e - s]
    return Kc.transpose(1, 2).contiguous(), Vc.transpose(1, 2).contiguous()   # [B, Hkv, kv_len, D]


def ref_fp32(Q, K, V, scale):
    B, H, _, d = Q.shape; Hkv, Sk = K.shape[1], K.shape[2]
    Kx = K.repeat_interleave(H // Hkv, 1).float(); Vx = V.repeat_interleave(H // Hkv, 1).float()
    s = (Q.float() @ Kx.transpose(-1, -2)) * scale
    return (torch.softmax(s, -1) @ Vx)


torch.manual_seed(0)
ok = True
# (H, Hkv, d, block_size, kv_len, veqk)
cases = [
    (8, 2, 128, 16, 100, False),   # GQA, partial final block (100 = 6*16 + 4)
    (8, 8, 128, 32, 256, False),   # MHA (GF<=1 path)
    (16, 2, 256, 16, 130, False),  # GQA d256, partial
    (8, 1, 64, 32, 200, False),    # MQA
    (8, 2, 512, 8, 96, True),      # GQA d512, V == K (Gemma-4 global)
]
for (H, Hkv, d, block_size, kv_len, veqk) in cases:
    B = 3
    scale = 1.0 / math.sqrt(d)
    nblk = (kv_len + block_size - 1) // block_size
    num_blocks = B * nblk + 5                       # a few spare physical blocks
    Kpool = torch.randn(num_blocks, block_size, Hkv, d, device="cuda", dtype=torch.float16) * 0.5
    Vpool = Kpool if veqk else torch.randn(num_blocks, block_size, Hkv, d, device="cuda", dtype=torch.float16) * 0.5
    # each sequence gets `nblk` distinct physical blocks in shuffled order (non-contiguous)
    perm = torch.randperm(num_blocks, device="cuda")[: B * nblk].view(B, nblk).to(torch.int32)
    block_table = perm
    Q = torch.randn(B, H, 1, d, device="cuda", dtype=torch.float16) * 0.5

    O, _ = fa2_sm70.decode_paged(Q, Kpool, Vpool, block_table, block_size, kv_len, scale=scale)

    Kc, Vc = gather(Kpool, Vpool, block_table, block_size, kv_len)
    Oref_dec, _ = fa2_sm70.decode(Q, Kc, Vc, scale=scale)          # trusted contiguous decode
    Oref_fp32 = ref_fp32(Q, Kc, Vc, scale)                        # independent softmax oracle
    e_dec = (O.float() - Oref_dec.float()).norm().item() / (Oref_dec.float().norm().item() + 1e-9)
    e_ref = (O.float() - Oref_fp32).norm().item() / (Oref_fp32.norm().item() + 1e-9)
    good = e_dec < 5e-3 and e_ref < 5e-3; ok &= good
    print(f"  H{H}/{Hkv} d{d} bs{block_size} kv_len{kv_len} veqk={int(veqk)}: "
          f"vs_decode={e_dec:.2e} vs_fp32={e_ref:.2e} {'OK' if good else 'FAIL'}")
print("PAGED (uniform):", "ALL OK" if ok else "FAILED")

# ---- per-sequence variable lengths (ragged paged batch) --------------------------------------
print("paged ragged (per-sequence seqlens):")
okr = True
for (H, Hkv, d, block_size) in [(8, 2, 128, 16), (8, 8, 64, 32), (16, 2, 256, 16)]:
    scale = 1.0 / math.sqrt(d)
    seqlens = [37, 128, 5, 100]                 # distinct valid lengths per sequence
    B = len(seqlens); mx = max(seqlens)
    nblk_max = (mx + block_size - 1) // block_size
    num_blocks = B * nblk_max + 5
    Kpool = torch.randn(num_blocks, block_size, Hkv, d, device="cuda", dtype=torch.float16) * 0.5
    Vpool = torch.randn(num_blocks, block_size, Hkv, d, device="cuda", dtype=torch.float16) * 0.5
    block_table = torch.randperm(num_blocks, device="cuda")[: B * nblk_max].view(B, nblk_max).to(torch.int32)
    Q = torch.randn(B, H, 1, d, device="cuda", dtype=torch.float16) * 0.5
    O, _ = fa2_sm70.decode_paged(Q, Kpool, Vpool, block_table, block_size, mx, scale=scale, seqlens=seqlens)
    err = 0.0
    for i in range(B):
        Li = seqlens[i]
        nb = (Li + block_size - 1) // block_size
        kparts, vparts = [], []
        for lb in range(nb):
            pb = int(block_table[i, lb]); s = lb * block_size; e = min(s + block_size, Li)
            kparts.append(Kpool[pb, :e - s]); vparts.append(Vpool[pb, :e - s])
        k = torch.cat(kparts, 0).transpose(0, 1); v = torch.cat(vparts, 0).transpose(0, 1)   # [Hkv,Li,d]
        oref = ref_fp32(Q[i:i + 1], k[None], v[None], scale)     # [1,H,1,d]
        err = max(err, (O[i:i + 1].float() - oref).norm().item() / (oref.norm().item() + 1e-9))
    good = err < 5e-3; okr &= good
    print(f"  H{H}/{Hkv} d{d} bs{block_size} seqlens={seqlens}: relL2={err:.2e} {'OK' if good else 'FAIL'}")
print("PAGED (ragged):", "ALL OK" if okr else "FAILED")
assert ok and okr
