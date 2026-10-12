# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Prefill at head_dim=256 -- the geometry of the hybrid-attention models this kernel is deployed on
# (Qwen3.5: 24 heads / 4 KV, d=256, full attention on every 4th layer). Three columns:
#   ours      the hand-written Volta block mainloop (fa2_sm70/csrc/volta_fwd_block.cuh)
#   cutlass   THIS repo's previous path -- cutlass FMHA + value split (FA2SM70_NO_VOLTA_FWD=1)
#   SDPA-eff  torch SDPBackend.EFFICIENT_ATTENTION, the vendor cutlass fmha_cutlassF_sm70 kernel
#
# WHY THE CHUNKED SHAPES ARE IN A SEPARATE TABLE. torch's `is_causal=True` aligns the mask to the
# TOP-LEFT, so with Sq < Sk it computes only the first Sq keys: its time stops growing with context
# and it reports impossible throughput (602 TFLOP/s against a 93.6 peak on this card). Chunked
# prefill needs BOTTOM-RIGHT alignment, so torch is simply not measuring the same thing and is left
# out of that table rather than quoted misleadingly.
import os, sys, math, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70
from torch.nn.attention import sdpa_kernel, SDPBackend

D = 256
SQUARE = [(1, 32, 2048), (1, 16, 4096), (1, 24, 4096), (2, 8, 4096), (1, 12, 8192), (1, 4, 8192)]
CHUNKED = [(12, 2, 8192, 16384), (12, 2, 8192, 32768), (12, 2, 8192, 65536),
           (24, 4, 8192, 16384), (24, 4, 8192, 65536)]


def bench(fn, it=8):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / it


def volta(on):                       # the gate reads getenv per call, so this toggles in-process
    if on: os.environ.pop("FA2SM70_NO_VOLTA_FWD", None)
    else:  os.environ["FA2SM70_NO_VOLTA_FWD"] = "1"


ext = fa2_sm70._ext.prefill_ext()
print(f"head_dim={D}, causal, fp16, V100-SXM2-32GB\n")
print("Square (Sq == Sk) -- all three paths compute the same thing")
print(f'{"B,H,S":>14} {"ours ms":>9} {"cutlass":>9} {"SDPA-eff":>9} | {"vs cutlass":>10} {"vs SDPA-eff":>11} {"%roofline":>10}  relL2')
for (B, H, S) in SQUARE:
    torch.manual_seed(S + H); sc = 1.0 / math.sqrt(D)
    q = (torch.randn(B, S, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    k = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    qb = q.transpose(1, 2).contiguous()
    volta(True);  o1 = ext.attn_fwd_qbshd(q, k, v, sc, True)[0]; t1 = bench(lambda: ext.attn_fwd_qbshd(q, k, v, sc, True))
    volta(False); o2 = ext.attn_fwd_qbshd(q, k, v, sc, True)[0]; t2 = bench(lambda: ext.attn_fwd_qbshd(q, k, v, sc, True))
    volta(True)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        oe = torch.nn.functional.scaled_dot_product_attention(qb, k, v, is_causal=True, scale=sc)
        t3 = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qb, k, v, is_causal=True, scale=sc))
    rel = ((o1.float() - oe.transpose(1, 2).float()).norm() / oe.float().norm()).item()
    fl = 4 * (S * (S + 1) / 2) * H * D * B
    print(f'{B},{H},{S:<8} {t1*1e3:>9.3f} {t2*1e3:>9.3f} {t3*1e3:>9.3f} | {t2/t1:>9.2f}x {t3/t1:>10.2f}x '
          f'{fl/t1/1e12/93.6*100:>9.1f}%  {rel:.1e}', flush=True)
    del q, k, v, qb, o1, o2, oe; torch.cuda.empty_cache()

print("\nChunked prefill (Sq < Sk, bottom-right causal) -- torch omitted, see the header")
print(f'{"H/Hkv":>7} {"Sq x Sk":>14} {"ours ms":>9} {"cutlass":>9} | {"speedup":>8} {"TFLOP/s":>8}  relL2')
for (H, Hkv, Sq, Sk) in CHUNKED:
    torch.manual_seed(Sk + H); sc = 1.0 / math.sqrt(D)
    q = (torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    k = (torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    v = (torch.randn(1, Hkv, Sk, D, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    volta(True);  o1 = ext.attn_fwd_qbshd(q, k, v, sc, True)[0]; t1 = bench(lambda: ext.attn_fwd_qbshd(q, k, v, sc, True), 5)
    volta(False); o2 = ext.attn_fwd_qbshd(q, k, v, sc, True)[0]; t2 = bench(lambda: ext.attn_fwd_qbshd(q, k, v, sc, True), 5)
    volta(True)
    rel = ((o1.float() - o2.float()).norm() / o2.float().norm()).item()
    fl = 4 * (Sq * (Sk - Sq) + Sq * (Sq + 1) / 2) * H * D
    print(f'{H}/{Hkv:<5} {Sq}x{Sk:<8} {t1*1e3:>9.2f} {t2*1e3:>9.2f} | {t2/t1:>7.2f}x {fl/t1/1e12:>8.1f}  {rel:.1e}', flush=True)
    del q, k, v, o1, o2; torch.cuda.empty_cache()
