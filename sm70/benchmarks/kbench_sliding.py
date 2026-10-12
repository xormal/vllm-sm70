#!/usr/bin/env python
"""Head-to-head KERNEL bench on Gemma-4 sliding-layer shapes (d=256, GQA 16/8, causal sliding window
1024) -- the shapes FlashAttention-2 *can* run. ATTN=fa2_sm70 (our sm70 kernel, V100) vs ATTN=flash
(real flash_attn_func, A100/PG199). Prints per-call latency + effective TFLOP/s. With SAVE/LOAD it also
dumps / checks outputs on identical inputs so the two kernels can be compared numerically.
  V100:  ATTN=fa2_sm70 SAVE=/tmp/qkv.pt python benchmarks/kbench_sliding.py   # times ours + saves Q,K,V,O
  A100:  ATTN=flash LOAD=/tmp/qkv.pt python benchmarks/kbench_sliding.py       # times FA2 on the same inputs
"""
import os, time, math, torch
ATTN = os.environ.get("ATTN", "fa2_sm70")
H, Hkv, D, W = 16, 8, 256, 1024
scale = 1.0 / math.sqrt(D)

def run(q, k, v):                                    # q[B,H,S,D], k/v[B,Hkv,S,D] -> O[B,H,S,D]
    if ATTN == "fa2_sm70":
        import fa2_sm70
        return fa2_sm70.attention(q, k, v, causal=True, scale=scale, window=W)[0]
    from flash_attn import flash_attn_func            # BHSD -> BSHD -> FA2 -> BHSD
    o = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                        causal=True, window_size=(W - 1, 0), softmax_scale=scale)
    return o.transpose(1, 2).contiguous()

def t(fn, it=50, wu=15):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / it

dev = torch.cuda.get_device_name(0)
print(f"[{ATTN}] {dev} cc{torch.cuda.get_device_capability(0)}  (d={D}, GQA {H}/{Hkv}, sliding W={W})")
print(f"{'S':>7}{'lat us':>10}{'TFLOP/s':>10}")
for S in [512, 1024, 2048, 4096, 8192, 16384]:
    torch.manual_seed(S)
    q = torch.randn(1, H, S, D, device="cuda", dtype=torch.float16) * 0.3
    k = torch.randn(1, Hkv, S, D, device="cuda", dtype=torch.float16) * 0.3
    v = torch.randn(1, Hkv, S, D, device="cuda", dtype=torch.float16) * 0.3
    lat = t(lambda: run(q, k, v))
    win = min(S, W)
    flops = 2 * 2 * H * S * win * D                  # QK + PV over the (windowed) causal band
    print(f"{S:>7}{lat*1e6:>10.0f}{flops/lat/1e12:>10.1f}")

# ---- numeric agreement on identical inputs (SAVE on one host, LOAD on the other) ----
if os.environ.get("SAVE"):
    S = 2048; torch.manual_seed(12345)
    q = torch.randn(1, H, S, D, device="cuda", dtype=torch.float16) * 0.3
    k = torch.randn(1, Hkv, S, D, device="cuda", dtype=torch.float16) * 0.3
    v = torch.randn(1, Hkv, S, D, device="cuda", dtype=torch.float16) * 0.3
    o = run(q, k, v)
    torch.save({"q": q.cpu(), "k": k.cpu(), "v": v.cpu(), "o": o.float().cpu()}, os.environ["SAVE"])
    print(f"[{ATTN}] saved Q,K,V,O (S={S}) to {os.environ['SAVE']}")
if os.environ.get("LOAD"):
    d = torch.load(os.environ["LOAD"])
    q, k, v = d["q"].cuda(), d["k"].cuda(), d["v"].cuda()
    o_other = d["o"].cuda()                           # the OTHER kernel's output (fp32)
    o = run(q, k, v).float()
    # fp32 oracle
    g = H // Hkv; Kx = k.repeat_interleave(g, 1).float(); Vx = v.repeat_interleave(g, 1).float()
    S = q.shape[2]; i = torch.arange(S, device="cuda")[:, None]; j = torch.arange(S, device="cuda")[None, :]
    m = torch.where((j <= i) & (j > i - W), 0.0, float("-inf"))
    ref = torch.softmax((q.float() @ Kx.transpose(-1, -2)) * scale + m, -1) @ Vx
    rl = lambda a, b: (a - b).norm().item() / (b.norm().item() + 1e-9)
    print(f"[{ATTN}] vs OTHER kernel: relL2={rl(o, o_other):.2e}   |   vs fp32 oracle: this={rl(o, ref):.2e} other={rl(o_other, ref):.2e}")
