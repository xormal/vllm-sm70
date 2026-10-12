# Decode (Sq=1) latency & throughput vs context length, ours vs torch SDPA-EFFICIENT (the cutlass FMHA
# torch falls back to for a single query on V100 -- parallelism-starved at Sq=1). Two head configs:
# Gemma-4-12B decode (H16/Hkv8/d256) and a Llama-style MHA (H32/Hkv32/d128). fp16, batch 1.
import math, time, torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import fa2_sm70

dev = "cuda"
def t(fn, it=100, wu=25):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / it

def sdpa_eff(q, K, V, scale):
    g = q.shape[1] // K.shape[1]                              # V100 efficient backend lacks enable_gqa at
    if g > 1:                                                 # Sq=1 -> expand K/V (how GQA runs on SDPA here)
        K = K.repeat_interleave(g, 1); V = V.repeat_interleave(g, 1)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        return F.scaled_dot_product_attention(q, K, V, scale=scale)

def oracle(q, K, V, scale):                                  # fp32 exact reference (expand GQA)
    g = q.shape[1] // K.shape[1]
    Kx = K.float().repeat_interleave(g, 1); Vx = V.float().repeat_interleave(g, 1)
    return torch.softmax((q.float() @ Kx.transpose(-1, -2)) * scale, -1) @ Vx

configs = [
    ("Gemma-4-12B  H16/Hkv8 d256", 16, 8, 256),
    ("MHA          H32/Hkv32 d128", 32, 32, 128),
]
lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

for name, H, Hkv, d in configs:
    scale = 1.0 / math.sqrt(d)
    print(f"\n{name}   (decode(q,K,V), the path the model calls; vs SDPA-EFFICIENT at Sq=1)")
    print(f"{'ctx len':>8}{'ours us':>10}{'tok/s':>9}{'GB/s':>7}{'SDPA-eff us':>13}{'tok/s':>8}{'  speedup':>10}{'  relL2':>9}")
    print("-" * 84)
    for L in lengths:
        q = torch.randn(1, H, 1, d, device=dev, dtype=torch.float16) * 0.3
        K = torch.randn(1, Hkv, L, d, device=dev, dtype=torch.float16) * 0.3
        V = torch.randn(1, Hkv, L, d, device=dev, dtype=torch.float16) * 0.3
        kv_bytes = 2 * Hkv * L * d * 2                            # K+V, fp16
        o = fa2_sm70.decode(q, K, V, scale=scale)[0]
        ref = oracle(q, K, V, scale)
        e = (o.float() - ref).norm().item() / (ref.norm().item() + 1e-9)
        to = t(lambda: fa2_sm70.decode(q, K, V, scale=scale)[0])
        ts = t(lambda: sdpa_eff(q, K, V, scale))
        print(f"{L:>8}{to*1e6:>10.0f}{1/to:>9.0f}{kv_bytes/to/1e9:>7.0f}"
              f"{ts*1e6:>13.0f}{1/ts:>8.0f}{ts/to:>9.1f}x{e:>9.0e}")

print("\nours = fa2_sm70.decode (split-KV flash-decoding; bare call takes the exact online path for GQA).")
print("tok/s = single-stream decode rate (1 / step latency). Decode is HBM-bound; GB/s is KV read/step.")
print("SDPA-eff is the same cutlass FMHA our prefill wraps -- but at Sq=1 it has ~1 CTA/head of work,")
print("so it is parallelism-starved; the split-KV kernel fills the 80 SMs, winning more at long context.")
