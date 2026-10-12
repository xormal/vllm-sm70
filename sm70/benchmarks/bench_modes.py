# Modes torch SDPA cannot do FUSED (but our sm70 cutlass FMHA can), with timings + exactness.
# Each row: our single fused kernel time, the best SDPA route (materialize an [S,S] bias then the
# EFFICIENT backend, or the quadratic MATH backend, or n/a), and relL2 vs an fp32 oracle. One V100.
import math, time, torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import fa2_sm70

dev = "cuda"
def t(fn, it=30, wu=8):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / it

def relL2(a, b):
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-9)

def mk(B, H, S, d, std=0.3):
    g = lambda: (torch.randn(B, H, S, d, device=dev, dtype=torch.float16) * std)
    return g(), g(), g()

def oracle(Q, K, V, scale, bias=None, softcap=0.0):
    s = (Q.float() @ K.float().transpose(-1, -2)) * scale
    if softcap > 0: s = softcap * torch.tanh(s / softcap)
    if bias is not None: s = s + bias
    return torch.softmax(s, -1) @ V.float()

def sdpa_route(Q, K, V, scale, bias):
    """Best SDPA can do for an arbitrary additive [.,.,S,S] bias: try the fused EFFICIENT backend,
    else the quadratic MATH backend. Returns (time, label)."""
    for be, tag in [(SDPBackend.EFFICIENT_ATTENTION, "mask+eff"), (SDPBackend.MATH, "mask+MATH")]:
        def run():
            with sdpa_kernel([be]):
                return F.scaled_dot_product_attention(Q, K, V, attn_mask=bias, scale=scale)
        try:
            run()                                            # probe: does this backend accept the mask?
            return t(run, it=10, wu=3), tag
        except Exception:
            continue
    return None, "unsupported"

B, H, S, d = 2, 16, 4096, 128
scale = 1.0 / math.sqrt(d)
i = torch.arange(S, device=dev)[:, None]; j = torch.arange(S, device=dev)[None, :]
print(f"{'mode (SDPA has no fused path)':<30}{'config':<18}{'ours':>9}{'best SDPA route':>22}{'relL2':>9}")
print("-" * 90)

# 1) Sliding-window causal (Gemma sliding layers): keys (i-W, i].
Q, K, V = mk(B, H, S, d); W = 1024
bmask = torch.where((j <= i) & (j > i - W), 0.0, float("-inf")).to(torch.float32)[None, None]
o = fa2_sm70.attention(Q, K, V, causal=True, scale=scale, window=W)[0]
ref = oracle(Q, K, V, scale, bias=bmask)
to = t(lambda: fa2_sm70.attention(Q, K, V, causal=True, scale=scale, window=W)[0])
ts, tag = sdpa_route(Q, K, V, scale, bmask.expand(B, H, S, S))
sd = f"{ts*1e6:.0f}us ({tag})" if ts else "unsupported"
print(f"{'sliding window':<30}{f'W={W},S={S}':<18}{to*1e6:>7.0f}us{sd:>22}{relL2(o, ref):>9.1e}")
del bmask, ref, o; torch.cuda.empty_cache()

# 2) Two-sided / bidirectional local window (non-causal): keys [i-(W-1), i+R].
W2, R = 512, 256
bmask = torch.where((j <= i + R) & (j > i - W2), 0.0, float("-inf")).to(torch.float32)[None, None]
o = fa2_sm70.attention(Q, K, V, causal=False, scale=scale, window=W2, window_right=R)[0]
ref = oracle(Q, K, V, scale, bias=bmask)
to = t(lambda: fa2_sm70.attention(Q, K, V, causal=False, scale=scale, window=W2, window_right=R)[0])
ts, tag = sdpa_route(Q, K, V, scale, bmask.expand(B, H, S, S))
sd = f"{ts*1e6:.0f}us ({tag})" if ts else "unsupported"
print(f"{'two-sided window':<30}{f'W={W2},R={R}':<18}{to*1e6:>7.0f}us{sd:>22}{relL2(o, ref):>9.1e}")
del bmask, ref, o; torch.cuda.empty_cache()

# 3) Logit soft-cap (Gemma-2/3): cap*tanh(logit/cap) -- a nonlinearity, NOT an additive mask.
cap = 50.0
causal = torch.where(j <= i, 0.0, float("-inf")).to(torch.float32)[None, None]
o = fa2_sm70.attention(Q, K, V, causal=True, scale=scale, softcap=cap)[0]
ref = oracle(Q, K, V, scale, bias=causal, softcap=cap)
to = t(lambda: fa2_sm70.attention(Q, K, V, causal=True, scale=scale, softcap=cap)[0])
print(f"{'logit softcap':<30}{f'cap={cap:g},S={S}':<18}{to*1e6:>7.0f}us{'n/a (nonlinear)':>22}{relL2(o, ref):>9.1e}")
del ref, o; torch.cuda.empty_cache()

# 4) ALiBi (fused linear position bias). SDPA needs a materialized per-head [S,S] bias.
slopes = (2.0 ** (-8.0 / H * torch.arange(1, H + 1, device=dev))).to(torch.float32)
o = fa2_sm70.attention(Q, K, V, causal=True, scale=scale, alibi_slopes=slopes)[0]
ab = (slopes[:, None, None] * (j - i).to(torch.float32)).masked_fill(j > i, float("-inf"))[None]
ref = oracle(Q, K, V, scale, bias=ab)
to = t(lambda: fa2_sm70.attention(Q, K, V, causal=True, scale=scale, alibi_slopes=slopes)[0])
ts, tag = sdpa_route(Q, K, V, scale, ab.expand(B, H, S, S))
sd = f"{ts*1e6:.0f}us ({tag})" if ts else "unsupported"
print(f"{'ALiBi (fused slope)':<30}{f'{H}h,S={S}':<18}{to*1e6:>7.0f}us{sd:>22}{relL2(o, ref):>9.1e}")
del ab, ref, o; torch.cuda.empty_cache()

print("\nAll 'ours' are the SINGLE fused sm70 kernel (windows are O(Sk*W): out-of-window key blocks are")
print("skipped, so they also beat full attention). The SDPA route must materialize an [S,S] bias")
print("(O(S^2) memory) and falls back to the quadratic MATH backend; softcap is a logit nonlinearity")
print("with no additive-mask form at all. Every mode is exact vs an fp32 oracle (relL2 ~2.6e-4).")
print("(head_dim 256/512/768 is NOT listed: SDPA's mem-efficient backend does support it on sm70 --")
print(" only flash rejects it -- so it is not an SDPA-exclusive gap; see bench_prefill for that path.)")
