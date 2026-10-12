# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Shared helpers: timing, fp32 reference, and SDPA backend wrappers (default + EFFICIENT_ATTENTION).
import os, sys, math, time, contextlib, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    HAVE_BACKEND = True
except Exception:                       # older torch: fall back to the legacy context manager
    HAVE_BACKEND = False


@contextlib.contextmanager
def efficient_backend():
    """Force SDPA onto the memory-efficient (cutlass FMHA) backend -- the direct sm_70 analog."""
    if HAVE_BACKEND:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            yield
    else:
        prev = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled())
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        try:
            yield
        finally:
            torch.backends.cuda.enable_flash_sdp(prev[0]); torch.backends.cuda.enable_math_sdp(prev[1]); torch.backends.cuda.enable_mem_efficient_sdp(prev[2])


V100_PEAK_TFLOPS = 93.6   # MEASURED fp16 tensor ceiling on V100-32GB, not the 125.3 data-sheet peak


def timed(fn, it=100, wu=20):
    """Standard timing protocol: `wu` warm-up iters (build/cache/clock ramp), then a single
    synchronize, `it` timed iters, one final synchronize. Inputs must be allocated ONCE outside `fn`
    (no per-iter allocation). Returns mean seconds/iter. Use it=30 for slow (backward) kernels."""
    for _ in range(wu):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it


def attn_flops(B, H, Sq, Sk, d, causal, passes="fwd"):
    """FLOP accounting for attention. Forward = 2 matmuls (QK^T, PV) = 4*B*H*Sq*Sk*d; backward ~= 2.5x
    the forward. Causal halves the (i,j) work. `passes`: 'fwd' | 'bwd' | 'fwdbwd'."""
    f = 4.0 * B * H * Sq * Sk * d * (0.5 if causal else 1.0)
    return {"fwd": f, "bwd": 2.5 * f, "fwdbwd": 3.5 * f}[passes]


def roofline_pct(flops, seconds, peak_tflops=V100_PEAK_TFLOPS):
    """Achieved % of the tensor-core roofline for a measured kernel."""
    return 100.0 * (flops / seconds) / (peak_tflops * 1e12)


def gpu_state():
    """One-line GPU run conditions (SM clock / power / temp) so a benchmark number is reproducible.
    Clocks throttle under power/thermal limits -- report them alongside latency."""
    import subprocess
    try:
        q = "clocks.sm,power.draw,temperature.gpu,name"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return out
    except Exception:
        return "nvidia-smi unavailable"


def exact_ref(Q, K, V, scale, causal):
    B, H, Sq, d = Q.shape; Hkv, Sk = K.shape[1], K.shape[2]
    Kx = K.repeat_interleave(H // Hkv, 1).float(); Vx = V.repeat_interleave(H // Hkv, 1).float()
    s = (Q.float() @ Kx.transpose(-1, -2)) * scale
    if causal:
        off = Sk - Sq; qi = torch.arange(Sq, device=Q.device).view(Sq, 1); ki = torch.arange(Sk, device=Q.device).view(1, Sk)
        s = s.masked_fill(ki > qi + off, float("-inf"))
    return torch.softmax(s, -1) @ Vx


def relL2(O, Oref):
    return ((O.float() - Oref).norm() / Oref.norm()).item()


def expand_kv(K, V, H):
    Hkv = K.shape[1]
    if Hkv == H:
        return K, V
    return K.repeat_interleave(H // Hkv, 1), V.repeat_interleave(H // Hkv, 1)
