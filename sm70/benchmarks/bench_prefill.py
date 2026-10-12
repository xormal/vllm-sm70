# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Prefill benchmark: fa2_sm70.attention vs torch SDPA (default dispatch) vs SDPA forced onto
# SDPBackend.EFFICIENT_ATTENTION (the cutlass-FMHA memory-efficient backend our prefill builds on).
# Reports latency, % of the ACHIEVABLE tensor roofline, speedup, and relL2 accuracy.
#
# THE ROOFLINE IS THE MEASURED 93.6 TFLOP/s, NOT THE 125.3 ON THE SPEC SHEET. 125.3 assumes the
# boost clock and perfect HMMA issue; a dense fp16 GEMM on this card tops out at 93.6 (see
# docs/VOLTA_SM70.md, [spec] vs [measured]). Quoting %roofline against the spec number silently
# deflates every row by 1.34x and makes a kernel at 60% of what the card can do look like 45%.
# bench_prefill_d256.py uses the same 93.6, so the two tables in the README are comparable.
#
# The table has two halves on purpose:
#   * FULL MACHINE (B*H large): every SM already has work, so the win is the BHSD-direct output
#     (no vendor transpose copy) plus the causal block-order reversal -- a few percent.
#   * UNDER-FILLED (B*H small): the causal split-K path engages and the win is multi-fold. These
#     shapes are the small-batch serving regime; without them the split-K claim is untestable.
#
# The `no-split` column is produced by RE-RUNNING THIS SCRIPT in a child process with
# FA2SM70_FWD_SPLITS=1, so the A/B is real (the switch is read once per process) and one command
# reproduces both sides.
# Run:  python benchmarks/bench_prefill.py
import json, math, os, subprocess, sys
import torch
import torch.nn.functional as F
from _common import timed, exact_ref, relL2, expand_kv, efficient_backend
import fa2_sm70

# (B, H, Sk, d, causal, label)
SHAPES = [
    (1, 32, 1024, 128, True, "full"),
    (1, 32, 2048, 128, True, "full"),
    (1, 32, 4096, 128, True, "full"),
    (1, 16, 2048,  64, True, "full"),
    (1, 32, 8192, 128, True, "full"),
    (1,  1, 2048, 128, True, "thin"),
    (1,  1, 4096, 128, True, "thin"),
    (1,  2, 2048, 128, True, "thin"),
    (1,  1, 2048,  64, True, "thin"),
    (1,  2, 4096,  64, True, "thin"),
]

CHILD = "--emit-json" in sys.argv


def measure(B, H, Sk, d, causal):
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(d)
    Q = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.float16) * 0.5
    K = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.float16) * 0.5
    V = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.float16) * 0.5
    to = timed(lambda: fa2_sm70.attention(Q, K, V, causal=causal, scale=scale)[0])
    if CHILD:                                    # baseline child: only our own latency is needed
        return to, None, None, None, None
    Ks, Vs = expand_kv(K, V, H)
    Oref = exact_ref(Q, K, V, scale, causal)
    ts = timed(lambda: F.scaled_dot_product_attention(Q, Ks, Vs, is_causal=causal, scale=scale))
    with efficient_backend():
        te = timed(lambda: F.scaled_dot_product_attention(Q, Ks, Vs, is_causal=causal, scale=scale))
        Oe = F.scaled_dot_product_attention(Q, Ks, Vs, is_causal=causal, scale=scale)
    Oo = fa2_sm70.attention(Q, K, V, causal=causal, scale=scale)[0]
    return to, ts, te, relL2(Oo, Oref), relL2(Oe, Oref)


def key(B, H, Sk, d):
    return f"{B},{H},{Sk},{d}"


if CHILD:                                        # child: emit our latency per shape as JSON, nothing else
    out = {key(B, H, S, d): measure(B, H, S, d, c)[0] for (B, H, S, d, c, _) in SHAPES}
    print("JSON " + json.dumps(out))
    sys.exit(0)

# Parent: get the no-split baseline from a child process with the switch off.
nosplit = {}
if not os.environ.get("FA2SM70_FWD_SPLITS"):
    env = dict(os.environ, FA2SM70_FWD_SPLITS="1")
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--emit-json"],
                           env=env, capture_output=True, text=True, timeout=3600)
        for line in r.stdout.splitlines():
            if line.startswith("JSON "):
                nosplit = json.loads(line[5:])
    except Exception as e:                       # baseline is a nicety, not a requirement
        print(f"(no-split baseline unavailable: {e})")

hdr = (f"{'B,H,Sk,d (causal)':<22}{'ours':>9}{'no-split':>10}{'SDPA':>9}{'SDPA-eff':>10}"
       f"{'ours%rf':>9}{'ours/eff':>9}   relL2(ours/eff)")
print(hdr)
print("-" * len(hdr))
prev = None
for (B, H, Sk, d, causal, tag) in SHAPES:
    if tag != prev:
        print(f"--- {'FULL MACHINE (B*H large: split-K stays off)' if tag == 'full' else 'UNDER-FILLED (B*H small: causal split-K engages)'}")
        prev = tag
    to, ts, te, ro, re_ = measure(B, H, Sk, d, causal)
    # TWO matmuls: QK^T and PV, each 2*B*H*Sq*Sk*d flops -> 4*, then causal halves the (i,j) work.
    # This read 2.0* and under-reported every %roofline in the README by exactly 2x -- which matters,
    # because "how much headroom is left" is read straight off that column. Same formula as
    # _common.attn_flops(); a shape measured by both must not disagree.
    fl = 4.0 * B * H * Sk * Sk * d * (0.5 if causal else 1.0)
    rf = 100 * fl / to / 1e12 / 93.6
    tn = nosplit.get(key(B, H, Sk, d))
    ns = f"{tn*1e3:>9.3f}m" if tn else f"{'-':>10}"
    print(f"{f'{B},{H},{Sk},{d} c{int(causal)}':<22}{to*1e3:>8.3f}m{ns}{ts*1e3:>8.3f}m{te*1e3:>9.3f}m"
          f"{rf:>8.1f}%{te/to:>8.2f}x   {ro:.1e}/{re_:.1e}")

print("""
SDPA-eff = EFFICIENT_ATTENTION = the cutlass fmha_cutlassF_sm70 kernel this prefill builds on.
Three levers separate us from it, all exact (relL2 is unchanged in every row):
  1. BHSD-direct output (kOutputBHSD) -- skips the vendor's BSHD->BHSD permute+contiguous copy of O.
  2. Causal block-order reversal -- the dispatcher hands out query tiles by ascending blockIdx, so
     under a causal mask the ragged tail wave gets the HEAVIEST tiles; we reverse the mapping.
  3. Causal split-K (the `no-split` column is it switched off) -- a schedule cannot be shorter than
     its longest task, so when B*H under-fills the machine the heaviest query tile sets the floor.
     Splitting the key range fixes that; partials are fp32 and merged by an LSE-weighted combine.
Off-switches for a clean A/B: FA2SM70_FWD_SPLITS=1 (no split-K), FA2SM70_FWD_REV=0 (no reversal).""")
