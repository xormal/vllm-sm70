# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Prefill correctness: fa2_sm70.attention vs an fp32 exact-softmax reference (independent oracle)
# and vs torch SDPA. Run:  python tests/test_attention.py
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70


def ref(Q, K, V, scale, causal):
    B, H, Sq, d = Q.shape; Hkv, Sk = K.shape[1], K.shape[2]
    Kx = K.repeat_interleave(H // Hkv, 1).float(); Vx = V.repeat_interleave(H // Hkv, 1).float()
    s = (Q.float() @ Kx.transpose(-1, -2)) * scale
    if causal:
        off = Sk - Sq; qi = torch.arange(Sq, device=Q.device).view(Sq, 1); ki = torch.arange(Sk, device=Q.device).view(1, Sk)
        s = s.masked_fill(ki > qi + off, float("-inf"))
    return torch.softmax(s, -1) @ Vx, torch.logsumexp(s, -1)


SHAPES = [(1,8,8,256,256,64,False), (2,8,2,300,300,128,True),
          (1,4,4,128,128,96,True), (1,16,16,512,512,128,True),
          (1,16,8,1024,1024,256,True), (1,16,1,2048,2048,512,True),   # MLA-scale d
          (1,4,4,256,256,768,False), (1,4,4,256,256,1024,True)]       # head_dim > 512

# Thin (small B*H) causal shapes: these UNDER-FILL the 80 SMs, which is exactly when the prefill
# planner engages split-K over the key range (README §2.6). Without them the split path would be
# essentially untested -- the shapes above are dense enough that the planner declines to split.
# 777 is deliberately not a multiple of the query tile (ragged last tile + ragged split slices).
THIN = [(1,1,1,2048,2048,128,True), (1,1,1,2048,2048,64,True),
        (1,2,2,4096,4096,64,True),  (1,1,1,777,777,128,True),
        # d>128 MUST be here. The split path was verified only at d=64/128 and shipped without a
        # head-dim gate; at d=256/512 the kernel keeps the output in a gmem accumulator instead of
        # registers, the per-split partial buffer collided with it, and O came out with relL2 up to
        # 3.6e+01 while the LSE stayed correct -- so training saw it as wrong dQ/dK with dV intact,
        # three steps away from the cause.
        (1,1,1,2048,2048,256,True), (1,1,1,2048,2048,512,True)]


def build(B, H, Hkv, Sq, Sk, d, seed):
    torch.manual_seed(seed)
    return (torch.randn(B,H,Sq,d, device="cuda", dtype=torch.float16)*0.5,
            torch.randn(B,Hkv,Sk,d, device="cuda", dtype=torch.float16)*0.5,
            torch.randn(B,Hkv,Sk,d, device="cuda", dtype=torch.float16)*0.5)


# Child mode: recompute the thin shapes under whatever FA2SM70_FWD_SPLITS the parent set, and dump
# them. The switch is read once per process, so a same-process A/B is impossible by construction.
if "--dump-thin" in sys.argv:
    out = []
    for i, (B, H, Hkv, Sq, Sk, d, causal) in enumerate(THIN):
        Q, K, V = build(B, H, Hkv, Sq, Sk, d, 100 + i)
        out.append(fa2_sm70.attention(Q, K, V, causal=causal, scale=1.0/math.sqrt(d))[0].cpu())
    torch.save(out, sys.argv[sys.argv.index("--dump-thin") + 1])
    sys.exit(0)

torch.manual_seed(0)
ok = True
for (B, H, Hkv, Sq, Sk, d, causal) in SHAPES:
    scale = 1.0 / math.sqrt(d)
    Q = torch.randn(B,H,Sq,d, device="cuda", dtype=torch.float16)*0.5
    K = torch.randn(B,Hkv,Sk,d, device="cuda", dtype=torch.float16)*0.5
    V = torch.randn(B,Hkv,Sk,d, device="cuda", dtype=torch.float16)*0.5
    O, L = fa2_sm70.attention(Q, K, V, causal=causal, scale=scale)
    Oref, Lref = ref(Q, K, V, scale, causal)
    oe = (O.float()-Oref).abs().max().item(); le = (L-Lref).abs().max().item()
    good = oe < 4e-3 and le < 5e-3; ok &= good
    print(f"  H{H}/{Hkv} S{Sq}/{Sk} d{d} causal={causal}: O_err={oe:.2e} LSE_err={le:.2e} {'OK' if good else 'FAIL'}")

# --- thin shapes vs the fp32 oracle (exercises the split-K path) -----------------------------
mine = []
for i, (B, H, Hkv, Sq, Sk, d, causal) in enumerate(THIN):
    scale = 1.0 / math.sqrt(d)
    Q, K, V = build(B, H, Hkv, Sq, Sk, d, 100 + i)
    O, L = fa2_sm70.attention(Q, K, V, causal=causal, scale=scale)
    mine.append(O.cpu())
    Oref, Lref = ref(Q, K, V, scale, causal)
    oe = (O.float()-Oref).abs().max().item(); le = (L-Lref).abs().max().item()
    good = oe < 4e-3 and le < 5e-3; ok &= good
    print(f"  [thin] B{B} H{H} S{Sq} d{d}: O_err={oe:.2e} LSE_err={le:.2e} {'OK' if good else 'FAIL'}")

# --- split-K must not change the answer: same inputs, switch off, in a child process ----------
import subprocess, tempfile
with tempfile.TemporaryDirectory() as td:
    dump = os.path.join(td, "nosplit.pt")
    env = dict(os.environ, FA2SM70_FWD_SPLITS="1",
               PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--dump-thin", dump],
                       env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not os.path.exists(dump):
        print(f"  [split A/B] FAIL: no-split child did not run ({r.returncode})\n{r.stderr[-500:]}")
        ok = False
    else:
        for (shape, a, b) in zip(THIN, mine, torch.load(dump)):
            dif = (a.float() - b.float()).abs().max().item()
            good = dif < 4e-3; ok &= good           # fp32 partials -> the merge must add no error
            print(f"  [split A/B] B{shape[0]} H{shape[1]} S{shape[3]} d{shape[5]}: "
                  f"|split - nosplit|={dif:.2e} {'OK' if good else 'FAIL'}")

print("PREFILL:", "ALL OK" if ok else "FAILED"); assert ok
