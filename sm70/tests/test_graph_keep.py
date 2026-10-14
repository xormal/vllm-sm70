# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# Static device buffers that grow must outlive a growth step: a captured CUDA graph keeps the old address.
# Run:  python tests/test_graph_keep.py      (GPU; a few seconds, ~100 MiB)
#
# [14.10.2026] Production crashed (illegal memory access, Xid 31 MMU faults on both TP ranks, a draft token id
# 85899345940 = two int32 selection counts read as int64). The split-K workspaces of w8a16_gemv (`кэш`,
# `кэш_hmma`, `кэш_m4`) and the per-head selection buffer `_Eg` of sparse_select_rows were reallocated when the
# batch grew and the old tensor released, while the full decode graph and the MTP drafter's graphs kept writing
# to it. Each check: capture a graph at a small batch, grow the buffer with an eager call, fill "victim" tensors
# of the old size with a sentinel (the allocator hands them the released block), replay the graph. Victims must
# stay intact (before the fix: 573440 and 16384 elements overwritten), and the graph's own result unchanged.
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fa2_sm70

ZN = 1.2345e7
ok = True


def replay_after_growth(capture, grow, victim_numel, n_victims):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            capture()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = capture()
    g.replay(); torch.cuda.synchronize()
    before = out.clone()
    grow(); torch.cuda.synchronize()
    victims = [torch.full((victim_numel,), ZN, device="cuda") for _ in range(n_victims)]
    torch.cuda.synchronize()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    hit = sum(int((v != ZN).sum().item()) for v in victims)
    return hit, bool(torch.equal(out, before))


# --- w8a16_gemv split-K workspace (M=4 in the graph, M=64 eager)
w8 = fa2_sm70._ext.w8a16_ext()
torch.manual_seed(0)
K, N, G = 17408, 5120, 128
qw = torch.randint(-2**31, 2**31 - 1, (K // 4, N), device="cuda", dtype=torch.int32)
qz = torch.randint(-2**31, 2**31 - 1, (K // G, N // 4), device="cuda", dtype=torch.int32)
sc = torch.rand(K // G, N, device="cuda", dtype=torch.float16) * 0.02 + 0.001
x4 = torch.randn(4, K, device="cuda", dtype=torch.float16) * 0.05
x64 = torch.randn(64, K, device="cuda", dtype=torch.float16) * 0.05
hit, same = replay_after_growth(lambda: w8.w8a16_gemv(x4, qw, qz, sc, G, False),
                                lambda: w8.w8a16_gemv(x64, qw, qz, sc, G, False), 1 << 20, 12)
good = hit == 0 and same; ok &= good
print(f"  w8a16_gemv split-K: victim elements hit {hit}, graph result unchanged {same}: {'OK' if good else 'FAIL'}")
del qw, qz, sc

# --- sparse_select_rows per-head buffer _Eg (topf>0; 8 rows in the graph, 24 eager)
pf = fa2_sm70._ext.prefill_ext()
H, Hkv, D, B, BS, PAGES, COLS = 12, 2, 256, 64, 16384, 24, 4
g_ = H // Hkv
NB = COLS * BS // B
KW = dict(SINK=2, WIN=8, REC=2, gamma=0.0, grp=1, topf=0.15, qGrp=4, qCausal=1, minblk=256)
kbar = (torch.randn(PAGES * (BS // B), Hkv, D, device="cuda") * 0.5).half()


def inputs(reqs, seed):
    t = torch.Generator(device="cpu").manual_seed(seed)
    n = reqs * 4
    q = (torch.randn(n, H, D, generator=t) * 0.5).half().cuda()
    bt = torch.stack([torch.randperm(PAGES, generator=t)[:COLS] for _ in range(reqs)]).int().cuda()
    sl = torch.randint(20000, COLS * BS, (reqs,), generator=t).int().cuda()
    bufs = (torch.zeros(n, Hkv, dtype=torch.int32, device="cuda"), torch.zeros(n, Hkv, NB, dtype=torch.int32, device="cuda"),
            torch.zeros(n, Hkv, g_ * NB, device="cuda"), torch.zeros(n, Hkv, g_ * NB, device="cuda"))
    return q, bt, sl, bufs


def select(q, bt, sl, bufs):
    pf.sparse_select_rows(q, kbar, bt, sl, None, D ** -0.5, 0.02, B, *bufs, BS, **KW)
    return bufs[1]


a8, a24 = inputs(2, 11), inputs(6, 13)
hit, same = replay_after_growth(lambda: select(*a8), lambda: select(*a24), 8 * Hkv * NB, 64)
good = hit == 0 and same; ok &= good
print(f"  sparse_select_rows _Eg: victim elements hit {hit}, graph selection unchanged {same}: {'OK' if good else 'FAIL'}")

print("GRAPH KEEP:", "ALL OK" if ok else "FAILED"); assert ok
