# Benchmark protocol

Standardized so numbers are comparable across runs and machines. Helpers live in `_common.py`.

## Method (enforced by `_common.timed`)

- **Warm-up**: 20 untimed iterations (JIT build, cache fill, clock ramp) before timing.
- **Timing**: one `torch.cuda.synchronize()`, then `it` timed iterations, then a final synchronize;
  report **mean seconds/iter**. `it=100` for fast kernels, `it=30` for the slow (backward) path.
- **Allocation policy**: all tensors are allocated **once, outside** the timed closure — never inside
  `fn` (per-iter allocation would measure the allocator, not the kernel).
- **Kernel-only vs end-to-end**: report them **separately**. Kernel-only feeds pre-shaped inputs
  straight to the extension op; end-to-end includes the Python wrapper (layout prep, LSE/delta, GQA
  expand/reduce). The backward gap between the two is the wrapper overhead (~1-2 ms, measured).
- **FLOP accounting**: `attn_flops(B,H,Sq,Sk,d,causal,passes)` — forward = `4·B·H·Sq·Sk·d` (2 matmuls),
  backward ≈ 2.5×, causal halves it. `roofline_pct(flops, seconds)` gives % of the **measured 93.6
  TFLOP/s** ceiling — not the 125.3 on the data sheet. 125.3 assumes the boost clock and perfect HMMA
  issue; a dense fp16 GEMM on this card reaches 93.6 (`docs/VOLTA_SM70.md`). Quoting the spec number
  deflates every result by 1.34× and flatters the hardware instead of the kernel.
- **Run conditions**: print `gpu_state()` (SM clock / power / temp / name) with every result — V100
  clocks throttle under power/thermal limits, so a latency number is only meaningful with them.

## Baselines

Benchmarks compare against `torch` SDPA on **both** its default dispatch and the memory-efficient
(cutlass FMHA) backend forced via `torch.nn.attention.sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)` —
the direct `sm_70` analog of our kernels — via `_common.efficient_backend()`. Accuracy is always
reported next to latency as `relL2` vs an fp32 exact-softmax reference (`_common.exact_ref`).

## Suites

| script | kind | what |
|---|---|---|
| `bench_prefill.py` | kernel-only | prefill latency, % roofline, speedup, accuracy vs SDPA (default + efficient); `head_dim` 64/128 |
| `bench_prefill_d256.py` | kernel-only | prefill at `head_dim=256` — the hand-written Volta mainloop vs this repo's own cutlass path vs SDPA-efficient, plus chunked (`Sq<Sk`) shapes |
| `bench_bwd_large.py` | kernel-only | isolated backward at large `head_dim` (256/512) vs SDPA-efficient |
| `bench_decode.py`  | kernel-only | decode (`Sq=1`) fp16/int8/int4 + deferred/MQA/VEQK vs SDPA-efficient |
| `bench_decode_lengths.py` | kernel-only | decode tok/s vs context length (512…32k), ours vs SDPA-efficient, two head configs |
| `bench_train.py`   | kernel-only | forward vs forward+backward latency; honest vs SDPA-efficient fwd+bwd |
| `bench_modes.py`   | kernel-only | window / two-sided / ALiBi / softcap — modes SDPA has no fused path for (vs its mask+MATH route) |
| `kbench_sliding.py`| kernel-only | head-to-head vs **real FlashAttention-2** on the Gemma-4 sliding shape (set `ATTN=flash` on an sm_80+ box); `SAVE`/`LOAD` for a numeric cross-check |
| `bench_fwd_roles.py` | kernel-only | the byte-KV `d=256` role kernel vs our fp16 kernel, our previous two-block int8 path, cutlass and SDPA-eff; prints results and writes JSON only when `RESULTS_PATH` points outside the repository |
| `bench_slots_ab.py` | kernel-only | **paired-ratio stopwatch** over the byte-KV formats (`nosc` 0/1/6/7/8) and the issue-slot variants; medians of PAIRED ratios with the formats interleaved WITHIN a round, plus the foreign-process count and clock/power before and after |
| `bench_i8_beats_e5m2.py` | kernel-only | the same comparison on a second harness and different shapes — kept deliberately as an independent second opinion on the int8-vs-e5m2 margin |
| `bench_kspl.py` | kernel-only | key-split (`KSPL`) vs the shipped role kernel on the Qwen3.8 prefill shape. **Correctness first, stopwatch second**: it checks both variants against an fp32 reference computed by torch over the same dequantized int8 (a *different* path), and it treats a disagreeing **baseline** as a broken instrument rather than a broken kernel — that positive control is what caught two harness defects (format 8 wants an *interleaved* scale table; near-uniform softmax makes relL2 degenerate). Verdict: correct and slightly more accurate, but x0.905–0.945 |

**Why `bench_slots_ab.py` reports medians of paired ratios rather than a ratio of medians.** V100
clocks drift over a run; interleaving the variants inside one round hands the drift to both of them
equally, and it cancels in the ratio. A ratio of medians does not have that property, and at the 1–6%
margins these formats differ by, the difference between the two is the whole result. The same script
prints the number of foreign processes on the card before and after — a number taken with a neighbour
on the card is void, and on a shared box that happens more often than not.

End-to-end (whole-model tok/s, in `examples/`, not here): `run_gemma_fa2_sm70.py`, `e2e_throughput.py`,
`compare_run.py` — every prefill/decoded token through the kernel; see `REPRODUCE.md` /
`RESULTS_gemma4_A100_vs_V100.md`.

Run: `CUDA_VISIBLE_DEVICES=0 python benchmarks/<script>.py` (from the package root).
