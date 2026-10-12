# Kernel catalogue: what these are, what they are for, how to run them

> Russian version: [`KERNELS.ru.md`](KERNELS.ru.md). Machine properties: [`VOLTA_SM70.md`](VOLTA_SM70.md).
> The mathematics behind the algorithms: [`../README.md`](../README.md) §2.

## 0. What this is

A set of CUDA kernels for an **inference engine on Volta (`sm_70`)** — V100, Quadro GV100, Titan V.
Not "a FlashAttention port": a compute layer for the whole engine — attention in all three regimes
(prefill, decode, training), the linear scan, quantised-weight unpacking, paged KV-pool servicing.
The starting point is blunt: on `sm_70` neither FlashAttention-2 (needs `sm_80+`) nor FlashInfer
(needs `sm_75+`) runs at all, and the stock PyTorch/vLLM path refuses half of the modern formats
(`fp8_e4m3`, MXFP4, bf16 in the efficient backend). Here those regimes either work, or they exist
nowhere on this hardware.

Every kernel is **exact** (except the ones explicitly marked research) and checked against an fp32
exact-softmax oracle. Quantising the KV cache is the caller's decision, not an approximation inside
the kernel: the kernel computes exactly the sum an fp16 kernel would compute on the dequantised
values.

**Map: seven extensions, ~40 ops.**

| extension | source | what | built by |
|---|---|---|---|
| `fa2sm70_prefill` | `csrc/attn_fwd_cutlass.cu` (+ `volta_fwd_*.cuh`) | prefill: cutlass path, hand-written `d=256` mainloop, byte KV | `prefill_ext()` |
| `fa2sm70_backward` | `csrc/attn_bwd_cutlass.cu` (+ `volta_bwd_block.cuh`) | training: `dQ/dK/dV`, dropout replay | `backward_ext()` |
| `fa2sm70_decode` | `csrc/split_decode.cu` | decode `Sq=1`: split-KV, deferred rescale, paged pool, KV store | `decode_ext()` |
| `fa2sm70_gdn` | `csrc/gdn_scan.cu` (+ `volta_gdn_*.cuh`) | gated-delta linear scan (replaces five Triton kernels) | `gdn_ext()` |
| `fa2sm70_w8a16` | `csrc/w8a16_gemv.cu` | GPTQ-int8 weights: GEMV and an HMMA path at `M>=8` | `w8a16_ext()` |
| `fa2sm70_mxfp4` | `csrc/mxfp4_dequant.cu` | MXFP4 (OCP Microscaling): unpack and fused GEMV | `mxfp4_ext()` |
| `fa2sm70_sparse` | `csrc/attn_fwd_sparse.cu` | RESEARCH: key-tile selection + a frozen dense `d=256` copy for honest A/B | `sparse_ext()` |

The extensions are independent: needing only decode does not mean compiling prefill.

---

## 1. Building, and where it lives

The build is **JIT, on demand**: the first `prefill_ext()` call compiles and caches a `.so`; later
calls take it from the cache. The full set is 5–20 minutes the first time (`attn_fwd_cutlass.cu`
alone takes 5–8), seconds thereafter.

```bash
source /path/to/kernels/cudaenv.sh     # or set the variables below by hand
python -c "import fa2_sm70; fa2_sm70._ext.prefill_ext()"
```

What the environment must carry (`_ext._preflight()` checks and prints a readable reason on refusal):

| variable | why |
|---|---|
| `CUDA_HOME` | root of a CUDA 12.x whose `nvcc` supports `sm_70` |
| `CC` / `CXX` / `CUDAHOSTCXX` | a host compiler `nvcc` accepts (CUDA 12.8 rejects gcc > 14) |
| `TORCH_CUDA_ARCH_LIST=7.0` | set automatically |
| `FA2SM70_BUILD_DIR` | where the cache goes if the default directory is read-only |

The usual build failure is **not** a missing CUDA — it is the host gcc from a conda env: `nvcc 12.8`
refuses gcc > 14, and torch picks exactly that one by default. Hence the explicit
`CC/CXX/CUDAHOSTCXX`.

---

## 2. Three entry levels

**Level 1 — the `fa2_sm70` Python API.** Ready-made functions with checks, `head_dim` padding,
bf16→fp16 casting and split-count selection. This is what 90% of callers want.

```python
import fa2_sm70, torch
Q = torch.randn(1, 16, 4096, 128, device="cuda", dtype=torch.float16)
K = V = torch.randn(1, 4, 4096, 128, device="cuda", dtype=torch.float16)   # GQA 4:1
O, lse = fa2_sm70.attention(Q, K, V, causal=True)
```

**Level 2 — raw extension ops.** No wrapper: your own layouts, your own scales, your own invariants.
This is where the byte paths and the diagnostic variants live.

```python
ext = fa2_sm70._ext.prefill_ext()
O, lse = ext.attn_fwd_volta_i8(Q, Kq, Ks, Vq, Vs, scale, causal, nosc)
```

**Level 3 — the engine.** Two routes, and the second one supersedes the first:

* **`FA2_SM70`, a first-class vLLM backend** (current fork): its own enum member, its own file, its
  own declared capabilities, and a refusal at start-up rather than a failure mid-forward. Serves
  `head_dim` 64/128/256/512, sliding window, chunked prefill, and a fp16 / `fp8_e4m3` /
  `int8_per_token_head` KV pool whose format is marked ON THE CACHE and verified by every reader.
  Enabled with `VLLM_SM70_FA2=1` or `--attention-backend FA2_SM70`.
* **`integrations/vllm_sm70/`** — the older module substitution (put it first on `PYTHONPATH`). Still
  works and is what the measurements in the README's §6 were taken on, but substituting somebody
  else's package means keeping their signatures, so the store and the attention can end up gated by
  different conditions — which is exactly how a private byte format once leaked to a foreign reader.

Acceptance rule for either: **the graft is verified by behaviour, never by a file being present** — a
server that came up without our kernels looks exactly the same. Route counters written BY THE WORKER
ON THE WORK are the cheapest honest proof.

---

## 3. The catalogue

### 3.1. Prefill, exact (`head_dim <= 1024`)

**What it is.** Fused attention with an online softmax: `S` is never materialised, memory is `O(S)`.
The inner loop is cutlass Volta FMHA (the same kernel behind `SDPBackend.EFFICIENT_ATTENTION`), but
with **our scheduling** of causal work across 80 SMs and our layout glue.

**What it is for.** Training and prefill at any length; GQA/MQA with no K/V expansion; variable
length via `cu_seqlens`; the modes SDPA has no fused path for (window, two-sided window, soft-cap,
ALiBi).

```python
O, lse = fa2_sm70.attention(Q, K, V, causal=True)                    # Q [B,H,Sq,D], K/V [B,Hkv,Sk,D]
O, lse = fa2_sm70.attention(Q, K, V, causal=True, window=1024)       # sliding window (Gemma local)
O, lse = fa2_sm70.attention(Q, K, V, window=512, window_right=512)   # two-sided window
O, lse = fa2_sm70.attention(Q, K, V, causal=True, softcap=50.0)      # logit soft-cap
O, lse = fa2_sm70.attention(Q, K, V, causal=True, alibi_slopes=sl)   # ALiBi, sl [H] fp32
O, lse = fa2_sm70.attention_varlen(Q, K, V, cu_q, cu_k, max_q, max_k, causal=True)
```

**What comes back.** `O` in the input dtype and `lse` fp32 `[B,H,Sq]` — `log sum exp(scale*QK^T)`.
`lse` is what backward and chunked-prefill stitching need.

**Limits.** `D <= 1024` (a non-multiple of 8 is zero-padded); bf16 is cast to fp16 — `sm_70` has no
bf16 in hardware; `D > 128` goes through the output-accumulator path.

**Raw ops:** `attn_fwd_cutlass`, `attn_fwd_cutlass_varlen`, `attn_fwd_cutlass_dropout`,
`attn_fwd_qbshd` (Q and O both `[B,S,H,D]` — chunked prefill with no transposes).

### 3.2. Prefill at `head_dim = 256`: the hand-written Volta mainloop

**What it is.** Here cutlass is replaced outright: our own block mainloop on `mma.sync` HMMA.884, our
own shared-memory layout, Q held in registers. The reason is that at `d=256` cutlass's template
constraints (`pitch_linear_thread_map`, tile width) cost 29% of the speed; see `VOLTA_SM70.md` §5.

**What it is for.** Models with a large head: Gemma-4 (`d=256`), MLA-like configurations.

```python
ext = fa2_sm70._ext.prefill_ext()
O, lse = ext.attn_fwd_volta(Q, K, V, scale, causal)      # BSHD in and out
```

`attn_fwd_vsplit` is the previous `d=256` route via two `head_dim_value=128` passes, kept as an A/B
baseline.

### 3.3. Prefill on a BYTE KV cache: splitting warps by role

**What it is.** A separate kernel whose K and V arrive one byte wide. One block, 12 warps: **four
haul** (staging a whole K/V tile into a shadow buffer), **eight compute** (both GEMMs and the
softmax). They are coupled by named barriers, **one rendezvous per tile**, and the buffer is double.
A single block is not subject to the 48 KB per-block threshold, so all 96 KB of shared memory are
available.

**Why at all.** A 250K context on 2×V100 **does not physically fit** in fp16. The byte cache is a
requirement, not an optimisation; the only question is how fast Volta can be made to compute on it.
`sm_70` has **no int8 tensor instruction** (`IMMA` arrived with Turing), so bytes reach the tensor
cores by being encoded **exactly** into an fp16 mantissa: `0x6400 | u` *is* `1024 + u`. The `1024`
offset is not subtracted per element but **folded into a rank-1 correction**
(`Σa(C+b) = Σab + C·Σa`). Expanding a byte costs **three instructions per four bytes** — it is a byte
permute, not arithmetic.

**Formats (`nosc`).** The same kernel is instantiated for different ways of delivering the byte:

| `nosc` | format | scale | status |
|---|---|---|---|
| `0` | signed int8 (two's complement) | per-position table | SHIPPED |
| `1` | e5m2 | none needed (the format carries its exponent) | IMMOVABLE reference point |
| `6` | biased byte (`u = v+128`) | per-position table | the XOR left the kernel for the quantiser |
| `7` | biased byte | per-CHANNEL, folded outside the kernel | no table at all |
| **`8`** | **biased byte** | **INTERLEAVED `[ks,vs]` table by position** | **FASTEST, bitwise equal to `0`** |
| `9` | biased byte | interleaved table in fp16 | half the table traffic |
| **`11`** | **`fp8_e4m3` directly** | none needed (the format carries its exponent) | **for a FOREIGN pool: 1.10× over format 0** |
| `2`,`3`,`4`,`5`,`10` | falsifiers and diagnostics | — | not for production |

The public entry point for the shipped format 8 (Q in **BSHD** layout `[B,Sq,H,256]`):

```python
cache  = fa2_sm70.quantize_kv_prefill(K, V)              # K/V [B,Hkv,Sk,256] -> biased byte + [.,.,.,2] fp32
O, lse = fa2_sm70.prefill_bytes(Q, cache, causal=True)   # O [B,Sq,H,256] fp16
```

Raw entry points, if you keep the layouts and scales yourself:

```python
ext = fa2_sm70._ext.prefill_ext()
# format 0 -- the shipped int8: Kq/Vq int8 [B,Hkv,Sk,D], Ks/Vs fp32 [B,Hkv,Sk]
O, lse = ext.attn_fwd_volta_i8(Q, Kq, Ks, Vq, Vs, scale, causal, 0)
# format 8 -- biased byte + interleaved table: Kq/Vq are bytes (v+128), Ks == Vs == [B,Hkv,Sk,2] fp32
O, lse = ext.attn_fwd_volta_i8(Q, Kb, KVs, Vb, KVs, scale, causal, 8)
# e5m2 -- no scales, no arguments for them
O, lse = ext.attn_fwd_volta_e5m2(Q, Kbytes, Vbytes, scale, causal)
# e4m3 -- the same, with fp8_e4m3 bytes (vLLM's KV-pool format)
O, lse = ext.attn_fwd_volta_e4m3(Q, Kbytes, Vbytes, scale, causal)
```

**Why format 8 is faster.** The cost of the scale table is **not volume** (0.8% of traffic) but the
**64-bit address recomputation on every access**: a SASS transcript shows +14 `LDG` but +36 `IMAD`
and +11 `IADD3`. Interleaving K and V by position brings both values in **one `.64` load** — one
address instead of two — and the tile pointer is computed once.

**A correction about e5m2.** `fp8_e5m2` shares fp16's field order and exponent bias (15), so an e5m2
byte **is** the high byte of an fp16: the expansion is exact and costs one permute. For a long time
that made e5m2 the fastest byte route. Format 8 removes that: it is **faster than e5m2** while
carrying int8 accuracy — an order of magnitude better (e5m2 has three mantissa bits against eight for
int8-with-scale). Current numbers are in the README's *Measured* section.

**Choosing a format.** If the KV cache is already byte-wide (the serving case), take format 8. If you
are comparing routes, leave reference `1` alone: it deliberately keeps an idle multiply by `1.f` so
that it stays a fixed point of comparison.

### 3.4. Decode (`Sq == 1`): split-KV flash-decoding

**What it is.** At `Sq=1` there is almost no work per head, and an ordinary attention kernel leaves
the machine empty. The fix is **splitting along keys**: the KV is cut into `num_splits` chunks, each
computed by its own block, and the partials are stitched by the online-softmax rule. Plus **deferred
rescale**: instead of a serial `max` update, a certified bound `U >= max_j ||k_j||` makes the sum
**associative** — the serial dependency disappears and ILP goes up.

**What it is for.** All autoregressive generation. KV formats: fp16 (exact), int8 (~0.7% relL2), int4
(coarse, two-level scale), `fp8_e4m3` (in the paged path).

```python
O, lse = fa2_sm70.decode(Q, K, V)                          # Q [B,H,1,D], K/V [B,Hkv,Sk,D]
cache  = fa2_sm70.quantize_kv(K, V, bits=8)                # quantise ONCE, at write time
O, lse = fa2_sm70.decode_quantized(Q, cache)
O, lse = fa2_sm70.decode_paged(Q, Kc, Vc, block_table, block_size, kv_len)   # paged pool
kv = fa2_sm70.KVCache(B, Hkv, max_len, D); kv.append(k, v); O, _ = kv.decode(q)
```

**A subtlety about `deferred`.** The deferred path needs `max ||k||`. When the cache is ours
(`KVCache`, `decode_paged`) that quantity is maintained incrementally and costs nothing. For a
**bare** decode with no cache and `Hkv>=2`, computing it would cost a whole extra read of K, so
`decode()` transparently takes the exact online path there instead — up to 3.3× faster. Pass
`k_absmax` explicitly to force the deferred route.

**A subtlety about MQA/GQA.** `mqa="auto"` fuses `GF` query heads into one CTA so the shared KV is
read from HBM **once** instead of once per head: bit-identical, 1.4–1.6× when the KV is large.

**Split count.** The target is a **CTA count, not a chunk size** (`_auto_splits`). These are not the
same thing: a "2048-token chunk" rule yields three times too few blocks to fill 80 SMs on a long
context.

**Sliding window at decode: `kv_start`.** `flash_decode_defer_mqa_paged_into` takes a final optional
argument `kv_start` — int32 `[B]`, the **left bound of the key range per sequence** (`None` = 0 = the
previous behaviour bit-for-bit; requires a non-empty `seqlens`). The right bound is already carried by
`seqlens[b]`, so a sliding layer's window is exactly this one number: `kv_start = max(0, seq_len - W)`.
It is **exact to the token**, and that is the whole point: substituting the `block_table` row (the only
workaround that avoids touching the kernel) can only round to `block_size` and drags in up to
`block_size-1` extra keys. Measured on the Gemma-4 shape (d=256, Hkv=8, W=1024, block_size=16, worst
remainder): the truncation gives relL2 **1.2e-1**, `kv_start` gives 2.1e-4 (fp16 noise) — at the same
cost. Gathering the window into a dense buffer is exact too, but 3.3-4.4x more expensive. With no bound
at all the kernel reads the whole length: 5.3x slower at T=16K, **and wrong**, because the engine has
already freed the pages to the left of the window.

**Also in this extension:** `gather_paged_kv*` (gathering pages into a contiguous buffer, byte pools
included), `reshape_and_cache_i8` / `reshape_and_cache_e4m3` (writing the KV store in our format),
`gptq_dequant_w8`.

### 3.5. Training: the exact backward

**What it is.** `dQ/dK/dV` with no `S` materialisation, via recomputation. `head_dim <= 512`. There is
**reproducible dropout** (replay from seed/offset — the same mask pattern as the forward, with no
stored mask).

```python
O = fa2_sm70.flash_attn_func(Q, K, V, causal=True, dropout_p=0.1)   # differentiable
O.sum().backward()                                                   # dQ/dK/dV -- our kernels
O = fa2_sm70.flash_attn_varlen_func(Q, K, V, cu_q, cu_k, max_q, max_k, causal=True)
```

The output layout is settable and the wrapper does **not** `permute + contiguous`: the transpose in
the glue cost more than the kernel itself (see `ALGORITHMS_LENS.md`, "the price of a seam").

### 3.6. The gated-delta linear scan (GDN)

**What it is.** The five chunked-scan kernels of `chunk_gated_delta_rule` on our HMMA instead of
Triton (`gdn_kkt`, `gdn_solve_tril`, `gdn_wu`, `gdn_h`, `gdn_o`) — Triton emits no HMMA.884 on
`sm_70` at all, so the stock path computes the scan on the ALU.

**What it is for.** Hybrid models with a linear mixer (Qwen3-Next and relatives). The scan is 10% of
prefill there, and that is not "only 10%": a third of its own time is pure seam.

```python
from fa2_sm70 import gdn
if gdn.supported(q, k, v, g, beta, initial_state, cu_seqlens):     # checked BEFORE any work
    o, ht = gdn.chunk_gated_delta_rule_volta(q, k, v, g, beta, ...)
```

`gdn.supported(...)` is not cosmetic: the kernels are instantiated for specific shapes, and the gate
must refuse up front rather than fail halfway.

### 3.7. Quantised weights

**W8A16 (GPTQ-int8).** `w8a16_gemv` is a fused GEMV that consumes the weights **packed** — no
dequantisation to fp16 first, so no extra read and no extra slab of memory. `w8a16_gemm_hmma` is the
tensor-core path at `M >= 8`. `w8a16_prefers_packed` returns the **measured** crossover against a
dense fp16 GEMM, not a guessed one.

**MXFP4 (OCP Microscaling).** `mxfp4_dequant` is a bit-exact unpacker, `mxfp4_gemv` a fused GEMV on
packed weights. Stock vLLM on `sm_70` simply refuses: `mxfp4 is not supported ... Minimum
capability: 80`.

```python
ext = fa2_sm70._ext.w8a16_ext();  y = ext.w8a16_gemv(x, qweight, scales, zeros, g_idx)
W  = fa2_sm70.mxfp4_dequant(blocks, scales)
```

### 3.8. Research: key-tile selection

`fa2sm70_sparse` is the sparsity mechanism: `block_summary` (mean key and radius per tile), `select`
(tile selection through the tensor cores plus an exact top-k), `fwd_sparse` (a forward driven by a
work queue over a bitmap). Beside it sits `fwd_ref`, a **frozen copy** of the dense `d=256` kernel, so
that A/B compares kernel against kernel rather than against a different revision of the glue.

The honest status: **2–4× is not free**. The lever here is depth across layers, not the fraction of
keys dropped.

---

## 4. Which kernel for which job

| job | take |
|---|---|
| prefill, `d <= 128`, long context | `fa2_sm70.attention` |
| prefill, `d = 256` | `attn_fwd_volta` (raw) or `attention` — the path selects itself |
| prefill, KV already byte-wide | `attn_fwd_volta_i8(..., nosc=8)` |
| chunked prefill, `[B,S,H,D]` in and out | `attn_fwd_qbshd` — no transposes |
| ragged batch without padding | `attention_varlen` |
| decode, cache is yours | `KVCache` + `decode` |
| decode, paged pool (serving) | `decode_paged` |
| decode, memory costs more than accuracy | `quantize_kv(bits=8)` + `decode_quantized` |
| training | `flash_attn_func` (autograd) |
| Gemma-style local layers | `attention(window=..., softcap=...)` |
| hybrid with a linear mixer | `gdn.chunk_gated_delta_rule_volta` |
| GPTQ-int8 / MXFP4 weights | `w8a16_ext()` / `mxfp4_ext()` |

---

## 5. Making sure it is actually ours that ran

Three times in one session it turned out that something was "declared and not executed": the server
comes up, answers, works — on somebody else's path. So check by **behaviour**:

1. **Route printing.** The vLLM glue prints a `route:` line at every path choice.
2. **A/B with the switch.** Run the same request with our path and without it. If the end-to-end
   metric moved by 0.03%, the server never went there.
3. **Bitwise equality where it is obligatory.** `torch.equal` between format 8 and format 0, between
   e5m2 and the fp16 kernel, between the int8 path and its own dequantised reference.
4. **SASS after every build.** `tools/sass_check.py`: stack frame, `LDL`/`STL`, register count. The
   compiler's spill report is incomplete — a stack frame appears at ZERO declared spills.

Tests: `python tests/run_all.py` (20 suites) or `pytest -q tests/`.

---

## 6. Limits and mines

- **bf16 on `sm_70` exists only as a cast to fp16.** There is no bf16 in the hardware. Accuracy after
  the cast is verified; expectations of "bf16 dynamic range" are not met.
- **A private format is legal only while every reader is ours.** The KV store and the engine's
  attention backend can be gated by DIFFERENT conditions — and then our int8 gets read as `e4m3`. The
  format is marked **on the cache**, and the reader checks the mark.
- **Staging buffers get zeros, plus tile slack.** A tile reads its own end past the valid data;
  `torch.empty` sized exactly gives NaN on the first long request.
- **The tail key tile must be clamped by `Sk`.** The mask sets `-inf` and those columns' `P` is
  exactly zero — but `0 · inf = NaN` still poisons the whole output row through the second GEMM. An
  address clamp costs one integer instruction; a predicated load costs 7.7% of the speed.
- **`num_splits` is derived from a CTA count.** A chunk-size rule silently under-fills the machine.
- **V100 clocks drift under load.** Any timing taken without `nvidia-smi -lgc`, and without the clock
  and power printed next to the result, is void.
