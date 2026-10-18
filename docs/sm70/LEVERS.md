# Levers of the reference deployment

The runtime is configured with environment variables ("levers"). They are read by the vLLM code
under `vllm/` and the integration layer under `sm70/integrations/vllm_sm70/`. The production set,
105 levers with exact values, is [`configs/qwen38-27b-int8-clef.env`](../../configs/qwen38-27b-int8-clef.env).
This page explains the ones that change the computation or the memory layout. All values below are the
production values. Every default was chosen by measurement on 2× V100 32 GB, and the code next to each
lever carries the measurement that justified it (search the name under `vllm/` and `sm70/`).

Levers marked **stand** are fixes verified on a test pair of GPUs on 2026-10-18 (VALIDATION.md, "Stand
2026-10-18") and not yet enabled in production. Their production value is `0`; each is a no-op at `0`.

**Rule:** change one lever at a time, then re-run the gates in [VALIDATION.md](VALIDATION.md). Several
levers change the structure of compiled graphs. `scripts/serve_qwen38_clef.sh` fingerprints those in
`VLLM_CACHE_ROOT`, so a graph compiled under one set is never replayed under another.

## Attention (backend `FA2_SM70`)

| Lever | Value | Effect |
|---|---|---|
| `VLLM_ATTENTION_BACKEND` | `FA2_SM70` | Volta attention backend (`vllm/v1/attention/backends/fa2_sm70.py`, kernels in `sm70/fa2_sm70`) |
| `FA2SM70_KVDT` | `int8_per_token_head` | KV cache dtype: int8 with a per-token, per-head scale. More accurate than `fp8_e4m3`, with a measured ~5.6% prefill cost |
| `FA2SM70_KV_I8`, `FA2SM70_KV_I8_BYTES` | `1`, `1` | int8 KV store/load on the native path |
| `FA2SM70_KV_BYTES` | `10630627328` | Fixed KV budget (skips vLLM's memory profiling). Sized so that 262,144 tokens fit on two 32 GB cards with room for activations |
| `FA2SM70_SPLIT_POOLS` | `1` | Separate page pools for attention and Gated DeltaNet state; the attention block grows to the state block (needed for 262K) |
| `FA2SM70_GDN_I16`, `FA2SM70_MAMBA_BLK_MULT` | `1`, `8` | int16 GDN state with scale (page 1596 → 828 KiB) and a coarser state block; together they give the 262K capacity |
| `FA2SM70_UNIFORM`, `FA2SM70_UNIFORM_MAXLEN` | `1`, `65000` | Uniform virtual-batch decode path for speculative positions up to 65K context, the general path beyond |
| `FA2SM70_MAX_SPLITS` | `64` | Upper bound on split-K chunks of paged decode |
| `FA2SM70_SPARSE_PCT`, `FA2SM70_SPARSE_ALPHA` | `100`, `0.02` | Sparse prefill attention: block selection with a measured error bound. Takes 250K prefill from ~390 s (dense) to ~190–215 s; validated with multi-needle tests against the dense path |
| `FA2SM70_SPARSE_MASS`, `FA2SM70_SPARSE_GAMMA`, `FA2SM70_SPARSE_BETA`, `FA2SM70_SPARSE_B` | `0.85`, `0.5`, `0`, `64` | Selection rule parameters (mass threshold, spread correction, block size) |
| `FA2SM70_SPARSE_MINSQ` | `2048` | The last, partial query chunk stays dense (needle retention) |
| `FA2SM70_SPARSE_DEC`, `FA2SM70_SPARSE_DEC_TOPF`, `FA2SM70_SPARSE_DEC_MINBLK` | `1`, `0.15`, `256` | Sparse decode: per-row/per-head top block fraction on long contexts; bit-identical to dense when all blocks are kept |
| `FA2SM70_BLKMEAN_ALL` | `0` (**stand**: `1`) | Sparse decode: after a decode step `blk_mean_update` refreshes the mean keys of every sub-block in its 64-slot slice, not only the first one. With MTP a slice holds 4 slots of up to 16 requests; at `0` every request but the first in the batch keeps stale means for the sub-blocks decode wrote, and the selection drops them once they leave the 512-token recent window |

## Linear layers

| Lever | Value | Effect |
|---|---|---|
| `FA2SM70_W8` | `1` | GPTQ-int8 GEMV on our tensor-core kernel for small M (decode); larger M goes to the turbomind body |
| `FA2SM70_TM8`, `FA2SM70_TM8_ONLY`, `FA2SM70_TM8_NMIN` | `1`, `1`, `3072` | turbomind s884 int8 GEMM (`sm70/third_party/sm70_turbomind`) for prefill-size M; with `TM8_ONLY` the original GPTQ layout is freed after repacking |
| `FA2SM70_TM8_RECON_MMIN`, `FA2SM70_TM8_RECON_NCH`, `FA2SM70_TM8_RECON_OUT` | `1024`, `4`, `1` | For M ≥ 1024, reconstruct fp16 weight chunks and use cuBLAS (×1.21 on the linear part), writing straight into the output slice |
| `FA2SM70_W12` | `1` | `self_attn.{q,k,v}_proj` (kept BF16 in the checkpoint) repacked to 12-bit at load and served by the W12A16 kernel; packed weights cached in `FA2SM70_W12_CACHE` |
| `FA2SM70_LMH` | `1` | int8 `lm_head` on our kernel (top-512 candidates, exact rescoring); frees ~620 MB per rank |
| `FA2SM70_FMLP` | `0` | Fused MLP tail (off in production) |
| `FA2SM70_EXCH12` | topology | TP all-reduce in fp16 on NVLink, int12-compressed on a PCIe bridge (compression loses 0.79–0.81× on NVLink). Chosen by the launcher |
| `FA2SM70_AR2` | `0` | Alternative all-reduce (off) |

## Speculative decoding (native MTP, k = 3)

| Lever | Value | Effect |
|---|---|---|
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3}` | The model's own MTP head as the drafter |
| `FA2SM70_MTP_W8`, `FA2SM70_MTP_W8_NSPLIT` | `1`, `4` | MTP head in int8 on our kernel; wide body split in 4 column chunks (no 170 MB scratch buffer) |
| `FA2SM70_MTP_I4`, `FA2SM70_MTP_I4_R`, `FA2SM70_MTP_I4_N` | `1`, `2048`, `256` | Two-stage drafter argmax: int4 slice of the first 2048 dims → top-256 → exact rescoring |
| `FA2SM70_MTP_GRAPH` | `1` | Own CUDA graph for the drafter body |
| `FA2SM70_MTP_NGRAM`, `FA2SM70_MTP_NGRAM_FIX` | `1`, `1` | n-gram proposals from the context when they out-accept the MTP head. Wins on copying, costs nothing on prose; the target still verifies every token |
| `FA2SM70_DRAFT_TM8` | `1` | Drafter linear layers on the turbomind int8 body |
| `FA2SM70_GRAN_SRC`, `FA2SM70_GRAN_DST`, `FA2SM70_GRAN_CONV_ALL` | `1`, `1`, `0` | Block-granularity fixes for state copies under speculation |
| `FA2SM70_UNIFORM_PREFILL_FIX` | `1` | A step counts as uniform speculative decode only if every row carries k draft tokens. Without it a prefill chunk of exactly k+1 = 4 tokens (one request, prompt length 2048·m + 4) replays the full speculative graph on stale GDN spec state, and the answer turns to garbage. See VALIDATION.md, 2026-10-18 |
| `FA2SM70_PAD_META_FIX` | `0` (**stand**: `1`) | Padded rows of the drafter batch get the placeholder slot and length again when the number of real rows shrinks. At `0` they keep the previous call's real slots, and the draft layer writes K/V into them |
| `FA2SM70_PAD_META_PROBE` | unset | Diagnostics: every 2000 drafter calls, log how many padded rows carried a real slot |
| `FA2SM70_ASYNC_SCHED` | `0` | Async scheduling off (required for MTP correctness at block boundaries) |
| `FA2SM70_MTP` | `0` | Legacy switch of the old launcher. The MTP config is passed explicitly on the command line |

## Memory and runtime

| Lever | Value | Effect |
|---|---|---|
| `FA2SM70_MAXBATCH` | `2048` | Chunked-prefill size (`--max-num-batched-tokens`); 4096 runs out of memory on 250K prompts after image traffic |
| `FA2SM70_SEQS` | `16` | `--max-num-seqs` |
| `FA2SM70_MEM_TRIM_MB` | `400` | Return cached-but-free allocator blocks to the driver before a prefill step when they exceed this. With Clef enabled, the threshold is lowered by Clef's own footprint |
| `FA2SM70_KEEP_BUFS` | `1` | Keep every attention buffer that is replaced when it grows (bit mask: 1 all, 2 per-layer fields, 4 module caches). Captured CUDA graphs keep writing to the old address; releasing it let replays corrupt whatever reused the memory. See VALIDATION.md, 2026-10-14 |
| `FA2SM70_DRAFT_GUARD` | `1` | A draft token id outside [0, vocabulary) is dropped (that request runs the step without speculation) and logged with its hex value, row and request, instead of crashing the engine (`OverflowError` in `token_ids_cpu`) or reading out of bounds in the rejection sampler. Costs B×k integer comparisons per step |
| `FA2SM70_SELBUF_PROBE` | `20000` | Probe for that defect: a retired selection buffer is filled with a sentinel and checked every N calls; `[SELBUF] ... ИЗМЕНЁН` in the log means a graph wrote to it. `0` disables the probe |
| `FA2SM70_FREE_UNHASHED_HEAD` | `0` (**stand**: `1`) | Freed blocks without a hash go to the head of the free queue (taken first), hashed ones to the tail. At `0` everything goes to the tail, and admitting Clef records evicts the early-released hashed GDN blocks of the prefix cache before empty ones |
| `FA2SM70_NO_SPEC_BLOCKS_MAXTOK1` | `0` (**stand**: `1`) | Do not reserve the speculative GDN blocks for requests with `max_tokens=1` (Clef records): they never decode |
| `FA2SM70_W8_OOM_RETRY` | `1` | On an allocator OOM in a foreign GEMM body: empty the cache and retry once |
| `FA2SM70_CG` | `1` | Full CUDA graph for decode (+20% decode on short contexts) |
| `FA2SM70_MODE` | `dialog` | Group sizing tuned for interactive traffic |
| `FA2SM70_THINK` | `false` | Default `enable_thinking` of the chat template (per-request override allowed) |
| `FA2SM70_MM_DRAFT_SHIFT` | `1` | Schedule an image's encoder when the MTP drafter's window (one position ahead of the target's) reaches it; without it a prefill chunk ending exactly at an image's first token crashes the engine (`Encoder cache miss`). Also lets the drafter skip a missing encoder output instead of asserting. See VALIDATION.md, 2026-10-13 |

## Clef

| Lever | Value | Effect |
|---|---|---|
| `FA2SM70_CLEF` | `1` | Enables `/v1/systemone`. `0` serves the base model with zero changes to its code path |
| `FA2SM70_CLEF_DIR` | `<model>/clef` | Adapter directory (LoRA, de-abliteration, joint head) |
| `FA2SM70_CLEF_R` | `2` | Concurrent Clef records (head slots) |
| `FA2SM70_CLEF_NMAX` | `32768` | Maximum record length in tokens; a request's `max_length` can only lower it. Head memory is 2 KiB per token per slot: 32,768 costs +64 MiB on rank 0 over 16,384 (Clef's own default and the code default). The head is verified against a PyTorch fp32 reference at 32,768 |
| `FA2SM70_CLEF_A1` | `0` (**stand**: `1`) | A record preempted mid-prefill and restarted from position 0 releases its head slot and is admitted again. At `0` its span means are accumulated twice |
| `FA2SM70_CLEF_MEDIA` | `1` | Accept `"images"` in Clef records. `0` (code default) rejects them with 400 |
| `FA2SM70_CLEF_IMGMAX` | `16` | Images per Clef record |
| `FA2SM70_CLEF_NSLOT` | `3` | Ring slots for streaming LoRA layers 40–63 from pinned host memory |
| `FA2SM70_CLEF_OUT` | `/dev/shm/...` | Where workers leave results for the API process |
| `FA2SM70_CLEF_NODELTA` | unset | Diagnostics: head only, no weight correction |
| `FA2SM70_CLEF_NOBA`, `FA2SM70_CLEF_NOWRAP`, `FA2SM70_CLEF_DIAG` | unset | Diagnostics used to bisect side effects (see VALIDATION.md) |
