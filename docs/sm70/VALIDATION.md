# Validation

How the reference deployment was validated, and how to repeat the checks. The probe client is
`sm70/integrations/vllm_sm70/clef/proverka_stenda.py`, standard library only. Its messages are in
Russian; the meaning of each mode is given below.

```bash
P=sm70/integrations/vllm_sm70/clef/proverka_stenda.py
python $P baza    8000 base.json        # 3 chats, greedy, 96 tokens with logprobs (one ~7K tokens)
python $P sravn   a.json b.json         # bitwise comparison of two `baza` runs
python $P clef    8000 clef.json 2      # 10 Clef records, 152..16,384 tokens, each sent twice
python $P clefsr  ref.json clef.json    # max |Δp| and decision agreement between two `clef` runs
python $P smes    8000 mixed.json 3     # 10 Clef records + 6 chats concurrently, 3 rounds
python $P kartinki 8000 img.json clef   # 8 requests x 2 images (1280x960 + 640x480) at once (+2 Clef records)
python $P igla    8000 245000 n.json    # code word at the START of a ~245K-token prompt
python $P dlinnye 8000 long.json 32768  # Clef length test: facts with known answers at 5/50/95 %
                                        # of 4K/12K/24K/31K-token records (needs FA2SM70_CLEF_NMAX=32768)
python $P clefkart 8000 img.json        # Clef with images: 15 records with known answers (color, digit,
                                        # shape, word, the second of two images, image + ~11K tokens of
                                        # text), a repeat for determinism, and refusals (video, bad URL,
                                        # "images" not a list); needs FA2SM70_CLEF_MEDIA=1
```

Offline gate for the image path of `/v1/systemone` (CPU, no server): the record built by the endpoint
must equal Clef's own `encode_record`, token for token and span for span, and vLLM's multimodal processor
must expand the collapsed record the endpoint sends into exactly that record. It runs the endpoint's real
chain (`_url` → vLLM media connector → token count → record → vLLM expansion), because the parts can pass
separately while the chain fails (see 2026-10-13 below). Pass a COPY of the directory with Clef's
`joint_schema_model.py`:

```bash
python -I sm70/integrations/vllm_sm70/clef/sverka_media_offline.py \
    sm70/integrations/vllm_sm70 /path/to/copy-of-clef /models/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef
# expected last line: ИТОГ: СОШЛОСЬ ("matched")
```

`clefsr` against a reference run requires a server on the original Clef weights: serve
`Cloudflare/clef` in fp16 with `FA2SM70_CLEF=1 FA2SM70_CLEF_NODELTA=1 FA2SM70_CLEF_DEABL=""` and
`FA2SM70_CLEF_DIR` pointing at a directory that contains `joint_head_fp16.safetensors`.

## Results of the reference deployment (2026-10-11 … 10-12)

2× V100 32 GB. Stand: PCIe pair, one card capped at 997 MHz. Production: NVLink pair. Production
configuration (MTP k=3, fp16 all-reduce).

| Gate | Result |
|---|---|
| Clef vs Cloudflare/clef fp16 (35 decisions, 10 records) | 35/35, max \|Δp\| 0.0093 (stand); 34/35 in production (one 0.200/0.199 near-tie in a meaningless record, which the reference scores 0.1916/0.1913) |
| Control without the weight correction | 24/35, max \|Δp\| 0.64 |
| Clef repeats / across server restarts | max \|Δp\| ≤ 0.0001 / 0.0000 |
| Native Clef head vs PyTorch fp32 reference, N = 16,000 and 32,768 tokens | max \|Δp\| 1.5e-4, argmax 8/8, repeat bitwise |
| Mixed load (10 Clef + 6 chats) × 3 rounds | 16/16 each round; Clef vs single runs max \|Δp\| 0.0185, 34/35 (same near-tie) |
| Heavy order: images ×2 → 243K needle → images, Clef inside every storm | images 8/8 ×3, Clef 2/2 ×3, needle found, ≥ 912 MiB free at peak (stand), 782 MiB (production) |
| Clef memory | +290 MiB on rank 0, +121 MiB on rank 1 |
| **Fresh clone of this repository, built from source in a new virtualenv, serving the Hugging Face release** (2026-10-12, PCIe pair) | base outputs **bitwise identical** to the stand runs in production configuration; Clef max \|Δp\| 0.0000 vs those runs and 0.0093 (35/35) vs Cloudflare/clef fp16; images 8/8, mixed load 16/16, 243,173-token needle found; ≥ 660 MiB free at peak |
| Base model with Clef enabled vs disabled | first token bitwise identical; decode differs at ~1e-3 in logprobs (see below) |

### Clef: records up to 32K and images (2026-10-13, production)

`FA2SM70_CLEF_NMAX=32768`, `FA2SM70_CLEF_MEDIA=1`, NVLink pair, under live traffic.

| Gate | Result |
|---|---|
| Offline image gate (endpoint record vs Clef `encode_record`; vLLM expansion vs endpoint record) | 3/3 records (1 image of 1,200 tokens, 2 images, text only) |
| Clef text, 10 records, vs production of 2026-10-12 | 35/35, max \|Δp\| 0.0058 when quiet; 34/35, 0.018 under live load (the same 0.200/0.199 near-tie) |
| `clefkart`: 15 image records with known answers | 16/16 decisions correct, p ≈ 0.99; repeat bitwise, also across two runs; video / bad URL / non-list → 400 |
| `dlinnye`: facts at 5/50/95 % of 4K, 12K, 24K, 31K-token records + records without facts | 40/40; a full 32,768-token record takes 28 s |
| `kartinki … clef`: 8 chat requests × 2 images + 2 Clef records | 8/8 and 2/2 |
| 243,173-token needle | found; 228.4 s alone at 1293–1342 MHz, 78–82 °C (2026-10-12: 219.9 / 241.2 s); 303.6 s with 2–6 other requests running |
| Clef memory | rank 0: 354 MiB (+64 MiB for 32K), rank 1: 121 MiB; ≥ 602 MiB free at the needle peak, no allocation failures |

The first rollout returned 400 for every image record: vLLM's media connector returns images wrapped in
`MediaWithBytes` (PIL image plus original bytes, used for cache hashing), and the transformers image
processor used to count image tokens rejects the wrapper. The offline gate of that time tested the
connector and the processor separately on plain PIL images and missed it. The endpoint now unwraps the
image for counting only and hands the wrapper to the engine, like chat does; the offline gate runs the
whole chain and fails on the old code with the same error.

### Incident 2026-10-13: `Encoder cache miss` in the MTP drafter

**What happened.** Production died at 14:42 with `AssertionError: Encoder cache miss` in
`propose_draft_token_ids → _gather_mm_embeddings`. Before that it had served ~11 hours and 4,805 Clef
records, most of them with images. The autostart loop brought it back in 6 minutes.

**Cause: a vLLM scheduler defect, made frequent by Clef records with images.**
* The MTP/eagle drafter reads the target's window shifted by one position.
* `_try_schedule_encoder_inputs` honours that shift when deciding whether an image is needed at all,
  but not when it checks whether the window contains any of the image's embeddings.
* So when a prefill chunk ends exactly at an image's first token and the image is not cached, the
  encoder is skipped while the drafter reads that token.
* Clef records put the image right after `STATE:`, at a fixed offset. Under mixed load a chunk
  sometimes ends exactly there.
* The assertion had never appeared in the logs before images were enabled for Clef.

**Fix: `FA2SM70_MM_DRAFT_SHIFT=1`.** It is a lever because the virtualenv is shared.
* The scheduler includes the drafter's shift in that check.
* The drafter skips a missing encoder output with a one-time warning instead of asserting. This is a
  safety net only: draft tokens are verified by the target anyway.

**Validation:**
* `sm70/tests/test_mm_draft_shift.py` (CPU) reproduces the defect with the lever off and passes with it
  on; neighbouring cases schedule no extra encoder work.
* In production, on an idle server, a chat whose image starts exactly at position 2048 (the first
  chunk boundary) was answered correctly ("Blue.") with no warning in the log; before the fix this
  request crashes the engine.
* After the restart:
  * Clef with images 16/16;
  * chat images 8/8 together with Clef 2/2;
  * Clef text 34/35 against 2026-10-12: max |Δp| 0.0058, the flip is again a 0.20/0.20 tie in the
    meaningless record 9.

**Lesson for restarts.** A graceful shutdown can hang on "Waiting for connections to close" when a
client keeps a streaming request open after the engine is gone. A second signal did not help; the
API server had to be killed. The GPUs were already free at that point.

### Incident 2026-10-14: graph replays wrote to released selection buffers

**What happened.** Production died three times in two hours:
* 13:22, `CUDA error: an illegal memory access` with an MMU fault (Xid 31) on both GPUs at once;
* 13:33, `OverflowError: Python integer 85899345940 out of bounds for int32` while scheduled draft
  tokens were written into `token_ids_cpu`;
* 14:24, an MMU fault again.

Two crashes came 10 s and 73 s after the number of concurrent requests reached a new maximum for
the process. In the month before, no crash of this class appeared, with the same code and similar
traffic. Why it started that day is not established.

**Cause: buffers that are replaced when they grow, while captured CUDA graphs keep their address.**
* The sparse-decode selection buffers (`_sel_bufs`, `_sel_bufs_h` in
  `vllm/v1/attention/backends/fa2_sm70.py`) are module-level and shared by every attention layer,
  including the MTP drafter's.
* When the virtual batch (requests × (k+1) rows) grew past its largest size so far, a new tuple
  was allocated and the old one released.
* The full decode graph and the drafter's own graphs, captured lazily under load, had baked the old
  address and kept writing to it. That memory went to other tensors or back to the driver (memory
  trim).
* 85899345940 is `0x00000014_00000014`: a pair of int32 selection counts (20, 20) of one row with two
  KV heads, read as one int64 draft token id.
* The same class was fixed for the GDN state indices on 2026-09-08. The selection buffers came later
  with sparse decode and were never covered.

**Fix.**
* The selection buffers now retire through the same keep-list as the other module caches.
* `FA2SM70_KEEP_BUFS=1` keeps every retired attention buffer, the selection buffers and the per-layer
  workspace alike.
* `FA2SM70_SELBUF_PROBE=N` fills a retired selection buffer with a sentinel and checks it every N
  calls, outside graph capture.

**Validation:**
* `sm70/tests/test_selbuf_keep.py` (CPU) covers three properties:
  * without the levers the old buffer is released exactly as before;
  * with them it is kept;
  * the probe reports a write to a retired buffer and stays silent otherwise.
* **In production the probe proved the mechanism.** Minutes after the restart the rows grew from 8 to
  12, then further. On both ranks the sentinel was overwritten in every retired tensor (`_SELBUF[0..3]`,
  `_SELBUF_H[0..1]`), i.e. captured graphs kept writing to the old addresses.
* Ruled out by fuzzing:
  * the context n-gram drafter: 1.6 M calls, never an id outside the vocabulary;
  * the draft hand-off from runner to scheduler: 264 k steps on the real code. It can produce this value
    only by reading a pinned row that was not written in this step, which the real code does not do;
  * the int4 draft argmax: inactive in production;
  * hardware: NVLink, PCIe and ECC counters clean.
* After the restart: chat, Clef, and an image exactly at the first chunk boundary all answered
  correctly.

**Follow-up the same day: the same defect in the kernels, and a guard.**
* An audit of every buffer reachable from a captured graph found three more that were released on
  growth. All three are C++ `static` buffers, so `FA2SM70_KEEP_BUFS` cannot reach them:
  * `_Eg` of `sparse_select_rows`: per-head selection with `topf > 0`, production 0.15;
  * the split-K workspaces `кэш` and `кэш_hmma` of `w8a16_gemv`.
* Until the kernel fix, production ran with `FA2SM70_MTP_GRAPH=0`: no drafter graphs.
* Proof on a GPU with the production kernel code, `sm70/tests/test_graph_keep.py`. Capture a graph at a
  small batch, grow the buffer eagerly, place "victim" tensors of the old size, replay the graph:

  | Buffer | Before the fix | After the fix |
  |---|---|---|
  | `w8a16_gemv` split-K | 573,440 victim elements overwritten | 0 |
  | `_Eg` | 16,384 elements overwritten (exactly the old capacity) | 0 |

  The selection of the old and new builds is bitwise identical on the same inputs.
* **Fix.**
  * The kernels keep every retired buffer in a static list.
  * `_Eg` starts at 64 rows (16 requests × 4), so in production it does not grow at all.
* `FA2SM70_DRAFT_GUARD=1` drops a draft row with an id outside [0, vocabulary) and logs it, instead of
  crashing the engine. CPU test: `sm70/tests/test_draft_guard.py`.
* Deployed 2026-10-14 17:35 with `FA2SM70_MTP_GRAPH=1` restored. The kernels were prebuilt into a new
  build directory with the production environment, so startup did not recompile.
* Checks after the restart:
  * chat answered correctly;
  * Clef 0.9921;
  * an image exactly at the first chunk boundary on an idle server: "Blue.";
  * the drafter graphs were captured again.

### Fix 2026-10-18: a 4-token prefill chunk ran as uniform speculative decode (production)

Found by the 2026-10-14 fuzzers, reproduced and fixed on a test pair of GPUs, in production since
2026-10-18 17:14 (`FA2SM70_UNIFORM_PREFILL_FIX=1`).

* `_is_uniform_decode` looks only at the number of scheduled tokens. A single request whose last prefill
  chunk is exactly k+1 = 4 tokens (prompt length 2048·m + 4) looks like one speculative decode row.
* That step replays the full speculative CUDA graph. The GDN metadata builder does not refresh the
  persistent spec state for it, so the graph runs on stale state and writes into another request's blocks.
  The answer turns to garbage from the first token.
* Fix (`gpu_model_runner._fa2sm70_ne_odnorodnyj`): a step is uniform only if every row carries exactly k
  draft tokens; otherwise `force_uniform_decode=False` and the step takes the piecewise path.
* Stand, `sm70/tests/server/probe_uniform.py` (one request of length 2048·m + 4 against the same request
  chunked differently next to a decoding neighbour): before the fix L = 2052 and 4100 gave different greedy
  tokens (max |Δ logprob| 4.96 and 5.30); after it the tokens match, and the residual |Δ logprob| (0.18,
  0.16) is the server's own repeat noise. Controls 2051 and 2053 (last chunk 3 and 5 tokens) unchanged.
  Base logprobs bitwise identical to the unpatched server, Clef 70/70 with max |Δp| 0.

### Stand 2026-10-18: fixes verified (in production since 2026-10-19)

A test pair of GPUs (GPU0-1, PCIe, `NCCL_P2P_DISABLE=1`) ran the production stack with a vLLM copy, so the
production venv stayed untouched. Every fix is a lever with default `0` (LEVERS.md, marked **stand**).

* **Sparse decode: stale mean keys for every request but the first in a decode batch** (`FA2SM70_BLKMEAN_ALL`,
  `sm70/fa2_sm70/csrc/sparse_select.cu`). This defect **is live in production** until the lever is enabled.
  * `blk_mean_update` takes the slot list in slices of 64 and recomputed only the sub-block of the first slot.
    After a decode step a slice holds k+1 = 4 slots of up to 16 requests (64 requests without MTP). The
    sub-blocks decode wrote for the other requests kept old means (zeros for a new sub-block).
  * Sparse decode (contexts from 16,384 tokens) then skipped those sub-blocks once they left the 512-token
    recent window. A prefill chunk that runs alone is correct; in a mixed step (decode rows followed by a
    prefill chunk) the slices no longer start at sub-block boundaries, and some prompt sub-blocks of the
    chunk are missed too (unit test, mixed slice: 5 of 8 right before the fix).
  * Symptom (`sm70/tests/server/probe_recall.py`: 8 parallel chats, 22-26K context, the model writes a code
    first and repeats it after 1-2K tokens): 5/16 correct, the wrong ones "almost right" (482910 instead of
    482915). A live audit of the table (means recomputed from the KV every 50 steps) showed batch position 0
    clean and positions 1..6 100 % stale.
  * With the fix: recall 16/16, audit 0 stale over 2,350 steps with up to 7 requests and batch reshuffles.
    `sm70/tests/test_blkmean_update.py` (GPU): 7/33 touched sub-blocks right before, 33/33 after.
* **Padded drafter rows** (`FA2SM70_PAD_META_FIX`, open defect below). Under real load, 752 of 2000 padded
  drafter calls carried real slots of a previous call in the padded tail: 3858 rows where the draft layer
  wrote K/V into another request's slot. With the fix: 0. MTP acceptance was indistinguishable in this
  scenario (2.29 without, 2.31 with) and recall 16/16 either way, so the stray writes mostly land on
  positions the drafter rewrites anyway. The fix costs nothing on a normal step.
* **Prefix cache under Clef load** (`FA2SM70_FREE_UNHASHED_HEAD`, `FA2SM70_NO_SPEC_BLOCKS_MAXTOK1`,
  `sm70/tests/server/probe_prefix2.py`). A document repeated after Clef load: 21,391 tokens hit 0 → 16,384
  (16.2 s → 4.1 s); 56,951 tokens hit 0 → 49,152 (44.5 s → 6.9 s); the control without load is unchanged
  (16,384 hit, 4.3 s → 4.2 s).
* **Clef record preempted mid-prefill** (`FA2SM70_CLEF_A1`, open defect below; `sm70/tests/server/probe_a1.py`
  with the KV pool forced to 5 blocks). The record was preempted 55 times; its answer equals the record run
  alone, max |Δp| 0.0001.
* **Clef records of 34,816 tokens** (`FA2SM70_CLEF_NMAX=34816`, in production since 2026-10-19): `proverka_stenda.py
  dlinnye 34816` 50/50 including a fact at 95 % of a 34,816-token record; head against PyTorch fp32 at
  N = 34,816: max |Δlogit| 6.79e-4, max |Δp| 7.0e-5, argmax 8/8 (`proverka_golovy.py`, now sized with
  `Nmax = max(16384, N)`); +160 MiB on rank 0.
* The stand then served real traffic with all of the above enabled: Clef equal to the reference (158 values,
  max |Δ| 0), needle at 60K found, mixed Clef + chat load all correct, MTP acceptance 2.33.

### Deployment 2026-10-19: the stand fixes in production

`FA2SM70_BLKMEAN_ALL=1`, `FA2SM70_PAD_META_FIX=1`, `FA2SM70_FREE_UNHASHED_HEAD=1`, `FA2SM70_NO_SPEC_BLOCKS_MAXTOK1=1`,
`FA2SM70_CLEF_A1=1`, `FA2SM70_CLEF_NMAX=34816`, `FA2SM70_CLEF_R=4` (now in `configs/qwen38-27b-int8-clef.env`).
Production restarted 2026-10-19 02:40, ready 02:44, no kernel rebuild at startup (the prefill kernel with the
`blk_mean_update` fix was prebuilt into a new build directory with the GPUs hidden).

* Levers confirmed from the startup log of every worker; KV capacity 262,144 tokens; free GPU memory
  898 / 1182 MiB.
* Clef 34K (`dlinnye 34816`) 50/50; recall (`probe_recall.py`) 4/4.
* Clef against the stand with the same fixes: max |Δp| 0.0227, 34/35; the flip is record 9, a tie of six
  options (stand 0.1984 against 0.1959). The stand service with the fixes equals the stand reference to
  0.0000 on all 10 records, i.e. the fixes do not change Clef numbers. Production differs from the stand
  already on short records (0.0013 on 369 tokens): the all-reduce summation order differs between NVLink
  and host-staged PCIe.
* Load since then: [docs/qwen38-27b-sm70-performance.md](../qwen38-27b-sm70-performance.md), section 3.

### Host-RAM KV offload (2026-10-16 … 10-18, off in production)

`FA2SM70_KV_OFFLOAD=1` (`sm70/integrations/vllm_sm70/fa2sm70_kv_offload.py`). The stock
`OffloadingConnector` does not fit this model: it is not `SupportsHMA` (vLLM then disables the hybrid manager
that GDN needs), it wants one page size for all groups, and it copies by `stride(0)`, which misses the carved
int8 scale tables and the selection side tables.

* CPU harness on the real `Scheduler`: 0 label failures, corruption control 29,526; GPU: byte-exact
  round trip of every piece of a block.
* Stand, without the prefix-cache levers, after Clef load (the GPU cache held 0 of the documents): a
  21,391-token repeat got 16,384 tokens from RAM (4.2 s instead of 16.2 s), a 56,951-token repeat 49,152
  (7.0 s instead of 44.5 s). Peak GPU memory unchanged.
* Identity (`sm70/tests/server/probe_ozu_ident.py`, `probe_ozu_ident2.py`): one block restored from RAM is
  byte-equal to a GPU hit (tokens, logprobs, top-5). With 2 and 3 blocks the prefill output and the first
  ~19 tokens are byte-equal and logprobs diverge from position ~20 by ~1e-2. Two repeats that both hit the
  GPU cache on the same blocks diverge from position 20 as well (4.5e-3): that is the server's own repeat
  nondeterminism, not the connector.
* With `FA2SM70_FREE_UNHASHED_HEAD` and `FA2SM70_NO_SPEC_BLOCKS_MAXTOK1` the GPU cache alone keeps those
  documents, and the RAM tier is not reached in the same scenario.

### Open defects (fuzzing, 2026-10-14)

CPU fuzzers driving the real code found defects that do not crash but change results silently. The
fixes are tested on copies only and wait for quality gates on a test pair of GPUs: graph versus eager,
MTP acceptance, needle ladder, Clef parity.

* **GDN accepted-token count in the full graph** (`vllm/v1/attention/backends/gdn_attn.py`).
  * With `FA2SM70_GDN_SHARED_BUF=1`, `update_block_table` skips the `num_accepted` refresh for groups 1..9.
  * Their per-builder buffer, baked into the full decode graph at capture, therefore stays at the capture
    values.
  * Result: in ~84 % of speculative full-graph steps, 43 of the 48 GDN layers start from the state of a
    wrong position.
  * The GRAN_SRC/DST state-block-boundary substitutions are also applied to group 0 only.
  * Fix: share the buffer, and rebuild groups 1..9 on boundary steps. Fuzzer mismatches go to 0.
  * **Does not apply to the production layout.** Production builds a single GDN group: the startup log
    `[fa2_sm70 ГРУППЫ]` shows group 0 = `GDNAttentionMetadataBuilder` with all 48 layers and group 1 = the
    17 attention layers (16 + MTP). The defect needs GDN groups 1..9, which other block layouts produce.
* **Clef record preempted mid-prefill.**
  * The runner keeps the request, so `req_begin` is not called again.
  * `span_add_k` then accumulates the already processed positions a second time.
  * Fix: track the expected next position per slot and restart the record when it does not match.
    Behind `FA2SM70_CLEF_A1`, verified on the stand 2026-10-18 (above); in production since 2026-10-19.
* **Padded rows of the MTP drafter batch** (`eagle.py`, `_дополнить_мету`). When the number of real rows
  shrinks for the same padded size, the padded rows keep the previous step's sequence lengths and slots,
  and the draft layer writes K/V into other requests' slots. Fix behind `FA2SM70_PAD_META_FIX`, verified on
  the stand 2026-10-18 (above); in production since 2026-10-19.
* **Configuration limit.** The sparse decode selection supports at most 4096 key blocks per row. With
  `FA2SM70_SPARSE_B=64` that is exactly a 262,144-token table.
* Ruled out by the same fuzzers: the encoder scheduling for the drafter (900,000 cases, 0 violations in
  the production mode), the virtual batch and selection bounds, and Clef segments without preemption.

## Known effects and hazards

* **Decode numerics depend on GPU memory layout.** With MTP speculation, any change in the memory
  allocated at startup before the KV pool and CUDA graphs shifts decode logprobs by ~1e-3.
  A plain 290 MiB allocation with Clef disabled reproduces exactly the shift that Clef causes:
  `FA2SM70_CLEF_DIAG=nosetup,dummy290` is bitwise equal to Clef enabled. Clef therefore does
  not alter the base computation, but "bitwise identical to before" cannot be a gate for any
  change that allocates memory at startup. Compare against an equally sized dummy allocation
  instead, or gate on behavior (needles, ladders). The mechanism inside the base stack is not
  yet identified.
* **`/v1/completions` with `echo` + `logprobs` on long prompts** computes full-vocabulary logits for
  every prompt position of a chunk: 2048 × 248,320 × 4 B = 1.89 GiB per rank. This exhausts
  memory and kills the engine. It is a property of vLLM's prompt-logprobs path, not of Clef. Do not
  expose prompt logprobs to untrusted clients (or limit them) until this is reworked.
* **DFlash2 drafter configuration.** With the alternative DFlash2 drafter instead of MTP,
  Clef's 290 MiB on rank 0 leaves too little headroom: mixed Clef + chat load ran out of memory in
  `qwen3_dflash._project_context_kv`. Shrink Clef (fewer ring slots, `FA2SM70_CLEF_R=1`) before
  combining them.
* **Preemption thrash in an overcommitted pool.** vLLM admits a waiting request when its first chunk fits,
  without checking that the whole prompt will. With the pool forced to 5 blocks, a 24K Clef record next to
  a decoding chat was admitted, could not grow, was preempted and recomputed 55 times in a row; for 17 min
  the chat decoded at ~1.3 tokens/s instead of ~25 (answers stayed correct). With the production pool of
  262,144 tokens this needs the concurrent contexts to exceed the pool.
* **NCCL P2P over a PCIe pair can hang at start.** On GPU0-1 (PHB) the server hung several times at NCCL
  P2P connection setup when started seconds after the previous process left the cards; a bare NCCL test
  hung at the same time and passed minutes later. A 60 s pause did not cure it, `NCCL_P2P_DISABLE=1` did.
  The price is prefill: ~820 tokens/s flat from 1K to 64K on that pair (the launcher's own measurement of
  2026-08-21 credits P2P over PCIe with ×1.52 prefill). Decode 51-62 tokens/s, following MTP acceptance.
* **Start the server from a neutral directory.** The kernel warm-up runs `python -u -` with a here-document,
  which puts the current directory first in `sys.path`. Started from a directory that holds another
  `fa2_sm70` tree, the warm-up builds that tree into `FA2SM70_BUILD_DIR`, and the workers rebuild it back
  (+30 min at start). Two different command hashes in a row in `.ninja_log` give it away.
* **Thermal state matters.** On hot cards the 250K prefill varies between ~187 s and ~240 s. Read
  clocks (`nvidia-smi --query-gpu=clocks.sm,temperature.gpu`) next to every timing.
