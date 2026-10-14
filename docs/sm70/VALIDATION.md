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
* **Thermal state matters.** On hot cards the 250K prefill varies between ~187 s and ~240 s. Read
  clocks (`nvidia-smi --query-gpu=clocks.sm,temperature.gpu`) next to every timing.
