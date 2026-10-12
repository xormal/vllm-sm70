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
