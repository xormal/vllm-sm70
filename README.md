# vLLM-SM70

**English** · [Русский](README.ru.md)

**The production vLLM stack for NVIDIA Volta (V100, `sm_70`).** It serves Qwen3.8-27B (GPTQ-Int8)
with a 262,144-token context on two 32 GB cards, plus Cloudflare's Clef decision model on the same
weights.

This repository is exactly the code that has served production traffic since 2026-10-12: the vLLM
fork, the Volta kernel library and the integration layer, with the launcher and lever set of the
running server. It is a fork of [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (a Volta-focused
fork of [vllm-project/vllm](https://github.com/vllm-project/vllm)), branch `sm70`.

Model weights and Clef adapters: **[alexxorm/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef](https://huggingface.co/alexxorm/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef)** on Hugging Face.

| | |
|---|---|
| **Model** | [Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef](https://huggingface.co/alexxorm/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef) (int8 GPTQ of [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16), base [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)) + adapters for [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) |
| **Hardware** | 2× NVIDIA V100 32 GB, tensor parallel 2, NVLink or PCIe |
| **Context** | 262,144 tokens, int8 KV cache |
| **Modalities** | text, images, video, tool calls (OpenAI-compatible API) |
| **Decisions** | `POST /v1/systemone`: Clef (Jev/SystemOne API) on the same weights, records up to 32,768 tokens with images, +0.35 / +0.12 GB of GPU memory |
| **Speculation** | native MTP head, k = 3, with n-gram assist |

---

## Why

Modern attention kernels (FlashAttention ≥ 2 targets `sm_80+`, FlashInfer `sm_75+`) and most quantized
GEMM paths do not run on Volta. Upstream vLLM on a V100 is therefore slow, short-context, or does not start at
all for hybrid models such as Qwen3.5/3.8 (Gated DeltaNet + full attention). This stack closes the gap
with its own kernels and integration:

* **Attention:** CUTLASS-based FMHA prefill, sparse prefill with a measured error bound, split-K
  paged decode with an int8 KV cache, and a virtual-batch path for speculative positions. All kernels are
  verified against fp32 exact-softmax oracles.
* **Linear layers:** GPTQ-int8 GEMV on tensor cores for decode, turbomind s884 int8 GEMM and
  cuBLAS-reconstruct for prefill, 12-bit (W12) attention projections, and an int8 `lm_head`.
* **Hybrid model plumbing:** int16 Gated DeltaNet state pools, split KV pools, block sizes aligned for 262K
  on 64 GB of total GPU memory, and shared GDN metadata across groups.
* **Speculation:** the model's own MTP head (int8, own CUDA graph, int4-sliced vocabulary argmax), with
  n-gram proposals when they out-accept it.
* **Clef on the same weights:** per-row low-rank weight correction inside the int8 GEMM kernels plus a native
  CUDA joint schema head. Clef requests get Clef's weights, chat requests in the same batch are untouched,
  and no second model is loaded.

## Measured on the reference deployment

| | |
|---|---|
| 250K-token prefill | 186.8 s (2026-09-15), median 215 s (2026-09-23), NVLink pair; depends on card temperature |
| Decode, single stream | 55.8 / 58.4 / 48.6 tok/s at 32K / 128K / 250K on filler, 63.8 tok/s on natural text (MTP k=3) |
| Long-context retrieval | code word at the start of a 243,173-token prompt reproduced verbatim (2026-10-12) |
| Clef vs the original Clef in fp16 | 35/35 decisions, max \|Δp\| 0.0093 on 10 records of 152–16,384 tokens |
| Clef with images and long records (2026-10-13, production) | 16/16 known-answer image records (color, digit, shape, word, order of two images, image + 11K tokens of text), repeat bitwise; facts at 5/50/95 % of 4K–32K-token records 40/40 |
| Load | 10 Clef records + 6 chats concurrently, and 8 × 2-image storms followed by a 243K prompt: no failures |

Details, tools and known effects: [docs/sm70/VALIDATION.md](docs/sm70/VALIDATION.md).

## Requirements

* 2× NVIDIA V100 32 GB (tested: one NVLink pair and one PCIe pair), driver supporting CUDA 12.8 (tested 580).
  The two GPUs must be **dedicated** to the server: the memory plan is fixed (`--kv-cache-memory-bytes`,
  `--gpu-memory-utilization 0.96`), so another process allocating GPU memory while the server starts makes the
  start fail with an out-of-memory error.
* CUDA **12.8** toolkit (`nvcc`) and a host gcc it accepts (tested gcc 11.4). The sm70 kernels are JIT-compiled.
* Python **3.12**, ≥ 32 GB RAM for the build with `MAX_JOBS=8` (~3 GB per job), ~35 GB disk for the model, ~2 GB for build products.

## Installation

```bash
git clone -b sm70 https://github.com/xormal/vllm-sm70.git
cd vllm-sm70
CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=8 scripts/install_from_source.sh
```

[`scripts/install_from_source.sh`](scripts/install_from_source.sh) creates `.venv` with the exact package
versions of the deployment ([`requirements-sm70.txt`](requirements-sm70.txt): torch 2.9.1+cu128,
transformers 4.57.6, triton 3.5.1, …). It then builds `flash-attention-v100` and vLLM for `sm_70`. The vLLM
build takes about an hour with `MAX_JOBS=8`. If your pip configuration adds unreachable indexes, run it
with `PIP_CONFIG_FILE=/dev/null`.

**Offline or slow network.** vLLM's CMake configuration clones its dependencies from GitHub. You can
point it at local checkouts at the pinned revisions instead:

| Variable | Repository | Revision |
|---|---|---|
| `VLLM_CUTLASS_SRC_DIR` | NVIDIA/cutlass | `v4.2.1` |
| `TRITON_KERNELS_SRC_DIR` | triton-lang/triton, **the `python/triton_kernels/triton_kernels` subdirectory** | `v3.5.0` |
| `FLASH_MLA_SRC_DIR` | vllm-project/FlashMLA | `c2afa9cb93e674d5a9120a170a6da57b89267208` |
| `QUTLASS_SRC_DIR` | IST-DASLab/qutlass | `830d2c4537c7396e14a02a46fbddd18b5d107c65` |
| `VLLM_FLASH_ATTN_SRC_DIR` | vllm-project/flash-attention (with submodules) | `2adfc8c2177c5b0e8ddeedfd5a8990d80eb496ff` |

## Serving

```bash
huggingface-cli download alexxorm/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef \
    --local-dir /models/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef

PY=.venv/bin/python MODEL_DIR=/models/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef \
CARDS=0,1 PORT=8000 scripts/serve_qwen38_clef.sh
```

[`scripts/serve_qwen38_clef.sh`](scripts/serve_qwen38_clef.sh) reproduces the production server exactly:
it loads the 103 levers of [`configs/qwen38-27b-int8-clef.env`](configs/qwen38-27b-int8-clef.env), probes the
toolchain, chooses fp16 or int12 all-reduce from the NVLink topology, refuses busy GPUs or ports, builds
all kernels in one process, and starts the OpenAI-compatible server with the production arguments. The first
start compiles the kernels (15–30 min) and repacks the attention weights to W12. Both are cached in
`STATE_DIR` (default `~/.cache/vllm-sm70`), and later starts take a few minutes.

| Variable | Default | |
|---|---|---|
| `MODEL_DIR` | required | checkpoint directory; Clef adapters are read from `MODEL_DIR/clef` |
| `CARDS` | `0,1` | the two GPUs (nvidia-smi indices) |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | listen address |
| `SERVED_NAMES` | `Qwen3.8-27B-ABLITERATED` | model names in the API |
| `THINK` | `false` | default `enable_thinking` (per-request override allowed) |
| `FA2SM70_CLEF` | `1` | `0` serves the base model only |
| `FA2SM70_CLEF_NMAX`, `FA2SM70_CLEF_MEDIA` | `32768`, `1` | Clef record limit in tokens; images in Clef records (`0` rejects them) |
| `STATE_DIR`, `PY`, `CUDA_HOME` | see script | cache directory, Python, toolkit |
| `DRYRUN=1` | | print the resolved environment and command without starting |

### Chat, vision, tools

The standard OpenAI-compatible endpoints: `/v1/chat/completions` (text and images up to 20 per request,
video up to 2), `/v1/completions`, `/v1/responses`, tool calling (`qwen3_coder` parser) and reasoning
(`qwen3` parser).

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-27B-ABLITERATED",
  "messages": [{"role": "user", "content": "Explain warp divergence in one paragraph."}]}'
```

### Decisions: `POST /v1/systemone` (Clef)

```bash
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "clef",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle the message?",
                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
    "urgency":    {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage":     {"type": "noul", "instructions": "Is a service down?"}}}'
```

With images (`"images"`: http(s) URLs, data URLs, bare base64 or chat-style `{"image_url": {"url": ...}}`):

```bash
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "clef",
  "state": {"task": "Triage the attached screenshot."},
  "images": ["https://example.com/screenshot.png"],
  "questions": {"error": {"type": "noul", "instructions": "Does the screenshot show an error message?"}}}'
```

```json
{"model": "clef", "answers": {
  "department": {"type": "choice", "choice": "technical", "confidence": 0.9185,
                 "probabilities": {"billing": 0.0815, "technical": 0.9185}},
  "urgency":    {"type": "score", "score": 1.8064, "confidence": 0.8546,
                 "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                 "probabilities": {"0": 0.0482, "1": 0.0972, "2": 0.8546}},
  "outage":     {"type": "noul", "noul": 0.8999}},
 "usage": {"input_tokens": 300, "output_tokens": 0}}
```

* **Question types:** `choice` (dict of options → `choice`, `probabilities`), `score` (ordered list →
  expected `score`, `probabilities`), `noul` (→ probability of "yes"). `state` is a string or any JSON value.
* **Limits:** 32,768 tokens per record, images included (`FA2SM70_CLEF_NMAX`; Clef's own default
  `max_length` is 16,384, lower it per request with `"max_length"`), up to 16 images per record, 64
  questions, 448 options in total, 256 options per question. Videos are rejected with 400. Two records are
  processed concurrently and further ones wait.
* **Images** are encoded as in Clef's `encode_record`: the image tokens follow `STATE:\n`. Clef's vision
  encoder is bitwise identical to this checkpoint's (333/333 tensors), so the served encoder is reused as is.
  Images are fetched like in chat (same media connector, same domain and path restrictions). The worker
  checks that the record the engine received (length and checksum) equals the one the endpoint encoded
  and refuses the request otherwise.
* **How it works.** The record is encoded with Clef's template and runs as a one-token prefill. For its
  rows only, the int8 GEMM kernels add `deabl` (rank 1, layers 2–63) and Clef's exact rank-256 LoRA (layers
  40–63, streamed from pinned host memory), which turns the served weights into Clef's. The final hidden
  states are tapped into Clef's joint schema head, a native CUDA implementation. Clef requests use their
  own `cache_salt` and never read the prefix cache. Code: [`sm70/integrations/vllm_sm70/fa2sm70_clef.py`](sm70/integrations/vllm_sm70/fa2sm70_clef.py),
  [`sm70/fa2_sm70/csrc/clef.cu`](sm70/fa2_sm70/csrc/clef.cu).

## Configuration

The deployment is driven by environment levers. [docs/sm70/LEVERS.md](docs/sm70/LEVERS.md) explains the
ones that change computation or memory, and [`configs/qwen38-27b-int8-clef.env`](configs/qwen38-27b-int8-clef.env)
holds all of them with production values. Change one at a time and re-run the gates.

## Repository layout

```
vllm/                     vLLM 0.0.3 (1Cat-vLLM) + the production Python runtime:
  v1/attention/backends/fa2_sm70.py      Volta attention backend (FA2_SM70)
  v1/attention/backends/gdn_attn.py, model_executor/layers/fla/   Gated DeltaNet path
  v1/spec_decode/                        MTP (eagle.py), DFlash/DFlash2 drafters, n-gram assist
  v1/core/, v1/kv_cache_interface.py     split KV pools, hybrid block alignment
  v1/worker/gpu_model_runner.py          memory trimming, step phases, Clef hook points
csrc/, flash-attention-v100/, lmdeploy/  1Cat-vLLM C++/CUDA (GPTQ/AWQ sm70 GEMM, FA-V100, TurboMind)
sm70/                     Volta kernel library and integration layer (MIT)
  fa2_sm70/               JIT CUDA extensions (prefill, decode, GDN, W8/W12/tm8, exchange, Clef, ...)
  fa2_src/                bundled CUTLASS headers (BSD-3-Clause) and Volta FMHA primitives
  third_party/sm70_turbomind/  TurboMind GEMM headers used by the tm8 extension (Apache-2.0)
  integrations/vllm_sm70/ runtime layer loaded via sitecustomize: linear paths, all-reduce, MTP head,
                          flash_attn_v100 shim, Clef (fa2sm70_clef*.py, clef/), quantize/
  docs/, tests/, benchmarks/   kernel documentation and oracle tests; see sm70/README.md
scripts/                  install_from_source.sh, serve_qwen38_clef.sh
configs/                  production lever set
docs/sm70/                LEVERS.md, VALIDATION.md
docs/upstream/            the original 1Cat-vLLM README
```

The kernel library has its own documentation: [sm70/README.md](sm70/README.md) and
[sm70/docs/KERNELS.md](sm70/docs/KERNELS.md). Many code comments in `sm70/` and in the runtime files are
engineering notes in Russian. They record the measurement behind each decision.

## Changes relative to 1Cat-vLLM

The `sm70` branch starts at 1Cat-vLLM `437b358e5` and adds, in order:

1. the working-tree changes the deployed binaries were built from (GPTQ/AWQ sm70 GEMM, flash-attention-v100 paged paths);
2. the production Python runtime of vLLM (44 modified and 8 new modules);
3. `sm70/`: kernels, integration layer, Clef runtime and vendored TurboMind headers;
4. the launcher, the production lever set and the pinned environment, plus the scripted source build;
5. this documentation (English and Russian), lever reference and validation guide;
6. removal of upstream vLLM CI workflows that do not apply to a standalone repository;
7. fixes found by building and serving from a fresh clone: the build PATH, and the out-of-tree
   TurboMind header needed by the tm8 extension.

`git log 437b358e5..sm70` shows each step with a description.

## Known limitations

* Volta only. Other architectures are untested, and some paths assume `sm_70`.
* `/v1/completions` with `echo` and `logprobs` on long prompts allocates full-vocabulary logits per
  prompt chunk (1.89 GiB per rank) and can exhaust memory. Do not expose prompt logprobs to untrusted
  clients.
* With speculation, decode logprobs depend at the ~1e-3 level on the GPU memory layout at startup
  (see VALIDATION.md). Bitwise comparisons are only meaningful between identical configurations.
* Clef records go up to 32,768 tokens. The native head is verified against PyTorch fp32 at 32,768 and a
  fact-retrieval test passes 40/40 up to 32K, but Cloudflare did not evaluate Clef beyond its own 16,384
  default.
* Clef with images is checked on known-answer records and for determinism, not yet against the original
  Clef's probabilities. This checkpoint ships the fast Qwen2-VL image processor while Clef uses the slow
  one: image token counts are identical, pixel values may differ slightly. Videos are not supported in
  `/v1/systemone`.

## License

* vLLM and 1Cat-vLLM code (everything outside `sm70/`): Apache License 2.0, see [LICENSE](LICENSE).
* `sm70/`: MIT, see [sm70/LICENSE](sm70/LICENSE), except the vendored third-party code:
  CUTLASS in `sm70/fa2_src/cutlass` (BSD-3-Clause, see its `LICENSE.txt`) and TurboMind headers in
  `sm70/third_party/sm70_turbomind` (Apache-2.0, see its `LICENSE`).
* Model weights are distributed separately under Apache-2.0 (see the model card).

## Acknowledgements

[vLLM](https://github.com/vllm-project/vllm), [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) by 1Cat AI,
[CUTLASS](https://github.com/NVIDIA/cutlass), [LMDeploy/TurboMind](https://github.com/InternLM/lmdeploy),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (vendored in vLLM), the
[Qwen](https://huggingface.co/Qwen) team, [Blackfrost](https://huggingface.co/Blackfrost-AI), and
[Cloudflare](https://huggingface.co/Cloudflare/clef) for Clef.
