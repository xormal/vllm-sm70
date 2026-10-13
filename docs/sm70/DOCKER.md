# Docker

One image serves every supported layout: 2, 4 or 8 V100 cards, 16 or 32 GB, NVLink, PCIe or a mix,
on all cards of a host or only on the ones you allow. A planner splits the allowed cards into groups,
and every group runs one replica in its own container behind nginx.

| | |
|---|---|
| Image | `docker/Dockerfile.sm70`: CUDA 12.8 (pinned: CUDA 13 dropped Volta), Python 3.12, the pinned environment of the reference deployment, vLLM and flash-attention-v100 built for `sm_70`, the fa2_sm70 kernels compiled into the image |
| Planner | `scripts/plan_gpus.py`: standard library only; runs on the host or in the image |
| Launcher | `scripts/serve_qwen38_clef.sh`: TP = number of cards (2, 4, 8); memory profile from `configs/profiles/` |
| Model | not in the image: mount the [Hugging Face release](https://huggingface.co/alexxorm/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef) at `/models/model`, read-only |
| State | `/state`: 12-bit weight cache, torch.compile cache, Triton cache. One volume per replica |

## Host requirements

* NVIDIA driver >= 570 (CUDA 12.8).
* Docker >= 25 with Compose v2, and the NVIDIA Container Toolkit. With CDI the Docker daemon does not
  need a restart:

  ```bash
  sudo apt-get install -y nvidia-container-toolkit        # NVIDIA's apt repository
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  nvidia-ctk cdi list                                      # nvidia.com/gpu=0 ..., =GPU-<uuid> ..., =all
  ```

  Without CDI, use the classic runtime (`nvidia-ctk runtime configure --runtime=docker`, then restart
  dockerd) and pass `--runtime nvidia` to the planner.
* About 40 GB of disk for the image and its build cache, 35 GB for the model, a few GB per replica in `/state`.

## Build

```bash
docker build -f docker/Dockerfile.sm70 --build-arg MAX_JOBS=8 -t vllm-sm70:$(git rev-parse --short HEAD) .
```

About 1.5 hours without a GPU: vLLM for `sm_70` takes most of it, the fa2_sm70 kernels about 20 minutes.
Layers are ordered so that editing `sm70/` rebuilds only the kernels, and editing scripts, configs or docs
rebuilds nothing heavy. `MAX_JOBS` costs ~3 GB of RAM per job.

## Plan and run

```bash
scripts/plan_gpus.py --out deploy/                 # all cards; or --gpus 2,3,6,7 / --gpus GPU-<uuid>,...
cd deploy && MODEL_DIR=/models/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef SM70_IMAGE=vllm-sm70:<tag> \
    docker compose up -d
docker compose ps                                  # healthy = the replica has answered a chat and a Clef request
curl localhost:8000/v1/models
```

The planner writes:
* `compose.yaml`: one service per replica, cards pinned by **UUID**. Indices change inside containers
  and when cards are added. Each replica listens on `127.0.0.1:8001…` only.
* `nginx.conf`: `least_conn` over the replicas, which nginx publishes on `HOST_PORT` (default 8000), with
  timeouts long enough for a 250K-token prefill.
* `plan.json`.

Without the planner, a single replica is just:

```bash
docker run -d --name sm70 --device nvidia.com/gpu=GPU-<uuid-a> --device nvidia.com/gpu=GPU-<uuid-b> \
    --shm-size 8g -p 8000:8000 -v /models/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef:/models/model:ro \
    -v sm70-state:/state vllm-sm70:<tag>
```

The container serves on every GPU it was given (`CARDS` narrows it). Other commands are `plan`,
`prebuild`, `gates <mode> ...` (the validation client of VALIDATION.md) and `bash`.

## How cards are grouped

1. **Group size.** The group size is the smallest TP whose weight shard fits the smallest allowed card:
   32 GB → TP=2 and 16 GB → TP=4. `--tp 4` or `--tp 8` raises it. With mixed cards everything is sized
   for 16 GB.
2. **Partition.** Groups maximise the weakest link inside them first: a tensor-parallel all-reduce runs at
   the pace of its slowest hop. Ties go to total link quality, then to cards on one NUMA node. A group
   on one NUMA node gets that node's CPUs (`cpuset`).
3. **Exchange.**
   * NVLink pair: fp16 NCCL.
   * Pair without NVLink: the int12-compressed exchange (`FA2SM70_EXCH12=1`). It was measured as a gain on
     PCIe and a 0.79–0.81× loss on NVLink.
   * Groups of 4 or 8: fp16 NCCL, which uses NVLink where the group has it. The int12 exchange is an
     all-gather and moves more data than a ring all-reduce beyond two ranks.
4. **One server on 8 cards.** `--single` puts all allowed cards into one TP=8 server. It is
   experimental and only lowers single-request latency, if at all. The context limit stays 262,144
   tokens, while replicas multiply throughput.

Replicas are separate containers, not vLLM's in-process data parallelism: two sm70 pieces (the int12
exchange and the tm8 tuning broadcast) use the global process group, which in-process DP would mix up.

| Host | Plan |
|---|---|
| 2× V100 32 GB | 1 replica, TP=2, **validated** (NVLink and PCIe pairs) |
| 4× V100 32 GB, NVLink only between 2–3 (the reference host) | 2 replicas: 0–1 on PCIe (int12), 2–3 on NVLink (fp16) |
| 4× V100 16 GB, all pairs NVLink | 1 replica, TP=4 (experimental) |
| 8× V100 32 GB, DGX-1 hybrid cube-mesh | 4 replicas on NV2 pairs 0–3, 1–2, 4–7, 5–6, each on one socket |
| 8× V100 16 GB, DGX-1 | 2 replicas, TP=4, quads 0–3 and 4–7 (all pairs NVLink) |
| 8× V100 PCIe 32 GB, two sockets | 4 replicas on the PIX pairs, never across sockets |

These plans are `sm70/tests/test_plan_gpus.py` (CPU, standard library). The DGX-1 and PCIe topologies
are synthetic fixtures; the reference host's is captured.

## Memory profiles

`tp2-32g` is the production configuration. `tp4-32g`, `tp4-16g`, `tp8-32g` and `tp8-16g` derive their KV
pool and context limit from production's measured memory split. They are **experimental until
calibrated** on their hardware; [configs/profiles/README.md](../../configs/profiles/README.md) has the
derivation and the calibration procedure. A 16 GB layout can be rehearsed on 32 GB cards by holding
16 GiB on each card with another process.

## Production hardening

* Images and videos in chat and in `/v1/systemone` are fetched by the server. Set
  `ALLOWED_MEDIA_DOMAINS` (comma-separated) when clients are not trusted.
* Set `VLLM_API_KEY` to require a bearer token. Expose only the router.
* `/v1/completions` with `echo` and `logprobs` on long prompts can exhaust GPU memory (README, "Known
  limitations"). Filter it at the router if clients are untrusted.

## Validation status

| What | Where | Result |
|---|---|---|
| TP=2, 32 GB, NVLink and PCIe | reference host, outside Docker | production since 2026-09; gates in VALIDATION.md |
| Kernels compiled without a GPU (`FA2SM70_PREBUILD=1`) | reference host | same nvcc flags as the production build (`sm_70`), loads without a device |
| Launcher at 2/4/8 cards, by index and by UUID | reference host, dry runs | TP=2 environment and command identical to the previous launcher |
| Planner | CI fixtures | 13 tests |
| Encoder scheduling for the MTP drafter (`FA2SM70_MM_DRAFT_SHIFT`) | CPU test + production, 2026-10-13 | see VALIDATION.md, incident 2026-10-13 |
| Clef split at TP=8 (KV heads replicated) | CPU, real adapters | rows equal vLLM's sharding at TP=1/2/4/8; TP=2 bitwise unchanged |
| Image on GPUs, TP=4, 16 GB profiles, 8 cards | not yet | needs GPU windows / hardware |

## Troubleshooting

* **`GPU n already holds … MiB`**: another process or container uses the card. Fix the plan, or set
  `FORCE=1` if you know why.
* **vLLM refuses at start-up because the KV cache is too small for `max_model_len`**: the profile is too
  optimistic for this card. Lower `MAX_MODEL_LEN` or `FA2SM70_KV_BYTES` and calibrate.
* **NCCL hangs or errors on PCIe across sockets**: try `NCCL_P2P_DISABLE=1` for that replica.
* **Unhealthy but `/health` is 200**: the warm-up requests have not been answered yet. The first start
  builds the 12-bit weight cache and compiles graphs into `/state`. Read `docker compose logs <replica>`.
