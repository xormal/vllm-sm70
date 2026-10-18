#!/usr/bin/env bash
# Serve Qwen3.8-27B-ABLITERATED GPTQ-Int8 (+ Clef on /v1/systemone) on two NVIDIA V100 32 GB
# with the exact configuration of the reference deployment.
#
#   MODEL_DIR=/path/to/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef CARDS=0,1 PORT=8000 \
#       scripts/serve_qwen38_clef.sh
#
# Environment (all optional except MODEL_DIR):
#   MODEL_DIR      checkpoint directory (the Hugging Face release; Clef adapters in MODEL_DIR/clef)
#   CARDS          two GPU indices as shown by nvidia-smi                 (default 0,1)
#   HOST, PORT     listen address                                         (default 0.0.0.0, 8000)
#   SERVED_NAMES   space-separated model names for the OpenAI API         (default Qwen3.8-27B-ABLITERATED)
#   THINK          default enable_thinking for chat requests              (default false)
#   STATE_DIR      build products, W12 weight cache, torch.compile cache  (default ~/.cache/vllm-sm70)
#   CONFIG         lever file                                             (default configs/qwen38-27b-int8-clef.env)
#   PY             python of the virtualenv with this repo installed      (default: python3 on PATH)
#   CUDA_HOME      CUDA 12.x toolkit used to JIT-build the sm70 kernels    (default: from nvcc on PATH)
#   FA2SM70_CLEF=0 serve without Clef (the base model only)
#   FORCE=1        start even if the GPUs already hold memory
#   DRYRUN=1       print the resolved environment and command, start nothing
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
say() { printf '[serve] %s\n' "$*"; }
die() { printf '[serve] ERROR: %s\n' "$*" >&2; exit 1; }

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the checkpoint directory}"
[ -f "$MODEL_DIR/config.json" ] || die "no config.json in $MODEL_DIR"
CARDS="${CARDS:-0,1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_NAMES="${SERVED_NAMES:-Qwen3.8-27B-ABLITERATED}"
THINK="${THINK:-false}"
STATE_DIR="${STATE_DIR:-$HOME/.cache/vllm-sm70}"
CONFIG="${CONFIG:-$REPO/configs/qwen38-27b-int8-clef.env}"
PY="${PY:-python3}"
[ "$(echo "$CARDS" | tr ',' '\n' | grep -c .)" = "2" ] || die "CARDS must name exactly two GPUs (got '$CARDS')"

# ---------------------------------------------------------------- 1. production lever set
# Sourced with allexport: every lever reaches the API server, the engine core and both workers.
# Values set in the caller's environment win (e.g. FA2SM70_CLEF=0), exactly like the original launcher.
while IFS= read -r line; do
  case "$line" in ''|'#'*) continue ;; esac
  key="${line%%=*}"
  [ -n "${!key+x}" ] || export "$line"
done < "$CONFIG"

# ---------------------------------------------------------------- 2. toolchain for the JIT kernels
if [ -z "${CUDA_HOME:-}" ]; then
  nvcc_path="$(command -v nvcc || true)"
  [ -n "$nvcc_path" ] || die "nvcc not found: set CUDA_HOME to a CUDA 12.x toolkit"
  CUDA_HOME="$(cd "$(dirname "$nvcc_path")/.." && pwd)"
fi
export CUDA_HOME
[ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc in $CUDA_HOME/bin"
command -v ninja >/dev/null || "$PY" -c "import ninja" 2>/dev/null || die "ninja is required (pip install ninja)"
# The host compiler must be accepted by this nvcc for sm_70; probe instead of trusting the default.
probe_dir="$(mktemp -d)"; trap 'rm -rf "$probe_dir"' EXIT
printf '#include <cuda_runtime.h>\n__global__ void k(){}\nint main(){return 0;}\n' > "$probe_dir/p.cu"
cc_ok=""
for cand in ${CC:-} /usr/bin/gcc /usr/bin/gcc-14 /usr/bin/gcc-13 /usr/bin/gcc-12 /usr/bin/gcc-11; do
  [ -x "$cand" ] || continue
  if "$CUDA_HOME/bin/nvcc" -ccbin "$cand" -arch=sm_70 -o "$probe_dir/p" "$probe_dir/p.cu" >/dev/null 2>&1; then
    cc_ok="$cand"; break
  fi
done
[ -n "$cc_ok" ] || die "no host gcc accepted by $CUDA_HOME/bin/nvcc for sm_70"
export CC="$cc_ok" CXX="${cc_ok/gcc/g++}" CUDAHOSTCXX="${cc_ok/gcc/g++}"
export TORCH_CUDA_ARCH_LIST=7.0
# torch's JIT loader shells out to `ninja` and `nvcc` in EVERY worker: put the virtualenv's bin
# (pip's ninja lives there) and the toolkit on PATH even when PY is called without activation.
PY_BIN="$(dirname "$(command -v "$PY")")"
export PATH="$PY_BIN:$CUDA_HOME/bin:$PATH"

# ---------------------------------------------------------------- 3. paths
export FA2SM70_PATH="$REPO/sm70"
export PYTHONPATH="$REPO/sm70/integrations/vllm_sm70:$REPO/sm70${PYTHONPATH:+:$PYTHONPATH}"
export FA2SM70_BUILD_DIR="${FA2SM70_BUILD_DIR:-$STATE_DIR/build}"
export FA2SM70_W12_CACHE="${FA2SM70_W12_CACHE:-$STATE_DIR/w12/$(basename "$MODEL_DIR")}"
# torch.compile caches are keyed by vLLM on model/engine config and traced source files, but NOT on
# our levers -- and several levers change the structure of graph arguments. The fingerprint keeps a
# graph compiled under one lever set from being replayed under another (silent misuse otherwise).
fp="${FA2SM70_W8:-}|${FA2SM70_TM8:-}|${FA2SM70_TM8_ONLY:-}|${FA2SM70_TM8_NMIN:-}|${FA2SM70_W12:-}|${FA2SM70_DRAFT_TM8:-}|${FA2SM70_LMH:-}|${FA2SM70_FMLP:-}|${FA2SM70_GDN_I16:-}|${FA2SM70_MTP_W8:-}|${FA2SM70_MTP_W8_NSPLIT:-}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$STATE_DIR/vllm-cache-$(printf '%s' "$fp" | md5sum | cut -c1-8)}"
if [ "${FA2SM70_CLEF:-0}" = "1" ]; then
  export FA2SM70_CLEF_DIR="${FA2SM70_CLEF_DIR:-$MODEL_DIR/clef}"
  for f in clef_lora_r256_fp16.safetensors deabl_abl_fp16.safetensors joint_head_fp16.safetensors; do
    [ -f "$FA2SM70_CLEF_DIR/$f" ] || die "Clef enabled but $FA2SM70_CLEF_DIR/$f is missing (FA2SM70_CLEF=0 to serve without it)"
  done
  export FA2SM70_CLEF_OUT="${FA2SM70_CLEF_OUT:-/dev/shm/vllm-sm70-clef-$PORT}"
fi
mkdir -p "$FA2SM70_BUILD_DIR" "$FA2SM70_W12_CACHE" "$VLLM_CACHE_ROOT"

# ---------------------------------------------------------------- 4. devices, topology, runtime env
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$CARDS" FA2SM70_CVD="$CARDS"
c1="${CARDS%%,*}"; c2="${CARDS##*,}"
if [ -z "${FA2SM70_EXCH12:-}" ]; then
  # int12-compressed TP exchange pays off on a PCIe bridge and loses on NVLink (measured 0.79-0.81x).
  if nvidia-smi topo -m 2>/dev/null | awk -v a="GPU$c1" -v b="$((c2 + 2))" '$1==a{print $b}' | grep -q '^NV'; then
    export FA2SM70_EXCH12=0; say "GPU$c1-GPU$c2: NVLink -> fp16 all-reduce"
  else
    export FA2SM70_EXCH12=1; say "GPU$c1-GPU$c2: no NVLink -> int12-compressed all-reduce"
  fi
fi
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}" NCCL_CUMEM_ENABLE=0 NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "$MODEL_DIR"
  --served-model-name $SERVED_NAMES
  --trust-remote-code
  --quantization gptq
  --dtype float16
  --tensor-parallel-size 2
  --disable-custom-all-reduce
  --gpu-memory-utilization 0.96
  --kv-cache-memory-bytes "${FA2SM70_KV_BYTES}"
  --kv-cache-dtype "${FA2SM70_KVDT}"
  --enable-prefix-caching
  --max-model-len 262144
  --max-num-seqs "${FA2SM70_SEQS}"
  --max-num-batched-tokens "${FA2SM70_MAXBATCH}"
  --mm-processor-cache-gb 1
  --limit-mm-per-prompt '{"image":20,"video":2}'
  --chat-template-content-format openai
  --default-chat-template-kwargs "{\"enable_thinking\": $THINK}"
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1,2,4,8]}'
  --host "$HOST"
  --port "$PORT"
  --mamba-cache-mode all
  --speculative-config "{\"method\":\"mtp\",\"model\":\"$MODEL_DIR\",\"num_speculative_tokens\":3}"
)

# Host-RAM KV offload (second prefix-cache tier, sm70/integrations/vllm_sm70/fa2sm70_kv_offload.py).
# Off in production; FA2SM70_KV_OFFLOAD_GB is pinned host memory for the whole server, split between ranks.
if [ "${FA2SM70_KV_OFFLOAD:-0}" = "1" ]; then
  ARGS+=(--kv-transfer-config '{"kv_connector":"Fa2Sm70OffloadConnector","kv_connector_module_path":"fa2sm70_kv_offload","kv_role":"kv_both"}'
        --no-disable-hybrid-kv-cache-manager)
  say "KV offload to host RAM: ${FA2SM70_KV_OFFLOAD_GB:-16} GiB for the server"
fi

if [ "${DRYRUN:-0}" = "1" ]; then
  say "resolved environment:"
  env | grep -E '^(FA2SM70_|VLLM_|CUDA_|NCCL_|PYTORCH_|TORCH_|CC=|CXX=|PYTHONPATH=)' | sort | sed 's/^/    /'
  say "command: $PY ${ARGS[*]}"
  exit 0
fi

# ---------------------------------------------------------------- 5. preflight
for c in ${CARDS//,/ }; do
  used=$(nvidia-smi -i "$c" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
  [ "${used:-0}" -lt 500 ] || [ "${FORCE:-0}" = "1" ] || die "GPU $c already holds ${used} MiB (FORCE=1 to start anyway)"
done
if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":$PORT "; then die "port $PORT is in use"; fi

# ---------------------------------------------------------------- 6. build all kernels in ONE process
# Otherwise both TP workers build lazily at the same time and contend for the build lock. The first
# build in a clean STATE_DIR takes 15-30 minutes (CUTLASS prefill alone ~9 min); later starts take seconds.
say "building / loading sm70 kernels in $FA2SM70_BUILD_DIR"
"$PY" -u - <<'PYEOF'
import sys, time
import fa2_sm70._ext as e
names = ["decode_ext", "w8a16_ext", "gdn_ext", "prefill_ext", "tm8_ext", "w12a16_ext", "exch_ext",
         "conv_ext", "i4_ext"]
import os
if os.environ.get("FA2SM70_CLEF") == "1":
    names.append("clef_ext")
bad = 0
for n in names:
    t = time.time()
    try:
        getattr(e, n)()
        print(f"[serve]   ok {n} ({time.time() - t:.1f} s)", flush=True)
    except Exception as ex:  # noqa: BLE001
        bad += 1
        print(f"[serve]   FAILED {n}: {str(ex)[-1500:]}", flush=True)
sys.exit(1 if bad else 0)
PYEOF

say "starting vLLM on GPUs $CARDS, http://$HOST:$PORT (Clef: ${FA2SM70_CLEF:-0})"
exec "$PY" "${ARGS[@]}"
