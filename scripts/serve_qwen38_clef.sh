#!/usr/bin/env bash
# Serve Qwen3.8-27B-ABLITERATED GPTQ-Int8 (+ Clef on /v1/systemone) on 2, 4 or 8 NVIDIA V100, tensor
# parallel over all given cards. Two 32 GB cards run the exact configuration of the reference deployment;
# other card counts and 16 GB cards use the memory profiles in configs/profiles/ (experimental, see its
# README). To split more cards into several replicas use scripts/plan_gpus.py.
#
#   MODEL_DIR=/path/to/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef CARDS=0,1 PORT=8000 \
#       scripts/serve_qwen38_clef.sh
#
# Environment (all optional except MODEL_DIR):
#   MODEL_DIR      checkpoint directory (the Hugging Face release; Clef adapters in MODEL_DIR/clef)
#   CARDS          2, 4 or 8 GPUs: nvidia-smi indices or GPU-UUIDs        (default 0,1; in the container: all visible)
#   PROFILE        memory profile in configs/profiles/                    (default tp<cards>-<16g|32g> of the smallest card)
#   MAX_MODEL_LEN  context limit                                          (default from the profile)
#   ALLOWED_MEDIA_DOMAINS  comma-separated hosts images/videos may be fetched from (default: any)
#   VLLM_API_KEY   require this bearer token on the API                   (default: no authentication)
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
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_NAMES="${SERVED_NAMES:-Qwen3.8-27B-ABLITERATED}"
THINK="${THINK:-false}"
STATE_DIR="${STATE_DIR:-$HOME/.cache/vllm-sm70}"
CONFIG="${CONFIG:-$REPO/configs/qwen38-27b-int8-clef.env}"
PY="${PY:-python3}"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (no NVIDIA driver / GPUs not passed to the container)"

# ---------------------------------------------------------------- 0. cards, tensor parallel, profile
if [ -z "${CARDS:-}" ]; then
  if [ "${SM70_IN_CONTAINER:-0}" = "1" ]; then
    CARDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    CARDS=0,1
  fi
fi
CARD_IDX=()
for c in ${CARDS//,/ }; do
  i="$(nvidia-smi -i "$c" --query-gpu=index --format=csv,noheader 2>/dev/null)" || die "no GPU '$c' (CARDS=$CARDS)"
  CARD_IDX+=("$i")
done
TP=${#CARD_IDX[@]}
case "$TP" in 2|4|8) ;; *) die "CARDS must name 2, 4 or 8 GPUs (got '$CARDS'); scripts/plan_gpus.py splits more" ;; esac
[ "$(printf '%s\n' "${CARD_IDX[@]}" | sort -u | wc -l)" = "$TP" ] || die "CARDS names a GPU twice ($CARDS)"
MEM_MIN="$(for i in "${CARD_IDX[@]}"; do nvidia-smi -i "$i" --query-gpu=memory.total --format=csv,noheader,nounits; done | sort -n | head -1)"
if [ "$MEM_MIN" -ge 30000 ]; then CARD_CLASS=32g
elif [ "$MEM_MIN" -ge 15000 ]; then CARD_CLASS=16g
else die "a ${MEM_MIN} MiB card is too small (V100 16 GB or 32 GB required)"; fi
[ "$CARD_CLASS" = 32g ] || [ "$TP" -ge 4 ] || die "16 GB cards need at least 4 GPUs (the weight shard of TP=2 does not fit)"
PROFILE="${PROFILE:-tp${TP}-${CARD_CLASS}}"
PROFILE_FILE="$REPO/configs/profiles/$PROFILE.env"
[ -f "$PROFILE_FILE" ] || die "no memory profile $PROFILE_FILE"

# ---------------------------------------------------------------- 1. production lever set + profile
# Sourced with allexport: every lever reaches the API server, the engine core and all workers.
# Precedence: the caller's environment (e.g. FA2SM70_CLEF=0), then the profile, then the base lever file.
load_levers() {
  while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    key="${line%%=*}"
    [ -n "${!key+x}" ] || export "$line"
  done < "$1"
}
load_levers "$PROFILE_FILE"
load_levers "$CONFIG"
case "$PROFILE" in tp2-32g) ;; *) say "profile $PROFILE is EXPERIMENTAL (configs/profiles/README.md)" ;; esac

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
c1="${CARD_IDX[0]}"; c2="${CARD_IDX[1]}"
if [ "$TP" -ge 4 ]; then
  # The int12 exchange is an all-gather: (N-1)*0.76 of the data per rank against 2(N-1)/N for a ring
  # all-reduce, so it only wins for a pair. NCCL routes over NVLink where the group has it.
  [ "${FA2SM70_EXCH12:-0}" = "0" ] || say "FA2SM70_EXCH12=1 ignored: it loses to NCCL beyond 2 ranks"
  export FA2SM70_EXCH12=0; say "TP=$TP on GPUs $CARDS -> fp16 NCCL all-reduce"
elif [ -z "${FA2SM70_EXCH12:-}" ]; then
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
  --tensor-parallel-size "$TP"
  --disable-custom-all-reduce
  --gpu-memory-utilization 0.96
  --kv-cache-memory-bytes "${FA2SM70_KV_BYTES}"
  --kv-cache-dtype "${FA2SM70_KVDT}"
  --enable-prefix-caching
  --max-model-len "${MAX_MODEL_LEN:-262144}"
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

# Images and videos are fetched server-side (chat and /v1/systemone): restrict the hosts when the API is
# reachable by untrusted clients. VLLM_API_KEY, if set, is read by vLLM itself.
[ -z "${ALLOWED_MEDIA_DOMAINS:-}" ] || ARGS+=(--allowed-media-domains ${ALLOWED_MEDIA_DOMAINS//,/ })

if [ "${DRYRUN:-0}" = "1" ]; then
  say "resolved environment:"
  say "cards $CARDS, TP=$TP, profile $PROFILE, max context ${MAX_MODEL_LEN:-262144}"
  env | grep -E '^(FA2SM70_|VLLM_|CUDA_|NCCL_|PYTORCH_|TORCH_|CC=|CXX=|PYTHONPATH=)' | grep -v '^VLLM_API_KEY=' | sort | sed 's/^/    /'
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
# the same list as the image build (sm70/fa2_sm70/prebuild.py); a prebuilt directory just loads
"$PY" -u -m fa2_sm70.prebuild || die "sm70 kernel build failed (see above)"

say "starting vLLM on GPUs $CARDS (TP=$TP, profile $PROFILE), http://$HOST:$PORT (Clef: ${FA2SM70_CLEF:-0})"
exec "$PY" "${ARGS[@]}"
