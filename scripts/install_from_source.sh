#!/usr/bin/env bash
# Build vllm-sm70 from source into a fresh virtualenv: Python 3.12, CUDA 12.8, NVIDIA Volta (sm_70).
#
#   scripts/install_from_source.sh                 # creates .venv in the repository
#   VENV=/opt/vllm-sm70 MAX_JOBS=12 scripts/install_from_source.sh
#
# Environment:
#   VENV       virtualenv to create or reuse              (default <repo>/.venv)
#   PYTHON     interpreter used to create it              (default python3.12)
#   CUDA_HOME  CUDA 12.8 toolkit                          (default: from nvcc on PATH, else /usr/local/cuda-12.8)
#   MAX_JOBS   parallel compile jobs (~3 GB RAM each)     (default 8)
#
# The vLLM C++/CUDA build for sm_70 takes about an hour with MAX_JOBS=8; flash-attention-v100 a few
# minutes. The fa2_sm70 kernels under sm70/ are NOT built here: they are JIT-compiled by
# scripts/serve_qwen38_clef.sh on first start (cached in STATE_DIR).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
PYTHON="${PYTHON:-python3.12}"
MAX_JOBS="${MAX_JOBS:-8}"
say() { printf '[install] %s\n' "$*"; }
die() { printf '[install] ERROR: %s\n' "$*" >&2; exit 1; }

if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null; then CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  else CUDA_HOME=/usr/local/cuda-12.8; fi
fi
[ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc in $CUDA_HOME/bin (set CUDA_HOME to a CUDA 12.8 toolkit)"
"$CUDA_HOME/bin/nvcc" --version | grep -q "release 12\." || die "CUDA 12.x toolkit required"
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS NVCC_THREADS=1

# A host gcc that this nvcc accepts for sm_70 (conda compilers on PATH are a common trap).
probe_dir="$(mktemp -d)"; trap 'rm -rf "$probe_dir"' EXIT
printf '#include <cuda_runtime.h>\n__global__ void k(){}\nint main(){return 0;}\n' > "$probe_dir/p.cu"
cc_ok=""
for cand in ${CC:-} /usr/bin/gcc /usr/bin/gcc-13 /usr/bin/gcc-12 /usr/bin/gcc-11; do
  [ -x "$cand" ] || continue
  if "$CUDA_HOME/bin/nvcc" -ccbin "$cand" -arch=sm_70 -o "$probe_dir/p" "$probe_dir/p.cu" >/dev/null 2>&1; then
    cc_ok="$cand"; break
  fi
done
[ -n "$cc_ok" ] || die "no host gcc accepted by nvcc for sm_70"
export CC="$cc_ok" CXX="${cc_ok/gcc/g++}" CUDAHOSTCXX="${cc_ok/gcc/g++}"
say "CUDA_HOME=$CUDA_HOME CC=$CC MAX_JOBS=$MAX_JOBS"

[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
PY="$VENV/bin/python"
# The build must use the venv's cmake (>= 3.26) and ninja, not whatever the system has.
export PATH="$VENV/bin:$PATH"
"$PY" -m pip install --upgrade pip
say "installing the pinned environment (requirements-sm70.txt)"
"$PY" -m pip install -r "$REPO/requirements-sm70.txt"

say "building flash-attention-v100"
"$PY" -m pip install --no-build-isolation --no-deps "$REPO/flash-attention-v100"

say "building vLLM for sm_70 (editable)"
cd "$REPO"
rm -rf build vllm.egg-info .deps/*-build .deps/*-subbuild
export VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.0.3+sm70}"
"$PY" -m pip install --no-build-isolation --no-deps -e .

"$PY" - <<'EOF'
import torch, vllm, flash_attn_v100_cuda  # noqa: F401
print(f"[install] ok: vllm {vllm.__version__}, torch {torch.__version__}, "
      f"CUDA available: {torch.cuda.is_available()}")
EOF
say "done. Serve with: PY=$PY MODEL_DIR=... scripts/serve_qwen38_clef.sh"
