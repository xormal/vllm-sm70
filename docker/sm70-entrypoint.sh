#!/usr/bin/env bash
# vllm-sm70 container entrypoint (PID 1 is tini).
#   serve [default]  start the server on every GPU given to the container (or CARDS=...); see
#                    scripts/serve_qwen38_clef.sh for the environment
#   plan [args]      plan replicas over the GPUs visible to this container (scripts/plan_gpus.py)
#   prebuild [names] (re)build the sm70 kernels
#   gates [args]     validation client (sm70/integrations/vllm_sm70/clef/proverka_stenda.py)
#   anything else    exec it (e.g. bash)
set -euo pipefail
R=/opt/vllm-sm70
cmd="${1:-serve}"
[ $# -gt 0 ] && shift
case "$cmd" in
  serve)
    rm -f /tmp/sm70-ready
    "$R/docker/sm70-warmup.sh" &
    exec "$R/scripts/serve_qwen38_clef.sh" "$@" ;;
  plan) exec "$PY" -I "$R/scripts/plan_gpus.py" "$@" ;;     # standard library only: skip sitecustomize
  prebuild) exec "$PY" -m fa2_sm70.prebuild "$@" ;;
  gates) exec "$PY" "$R/sm70/integrations/vllm_sm70/clef/proverka_stenda.py" "$@" ;;
  *) exec "$cmd" "$@" ;;
esac
