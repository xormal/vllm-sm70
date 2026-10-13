#!/usr/bin/env bash
# healthy = warm-up answered once (sm70-warmup.sh) AND the server still answers /health now
[ -f /tmp/sm70-ready ] && curl -sf -m 5 -o /dev/null "http://127.0.0.1:${PORT:-8000}/health"
