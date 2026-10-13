#!/usr/bin/env bash
# Marks the container ready (/tmp/sm70-ready) once the server has ANSWERED: /health is 200 before the
# first real request has finished compiling and capturing, and a request in that window can take
# minutes. The healthcheck waits for the marker.
P="${PORT:-8000}"
U="http://127.0.0.1:$P"
AUTH=()
[ -z "${VLLM_API_KEY:-}" ] || AUTH=(-H "Authorization: Bearer $VLLM_API_KEY")
NAME="${SERVED_NAMES:-Qwen3.8-27B-ABLITERATED}"; NAME="${NAME%% *}"
until curl -sf -m 5 -o /dev/null "$U/health"; do sleep 10; done
ok=0
for _ in 1 2 3 4 5; do
  if curl -sf -m 1200 -o /dev/null "${AUTH[@]}" -H 'Content-Type: application/json' "$U/v1/chat/completions" \
       -d "{\"model\":\"$NAME\",\"max_tokens\":4,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}"; then
    ok=1; break
  fi
  sleep 30
done
if [ "$ok" = 1 ] && [ "${FA2SM70_CLEF:-1}" != "0" ]; then
  curl -sf -m 600 -o /dev/null "${AUTH[@]}" -H 'Content-Type: application/json' "$U/v1/systemone" \
    -d '{"model":"clef","state":"warm-up","questions":{"q":{"type":"noul","instructions":"Is this a warm-up?"}}}' || ok=0
fi
what="chat"; [ "${FA2SM70_CLEF:-1}" = "0" ] || what="chat and Clef"
if [ "$ok" = 1 ]; then touch /tmp/sm70-ready; echo "[warmup] ready: $what answered"
else echo "[warmup] the server did not answer warm-up requests; container stays unhealthy" >&2; fi
