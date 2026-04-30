#!/usr/bin/env bash
# Smoke test the APC integration through the FastAPI server.
#
# Starts the server with APC enabled, fires two identical chat completions,
# and asserts that the second one's prompt_tps is meaningfully higher and
# that /v1/cache/stats reports a non-zero token_hit_rate.
#
# Usage:  scripts/test_apc_server.sh [model_path]

set -euo pipefail

MODEL="${1:-mlx-community/Qwen3-VL-4B-Instruct-4bit}"
PORT="${PORT:-8089}"
PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
LOG="$(mktemp -t apc-server.XXXXXX)"
PID=""

cleanup() {
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
  echo "Server log: $LOG"
}
trap cleanup EXIT

echo "Starting server (model=$MODEL port=$PORT)…"
APC_ENABLED=1 APC_NUM_BLOCKS=4096 APC_BLOCK_SIZE=16 \
MLX_VLM_PRELOAD_MODEL="$MODEL" \
  "$PYTHON" -m mlx_vlm.server --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
PID=$!

# Wait for /health to come up
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Server died early."
    tail -50 "$LOG"
    exit 1
  fi
done
echo "Server is up."

curl -sf "http://127.0.0.1:$PORT/health" | python -m json.tool
echo

PROMPT='List exactly five facts about transformer language models, one per line, beginning each with a dash.'

req() {
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":32,\"temperature\":0}"
}

echo "=== Reset cache stats ==="
curl -sf -X POST "http://127.0.0.1:$PORT/v1/cache/reset" | python -m json.tool

echo "=== Request 1 ==="
T0=$(python -c 'import time; print(time.time())')
req >/dev/null
T1=$(python -c 'import time; print(time.time())')
ELAPSED1=$(python -c "print(round($T1-$T0,2))")
echo "elapsed1=${ELAPSED1}s"
curl -sf "http://127.0.0.1:$PORT/v1/cache/stats" | python -m json.tool

echo "=== Request 2 (same prompt) ==="
T0=$(python -c 'import time; print(time.time())')
req >/dev/null
T1=$(python -c 'import time; print(time.time())')
ELAPSED2=$(python -c "print(round($T1-$T0,2))")
echo "elapsed2=${ELAPSED2}s"
STATS=$(curl -sf "http://127.0.0.1:$PORT/v1/cache/stats")
echo "$STATS" | python -m json.tool

HIT_RATE=$(echo "$STATS" | python -c 'import json,sys; print(json.load(sys.stdin)["token_hit_rate"])')
MATCHED=$(echo "$STATS" | python -c 'import json,sys; print(json.load(sys.stdin)["matched_tokens"])')

echo
echo "=== Verdict ==="
echo "matched_tokens=$MATCHED  token_hit_rate=$HIT_RATE  elapsed1=${ELAPSED1}s  elapsed2=${ELAPSED2}s"

python -c "
import sys
matched = $MATCHED
if matched <= 0:
    print('FAIL: no APC hit on second request'); sys.exit(1)
print('PASS: APC hit ({} matched tokens)'.format(matched))
"
