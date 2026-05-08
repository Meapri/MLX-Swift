#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SWIFT_BUILD_JOBS="${SWIFT_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

MODEL="${MLXVLM_REAL_GEMMA4_MODEL:-/Users/naen/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/cc3b666c01c20395e0dcebd53854504c7d9821f9}"
DRAFT_MODEL="${MLXVLM_REAL_GEMMA4_MTP_DRAFT_MODEL:-/Users/naen/.cache/huggingface/hub/models--mlx-community--gemma-4-E4B-it-assistant-bf16/snapshots/844e008e06ef5562bdb89428d851d0634d119dcd}"

if [[ ! -d "$MODEL" ]]; then
  echo "real Gemma4 MTP diagnostic skipped: model directory not found: $MODEL"
  exit 0
fi

if [[ ! -d "$DRAFT_MODEL" ]]; then
  echo "real Gemma4 MTP diagnostic skipped: draft model directory not found: $DRAFT_MODEL"
  exit 0
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "real Gemma4 MTP diagnostic skipped: rsync is required for the identity-safe build copy"
  exit 0
fi

TMP_ROOT="${TMPDIR:-/tmp}/mlx-vlm-swift-real-gemma4-mtp-diagnostic-$$"
PORT="${MLXVLM_REAL_GEMMA4_MTP_PORT:-$((15000 + ($$ % 20000)))}"
SCRATCH_DIR="${MLXVLM_REAL_GEMMA4_MTP_SCRATCH:-${TMPDIR:-/tmp}/mlx-vlm-swift-real-gemma4-mtp-diagnostic-build}"
BUILD_DIR="$TMP_ROOT/source"
SERVER_LOG="$TMP_ROOT/server.log"
mkdir -p "$BUILD_DIR"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

rsync -a --exclude .build --exclude .git ./ "$BUILD_DIR/"
cd "$BUILD_DIR"

export MLXVLM_ENABLE_MLX_BACKEND=1
export MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1
export MLXVLM_ENABLE_REAL_MLX_API=1
export CLANG_MODULE_CACHE_PATH="$BUILD_DIR/.build/clang-module-cache"

swift build --disable-sandbox --jobs "$SWIFT_BUILD_JOBS" --scratch-path "$SCRATCH_DIR" --product mlx-vlm-swift
BIN="$SCRATCH_DIR/arm64-apple-macosx/debug/mlx-vlm-swift"

PLAN="$("$BIN" inspect-gemma4-assistant-draft-plan --model "$DRAFT_MODEL")"
echo "$PLAN" | grep -q '"modelType" : "gemma4_assistant"'
echo "$PLAN" | grep -q '"requiredDraftKind" : "mtp"'
echo "$PLAN" | grep -q '"draftLayerCount" : 4'
echo "$PLAN" | grep -q '"backboneHiddenSize" : 2560'
echo "$PLAN" | grep -q '"useOrderedEmbeddings" : true'
echo "$PLAN" | grep -q '"missingTensorKeys" : \['
echo "$PLAN" | grep -q '"unexpectedCriticalTensorKeys" : \['
echo "$PLAN" | grep -q '"nativeSwiftMTPReady" : false'
echo "$PLAN" | grep -q 'target Gemma4 prefill must return the last pre-norm hidden state'

"$BIN" serve \
  --model "$MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --draft-kind mtp \
  --draft-block-size 6 \
  --port "$PORT" \
  --max-tokens 8 \
  --temperature 0.0 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in {1..120}; do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited before readiness"
    cat "$SERVER_LOG"
    exit 1
  fi
  sleep 1
done

HEALTH="$(curl -fsS -m 5 "http://127.0.0.1:$PORT/health")"
echo "$HEALTH" | grep -q '"backend_ready":true'
echo "$HEALTH" | grep -q '"backend":"mlx-swift-vlm"'

RESPONSE="$(
  curl -sS -m 120 -i "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","messages":[{"role":"user","content":"Say hi."}],"max_tokens":8,"temperature":0}'
)"

echo "$RESPONSE" | grep -q 'HTTP/1.1 500 Internal Server Error'
echo "$RESPONSE" | grep -q 'gemma4_assistant'
echo "$RESPONSE" | grep -q 'Gemma4AssistantForCausalLM'
echo "$RESPONSE" | grep -q 'native Swift Gemma4Assistant draft-model support is required'
