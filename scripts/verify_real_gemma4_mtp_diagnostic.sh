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
export MLXVLM_ENABLE_GEMMA4_ASSISTANT_RUNTIME=1
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

TARGET_PLAN="$("$BIN" inspect-gemma4-mtp-target-plan --model "$MODEL" --draft-model "$DRAFT_MODEL")"
echo "$TARGET_PLAN" | grep -q '"targetModelType" : "gemma4"'
echo "$TARGET_PLAN" | grep -q '"draftModelType" : "gemma4_assistant"'
echo "$TARGET_PLAN" | grep -q '"isGemma4Target" : true'
echo "$TARGET_PLAN" | grep -q '"isGemma4AssistantDraft" : true'
echo "$TARGET_PLAN" | grep -q '"hiddenSizeMatches" : true'
echo "$TARGET_PLAN" | grep -q '"targetHiddenSize" : 2560'
echo "$TARGET_PLAN" | grep -q '"draftBackboneHiddenSize" : 2560'
echo "$TARGET_PLAN" | grep -q '"firstKVSharedLayerIndex" : 24'
echo "$TARGET_PLAN" | grep -q '"targetKVSharedLayerCount" : 18'
echo "$TARGET_PLAN" | grep -q '"producerLayerIndex" : 23'
echo "$TARGET_PLAN" | grep -q '"producerLayerIndex" : 22'
echo "$TARGET_PLAN" | grep -q 'target forward must export K\\\/V pairs from the latest non-shared producer layer'

LOAD="$("$BIN" load-gemma4-assistant-draft --model "$DRAFT_MODEL")"
echo "$LOAD" | grep -q '"assistantRuntimeCompiled" : true'
echo "$LOAD" | grep -q '"realMLXAPIImplementationCompiled" : true'
echo "$LOAD" | grep -q '"rawModelType" : "gemma4_assistant"'
echo "$LOAD" | grep -q '"loadedTensorCount" : 50'

SMOKE="$("$BIN" smoke-gemma4-assistant-draft --model "$DRAFT_MODEL" --draft-block-size 4)"
echo "$SMOKE" | grep -q '"assistantRuntimeCompiled" : true'
echo "$SMOKE" | grep -q '"realMLXAPIImplementationCompiled" : true'
echo "$SMOKE" | grep -q '"loadedTensorCount" : 50'
echo "$SMOKE" | grep -q '"outputDType" : "int32"'
echo "$SMOKE" | grep -A3 '"outputShape" : \[' | grep -q '1,'
echo "$SMOKE" | grep -A3 '"outputShape" : \[' | grep -q '3'

TARGET_TEXT_LOAD="$("$BIN" load-gemma4-mtp-target-text --model "$MODEL")"
echo "$TARGET_TEXT_LOAD" | grep -q '"targetTextRuntimeCompiled" : true'
echo "$TARGET_TEXT_LOAD" | grep -q '"loaded" : true'
echo "$TARGET_TEXT_LOAD" | grep -q '"quantized" : true'
echo "$TARGET_TEXT_LOAD" | grep -q '"hiddenSize" : 2560'
echo "$TARGET_TEXT_LOAD" | grep -q '"hiddenLayers" : 42'
echo "$TARGET_TEXT_LOAD" | grep -q '"kvSharedLayers" : 18'
echo "$TARGET_TEXT_LOAD" | grep -q '"firstKVSharedLayerIndex" : 24'

TARGET_TEXT_SMOKE="$("$BIN" smoke-gemma4-mtp-target-text --model "$MODEL" --tokens 2,106)"
echo "$TARGET_TEXT_SMOKE" | grep -q '"targetTextRuntimeCompiled" : true'
echo "$TARGET_TEXT_SMOKE" | grep -q '"passed" : true'
echo "$TARGET_TEXT_SMOKE" | grep -q '"full_attention"'
echo "$TARGET_TEXT_SMOKE" | grep -q '"sliding_attention"'
echo "$TARGET_TEXT_SMOKE" | grep -A5 '"hiddenShape" : \[' | grep -q '2560'

TARGET_ADAPTER="$("$BIN" inspect-gemma4-mtp-target-adapter)"
echo "$TARGET_ADAPTER" | grep -q '"targetAdapterCompiled" : true'
echo "$TARGET_ADAPTER" | grep -q '"supportsHiddenSlotSelection" : true'
echo "$TARGET_ADAPTER" | grep -q '"supportsSharedKVSnapshot" : true'
echo "$TARGET_ADAPTER" | grep -q '"supportsScalarCacheRollback" : true'
echo "$TARGET_ADAPTER" | grep -q '"supportsDraftBinding" : true'
echo "$TARGET_ADAPTER" | grep -q 'route requests through the Swift target text runtime'

TARGET_ADAPTER_SMOKE="$("$BIN" smoke-gemma4-mtp-target-adapter)"
echo "$TARGET_ADAPTER_SMOKE" | grep -q '"targetAdapterCompiled" : true'
echo "$TARGET_ADAPTER_SMOKE" | grep -q '"passed" : true'
echo "$TARGET_ADAPTER_SMOKE" | grep -q '"cacheOffsetBeforeRollback" : 6'
echo "$TARGET_ADAPTER_SMOKE" | grep -q '"cacheOffsetAfterRollback" : 4'
echo "$TARGET_ADAPTER_SMOKE" | grep -q '"trimmedCacheTokenCount" : 2'
echo "$TARGET_ADAPTER_SMOKE" | grep -q '"fullAttentionKVLength" : 4'
echo "$TARGET_ADAPTER_SMOKE" | grep -q '"slidingAttentionKVLength" : 4'
echo "$TARGET_ADAPTER_SMOKE" | grep -A4 '"hiddenShape" : \[' | grep -q '1,'
echo "$TARGET_ADAPTER_SMOKE" | grep -A4 '"hiddenShape" : \[' | grep -q '8'

ROUND_PLAN="$("$BIN" inspect-mtp-round-plan --draft-tokens 10,11,12 --target-tokens 10,99,98,97 --emitted 1 --max-tokens 8 --draft-block-size 4 --position 42 --shared-kv-length 46)"
echo "$ROUND_PLAN" | grep -q '"acceptedCount" : 1'
echo "$ROUND_PLAN" | grep -q '"newTokens" : \['
echo "$ROUND_PLAN" | grep -q '10,'
echo "$ROUND_PLAN" | grep -q '99'
echo "$ROUND_PLAN" | grep -q '"hiddenSlotIndex" : 1'
echo "$ROUND_PLAN" | grep -q '"positionAfterRound" : 44'
echo "$ROUND_PLAN" | grep -q '"rollbackRequired" : true'
echo "$ROUND_PLAN" | grep -q '"rejectedTokenCount" : 2'
echo "$ROUND_PLAN" | grep -q '"sharedKVValidLength" : 44'

SESSION_PLAN="$("$BIN" inspect-mtp-session --first-bonus 9 --draft-round 10,11,12 --target-round 10,99,98,97 --draft-round 100,101,102 --target-round 100,101,102,103 --max-tokens 6 --draft-block-size 4 --position 40)"
echo "$SESSION_PLAN" | grep -q '"inputBonusToken" : 9'
echo "$SESSION_PLAN" | grep -q '"inputBonusToken" : 99'
echo "$SESSION_PLAN" | grep -q '"roundIndex" : 2'
echo "$SESSION_PLAN" | grep -q '"bonusToken" : 102'
echo "$SESSION_PLAN" | grep -q '"emittedTokenCount" : 6'
echo "$SESSION_PLAN" | grep -q '"finished" : true'
echo "$SESSION_PLAN" | grep -q '"sharedKVSequenceLength" : 46'

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
echo "$RESPONSE" | grep -q 'Gemma4 MTP assistant architecture'
echo "$RESPONSE" | grep -q 'Native Swift Gemma4Assistant loading now succeeds'
echo "$RESPONSE" | grep -q 'MTP accept.*rollback session loop has been ported into Swift Core'
echo "$RESPONSE" | grep -q 'Swift target adapter contract is compiled'
echo "$RESPONSE" | grep -q 'Swift Gemma4 target text runtime now exports pre-norm hidden states plus shared K\\\/V'
echo "$RESPONSE" | grep -q 'still needs to route --draft-kind mtp requests'
