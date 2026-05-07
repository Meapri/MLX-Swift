#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SWIFT_BUILD_JOBS="${SWIFT_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

MODEL="${MLXVLM_REAL_SMOKE_MODEL:-/Users/naen/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/cc3b666c01c20395e0dcebd53854504c7d9821f9}"

if [[ ! -d "$MODEL" ]]; then
  echo "real Gemma4 smoke skipped: model directory not found: $MODEL"
  exit 0
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "real Gemma4 smoke skipped: rsync is required for the identity-safe build copy"
  exit 0
fi

TMP_ROOT="${TMPDIR:-/tmp}/mlx-vlm-swift-real-gemma4-smoke-$$"
PORT="${MLXVLM_REAL_SMOKE_PORT:-$((12000 + ($$ % 20000)))}"
SCRATCH_DIR="${MLXVLM_REAL_SMOKE_SCRATCH:-${TMPDIR:-/tmp}/mlx-vlm-swift-real-gemma4-smoke-build}"
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

"$BIN" serve --model "$MODEL" --port "$PORT" --max-tokens 8 --temperature 0.0 > "$SERVER_LOG" 2>&1 &
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
echo "$HEALTH" | grep -q '"embedding_backend_ready":false'

TOKENIZE_RESPONSE="$(
  curl -fsS -m 30 "http://127.0.0.1:$PORT/v1/tokenize" \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","text":"Hello","add_special_tokens":true}'
)"
echo "$TOKENIZE_RESPONSE" | grep -q '"supported":true'
echo "$TOKENIZE_RESPONSE" | grep -q '"backend":"mlx-swift-vlm"'
echo "$TOKENIZE_RESPONSE" | grep -q '"token_ids":\['

CHAT_RESPONSE="$(
  curl -fsS -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","messages":[{"role":"user","content":"Say hi in one short sentence."}],"max_tokens":8,"temperature":0}'
)"
echo "$CHAT_RESPONSE" | grep -q '"content":"Hello\."'
echo "$CHAT_RESPONSE" | grep -q '"prompt_tokens":16'
echo "$CHAT_RESPONSE" | grep -q '"completion_tokens":2'

JSON_MODE_RESPONSE="$(
  curl -fsS -m 90 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","messages":[{"role":"user","content":"Say hello."}],"response_format":{"type":"json_object"},"max_tokens":4,"temperature":0}'
)"
echo "$JSON_MODE_RESPONSE" | grep -q '"content":"{' || echo "$JSON_MODE_RESPONSE" | grep -q '"content":"\['

TOOL_RESPONSE="$(
  curl -fsS -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","messages":[{"role":"user","content":"Say hi."}],"tools":[{"type":"function","function":{"name":"lookup","description":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],"max_tokens":8,"temperature":0}'
)"
echo "$TOOL_RESPONSE" | grep -q '"choices"'

TOOL_CALL_RESPONSE="$(
  curl -fsS -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","messages":[{"role":"system","content":"You must answer only by calling the lookup function. Do not write ordinary text."},{"role":"user","content":"Use the lookup function with query exactly swift mlx."}],"tools":[{"type":"function","function":{"name":"lookup","description":"lookup information","parameters":{"type":"object","properties":{"query":{"type":"string","description":"search query"}},"required":["query"]}}}],"tool_choice":{"type":"function","function":{"name":"lookup"}},"max_tokens":64,"temperature":0}'
)"
echo "$TOOL_CALL_RESPONSE" | grep -q '"finish_reason":"tool_calls"'
echo "$TOOL_CALL_RESPONSE" | grep -q '"tool_calls"'
echo "$TOOL_CALL_RESPONSE" | grep -q '"name":"lookup"'
echo "$TOOL_CALL_RESPONSE" | grep -q '"arguments":"{\\"query\\":\\"swift mlx\\"}"'

RESPONSES_TOOL_CALL_RESPONSE="$(
  curl -fsS -m 120 "http://127.0.0.1:$PORT/v1/responses" \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","input":[{"role":"system","content":"You must answer only by calling the lookup function. Do not write ordinary text."},{"role":"user","content":"Use the lookup function with query exactly swift mlx."}],"tools":[{"type":"function","function":{"name":"lookup","description":"lookup information","parameters":{"type":"object","properties":{"query":{"type":"string","description":"search query"}},"required":["query"]}}}],"tool_choice":{"type":"function","function":{"name":"lookup"}},"max_output_tokens":64,"temperature":0,"top_p":1,"truncation":"auto","user":"gemma-user","metadata":{"trace":"gemma-smoke"}}'
)"
echo "$RESPONSES_TOOL_CALL_RESPONSE" | grep -q '"object":"response"'
echo "$RESPONSES_TOOL_CALL_RESPONSE" | grep -q '"max_output_tokens":64'
echo "$RESPONSES_TOOL_CALL_RESPONSE" | grep -q '"temperature":0'
echo "$RESPONSES_TOOL_CALL_RESPONSE" | grep -q '"top_p":1'
echo "$RESPONSES_TOOL_CALL_RESPONSE" | grep -q '"truncation":"auto"'
echo "$RESPONSES_TOOL_CALL_RESPONSE" | grep -q '"user":"gemma-user"'
echo "$RESPONSES_TOOL_CALL_RESPONSE" | grep -q '"metadata":{"trace":"gemma-smoke"}'
echo "$RESPONSES_TOOL_CALL_RESPONSE" | grep -q '"type":"function_call"'
echo "$RESPONSES_TOOL_CALL_RESPONSE" | grep -q '"call_id"'
echo "$RESPONSES_TOOL_CALL_RESPONSE" | grep -q '"name":"lookup"'
echo "$RESPONSES_TOOL_CALL_RESPONSE" | grep -q '"arguments":"{\\"query\\":\\"swift mlx\\"}"'

RESPONSES_TOOL_STREAM="$(mktemp)"
curl -sS -m 120 -i "http://127.0.0.1:$PORT/v1/responses" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","input":[{"role":"system","content":"You must answer only by calling the lookup function. Do not write ordinary text."},{"role":"user","content":"Use the lookup function with query exactly swift mlx."}],"tools":[{"type":"function","function":{"name":"lookup","description":"lookup information","parameters":{"type":"object","properties":{"query":{"type":"string","description":"search query"}},"required":["query"]}}}],"tool_choice":{"type":"function","function":{"name":"lookup"}},"max_output_tokens":64,"temperature":0,"top_p":1,"truncation":"auto","user":"gemma-user","metadata":{"trace":"gemma-smoke"},"stream":true}' > "$RESPONSES_TOOL_STREAM"
grep -qi '^Content-Type: text/event-stream' "$RESPONSES_TOOL_STREAM"
if grep -qi '^Content-Length:' "$RESPONSES_TOOL_STREAM"; then
  echo "OpenAI Responses tool stream unexpectedly included Content-Length"
  cat "$RESPONSES_TOOL_STREAM"
  exit 1
fi
grep -q 'event: response.function_call_arguments.done' "$RESPONSES_TOOL_STREAM"
grep -q '"max_output_tokens":64' "$RESPONSES_TOOL_STREAM"
grep -q '"temperature":0' "$RESPONSES_TOOL_STREAM"
grep -q '"top_p":1' "$RESPONSES_TOOL_STREAM"
grep -q '"truncation":"auto"' "$RESPONSES_TOOL_STREAM"
grep -q '"user":"gemma-user"' "$RESPONSES_TOOL_STREAM"
grep -q '"metadata":{"trace":"gemma-smoke"}' "$RESPONSES_TOOL_STREAM"
grep -q '"type":"function_call"' "$RESPONSES_TOOL_STREAM"
grep -q '"call_id"' "$RESPONSES_TOOL_STREAM"
grep -q '"name":"lookup"' "$RESPONSES_TOOL_STREAM"
grep -q '"arguments":"{\\"query\\":\\"swift mlx\\"}"' "$RESPONSES_TOOL_STREAM"
grep -q 'data: \[DONE\]' "$RESPONSES_TOOL_STREAM"

GREEN_PNG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
OLLAMA_IMAGE_RESPONSE="$(
  curl -fsS -m 90 "http://127.0.0.1:$PORT/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"gemma4\",\"prompt\":\"What color is this image? Answer one word.\",\"images\":[\"$GREEN_PNG\"],\"max_tokens\":4,\"temperature\":0}"
)"
echo "$OLLAMA_IMAGE_RESPONSE" | grep -qi '"response":"green"'

OPENAI_IMAGE_RESPONSE="$(
  curl -fsS -m 90 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"gemma4\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What color is this image? Answer one word.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$GREEN_PNG\"}}]}],\"max_tokens\":4,\"temperature\":0}"
)"
echo "$OPENAI_IMAGE_RESPONSE" | grep -qi '"content":"green"'

OPENAI_STREAM="$(mktemp)"
curl -sS -m 60 -i "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"Say hi in one short sentence."}],"max_tokens":8,"temperature":0,"stream":true}' > "$OPENAI_STREAM"
grep -qi '^Content-Type: text/event-stream' "$OPENAI_STREAM"
if grep -qi '^Content-Length:' "$OPENAI_STREAM"; then
  echo "OpenAI stream unexpectedly included Content-Length"
  cat "$OPENAI_STREAM"
  exit 1
fi
grep -q 'data: ' "$OPENAI_STREAM"
grep -q 'data: \[DONE\]' "$OPENAI_STREAM"
grep -q '"prompt_tokens":16' "$OPENAI_STREAM"
grep -q '"completion_tokens":2' "$OPENAI_STREAM"

OLLAMA_STREAM="$(mktemp)"
curl -sS -m 60 -i "http://127.0.0.1:$PORT/api/chat" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"Say hi in one short sentence."}],"max_tokens":8,"temperature":0,"stream":true}' > "$OLLAMA_STREAM"
grep -qi '^Content-Type: application/x-ndjson' "$OLLAMA_STREAM"
if grep -qi '^Content-Length:' "$OLLAMA_STREAM"; then
  echo "Ollama stream unexpectedly included Content-Length"
  cat "$OLLAMA_STREAM"
  exit 1
fi
grep -q '"prompt_eval_count":16' "$OLLAMA_STREAM"
grep -q '"eval_count":2' "$OLLAMA_STREAM"

EMBEDDING_RESPONSE="$(
  curl -sS -m 20 -i "http://127.0.0.1:$PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","input":"hello"}'
)"
echo "$EMBEDDING_RESPONSE" | grep -q 'HTTP/1.1 501 Not Implemented'
echo "$EMBEDDING_RESPONSE" | grep -q '"activeBackend":"mlx-swift-vlm"'
echo "$EMBEDDING_RESPONSE" | grep -q '"inputCount":1'
echo "$EMBEDDING_RESPONSE" | grep -q '"fallbackPolicy":"diagnostic-501-no-generated-embedding"'
echo "$EMBEDDING_RESPONSE" | grep -q '"unavailableReason"'

echo "real Gemma4 smoke passed"
