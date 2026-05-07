#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SWIFT_BUILD_JOBS="${SWIFT_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

MODEL="${MLXVLM_REAL_QWEN_VL_MODEL:-${MLXVLM_REAL_QWEN25VL_MODEL:-mlx-community/Qwen2.5-VL-3B-Instruct-3bit}}"
ALLOW_REMOTE="${MLXVLM_REAL_QWEN_VL_ALLOW_REMOTE:-${MLXVLM_REAL_QWEN25VL_ALLOW_REMOTE:-0}}"
DOWNLOAD_DIR="${MLXVLM_REAL_QWEN_VL_DOWNLOAD_DIR:-${MLXVLM_REAL_QWEN25VL_DOWNLOAD_DIR:-}}"
REQUEST_MODEL="${MLXVLM_REAL_QWEN_VL_REQUEST_MODEL:-qwen25vl}"
SMOKE_LABEL="${MLXVLM_REAL_QWEN_VL_LABEL:-Qwen VL}"

download_hf_model() {
  local model_id="$1"
  local target_root="${DOWNLOAD_DIR:-$HOME/Models/${model_id//\//-}}"

  if ! command -v curl >/dev/null 2>&1; then
    echo "real $SMOKE_LABEL video smoke skipped: curl is required to download $model_id" >&2
    exit 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "real $SMOKE_LABEL video smoke skipped: python3 is required to read the Hugging Face file list for $model_id" >&2
    exit 0
  fi

  mkdir -p "$target_root"
  echo "downloading $model_id into $target_root" >&2
  curl -fsS "https://huggingface.co/api/models/$model_id" |
    python3 -c 'import json, sys
data = json.load(sys.stdin)
for sibling in data.get("siblings", []):
    name = sibling.get("rfilename")
    if name and not name.endswith("/"):
        print(name)
' |
    while IFS= read -r file; do
      local destination="$target_root/$file"
      local url="https://huggingface.co/$model_id/resolve/main/$file"
      mkdir -p "$(dirname "$destination")"
      local expected_size
      expected_size="$(
        curl -fsSI -L "$url" |
          awk 'BEGIN { IGNORECASE = 1 } /^Content-Length:/ { gsub("\r", "", $2); size = $2 } END { print size }'
      )"
      local local_size=""
      if [[ -f "$destination" ]]; then
        local_size="$(stat -f%z "$destination" 2>/dev/null || stat -c%s "$destination" 2>/dev/null || true)"
      fi
      if [[ -n "$expected_size" && -n "$local_size" && "$local_size" == "$expected_size" ]]; then
        continue
      fi
      if [[ -z "$expected_size" && -s "$destination" ]]; then
        continue
      fi
      curl -fL -C - --retry 5 --retry-delay 5 -o "$destination" "$url"
    done
  echo "$target_root"
}

if ! command -v rsync >/dev/null 2>&1; then
  echo "real $SMOKE_LABEL video smoke skipped: rsync is required for the identity-safe build copy"
  exit 0
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "real $SMOKE_LABEL video smoke skipped: ffmpeg is required to create a tiny video fixture"
  exit 0
fi

if [[ "$MODEL" == */* && ! -d "$MODEL" && "$ALLOW_REMOTE" != "1" ]]; then
  echo "real $SMOKE_LABEL video smoke skipped: set MLXVLM_REAL_QWEN_VL_MODEL to a local model directory, or set MLXVLM_REAL_QWEN_VL_ALLOW_REMOTE=1 to download $MODEL"
  exit 0
fi

if [[ "$MODEL" == */* && ! -d "$MODEL" && "$ALLOW_REMOTE" == "1" ]]; then
  MODEL="$(download_hf_model "$MODEL")"
fi

TMP_ROOT="${MLXVLM_REAL_QWEN_VL_TMP_ROOT:-${TMPDIR:-/tmp}/mlx-vlm-swift-real-qwen-vl-video-smoke-$$}"
PORT="${MLXVLM_REAL_QWEN_VL_PORT:-${MLXVLM_REAL_QWEN25VL_PORT:-$((15000 + ($$ % 20000)))}}"
SCRATCH_DIR="${MLXVLM_REAL_QWEN_VL_SCRATCH:-${MLXVLM_REAL_QWEN25VL_SCRATCH:-${TMPDIR:-/tmp}/mlx-vlm-swift-real-qwen-vl-video-build}}"
READY_TIMEOUT_SECONDS="${MLXVLM_REAL_QWEN_VL_READY_TIMEOUT:-${MLXVLM_REAL_QWEN25VL_READY_TIMEOUT:-1800}}"
BUILD_DIR="$TMP_ROOT/source"
SERVER_LOG="$TMP_ROOT/server.log"
IMAGE="$TMP_ROOT/green.png"
VIDEO="$TMP_ROOT/green.mp4"
mkdir -p "$BUILD_DIR"

SERVER_PID=""
cleanup() {
  local status=$?
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ "$status" -eq 0 && "${MLXVLM_REAL_QWEN_VL_KEEP_TMP:-${MLXVLM_REAL_QWEN25VL_KEEP_TMP:-0}}" != "1" ]]; then
    rm -rf "$TMP_ROOT"
  elif [[ "$status" -ne 0 ]]; then
    echo "real $SMOKE_LABEL video smoke failed; temp dir preserved at $TMP_ROOT"
    if [[ -f "$SERVER_LOG" ]]; then
      echo "server log:"
      cat "$SERVER_LOG"
    fi
  fi
}
trap cleanup EXIT

ffmpeg -v error \
  -f lavfi -i color=c=green:s=64x64 \
  -frames:v 1 \
  "$IMAGE"

ffmpeg -v error \
  -f lavfi -i color=c=green:s=64x64:d=1:r=2 \
  -pix_fmt yuv420p \
  "$VIDEO"

rsync -a --exclude .build --exclude .git ./ "$BUILD_DIR/"
cd "$BUILD_DIR"

export MLXVLM_ENABLE_MLX_BACKEND=1
export MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1
export MLXVLM_ENABLE_REAL_MLX_API=1
export MLXVLM_ENABLE_HUGGINGFACE_DOWNLOADER=1
export CLANG_MODULE_CACHE_PATH="$BUILD_DIR/.build/clang-module-cache"

swift build --disable-sandbox --jobs "$SWIFT_BUILD_JOBS" --scratch-path "$SCRATCH_DIR" --product mlx-vlm-swift
BIN="$SCRATCH_DIR/arm64-apple-macosx/debug/mlx-vlm-swift"

"$BIN" serve --model "$MODEL" --use-latest --port "$PORT" --max-tokens 16 --temperature 0.0 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 "$READY_TIMEOUT_SECONDS"); do
  if curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited before readiness"
    cat "$SERVER_LOG"
    exit 1
  fi
  sleep 1
done

if [[ "$READY" != "1" ]]; then
  echo "server did not become ready within ${READY_TIMEOUT_SECONDS}s"
  exit 1
fi

HEALTH="$(curl -fsS -m 5 "http://127.0.0.1:$PORT/health")"
echo "$HEALTH" | grep -q '"backend_ready":true'
echo "$HEALTH" | grep -q '"backend":"mlx-swift-vlm"'

TOKENIZE_RESPONSE="$(
  curl -fsS -m 30 "http://127.0.0.1:$PORT/v1/tokenize" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$REQUEST_MODEL\",\"text\":\"Hello\",\"add_special_tokens\":true}"
)"
echo "$TOKENIZE_RESPONSE" | grep -q '"supported":true'
echo "$TOKENIZE_RESPONSE" | grep -q '"backend":"mlx-swift-vlm"'
echo "$TOKENIZE_RESPONSE" | grep -q '"token_ids":\['

GREEN_PNG="$(base64 < "$IMAGE" | tr -d '\n')"
OPENAI_IMAGE_RESPONSE="$(
  curl -fsS -m 180 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$REQUEST_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Answer with one short sentence about this image.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$GREEN_PNG\"}}]}],\"max_tokens\":16,\"temperature\":0}"
)"
echo "$OPENAI_IMAGE_RESPONSE" | grep -q '"choices"'
echo "$OPENAI_IMAGE_RESPONSE" | grep -q '"content":"'
echo "$OPENAI_IMAGE_RESPONSE" | grep -q '"prompt_tokens"'
echo "$OPENAI_IMAGE_RESPONSE" | grep -q '"completion_tokens"'

OPENAI_VIDEO_RESPONSE="$(
  curl -fsS -m 180 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$REQUEST_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Answer with one short sentence about the video.\"},{\"type\":\"video\",\"video\":\"file://$VIDEO\",\"fps\":1,\"nframes\":2,\"min_frames\":1,\"max_frames\":2}]}],\"max_tokens\":16,\"temperature\":0}"
)"
echo "$OPENAI_VIDEO_RESPONSE" | grep -q '"choices"'
echo "$OPENAI_VIDEO_RESPONSE" | grep -q '"content":"'
echo "$OPENAI_VIDEO_RESPONSE" | grep -q '"prompt_tokens"'
echo "$OPENAI_VIDEO_RESPONSE" | grep -q '"completion_tokens"'

OPENAI_STREAM="$(mktemp)"
curl -sS -m 180 -i "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$REQUEST_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one short sentence.\"}],\"max_tokens\":16,\"temperature\":0,\"stream\":true}" > "$OPENAI_STREAM"
grep -qi '^Content-Type: text/event-stream' "$OPENAI_STREAM"
if grep -qi '^Content-Length:' "$OPENAI_STREAM"; then
  echo "OpenAI $SMOKE_LABEL stream unexpectedly included Content-Length"
  cat "$OPENAI_STREAM"
  exit 1
fi
grep -q 'data: ' "$OPENAI_STREAM"
grep -q 'data: \[DONE\]' "$OPENAI_STREAM"
grep -q '"prompt_tokens"' "$OPENAI_STREAM"
grep -q '"completion_tokens"' "$OPENAI_STREAM"

echo "real $SMOKE_LABEL video smoke passed"
