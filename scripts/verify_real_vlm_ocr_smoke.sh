#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SWIFT_BUILD_JOBS="${SWIFT_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

DEFAULT_MODEL_ID="mlx-community/gemma-4-e4b-it-4bit"
DEFAULT_LOCAL_MODEL="/Users/naen/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/cc3b666c01c20395e0dcebd53854504c7d9821f9"
MODEL="${MLXVLM_REAL_OCR_MODEL:-$DEFAULT_LOCAL_MODEL}"
ALLOW_REMOTE="${MLXVLM_REAL_OCR_ALLOW_REMOTE:-0}"
DOWNLOAD_DIR="${MLXVLM_REAL_OCR_DOWNLOAD_DIR:-}"
REQUEST_MODEL="${MLXVLM_REAL_OCR_REQUEST_MODEL:-vlm-ocr-smoke}"
SMOKE_LABEL="${MLXVLM_REAL_OCR_LABEL:-VLM OCR}"
EXPECTED_TEXT="${MLXVLM_REAL_OCR_EXPECTED_TEXT:-HELLO}"

download_hf_model() {
  local model_id="$1"
  local target_root="${DOWNLOAD_DIR:-$HOME/Models/${model_id//\//-}}"

  if ! command -v curl >/dev/null 2>&1; then
    echo "real $SMOKE_LABEL smoke skipped: curl is required to download $model_id" >&2
    exit 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "real $SMOKE_LABEL smoke skipped: python3 is required to read the Hugging Face file list for $model_id" >&2
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
  echo "real $SMOKE_LABEL smoke skipped: rsync is required for the identity-safe build copy"
  exit 0
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "real $SMOKE_LABEL smoke skipped: ffmpeg is required to create an OCR image fixture"
  exit 0
fi

if [[ "$MODEL" == */* && ! -d "$MODEL" && "$ALLOW_REMOTE" != "1" ]]; then
  echo "real $SMOKE_LABEL smoke skipped: set MLXVLM_REAL_OCR_MODEL to a local model directory, or set MLXVLM_REAL_OCR_ALLOW_REMOTE=1 to download $MODEL"
  exit 0
fi

if [[ "$MODEL" == */* && ! -d "$MODEL" && "$ALLOW_REMOTE" == "1" ]]; then
  MODEL="$(download_hf_model "$MODEL")"
fi

TMP_ROOT="${MLXVLM_REAL_OCR_TMP_ROOT:-${TMPDIR:-/tmp}/mlx-vlm-swift-real-ocr-smoke-$$}"
PORT="${MLXVLM_REAL_OCR_PORT:-$((19000 + ($$ % 20000)))}"
SCRATCH_DIR="${MLXVLM_REAL_OCR_SCRATCH:-${TMPDIR:-/tmp}/mlx-vlm-swift-real-ocr-build}"
READY_TIMEOUT_SECONDS="${MLXVLM_REAL_OCR_READY_TIMEOUT:-1800}"
BUILD_DIR="$TMP_ROOT/source"
SERVER_LOG="$TMP_ROOT/server.log"
IMAGE="$TMP_ROOT/ocr.png"
PPM_IMAGE="$TMP_ROOT/ocr.ppm"
OCR_RESPONSE_FILE="$TMP_ROOT/ocr_response.json"
mkdir -p "$BUILD_DIR"

SERVER_PID=""
cleanup() {
  local status=$?
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ "$status" -eq 0 && "${MLXVLM_REAL_OCR_KEEP_TMP:-0}" != "1" ]]; then
    rm -rf "$TMP_ROOT"
  elif [[ "$status" -ne 0 ]]; then
    echo "real $SMOKE_LABEL smoke failed; temp dir preserved at $TMP_ROOT"
    if [[ -f "$SERVER_LOG" ]]; then
      echo "server log:"
      cat "$SERVER_LOG"
    fi
  fi
}
trap cleanup EXIT

create_ocr_ppm() {
  local output="$1"
  awk -v text="$EXPECTED_TEXT" -v scale=10 -v margin=28 '
BEGIN {
  font["S",0] = "01111"; font["S",1] = "10000"; font["S",2] = "10000"; font["S",3] = "01110"; font["S",4] = "00001"; font["S",5] = "00001"; font["S",6] = "11110"
  font["W",0] = "10001"; font["W",1] = "10001"; font["W",2] = "10001"; font["W",3] = "10101"; font["W",4] = "10101"; font["W",5] = "11011"; font["W",6] = "10001"
  font["I",0] = "11111"; font["I",1] = "00100"; font["I",2] = "00100"; font["I",3] = "00100"; font["I",4] = "00100"; font["I",5] = "00100"; font["I",6] = "11111"
  font["F",0] = "11111"; font["F",1] = "10000"; font["F",2] = "10000"; font["F",3] = "11110"; font["F",4] = "10000"; font["F",5] = "10000"; font["F",6] = "10000"
  font["T",0] = "11111"; font["T",1] = "00100"; font["T",2] = "00100"; font["T",3] = "00100"; font["T",4] = "00100"; font["T",5] = "00100"; font["T",6] = "00100"
  font["4",0] = "10010"; font["4",1] = "10010"; font["4",2] = "10010"; font["4",3] = "11111"; font["4",4] = "00010"; font["4",5] = "00010"; font["4",6] = "00010"
  font["2",0] = "01110"; font["2",1] = "10001"; font["2",2] = "00001"; font["2",3] = "00010"; font["2",4] = "00100"; font["2",5] = "01000"; font["2",6] = "11111"
  charw = 5
  charh = 7
  spacing = 2
  cellw = (charw + spacing) * scale
  w = (2 * margin) + (length(text) * cellw) - (spacing * scale)
  h = (2 * margin) + (charh * scale)
  print "P3"
  print w " " h
  print 255
  for (y = 0; y < h; y++) {
    line = ""
    for (x = 0; x < w; x++) {
      black = 0
      tx = x - margin
      ty = y - margin
      if (tx >= 0 && ty >= 0 && ty < charh * scale) {
        ci = int(tx / cellw)
        within = tx - (ci * cellw)
        if (ci >= 0 && ci < length(text) && within < charw * scale) {
          ch = substr(text, ci + 1, 1)
          row = int(ty / scale)
          col = int(within / scale)
          pattern = font[ch,row]
          if (substr(pattern, col + 1, 1) == "1") {
            black = 1
          }
        }
      }
      line = line (black ? "0 0 0 " : "255 255 255 ")
    }
    print line
  }
}
' > "$output"
}

create_ocr_png_with_swift() {
  /usr/bin/swift - "$IMAGE" "$EXPECTED_TEXT" <<'SWIFT'
import AppKit
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
    exit(2)
}
let output = arguments[1]
let text = arguments[2]
let size = NSSize(width: 640, height: 240)
let image = NSImage(size: size)
image.lockFocus()
NSColor.white.setFill()
NSBezierPath(rect: NSRect(origin: .zero, size: size)).fill()
let paragraph = NSMutableParagraphStyle()
paragraph.alignment = .center
let attributes: [NSAttributedString.Key: Any] = [
    .font: NSFont.boldSystemFont(ofSize: 120),
    .foregroundColor: NSColor.black,
    .paragraphStyle: paragraph,
]
let textSize = text.size(withAttributes: attributes)
let rect = NSRect(
    x: 0,
    y: (size.height - textSize.height) / 2,
    width: size.width,
    height: textSize.height
)
text.draw(in: rect, withAttributes: attributes)
image.unlockFocus()
guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let png = bitmap.representation(using: .png, properties: [:])
else {
    exit(3)
}
try png.write(to: URL(fileURLWithPath: output))
SWIFT
}

if ! create_ocr_png_with_swift
then
  create_ocr_ppm "$PPM_IMAGE"
  if ! ffmpeg -v error -i "$PPM_IMAGE" -frames:v 1 "$IMAGE"
  then
    echo "real $SMOKE_LABEL smoke skipped: could not create the OCR image fixture"
    exit 0
  fi
fi

if [[ ! -s "$IMAGE" ]]
then
  echo "real $SMOKE_LABEL smoke skipped: OCR image fixture was not created"
  exit 0
fi

rsync -a --exclude .build --exclude .git ./ "$BUILD_DIR/"
cd "$BUILD_DIR"

export MLXVLM_ENABLE_MLX_BACKEND=1
export MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1
export MLXVLM_ENABLE_REAL_MLX_API=1
export MLXVLM_ENABLE_HUGGINGFACE_DOWNLOADER=1
export CLANG_MODULE_CACHE_PATH="$BUILD_DIR/.build/clang-module-cache"

swift build --disable-sandbox --jobs "$SWIFT_BUILD_JOBS" --scratch-path "$SCRATCH_DIR" --product mlx-vlm-swift
BIN="$SCRATCH_DIR/arm64-apple-macosx/debug/mlx-vlm-swift"

"$BIN" serve --model "$MODEL" --use-latest --port "$PORT" --max-tokens 24 --temperature 0.0 > "$SERVER_LOG" 2>&1 &
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

OCR_PNG="$(base64 < "$IMAGE" | tr -d '\n')"
curl -fsS -m 240 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$REQUEST_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Read the large text in the image. Reply with only the exact text, no punctuation.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$OCR_PNG\"}}]}],\"max_tokens\":24,\"temperature\":0}" \
  > "$OCR_RESPONSE_FILE"
OCR_RESPONSE="$(cat "$OCR_RESPONSE_FILE")"
echo "$OCR_RESPONSE" | grep -q '"choices"'
echo "$OCR_RESPONSE" | grep -q '"content":"'
echo "$OCR_RESPONSE" | grep -q '"prompt_tokens"'
echo "$OCR_RESPONSE" | grep -q '"completion_tokens"'
if ! echo "$OCR_RESPONSE" | tr '[:lower:]' '[:upper:]' | grep -q "$EXPECTED_TEXT"; then
  echo "real $SMOKE_LABEL smoke did not find expected OCR text $EXPECTED_TEXT"
  cat "$OCR_RESPONSE_FILE"
  exit 1
fi

echo "real $SMOKE_LABEL smoke passed"
