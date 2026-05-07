#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
SCRATCH_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-real-mlx-backend-build"

candidate_path() {
  local package_name="$1"
  local explicit_var="$2"
  local alias_name="$3"
  local explicit_path="${!explicit_var:-}"
  if [[ -n "$explicit_path" ]]; then
    printf '%s\n' "$explicit_path"
    return 0
  fi

  for path in \
    "$ROOT_DIR/vendor/$alias_name" \
    "$ROOT_DIR/Vendor/$alias_name" \
    "$ROOT_DIR/Dependencies/$alias_name" \
    "$(dirname "$ROOT_DIR")/$alias_name" \
    "$ROOT_DIR/vendor/$package_name" \
    "$ROOT_DIR/Vendor/$package_name" \
    "$ROOT_DIR/Dependencies/$package_name" \
    "$ROOT_DIR/.build/checkouts/$package_name" \
    "$(dirname "$ROOT_DIR")/$package_name"
  do
    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done

  return 1
}

MLX_SWIFT_PATH="$(candidate_path mlx-swift MLXVLM_MLX_SWIFT_PATH MLXSwift || true)"
MLX_SWIFT_LM_PATH="$(candidate_path mlx-swift-lm MLXVLM_MLX_SWIFT_LM_PATH MLXSwiftLM || true)"
SWIFT_TOKENIZERS_MLX_PATH="$(candidate_path swift-tokenizers-mlx MLXVLM_SWIFT_TOKENIZERS_MLX_PATH SwiftTokenizersMLX || true)"

if [[ -z "$MLX_SWIFT_PATH" || -z "$MLX_SWIFT_LM_PATH" || -z "$SWIFT_TOKENIZERS_MLX_PATH" ]]; then
  echo "MLX Swift dependencies are not available locally."
  echo
  echo "Provide local checkouts with either:"
  echo "  vendor/MLXSwift, vendor/MLXSwiftLM, and vendor/SwiftTokenizersMLX"
  echo "  or vendor/mlx-swift, vendor/mlx-swift-lm, and vendor/swift-tokenizers-mlx when the root package identity does not conflict"
  echo "  or MLXVLM_MLX_SWIFT_PATH=/path/to/mlx-swift"
  echo "     MLXVLM_MLX_SWIFT_LM_PATH=/path/to/mlx-swift-lm"
  echo "     MLXVLM_SWIFT_TOKENIZERS_MLX_PATH=/path/to/swift-tokenizers-mlx"
  echo
  echo "Current backend plan:"
  export CLANG_MODULE_CACHE_PATH="$ROOT_DIR/.build/clang-module-cache"
  swift build --disable-sandbox
  "$ROOT_DIR/.build/arm64-apple-macosx/debug/mlx-vlm-swift" backend-plan
  exit 2
fi

export MLXVLM_ENABLE_MLX_BACKEND=1
export MLXVLM_ENABLE_REAL_MLX_API=1
export MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1
export MLXVLM_USE_LOCAL_MLX=1
export MLXVLM_MLX_SWIFT_PATH="$MLX_SWIFT_PATH"
export MLXVLM_MLX_SWIFT_LM_PATH="$MLX_SWIFT_LM_PATH"
export MLXVLM_SWIFT_TOKENIZERS_MLX_PATH="$SWIFT_TOKENIZERS_MLX_PATH"
export CLANG_MODULE_CACHE_PATH="$ROOT_DIR/.build/clang-module-cache"

swift build --disable-sandbox --scratch-path "$SCRATCH_DIR" --target MLXVLMMLXBackend
swift build --disable-sandbox --scratch-path "$SCRATCH_DIR" --product mlx-vlm-swift
"$SCRATCH_DIR/arm64-apple-macosx/debug/mlx-vlm-swift" backend-availability
