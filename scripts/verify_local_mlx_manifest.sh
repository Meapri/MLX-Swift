#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
SWIFT_BUILD_JOBS="${SWIFT_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

TMP_ROOT="${TMPDIR:-/tmp}/mlx-vlm-swift-local-mock-deps"
MLX_SWIFT_DIR="$TMP_ROOT/MLXSwift"
MLX_SWIFT_LM_DIR="$TMP_ROOT/MLXSwiftLM"
SWIFT_TOKENIZERS_MLX_DIR="$TMP_ROOT/SwiftTokenizersMLX"
SCRATCH_DIR="$TMP_ROOT/build"
rm -rf "$TMP_ROOT"
mkdir -p \
  "$MLX_SWIFT_DIR/Sources/MLX" \
  "$MLX_SWIFT_LM_DIR/Sources/MLXLMCommon" \
  "$MLX_SWIFT_LM_DIR/Sources/MLXLLM" \
  "$MLX_SWIFT_LM_DIR/Sources/MLXVLM" \
  "$SWIFT_TOKENIZERS_MLX_DIR/Sources/MLXLMTokenizers"

cat > "$MLX_SWIFT_DIR/Package.swift" <<'SWIFT'
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MLXSwift",
    products: [
        .library(name: "MLX", targets: ["MLX"]),
    ],
    targets: [
        .target(name: "MLX"),
    ]
)
SWIFT

cat > "$MLX_SWIFT_DIR/Sources/MLX/MLX.swift" <<'SWIFT'
public enum MLXMockRuntime {
    public static let available = true
}
SWIFT

cat > "$MLX_SWIFT_LM_DIR/Package.swift" <<'SWIFT'
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MLXSwiftLM",
    products: [
        .library(name: "MLXLMCommon", targets: ["MLXLMCommon"]),
        .library(name: "MLXLLM", targets: ["MLXLLM"]),
        .library(name: "MLXVLM", targets: ["MLXVLM"]),
    ],
    targets: [
        .target(name: "MLXLMCommon"),
        .target(name: "MLXLLM", dependencies: ["MLXLMCommon"]),
        .target(name: "MLXVLM", dependencies: ["MLXLMCommon"]),
    ]
)
SWIFT

cat > "$MLX_SWIFT_LM_DIR/Sources/MLXLMCommon/MLXLMCommon.swift" <<'SWIFT'
public enum MLXLMCommonMockRuntime {
    public static let available = true
}

public final class ModelContainer: Sendable {
    public init() {}
}

public struct GenerateParameters: Sendable {
    public var maxTokens: Int?
    public var maxKVSize: Int?
    public var kvBits: Int?
    public var kvGroupSize: Int
    public var quantizedKVStart: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int
    public var presencePenalty: Float?
    public var presenceContextSize: Int
    public var frequencyPenalty: Float?
    public var frequencyContextSize: Int
    public var prefillStepSize: Int

    public init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        presencePenalty: Float? = nil,
        presenceContextSize: Int = 20,
        frequencyPenalty: Float? = nil,
        frequencyContextSize: Int = 20,
        prefillStepSize: Int = 512
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
        self.frequencyPenalty = frequencyPenalty
        self.frequencyContextSize = frequencyContextSize
        self.prefillStepSize = prefillStepSize
    }
}
SWIFT

cat > "$MLX_SWIFT_LM_DIR/Sources/MLXLLM/MLXLLM.swift" <<'SWIFT'
public enum MLXLLMMockRuntime {
    public static let available = true
}
SWIFT

cat > "$MLX_SWIFT_LM_DIR/Sources/MLXVLM/MLXVLM.swift" <<'SWIFT'
public final class VLMModelFactory: Sendable {
    public static let shared = VLMModelFactory()
    public init() {}
}

public enum MLXVLMMockRuntime {
    public static let available = true
}
SWIFT

cat > "$SWIFT_TOKENIZERS_MLX_DIR/Package.swift" <<'SWIFT'
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftTokenizersMLX",
    products: [
        .library(name: "MLXLMTokenizers", targets: ["MLXLMTokenizers"]),
    ],
    targets: [
        .target(name: "MLXLMTokenizers"),
    ]
)
SWIFT

cat > "$SWIFT_TOKENIZERS_MLX_DIR/Sources/MLXLMTokenizers/MLXLMTokenizers.swift" <<'SWIFT'
public struct TokenizersLoader: Sendable {
    public init() {}
}
SWIFT

export MLXVLM_ENABLE_MLX_BACKEND=1
export MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1
export MLXVLM_USE_LOCAL_MLX=1
export MLXVLM_MLX_SWIFT_PATH="$MLX_SWIFT_DIR"
export MLXVLM_MLX_SWIFT_LM_PATH="$MLX_SWIFT_LM_DIR"
export MLXVLM_SWIFT_TOKENIZERS_MLX_PATH="$SWIFT_TOKENIZERS_MLX_DIR"
export CLANG_MODULE_CACHE_PATH="$ROOT_DIR/.build/clang-module-cache"

swift build --disable-sandbox --jobs "$SWIFT_BUILD_JOBS" --scratch-path "$SCRATCH_DIR" --target MLXVLMMLXBackend
swift build --disable-sandbox --jobs "$SWIFT_BUILD_JOBS" --scratch-path "$SCRATCH_DIR" --product mlx-vlm-swift
AVAILABILITY_JSON="$(
  "$SCRATCH_DIR/arm64-apple-macosx/debug/mlx-vlm-swift" backend-availability
)"

echo "$AVAILABILITY_JSON" | grep -q '"canImportMLX" : true'
echo "$AVAILABILITY_JSON" | grep -q '"canImportMLXLMCommon" : true'
echo "$AVAILABILITY_JSON" | grep -q '"canImportMLXLLM" : true'
echo "$AVAILABILITY_JSON" | grep -q '"canImportMLXVLM" : true'
echo "$AVAILABILITY_JSON" | grep -q '"canImportMLXLMTokenizers" : true'
echo "$AVAILABILITY_JSON" | grep -q '"realMLXAPIImplementationCompiled" : false'
echo "$AVAILABILITY_JSON" | grep -q '"backendImplementationReady" : false'
echo "$AVAILABILITY_JSON" | grep -q '"canCreateBackend" : false'
