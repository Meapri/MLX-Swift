#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
SWIFT_BUILD_JOBS="${SWIFT_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

TMP_ROOT="${TMPDIR:-/tmp}/mlx-vlm-swift-mock-real-mlx-api"
MLX_SWIFT_DIR="$TMP_ROOT/MLXSwift"
MLX_SWIFT_LM_DIR="$TMP_ROOT/MLXSwiftLM"
SWIFT_TOKENIZERS_MLX_DIR="$TMP_ROOT/SwiftTokenizersMLX"
SCRATCH_DIR="$TMP_ROOT/build"
MODEL_DIR="$TMP_ROOT/model"
rm -rf "$TMP_ROOT"
mkdir -p \
  "$MLX_SWIFT_DIR/Sources/MLX" \
  "$MLX_SWIFT_LM_DIR/Sources/MLXLMCommon" \
  "$MLX_SWIFT_LM_DIR/Sources/MLXLLM" \
  "$MLX_SWIFT_LM_DIR/Sources/MLXVLM" \
  "$MLX_SWIFT_LM_DIR/Sources/MLXEmbedders" \
  "$SWIFT_TOKENIZERS_MLX_DIR/Sources/MLXLMTokenizers" \
  "$MODEL_DIR"

cat > "$MLX_SWIFT_DIR/Package.swift" <<'SWIFT'
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MLXSwift",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MLX", targets: ["MLX"]),
    ],
    targets: [
        .target(name: "MLX"),
    ]
)
SWIFT

cat > "$MLX_SWIFT_DIR/Sources/MLX/MLX.swift" <<'SWIFT'
import Foundation

public enum DType: String, Sendable {
    case bool
    case uint8
    case uint16
    case uint32
    case uint64
    case int8
    case int16
    case int32
    case int64
    case float16
    case bfloat16
    case float32
    case float64
}

public final class MLXArray: @unchecked Sendable {
    public let data: Data
    public let shape: [Int]
    public let dtype: DType

    public init(_ data: Data, _ shape: [Int], dtype: DType) {
        self.data = data
        self.shape = shape
        self.dtype = dtype
    }

    public convenience init(_ values: [Int]) {
        self.init(Data(), [values.count], dtype: .int32)
    }

    public convenience init(_ value: Float) {
        self.init(Data(), [], dtype: .float32)
    }

    public var size: Int {
        shape.reduce(1, *)
    }

    public static func zeros(like other: MLXArray) -> MLXArray {
        MLXArray(Data(), other.shape, dtype: other.dtype)
    }

    public static func zeros(_ shape: [Int], dtype: DType = .float32) -> MLXArray {
        MLXArray(Data(), shape, dtype: dtype)
    }

    public func asType(_ dtype: DType) -> MLXArray {
        MLXArray(data, shape, dtype: dtype)
    }

    public subscript(_ rows: PartialRangeFrom<Int>, _ columns: MLXArray) -> MLXArray {
        get { self }
        set {}
    }

    public func eval() {}

    public func asArray<T>(_ type: T.Type) -> [T] {
        []
    }

    public func map<T>(_ transform: (MLXArray) throws -> T) rethrows -> [T] {
        [try transform(self)]
    }
}

public func + (lhs: MLXArray, rhs: MLXArray) -> MLXArray {
    lhs
}

public func stacked(_ arrays: [MLXArray]) -> MLXArray {
    let rowCount = arrays.count
    let columnCount = arrays.first?.shape.first ?? 0
    return MLXArray(Data(), [rowCount, columnCount], dtype: .int32)
}

infix operator .!=: ComparisonPrecedence

public func .!= (lhs: MLXArray, rhs: Int) -> MLXArray {
    MLXArray(Data(), lhs.shape, dtype: .bool)
}

public enum MLXRandom {
    public static func seed(_ value: UInt64) {}
}
SWIFT

cat > "$MLX_SWIFT_LM_DIR/Package.swift" <<'SWIFT'
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MLXSwiftLM",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MLXLMCommon", targets: ["MLXLMCommon"]),
        .library(name: "MLXLLM", targets: ["MLXLLM"]),
        .library(name: "MLXVLM", targets: ["MLXVLM"]),
        .library(name: "MLXEmbedders", targets: ["MLXEmbedders"]),
    ],
    dependencies: [
        .package(name: "MLXSwift", path: "../MLXSwift"),
    ],
    targets: [
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: "MLXSwift"),
            ]
        ),
        .target(name: "MLXLLM", dependencies: ["MLXLMCommon"]),
        .target(name: "MLXVLM", dependencies: ["MLXLMCommon"]),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "MLXSwift"),
            ]
        ),
    ]
)
SWIFT

cat > "$MLX_SWIFT_LM_DIR/Sources/MLXLMCommon/MLXLMCommon.swift" <<'SWIFT'
import CoreGraphics
import Foundation
import MLX

public enum MLXLMCommonMockRuntime {
    public static let available = true
}

public struct LMInputText: Sendable {
    public var tokens: MLXArray

    public init(tokens: MLXArray = MLXArray([1])) {
        self.tokens = tokens
    }
}

public struct LMInput: Sendable {
    public var text: LMInputText

    public init(text: LMInputText = LMInputText()) {
        self.text = text
    }
}

public protocol Tokenizer: Sendable {
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertIdToToken(_ id: Int) -> String?

    var bosToken: String? { get }
    var eosToken: String? { get }
    var unknownToken: String? { get }
}

extension Tokenizer {
    public var eosTokenId: Int? {
        guard let eosToken else { return nil }
        return convertTokenToId(eosToken)
    }
}

public enum JSONValue: Sendable, CustomStringConvertible {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public var description: String {
        "json"
    }
}

public struct ToolCall: Sendable {
    public struct Function: Sendable {
        public let name: String
        public let arguments: [String: JSONValue]

        public init(name: String, arguments: [String: JSONValue]) {
            self.name = name
            self.arguments = arguments
        }
    }

    public let function: Function

    public init(function: Function = Function(name: "mock", arguments: [:])) {
        self.function = function
    }
}

public enum GenerateStopReason: Sendable {
    case stop
    case length
    case cancelled
}

public struct GenerateCompletionInfo: Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let stopReason: GenerateStopReason

    public init(
        promptTokenCount: Int = 1,
        generationTokenCount: Int = 1,
        stopReason: GenerateStopReason = .stop
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.stopReason = stopReason
    }
}

public enum Generation: Sendable {
    case chunk(String)
    case info(GenerateCompletionInfo)
    case toolCall(ToolCall)
}

public protocol LogitProcessor {
    mutating func prompt(_ prompt: MLXArray)
    func process(logits: MLXArray) -> MLXArray
    mutating func didSample(token: MLXArray)
}

extension LogitProcessor {
    public mutating func prompt(_ prompt: MLXArray) {}
    public func process(logits: MLXArray) -> MLXArray { logits }
    public mutating func didSample(token: MLXArray) {}
}

public struct MockSampler: Sendable {
    public init() {}
}

public struct TokenIterator: Sendable {
    public init(
        input: LMInput,
        model: MockModel,
        processor: any LogitProcessor,
        sampler: MockSampler,
        prefillStepSize: Int,
        maxTokens: Int?
    ) throws {}
}

public func generateTask(
    promptTokenCount: Int,
    modelConfiguration: MockModelConfiguration,
    tokenizer: any Tokenizer,
    iterator: TokenIterator
) -> (AsyncStream<Generation>, Task<Void, Never>?) {
    let stream = AsyncStream<Generation> { continuation in
        continuation.yield(.chunk("mock"))
        continuation.yield(.info(GenerateCompletionInfo(promptTokenCount: promptTokenCount)))
        continuation.finish()
    }
    return (stream, nil)
}

public func generate(
    input: LMInput,
    parameters: GenerateParameters,
    context: ModelContext,
    draftModel: MockModel,
    numDraftTokens: Int
) throws -> AsyncStream<Generation> {
    AsyncStream { continuation in
        continuation.yield(.chunk("mock"))
        continuation.yield(.info(GenerateCompletionInfo()))
        continuation.finish()
    }
}

public enum Chat {
    public struct Message: Sendable {
        public var role: Role
        public var content: String
        public var images: [UserInput.Image]
        public var videos: [UserInput.Video]

        public init(role: Role, content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []) {
            self.role = role
            self.content = content
            self.images = images
            self.videos = videos
        }

        public static func system(_ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []) -> Self {
            Self(role: .system, content: content, images: images, videos: videos)
        }

        public static func user(_ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []) -> Self {
            Self(role: .user, content: content, images: images, videos: videos)
        }

        public static func assistant(_ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []) -> Self {
            Self(role: .assistant, content: content, images: images, videos: videos)
        }

        public static func tool(_ content: String) -> Self {
            Self(role: .tool, content: content)
        }

        public enum Role: String, Sendable {
            case user
            case assistant
            case system
            case tool
        }
    }
}

public typealias ToolSpec = [String: any Sendable]
public typealias Message = [String: any Sendable]

public struct UserInput: Sendable {
    public enum Image: Sendable {
        case url(URL)
    }

    public enum Video: Sendable {
        case url(URL)
    }

    public struct Processing: Sendable {
        public var resize: CGSize?

        public init(resize: CGSize? = nil) {
            self.resize = resize
        }
    }

    public var chat: [Chat.Message]
    public var messages: [Message]?
    public var images: [Image]
    public var videos: [Video]
    public var processing: Processing
    public var tools: [ToolSpec]?
    public var additionalContext: [String: any Sendable]?

    public init(
        chat: [Chat.Message],
        processing: Processing = Processing(),
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) {
        self.chat = chat
        self.messages = nil
        self.images = []
        self.videos = []
        self.processing = processing
        self.tools = tools
        self.additionalContext = additionalContext
    }

    public init(
        messages: [Message],
        images: [Image] = [],
        videos: [Video] = [],
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) {
        self.chat = []
        self.messages = messages
        self.images = images
        self.videos = videos
        self.processing = Processing()
        self.tools = tools
        self.additionalContext = additionalContext
    }
}

public struct MockModel: Sendable {
    public init() {}
}

public struct MockModelConfiguration: Sendable {
    public init() {}
}

public struct MockTokenizer: Tokenizer {
    public init() {}

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if text.hasPrefix("{") {
            return [123]
        }
        if text.hasPrefix("[") {
            return [91]
        }
        return addSpecialTokens ? [1, 2] : [2]
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        "mock"
    }

    public func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "</s>": return 0
        case "{": return 123
        case "[": return 91
        default: return nil
        }
    }

    public func convertIdToToken(_ id: Int) -> String? {
        id == 0 ? "</s>" : nil
    }

    public let bosToken: String? = "<s>"
    public let eosToken: String? = "</s>"
    public let unknownToken: String? = "<unk>"
}

public struct ModelContext: Sendable {
    public var model: MockModel
    public var configuration: MockModelConfiguration
    public var tokenizer: any Tokenizer

    public init(
        model: MockModel = MockModel(),
        configuration: MockModelConfiguration = MockModelConfiguration(),
        tokenizer: any Tokenizer = MockTokenizer()
    ) {
        self.model = model
        self.configuration = configuration
        self.tokenizer = tokenizer
    }
}

public final class ModelContainer: Sendable {
    private let context = ModelContext()

    public init() {}

    public var tokenizer: any Tokenizer {
        context.tokenizer
    }

    public func prepare(input: consuming sending UserInput) async throws -> sending LMInput {
        LMInput()
    }

    public func generate(
        input: consuming sending LMInput,
        parameters: GenerateParameters
    ) async throws -> AsyncStream<Generation> {
        AsyncStream { continuation in
            continuation.yield(.chunk("mock"))
            continuation.yield(.info(GenerateCompletionInfo()))
            continuation.finish()
        }
    }

    public func perform<R: Sendable>(
        _ action: @Sendable (ModelContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await action(context)
    }

    public func perform<Value, R: Sendable>(
        nonSendable value: consuming Value,
        _ action: (ModelContext, consuming Value) async throws -> sending R
    ) async rethrows -> sending R {
        try await action(context, value)
    }

    public func update(_ action: @Sendable (ModelContext) -> Void) async {
        action(context)
    }
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

    public func processor() -> (any LogitProcessor)? {
        nil
    }

    public func sampler() -> MockSampler {
        MockSampler()
    }
}

public protocol ModelAdapter: Sendable {
    func load(into model: MockModel) throws
    func unload(from model: MockModel)
}

public struct MockModelAdapter: ModelAdapter {
    public init() {}
    public func load(into model: MockModel) throws {}
    public func unload(from model: MockModel) {}
}

public struct ModelAdapterRegistry: Sendable {
    public init() {}

    public func createAdapter(directory: URL, adapterType: String) throws -> any ModelAdapter {
        MockModelAdapter()
    }
}

public final class ModelAdapterFactory: Sendable {
    public static let shared = ModelAdapterFactory()
    public let registry = ModelAdapterRegistry()
    public init() {}
}
SWIFT

cat > "$MLX_SWIFT_LM_DIR/Sources/MLXLLM/MLXLLM.swift" <<'SWIFT'
import Foundation
import MLXLMCommon

public enum MLXLLMMockRuntime {
    public static let available = true
}

public final class LLMModelFactory: Sendable {
    public static let shared = LLMModelFactory()
    public init() {}

    public func loadContainer<T>(
        from directory: URL,
        using tokenizerLoader: T
    ) async throws -> MLXLMCommon.ModelContainer {
        MLXLMCommon.ModelContainer()
    }
}
SWIFT

cat > "$MLX_SWIFT_LM_DIR/Sources/MLXVLM/MLXVLM.swift" <<'SWIFT'
import Foundation
import MLXLMCommon

public final class VLMModelFactory: Sendable {
    public static let shared = VLMModelFactory()
    public init() {}

    public func loadContainer<T>(
        from directory: URL,
        using tokenizerLoader: T
    ) async throws -> MLXLMCommon.ModelContainer {
        MLXLMCommon.ModelContainer()
    }
}

public enum MLXVLMMockRuntime {
    public static let available = true
}
SWIFT

cat > "$MLX_SWIFT_LM_DIR/Sources/MLXEmbedders/MLXEmbedders.swift" <<'SWIFT'
import Foundation
import MLX
import MLXLMCommon

public struct MockTokenizer: Tokenizer {
    public init() {}

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        addSpecialTokens ? [1, 2] : [2]
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        "mock"
    }

    public func convertTokenToId(_ token: String) -> Int? {
        token == "</s>" ? 0 : nil
    }

    public func convertIdToToken(_ id: Int) -> String? {
        id == 0 ? "</s>" : nil
    }

    public let bosToken: String? = "<s>"
    public let eosToken: String? = "</s>"
    public let unknownToken: String? = "<unk>"
}

public struct MockEmbeddingModel: Sendable {
    public init() {}

    public func callAsFunction(
        _ inputs: MLXArray,
        positionIds: MLXArray?,
        tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> MLXArray {
        inputs
    }
}

public struct MockPooling: Sendable {
    public init() {}

    public func callAsFunction(
        _ output: MLXArray,
        mask: MLXArray?,
        normalize: Bool,
        applyLayerNorm: Bool
    ) -> MLXArray {
        output
    }
}

public struct EmbedderModelContext: Sendable {
    public let tokenizer: any Tokenizer
    public let model: MockEmbeddingModel
    public let pooling: MockPooling

    public init(
        tokenizer: any Tokenizer = MockTokenizer(),
        model: MockEmbeddingModel = MockEmbeddingModel(),
        pooling: MockPooling = MockPooling()
    ) {
        self.tokenizer = tokenizer
        self.model = model
        self.pooling = pooling
    }
}

public final class EmbedderModelContainer: Sendable {
    private let context = EmbedderModelContext()

    public init() {}

    public func perform<R: Sendable>(
        _ action: @Sendable (EmbedderModelContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await action(context)
    }
}

public final class EmbedderModelFactory: Sendable {
    public static let shared = EmbedderModelFactory()
    public init() {}

    public func loadContainer<T>(
        from directory: URL,
        using tokenizerLoader: T
    ) async throws -> EmbedderModelContainer {
        EmbedderModelContainer()
    }
}
SWIFT

cat > "$SWIFT_TOKENIZERS_MLX_DIR/Package.swift" <<'SWIFT'
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftTokenizersMLX",
    platforms: [.macOS(.v14)],
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

cat > "$MODEL_DIR/config.json" <<'JSON'
{
  "model_type": "qwen2_vl",
  "hidden_size": 1536,
  "num_hidden_layers": 28,
  "intermediate_size": 8960,
  "num_attention_heads": 12,
  "rms_norm_eps": 0.000001,
  "vocab_size": 151936,
  "vision_config": {"model_type": "qwen2_vl"}
}
JSON

cat > "$MODEL_DIR/tokenizer_config.json" <<'JSON'
{
  "chat_template": "verify-template",
  "added_tokens_decoder": {
    "151655": {"content": "<|image_pad|>"},
    "151656": {"content": "<|video_pad|>"},
    "151644": {"content": "<|im_start|>"},
    "151645": {"content": "<|im_end|>"}
  }
}
JSON

cat > "$MODEL_DIR/tokenizer.json" <<'JSON'
{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"hello": 0, "world": 1, "user": 2},
    "merges": ["hello world"]
  },
  "added_tokens": [
    {"id": 151655, "content": "<|image_pad|>"},
    {"id": 151656, "content": "<|video_pad|>"},
    {"id": 151644, "content": "<|im_start|>"},
    {"id": 151645, "content": "<|im_end|>"}
  ],
  "normalizer": {"type": "Sequence"},
  "pre_tokenizer": {"type": "ByteLevel"},
  "decoder": {"type": "ByteLevel"}
}
JSON

cat > "$MODEL_DIR/model.safetensors.index.json" <<'JSON'
{
  "metadata": {"total_size": 8},
  "weight_map": {
    "model.embed_tokens.weight": "model.safetensors",
    "visual.patch_embed.proj.weight": "model.safetensors"
  }
}
JSON

SAFETENSORS_HEADER='{"model.embed_tokens.weight":{"dtype":"F16","shape":[2,1],"data_offsets":[0,4]},"visual.patch_embed.proj.weight":{"dtype":"F16","shape":[2,1],"data_offsets":[4,8]}}'
printf '\244\0\0\0\0\0\0\0%s\000\074\000\100\000\102\000\104' "$SAFETENSORS_HEADER" > "$MODEL_DIR/model.safetensors"

export MLXVLM_ENABLE_MLX_BACKEND=1
export MLXVLM_ENABLE_REAL_MLX_API=1
export MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1
export MLXVLM_USE_LOCAL_MLX=1
export MLXVLM_MLX_SWIFT_PATH="$MLX_SWIFT_DIR"
export MLXVLM_MLX_SWIFT_LM_PATH="$MLX_SWIFT_LM_DIR"
export MLXVLM_SWIFT_TOKENIZERS_MLX_PATH="$SWIFT_TOKENIZERS_MLX_DIR"
export CLANG_MODULE_CACHE_PATH="$ROOT_DIR/.build/clang-module-cache"

swift build --disable-sandbox --jobs "$SWIFT_BUILD_JOBS" --scratch-path "$SCRATCH_DIR" --target MLXVLMMLXBackend
swift build --disable-sandbox --jobs "$SWIFT_BUILD_JOBS" --scratch-path "$SCRATCH_DIR" --product mlx-vlm-swift

BIN="$SCRATCH_DIR/arm64-apple-macosx/debug/mlx-vlm-swift"
"$BIN" self-test

AVAILABILITY_JSON="$("$BIN" backend-availability)"
grep -q '"canImportMLX" : true' <<< "$AVAILABILITY_JSON"
grep -q '"canImportMLXLMCommon" : true' <<< "$AVAILABILITY_JSON"
grep -q '"canImportMLXLLM" : true' <<< "$AVAILABILITY_JSON"
grep -q '"canImportMLXVLM" : true' <<< "$AVAILABILITY_JSON"
grep -q '"canImportMLXLMTokenizers" : true' <<< "$AVAILABILITY_JSON"
grep -q '"realMLXAPIImplementationCompiled" : true' <<< "$AVAILABILITY_JSON"
grep -q '"backendImplementationReady" : true' <<< "$AVAILABILITY_JSON"
grep -q '"canCreateBackend" : true' <<< "$AVAILABILITY_JSON"

LOAD_JSON="$("$BIN" preflight-mlx-backend-load --model "$MODEL_DIR" --max-tensors 2 --max-total-bytes 8)"
grep -q '"canPrepareWeightPayloads" : true' <<< "$LOAD_JSON"
grep -q '"realMLXAPIImplementationCompiled" : true' <<< "$LOAD_JSON"
grep -q '"loadedArrayCount" : 2' <<< "$LOAD_JSON"
grep -q '"canCreateMLXArrays" : true' <<< "$LOAD_JSON"
grep -q '"canInstantiateModelModules" : false' <<< "$LOAD_JSON"
grep -q '"canRunGeneration" : false' <<< "$LOAD_JSON"
grep -q '"language_model.model.embed_tokens.weight"' <<< "$LOAD_JSON"

CONTAINER_JSON="$("$BIN" inspect-mlx-container --model "$MODEL_DIR" --max-tensors 2 --max-total-bytes 8)"
grep -q '"preparedWeightTensorCount" : 2' <<< "$CONTAINER_JSON"
grep -q '"loadedArrayCount" : 2' <<< "$CONTAINER_JSON"
grep -q '"arrayBacked" : true' <<< "$CONTAINER_JSON"
grep -q '"moduleInstantiationReady" : false' <<< "$CONTAINER_JSON"
grep -q '"generationReady" : false' <<< "$CONTAINER_JSON"

MODULE_PLAN_JSON="$("$BIN" inspect-mlx-module-plan --model "$MODEL_DIR" --max-tensors 2 --max-total-bytes 8)"
grep -q '"moduleConstructionReady" : false' <<< "$MODULE_PLAN_JSON"
grep -q '"loadedArrayCount" : 2' <<< "$MODULE_PLAN_JSON"
grep -q '"phase" : "token-embedding"' <<< "$MODULE_PLAN_JSON"
grep -q '"phase" : "vision-patch-embedding"' <<< "$MODULE_PLAN_JSON"
grep -q '"constructible" : false' <<< "$MODULE_PLAN_JSON"
grep -q 'Qwen VL module construction is not ready yet.' <<< "$MODULE_PLAN_JSON"

FORWARD_PLAN_JSON="$("$BIN" inspect-mlx-forward-plan --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' --max-tensors 2 --max-total-bytes 8)"
grep -q '"forwardReady" : false' <<< "$FORWARD_PLAN_JSON"
grep -q '"nextTokenSelectionReady" : false' <<< "$FORWARD_PLAN_JSON"
grep -q '"vocabSize" : 151936' <<< "$FORWARD_PLAN_JSON"
grep -q '"logitsShape"' <<< "$FORWARD_PLAN_JSON"
grep -q 'Qwen VL modules must be constructed before forward can run.' <<< "$FORWARD_PLAN_JSON"

GENERATION_LOOP_JSON="$("$BIN" inspect-mlx-generation-loop --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' --max-tensors 2 --max-total-bytes 8)"
grep -q '"generationLoopReady" : false' <<< "$GENERATION_LOOP_JSON"
grep -q '"expectedFirstStep" : "prefill"' <<< "$GENERATION_LOOP_JSON"
grep -q '"expectedLoop" : "decode-next-token-until-stop-or-length"' <<< "$GENERATION_LOOP_JSON"
grep -q '"sampler" : "greedy"' <<< "$GENERATION_LOOP_JSON"
grep -q '"assemblerHandoffReady" : true' <<< "$GENERATION_LOOP_JSON"
grep -q 'MLX-backed decode loop is not implemented yet.' <<< "$GENERATION_LOOP_JSON"

GENERATE_PARAMETERS_BRIDGE_JSON="$("$BIN" inspect-mlx-generate-parameters-bridge --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}],"max_tokens":33,"temperature":0.7,"top_p":0.8,"top_k":12,"min_p":0.05,"repetition_penalty":1.1}' --kv-bits 8 --kv-group-size 128 --max-kv-size 4096)"
grep -q '"canBridgeToGenerateParameters" : true' <<< "$GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q '"maxTokens" : 33' <<< "$GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q '"maxKVSize" : 4096' <<< "$GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q '"kvBits" : 8' <<< "$GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q '"kvGroupSize" : 128' <<< "$GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q '"temperature" : 0.7' <<< "$GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q '"topP" : 0.8' <<< "$GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q '"topK" : 12' <<< "$GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q '"minP" : 0.05' <<< "$GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q '"repetitionPenalty" : 1.1' <<< "$GENERATE_PARAMETERS_BRIDGE_JSON"

MODEL_FACTORY_BRIDGE_JSON="$("$BIN" inspect-mlx-model-factory-bridge --model "$MODEL_DIR")"
grep -q '"canReferenceVLMModelFactory" : true' <<< "$MODEL_FACTORY_BRIDGE_JSON"
grep -q '"canReferenceModelContainer" : true' <<< "$MODEL_FACTORY_BRIDGE_JSON"
grep -q '"canReferenceGenerateParameters" : true' <<< "$MODEL_FACTORY_BRIDGE_JSON"
grep -q '"canReferenceTokenizersLoader" : true' <<< "$MODEL_FACTORY_BRIDGE_JSON"
grep -q '"canReferenceGenerationEntrypoints" : true' <<< "$MODEL_FACTORY_BRIDGE_JSON"
grep -q '"canLoadLocalModelContainer" : true' <<< "$MODEL_FACTORY_BRIDGE_JSON"

DECODE_STATE_JSON="$("$BIN" inspect-mlx-decode-state --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' --max-tensors 2 --max-total-bytes 8)"
grep -q '"canInitializeDecodeState" : true' <<< "$DECODE_STATE_JSON"
grep -q '"nextTokenInputShape"' <<< "$DECODE_STATE_JSON"
grep -q '"eosTokenIDs" : \[' <<< "$DECODE_STATE_JSON"
grep -q '"max-completion-tokens"' <<< "$DECODE_STATE_JSON"

PIPELINE_JSON="$("$BIN" inspect-mlx-pipeline --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' --max-tensors 2 --max-total-bytes 8)"
grep -q '"pipelineReady" : false' <<< "$PIPELINE_JSON"
grep -q '"loadedArrayCount" : 2' <<< "$PIPELINE_JSON"
grep -q '"decodeState"' <<< "$PIPELINE_JSON"
grep -q '"generateParametersBridge"' <<< "$PIPELINE_JSON"
grep -q '"modelFactoryBridge"' <<< "$PIPELINE_JSON"
grep -q '"canBridgeToGenerateParameters" : true' <<< "$PIPELINE_JSON"
grep -q '"canReferenceVLMModelFactory" : true' <<< "$PIPELINE_JSON"
grep -q '"canLoadLocalModelContainer" : true' <<< "$PIPELINE_JSON"
grep -q '"requestModel" : "verify"' <<< "$PIPELINE_JSON"
grep -q 'MLX-backed decode loop is not implemented yet.' <<< "$PIPELINE_JSON"

LIGHTWEIGHT_PIPELINE_JSON="$("$BIN" inspect-mlx-pipeline --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' --skip-weight-payloads)"
grep -q '"pipelineReady" : false' <<< "$LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"preparedWeightTensorCount" : 0' <<< "$LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"canCreateMLXArrays" : false' <<< "$LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"canBridgeToGenerateParameters" : true' <<< "$LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"canReferenceVLMModelFactory" : true' <<< "$LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"canLoadLocalModelContainer" : true' <<< "$LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"requestModel" : "verify"' <<< "$LIGHTWEIGHT_PIPELINE_JSON"
