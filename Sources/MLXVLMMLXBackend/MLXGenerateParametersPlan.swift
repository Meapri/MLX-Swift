import Foundation
import MLXVLMCore

public struct MLXGenerateParametersPlan: Codable, Equatable, Sendable {
    public let sourceAPI: String
    public let prefillStepSize: Int
    public let maxTokens: Int
    public let maxKVSize: Int?
    public let kvBits: Int?
    public let kvGroupSize: Int
    public let quantizedKVStart: Int
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let minP: Float
    public let repetitionPenalty: Float?
    public let repetitionContextSize: Int
    public let presencePenalty: Float?
    public let presenceContextSize: Int
    public let frequencyPenalty: Float?
    public let frequencyContextSize: Int
    public let stopSequences: [String]
    public let unsupportedRuntimeOptions: [String]
    public let warnings: [String]

    public init(runtime: GenerationRuntimePlan) {
        self.sourceAPI = "mlx-swift-lm GenerateParameters"
        self.prefillStepSize = 512
        self.maxTokens = runtime.maxCompletionTokens
        self.maxKVSize = runtime.maxKVSize
        self.kvBits = runtime.kvBits.flatMap(Self.integralInt)
        self.kvGroupSize = runtime.kvGroupSize ?? 64
        self.quantizedKVStart = 0
        self.temperature = Float(runtime.sampling.temperature)
        self.topP = Float(runtime.sampling.topP)
        self.topK = runtime.sampling.topK
        self.minP = Float(runtime.sampling.minP ?? 0)
        self.repetitionPenalty = runtime.sampling.repetitionPenalty.map(Float.init)
        self.repetitionContextSize = runtime.sampling.repeatLastN ?? 20
        self.presencePenalty = runtime.sampling.presencePenalty.map(Float.init)
        self.presenceContextSize = runtime.sampling.repeatLastN ?? 20
        self.frequencyPenalty = runtime.sampling.frequencyPenalty.map(Float.init)
        self.frequencyContextSize = runtime.sampling.repeatLastN ?? 20
        self.stopSequences = runtime.stopSequences

        var unsupported: [String] = []
        if runtime.sampling.sampler == "mirostat" {
            unsupported.append("mirostat")
        }
        if runtime.sampling.typicalP != nil {
            unsupported.append("typical_p")
        }
        if runtime.sampling.tfsZ != nil {
            unsupported.append("tfs_z")
        }
        if runtime.kvQuantizationScheme != nil {
            unsupported.append("kv_quantization_scheme")
        }
        if runtime.quantizeActivations != nil {
            unsupported.append("quantize_activations")
        }
        if runtime.visionCacheSize != nil {
            unsupported.append("vision_cache_size")
        }
        self.unsupportedRuntimeOptions = unsupported

        var warnings: [String] = []
        if runtime.kvBits != nil && kvBits == nil {
            warnings.append("mlx-swift-lm GenerateParameters expects integral kvBits.")
        }
        if runtime.sampling.seed != 0 {
            warnings.append("Seed handling is outside GenerateParameters and must be applied through MLX random state when the real backend is wired.")
        }
        if runtime.sampling.penalizeNewline != nil {
            warnings.append("Newline penalty is not represented directly in mlx-swift-lm GenerateParameters.")
        }
        if !unsupported.isEmpty {
            warnings.append("Some Ollama/OpenAI runtime options require separate backend support beyond mlx-swift-lm GenerateParameters.")
        }
        self.warnings = warnings
    }

    public var canInstantiateGenerateParameters: Bool {
        kvBits != nil || warnings.allSatisfy { !$0.contains("kvBits") }
    }

    private static func integralInt(_ value: Double) -> Int? {
        guard value.isFinite, value.rounded() == value else {
            return nil
        }
        return Int(value)
    }
}
