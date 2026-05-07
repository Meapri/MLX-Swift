import Foundation
import MLXVLMCore

#if MLXVLM_REAL_MLX_API && canImport(MLXLMCommon)
import MLXLMCommon
#endif

public struct MLXGenerateParametersBridgeReport: Codable, Equatable, Sendable {
    public let sourceAPI: String
    public let realMLXAPIImplementationCompiled: Bool
    public let canImportMLXLMCommon: Bool
    public let canBridgeToGenerateParameters: Bool
    public let prefillStepSize: Int?
    public let maxTokens: Int?
    public let maxKVSize: Int?
    public let kvBits: Int?
    public let kvGroupSize: Int?
    public let quantizedKVStart: Int?
    public let temperature: Float?
    public let topP: Float?
    public let topK: Int?
    public let minP: Float?
    public let repetitionPenalty: Float?
    public let repetitionContextSize: Int?
    public let presencePenalty: Float?
    public let presenceContextSize: Int?
    public let frequencyPenalty: Float?
    public let frequencyContextSize: Int?
    public let stopSequenceCount: Int
    public let externalRuntimeResponsibilities: [String]
    public let blockingReasons: [String]

    public init(
        plan: MLXGenerateParametersPlan,
        probe: MLXRuntimeProbe = MLXRuntimeProbe()
    ) {
        self.sourceAPI = plan.sourceAPI
        self.realMLXAPIImplementationCompiled = probe.realMLXAPIImplementationCompiled
        self.canImportMLXLMCommon = probe.canImportMLXLMCommon
        self.stopSequenceCount = plan.stopSequences.count
        self.externalRuntimeResponsibilities = Self.externalRuntimeResponsibilities(for: plan)

        #if MLXVLM_REAL_MLX_API && canImport(MLXLMCommon)
        if plan.canInstantiateGenerateParameters {
            let parameters = MLXGenerateParametersBridge.makeGenerateParameters(from: plan)
            self.canBridgeToGenerateParameters = true
            self.prefillStepSize = parameters.prefillStepSize
            self.maxTokens = parameters.maxTokens
            self.maxKVSize = parameters.maxKVSize
            self.kvBits = parameters.kvBits
            self.kvGroupSize = parameters.kvGroupSize
            self.quantizedKVStart = parameters.quantizedKVStart
            self.temperature = parameters.temperature
            self.topP = parameters.topP
            self.topK = parameters.topK
            self.minP = parameters.minP
            self.repetitionPenalty = parameters.repetitionPenalty
            self.repetitionContextSize = parameters.repetitionContextSize
            self.presencePenalty = parameters.presencePenalty
            self.presenceContextSize = parameters.presenceContextSize
            self.frequencyPenalty = parameters.frequencyPenalty
            self.frequencyContextSize = parameters.frequencyContextSize
            self.blockingReasons = []
        } else {
            self.canBridgeToGenerateParameters = false
            self.prefillStepSize = nil
            self.maxTokens = nil
            self.maxKVSize = nil
            self.kvBits = nil
            self.kvGroupSize = nil
            self.quantizedKVStart = nil
            self.temperature = nil
            self.topP = nil
            self.topK = nil
            self.minP = nil
            self.repetitionPenalty = nil
            self.repetitionContextSize = nil
            self.presencePenalty = nil
            self.presenceContextSize = nil
            self.frequencyPenalty = nil
            self.frequencyContextSize = nil
            self.blockingReasons = ["GenerateParameters plan contains values that cannot be represented by mlx-swift-lm."]
        }
        #else
        self.canBridgeToGenerateParameters = false
        self.prefillStepSize = nil
        self.maxTokens = nil
        self.maxKVSize = nil
        self.kvBits = nil
        self.kvGroupSize = nil
        self.quantizedKVStart = nil
        self.temperature = nil
        self.topP = nil
        self.topK = nil
        self.minP = nil
        self.repetitionPenalty = nil
        self.repetitionContextSize = nil
        self.presencePenalty = nil
        self.presenceContextSize = nil
        self.frequencyPenalty = nil
        self.frequencyContextSize = nil
        self.blockingReasons = ["MLXLMCommon.GenerateParameters is not available in this build."]
        #endif
    }

    private static func externalRuntimeResponsibilities(for plan: MLXGenerateParametersPlan) -> [String] {
        var responsibilities = [
            "stop-sequence matching",
            "eos-token matching",
        ]
        if plan.stopSequences.isEmpty {
            responsibilities.removeAll { $0 == "stop-sequence matching" }
        }
        if !plan.unsupportedRuntimeOptions.isEmpty {
            let unsupportedOptions = plan.unsupportedRuntimeOptions.joined(separator: ",")
            responsibilities.append("unsupported runtime options: \(unsupportedOptions)")
        }
        if plan.warnings.contains(where: { $0.contains("Seed handling") }) {
            responsibilities.append("seed application through MLX random state")
        }
        return responsibilities
    }
}

public struct MLXGenerateParametersBridge: Sendable {
    public init() {}

    public func report(for plan: MLXGenerateParametersPlan) -> MLXGenerateParametersBridgeReport {
        MLXGenerateParametersBridgeReport(plan: plan)
    }
}

#if MLXVLM_REAL_MLX_API && canImport(MLXLMCommon)
extension MLXGenerateParametersBridge {
    public static func makeGenerateParameters(from plan: MLXGenerateParametersPlan) -> GenerateParameters {
        GenerateParameters(
            maxTokens: plan.maxTokens,
            maxKVSize: plan.maxKVSize,
            kvBits: plan.kvBits,
            kvGroupSize: plan.kvGroupSize,
            quantizedKVStart: plan.quantizedKVStart,
            temperature: plan.temperature,
            topP: plan.topP,
            topK: plan.topK,
            minP: plan.minP,
            repetitionPenalty: plan.repetitionPenalty,
            repetitionContextSize: plan.repetitionContextSize,
            presencePenalty: plan.presencePenalty,
            presenceContextSize: plan.presenceContextSize,
            frequencyPenalty: plan.frequencyPenalty,
            frequencyContextSize: plan.frequencyContextSize,
            prefillStepSize: plan.prefillStepSize
        )
    }
}
#endif
