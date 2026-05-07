import Foundation
import MLXVLMCore

public struct MLXQwenVLDecodeStatePlan: Codable, Equatable, Sendable {
    public let modelID: String
    public let canonicalModelType: String
    public let promptTokenCount: Int
    public let maxCompletionTokens: Int
    public let totalTokenLimit: Int
    public let initialPosition: Int
    public let nextTokenInputShape: [Int]?
    public let kvCacheEnabled: Bool
    public let kvBits: Double?
    public let kvQuantizationScheme: String?
    public let kvGroupSize: Int?
    public let maxKVSize: Int?
    public let visionCacheSize: Int?
    public let eosTokenIDs: [Int]
    public let stopSequences: [String]
    public let stopConditionOrder: [String]
    public let stream: Bool
    public let outputAssemblerReady: Bool
    public let canInitializeDecodeState: Bool
    public let blockingReasons: [String]

    public init(
        container: MLXWeightBackedModelContainer,
        forwardPlan: MLXQwenVLForwardPlan,
        generationLoopPlan: MLXQwenVLGenerationLoopPlan,
        input: ProcessedGenerationInput
    ) {
        let promptTokenCount = forwardPlan.promptTokenCount
        let maxCompletionTokens = input.runtime.maxCompletionTokens
        let eosTokenIDs = container.context.qwenVLConfig?.eosTokenIDs ?? []
        self.modelID = forwardPlan.modelID
        self.canonicalModelType = forwardPlan.canonicalModelType
        self.promptTokenCount = promptTokenCount
        self.maxCompletionTokens = maxCompletionTokens
        self.totalTokenLimit = promptTokenCount + max(0, maxCompletionTokens)
        self.initialPosition = promptTokenCount
        self.nextTokenInputShape = promptTokenCount > 0 ? [1, 1] : nil
        self.kvCacheEnabled = input.runtime.useCache ?? true
        self.kvBits = input.runtime.kvBits
        self.kvQuantizationScheme = input.runtime.kvQuantizationScheme
        self.kvGroupSize = input.runtime.kvGroupSize
        self.maxKVSize = input.runtime.maxKVSize
        self.visionCacheSize = input.runtime.visionCacheSize
        self.eosTokenIDs = eosTokenIDs
        self.stopSequences = input.runtime.stopSequences
        self.stopConditionOrder = Self.stopConditionOrder(
            eosTokenIDs: eosTokenIDs,
            stopSequences: input.runtime.stopSequences
        )
        self.stream = input.runtime.stream
        self.outputAssemblerReady = generationLoopPlan.assemblerHandoffReady

        var reasons: [String] = []
        if promptTokenCount <= 0 {
            reasons.append("Prompt token state is required before decode state can be initialized.")
        }
        if maxCompletionTokens <= 0 {
            reasons.append("maxCompletionTokens must be positive before decode state can be initialized.")
        }
        if !generationLoopPlan.assemblerHandoffReady {
            reasons.append("Generation output assembler handoff is required before decode can emit tokens.")
        }
        if eosTokenIDs.isEmpty {
            reasons.append("No EOS token IDs were found in the normalized Qwen VL config; decode must rely on length or stop sequences.")
        }
        self.blockingReasons = Self.unique(reasons)
        self.canInitializeDecodeState = promptTokenCount > 0 &&
            maxCompletionTokens > 0 &&
            generationLoopPlan.assemblerHandoffReady
    }

    private static func stopConditionOrder(
        eosTokenIDs: [Int],
        stopSequences: [String]
    ) -> [String] {
        var result: [String] = []
        if !eosTokenIDs.isEmpty {
            result.append("eos-token")
        }
        if !stopSequences.isEmpty {
            result.append("stop-sequence")
        }
        result.append("max-completion-tokens")
        return result
    }

    private static func unique(_ values: [String]) -> [String] {
        var seen: Set<String> = []
        var result: [String] = []
        for value in values where !value.isEmpty && !seen.contains(value) {
            seen.insert(value)
            result.append(value)
        }
        return result
    }
}

public struct MLXQwenVLDecodeStatePlanner {
    public init() {}

    public func plan(
        container: MLXWeightBackedModelContainer,
        forwardPlan: MLXQwenVLForwardPlan,
        generationLoopPlan: MLXQwenVLGenerationLoopPlan,
        input: ProcessedGenerationInput
    ) -> MLXQwenVLDecodeStatePlan {
        MLXQwenVLDecodeStatePlan(
            container: container,
            forwardPlan: forwardPlan,
            generationLoopPlan: generationLoopPlan,
            input: input
        )
    }
}
