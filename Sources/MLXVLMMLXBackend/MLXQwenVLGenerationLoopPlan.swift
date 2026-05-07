import Foundation
import MLXVLMCore

public struct MLXQwenVLGenerationLoopPlan: Codable, Equatable, Sendable {
    public let modelID: String
    public let canonicalModelType: String
    public let forwardReady: Bool
    public let generationLoopReady: Bool
    public let promptTokenCount: Int
    public let maxCompletionTokens: Int
    public let stream: Bool
    public let stopSequenceCount: Int
    public let sampler: String
    public let deterministic: Bool
    public let generateParameters: MLXGenerateParametersPlan
    public let requiredSamplerFeatures: [String]
    public let requiresAdvancedSampler: Bool
    public let assemblerHandoffReady: Bool
    public let expectedFirstStep: String
    public let expectedLoop: String
    public let blockingReasons: [String]

    public init(forwardPlan: MLXQwenVLForwardPlan, input: ProcessedGenerationInput) {
        let sampling = input.runtime.sampling
        self.modelID = forwardPlan.modelID
        self.canonicalModelType = forwardPlan.canonicalModelType
        self.forwardReady = forwardPlan.forwardReady
        self.promptTokenCount = forwardPlan.promptTokenCount
        self.maxCompletionTokens = input.runtime.maxCompletionTokens
        self.stream = input.runtime.stream
        self.stopSequenceCount = input.runtime.stopSequences.count
        self.sampler = sampling.sampler
        self.deterministic = sampling.deterministic
        self.generateParameters = MLXGenerateParametersPlan(runtime: input.runtime)
        self.requiredSamplerFeatures = sampling.backendMinimumFeatures
        self.requiresAdvancedSampler = sampling.requiresAdvancedSampler
        self.assemblerHandoffReady = forwardPlan.promptTokenCount > 0
        self.expectedFirstStep = "prefill"
        self.expectedLoop = "decode-next-token-until-stop-or-length"

        var reasons = forwardPlan.blockingReasons + sampling.warnings + generateParameters.warnings
        if !forwardPlan.forwardReady {
            reasons.append("Forward pass must be ready before generation loop can run.")
        }
        if input.runtime.maxCompletionTokens <= 0 {
            reasons.append("maxCompletionTokens must be positive before generation loop can run.")
        }
        if !generateParameters.canInstantiateGenerateParameters {
            reasons.append("mlx-swift-lm GenerateParameters cannot be instantiated for this request yet.")
        }
        if sampling.requiresAdvancedSampler {
            reasons.append("Advanced sampler features must be implemented before this request can run.")
        }
        reasons.append("MLX-backed decode loop is not implemented yet.")
        self.blockingReasons = Self.unique(reasons)
        self.generationLoopReady = forwardPlan.forwardReady &&
            input.runtime.maxCompletionTokens > 0 &&
            generateParameters.canInstantiateGenerateParameters &&
            !sampling.requiresAdvancedSampler &&
            false
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

public struct MLXQwenVLGenerationLoopPlanner {
    public init() {}

    public func plan(
        forwardPlan: MLXQwenVLForwardPlan,
        input: ProcessedGenerationInput
    ) -> MLXQwenVLGenerationLoopPlan {
        MLXQwenVLGenerationLoopPlan(forwardPlan: forwardPlan, input: input)
    }
}
