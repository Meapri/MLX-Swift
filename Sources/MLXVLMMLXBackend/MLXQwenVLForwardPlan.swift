import Foundation
import MLXVLMCore

public struct MLXQwenVLForwardPlan: Codable, Equatable, Sendable {
    public let modelID: String
    public let canonicalModelType: String
    public let moduleConstructionReady: Bool
    public let forwardReady: Bool
    public let promptTokenCount: Int
    public let requiresTokenizerImplementation: Bool
    public let inputIDsShape: [Int]?
    public let pixelShape: [Int]?
    public let logitsShape: [Int]?
    public let nextTokenSelectionReady: Bool
    public let vocabSize: Int?
    public let effectiveContextLength: Int?
    public let maxCompletionTokens: Int
    public let blockingReasons: [String]

    public init(
        container: MLXWeightBackedModelContainer,
        modulePlan: MLXQwenVLModuleConstructionPlan,
        input: ProcessedGenerationInput
    ) {
        let tokenIDs = input.tokenIDs ?? []
        let promptTokenCount = tokenIDs.count
        let vocabSize = container.context.descriptor.configVocabSize
        let requiresTokenizerImplementation = input.preflight.tokenization?.requiresTokenizerImplementation ?? true
        self.modelID = container.context.descriptor.id
        self.canonicalModelType = container.context.descriptor.canonicalModelType
        self.moduleConstructionReady = modulePlan.moduleConstructionReady
        self.promptTokenCount = promptTokenCount
        self.requiresTokenizerImplementation = requiresTokenizerImplementation
        self.inputIDsShape = promptTokenCount > 0 ? [1, promptTokenCount] : nil
        self.pixelShape = input.pixelShape
        self.vocabSize = vocabSize
        self.effectiveContextLength = input.runtime.effectiveContextLength
        self.maxCompletionTokens = input.runtime.maxCompletionTokens
        self.logitsShape = if let vocabSize, promptTokenCount > 0 {
            [1, promptTokenCount, vocabSize]
        } else {
            nil
        }

        var reasons = modulePlan.blockingReasons + input.preflight.blockingReasons
        if !modulePlan.moduleConstructionReady {
            reasons.append("Qwen VL modules must be constructed before forward can run.")
        }
        if promptTokenCount == 0 {
            reasons.append("Token IDs are required before forward can run.")
        }
        if requiresTokenizerImplementation {
            reasons.append("A full tokenizer implementation is required before forward can run reliably.")
        }
        if vocabSize == nil {
            reasons.append("Config vocab_size is required to determine logits shape.")
        }
        if input.runtime.maxCompletionTokens <= 0 {
            reasons.append("At least one completion token is required for generation.")
        }
        self.blockingReasons = Self.unique(reasons)
        self.forwardReady = modulePlan.moduleConstructionReady &&
            promptTokenCount > 0 &&
            !requiresTokenizerImplementation &&
            vocabSize != nil &&
            input.runtime.maxCompletionTokens > 0
        self.nextTokenSelectionReady = forwardReady && logitsShape != nil
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

public struct MLXQwenVLForwardPlanner {
    public init() {}

    public func plan(
        container: MLXWeightBackedModelContainer,
        modulePlan: MLXQwenVLModuleConstructionPlan,
        input: ProcessedGenerationInput
    ) -> MLXQwenVLForwardPlan {
        MLXQwenVLForwardPlan(
            container: container,
            modulePlan: modulePlan,
            input: input
        )
    }
}
