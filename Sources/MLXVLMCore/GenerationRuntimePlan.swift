import Foundation

public struct GenerationRuntimePlan: Codable, Equatable, Sendable {
    public let requestedContextLength: Int?
    public let modelMaxPositionEmbeddings: Int?
    public let effectiveContextLength: Int?
    public let promptTokenCount: Int?
    public let maxCompletionTokens: Int
    public let totalTokenBudget: Int?
    public let exceedsContextLength: Bool?
    public let useCache: Bool?
    public let kvBits: Double?
    public let kvQuantizationScheme: String?
    public let kvGroupSize: Int?
    public let maxKVSize: Int?
    public let visionCacheSize: Int?
    public let quantizeActivations: Bool?
    public let keepAlive: String?
    public let stopSequences: [String]
    public let stream: Bool
    public let sampling: GenerationSamplingPlan

    public init(
        requestedContextLength: Int?,
        modelMaxPositionEmbeddings: Int?,
        effectiveContextLength: Int?,
        promptTokenCount: Int?,
        maxCompletionTokens: Int,
        totalTokenBudget: Int?,
        exceedsContextLength: Bool?,
        useCache: Bool?,
        kvBits: Double?,
        kvQuantizationScheme: String?,
        kvGroupSize: Int?,
        maxKVSize: Int?,
        visionCacheSize: Int?,
        quantizeActivations: Bool?,
        keepAlive: String?,
        stopSequences: [String],
        stream: Bool,
        sampling: GenerationSamplingPlan
    ) {
        self.requestedContextLength = requestedContextLength
        self.modelMaxPositionEmbeddings = modelMaxPositionEmbeddings
        self.effectiveContextLength = effectiveContextLength
        self.promptTokenCount = promptTokenCount
        self.maxCompletionTokens = maxCompletionTokens
        self.totalTokenBudget = totalTokenBudget
        self.exceedsContextLength = exceedsContextLength
        self.useCache = useCache
        self.kvBits = kvBits
        self.kvQuantizationScheme = kvQuantizationScheme
        self.kvGroupSize = kvGroupSize
        self.maxKVSize = maxKVSize
        self.visionCacheSize = visionCacheSize
        self.quantizeActivations = quantizeActivations
        self.keepAlive = keepAlive
        self.stopSequences = stopSequences
        self.stream = stream
        self.sampling = sampling
    }

    public init(
        request: GenerationRequest,
        qwenVLConfig: QwenVLModelConfig?,
        tokenIDs: [Int]?
    ) {
        let modelLimit = qwenVLConfig?.textConfig.maxPositionEmbeddings
        let requestedLimit = request.parameters.contextLength
        let effectiveLimit: Int?
        if let requestedLimit, let modelLimit {
            effectiveLimit = min(requestedLimit, modelLimit)
        } else {
            effectiveLimit = requestedLimit ?? modelLimit
        }

        let promptTokenCount = tokenIDs?.count
        let totalTokenBudget = promptTokenCount.map { $0 + request.parameters.maxTokens }
        let exceedsContextLength: Bool?
        if let totalTokenBudget, let effectiveLimit {
            exceedsContextLength = totalTokenBudget > effectiveLimit
        } else {
            exceedsContextLength = nil
        }

        self.init(
            requestedContextLength: requestedLimit,
            modelMaxPositionEmbeddings: modelLimit,
            effectiveContextLength: effectiveLimit,
            promptTokenCount: promptTokenCount,
            maxCompletionTokens: request.parameters.maxTokens,
            totalTokenBudget: totalTokenBudget,
            exceedsContextLength: exceedsContextLength,
            useCache: qwenVLConfig?.textConfig.useCache,
            kvBits: request.parameters.kvBits,
            kvQuantizationScheme: request.parameters.kvQuantizationScheme,
            kvGroupSize: request.parameters.kvGroupSize,
            maxKVSize: request.parameters.maxKVSize,
            visionCacheSize: request.parameters.visionCacheSize,
            quantizeActivations: request.parameters.quantizeActivations,
            keepAlive: request.parameters.keepAlive,
            stopSequences: request.parameters.stopSequences,
            stream: request.stream,
            sampling: GenerationSamplingPlanner().plan(parameters: request.parameters)
        )
    }
}
