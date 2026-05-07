import Foundation

public struct ModelMemoryEstimate: Codable, Equatable, Sendable {
    public let weightFileBytes: Int64
    public let readableWeightBytes: Int64
    public let unreadableTensorCount: Int
    public let quantizationSummary: String?
    public let kvCacheTokenCapacity: Int?
    public let kvCacheElementBits: Double?
    public let estimatedKVCacheBytes: Int64?
    public let visionCacheEntryCount: Int?
    public let estimatedVisionCacheBytes: Int64?
    public let estimatedTotalRuntimeBytes: Int64
    public let assumptions: [String]

    public init(
        weightFileBytes: Int64,
        readableWeightBytes: Int64,
        unreadableTensorCount: Int,
        quantizationSummary: String?,
        kvCacheTokenCapacity: Int?,
        kvCacheElementBits: Double?,
        estimatedKVCacheBytes: Int64?,
        visionCacheEntryCount: Int?,
        estimatedVisionCacheBytes: Int64?,
        assumptions: [String]
    ) {
        self.weightFileBytes = weightFileBytes
        self.readableWeightBytes = readableWeightBytes
        self.unreadableTensorCount = unreadableTensorCount
        self.quantizationSummary = quantizationSummary
        self.kvCacheTokenCapacity = kvCacheTokenCapacity
        self.kvCacheElementBits = kvCacheElementBits
        self.estimatedKVCacheBytes = estimatedKVCacheBytes
        self.visionCacheEntryCount = visionCacheEntryCount
        self.estimatedVisionCacheBytes = estimatedVisionCacheBytes
        self.estimatedTotalRuntimeBytes = readableWeightBytes
            + (estimatedKVCacheBytes ?? 0)
            + (estimatedVisionCacheBytes ?? 0)
        self.assumptions = assumptions
    }
}

public struct ModelMemoryEstimator {
    public init() {}

    public func estimate(
        descriptor: ModelDescriptor,
        weightDataCatalog: WeightDataCatalog,
        qwenVLConfig: QwenVLModelConfig?,
        parameters: GenerationParameters = GenerationParameters()
    ) -> ModelMemoryEstimate {
        var assumptions: [String] = [
            "Weight bytes are safetensors payload bytes, not Metal allocation overhead.",
            "Runtime estimates exclude tokenizer, HTTP buffers, temporary activations, and allocator fragmentation.",
        ]

        let kvTokenCapacity = tokenCapacity(config: qwenVLConfig, parameters: parameters)
        let kvElementBits = parameters.kvBits ?? 16
        let estimatedKVBytes = qwenVLConfig.flatMap { config in
            kvTokenCapacity.map { capacity in
                estimateKVCacheBytes(config: config, tokens: capacity, elementBits: kvElementBits)
            }
        }
        if parameters.kvBits != nil {
            assumptions.append("KV cache estimate uses requested quantized element width \(kvElementBits) bits.")
        } else {
            assumptions.append("KV cache estimate assumes 16-bit key/value cache elements.")
        }
        if parameters.maxKVSize != nil {
            assumptions.append("KV cache token capacity is capped by maxKVSize.")
        }

        let visionEntries = parameters.visionCacheSize
        let estimatedVisionBytes = qwenVLConfig.flatMap { config in
            visionEntries.map { entries in
                Int64(entries * config.textConfig.hiddenSize * 4)
            }
        }
        if visionEntries != nil {
            assumptions.append("Vision cache estimate assumes one projected feature vector per entry at Float32 width.")
        }

        return ModelMemoryEstimate(
            weightFileBytes: descriptor.totalWeightBytes,
            readableWeightBytes: weightDataCatalog.totalReadableBytes,
            unreadableTensorCount: weightDataCatalog.unreadableTensorCount,
            quantizationSummary: descriptor.quantizationMetadata?.summary,
            kvCacheTokenCapacity: kvTokenCapacity,
            kvCacheElementBits: qwenVLConfig == nil ? nil : kvElementBits,
            estimatedKVCacheBytes: estimatedKVBytes,
            visionCacheEntryCount: visionEntries,
            estimatedVisionCacheBytes: estimatedVisionBytes,
            assumptions: assumptions
        )
    }

    private func tokenCapacity(
        config: QwenVLModelConfig?,
        parameters: GenerationParameters
    ) -> Int? {
        guard let config else {
            return nil
        }
        let context = parameters.contextLength ?? config.textConfig.maxPositionEmbeddings
        if let maxKVSize = parameters.maxKVSize {
            return min(context, maxKVSize)
        }
        return context
    }

    private func estimateKVCacheBytes(
        config: QwenVLModelConfig,
        tokens: Int,
        elementBits: Double
    ) -> Int64 {
        let text = config.textConfig
        let headDim = text.hiddenSize / text.numAttentionHeads
        let elements = Double(text.numHiddenLayers)
            * Double(tokens)
            * Double(text.numKeyValueHeads)
            * Double(headDim)
            * 2.0
        return Int64((elements * elementBits / 8.0).rounded(.up))
    }
}
