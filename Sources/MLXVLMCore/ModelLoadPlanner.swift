import Foundation

public struct ModelLoadPlan: Codable, Equatable, Sendable {
    public let descriptor: ModelDescriptor
    public let normalizedConfig: JSONValue
    public let qwenVLConfig: QwenVLModelConfig?
    public let qwenVLArchitecture: QwenVLArchitecturePlan?
    public let capabilities: ModelCapabilityPlan
    public let tokenizerCatalog: TokenizerCatalog?
    public let tokenizerPlan: TokenizerImplementationPlan
    public let chatTemplatePlan: ChatTemplatePlan
    public let configNormalization: ModelConfigNormalizationPlan
    public let adapterMetadata: AdapterMetadata
    public let weightCatalog: WeightCatalog
    public let weightDataCatalog: WeightDataCatalog
    public let memoryEstimate: ModelMemoryEstimate
    public let compatibilityChecks: [CompatibilityCheck]
    public let metadataReady: Bool
    public let generationReady: Bool
    public let canLoadMetadata: Bool
    public let canAttemptGeneration: Bool
    public let blockingReasons: [String]

    public init(
        descriptor: ModelDescriptor,
        normalizedConfig: JSONValue,
        qwenVLConfig: QwenVLModelConfig?,
        qwenVLArchitecture: QwenVLArchitecturePlan?,
        capabilities: ModelCapabilityPlan,
        tokenizerCatalog: TokenizerCatalog?,
        tokenizerPlan: TokenizerImplementationPlan,
        chatTemplatePlan: ChatTemplatePlan,
        configNormalization: ModelConfigNormalizationPlan,
        adapterMetadata: AdapterMetadata,
        weightCatalog: WeightCatalog,
        weightDataCatalog: WeightDataCatalog,
        memoryEstimate: ModelMemoryEstimate,
        compatibilityChecks: [CompatibilityCheck],
        metadataReady: Bool,
        generationReady: Bool,
        canLoadMetadata: Bool,
        canAttemptGeneration: Bool,
        blockingReasons: [String]
    ) {
        self.descriptor = descriptor
        self.normalizedConfig = normalizedConfig
        self.qwenVLConfig = qwenVLConfig
        self.qwenVLArchitecture = qwenVLArchitecture
        self.capabilities = capabilities
        self.tokenizerCatalog = tokenizerCatalog
        self.tokenizerPlan = tokenizerPlan
        self.chatTemplatePlan = chatTemplatePlan
        self.configNormalization = configNormalization
        self.adapterMetadata = adapterMetadata
        self.weightCatalog = weightCatalog
        self.weightDataCatalog = weightDataCatalog
        self.memoryEstimate = memoryEstimate
        self.compatibilityChecks = compatibilityChecks
        self.metadataReady = metadataReady
        self.generationReady = generationReady
        self.canLoadMetadata = canLoadMetadata
        self.canAttemptGeneration = canAttemptGeneration
        self.blockingReasons = blockingReasons
    }
}

public struct ModelLoadPlanner {
    public let backend: BackendStatus

    public init(backend: BackendStatus = .compatibilityShell) {
        self.backend = backend
    }

    public func plan(descriptor: ModelDescriptor) -> ModelLoadPlan {
        let compatibility = ModelCompatibilityValidator.validate(descriptor: descriptor, backend: backend)
        let capabilities = ModelCapabilityPlanner().plan(descriptor: descriptor)
        let weightCatalog = WeightCatalogBuilder().catalog(for: descriptor)
        let weightDataCatalog = WeightDataCatalogBuilder().catalog(for: descriptor)
        let tokenizerCatalog = TokenizerCatalogBuilder().catalog(for: descriptor)
        let tokenizerPlan = TokenizerImplementationPlanner().plan(
            descriptor: descriptor,
            catalog: tokenizerCatalog
        )
        let chatTemplatePlan = ChatTemplatePlanner().plan(descriptor: descriptor)
        let normalizedConfig = (try? ModelStore().loadNormalizedConfig(pathOrIdentifier: descriptor.path)) ?? .object([:])
        let qwenConfig = loadQwenConfigIfSupported(descriptor: descriptor)
        let qwenArchitecture = qwenConfig.map {
            QwenVLArchitecturePlanner().plan(config: $0, weightCatalog: weightCatalog)
        }
        let memoryEstimate = ModelMemoryEstimator().estimate(
            descriptor: descriptor,
            weightDataCatalog: weightDataCatalog,
            qwenVLConfig: qwenConfig
        )
        let blockingReasons = blockingReasons(
            compatibility: compatibility,
            capabilities: capabilities,
            qwenConfig: qwenConfig,
            descriptor: descriptor
        )

        return ModelLoadPlan(
            descriptor: descriptor,
            normalizedConfig: normalizedConfig,
            qwenVLConfig: qwenConfig,
            qwenVLArchitecture: qwenArchitecture,
            capabilities: capabilities,
            tokenizerCatalog: tokenizerCatalog,
            tokenizerPlan: tokenizerPlan,
            chatTemplatePlan: chatTemplatePlan,
            configNormalization: descriptor.configNormalization,
            adapterMetadata: descriptor.adapterMetadata,
            weightCatalog: weightCatalog,
            weightDataCatalog: weightDataCatalog,
            memoryEstimate: memoryEstimate,
            compatibilityChecks: compatibility.checks,
            metadataReady: compatibility.metadataReady,
            generationReady: compatibility.generationReady,
            canLoadMetadata: compatibility.metadataReady && (!isQwenVL(descriptor) || qwenConfig != nil),
            canAttemptGeneration: compatibility.generationReady && blockingReasons.isEmpty,
            blockingReasons: blockingReasons
        )
    }

    private func loadQwenConfigIfSupported(descriptor: ModelDescriptor) -> QwenVLModelConfig? {
        guard isQwenVL(descriptor) else {
            return nil
        }
        return try? QwenVLModelConfig.load(fromModelDirectory: descriptor.path)
    }

    private func blockingReasons(
        compatibility: ModelCompatibilityReport,
        capabilities: ModelCapabilityPlan,
        qwenConfig: QwenVLModelConfig?,
        descriptor: ModelDescriptor
    ) -> [String] {
        var reasons = compatibility.checks
            .filter { $0.severity == .error && !$0.passed }
            .map(\.message)

        if isQwenVL(descriptor), qwenConfig == nil
        {
            reasons.append("Qwen VL config could not be parsed for this model directory.")
        }
        if !capabilities.supportsOllamaGenerationAPI {
            reasons.append("Model type \(descriptor.canonicalModelType) uses \(capabilities.primaryTask) and is not compatible with text generation endpoints.")
        }

        return reasons
    }

    private func isQwenVL(_ descriptor: ModelDescriptor) -> Bool {
        descriptor.canonicalModelType == "qwen2_vl" || descriptor.canonicalModelType == "qwen2_5_vl"
    }
}
