import Foundation
import MLXVLMCore

#if MLXVLM_REAL_MLX_API && canImport(MLXLMCommon) && canImport(MLXVLM) && canImport(MLXLMTokenizers)
import MLXLMCommon
import MLXLMTokenizers
import MLXVLM
#endif

public struct MLXVLMModelFactoryBridgeReport: Codable, Equatable, Sendable {
    public let sourceAPI: String
    public let localModelDirectory: String
    public let modelID: String
    public let canonicalModelType: String
    public let realMLXAPIImplementationCompiled: Bool
    public let canImportMLXLMCommon: Bool
    public let canImportMLXVLM: Bool
    public let canImportMLXLMTokenizers: Bool
    public let canReferenceVLMModelFactory: Bool
    public let canReferenceModelContainer: Bool
    public let canReferenceGenerateParameters: Bool
    public let canReferenceTokenizersLoader: Bool
    public let canReferenceGenerationEntrypoints: Bool
    public let canLoadLocalModelContainer: Bool
    public let requiredRuntimeInputs: [String]
    public let blockingReasons: [String]

    public init(
        descriptor: ModelDescriptor,
        probe: MLXRuntimeProbe = MLXRuntimeProbe()
    ) {
        self.sourceAPI = "MLXVLM.VLMModelFactory + MLXLMCommon.ModelContainer"
        self.localModelDirectory = descriptor.path
        self.modelID = descriptor.id
        self.canonicalModelType = descriptor.canonicalModelType
        self.realMLXAPIImplementationCompiled = probe.realMLXAPIImplementationCompiled
        self.canImportMLXLMCommon = probe.canImportMLXLMCommon
        self.canImportMLXVLM = probe.canImportMLXVLM
        self.canImportMLXLMTokenizers = probe.canImportMLXLMTokenizers
        self.requiredRuntimeInputs = [
            "local model directory",
            "TokenizerLoader implementation",
            "config.json model_type",
            "processor_config.json or preprocessor_config.json",
            "tokenizer.json/tokenizer_config.json",
            "safetensors weights",
        ]

        #if MLXVLM_REAL_MLX_API && canImport(MLXLMCommon) && canImport(MLXVLM) && canImport(MLXLMTokenizers)
        self.canReferenceVLMModelFactory = MLXVLMModelFactoryBridge.canReferenceVLMModelFactory()
        self.canReferenceModelContainer = MLXVLMModelFactoryBridge.canReferenceModelContainer()
        self.canReferenceGenerateParameters = MLXVLMModelFactoryBridge.canReferenceGenerateParameters()
        self.canReferenceTokenizersLoader = MLXVLMModelFactoryBridge.canReferenceTokenizersLoader()
        self.canReferenceGenerationEntrypoints = self.canReferenceModelContainer &&
            self.canReferenceGenerateParameters &&
            self.canReferenceTokenizersLoader
        self.canLoadLocalModelContainer = true
        self.blockingReasons = []
        #else
        self.canReferenceVLMModelFactory = false
        self.canReferenceModelContainer = false
        self.canReferenceGenerateParameters = false
        self.canReferenceTokenizersLoader = false
        self.canReferenceGenerationEntrypoints = false
        self.canLoadLocalModelContainer = false
        self.blockingReasons = [
            "MLXVLM.VLMModelFactory, MLXLMCommon.ModelContainer, and MLXLMTokenizers.TokenizersLoader are not available in this build."
        ]
        #endif
    }
}

public struct MLXVLMModelFactoryBridge: Sendable {
    public init() {}

    public func report(for descriptor: ModelDescriptor) -> MLXVLMModelFactoryBridgeReport {
        MLXVLMModelFactoryBridgeReport(descriptor: descriptor)
    }
}

#if MLXVLM_REAL_MLX_API && canImport(MLXLMCommon) && canImport(MLXVLM) && canImport(MLXLMTokenizers)
extension MLXVLMModelFactoryBridge {
    public static func canReferenceVLMModelFactory() -> Bool {
        _ = MLXVLM.VLMModelFactory.shared
        return true
    }

    public static func canReferenceModelContainer() -> Bool {
        _ = MLXLMCommon.ModelContainer.self
        return true
    }

    public static func canReferenceGenerateParameters() -> Bool {
        _ = MLXLMCommon.GenerateParameters.self
        return true
    }

    public static func canReferenceTokenizersLoader() -> Bool {
        _ = MLXLMTokenizers.TokenizersLoader.self
        return true
    }
}
#endif
