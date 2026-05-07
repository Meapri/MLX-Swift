import Foundation
import MLXVLMCore

public struct MLXWeightBackedModelContainerSummary: Codable, Equatable, Sendable {
    public let modelID: String
    public let canonicalModelType: String
    public let metadataReady: Bool
    public let preparedWeightTensorCount: Int
    public let loadedArrayCount: Int
    public let arrayBacked: Bool
    public let moduleInstantiationReady: Bool
    public let generationReady: Bool
    public let blockingReasons: [String]

    public init(
        context: ModelLoadContext,
        preflight: MLXBackendLoadPreflightReport
    ) {
        self.modelID = context.descriptor.id
        self.canonicalModelType = context.descriptor.canonicalModelType
        self.metadataReady = preflight.metadataReady
        self.preparedWeightTensorCount = preflight.preparedWeights?.tensorCount ?? 0
        self.loadedArrayCount = preflight.arrayLoadReport?.loadedArrayCount ?? 0
        self.arrayBacked = preflight.canCreateMLXArrays
        self.moduleInstantiationReady = preflight.canInstantiateModelModules
        self.generationReady = preflight.canRunGeneration
        self.blockingReasons = preflight.blockingReasons
    }
}

public struct MLXWeightBackedModelContainer: ModelContainer {
    public let context: ModelLoadContext
    public let preflight: MLXBackendLoadPreflightReport
    public let summary: MLXWeightBackedModelContainerSummary

    public init(context: ModelLoadContext, preflight: MLXBackendLoadPreflightReport) {
        self.context = context
        self.preflight = preflight
        self.summary = MLXWeightBackedModelContainerSummary(
            context: context,
            preflight: preflight
        )
    }
}

