import Foundation
import MLXVLMCore

public struct MLXBackendLoadPreflightReport: Codable, Equatable, Sendable {
    public let modelID: String
    public let canonicalModelType: String
    public let availability: MLXBackendAvailability
    public let metadataReady: Bool
    public let generationReady: Bool
    public let bindingPlan: MLXBackendBindingPlan
    public let preparedWeights: MLXPreparedWeightBundleSummary?
    public let preparedWeightError: String?
    public let arrayLoadReport: MLXArrayWeightLoadReport?
    public let canPrepareWeightPayloads: Bool
    public let canCreateMLXArrays: Bool
    public let canInstantiateModelModules: Bool
    public let canRunGeneration: Bool
    public let blockingReasons: [String]

    public init(
        descriptor: ModelDescriptor,
        availability: MLXBackendAvailability,
        loadPlan: ModelLoadPlan,
        bindingPlan: MLXBackendBindingPlan,
        preparedWeights: MLXPreparedWeightBundleSummary?,
        preparedWeightError: String?,
        arrayLoadReport: MLXArrayWeightLoadReport?
    ) {
        self.modelID = descriptor.id
        self.canonicalModelType = descriptor.canonicalModelType
        self.availability = availability
        self.metadataReady = loadPlan.metadataReady
        self.generationReady = loadPlan.generationReady
        self.bindingPlan = bindingPlan
        self.preparedWeights = preparedWeights
        self.preparedWeightError = preparedWeightError
        self.arrayLoadReport = arrayLoadReport
        self.canPrepareWeightPayloads = preparedWeights != nil
        self.canCreateMLXArrays = arrayLoadReport?.loadedAllRequestedArrays == true
        self.canInstantiateModelModules = canCreateMLXArrays && bindingPlan.missingRequiredKeys.isEmpty
        self.canRunGeneration = canInstantiateModelModules && loadPlan.canAttemptGeneration

        var reasons = availability.blockingReasons + loadPlan.blockingReasons + bindingPlan.blockingReasons
        if let preparedWeightError {
            reasons.append(preparedWeightError)
        }
        if let arrayLoadError = arrayLoadReport?.error {
            reasons.append(arrayLoadError)
        }
        if preparedWeights == nil {
            reasons.append("Prepared weight payloads are not available for MLX array creation.")
        }
        if !canCreateMLXArrays {
            reasons.append("MLXArray creation from safetensors payload bytes is not available yet.")
        }
        if !canInstantiateModelModules {
            reasons.append("MLX-backed Qwen VL module construction is not implemented yet.")
        }
        if !canRunGeneration {
            reasons.append("MLX-backed generation loop and sampling are not implemented yet.")
        }
        self.blockingReasons = Self.unique(reasons)
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
