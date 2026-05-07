import Foundation
import MLXVLMCore

public struct MLXBackendAvailability: Codable, Equatable, Sendable {
    public let dependencyPlan: BackendDependencyPlan
    public let runtimeProbe: MLXRuntimeProbe
    public let canCreateBackend: Bool
    public let blockingReasons: [String]

    public init(
        dependencyPlan: BackendDependencyPlan,
        runtimeProbe: MLXRuntimeProbe = MLXRuntimeProbe()
    ) {
        self.dependencyPlan = dependencyPlan
        self.runtimeProbe = runtimeProbe
        self.canCreateBackend = (dependencyPlan.canEnableMLXBackend || runtimeProbe.realMLXAPIImplementationCompiled) &&
            runtimeProbe.canImportRequiredModules &&
            runtimeProbe.backendImplementationReady
        var reasons = dependencyPlan.nextSteps
        if !runtimeProbe.canImportRequiredModules {
            reasons.append("Build MLXVLMMLXBackend with importable MLX, MLXLMCommon, MLXLLM, MLXVLM, and MLXLMTokenizers modules.")
        }
        if !runtimeProbe.backendImplementationReady {
            reasons.append("Enable the upstream mlx-swift-lm adapter first; add Swift-owned model modules only for Python mlx-vlm compatibility gaps that upstream does not cover.")
        }
        self.blockingReasons = reasons
    }
}

public enum MLXBackendFactoryError: Error, CustomStringConvertible, Sendable {
    case dependenciesUnavailable(MLXBackendAvailability)
    case asyncLoadRequired

    public var description: String {
        switch self {
        case .dependenciesUnavailable(let availability):
            return availability.blockingReasons.joined(separator: " ")
        case .asyncLoadRequired:
            return "MLX-backed VLM containers are loaded asynchronously; call makeBackendAsync(descriptor:rootPath:) instead."
        }
    }
}

public enum MLXBackendFactory {
    public static func availability(rootPath: String = FileManager.default.currentDirectoryPath) -> MLXBackendAvailability {
        MLXBackendAvailability(
            dependencyPlan: BackendDependencyPlanner().plan(rootPath: rootPath),
            runtimeProbe: MLXRuntimeProbe()
        )
    }

    public static func preflightLoad(
        descriptor: ModelDescriptor,
        rootPath: String = FileManager.default.currentDirectoryPath,
        weightOptions: MLXWeightPreparationOptions = MLXWeightPreparationOptions()
    ) -> MLXBackendLoadPreflightReport {
        let availability = availability(rootPath: rootPath)
        let loadPlan = ModelLoadPlanner().plan(descriptor: descriptor)
        let bindingPlan = MLXBackendBindingPlan(descriptor: descriptor, loadPlan: loadPlan)
        let preparedResult = Result {
            try MLXWeightPreparer().prepare(descriptor: descriptor, options: weightOptions)
        }
        let preparedWeights: MLXPreparedWeightBundleSummary?
        let preparedWeightError: String?
        let arrayLoadReport: MLXArrayWeightLoadReport?
        switch preparedResult {
        case .success(let bundle):
            preparedWeights = bundle.summary
            preparedWeightError = nil
            arrayLoadReport = MLXArrayWeightLoading.loadReport(bundle: bundle)
        case .failure(let error):
            preparedWeights = nil
            preparedWeightError = String(describing: error)
            arrayLoadReport = nil
        }

        return MLXBackendLoadPreflightReport(
            descriptor: descriptor,
            availability: availability,
            loadPlan: loadPlan,
            bindingPlan: bindingPlan,
            preparedWeights: preparedWeights,
            preparedWeightError: preparedWeightError,
            arrayLoadReport: arrayLoadReport
        )
    }

    public static func loadWeightBackedContainer(
        descriptor: ModelDescriptor,
        rootPath: String = FileManager.default.currentDirectoryPath,
        weightOptions: MLXWeightPreparationOptions = MLXWeightPreparationOptions()
    ) -> MLXWeightBackedModelContainer {
        let loadPlan = ModelLoadPlanner().plan(descriptor: descriptor)
        let context = ModelLoadContext(descriptor: descriptor, loadPlan: loadPlan)
        let preflight = preflightLoad(
            descriptor: descriptor,
            rootPath: rootPath,
            weightOptions: weightOptions
        )
        return MLXWeightBackedModelContainer(context: context, preflight: preflight)
    }

    public static func makeBackend(
        descriptor: ModelDescriptor,
        rootPath: String = FileManager.default.currentDirectoryPath
    ) throws -> any VLMBackend {
        let availability = availability(rootPath: rootPath)
        guard availability.canCreateBackend else {
            throw MLXBackendFactoryError.dependenciesUnavailable(availability)
        }

        throw MLXBackendFactoryError.asyncLoadRequired
    }

    public static func makeBackendAsync(
        descriptor: ModelDescriptor,
        rootPath: String = FileManager.default.currentDirectoryPath
    ) async throws -> any VLMBackend {
        let availability = availability(rootPath: rootPath)
        guard availability.canCreateBackend else {
            throw MLXBackendFactoryError.dependenciesUnavailable(availability)
        }

        _ = MLXMetalLibrarySupport.ensureDefaultLibraryAvailable()

        #if MLXVLM_REAL_MLX_API && canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXVLM) && canImport(MLXLMTokenizers)
        return try await MLXVLMUpstreamBackend.load(descriptor: descriptor)
        #else
        throw MLXBackendFactoryError.dependenciesUnavailable(availability)
        #endif
    }

    public static func makeEmbeddingBackendAsync(
        descriptor: ModelDescriptor,
        rootPath: String = FileManager.default.currentDirectoryPath
    ) async throws -> any EmbeddingBackend {
        let availability = availability(rootPath: rootPath)
        guard availability.canCreateBackend else {
            throw MLXBackendFactoryError.dependenciesUnavailable(availability)
        }

        _ = MLXMetalLibrarySupport.ensureDefaultLibraryAvailable()

        #if MLXVLM_REAL_MLX_API && canImport(MLXEmbedders) && canImport(MLXLMTokenizers)
        return try await MLXVLMUpstreamEmbeddingBackend.load(descriptor: descriptor)
        #else
        throw MLXVLMUpstreamEmbeddingBackendError.unavailable
        #endif
    }
}
