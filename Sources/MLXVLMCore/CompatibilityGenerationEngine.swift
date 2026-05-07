import Foundation

public struct GenerationUnavailableReport: Codable, Equatable, Sendable {
    public let error: String
    public let model: String
    public let canonicalModelType: String
    public let backend: BackendStatus
    public let preflight: GenerationPreflightPlan

    public init(
        error: String,
        model: String,
        canonicalModelType: String,
        backend: BackendStatus,
        preflight: GenerationPreflightPlan
    ) {
        self.error = error
        self.model = model
        self.canonicalModelType = canonicalModelType
        self.backend = backend
        self.preflight = preflight
    }
}

public struct CompatibilityGenerationEngine {
    public let descriptor: ModelDescriptor
    public let backend: CompatibilityVLMBackend

    public init(
        descriptor: ModelDescriptor,
        backend: BackendStatus = .compatibilityShell,
        mediaResolver: MediaReferenceResolver = MediaReferenceResolver(),
        imageGridConfig: QwenVLImageGridConfig = QwenVLImageGridConfig()
    ) {
        self.descriptor = descriptor
        self.backend = CompatibilityVLMBackend(
            descriptor: descriptor,
            status: backend,
            mediaResolver: mediaResolver,
            imageGridConfig: imageGridConfig
        )
    }

    public func unavailableReport(for request: GenerationRequest) -> GenerationUnavailableReport {
        backend.unavailableReport(for: request)
    }

    public func unavailableEmbeddingReport(for request: EmbeddingRequest) -> EmbeddingUnavailableReport {
        EmbeddingUnavailableReport(
            error: "Swift embedding backend is not wired yet.",
            model: descriptor.id,
            canonicalModelType: descriptor.canonicalModelType,
            backend: backend.status,
            request: request
        )
    }

    public func processedInput(for request: GenerationRequest) throws -> ProcessedGenerationInput {
        try backend.process(request)
    }
}
