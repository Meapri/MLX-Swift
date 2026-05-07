import Foundation

public struct ModelLoadContext: Codable, Equatable, Sendable {
    public let descriptor: ModelDescriptor
    public let loadPlan: ModelLoadPlan

    public init(descriptor: ModelDescriptor, loadPlan: ModelLoadPlan) {
        self.descriptor = descriptor
        self.loadPlan = loadPlan
    }

    public var qwenVLConfig: QwenVLModelConfig? {
        loadPlan.qwenVLConfig
    }

    public var normalizedConfig: JSONValue {
        loadPlan.normalizedConfig
    }

    public var qwenVLArchitecture: QwenVLArchitecturePlan? {
        loadPlan.qwenVLArchitecture
    }

    public var tokenizerCatalog: TokenizerCatalog? {
        loadPlan.tokenizerCatalog
    }

    public var tokenizerPlan: TokenizerImplementationPlan {
        loadPlan.tokenizerPlan
    }

    public var adapterMetadata: AdapterMetadata {
        loadPlan.adapterMetadata
    }

    public var capabilities: ModelCapabilityPlan {
        loadPlan.capabilities
    }

    public var weightCatalog: WeightCatalog {
        loadPlan.weightCatalog
    }

    public var weightDataCatalog: WeightDataCatalog {
        loadPlan.weightDataCatalog
    }

    public var memoryEstimate: ModelMemoryEstimate {
        loadPlan.memoryEstimate
    }
}

public struct ProcessedGenerationInput: Codable, Equatable, Sendable {
    public let request: GenerationRequest
    public let preflight: GenerationPreflightPlan
    public let runtime: GenerationRuntimePlan
    public let pixelShape: [Int]?
    public let tokenIDs: [Int]?

    public init(
        request: GenerationRequest,
        preflight: GenerationPreflightPlan,
        runtime: GenerationRuntimePlan,
        pixelShape: [Int]?,
        tokenIDs: [Int]?
    ) {
        self.request = request
        self.preflight = preflight
        self.runtime = runtime
        self.pixelShape = pixelShape
        self.tokenIDs = tokenIDs
    }
}

public struct VLMModelOutput: Codable, Equatable, Sendable {
    public let logitsShape: [Int]
    public let nextTokenID: Int?

    public init(logitsShape: [Int], nextTokenID: Int? = nil) {
        self.logitsShape = logitsShape
        self.nextTokenID = nextTokenID
    }
}

public protocol ModelContainer: Sendable {
    var context: ModelLoadContext { get }
}

public protocol VLMProcessor: Sendable {
    func process(request: GenerationRequest, context: ModelLoadContext) throws -> ProcessedGenerationInput
}

public protocol VLMModel: Sendable {
    var descriptor: ModelDescriptor { get }

    func forward(_ input: ProcessedGenerationInput) async throws -> VLMModelOutput
}

public protocol VLMGenerator: Sendable {
    func generate(input: ProcessedGenerationInput) async throws -> AsyncThrowingStream<GenerationChunk, Error>
}

public protocol VLMBackend: Sendable {
    var descriptor: ModelDescriptor { get }
    var status: BackendStatus { get }

    func loadContext() throws -> ModelLoadContext
    func process(_ request: GenerationRequest) throws -> ProcessedGenerationInput
    func generate(_ request: GenerationRequest) async throws -> AsyncThrowingStream<GenerationChunk, Error>
}

public protocol EmbeddingBackend: Sendable {
    var descriptor: ModelDescriptor { get }
    var status: BackendStatus { get }

    func embed(_ request: EmbeddingRequest) async throws -> CompletedEmbedding
}

public struct CompatibilityModelContainer: ModelContainer {
    public let context: ModelLoadContext

    public init(descriptor: ModelDescriptor, backend: BackendStatus = .compatibilityShell) {
        self.context = ModelLoadContext(
            descriptor: descriptor,
            loadPlan: ModelLoadPlanner(backend: backend).plan(descriptor: descriptor)
        )
    }
}

public struct CompatibilityProcessor: VLMProcessor {
    public let backend: BackendStatus
    public let mediaResolver: MediaReferenceResolver
    public let imageGridConfig: QwenVLImageGridConfig?

    public init(
        backend: BackendStatus = .compatibilityShell,
        mediaResolver: MediaReferenceResolver = MediaReferenceResolver(),
        imageGridConfig: QwenVLImageGridConfig? = nil
    ) {
        self.backend = backend
        self.mediaResolver = mediaResolver
        self.imageGridConfig = imageGridConfig
    }

    public func process(request: GenerationRequest, context: ModelLoadContext) throws -> ProcessedGenerationInput {
        let preflight = GenerationPreflightPlanner(
            descriptor: context.descriptor,
            backend: backend,
            mediaResolver: mediaResolver,
            imageGridConfig: imageGridConfig
        ).plan(request: request)

        return ProcessedGenerationInput(
            request: request,
            preflight: preflight,
            runtime: preflight.runtime,
            pixelShape: combinedPixelShape(preflight.imagePixels),
            tokenIDs: preflight.tokenization?.tokenIDs
        )
    }

    private func combinedPixelShape(_ report: QwenVLImagePixelPreflightReport) -> [Int]? {
        guard report.preparedCount > 0 else {
            return nil
        }
        let shapes = report.images.compactMap(\.pixelShape)
        guard let first = shapes.first, shapes.allSatisfy({ $0 == first }) else {
            return nil
        }
        return [shapes.count] + first
    }
}

public struct UnavailableVLMGenerator: VLMGenerator {
    public let descriptor: ModelDescriptor
    public let backend: BackendStatus

    public init(descriptor: ModelDescriptor, backend: BackendStatus = .compatibilityShell) {
        self.descriptor = descriptor
        self.backend = backend
    }

    public func generate(input: ProcessedGenerationInput) async throws -> AsyncThrowingStream<GenerationChunk, Error> {
        throw GenerationUnavailableError(
            report: GenerationUnavailableReport(
                error: "Swift inference backend is not wired yet.",
                model: descriptor.id,
                canonicalModelType: descriptor.canonicalModelType,
                backend: backend,
                preflight: input.preflight
            )
        )
    }
}

public struct CompatibilityVLMBackend: VLMBackend {
    public let descriptor: ModelDescriptor
    public let status: BackendStatus
    public let container: CompatibilityModelContainer
    public let processor: CompatibilityProcessor
    public let generator: UnavailableVLMGenerator

    public init(
        descriptor: ModelDescriptor,
        status: BackendStatus = .compatibilityShell,
        mediaResolver: MediaReferenceResolver = MediaReferenceResolver(),
        imageGridConfig: QwenVLImageGridConfig? = nil
    ) {
        self.descriptor = descriptor
        self.status = status
        self.container = CompatibilityModelContainer(descriptor: descriptor, backend: status)
        self.processor = CompatibilityProcessor(
            backend: status,
            mediaResolver: mediaResolver,
            imageGridConfig: imageGridConfig
        )
        self.generator = UnavailableVLMGenerator(descriptor: descriptor, backend: status)
    }

    public func loadContext() throws -> ModelLoadContext {
        container.context
    }

    public func process(_ request: GenerationRequest) throws -> ProcessedGenerationInput {
        try processor.process(request: request, context: loadContext())
    }

    public func generate(_ request: GenerationRequest) async throws -> AsyncThrowingStream<GenerationChunk, Error> {
        try await generator.generate(input: process(request))
    }

    public func unavailableReport(for request: GenerationRequest) -> GenerationUnavailableReport {
        do {
            let input = try process(request)
            return GenerationUnavailableReport(
                error: "Swift inference backend is not wired yet.",
                model: descriptor.id,
                canonicalModelType: descriptor.canonicalModelType,
                backend: status,
                preflight: input.preflight
            )
        } catch {
            let preflight = GenerationPreflightPlanner(
                descriptor: descriptor,
                backend: status
            ).plan(request: request)
            return GenerationUnavailableReport(
                error: String(describing: error),
                model: descriptor.id,
                canonicalModelType: descriptor.canonicalModelType,
                backend: status,
                preflight: preflight
            )
        }
    }
}

public struct GenerationUnavailableError: Error, CustomStringConvertible, Equatable, Sendable {
    public let report: GenerationUnavailableReport

    public init(report: GenerationUnavailableReport) {
        self.report = report
    }

    public var description: String {
        report.error
    }
}
