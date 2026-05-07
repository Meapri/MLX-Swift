import Foundation

public struct OllamaModelTag: Codable, Equatable, Sendable {
    public let name: String
    public let model: String
    public let modifiedAt: String
    public let size: Int64
    public let digest: String
    public let details: OllamaModelDetails

    enum CodingKeys: String, CodingKey {
        case name
        case model
        case modifiedAt = "modified_at"
        case size
        case digest
        case details
    }
}

public struct OllamaModelDetails: Codable, Equatable, Sendable {
    public let parentModel: String
    public let format: String
    public let family: String
    public let families: [String]
    public let parameterSize: String
    public let quantizationLevel: String

    enum CodingKeys: String, CodingKey {
        case parentModel = "parent_model"
        case format
        case family
        case families
        case parameterSize = "parameter_size"
        case quantizationLevel = "quantization_level"
    }
}

public struct OllamaTagsResponse: Codable, Equatable, Sendable {
    public let models: [OllamaModelTag]

    public init(models: [OllamaModelTag]) {
        self.models = models
    }
}

public struct OllamaShowRequest: Codable, Equatable, Sendable {
    public let fields: [String: JSONValue]

    public init(fields: [String: JSONValue] = [:]) {
        self.fields = fields
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.fields = (try? container.decode([String: JSONValue].self)) ?? [:]
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(fields)
    }

    public var modelName: String? {
        fields["model"]?.stringValue ?? fields["name"]?.stringValue
    }

    public var verbose: Bool {
        fields["verbose"]?.boolValue ?? false
    }
}

public struct OllamaShowResponse: Codable, Equatable, Sendable {
    public let modelfile: String
    public let parameters: String
    public let template: String
    public let details: OllamaModelDetails
    public let modelInfo: [String: JSONValue]

    enum CodingKeys: String, CodingKey {
        case modelfile
        case parameters
        case template
        case details
        case modelInfo = "model_info"
    }

    public init(descriptor: ModelDescriptor, backend: BackendStatus = .compatibilityShell) {
        let loadPlan = ModelLoadPlanner(backend: backend).plan(descriptor: descriptor)
        let normalizedConfigKeys = loadPlan.normalizedConfig.objectValue?.keys.sorted() ?? []
        self.modelfile = "FROM \(descriptor.path)"
        self.parameters = descriptor.quantizationMetadata.map { "quantization \($0.summary)" } ?? ""
        self.template = descriptor.tokenizerMetadata.chatTemplate ?? ""
        self.details = OllamaModelDetails(
            parentModel: "",
            format: "safetensors",
            family: descriptor.canonicalModelType,
            families: ["vision", "language"],
            parameterSize: "unknown",
            quantizationLevel: descriptor.quantizationMetadata?.summary ?? "unknown"
        )
        self.modelInfo = [
            "mlx_vlm.raw_model_type": .string(descriptor.rawModelType),
            "mlx_vlm.canonical_model_type": .string(descriptor.canonicalModelType),
            "mlx_vlm.weight_count": .number(Double(descriptor.weightFiles.count)),
            "mlx_vlm.has_chat_template": .bool(descriptor.hasChatTemplate),
            "mlx_vlm.backend_active": .string(backend.activeBackend),
            "mlx_vlm.backend_ready": .bool(backend.ready),
            "mlx_vlm.metadata_ready": .bool(loadPlan.metadataReady),
            "mlx_vlm.generation_ready": .bool(loadPlan.generationReady),
            "mlx_vlm.can_load_metadata": .bool(loadPlan.canLoadMetadata),
            "mlx_vlm.can_attempt_generation": .bool(loadPlan.canAttemptGeneration),
            "mlx_vlm.blocking_reasons": .array(loadPlan.blockingReasons.map(JSONValue.string)),
            "mlx_vlm.normalized_config_keys": .array(normalizedConfigKeys.map(JSONValue.string)),
            "mlx_vlm.config_text_source": descriptor.configNormalization.textConfigSource.map(JSONValue.string) ?? .null,
            "mlx_vlm.config_vision_source": descriptor.configNormalization.visionConfigSource.map(JSONValue.string) ?? .null,
            "mlx_vlm.config_audio_source": descriptor.configNormalization.audioConfigSource.map(JSONValue.string) ?? .null,
            "mlx_vlm.used_llm_config_as_text_config": .bool(descriptor.configNormalization.usedLLMConfigAsTextConfig),
            "mlx_vlm.inserted_empty_text_config": .bool(descriptor.configNormalization.insertedEmptyTextConfig),
            "mlx_vlm.inserted_empty_vision_config": .bool(descriptor.configNormalization.insertedEmptyVisionConfig),
            "mlx_vlm.inserted_empty_audio_config": .bool(descriptor.configNormalization.insertedEmptyAudioConfig),
            "mlx_vlm.tokenizer_backend": .string(loadPlan.tokenizerPlan.requiredBackend),
            "mlx_vlm.tokenizer_swift_execution_supported": .bool(loadPlan.tokenizerPlan.swiftExecutionSupported),
            "mlx_vlm.tokenizer_swift_execution_mode": loadPlan.tokenizerPlan.swiftExecutionMode.map(JSONValue.string) ?? .null,
            "mlx_vlm.chat_template_renderer": .string(loadPlan.chatTemplatePlan.requiredRenderer),
            "mlx_vlm.primary_task": .string(loadPlan.capabilities.primaryTask),
            "mlx_vlm.readable_weight_bytes": .number(Double(loadPlan.weightDataCatalog.totalReadableBytes)),
            "mlx_vlm.estimated_runtime_bytes": .number(Double(loadPlan.memoryEstimate.estimatedTotalRuntimeBytes)),
            "mlx_vlm.quantization_source": descriptor.quantizationMetadata.map { .string($0.source) } ?? .null,
            "mlx_vlm.quantization_bits": descriptor.quantizationMetadata?.bits.map { .number($0) } ?? .null,
            "mlx_vlm.quantization_group_size": descriptor.quantizationMetadata?.groupSize.map { .number(Double($0)) } ?? .null,
            "mlx_vlm.chat_template_source": descriptor.tokenizerMetadata.chatTemplateSource.map(JSONValue.string) ?? .null,
            "mlx_vlm.image_token_id": descriptor.tokenizerMetadata.imageTokenID.map { .number(Double($0)) } ?? .null,
            "mlx_vlm.video_token_id": descriptor.tokenizerMetadata.videoTokenID.map { .number(Double($0)) } ?? .null,
            "mlx_vlm.audio_token_id": descriptor.tokenizerMetadata.audioTokenID.map { .number(Double($0)) } ?? .null,
            "mlx_vlm.begin_image_token_id": descriptor.tokenizerMetadata.beginImageTokenID.map { .number(Double($0)) } ?? .null,
            "mlx_vlm.end_image_token_id": descriptor.tokenizerMetadata.endImageTokenID.map { .number(Double($0)) } ?? .null,
            "mlx_vlm.begin_audio_token_id": descriptor.tokenizerMetadata.beginAudioTokenID.map { .number(Double($0)) } ?? .null,
            "mlx_vlm.end_audio_token_id": descriptor.tokenizerMetadata.endAudioTokenID.map { .number(Double($0)) } ?? .null,
            "mlx_vlm.start_turn_token_id": descriptor.tokenizerMetadata.startTurnTokenID.map { .number(Double($0)) } ?? .null,
            "mlx_vlm.end_turn_token_id": descriptor.tokenizerMetadata.endTurnTokenID.map { .number(Double($0)) } ?? .null,
        ]
    }
}

public struct OllamaRunningModelsResponse: Codable, Equatable, Sendable {
    public let models: [OllamaRunningModel]

    public init(models: [OllamaRunningModel]) {
        self.models = models
    }
}

public struct OllamaRunningModel: Codable, Equatable, Sendable {
    public let name: String
    public let model: String
    public let size: Int64
    public let digest: String
    public let details: OllamaModelDetails
    public let expiresAt: String
    public let sizeVRAM: Int64

    enum CodingKeys: String, CodingKey {
        case name
        case model
        case size
        case digest
        case details
        case expiresAt = "expires_at"
        case sizeVRAM = "size_vram"
    }

    public init(descriptor: ModelDescriptor, expiresAt: String = "1970-01-01T00:00:00Z") {
        self.name = descriptor.id
        self.model = descriptor.id
        self.size = descriptor.totalWeightBytes
        self.digest = descriptor.weightFiles.map(\.name).joined(separator: ":")
        self.details = OllamaModelDetails(
            parentModel: "",
            format: "safetensors",
            family: descriptor.canonicalModelType,
            families: ["vision", "language"],
            parameterSize: "unknown",
            quantizationLevel: descriptor.quantizationMetadata?.summary ?? "unknown"
        )
        self.expiresAt = expiresAt
        self.sizeVRAM = 0
    }
}

public struct OllamaVersionResponse: Codable, Equatable, Sendable {
    public let version: String

    public init(version: String) {
        self.version = version
    }
}

public struct UnloadResponse: Codable, Equatable, Sendable {
    public let status: String
    public let model: String
    public let unloaded: Bool

    public init(status: String, model: String, unloaded: Bool = true) {
        self.status = status
        self.model = model
        self.unloaded = unloaded
    }
}

public struct PythonServerUnloadInfo: Codable, Equatable, Sendable {
    public let modelName: String?
    public let adapterName: String?

    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case adapterName = "adapter_name"
    }

    public init(modelName: String?, adapterName: String?) {
        self.modelName = modelName
        self.adapterName = adapterName
    }
}

public struct PythonServerUnloadResponse: Codable, Equatable, Sendable {
    public let status: String
    public let message: String
    public let unloaded: PythonServerUnloadInfo?

    public init(status: String, message: String, unloaded: PythonServerUnloadInfo? = nil) {
        self.status = status
        self.message = message
        self.unloaded = unloaded
    }
}

public enum OllamaResidencyActionKind: String, Codable, Equatable, Sendable {
    case load
    case unload
}

public struct OllamaResidencyAction: Codable, Equatable, Sendable {
    public let model: String
    public let action: OllamaResidencyActionKind
    public let keepAlive: String?

    enum CodingKeys: String, CodingKey {
        case model
        case action
        case keepAlive = "keep_alive"
    }

    public init(model: String, action: OllamaResidencyActionKind, keepAlive: String?) {
        self.model = model
        self.action = action
        self.keepAlive = keepAlive
    }
}

public struct OllamaModelResidency: Codable, Equatable, Sendable {
    public private(set) var isLoaded: Bool
    public private(set) var unloadCount: Int

    public init(isLoaded: Bool = true, unloadCount: Int = 0) {
        self.isLoaded = isLoaded
        self.unloadCount = unloadCount
    }

    public mutating func markLoaded() {
        isLoaded = true
    }

    public mutating func unload(model: String) -> UnloadResponse {
        isLoaded = false
        unloadCount += 1
        return UnloadResponse(status: "ok", model: model, unloaded: true)
    }

    public func runningModelsResponse(descriptor: ModelDescriptor) -> OllamaRunningModelsResponse {
        OllamaRunningModelsResponse(
            models: isLoaded ? [OllamaRunningModel(descriptor: descriptor)] : []
        )
    }
}

public struct OllamaModelOperationRequest: Codable, Equatable, Sendable {
    public let fields: [String: JSONValue]

    public init(fields: [String: JSONValue] = [:]) {
        self.fields = fields
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.fields = (try? container.decode([String: JSONValue].self)) ?? [:]
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(fields)
    }

    public var modelName: String? {
        fields["model"]?.stringValue
            ?? fields["name"]?.stringValue
            ?? fields["source"]?.stringValue
            ?? fields["destination"]?.stringValue
            ?? fields["from"]?.stringValue
    }

    public var requestedModelName: String? {
        fields["model"]?.stringValue ?? fields["name"]?.stringValue
    }

    public var sourceName: String? {
        fields["source"]?.stringValue ?? fields["from"]?.stringValue
    }

    public var destinationName: String? {
        fields["destination"]?.stringValue ?? fields["name"]?.stringValue
    }

    public var modelfile: String? {
        fields["modelfile"]?.stringValue
    }

    public var modelfileSourceName: String? {
        guard let modelfile else {
            return nil
        }
        for line in modelfile.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.uppercased().hasPrefix("FROM ") else {
                continue
            }
            let source = trimmed.dropFirst("FROM ".count).trimmingCharacters(in: .whitespacesAndNewlines)
            return source.isEmpty ? nil : source
        }
        return nil
    }

    public var useLatest: Bool {
        fields["use_latest"]?.boolValue ??
            fields["useLatest"]?.boolValue ??
            false
    }
}

public struct OllamaModelOperationReport: Codable, Equatable, Sendable {
    public let operation: String
    public let model: String?
    public let source: String?
    public let destination: String?
    public let accepted: Bool
    public let status: String
    public let path: String?
    public let error: String?
    public let backend: BackendStatus
    public let request: OllamaModelOperationRequest

    public init(
        operation: String,
        request: OllamaModelOperationRequest,
        backend: BackendStatus = .compatibilityShell,
        accepted: Bool = false,
        status: String = "unavailable",
        path: String? = nil,
        error: String? = nil
    ) {
        self.operation = operation
        self.model = request.modelName
        self.source = request.sourceName ?? request.modelfileSourceName
        self.destination = request.destinationName ?? request.requestedModelName
        self.accepted = accepted
        self.status = status
        self.path = path
        self.error = error ?? (accepted ? nil : "Ollama model operation '\(operation)' is not available in the dependency-free Swift compatibility shell.")
        self.backend = backend
        self.request = request
    }
}

public struct OllamaBlobOperationReport: Codable, Equatable, Sendable {
    public let operation: String
    public let digest: String
    public let accepted: Bool
    public let exists: Bool
    public let status: String
    public let bytes: Int?
    public let error: String?
    public let backend: BackendStatus

    public init(
        operation: String,
        digest: String,
        exists: Bool = false,
        backend: BackendStatus = .compatibilityShell,
        accepted: Bool = false,
        status: String = "unavailable",
        bytes: Int? = nil,
        error: String? = nil
    ) {
        self.operation = operation
        self.digest = digest
        self.accepted = accepted
        self.exists = exists
        self.status = status
        self.bytes = bytes
        self.error = error ?? (accepted ? nil : "Ollama blob operation '\(operation)' is not available in the dependency-free Swift compatibility shell.")
        self.backend = backend
    }
}

public struct APCCacheStatusResponse: Codable, Equatable, Sendable {
    public let enabled: Bool
    public let backend: BackendStatus
    public let status: String?
    public let note: String?

    public init(
        enabled: Bool = false,
        backend: BackendStatus = .compatibilityShell,
        status: String? = nil,
        note: String? = "Automatic Prefix Cache is not enabled in the Swift compatibility server."
    ) {
        self.enabled = enabled
        self.backend = backend
        self.status = status
        self.note = note
    }
}

public extension OllamaModelTag {
    init(descriptor: ModelDescriptor, modifiedAt: String = "1970-01-01T00:00:00Z") {
        self.init(name: descriptor.id, descriptor: descriptor, modifiedAt: modifiedAt)
    }

    init(name: String, descriptor: ModelDescriptor, modifiedAt: String = "1970-01-01T00:00:00Z") {
        self.name = name
        self.model = name
        self.modifiedAt = modifiedAt
        self.size = descriptor.totalWeightBytes
        self.digest = descriptor.weightFiles.map(\.name).joined(separator: ":")
        self.details = OllamaModelDetails(
            parentModel: "",
            format: "safetensors",
            family: descriptor.canonicalModelType,
            families: ["vision", "language"],
            parameterSize: "unknown",
            quantizationLevel: descriptor.quantizationMetadata?.summary ?? "unknown"
        )
    }
}
