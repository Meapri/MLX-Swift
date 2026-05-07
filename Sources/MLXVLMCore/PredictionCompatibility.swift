import Foundation

public struct PredictionMediaReference: Codable, Equatable, Sendable {
    public let kind: MediaKind
    public let reference: String

    public init(kind: MediaKind, reference: String) {
        self.kind = kind
        self.reference = reference
    }
}

public struct PredictionRequest: Codable, Equatable, Sendable {
    public let model: String?
    public let task: String?
    public let body: JSONValue
    public let mediaReferences: [PredictionMediaReference]

    public init(
        model: String?,
        task: String?,
        body: JSONValue,
        mediaReferences: [PredictionMediaReference]
    ) {
        self.model = model
        self.task = task
        self.body = body
        self.mediaReferences = mediaReferences
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let body = try container.decode(JSONValue.self)
        let object = body.objectValue ?? [:]
        self.init(
            model: object["model"]?.stringValue ?? object["name"]?.stringValue,
            task: object["task"]?.stringValue ?? object["type"]?.stringValue,
            body: body,
            mediaReferences: Self.collectMediaReferences(from: body)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(model, forKey: .model)
        try container.encodeIfPresent(task, forKey: .task)
        try container.encode(body, forKey: .body)
        try container.encode(mediaReferences, forKey: .mediaReferences)
    }

    enum CodingKeys: String, CodingKey {
        case model
        case task
        case body
        case mediaReferences
    }

    private static func collectMediaReferences(from value: JSONValue) -> [PredictionMediaReference] {
        var references: [PredictionMediaReference] = []
        collectMediaReferences(from: value, keyHint: nil, into: &references)
        return references
    }

    private static func collectMediaReferences(
        from value: JSONValue,
        keyHint: String?,
        into references: inout [PredictionMediaReference]
    ) {
        switch value {
        case .object(let object):
            if let kind = kind(for: keyHint), let string = object["url"]?.stringValue ?? object["data"]?.stringValue {
                references.append(PredictionMediaReference(kind: kind, reference: string))
            }
            for (key, child) in object {
                collectMediaReferences(from: child, keyHint: key, into: &references)
            }
        case .array(let array):
            for item in array {
                collectMediaReferences(from: item, keyHint: keyHint, into: &references)
            }
        case .string(let string):
            if let kind = kind(for: keyHint) {
                references.append(PredictionMediaReference(kind: kind, reference: string))
            }
        case .number, .bool, .null:
            break
        }
    }

    private static func kind(for key: String?) -> MediaKind? {
        guard let key = key?.lowercased() else {
            return nil
        }
        switch key {
        case "image", "images", "image_url", "image_urls", "input_image", "input_images":
            return .image
        case "video", "videos", "video_url", "video_urls", "input_video", "input_videos":
            return .video
        case "audio", "audios", "audio_url", "audio_urls", "input_audio", "input_audios":
            return .audio
        default:
            return nil
        }
    }
}

public struct PredictionPreflightPlan: Codable, Equatable, Sendable {
    public let requestModel: String
    public let canonicalModelType: String
    public let requestTask: String?
    public let primaryTask: String
    public let preferredSwiftEntryPoint: String
    public let capabilities: ModelCapabilityPlan
    public let media: MediaResolutionReport
    public let compatibilityChecks: [CompatibilityCheck]
    public let canAttemptPrediction: Bool
    public let blockingReasons: [String]

    public init(
        requestModel: String,
        canonicalModelType: String,
        requestTask: String?,
        primaryTask: String,
        preferredSwiftEntryPoint: String,
        capabilities: ModelCapabilityPlan,
        media: MediaResolutionReport,
        compatibilityChecks: [CompatibilityCheck],
        canAttemptPrediction: Bool,
        blockingReasons: [String]
    ) {
        self.requestModel = requestModel
        self.canonicalModelType = canonicalModelType
        self.requestTask = requestTask
        self.primaryTask = primaryTask
        self.preferredSwiftEntryPoint = preferredSwiftEntryPoint
        self.capabilities = capabilities
        self.media = media
        self.compatibilityChecks = compatibilityChecks
        self.canAttemptPrediction = canAttemptPrediction
        self.blockingReasons = blockingReasons
    }
}

public struct PredictionUnavailableReport: Codable, Equatable, Sendable {
    public let error: String
    public let model: String
    public let canonicalModelType: String
    public let backend: BackendStatus
    public let preflight: PredictionPreflightPlan

    public init(
        error: String,
        model: String,
        canonicalModelType: String,
        backend: BackendStatus,
        preflight: PredictionPreflightPlan
    ) {
        self.error = error
        self.model = model
        self.canonicalModelType = canonicalModelType
        self.backend = backend
        self.preflight = preflight
    }
}

public struct PredictionPreflightPlanner {
    public let descriptor: ModelDescriptor
    public let backend: BackendStatus
    public let mediaResolver: MediaReferenceResolver

    public init(
        descriptor: ModelDescriptor,
        backend: BackendStatus = .compatibilityShell,
        mediaResolver: MediaReferenceResolver = MediaReferenceResolver()
    ) {
        self.descriptor = descriptor
        self.backend = backend
        self.mediaResolver = mediaResolver
    }

    public func plan(request: PredictionRequest) -> PredictionPreflightPlan {
        let capabilities = ModelCapabilityPlanner().plan(descriptor: descriptor)
        let compatibility = ModelCompatibilityValidator.validate(descriptor: descriptor)
        let media = MediaResolutionReport(
            references: request.mediaReferences.map {
                mediaResolver.resolve(reference: $0.reference, kind: $0.kind)
            }
        )
        var blockingReasons: [String] = []
        if capabilities.supportsTextGeneration {
            blockingReasons.append("Model type \(descriptor.canonicalModelType) is a generative model; use Ollama/OpenAI generation endpoints instead of predictor endpoints.")
        } else {
            blockingReasons.append("Swift predictor inference for \(descriptor.canonicalModelType) (\(capabilities.primaryTask)) is not ported yet.")
        }
        if media.errorCount > 0 {
            blockingReasons.append("\(media.errorCount) media reference(s) are not loadable.")
        }
        return PredictionPreflightPlan(
            requestModel: request.model ?? descriptor.id,
            canonicalModelType: descriptor.canonicalModelType,
            requestTask: request.task,
            primaryTask: capabilities.primaryTask,
            preferredSwiftEntryPoint: capabilities.preferredSwiftEntryPoint,
            capabilities: capabilities,
            media: media,
            compatibilityChecks: compatibility.checks,
            canAttemptPrediction: false,
            blockingReasons: blockingReasons
        )
    }

    public func unavailableReport(for request: PredictionRequest) -> PredictionUnavailableReport {
        PredictionUnavailableReport(
            error: "Swift predictor backend is unavailable.",
            model: descriptor.id,
            canonicalModelType: descriptor.canonicalModelType,
            backend: backend,
            preflight: plan(request: request)
        )
    }
}
