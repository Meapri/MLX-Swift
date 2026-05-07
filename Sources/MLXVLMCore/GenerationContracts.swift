import Foundation

public enum MessageRole: String, Codable, Sendable {
    case system
    case developer
    case user
    case assistant
    case tool
}

public enum ContentPart: Codable, Equatable, Sendable {
    case text(String)
    case imagePlaceholder
    case imageURL(ImageReference)
    case audioPlaceholder
    case audioURL(AudioReference)
    case videoURL(VideoReference)

    enum CodingKeys: String, CodingKey {
        case type
        case text
        case content
        case image
        case imageURL = "image_url"
        case detail
        case resizedHeight = "resized_height"
        case resizedWidth = "resized_width"
        case inputAudio = "input_audio"
        case video
        case videoURL = "video_url"
        case inputVideo = "input_video"
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case fps
        case nframes
        case minFrames = "min_frames"
        case maxFrames = "max_frames"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "text", "input_text", "output_text":
            if let text = try? container.decode(String.self, forKey: .text) {
                self = .text(text)
            } else {
                self = .text(try container.decode(String.self, forKey: .content))
            }
        case "image":
            if let url = Self.decodeOptionalImageURL(container, key: .image) {
                self = .imageURL(Self.imageReference(url: url, container: container))
            } else {
                self = .imagePlaceholder
            }
        case "image_url", "input_image":
            if let url = Self.decodeOptionalImageURL(container, key: .imageURL) {
                self = .imageURL(Self.imageReference(url: url, container: container))
            } else {
                self = .imagePlaceholder
            }
        case "audio":
            self = .audioPlaceholder
        case "input_audio":
            if let raw = try? container.decode(String.self, forKey: .inputAudio) {
                self = .audioURL(AudioReference(data: raw))
            } else {
                let nested = try container.decode(InputAudio.self, forKey: .inputAudio)
                self = .audioURL(AudioReference(data: nested.data, format: nested.format))
            }
        case "video", "input_video", "video_url":
            self = .videoURL(
                VideoReference(
                    url: Self.decodeOptionalVideoURL(container) ?? "",
                    minPixels: try? container.decode(Int.self, forKey: .minPixels),
                    maxPixels: try? container.decode(Int.self, forKey: .maxPixels),
                    fps: try? container.decode(Double.self, forKey: .fps),
                    nframes: try? container.decode(Int.self, forKey: .nframes),
                    minFrames: try? container.decode(Int.self, forKey: .minFrames),
                    maxFrames: try? container.decode(Int.self, forKey: .maxFrames)
                )
            )
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unsupported content part type: \(type)"
            )
        }
    }

    private static func decodeOptionalImageURL(
        _ container: KeyedDecodingContainer<CodingKeys>,
        key: CodingKeys
    ) -> String? {
        if let raw = try? container.decode(String.self, forKey: key) {
            return raw
        }
        if let nested = try? container.decode(ImageURL.self, forKey: key) {
            return nested.url
        }
        return nil
    }

    private static func decodeOptionalVideoURL(_ container: KeyedDecodingContainer<CodingKeys>) -> String? {
        if let raw = try? container.decode(String.self, forKey: .video) {
            return raw
        }
        if let raw = try? container.decode(String.self, forKey: .videoURL) {
            return raw
        }
        if let nested = try? container.decode(ImageURL.self, forKey: .videoURL) {
            return nested.url
        }
        if let raw = try? container.decode(String.self, forKey: .inputVideo) {
            return raw
        }
        if let nested = try? container.decode(ImageURL.self, forKey: .inputVideo) {
            return nested.url
        }
        return nil
    }

    private static func imageReference(
        url: String,
        container: KeyedDecodingContainer<CodingKeys>
    ) -> ImageReference {
        ImageReference(
            url: url,
            detail: try? container.decode(String.self, forKey: .detail),
            resizedHeight: try? container.decode(Int.self, forKey: .resizedHeight),
            resizedWidth: try? container.decode(Int.self, forKey: .resizedWidth),
            minPixels: try? container.decode(Int.self, forKey: .minPixels),
            maxPixels: try? container.decode(Int.self, forKey: .maxPixels)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)
        case .imagePlaceholder:
            try container.encode("image", forKey: .type)
        case .imageURL(let image):
            try container.encode("image_url", forKey: .type)
            try container.encode(ImageURL(url: image.url), forKey: .imageURL)
            try container.encodeIfPresent(image.detail, forKey: .detail)
            try container.encodeIfPresent(image.resizedHeight, forKey: .resizedHeight)
            try container.encodeIfPresent(image.resizedWidth, forKey: .resizedWidth)
            try container.encodeIfPresent(image.minPixels, forKey: .minPixels)
            try container.encodeIfPresent(image.maxPixels, forKey: .maxPixels)
        case .audioPlaceholder:
            try container.encode("audio", forKey: .type)
        case .audioURL(let audio):
            try container.encode("input_audio", forKey: .type)
            if audio.format == nil {
                try container.encode(audio.data, forKey: .inputAudio)
            } else {
                try container.encode(InputAudio(data: audio.data, format: audio.format), forKey: .inputAudio)
            }
        case .videoURL(let url):
            try container.encode("video", forKey: .type)
            try container.encode(url.url, forKey: .video)
            try container.encodeIfPresent(url.minPixels, forKey: .minPixels)
            try container.encodeIfPresent(url.maxPixels, forKey: .maxPixels)
            try container.encodeIfPresent(url.fps, forKey: .fps)
            try container.encodeIfPresent(url.nframes, forKey: .nframes)
            try container.encodeIfPresent(url.minFrames, forKey: .minFrames)
            try container.encodeIfPresent(url.maxFrames, forKey: .maxFrames)
        }
    }
}

public struct ImageURL: Codable, Equatable, Sendable {
    public let url: String

    public init(url: String) {
        self.url = url
    }
}

public struct ImageReference: Codable, Equatable, Sendable {
    public let url: String
    public let detail: String?
    public let resizedHeight: Int?
    public let resizedWidth: Int?
    public let minPixels: Int?
    public let maxPixels: Int?

    public init(
        url: String,
        detail: String? = nil,
        resizedHeight: Int? = nil,
        resizedWidth: Int? = nil,
        minPixels: Int? = nil,
        maxPixels: Int? = nil
    ) {
        self.url = url
        self.detail = detail
        self.resizedHeight = resizedHeight
        self.resizedWidth = resizedWidth
        self.minPixels = minPixels
        self.maxPixels = maxPixels
    }
}

public struct InputAudio: Codable, Equatable, Sendable {
    public let data: String
    public let format: String?

    public init(data: String, format: String? = nil) {
        self.data = data
        self.format = format
    }
}

public struct AudioReference: Codable, Equatable, Sendable {
    public let data: String
    public let format: String?

    public init(data: String, format: String? = nil) {
        self.data = data
        self.format = format
    }
}

public struct VideoReference: Codable, Equatable, Sendable {
    public let url: String
    public let minPixels: Int?
    public let maxPixels: Int?
    public let fps: Double?
    public let nframes: Int?
    public let minFrames: Int?
    public let maxFrames: Int?

    public init(
        url: String,
        minPixels: Int? = nil,
        maxPixels: Int? = nil,
        fps: Double? = nil,
        nframes: Int? = nil,
        minFrames: Int? = nil,
        maxFrames: Int? = nil
    ) {
        self.url = url
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        self.fps = fps
        self.nframes = nframes
        self.minFrames = minFrames
        self.maxFrames = maxFrames
    }
}

public struct ChatMessage: Codable, Equatable, Sendable {
    public let role: MessageRole
    public let content: [ContentPart]
    public let name: String?
    public let reasoning: String?
    public let toolCalls: [JSONValue]?
    public let toolCallID: String?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case name
        case reasoning
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }

    public init(
        role: MessageRole,
        content: [ContentPart],
        name: String? = nil,
        reasoning: String? = nil,
        toolCalls: [JSONValue]? = nil,
        toolCallID: String? = nil
    ) {
        self.role = role
        self.content = content
        self.name = name
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }
}

public struct GenerationParameters: Codable, Equatable, Sendable {
    public var maxTokens: Int
    public var temperature: Double
    public var topP: Double
    public var topK: Int
    public var minP: Double?
    public var typicalP: Double?
    public var tfsZ: Double?
    public var seed: Int
    public var contextLength: Int?
    public var numKeep: Int?
    public var kvBits: Double?
    public var kvQuantizationScheme: String?
    public var kvGroupSize: Int?
    public var quantizedKVStart: Int?
    public var maxKVSize: Int?
    public var prefillStepSize: Int?
    public var visionCacheSize: Int?
    public var quantizeActivations: Bool?
    public var repetitionPenalty: Double?
    public var repeatLastN: Int?
    public var presencePenalty: Double?
    public var frequencyPenalty: Double?
    public var penalizeNewline: Bool?
    public var mirostat: Int?
    public var mirostatTau: Double?
    public var mirostatEta: Double?
    public var stopSequences: [String]
    public var keepAlive: String?
    public var enableThinking: Bool?
    public var thinkingBudget: Int?

    public init(
        maxTokens: Int = 512,
        temperature: Double = 0.0,
        topP: Double = 1.0,
        topK: Int = 0,
        minP: Double? = nil,
        typicalP: Double? = nil,
        tfsZ: Double? = nil,
        seed: Int = 0,
        contextLength: Int? = nil,
        numKeep: Int? = nil,
        kvBits: Double? = nil,
        kvQuantizationScheme: String? = nil,
        kvGroupSize: Int? = nil,
        quantizedKVStart: Int? = nil,
        maxKVSize: Int? = nil,
        prefillStepSize: Int? = nil,
        visionCacheSize: Int? = nil,
        quantizeActivations: Bool? = nil,
        repetitionPenalty: Double? = nil,
        repeatLastN: Int? = nil,
        presencePenalty: Double? = nil,
        frequencyPenalty: Double? = nil,
        penalizeNewline: Bool? = nil,
        mirostat: Int? = nil,
        mirostatTau: Double? = nil,
        mirostatEta: Double? = nil,
        stopSequences: [String] = [],
        keepAlive: String? = nil,
        enableThinking: Bool? = nil,
        thinkingBudget: Int? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.typicalP = typicalP
        self.tfsZ = tfsZ
        self.seed = seed
        self.contextLength = contextLength
        self.numKeep = numKeep
        self.kvBits = kvBits
        self.kvQuantizationScheme = kvQuantizationScheme
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.maxKVSize = maxKVSize
        self.prefillStepSize = prefillStepSize
        self.visionCacheSize = visionCacheSize
        self.quantizeActivations = quantizeActivations
        self.repetitionPenalty = repetitionPenalty
        self.repeatLastN = repeatLastN
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.penalizeNewline = penalizeNewline
        self.mirostat = mirostat
        self.mirostatTau = mirostatTau
        self.mirostatEta = mirostatEta
        self.stopSequences = stopSequences
        self.keepAlive = keepAlive
        self.enableThinking = enableThinking
        self.thinkingBudget = thinkingBudget
    }
}

public struct GenerationRequestMetadata: Codable, Equatable, Sendable {
    public let responseFormat: JSONValue?
    public let rawPrompt: Bool
    public let template: String?
    public let suffix: String?
    public let legacyContext: [Int]?
    public let tools: [JSONValue]?
    public let toolChoice: JSONValue?
    public let rawOptions: [String: JSONValue]?
    public let adapterPath: String?
    public let draftModel: String?
    public let draftKind: String?
    public let draftBlockSize: Int?
    public let logitBias: JSONValue?
    public let logprobs: Bool?
    public let topLogprobs: Int?
    public let resizeShape: [Int]?
    public let thinkingStartToken: String?
    public let user: String?
    public let responseInstructions: String?
    public let responseTruncation: String?
    public let responseMetadata: JSONValue?
    public let n: Int?
    public let streamOptions: JSONValue?
    public let modalities: [JSONValue]?
    public let audio: JSONValue?
    public let prediction: JSONValue?
    public let previousResponseID: String?
    public let include: [JSONValue]?
    public let parallelToolCalls: Bool?
    public let store: Bool?
    public let serviceTier: String?
    public let responseReasoning: JSONValue?
    public let tenantID: String?

    public init(
        responseFormat: JSONValue? = nil,
        rawPrompt: Bool = false,
        template: String? = nil,
        suffix: String? = nil,
        legacyContext: [Int]? = nil,
        tools: [JSONValue]? = nil,
        toolChoice: JSONValue? = nil,
        rawOptions: [String: JSONValue]? = nil,
        adapterPath: String? = nil,
        draftModel: String? = nil,
        draftKind: String? = nil,
        draftBlockSize: Int? = nil,
        logitBias: JSONValue? = nil,
        logprobs: Bool? = nil,
        topLogprobs: Int? = nil,
        resizeShape: [Int]? = nil,
        thinkingStartToken: String? = nil,
        user: String? = nil,
        responseInstructions: String? = nil,
        responseTruncation: String? = nil,
        responseMetadata: JSONValue? = nil,
        n: Int? = nil,
        streamOptions: JSONValue? = nil,
        modalities: [JSONValue]? = nil,
        audio: JSONValue? = nil,
        prediction: JSONValue? = nil,
        previousResponseID: String? = nil,
        include: [JSONValue]? = nil,
        parallelToolCalls: Bool? = nil,
        store: Bool? = nil,
        serviceTier: String? = nil,
        responseReasoning: JSONValue? = nil,
        tenantID: String? = nil
    ) {
        self.responseFormat = responseFormat
        self.rawPrompt = rawPrompt
        self.template = template
        self.suffix = suffix
        self.legacyContext = legacyContext
        self.tools = tools
        self.toolChoice = toolChoice
        self.rawOptions = rawOptions
        self.adapterPath = adapterPath
        self.draftModel = draftModel
        self.draftKind = draftKind
        self.draftBlockSize = draftBlockSize
        self.logitBias = logitBias
        self.logprobs = logprobs
        self.topLogprobs = topLogprobs
        self.resizeShape = resizeShape
        self.thinkingStartToken = thinkingStartToken
        self.user = user
        self.responseInstructions = responseInstructions
        self.responseTruncation = responseTruncation
        self.responseMetadata = responseMetadata
        self.n = n
        self.streamOptions = streamOptions
        self.modalities = modalities
        self.audio = audio
        self.prediction = prediction
        self.previousResponseID = previousResponseID
        self.include = include
        self.parallelToolCalls = parallelToolCalls
        self.store = store
        self.serviceTier = serviceTier
        self.responseReasoning = responseReasoning
        self.tenantID = tenantID
    }
}

public struct GenerationRequest: Codable, Equatable, Sendable {
    public let model: String
    public let messages: [ChatMessage]
    public let parameters: GenerationParameters
    public let metadata: GenerationRequestMetadata
    public let stream: Bool

    public init(
        model: String,
        messages: [ChatMessage],
        parameters: GenerationParameters = GenerationParameters(),
        metadata: GenerationRequestMetadata = GenerationRequestMetadata(),
        stream: Bool = false
    ) {
        self.model = model
        self.messages = messages
        self.parameters = parameters
        self.metadata = metadata
        self.stream = stream
    }
}

public struct GenerationChunk: Codable, Equatable, Sendable {
    public let text: String
    public let reasoning: String?
    public let tokenID: Int?
    public let logprob: GenerationTokenLogprob?
    public let isFinished: Bool
    public let finishReason: String?
    public let promptTokenCount: Int?
    public let completionTokenCount: Int?
    public let toolCalls: [GenerationToolCall]

    public init(
        text: String,
        reasoning: String? = nil,
        tokenID: Int? = nil,
        logprob: GenerationTokenLogprob? = nil,
        isFinished: Bool = false,
        finishReason: String? = nil,
        promptTokenCount: Int? = nil,
        completionTokenCount: Int? = nil,
        toolCalls: [GenerationToolCall] = []
    ) {
        self.text = text
        self.reasoning = reasoning
        self.tokenID = tokenID
        self.logprob = logprob
        self.isFinished = isFinished
        self.finishReason = finishReason
        self.promptTokenCount = promptTokenCount
        self.completionTokenCount = completionTokenCount
        self.toolCalls = toolCalls
    }
}

public struct GenerationTokenLogprob: Codable, Equatable, Sendable {
    public let token: String
    public let logprob: Double
    public let bytes: [Int]?
    public let topLogprobs: [GenerationTopLogprob]

    public init(
        token: String,
        logprob: Double,
        bytes: [Int]? = nil,
        topLogprobs: [GenerationTopLogprob] = []
    ) {
        self.token = token
        self.logprob = logprob
        self.bytes = bytes
        self.topLogprobs = topLogprobs
    }
}

public struct GenerationTopLogprob: Codable, Equatable, Sendable {
    public let token: String
    public let logprob: Double
    public let bytes: [Int]?

    public init(token: String, logprob: Double, bytes: [Int]? = nil) {
        self.token = token
        self.logprob = logprob
        self.bytes = bytes
    }
}

public struct GenerationToolCall: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let function: GenerationToolCallFunction

    public init(id: String, type: String = "function", function: GenerationToolCallFunction) {
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct GenerationToolCallFunction: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: [String: JSONValue]

    public init(name: String, arguments: [String: JSONValue]) {
        self.name = name
        self.arguments = arguments
    }
}

public struct GenerationUsage: Codable, Equatable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int

    public init(promptTokens: Int = 0, completionTokens: Int = 0) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }

    public var totalTokens: Int {
        promptTokens + completionTokens
    }
}

public struct CompletedGeneration: Codable, Equatable, Sendable {
    public let model: String
    public let text: String
    public let finishReason: String
    public let usage: GenerationUsage
    public let toolCalls: [GenerationToolCall]
    public let logprobs: [GenerationTokenLogprob]

    public init(
        model: String,
        text: String,
        finishReason: String = "stop",
        usage: GenerationUsage = GenerationUsage(),
        toolCalls: [GenerationToolCall] = [],
        logprobs: [GenerationTokenLogprob] = []
    ) {
        self.model = model
        self.text = text
        self.finishReason = finishReason
        self.usage = usage
        self.toolCalls = toolCalls
        self.logprobs = logprobs
    }
}

public struct EmbeddingRequest: Codable, Equatable, Sendable {
    public let model: String
    public let texts: [String]
    public let tokenIDInputs: [[Int]]?
    public let truncate: Bool?
    public let parameters: GenerationParameters

    public init(
        model: String,
        texts: [String],
        tokenIDInputs: [[Int]]? = nil,
        truncate: Bool? = nil,
        parameters: GenerationParameters = GenerationParameters()
    ) {
        self.model = model
        self.texts = texts
        self.tokenIDInputs = tokenIDInputs
        self.truncate = truncate
        self.parameters = parameters
    }
}

public struct CompletedEmbedding: Codable, Equatable, Sendable {
    public let model: String
    public let embeddings: [[Float]]
    public let promptTokenCount: Int

    public init(
        model: String,
        embeddings: [[Float]],
        promptTokenCount: Int
    ) {
        self.model = model
        self.embeddings = embeddings
        self.promptTokenCount = promptTokenCount
    }

    public var inputCount: Int {
        embeddings.count
    }
}

public struct EmbeddingUnavailableReport: Codable, Equatable, Sendable {
    public let error: String
    public let model: String
    public let canonicalModelType: String
    public let backend: BackendStatus
    public let fallbackPolicy: String
    public let unavailableReason: String
    public let request: EmbeddingRequest
    public let inputCount: Int

    public init(
        error: String,
        model: String,
        canonicalModelType: String,
        backend: BackendStatus,
        fallbackPolicy: String = "diagnostic-501-no-generated-embedding",
        unavailableReason: String? = nil,
        request: EmbeddingRequest
    ) {
        self.error = error
        self.model = model
        self.canonicalModelType = canonicalModelType
        self.backend = backend
        self.fallbackPolicy = fallbackPolicy
        self.unavailableReason = unavailableReason ?? (
            backend.activeBackend == "mlx-swift-vlm"
                ? "The generation backend is loaded, but no MLXEmbedders-compatible embedding backend is available for this model directory."
                : "The Swift embedding backend is not available in this build."
        )
        self.request = request
        self.inputCount = max(request.texts.count, request.tokenIDInputs?.count ?? 0)
    }
}
