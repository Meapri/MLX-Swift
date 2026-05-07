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
    case imageURL(String)
    case audioURL(String)
    case videoURL(String)

    enum CodingKeys: String, CodingKey {
        case type
        case text
        case imageURL = "image_url"
        case inputAudio = "input_audio"
        case video
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "text", "input_text", "output_text":
            self = .text(try container.decode(String.self, forKey: .text))
        case "image_url", "input_image":
            if let raw = try? container.decode(String.self, forKey: .imageURL) {
                self = .imageURL(raw)
            } else {
                let nested = try container.decode(ImageURL.self, forKey: .imageURL)
                self = .imageURL(nested.url)
            }
        case "input_audio":
            if let raw = try? container.decode(String.self, forKey: .inputAudio) {
                self = .audioURL(raw)
            } else {
                let nested = try container.decode(InputAudio.self, forKey: .inputAudio)
                self = .audioURL(nested.data)
            }
        case "video":
            self = .videoURL((try? container.decode(String.self, forKey: .video)) ?? "")
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unsupported content part type: \(type)"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)
        case .imageURL(let url):
            try container.encode("image_url", forKey: .type)
            try container.encode(ImageURL(url: url), forKey: .imageURL)
        case .audioURL(let url):
            try container.encode("input_audio", forKey: .type)
            try container.encode(url, forKey: .inputAudio)
        case .videoURL(let url):
            try container.encode("video", forKey: .type)
            try container.encode(url, forKey: .video)
        }
    }
}

public struct ImageURL: Codable, Equatable, Sendable {
    public let url: String

    public init(url: String) {
        self.url = url
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
    public var maxKVSize: Int?
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
        maxKVSize: Int? = nil,
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
        self.maxKVSize = maxKVSize
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
    public let logitBias: JSONValue?
    public let logprobs: Bool?
    public let topLogprobs: Int?
    public let resizeShape: [Int]?
    public let thinkingStartToken: String?
    public let user: String?

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
        logitBias: JSONValue? = nil,
        logprobs: Bool? = nil,
        topLogprobs: Int? = nil,
        resizeShape: [Int]? = nil,
        thinkingStartToken: String? = nil,
        user: String? = nil
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
        self.logitBias = logitBias
        self.logprobs = logprobs
        self.topLogprobs = topLogprobs
        self.resizeShape = resizeShape
        self.thinkingStartToken = thinkingStartToken
        self.user = user
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
    public let tokenID: Int?
    public let isFinished: Bool
    public let finishReason: String?
    public let promptTokenCount: Int?
    public let completionTokenCount: Int?
    public let toolCalls: [GenerationToolCall]

    public init(
        text: String,
        tokenID: Int? = nil,
        isFinished: Bool = false,
        finishReason: String? = nil,
        promptTokenCount: Int? = nil,
        completionTokenCount: Int? = nil,
        toolCalls: [GenerationToolCall] = []
    ) {
        self.text = text
        self.tokenID = tokenID
        self.isFinished = isFinished
        self.finishReason = finishReason
        self.promptTokenCount = promptTokenCount
        self.completionTokenCount = completionTokenCount
        self.toolCalls = toolCalls
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

    public init(
        model: String,
        text: String,
        finishReason: String = "stop",
        usage: GenerationUsage = GenerationUsage(),
        toolCalls: [GenerationToolCall] = []
    ) {
        self.model = model
        self.text = text
        self.finishReason = finishReason
        self.usage = usage
        self.toolCalls = toolCalls
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
    public let request: EmbeddingRequest
    public let inputCount: Int

    public init(
        error: String,
        model: String,
        canonicalModelType: String,
        backend: BackendStatus,
        request: EmbeddingRequest
    ) {
        self.error = error
        self.model = model
        self.canonicalModelType = canonicalModelType
        self.backend = backend
        self.request = request
        self.inputCount = max(request.texts.count, request.tokenIDInputs?.count ?? 0)
    }
}
