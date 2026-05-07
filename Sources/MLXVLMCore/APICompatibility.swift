import Foundation

public enum APICompatibilityError: Error, CustomStringConvertible, Equatable {
    case missingPrompt
    case missingMessages
    case unsupportedContent(String)

    public var description: String {
        switch self {
        case .missingPrompt:
            return "Ollama generate request is missing prompt"
        case .missingMessages:
            return "Chat request is missing messages"
        case .unsupportedContent(let value):
            return "Unsupported message content: \(value)"
        }
    }
}

public struct OllamaGenerateRequest: Codable, Equatable, Sendable {
    public let model: String
    public let prompt: String?
    public let suffix: String?
    public let system: String?
    public let template: String?
    public let raw: Bool?
    public let format: JSONValue?
    public let context: [Int]?
    public let stream: Bool?
    public let images: [String]?
    public let options: [String: JSONValue]?
    public let keepAlive: CompatibleKeepAlive?

    enum CodingKeys: String, CodingKey {
        case model
        case prompt
        case suffix
        case system
        case template
        case raw
        case format
        case context
        case stream
        case images
        case options
        case keepAlive = "keep_alive"
    }

    public init(
        model: String,
        prompt: String? = nil,
        suffix: String? = nil,
        system: String? = nil,
        template: String? = nil,
        raw: Bool? = nil,
        format: JSONValue? = nil,
        context: [Int]? = nil,
        stream: Bool? = nil,
        images: [String]? = nil,
        options: [String: JSONValue]? = nil,
        keepAlive: CompatibleKeepAlive? = nil
    ) {
        self.model = model
        self.prompt = prompt
        self.suffix = suffix
        self.system = system
        self.template = template
        self.raw = raw
        self.format = format
        self.context = context
        self.stream = stream
        self.images = images
        self.options = options
        self.keepAlive = keepAlive
    }

    public func generationRequest(
        defaultModel: String,
        defaultParameters: GenerationParameters = GenerationParameters()
    ) throws -> GenerationRequest {
        guard let prompt else {
            throw APICompatibilityError.missingPrompt
        }
        var messages: [ChatMessage] = []
        if let system, !system.isEmpty {
            messages.append(ChatMessage(role: .system, content: [.text(system)]))
        }
        var content: [ContentPart] = [.text(prompt)]
        for image in images ?? [] {
            content.append(.imageURL(ImageReference(url: image)))
        }
        messages.append(ChatMessage(role: .user, content: content))
        return GenerationRequest(
            model: model.isEmpty ? defaultModel : model,
            messages: messages,
            parameters: GenerationParameters(
                options: options,
                defaults: defaultParameters,
                keepAlive: keepAlive?.value
            ),
            metadata: GenerationRequestMetadata(
                responseFormat: format,
                rawPrompt: raw ?? false,
                template: template,
                suffix: suffix,
                legacyContext: context,
                rawOptions: options,
                draftModel: options?.compatibleDraftModel,
                draftKind: options?.compatibleDraftKind,
                draftBlockSize: options?.compatibleDraftBlockSize
            ),
            stream: stream ?? false
        )
    }

    public func residencyAction(defaultModel: String) -> OllamaResidencyAction? {
        guard prompt?.isEmpty == true,
              system?.isEmpty != false,
              suffix?.isEmpty != false,
              images?.isEmpty != false,
              context?.isEmpty != false,
              keepAlive != nil
        else {
            return nil
        }
        let modelName = model.isEmpty ? defaultModel : model
        if keepAlive?.isZeroDuration == true {
            return OllamaResidencyAction(model: modelName, action: .unload, keepAlive: keepAlive?.value)
        }
        return OllamaResidencyAction(model: modelName, action: .load, keepAlive: keepAlive?.value)
    }
}

public struct OllamaChatRequest: Codable, Equatable, Sendable {
    public let model: String
    public let messages: [CompatibleChatMessage]
    public let format: JSONValue?
    public let tools: [JSONValue]?
    public let stream: Bool?
    public let options: [String: JSONValue]?
    public let keepAlive: CompatibleKeepAlive?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case format
        case tools
        case stream
        case options
        case keepAlive = "keep_alive"
    }

    public func generationRequest(
        defaultModel: String,
        defaultParameters: GenerationParameters = GenerationParameters()
    ) throws -> GenerationRequest {
        guard !messages.isEmpty else {
            throw APICompatibilityError.missingMessages
        }
        return GenerationRequest(
            model: model.isEmpty ? defaultModel : model,
            messages: try messages.map { try $0.chatMessage },
            parameters: GenerationParameters(
                options: options,
                defaults: defaultParameters,
                keepAlive: keepAlive?.value
            ),
            metadata: GenerationRequestMetadata(
                responseFormat: format,
                tools: tools,
                rawOptions: options,
                draftModel: options?.compatibleDraftModel,
                draftKind: options?.compatibleDraftKind,
                draftBlockSize: options?.compatibleDraftBlockSize
            ),
            stream: stream ?? false
        )
    }
}

public struct OpenAIChatCompletionRequest: Codable, Equatable, Sendable {
    public let model: String
    public let messages: [CompatibleChatMessage]
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let minP: Double?
    public let stop: CompatibleStop?
    public let seed: Int?
    public let repetitionPenalty: Double?
    public let presencePenalty: Double?
    public let frequencyPenalty: Double?
    public let logitBias: JSONValue?
    public let enableThinking: Bool?
    public let thinkingBudget: Int?
    public let thinkingStartToken: String?
    public let logprobs: Bool?
    public let topLogprobs: Int?
    public let resizeShape: CompatibleResizeShape?
    public let responseFormat: JSONValue?
    public let n: Int?
    public let tools: [JSONValue]?
    public let toolChoice: JSONValue?
    public let adapterPath: String?
    public let draftModel: String?
    public let draftKind: String?
    public let draftBlockSize: Int?
    public let user: String?
    public let metadata: JSONValue?
    public let parallelToolCalls: Bool?
    public let store: Bool?
    public let reasoning: JSONValue?
    public let serviceTier: String?
    public let streamOptions: JSONValue?
    public let modalities: [JSONValue]?
    public let audio: JSONValue?
    public let prediction: JSONValue?
    public let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case stop
        case seed
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case logitBias = "logit_bias"
        case enableThinking = "enable_thinking"
        case thinkingBudget = "thinking_budget"
        case thinkingStartToken = "thinking_start_token"
        case logprobs
        case topLogprobs = "top_logprobs"
        case resizeShape = "resize_shape"
        case responseFormat = "response_format"
        case n
        case tools
        case toolChoice = "tool_choice"
        case adapterPath = "adapter_path"
        case draftModel = "draft_model"
        case draftKind = "draft_kind"
        case draftBlockSize = "draft_block_size"
        case user
        case metadata
        case parallelToolCalls = "parallel_tool_calls"
        case store
        case reasoning
        case serviceTier = "service_tier"
        case streamOptions = "stream_options"
        case modalities
        case audio
        case prediction
        case stream
    }

    public func generationRequest(
        defaultModel: String,
        defaultParameters: GenerationParameters = GenerationParameters()
    ) throws -> GenerationRequest {
        guard !messages.isEmpty else {
            throw APICompatibilityError.missingMessages
        }
        return GenerationRequest(
            model: model.isEmpty ? defaultModel : model,
            messages: try messages.map { try $0.chatMessage },
            parameters: GenerationParameters(
                maxTokens: maxTokens ?? maxCompletionTokens ?? defaultParameters.maxTokens,
                temperature: temperature ?? defaultParameters.temperature,
                topP: topP ?? defaultParameters.topP,
                topK: topK ?? defaultParameters.topK,
                minP: minP ?? defaultParameters.minP,
                typicalP: defaultParameters.typicalP,
                tfsZ: defaultParameters.tfsZ,
                seed: seed ?? defaultParameters.seed,
                contextLength: defaultParameters.contextLength,
                numKeep: defaultParameters.numKeep,
                kvBits: defaultParameters.kvBits,
                kvQuantizationScheme: defaultParameters.kvQuantizationScheme,
                kvGroupSize: defaultParameters.kvGroupSize,
                quantizedKVStart: defaultParameters.quantizedKVStart,
                maxKVSize: defaultParameters.maxKVSize,
                prefillStepSize: defaultParameters.prefillStepSize,
                visionCacheSize: defaultParameters.visionCacheSize,
                quantizeActivations: defaultParameters.quantizeActivations,
                repetitionPenalty: repetitionPenalty ?? defaultParameters.repetitionPenalty,
                repeatLastN: defaultParameters.repeatLastN,
                presencePenalty: presencePenalty ?? defaultParameters.presencePenalty,
                frequencyPenalty: frequencyPenalty ?? defaultParameters.frequencyPenalty,
                penalizeNewline: defaultParameters.penalizeNewline,
                mirostat: defaultParameters.mirostat,
                mirostatTau: defaultParameters.mirostatTau,
                mirostatEta: defaultParameters.mirostatEta,
                stopSequences: stop?.values ?? defaultParameters.stopSequences,
                keepAlive: defaultParameters.keepAlive,
                enableThinking: enableThinking ?? defaultParameters.enableThinking,
                thinkingBudget: thinkingBudget ?? defaultParameters.thinkingBudget
            ),
            metadata: GenerationRequestMetadata(
                responseFormat: responseFormat,
                tools: tools,
                toolChoice: toolChoice,
                adapterPath: adapterPath,
                draftModel: draftModel,
                draftKind: draftKind,
                draftBlockSize: draftBlockSize,
                logitBias: logitBias,
                logprobs: logprobs,
                topLogprobs: topLogprobs,
                resizeShape: resizeShape?.values,
                thinkingStartToken: thinkingStartToken,
                user: user,
                responseMetadata: metadata,
                n: n,
                streamOptions: streamOptions,
                modalities: modalities,
                audio: audio,
                prediction: prediction,
                parallelToolCalls: parallelToolCalls,
                store: store,
                serviceTier: serviceTier,
                responseReasoning: reasoning
            ),
            stream: stream ?? false
        )
    }
}

public struct OpenAICompletionRequest: Codable, Equatable, Sendable {
    public let model: String
    public let prompt: CompatibleCompletionPrompt?
    public let suffix: String?
    public let maxTokens: Int?
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let minP: Double?
    public let stop: CompatibleStop?
    public let seed: Int?
    public let presencePenalty: Double?
    public let frequencyPenalty: Double?
    public let logitBias: JSONValue?
    public let logprobs: Int?
    public let echo: Bool?
    public let n: Int?
    public let bestOf: Int?
    public let streamOptions: JSONValue?
    public let user: String?
    public let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model
        case prompt
        case suffix
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case stop
        case seed
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case logitBias = "logit_bias"
        case logprobs
        case echo
        case n
        case bestOf = "best_of"
        case streamOptions = "stream_options"
        case user
        case stream
    }

    public func generationRequest(
        defaultModel: String,
        defaultParameters: GenerationParameters = GenerationParameters()
    ) throws -> GenerationRequest {
        guard let prompt else {
            throw APICompatibilityError.missingPrompt
        }
        let promptText = prompt.textValue ?? ""
        return GenerationRequest(
            model: model.isEmpty ? defaultModel : model,
            messages: [ChatMessage(role: .user, content: [.text(promptText)])],
            parameters: GenerationParameters(
                maxTokens: maxTokens ?? defaultParameters.maxTokens,
                temperature: temperature ?? defaultParameters.temperature,
                topP: topP ?? defaultParameters.topP,
                topK: topK ?? defaultParameters.topK,
                minP: minP ?? defaultParameters.minP,
                typicalP: defaultParameters.typicalP,
                tfsZ: defaultParameters.tfsZ,
                seed: seed ?? defaultParameters.seed,
                contextLength: defaultParameters.contextLength,
                numKeep: defaultParameters.numKeep,
                kvBits: defaultParameters.kvBits,
                kvQuantizationScheme: defaultParameters.kvQuantizationScheme,
                kvGroupSize: defaultParameters.kvGroupSize,
                quantizedKVStart: defaultParameters.quantizedKVStart,
                maxKVSize: defaultParameters.maxKVSize,
                prefillStepSize: defaultParameters.prefillStepSize,
                visionCacheSize: defaultParameters.visionCacheSize,
                quantizeActivations: defaultParameters.quantizeActivations,
                repetitionPenalty: defaultParameters.repetitionPenalty,
                repeatLastN: defaultParameters.repeatLastN,
                presencePenalty: presencePenalty ?? defaultParameters.presencePenalty,
                frequencyPenalty: frequencyPenalty ?? defaultParameters.frequencyPenalty,
                penalizeNewline: defaultParameters.penalizeNewline,
                mirostat: defaultParameters.mirostat,
                mirostatTau: defaultParameters.mirostatTau,
                mirostatEta: defaultParameters.mirostatEta,
                stopSequences: stop?.values ?? defaultParameters.stopSequences,
                keepAlive: defaultParameters.keepAlive,
                enableThinking: defaultParameters.enableThinking,
                thinkingBudget: defaultParameters.thinkingBudget
            ),
            metadata: GenerationRequestMetadata(
                rawPrompt: true,
                suffix: suffix,
                legacyContext: prompt.tokenIDs,
                logitBias: logitBias,
                logprobs: logprobs != nil,
                topLogprobs: logprobs,
                user: user,
                responseMetadata: completionMetadata,
                n: n,
                streamOptions: streamOptions
            ),
            stream: stream ?? false
        )
    }

    private var completionMetadata: JSONValue? {
        var object: [String: JSONValue] = [:]
        if let echo {
            object["echo"] = .bool(echo)
        }
        if let bestOf {
            object["best_of"] = .number(Double(bestOf))
        }
        return object.isEmpty ? nil : .object(object)
    }
}

public enum CompatibleCompletionPrompt: Codable, Equatable, Sendable {
    case string(String)
    case strings([String])
    case tokenIDs([Int])
    case tokenIDBatches([[Int]])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String].self) {
            self = .strings(value)
        } else if let value = try? container.decode([Int].self) {
            self = .tokenIDs(value)
        } else {
            self = .tokenIDBatches(try container.decode([[Int]].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .strings(let value):
            try container.encode(value)
        case .tokenIDs(let value):
            try container.encode(value)
        case .tokenIDBatches(let value):
            try container.encode(value)
        }
    }

    var textValue: String? {
        switch self {
        case .string(let value):
            return value
        case .strings(let value):
            return value.first
        case .tokenIDs, .tokenIDBatches:
            return nil
        }
    }

    var tokenIDs: [Int]? {
        switch self {
        case .tokenIDs(let value):
            return value
        case .tokenIDBatches(let value):
            return value.first
        case .string, .strings:
            return nil
        }
    }
}

public struct OpenAIResponsesRequest: Codable, Equatable, Sendable {
    public let model: String
    public let input: CompatibleResponsesInput
    public let instructions: String?
    public let maxTokens: Int?
    public let maxOutputTokens: Int?
    public let maxCompletionTokens: Int?
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let minP: Double?
    public let stop: CompatibleStop?
    public let seed: Int?
    public let repetitionPenalty: Double?
    public let presencePenalty: Double?
    public let frequencyPenalty: Double?
    public let logitBias: JSONValue?
    public let enableThinking: Bool?
    public let thinkingBudget: Int?
    public let thinkingStartToken: String?
    public let logprobs: Bool?
    public let topLogprobs: Int?
    public let resizeShape: CompatibleResizeShape?
    public let responseFormat: JSONValue?
    public let text: JSONValue?
    public let tools: [JSONValue]?
    public let toolChoice: JSONValue?
    public let adapterPath: String?
    public let draftModel: String?
    public let draftKind: String?
    public let draftBlockSize: Int?
    public let user: String?
    public let metadata: JSONValue?
    public let previousResponseID: String?
    public let include: [JSONValue]?
    public let parallelToolCalls: Bool?
    public let truncation: String?
    public let store: Bool?
    public let reasoning: JSONValue?
    public let serviceTier: String?
    public let n: Int?
    public let streamOptions: JSONValue?
    public let modalities: [JSONValue]?
    public let audio: JSONValue?
    public let prediction: JSONValue?
    public let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model
        case input
        case instructions
        case maxTokens = "max_tokens"
        case maxOutputTokens = "max_output_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case stop
        case seed
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case logitBias = "logit_bias"
        case enableThinking = "enable_thinking"
        case thinkingBudget = "thinking_budget"
        case thinkingStartToken = "thinking_start_token"
        case logprobs
        case topLogprobs = "top_logprobs"
        case resizeShape = "resize_shape"
        case responseFormat = "response_format"
        case text
        case tools
        case toolChoice = "tool_choice"
        case adapterPath = "adapter_path"
        case draftModel = "draft_model"
        case draftKind = "draft_kind"
        case draftBlockSize = "draft_block_size"
        case user
        case metadata
        case previousResponseID = "previous_response_id"
        case include
        case parallelToolCalls = "parallel_tool_calls"
        case truncation
        case store
        case reasoning
        case serviceTier = "service_tier"
        case n
        case streamOptions = "stream_options"
        case modalities
        case audio
        case prediction
        case stream
    }

    public func generationRequest(
        defaultModel: String,
        defaultParameters: GenerationParameters = GenerationParameters()
    ) throws -> GenerationRequest {
        var messages: [ChatMessage] = []
        if let instructions, !instructions.isEmpty {
            messages.append(ChatMessage(role: .system, content: [.text(instructions)]))
        }
        messages.append(contentsOf: try input.chatMessages)
        guard !messages.isEmpty else {
            throw APICompatibilityError.missingMessages
        }
        return GenerationRequest(
            model: model.isEmpty ? defaultModel : model,
            messages: messages,
            parameters: GenerationParameters(
                maxTokens: maxTokens ?? maxOutputTokens ?? maxCompletionTokens ?? defaultParameters.maxTokens,
                temperature: temperature ?? defaultParameters.temperature,
                topP: topP ?? defaultParameters.topP,
                topK: topK ?? defaultParameters.topK,
                minP: minP ?? defaultParameters.minP,
                typicalP: defaultParameters.typicalP,
                tfsZ: defaultParameters.tfsZ,
                seed: seed ?? defaultParameters.seed,
                contextLength: defaultParameters.contextLength,
                numKeep: defaultParameters.numKeep,
                kvBits: defaultParameters.kvBits,
                kvQuantizationScheme: defaultParameters.kvQuantizationScheme,
                kvGroupSize: defaultParameters.kvGroupSize,
                quantizedKVStart: defaultParameters.quantizedKVStart,
                maxKVSize: defaultParameters.maxKVSize,
                prefillStepSize: defaultParameters.prefillStepSize,
                visionCacheSize: defaultParameters.visionCacheSize,
                quantizeActivations: defaultParameters.quantizeActivations,
                repetitionPenalty: repetitionPenalty ?? defaultParameters.repetitionPenalty,
                repeatLastN: defaultParameters.repeatLastN,
                presencePenalty: presencePenalty ?? defaultParameters.presencePenalty,
                frequencyPenalty: frequencyPenalty ?? defaultParameters.frequencyPenalty,
                penalizeNewline: defaultParameters.penalizeNewline,
                mirostat: defaultParameters.mirostat,
                mirostatTau: defaultParameters.mirostatTau,
                mirostatEta: defaultParameters.mirostatEta,
                stopSequences: stop?.values ?? defaultParameters.stopSequences,
                keepAlive: defaultParameters.keepAlive,
                enableThinking: enableThinking ?? defaultParameters.enableThinking,
                thinkingBudget: thinkingBudget ?? defaultParameters.thinkingBudget
            ),
            metadata: GenerationRequestMetadata(
                responseFormat: text?["format"] ?? responseFormat ?? text,
                tools: tools,
                toolChoice: toolChoice,
                adapterPath: adapterPath,
                draftModel: draftModel,
                draftKind: draftKind,
                draftBlockSize: draftBlockSize,
                logitBias: logitBias,
                logprobs: logprobs,
                topLogprobs: topLogprobs,
                resizeShape: resizeShape?.values,
                thinkingStartToken: thinkingStartToken,
                user: user,
                responseInstructions: instructions,
                responseTruncation: truncation,
                responseMetadata: metadata,
                n: n,
                streamOptions: streamOptions,
                modalities: modalities,
                audio: audio,
                prediction: prediction,
                previousResponseID: previousResponseID,
                include: include,
                parallelToolCalls: parallelToolCalls,
                store: store,
                serviceTier: serviceTier,
                responseReasoning: reasoning
            ),
            stream: stream ?? false
        )
    }
}

private extension Dictionary where Key == String, Value == JSONValue {
    var compatibleDraftModel: String? {
        self["draft_model"]?.stringValue ?? self["draft-model"]?.stringValue
    }

    var compatibleDraftKind: String? {
        self["draft_kind"]?.stringValue ?? self["draft-kind"]?.stringValue
    }

    var compatibleDraftBlockSize: Int? {
        self["draft_block_size"]?.intValue ?? self["draft-block-size"]?.intValue
    }
}

public enum CompatibleResizeShape: Codable, Equatable, Sendable {
    case single(Int)
    case pair([Int])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Int.self) {
            self = .single(value)
        } else if let values = try? container.decode([Int].self), values.count == 1 || values.count == 2 {
            self = .pair(values)
        } else {
            throw APICompatibilityError.unsupportedContent("expected resize_shape integer or one/two integer array")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .single(let value):
            try container.encode(value)
        case .pair(let values):
            try container.encode(values)
        case .null:
            try container.encodeNil()
        }
    }

    public var values: [Int]? {
        switch self {
        case .single(let value):
            return [value, value]
        case .pair(let values):
            return values.count == 1 ? [values[0], values[0]] : values
        case .null:
            return nil
        }
    }
}

public struct OllamaEmbedRequest: Codable, Equatable, Sendable {
    public let model: String
    public let input: CompatibleEmbeddingInput
    public let truncate: Bool?
    public let options: [String: JSONValue]?
    public let keepAlive: CompatibleKeepAlive?

    enum CodingKeys: String, CodingKey {
        case model
        case input
        case truncate
        case options
        case keepAlive = "keep_alive"
    }

    public func embeddingRequest(
        defaultModel: String,
        defaultParameters: GenerationParameters = GenerationParameters()
    ) -> EmbeddingRequest {
        input.embeddingRequest(
            model: model.isEmpty ? defaultModel : model,
            truncate: truncate,
            parameters: GenerationParameters(
                options: options,
                defaults: defaultParameters,
                keepAlive: keepAlive?.value
            )
        )
    }
}

public struct OllamaEmbeddingsRequest: Codable, Equatable, Sendable {
    public let model: String
    public let prompt: String
    public let options: [String: JSONValue]?
    public let keepAlive: CompatibleKeepAlive?

    enum CodingKeys: String, CodingKey {
        case model
        case prompt
        case options
        case keepAlive = "keep_alive"
    }

    public func embeddingRequest(
        defaultModel: String,
        defaultParameters: GenerationParameters = GenerationParameters()
    ) -> EmbeddingRequest {
        EmbeddingRequest(
            model: model.isEmpty ? defaultModel : model,
            texts: [prompt],
            parameters: GenerationParameters(
                options: options,
                defaults: defaultParameters,
                keepAlive: keepAlive?.value
            )
        )
    }
}

public struct OpenAIEmbeddingRequest: Codable, Equatable, Sendable {
    public let model: String
    public let input: CompatibleEmbeddingInput

    public func embeddingRequest(defaultModel: String) -> EmbeddingRequest {
        input.embeddingRequest(
            model: model.isEmpty ? defaultModel : model,
            truncate: nil,
            parameters: GenerationParameters()
        )
    }
}

public enum CompatibleStop: Codable, Equatable, Sendable {
    case string(String)
    case strings([String])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String].self) {
            self = .strings(value)
        } else {
            throw APICompatibilityError.unsupportedContent("expected stop string or string array")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .strings(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }

    public var values: [String] {
        switch self {
        case .string(let value):
            return [value]
        case .strings(let value):
            return value
        case .null:
            return []
        }
    }
}

public enum CompatibleEmbeddingInput: Codable, Equatable, Sendable {
    case string(String)
    case strings([String])
    case tokenIDs([Int])
    case tokenIDRows([[Int]])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String].self) {
            self = .strings(value)
        } else if let value = try? container.decode([Int].self) {
            self = .tokenIDs(value)
        } else if let value = try? container.decode([[Int]].self) {
            self = .tokenIDRows(value)
        } else {
            throw APICompatibilityError.unsupportedContent("expected embedding input string, string array, token id array, or token id row array")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .strings(let value):
            try container.encode(value)
        case .tokenIDs(let value):
            try container.encode(value)
        case .tokenIDRows(let value):
            try container.encode(value)
        }
    }

    public func embeddingRequest(
        model: String,
        truncate: Bool?,
        parameters: GenerationParameters
    ) -> EmbeddingRequest {
        switch self {
        case .string(let value):
            return EmbeddingRequest(model: model, texts: [value], truncate: truncate, parameters: parameters)
        case .strings(let values):
            return EmbeddingRequest(model: model, texts: values, truncate: truncate, parameters: parameters)
        case .tokenIDs(let values):
            return EmbeddingRequest(model: model, texts: [], tokenIDInputs: [values], truncate: truncate, parameters: parameters)
        case .tokenIDRows(let rows):
            return EmbeddingRequest(model: model, texts: [], tokenIDInputs: rows, truncate: truncate, parameters: parameters)
        }
    }
}

public enum CompatibleKeepAlive: Codable, Equatable, Sendable {
    case string(String)
    case number(Double)
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else {
            throw APICompatibilityError.unsupportedContent("expected keep_alive string or number")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }

    public var value: String? {
        switch self {
        case .string(let value):
            return value
        case .number(let value):
            if value.rounded(.towardZero) == value {
                return String(Int(value))
            }
            return String(value)
        case .null:
            return nil
        }
    }

    public var isZeroDuration: Bool {
        switch self {
        case .number(let value):
            return value == 0
        case .string(let value):
            let normalized = value
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            return ["0", "0s", "0m", "0h", "0ms", "0us", "0ns"].contains(normalized)
        case .null:
            return false
        }
    }
}

public enum CompatibleResponsesInput: Codable, Equatable, Sendable {
    case string(String)
    case messages([CompatibleChatMessage])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let messages = try? container.decode([CompatibleChatMessage].self) {
            self = .messages(messages)
        } else {
            throw APICompatibilityError.unsupportedContent("expected response input string or message array")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .messages(let messages):
            try container.encode(messages)
        }
    }

    public var chatMessages: [ChatMessage] {
        get throws {
            switch self {
            case .string(let value):
                return [ChatMessage(role: .user, content: [.text(value)])]
            case .messages(let messages):
                return try messages.map { try $0.chatMessage }
            }
        }
    }
}

public struct CompatibleChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: CompatibleMessageContent?
    public let images: [String]?
    public let reasoning: String?
    public let toolCalls: [JSONValue]?
    public let toolCallID: String?
    public let name: String?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case images
        case reasoning
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
        case name
    }

    public var chatMessage: ChatMessage {
        get throws {
            let parsedRole = MessageRole(rawValue: role) ?? .user
            var parts = content?.contentParts ?? []
            for image in images ?? [] {
                parts.append(.imageURL(ImageReference(url: image)))
            }
            return ChatMessage(
                role: parsedRole,
                content: parts,
                name: name,
                reasoning: reasoning,
                toolCalls: toolCalls?.map { $0.normalizingToolCallFunctionArguments() },
                toolCallID: toolCallID
            )
        }
    }
}

public enum CompatibleMessageContent: Codable, Equatable, Sendable {
    case string(String)
    case parts([ContentPart])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([ContentPart].self) {
            self = .parts(value)
        } else {
            throw APICompatibilityError.unsupportedContent("expected string or content part array")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .parts(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }

    public var contentParts: [ContentPart] {
        switch self {
        case .string(let value):
            return [.text(value)]
        case .parts(let value):
            return value
        case .null:
            return []
        }
    }
}

public extension GenerationParameters {
    init(options: [String: JSONValue]?) {
        self.init(options: options, defaults: GenerationParameters())
    }

    init(
        options: [String: JSONValue]?,
        defaults: GenerationParameters,
        keepAlive: String? = nil
    ) {
        self.init(
            maxTokens: options?["num_predict"]?.intValue ?? options?["max_tokens"]?.intValue ?? defaults.maxTokens,
            temperature: options?["temperature"]?.doubleValue ?? defaults.temperature,
            topP: options?["top_p"]?.doubleValue ?? defaults.topP,
            topK: options?["top_k"]?.intValue ?? defaults.topK,
            minP: options?["min_p"]?.doubleValue ?? defaults.minP,
            typicalP: options?["typical_p"]?.doubleValue ?? defaults.typicalP,
            tfsZ: options?["tfs_z"]?.doubleValue ?? defaults.tfsZ,
            seed: options?["seed"]?.intValue ?? defaults.seed,
            contextLength: options?["num_ctx"]?.intValue ?? options?["context_length"]?.intValue ?? defaults.contextLength,
            numKeep: options?["num_keep"]?.intValue ?? defaults.numKeep,
            kvBits: options?["kv_bits"]?.doubleValue ?? options?["kv-bits"]?.doubleValue ?? defaults.kvBits,
            kvQuantizationScheme: options?["kv_quant_scheme"]?.stringValue ?? options?["kv-quant-scheme"]?.stringValue ?? defaults.kvQuantizationScheme,
            kvGroupSize: options?["kv_group_size"]?.intValue ?? options?["kv-group-size"]?.intValue ?? defaults.kvGroupSize,
            quantizedKVStart: options?["quantized_kv_start"]?.intValue ?? options?["quantized-kv-start"]?.intValue ?? defaults.quantizedKVStart,
            maxKVSize: options?["max_kv_size"]?.intValue ?? options?["max-kv-size"]?.intValue ?? defaults.maxKVSize,
            prefillStepSize: options?["prefill_step_size"]?.intValue ?? options?["prefill-step-size"]?.intValue ?? defaults.prefillStepSize,
            visionCacheSize: options?["vision_cache_size"]?.intValue ?? options?["vision-cache-size"]?.intValue ?? defaults.visionCacheSize,
            quantizeActivations: options?["quantize_activations"]?.boolValue ?? options?["quantize-activations"]?.boolValue ?? defaults.quantizeActivations,
            repetitionPenalty: options?["repeat_penalty"]?.doubleValue ?? defaults.repetitionPenalty,
            repeatLastN: options?["repeat_last_n"]?.intValue ?? defaults.repeatLastN,
            presencePenalty: options?["presence_penalty"]?.doubleValue ?? defaults.presencePenalty,
            frequencyPenalty: options?["frequency_penalty"]?.doubleValue ?? defaults.frequencyPenalty,
            penalizeNewline: options?["penalize_newline"]?.boolValue ?? defaults.penalizeNewline,
            mirostat: options?["mirostat"]?.intValue ?? defaults.mirostat,
            mirostatTau: options?["mirostat_tau"]?.doubleValue ?? defaults.mirostatTau,
            mirostatEta: options?["mirostat_eta"]?.doubleValue ?? defaults.mirostatEta,
            stopSequences: options?["stop"]?.stringArrayValue ?? defaults.stopSequences,
            keepAlive: keepAlive ?? defaults.keepAlive,
            enableThinking: defaults.enableThinking,
            thinkingBudget: defaults.thinkingBudget
        )
    }
}

private extension JSONValue {
    var stringArrayValue: [String]? {
        if let stringValue {
            return [stringValue]
        }
        if let array = arrayValue {
            return array.compactMap(\.stringValue)
        }
        return nil
    }
}
