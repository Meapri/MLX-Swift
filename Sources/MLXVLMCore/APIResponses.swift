import Foundation

public struct OllamaGenerateResponse: Codable, Equatable, Sendable {
    public let model: String
    public let createdAt: String
    public let response: String
    public let done: Bool
    public let totalDuration: Int?
    public let loadDuration: Int?
    public let promptEvalCount: Int
    public let evalCount: Int
    public let doneReason: String?

    enum CodingKeys: String, CodingKey {
        case model
        case createdAt = "created_at"
        case response
        case done
        case totalDuration = "total_duration"
        case loadDuration = "load_duration"
        case promptEvalCount = "prompt_eval_count"
        case evalCount = "eval_count"
        case doneReason = "done_reason"
    }

    public init(
        result: CompletedGeneration,
        createdAt: String = Self.defaultCreatedAt(),
        totalDuration: Int? = nil,
        loadDuration: Int? = nil,
        doneReason: String? = nil
    ) {
        self.model = result.model
        self.createdAt = createdAt
        self.response = result.text
        self.done = true
        self.totalDuration = totalDuration
        self.loadDuration = loadDuration
        self.promptEvalCount = result.usage.promptTokens
        self.evalCount = result.usage.completionTokens
        self.doneReason = doneReason
    }

    public static func defaultCreatedAt() -> String {
        ISO8601DateFormatter().string(from: Date())
    }
}

public struct OllamaGenerateStreamChunk: Codable, Equatable, Sendable {
    public let model: String
    public let createdAt: String
    public let response: String
    public let done: Bool
    public let promptEvalCount: Int?
    public let evalCount: Int?
    public let doneReason: String?

    enum CodingKeys: String, CodingKey {
        case model
        case createdAt = "created_at"
        case response
        case done
        case promptEvalCount = "prompt_eval_count"
        case evalCount = "eval_count"
        case doneReason = "done_reason"
    }

    public init(
        model: String,
        chunk: GenerationChunk,
        usage: GenerationUsage? = nil,
        createdAt: String = OllamaGenerateResponse.defaultCreatedAt()
    ) {
        self.model = model
        self.createdAt = createdAt
        self.response = chunk.text
        self.done = chunk.isFinished
        self.promptEvalCount = chunk.isFinished ? usage?.promptTokens : nil
        self.evalCount = chunk.isFinished ? usage?.completionTokens : nil
        self.doneReason = chunk.isFinished ? chunk.finishReason : nil
    }
}

public struct OllamaChatResponse: Codable, Equatable, Sendable {
    public let model: String
    public let createdAt: String
    public let message: OllamaChatMessage
    public let done: Bool
    public let promptEvalCount: Int
    public let evalCount: Int

    enum CodingKeys: String, CodingKey {
        case model
        case createdAt = "created_at"
        case message
        case done
        case promptEvalCount = "prompt_eval_count"
        case evalCount = "eval_count"
    }

    public init(result: CompletedGeneration, createdAt: String = ISO8601DateFormatter().string(from: Date())) {
        self.model = result.model
        self.createdAt = createdAt
        self.message = OllamaChatMessage(role: "assistant", content: result.text, toolCalls: result.toolCalls)
        self.done = true
        self.promptEvalCount = result.usage.promptTokens
        self.evalCount = result.usage.completionTokens
    }
}

public struct OllamaChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: String
    public let toolCalls: [OllamaToolCall]?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
    }

    public init(role: String, content: String, toolCalls: [GenerationToolCall] = []) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls.isEmpty ? nil : toolCalls.map { OllamaToolCall(toolCall: $0) }
    }
}

public struct OllamaChatStreamChunk: Codable, Equatable, Sendable {
    public let model: String
    public let createdAt: String
    public let message: OllamaChatMessage
    public let done: Bool
    public let promptEvalCount: Int?
    public let evalCount: Int?
    public let doneReason: String?

    enum CodingKeys: String, CodingKey {
        case model
        case createdAt = "created_at"
        case message
        case done
        case promptEvalCount = "prompt_eval_count"
        case evalCount = "eval_count"
        case doneReason = "done_reason"
    }

    public init(
        model: String,
        chunk: GenerationChunk,
        usage: GenerationUsage? = nil,
        createdAt: String = OllamaGenerateResponse.defaultCreatedAt()
    ) {
        self.model = model
        self.createdAt = createdAt
        self.message = OllamaChatMessage(role: "assistant", content: chunk.text, toolCalls: chunk.toolCalls)
        self.done = chunk.isFinished
        self.promptEvalCount = chunk.isFinished ? usage?.promptTokens : nil
        self.evalCount = chunk.isFinished ? usage?.completionTokens : nil
        self.doneReason = chunk.isFinished ? chunk.finishReason : nil
    }
}

public struct OpenAIChatCompletionResponse: Codable, Equatable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [OpenAIChatChoice]
    public let usage: OpenAIUsage

    public init(
        result: CompletedGeneration,
        id: String = "chatcmpl-swift",
        created: Int = Int(Date().timeIntervalSince1970)
    ) {
        self.id = id
        self.object = "chat.completion"
        self.created = created
        self.model = result.model
        self.choices = [
            OpenAIChatChoice(
                index: 0,
                message: OpenAIMessage(
                    role: "assistant",
                    content: result.text,
                    toolCalls: result.toolCalls
                ),
                finishReason: result.finishReason
            )
        ]
        self.usage = OpenAIUsage(
            promptTokens: result.usage.promptTokens,
            completionTokens: result.usage.completionTokens,
            totalTokens: result.usage.totalTokens
        )
    }
}

public struct OpenAIChatChoice: Codable, Equatable, Sendable {
    public let index: Int
    public let message: OpenAIMessage
    public let finishReason: String

    enum CodingKeys: String, CodingKey {
        case index
        case message
        case finishReason = "finish_reason"
    }

    public init(index: Int, message: OpenAIMessage, finishReason: String) {
        self.index = index
        self.message = message
        self.finishReason = finishReason
    }
}

public struct OpenAIChatCompletionStreamResponse: Codable, Equatable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [OpenAIChatStreamChoice]
    public let usage: OpenAIUsage?

    public init(
        model: String,
        chunk: GenerationChunk,
        id: String = "chatcmpl-swift",
        created: Int = Int(Date().timeIntervalSince1970),
        usage: GenerationUsage? = nil
    ) {
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = created
        self.model = model
        self.choices = [
            OpenAIChatStreamChoice(
                index: 0,
                delta: OpenAIChatDelta(content: chunk.text, toolCalls: chunk.toolCalls),
                finishReason: chunk.isFinished ? (chunk.finishReason ?? "stop") : nil
            )
        ]
        self.usage = chunk.isFinished
            ? usage.map {
                OpenAIUsage(
                    promptTokens: $0.promptTokens,
                    completionTokens: $0.completionTokens,
                    totalTokens: $0.totalTokens
                )
            }
            : nil
    }
}

public struct OpenAIChatStreamChoice: Codable, Equatable, Sendable {
    public let index: Int
    public let delta: OpenAIChatDelta
    public let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index
        case delta
        case finishReason = "finish_reason"
    }

    public init(index: Int, delta: OpenAIChatDelta, finishReason: String?) {
        self.index = index
        self.delta = delta
        self.finishReason = finishReason
    }
}

public struct OpenAIChatDelta: Codable, Equatable, Sendable {
    public let role: String?
    public let content: String?
    public let toolCalls: [OpenAIToolCall]?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
    }

    public init(role: String? = nil, content: String?, toolCalls: [GenerationToolCall] = []) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls.isEmpty ? nil : toolCalls.map { OpenAIToolCall(toolCall: $0) }
    }
}

public struct OpenAIMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: String?
    public let toolCalls: [OpenAIToolCall]?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
    }

    public init(role: String, content: String, toolCalls: [GenerationToolCall] = []) {
        self.role = role
        self.content = content.isEmpty && !toolCalls.isEmpty ? nil : content
        self.toolCalls = toolCalls.isEmpty ? nil : toolCalls.map { OpenAIToolCall(toolCall: $0) }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        if let content {
            try container.encode(content, forKey: .content)
        } else {
            try container.encodeNil(forKey: .content)
        }
        try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
    }
}

public struct OpenAIToolCall: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let function: OpenAIToolCallFunction

    public init(toolCall: GenerationToolCall) {
        self.id = toolCall.id
        self.type = toolCall.type
        self.function = OpenAIToolCallFunction(function: toolCall.function)
    }
}

public struct OpenAIToolCallFunction: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: String

    public init(function: GenerationToolCallFunction) {
        self.name = function.name
        self.arguments = Self.argumentsJSONString(function.arguments)
    }

    private static func argumentsJSONString(_ arguments: [String: JSONValue]) -> String {
        let data = (try? ResponseStreamFramer.streamEncoder().encode(arguments)) ?? Data("{}".utf8)
        return String(decoding: data, as: UTF8.self)
    }
}

public struct OllamaToolCall: Codable, Equatable, Sendable {
    public let function: OllamaToolCallFunction

    public init(toolCall: GenerationToolCall) {
        self.function = OllamaToolCallFunction(function: toolCall.function)
    }
}

public struct OllamaToolCallFunction: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: [String: JSONValue]

    public init(function: GenerationToolCallFunction) {
        self.name = function.name
        self.arguments = function.arguments
    }
}

public struct OpenAIUsage: Codable, Equatable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }

    public init(promptTokens: Int, completionTokens: Int, totalTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
    }
}

public struct OpenAIModelListResponse: Codable, Equatable, Sendable {
    public let object: String
    public let data: [OpenAIModelResponse]

    public init(models: [OpenAIModelResponse]) {
        self.object = "list"
        self.data = models
    }
}

public struct OpenAIModelResponse: Codable, Equatable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let ownedBy: String

    enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case ownedBy = "owned_by"
    }

    public init(id: String, created: Int = 0, ownedBy: String = "mlx-vlm-swift") {
        self.id = id
        self.object = "model"
        self.created = created
        self.ownedBy = ownedBy
    }
}

public struct OpenAIResponsesResponse: Codable, Equatable, Sendable {
    public let id: String
    public let object: String
    public let createdAt: Int
    public let status: String
    public let model: String
    public let output: [OpenAIResponseOutput]
    public let outputText: String
    public let usage: OpenAIUsage

    enum CodingKeys: String, CodingKey {
        case id
        case object
        case createdAt = "created_at"
        case status
        case model
        case output
        case outputText = "output_text"
        case usage
    }

    public init(
        result: CompletedGeneration,
        id: String = "resp-swift",
        createdAt: Int = Int(Date().timeIntervalSince1970)
    ) {
        self.id = id
        self.object = "response"
        self.createdAt = createdAt
        self.status = "completed"
        self.model = result.model
        self.output = [
            OpenAIResponseOutput(
                id: "\(id)-msg",
                type: "message",
                status: "completed",
                role: "assistant",
                content: [
                    OpenAIResponseContent(
                        type: "output_text",
                        text: result.text
                    )
                ]
            )
        ]
        self.outputText = result.text
        self.usage = OpenAIUsage(
            promptTokens: result.usage.promptTokens,
            completionTokens: result.usage.completionTokens,
            totalTokens: result.usage.totalTokens
        )
    }
}

public struct OpenAIResponseOutput: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let status: String
    public let role: String
    public let content: [OpenAIResponseContent]

    public init(
        id: String,
        type: String,
        status: String,
        role: String,
        content: [OpenAIResponseContent]
    ) {
        self.id = id
        self.type = type
        self.status = status
        self.role = role
        self.content = content
    }
}

public struct OpenAIResponseContent: Codable, Equatable, Sendable {
    public let type: String
    public let text: String

    public init(type: String, text: String) {
        self.type = type
        self.text = text
    }
}

public enum ResponseStreamFramer {
    public static func jsonLine<T: Encodable>(_ value: T, encoder: JSONEncoder = streamEncoder()) -> String {
        String(decoding: try! encoder.encode(value), as: UTF8.self) + "\n"
    }

    public static func serverSentEvent<T: Encodable>(
        _ value: T,
        encoder: JSONEncoder = streamEncoder()
    ) -> String {
        "data: \(String(decoding: try! encoder.encode(value), as: UTF8.self))\n\n"
    }

    public static func doneServerSentEvent() -> String {
        "data: [DONE]\n\n"
    }

    public static func streamEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}

public enum GenerationResponseAPI: String, Codable, Equatable, Sendable {
    case ollamaGenerate = "ollama-generate"
    case ollamaChat = "ollama-chat"
    case openAIChat = "openai-chat"
    case openAIResponses = "openai-responses"
}

public struct OllamaEmbedResponse: Codable, Equatable, Sendable {
    public let model: String
    public let embeddings: [[Float]]
    public let promptEvalCount: Int

    enum CodingKeys: String, CodingKey {
        case model
        case embeddings
        case promptEvalCount = "prompt_eval_count"
    }

    public init(result: CompletedEmbedding) {
        self.model = result.model
        self.embeddings = result.embeddings
        self.promptEvalCount = result.promptTokenCount
    }
}

public struct OllamaEmbeddingsResponse: Codable, Equatable, Sendable {
    public let embedding: [Float]

    public init(result: CompletedEmbedding) {
        self.embedding = result.embeddings.first ?? []
    }
}

public struct OpenAIEmbeddingResponse: Codable, Equatable, Sendable {
    public struct Item: Codable, Equatable, Sendable {
        public let object: String
        public let embedding: [Float]
        public let index: Int
    }

    public struct Usage: Codable, Equatable, Sendable {
        public let promptTokens: Int
        public let totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case totalTokens = "total_tokens"
        }
    }

    public let object: String
    public let data: [Item]
    public let model: String
    public let usage: Usage

    public init(result: CompletedEmbedding) {
        self.object = "list"
        self.data = result.embeddings.enumerated().map { index, embedding in
            Item(object: "embedding", embedding: embedding, index: index)
        }
        self.model = result.model
        self.usage = Usage(
            promptTokens: result.promptTokenCount,
            totalTokens: result.promptTokenCount
        )
    }
}

public struct GenerationAPIResponseRenderReport: Codable, Equatable, Sendable {
    public let api: GenerationResponseAPI
    public let stream: Bool
    public let contentType: String
    public let body: String
    public let frameCount: Int

    public init(
        api: GenerationResponseAPI,
        stream: Bool,
        contentType: String,
        body: String,
        frameCount: Int
    ) {
        self.api = api
        self.stream = stream
        self.contentType = contentType
        self.body = body
        self.frameCount = frameCount
    }
}

public enum GenerationAPIResponseRenderer {
    public static func renderCompleted(
        _ result: CompletedGeneration,
        api: GenerationResponseAPI,
        stream: Bool = false,
        chunks: [GenerationChunk] = []
    ) -> GenerationAPIResponseRenderReport {
        if stream {
            return renderStream(
                model: result.model,
                chunks: chunks.isEmpty ? [GenerationChunk(text: result.text, isFinished: true, finishReason: result.finishReason)] : chunks,
                usage: result.usage,
                api: api
            )
        }

        let body: String
        switch api {
        case .ollamaGenerate:
            body = encode(OllamaGenerateResponse(result: result))
        case .ollamaChat:
            body = encode(OllamaChatResponse(result: result))
        case .openAIChat:
            body = encode(OpenAIChatCompletionResponse(result: result))
        case .openAIResponses:
            body = encode(OpenAIResponsesResponse(result: result))
        }
        return GenerationAPIResponseRenderReport(
            api: api,
            stream: false,
            contentType: "application/json",
            body: body,
            frameCount: 1
        )
    }

    public static func renderStream(
        model: String,
        chunks: [GenerationChunk],
        usage: GenerationUsage,
        api: GenerationResponseAPI
    ) -> GenerationAPIResponseRenderReport {
        let frames: [String]
        let contentType: String
        switch api {
        case .ollamaGenerate:
            contentType = "application/x-ndjson"
            frames = chunks.map {
                ResponseStreamFramer.jsonLine(
                    OllamaGenerateStreamChunk(model: model, chunk: $0, usage: $0.isFinished ? usage : nil)
                )
            }
        case .ollamaChat:
            contentType = "application/x-ndjson"
            frames = chunks.map {
                ResponseStreamFramer.jsonLine(
                    OllamaChatStreamChunk(model: model, chunk: $0, usage: $0.isFinished ? usage : nil)
                )
            }
        case .openAIChat:
            contentType = "text/event-stream"
            frames = chunks.map {
                ResponseStreamFramer.serverSentEvent(
                    OpenAIChatCompletionStreamResponse(
                        model: model,
                        chunk: $0,
                        usage: $0.isFinished ? usage : nil
                    )
                )
            } + [ResponseStreamFramer.doneServerSentEvent()]
        case .openAIResponses:
            contentType = "text/event-stream"
            let text = chunks.map(\.text).joined()
            let finishReason = chunks.last(where: \.isFinished)?.finishReason ?? "stop"
            let result = CompletedGeneration(model: model, text: text, finishReason: finishReason, usage: usage)
            frames = [
                ResponseStreamFramer.serverSentEvent(OpenAIResponsesResponse(result: result)),
                ResponseStreamFramer.doneServerSentEvent(),
            ]
        }
        return GenerationAPIResponseRenderReport(
            api: api,
            stream: true,
            contentType: contentType,
            body: frames.joined(),
            frameCount: frames.count
        )
    }

    private static func encode<T: Encodable>(_ value: T) -> String {
        String(decoding: try! ResponseStreamFramer.streamEncoder().encode(value), as: UTF8.self)
    }
}
