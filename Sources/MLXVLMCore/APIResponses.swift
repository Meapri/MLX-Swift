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
        let split = ThinkingOutputSplitter().split(result.text)
        self.id = id
        self.object = "chat.completion"
        self.created = created
        self.model = result.model
        self.choices = [
            OpenAIChatChoice(
                index: 0,
                message: OpenAIMessage(
                    role: "assistant",
                    content: split.content,
                    reasoning: split.reasoning,
                    toolCalls: result.toolCalls
                ),
                finishReason: result.finishReason,
                logprobs: result.logprobs.isEmpty ? nil : OpenAIChatLogprobs(logprobs: result.logprobs)
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
    public let logprobs: OpenAIChatLogprobs?

    enum CodingKeys: String, CodingKey {
        case index
        case message
        case finishReason = "finish_reason"
        case logprobs
    }

    public init(
        index: Int,
        message: OpenAIMessage,
        finishReason: String,
        logprobs: OpenAIChatLogprobs? = nil
    ) {
        self.index = index
        self.message = message
        self.finishReason = finishReason
        self.logprobs = logprobs
    }
}

public struct OpenAICompletionResponse: Codable, Equatable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [OpenAICompletionChoice]
    public let usage: OpenAIUsage

    public init(
        result: CompletedGeneration,
        id: String = "cmpl-swift",
        created: Int = Int(Date().timeIntervalSince1970)
    ) {
        self.id = id
        self.object = "text_completion"
        self.created = created
        self.model = result.model
        self.choices = [
            OpenAICompletionChoice(
                text: result.text,
                index: 0,
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

public struct OpenAICompletionChoice: Codable, Equatable, Sendable {
    public let text: String
    public let index: Int
    public let logprobs: JSONValue?
    public let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case text
        case index
        case logprobs
        case finishReason = "finish_reason"
    }

    public init(text: String, index: Int, logprobs: JSONValue? = nil, finishReason: String?) {
        self.text = text
        self.index = index
        self.logprobs = logprobs
        self.finishReason = finishReason
    }
}

public struct OpenAICompletionStreamResponse: Codable, Equatable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [OpenAICompletionChoice]
    public let usage: OpenAIUsage?

    public init(
        model: String,
        chunk: GenerationChunk,
        id: String = "cmpl-swift",
        created: Int = Int(Date().timeIntervalSince1970),
        usage: GenerationUsage? = nil
    ) {
        self.id = id
        self.object = "text_completion.chunk"
        self.created = created
        self.model = model
        self.choices = [
            OpenAICompletionChoice(
                text: chunk.text,
                index: 0,
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
                delta: OpenAIChatDelta(content: chunk.text, reasoning: chunk.reasoning, toolCalls: chunk.toolCalls),
                finishReason: chunk.isFinished ? (chunk.finishReason ?? "stop") : nil,
                logprobs: chunk.logprob.map { OpenAIChatLogprobs(logprobs: [$0]) }
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
    public let logprobs: OpenAIChatLogprobs?

    enum CodingKeys: String, CodingKey {
        case index
        case delta
        case finishReason = "finish_reason"
        case logprobs
    }

    public init(
        index: Int,
        delta: OpenAIChatDelta,
        finishReason: String?,
        logprobs: OpenAIChatLogprobs? = nil
    ) {
        self.index = index
        self.delta = delta
        self.finishReason = finishReason
        self.logprobs = logprobs
    }
}

public struct OpenAIChatLogprobs: Codable, Equatable, Sendable {
    public let content: [OpenAIChatLogprobContent]

    public init(logprobs: [GenerationTokenLogprob]) {
        self.content = logprobs.map(OpenAIChatLogprobContent.init(logprob:))
    }
}

public struct OpenAIChatLogprobContent: Codable, Equatable, Sendable {
    public let token: String
    public let logprob: Double
    public let bytes: [Int]?
    public let topLogprobs: [OpenAIChatTopLogprob]

    enum CodingKeys: String, CodingKey {
        case token
        case logprob
        case bytes
        case topLogprobs = "top_logprobs"
    }

    public init(logprob: GenerationTokenLogprob) {
        self.token = logprob.token
        self.logprob = logprob.logprob
        self.bytes = logprob.bytes
        self.topLogprobs = logprob.topLogprobs.map(OpenAIChatTopLogprob.init(logprob:))
    }
}

public struct OpenAIChatTopLogprob: Codable, Equatable, Sendable {
    public let token: String
    public let logprob: Double
    public let bytes: [Int]?

    public init(logprob: GenerationTopLogprob) {
        self.token = logprob.token
        self.logprob = logprob.logprob
        self.bytes = logprob.bytes
    }
}

public struct OpenAIChatDelta: Codable, Equatable, Sendable {
    public let role: String?
    public let content: String?
    public let reasoning: String?
    public let toolCalls: [OpenAIToolCall]?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case reasoning
        case toolCalls = "tool_calls"
    }

    public init(
        role: String? = nil,
        content: String?,
        reasoning: String? = nil,
        toolCalls: [GenerationToolCall] = []
    ) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls.isEmpty ? nil : toolCalls.map { OpenAIToolCall(toolCall: $0) }
    }
}

public struct OpenAIMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: String?
    public let reasoning: String?
    public let toolCalls: [OpenAIToolCall]?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case reasoning
        case toolCalls = "tool_calls"
    }

    public init(
        role: String,
        content: String,
        reasoning: String? = nil,
        toolCalls: [GenerationToolCall] = []
    ) {
        self.role = role
        self.content = content.isEmpty && !toolCalls.isEmpty ? nil : content
        self.reasoning = reasoning
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
        try container.encodeIfPresent(reasoning, forKey: .reasoning)
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
    public let instructions: String?
    public let maxOutputTokens: Int?
    public let model: String
    public let output: [OpenAIResponseOutput]
    public let outputText: String
    public let temperature: Double?
    public let topP: Double?
    public let truncation: String
    public let usage: OpenAIUsage
    public let user: String?
    public let metadata: JSONValue?

    enum CodingKeys: String, CodingKey {
        case id
        case object
        case createdAt = "created_at"
        case status
        case instructions
        case maxOutputTokens = "max_output_tokens"
        case model
        case output
        case outputText = "output_text"
        case temperature
        case topP = "top_p"
        case truncation
        case usage
        case user
        case metadata
    }

    public init(
        result: CompletedGeneration,
        request: GenerationRequest? = nil,
        id: String = "resp-swift",
        createdAt: Int = Int(Date().timeIntervalSince1970)
    ) {
        let split = ThinkingOutputSplitter().split(result.text)
        var output: [OpenAIResponseOutput] = []
        if !split.content.isEmpty || split.reasoning?.isEmpty == false || result.toolCalls.isEmpty {
            output.append(
                OpenAIResponseOutput(
                    id: "\(id)-msg",
                    type: "message",
                    status: "completed",
                    role: "assistant",
                    content: [
                        OpenAIResponseContent(
                            type: "output_text",
                            text: split.content
                        )
                    ],
                    reasoning: split.reasoning
                )
            )
        }
        output += result.toolCalls.map(OpenAIResponseOutput.init(toolCall:))

        self.id = id
        self.object = "response"
        self.createdAt = createdAt
        self.status = "completed"
        self.instructions = request?.metadata.responseInstructions
        self.maxOutputTokens = request?.parameters.maxTokens
        self.model = result.model
        self.output = output
        self.outputText = split.content
        self.temperature = request?.parameters.temperature
        self.topP = request?.parameters.topP
        self.truncation = request?.metadata.responseTruncation ?? "disabled"
        self.usage = OpenAIUsage(
            promptTokens: result.usage.promptTokens,
            completionTokens: result.usage.completionTokens,
            totalTokens: result.usage.totalTokens
        )
        self.user = request?.metadata.user
        self.metadata = request?.metadata.responseMetadata
    }
}

public struct OpenAIResponseOutput: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let status: String
    public let role: String?
    public let content: [OpenAIResponseContent]?
    public let reasoning: String?
    public let callID: String?
    public let name: String?
    public let arguments: String?

    enum CodingKeys: String, CodingKey {
        case id
        case type
        case status
        case role
        case content
        case reasoning
        case callID = "call_id"
        case name
        case arguments
    }

    public init(
        id: String,
        type: String,
        status: String,
        role: String,
        content: [OpenAIResponseContent],
        reasoning: String? = nil
    ) {
        self.id = id
        self.type = type
        self.status = status
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.callID = nil
        self.name = nil
        self.arguments = nil
    }

    public init(toolCall: GenerationToolCall) {
        self.id = toolCall.id
        self.type = "function_call"
        self.status = "completed"
        self.role = nil
        self.content = nil
        self.reasoning = nil
        self.callID = toolCall.id
        self.name = toolCall.function.name
        self.arguments = Self.argumentsJSONString(toolCall.function.arguments)
    }

    private static func argumentsJSONString(_ arguments: [String: JSONValue]) -> String {
        let data = (try? ResponseStreamFramer.streamEncoder().encode(arguments)) ?? Data("{}".utf8)
        return String(decoding: data, as: UTF8.self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(type, forKey: .type)
        try container.encode(status, forKey: .status)
        try container.encodeIfPresent(role, forKey: .role)
        try container.encodeIfPresent(content, forKey: .content)
        try container.encodeIfPresent(reasoning, forKey: .reasoning)
        try container.encodeIfPresent(callID, forKey: .callID)
        try container.encodeIfPresent(name, forKey: .name)
        try container.encodeIfPresent(arguments, forKey: .arguments)
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

    public static func namedServerSentEvent<T: Encodable>(
        event: String,
        data value: T,
        encoder: JSONEncoder = streamEncoder()
    ) -> String {
        "event: \(event)\ndata: \(String(decoding: try! encoder.encode(value), as: UTF8.self))\n\n"
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

public enum OpenAIResponsesStreamFramer {
    public static func initialFrames(
        id: String = "resp-swift",
        messageID: String = "resp-swift-msg",
        model: String,
        createdAt: Int = Int(Date().timeIntervalSince1970),
        request: GenerationRequest? = nil
    ) -> [String] {
        let response = responseObject(
            id: id,
            model: model,
            createdAt: createdAt,
            status: "in_progress",
            output: [],
            outputText: "",
            usage: GenerationUsage(),
            request: request
        )
        let message = messageItem(id: messageID, status: "in_progress", content: [])
        let emptyPart = outputTextPart(text: "")
        return [
            event("response.created", [
                "type": .string("response.created"),
                "response": response,
            ]),
            event("response.in_progress", [
                "type": .string("response.in_progress"),
                "response": response,
            ]),
            event("response.output_item.added", [
                "type": .string("response.output_item.added"),
                "output_index": .number(0),
                "item": .object(message),
            ]),
            event("response.content_part.added", [
                "type": .string("response.content_part.added"),
                "item_id": .string(messageID),
                "output_index": .number(0),
                "content_index": .number(0),
                "part": .object(emptyPart),
            ]),
        ]
    }

    public static func deltaFrame(
        delta: String,
        messageID: String = "resp-swift-msg"
    ) -> String {
        event("response.output_text.delta", [
            "type": .string("response.output_text.delta"),
            "item_id": .string(messageID),
            "output_index": .number(0),
            "content_index": .number(0),
            "delta": .string(delta),
        ])
    }

    public static func finalFrames(
        id: String = "resp-swift",
        messageID: String = "resp-swift-msg",
        model: String,
        createdAt: Int = Int(Date().timeIntervalSince1970),
        text: String,
        usage: GenerationUsage,
        toolCalls: [GenerationToolCall] = [],
        request: GenerationRequest? = nil
    ) -> [String] {
        let split = ThinkingOutputSplitter().split(text)
        let cleanText = split.content
        let contentPart = outputTextPart(text: cleanText)
        let message = messageItem(
            id: messageID,
            status: "completed",
            content: [.object(contentPart)]
        )
        let functionItems = toolCalls.map(functionCallItem(from:))
        let output = [.object(message)] + functionItems.map(JSONValue.object)
        let response = responseObject(
            id: id,
            model: model,
            createdAt: createdAt,
            status: "completed",
            output: output,
            outputText: cleanText,
            usage: usage,
            request: request
        )
        let functionEvents = functionItems.enumerated().flatMap { index, item in
            let outputIndex = index + 1
            return [
                event("response.output_item.added", [
                    "type": .string("response.output_item.added"),
                    "output_index": .number(Double(outputIndex)),
                    "item": .object(item),
                ]),
                event("response.function_call_arguments.done", [
                    "type": .string("response.function_call_arguments.done"),
                    "item_id": item["id"] ?? .string(""),
                    "output_index": .number(Double(outputIndex)),
                    "arguments": item["arguments"] ?? .string("{}"),
                ]),
                event("response.output_item.done", [
                    "type": .string("response.output_item.done"),
                    "output_index": .number(Double(outputIndex)),
                    "item": .object(item),
                ]),
            ]
        }

        return [
            event("response.output_text.done", [
                "type": .string("response.output_text.done"),
                "item_id": .string(messageID),
                "output_index": .number(0),
                "content_index": .number(0),
                "text": .string(cleanText),
            ]),
            event("response.content_part.done", [
                "type": .string("response.content_part.done"),
                "item_id": .string(messageID),
                "output_index": .number(0),
                "content_index": .number(0),
                "part": .object(contentPart),
            ]),
            event("response.output_item.done", [
                "type": .string("response.output_item.done"),
                "output_index": .number(0),
                "item": .object(message),
            ]),
        ] + functionEvents + [
            event("response.completed", [
                "type": .string("response.completed"),
                "response": response,
            ]),
            ResponseStreamFramer.doneServerSentEvent(),
        ]
    }

    private static func event(_ name: String, _ data: [String: JSONValue]) -> String {
        ResponseStreamFramer.namedServerSentEvent(event: name, data: JSONValue.object(data))
    }

    private static func outputTextPart(text: String) -> [String: JSONValue] {
        [
            "type": .string("output_text"),
            "text": .string(text),
            "annotations": .array([]),
        ]
    }

    private static func messageItem(
        id: String,
        status: String,
        content: [JSONValue]
    ) -> [String: JSONValue] {
        [
            "id": .string(id),
            "type": .string("message"),
            "status": .string(status),
            "role": .string("assistant"),
            "content": .array(content),
        ]
    }

    private static func functionCallItem(from toolCall: GenerationToolCall) -> [String: JSONValue] {
        [
            "id": .string(toolCall.id),
            "type": .string("function_call"),
            "status": .string("completed"),
            "call_id": .string(toolCall.id),
            "name": .string(toolCall.function.name),
            "arguments": .string(argumentsJSONString(toolCall.function.arguments)),
        ]
    }

    private static func argumentsJSONString(_ arguments: [String: JSONValue]) -> String {
        let data = (try? ResponseStreamFramer.streamEncoder().encode(arguments)) ?? Data("{}".utf8)
        return String(decoding: data, as: UTF8.self)
    }

    private static func responseObject(
        id: String,
        model: String,
        createdAt: Int,
        status: String,
        output: [JSONValue],
        outputText: String,
        usage: GenerationUsage,
        request: GenerationRequest? = nil
    ) -> JSONValue {
        var response: [String: JSONValue] = [
            "id": .string(id),
            "object": .string("response"),
            "created_at": .number(Double(createdAt)),
            "status": .string(status),
            "model": .string(model),
            "output": .array(output),
            "output_text": .string(outputText),
            "truncation": .string(request?.metadata.responseTruncation ?? "disabled"),
            "usage": .object([
                "input_tokens": .number(Double(usage.promptTokens)),
                "output_tokens": .number(Double(usage.completionTokens)),
                "total_tokens": .number(Double(usage.totalTokens)),
            ]),
        ]
        if let instructions = request?.metadata.responseInstructions {
            response["instructions"] = .string(instructions)
        }
        if let maxOutputTokens = request?.parameters.maxTokens {
            response["max_output_tokens"] = .number(Double(maxOutputTokens))
        }
        if let temperature = request?.parameters.temperature {
            response["temperature"] = .number(temperature)
        }
        if let topP = request?.parameters.topP {
            response["top_p"] = .number(topP)
        }
        if let user = request?.metadata.user {
            response["user"] = .string(user)
        }
        if let metadata = request?.metadata.responseMetadata {
            response["metadata"] = metadata
        }
        return .object(response)
    }
}

public enum GenerationResponseAPI: String, Codable, Equatable, Sendable {
    case ollamaGenerate = "ollama-generate"
    case ollamaChat = "ollama-chat"
    case openAICompletions = "openai-completions"
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
        chunks: [GenerationChunk] = [],
        request: GenerationRequest? = nil
    ) -> GenerationAPIResponseRenderReport {
        if stream {
            return renderStream(
                model: result.model,
                chunks: chunks.isEmpty ? [GenerationChunk(text: result.text, isFinished: true, finishReason: result.finishReason)] : chunks,
                usage: result.usage,
                api: api,
                request: request
            )
        }

        let body: String
        switch api {
        case .ollamaGenerate:
            body = encode(OllamaGenerateResponse(result: result))
        case .ollamaChat:
            body = encode(OllamaChatResponse(result: result))
        case .openAICompletions:
            body = encode(OpenAICompletionResponse(result: result))
        case .openAIChat:
            body = encode(OpenAIChatCompletionResponse(result: result))
        case .openAIResponses:
            body = encode(OpenAIResponsesResponse(result: result, request: request))
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
        api: GenerationResponseAPI,
        request: GenerationRequest? = nil
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
        case .openAICompletions:
            contentType = "text/event-stream"
            frames = chunks.map {
                ResponseStreamFramer.serverSentEvent(
                    OpenAICompletionStreamResponse(
                        model: model,
                        chunk: $0,
                        usage: $0.isFinished ? usage : nil
                    )
                )
            } + [ResponseStreamFramer.doneServerSentEvent()]
        case .openAIChat:
            contentType = "text/event-stream"
            frames = openAIChatStreamChunks(chunks).map {
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
            let toolCalls = chunks.flatMap(\.toolCalls)
            frames = OpenAIResponsesStreamFramer.initialFrames(model: model, request: request)
                + chunks.filter { !$0.isFinished && !$0.text.isEmpty }.map {
                    OpenAIResponsesStreamFramer.deltaFrame(delta: $0.text)
                }
                + OpenAIResponsesStreamFramer.finalFrames(
                    model: model,
                    text: text,
                    usage: usage,
                    toolCalls: toolCalls,
                    request: request
                )
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

    private static func openAIChatStreamChunks(_ chunks: [GenerationChunk]) -> [GenerationChunk] {
        var splitter = StreamingThinkingOutputSplitter()
        var output = chunks.flatMap { splitter.process($0) }
        output.append(contentsOf: splitter.finish())
        return output
    }
}
