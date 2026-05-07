import Foundation
import MLXVLMCore

#if MLXVLM_REAL_MLX_API && canImport(MLXLMCommon) && canImport(MLXVLM) && canImport(MLXLMTokenizers)
import MLXLMCommon
import MLXLMTokenizers
import MLXVLM

public enum MLXVLMUpstreamBackendError: Error, CustomStringConvertible, Sendable {
    case unsupportedMessageRole(MessageRole)
    case unsupportedMediaReference(String)
    case unsupportedToolPayload
    case unsupportedToolSpec(MLXVLMCore.JSONValue)

    public var description: String {
        switch self {
        case .unsupportedMessageRole(let role):
            return "Unsupported chat message role for mlx-swift-lm Chat.Message: \(role.rawValue)"
        case .unsupportedMediaReference(let reference):
            return "Unsupported media reference for mlx-swift-lm UserInput: \(reference)"
        case .unsupportedToolPayload:
            return "Tool payload bridging to mlx-swift-lm ToolSpec is not implemented yet."
        case .unsupportedToolSpec(let value):
            return "Unsupported tool schema for mlx-swift-lm ToolSpec: \(value)"
        }
    }
}

public struct MLXVLMUpstreamModelContainer: MLXVLMCore.ModelContainer {
    public let context: ModelLoadContext
    public let upstream: MLXLMCommon.ModelContainer

    public init(context: ModelLoadContext, upstream: MLXLMCommon.ModelContainer) {
        self.context = context
        self.upstream = upstream
    }
}

public struct MLXVLMUpstreamProcessor: VLMProcessor {
    public let compatibilityProcessor: CompatibilityProcessor

    public init(compatibilityProcessor: CompatibilityProcessor = CompatibilityProcessor(backend: .mlxSwiftVLM)) {
        self.compatibilityProcessor = compatibilityProcessor
    }

    public func process(request: GenerationRequest, context: ModelLoadContext) throws -> ProcessedGenerationInput {
        try compatibilityProcessor.process(request: request, context: context)
    }
}

public struct MLXVLMUpstreamGenerator: VLMGenerator {
    public let container: MLXVLMUpstreamModelContainer

    public init(container: MLXVLMUpstreamModelContainer) {
        self.container = container
    }

    public func generate(input: ProcessedGenerationInput) async throws -> AsyncThrowingStream<GenerationChunk, Error> {
        let userInput = try MLXVLMUserInputBridge.userInput(from: input.request)
        let lmInput = try await container.upstream.prepare(input: userInput)
        let parameters = MLXGenerateParametersBridge.makeGenerateParameters(
            from: MLXGenerateParametersPlan(runtime: input.runtime)
        )
        let upstreamStream = try await container.upstream.generate(input: lmInput, parameters: parameters)

        return AsyncThrowingStream { continuation in
            Task {
                var emittedToolCall = false
                for await event in upstreamStream {
                    switch event {
                    case .chunk(let text):
                        continuation.yield(GenerationChunk(text: text))
                    case .toolCall(let call):
                        emittedToolCall = true
                        continuation.yield(GenerationChunk(text: "", toolCalls: [Self.toolCall(from: call)]))
                    case .info(let info):
                        continuation.yield(
                            GenerationChunk(
                                text: "",
                                isFinished: true,
                                finishReason: emittedToolCall ? "tool_calls" : finishReason(from: info.stopReason),
                                promptTokenCount: info.promptTokenCount,
                                completionTokenCount: info.generationTokenCount
                            )
                        )
                    }
                }
                continuation.finish()
            }
        }
    }

    private func finishReason(from stopReason: GenerateStopReason) -> String {
        switch stopReason {
        case .stop:
            return "stop"
        case .length:
            return "length"
        case .cancelled:
            return "cancelled"
        @unknown default:
            return "stop"
        }
    }

    private static func toolCall(from call: MLXLMCommon.ToolCall) -> GenerationToolCall {
        GenerationToolCall(
            id: "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased())",
            function: GenerationToolCallFunction(
                name: call.function.name,
                arguments: call.function.arguments.mapValues(jsonValue(from:))
            )
        )
    }

    private static func jsonValue(from value: MLXLMCommon.JSONValue) -> MLXVLMCore.JSONValue {
        switch value {
        case .null:
            return .null
        case .bool(let bool):
            return .bool(bool)
        case .int(let int):
            return .number(Double(int))
        case .double(let double):
            return .number(double)
        case .string(let string):
            return .string(string)
        case .array(let array):
            return .array(array.map(jsonValue(from:)))
        case .object(let object):
            return .object(object.mapValues(jsonValue(from:)))
        @unknown default:
            return .string(String(describing: value))
        }
    }
}

public struct MLXVLMUpstreamBackend: VLMBackend {
    public let descriptor: ModelDescriptor
    public let status: BackendStatus = .mlxSwiftVLM
    public let container: MLXVLMUpstreamModelContainer
    public let processor: MLXVLMUpstreamProcessor
    public let generator: MLXVLMUpstreamGenerator

    public init(
        descriptor: ModelDescriptor,
        upstream: MLXLMCommon.ModelContainer,
        processor: MLXVLMUpstreamProcessor = MLXVLMUpstreamProcessor()
    ) {
        let loadPlan = ModelLoadPlanner(backend: .mlxSwiftVLM).plan(descriptor: descriptor)
        self.descriptor = descriptor
        self.container = MLXVLMUpstreamModelContainer(
            context: ModelLoadContext(descriptor: descriptor, loadPlan: loadPlan),
            upstream: upstream
        )
        self.processor = processor
        self.generator = MLXVLMUpstreamGenerator(container: container)
    }

    public static func load(descriptor: ModelDescriptor) async throws -> MLXVLMUpstreamBackend {
        let container = try await MLXVLM.VLMModelFactory.shared.loadContainer(
            from: URL(fileURLWithPath: descriptor.path),
            using: MLXLMTokenizers.TokenizersLoader()
        )
        return MLXVLMUpstreamBackend(descriptor: descriptor, upstream: container)
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
}

public enum MLXVLMUserInputBridge {
    public static func userInput(from request: GenerationRequest) throws -> UserInput {
        UserInput(
            chat: try request.messages.map(chatMessage(from:)),
            tools: try toolSpecs(from: request.metadata.tools)
        )
    }

    private static func toolSpecs(from tools: [MLXVLMCore.JSONValue]?) throws -> [MLXLMCommon.ToolSpec]? {
        guard let tools else {
            return nil
        }
        return try tools.map { tool in
            guard case .object(let object) = tool else {
                throw MLXVLMUpstreamBackendError.unsupportedToolSpec(tool)
            }
            return try object.reduce(into: MLXLMCommon.ToolSpec()) { result, element in
                result[element.key] = try sendableValue(from: element.value)
            }
        }
    }

    private static func sendableValue(from value: MLXVLMCore.JSONValue) throws -> any Sendable {
        switch value {
        case .object(let object):
            return try object.reduce(into: [String: any Sendable]()) { result, element in
                result[element.key] = try sendableValue(from: element.value)
            }
        case .array(let values):
            return try values.map { try sendableValue(from: $0) } as [any Sendable]
        case .string(let value):
            return value
        case .number(let value):
            return value
        case .bool(let value):
            return value
        case .null:
            return NSNull()
        }
    }

    private static func chatMessage(from message: ChatMessage) throws -> MLXLMCommon.Chat.Message {
        let text = message.content.compactMap { part -> String? in
            if case .text(let value) = part {
                return value
            }
            return nil
        }.joined(separator: "\n")
        let images = try message.content.compactMap(image(from:))
        let videos = try message.content.compactMap(video(from:))

        switch message.role {
        case .system, .developer:
            return .system(text, images: images, videos: videos)
        case .user:
            return .user(text, images: images, videos: videos)
        case .assistant:
            return .assistant(text, images: images, videos: videos)
        case .tool:
            return .tool(text)
        }
    }

    private static func image(from part: ContentPart) throws -> UserInput.Image? {
        guard case .imageURL(let reference) = part else {
            return nil
        }
        return .url(try mediaURL(from: reference, kind: .image))
    }

    private static func video(from part: ContentPart) throws -> UserInput.Video? {
        guard case .videoURL(let reference) = part else {
            return nil
        }
        return .url(try mediaURL(from: reference, kind: .video))
    }

    private static func mediaURL(from reference: String, kind: MediaKind) throws -> URL {
        if let inlineURL = try inlineMediaURL(from: reference, kind: kind) {
            return inlineURL
        }
        if let url = URL(string: reference), url.scheme != nil {
            return url
        }
        if reference.hasPrefix("/") || reference.hasPrefix("~") || reference.hasPrefix(".") {
            let expanded = NSString(string: reference).expandingTildeInPath
            return URL(fileURLWithPath: expanded)
        }
        throw MLXVLMUpstreamBackendError.unsupportedMediaReference(reference)
    }

    private static func inlineMediaURL(from reference: String, kind: MediaKind) throws -> URL? {
        let media: (data: Data, mimeType: String?)?
        if reference.hasPrefix("data:") {
            media = try decodeDataURI(reference)
        } else {
            let normalized = reference
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "\n", with: "")
                .replacingOccurrences(of: "\r", with: "")
            if let data = Data(base64Encoded: normalized, options: [.ignoreUnknownCharacters]) {
                media = (data, nil)
            } else {
                media = nil
            }
        }

        guard let media else {
            return nil
        }
        return try writeInlineMedia(
            data: media.data,
            mimeType: media.mimeType,
            kind: kind
        )
    }

    private static func decodeDataURI(_ reference: String) throws -> (Data, String?) {
        guard let comma = reference.firstIndex(of: ",") else {
            throw MLXVLMUpstreamBackendError.unsupportedMediaReference(reference)
        }
        let metadata = String(reference[reference.index(reference.startIndex, offsetBy: 5)..<comma])
        let payload = String(reference[reference.index(after: comma)...])
        let metadataParts = metadata.split(separator: ";").map(String.init)
        guard metadataParts.contains("base64"),
              let data = Data(base64Encoded: payload, options: [.ignoreUnknownCharacters])
        else {
            throw MLXVLMUpstreamBackendError.unsupportedMediaReference(reference)
        }
        let mimeType = metadataParts.first?.isEmpty == false ? metadataParts.first : nil
        return (data, mimeType)
    }

    private static func writeInlineMedia(
        data: Data,
        mimeType: String?,
        kind: MediaKind
    ) throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("mlx-vlm-swift-inline-media", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let filename = UUID().uuidString + "." + fileExtension(for: data, mimeType: mimeType, kind: kind)
        let url = directory.appendingPathComponent(filename)
        try data.write(to: url, options: [.atomic])
        return url
    }

    private static func fileExtension(for data: Data, mimeType: String?, kind: MediaKind) -> String {
        switch mimeType?.lowercased() {
        case "image/png":
            return "png"
        case "image/jpeg", "image/jpg":
            return "jpg"
        case "image/webp":
            return "webp"
        case "image/gif":
            return "gif"
        case "video/mp4":
            return "mp4"
        case "video/quicktime":
            return "mov"
        default:
            if data.starts(with: [0x89, 0x50, 0x4E, 0x47]) {
                return "png"
            }
            if data.starts(with: [0xFF, 0xD8, 0xFF]) {
                return "jpg"
            }
            if data.starts(with: [0x47, 0x49, 0x46]) {
                return "gif"
            }
            return kind == .video ? "mp4" : "png"
        }
    }
}
#endif
