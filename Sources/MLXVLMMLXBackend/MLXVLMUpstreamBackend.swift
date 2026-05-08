import CoreGraphics
import Foundation
import MLXVLMCore

public enum MLXVLMUserInputBridgePolicy {
    public static func shouldUseRawMessages(_ messages: [ChatMessage]) -> Bool {
        messages.contains(where: shouldUseRawMessage)
    }

    private static func shouldUseRawMessage(_ message: ChatMessage) -> Bool {
        if message.reasoning?.isEmpty == false {
            return true
        }
        if message.toolCalls?.isEmpty == false {
            return true
        }
        if message.toolCallID?.isEmpty == false {
            return true
        }
        if message.name?.isEmpty == false {
            return true
        }
        return message.content.contains(where: requiresRawContent)
    }

    private static func requiresRawContent(_ part: ContentPart) -> Bool {
        switch part {
        case .text:
            return false
        case .imagePlaceholder, .audioPlaceholder, .audioURL, .videoURL:
            return true
        case .imageURL(let reference):
            return reference.detail?.isEmpty == false ||
                reference.resizedHeight != nil ||
                reference.resizedWidth != nil ||
                reference.minPixels != nil ||
                reference.maxPixels != nil
        }
    }
}

#if MLXVLM_REAL_MLX_API && canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXVLM) && canImport(MLXLMTokenizers)
import MLX
import MLXLLM
import MLXLMCommon
import MLXLMTokenizers
import MLXVLM

public enum MLXVLMUpstreamBackendError: Error, CustomStringConvertible, Sendable {
    case unsupportedMessageRole(MessageRole)
    case unsupportedMediaReference(String)
    case unsupportedToolSpec(MLXVLMCore.JSONValue)
    case unsupportedAdapterPath(String)
    case invalidAdapterConfig(String)
    case invalidAdapterWeights(String)
    case invalidDraftModel(String)

    public var description: String {
        switch self {
        case .unsupportedMessageRole(let role):
            return "Unsupported chat message role for mlx-swift-lm Chat.Message: \(role.rawValue)"
        case .unsupportedMediaReference(let reference):
            return "Unsupported media reference for mlx-swift-lm UserInput: \(reference)"
        case .unsupportedToolSpec(let value):
            return "Unsupported tool schema for mlx-swift-lm ToolSpec: \(value)"
        case .unsupportedAdapterPath(let path):
            return "Unsupported adapter_path for mlx-swift-lm LoRA bridge: \(path)"
        case .invalidAdapterConfig(let reason):
            return "Invalid adapter_config.json for mlx-swift-lm LoRA bridge: \(reason)"
        case .invalidAdapterWeights(let reason):
            return "Invalid adapters.safetensors for mlx-swift-lm LoRA bridge: \(reason)"
        case .invalidDraftModel(let reason):
            return "Invalid draft model for mlx-swift-lm speculative decoding: \(reason)"
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
        let adapter = try MLXVLMAdapterBridge.adapter(path: input.request.metadata.adapterPath)
        if let adapter {
            do {
                try await container.upstream.perform { context in
                    try adapter.load(into: context.model)
                }
            } catch {
                await container.upstream.update { context in
                    adapter.unload(from: context.model)
                }
                throw error
            }
        }

        let userInput = try MLXVLMUserInputBridge.userInput(from: input.request)
        let lmInput: LMInput
        let upstreamStream: AsyncStream<Generation>
        do {
            lmInput = try await container.upstream.prepare(input: userInput)
            let parameters = MLXGenerateParametersBridge.makeGenerateParameters(
                from: MLXGenerateParametersPlan(runtime: input.runtime)
            )
            if input.runtime.sampling.seed != 0 {
                MLXRandom.seed(UInt64(truncatingIfNeeded: input.runtime.sampling.seed))
            }
            if let processor = try await MLXVLMStructuredLogitProcessorFactory.processor(
                request: input.request,
                container: container.upstream,
                parameters: parameters
            ) {
                upstreamStream = try await container.upstream.perform(
                    nonSendable: (lmInput, processor)
                ) { context, values in
                    let (lmInput, processor) = values
                    let iterator = try TokenIterator(
                        input: lmInput,
                        model: context.model,
                        processor: processor,
                        sampler: parameters.sampler(),
                        prefillStepSize: parameters.prefillStepSize,
                        maxTokens: parameters.maxTokens
                    )
                    let (stream, _) = MLXLMCommon.generateTask(
                        promptTokenCount: lmInput.text.tokens.size,
                        modelConfiguration: context.configuration,
                        tokenizer: context.tokenizer,
                        iterator: iterator
                    )
                    return stream
                }
            } else if let draft = try await MLXVLMResolvedDraftModel.loadIfRequested(metadata: input.request.metadata) {
                let draftContainer = draft.container
                let numDraftTokens = draft.numDraftTokens
                upstreamStream = try await draftContainer.perform(nonSendable: lmInput) { draftContext, lmInput in
                    try await container.upstream.perform(
                        nonSendable: (lmInput, draftContext.model)
                    ) { context, values in
                        let (lmInput, draftModel) = values
                        return try MLXLMCommon.generate(
                            input: lmInput,
                            parameters: parameters,
                            context: context,
                            draftModel: draftModel,
                            numDraftTokens: numDraftTokens
                        )
                    }
                }
            } else {
                upstreamStream = try await container.upstream.generate(input: lmInput, parameters: parameters)
            }
        } catch {
            if let adapter {
                await container.upstream.update { context in
                    adapter.unload(from: context.model)
                }
            }
            throw error
        }

        return AsyncThrowingStream { continuation in
            Task {
                defer {
                    if let adapter {
                        Task {
                            await container.upstream.update { context in
                                adapter.unload(from: context.model)
                            }
                        }
                    }
                }
                var emittedToolCall = false
                let shouldParseGemma4ToolOutput =
                    input.request.metadata.tools?.isEmpty == false &&
                    ToolCallOutputParser.usesGemma4ToolCalls(descriptor: container.context.descriptor)
                var bufferedText = ""
                for await event in upstreamStream {
                    switch event {
                    case .chunk(let text):
                        if shouldParseGemma4ToolOutput {
                            bufferedText += text
                        } else {
                            continuation.yield(GenerationChunk(text: text))
                        }
                    case .toolCall(let call):
                        emittedToolCall = true
                        continuation.yield(GenerationChunk(text: "", toolCalls: [Self.toolCall(from: call)]))
                    case .info(let info):
                        var finishReason = emittedToolCall ? "tool_calls" : finishReason(from: info.stopReason)
                        if shouldParseGemma4ToolOutput {
                            if emittedToolCall {
                                if !bufferedText.isEmpty {
                                    continuation.yield(GenerationChunk(text: bufferedText))
                                }
                            } else {
                                let parsed = ToolCallOutputParser().parseGemma4ToolCalls(in: bufferedText)
                                if !parsed.text.isEmpty {
                                    continuation.yield(GenerationChunk(text: parsed.text))
                                }
                                for toolCall in parsed.toolCalls {
                                    continuation.yield(GenerationChunk(text: "", toolCalls: [toolCall]))
                                }
                                if !parsed.toolCalls.isEmpty {
                                    finishReason = "tool_calls"
                                }
                            }
                        }
                        continuation.yield(
                            GenerationChunk(
                                text: "",
                                isFinished: true,
                                finishReason: finishReason,
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

public enum MLXVLMAdapterBridge {
    public static func adapter(path: String?) throws -> (any ModelAdapter)? {
        guard let path, !path.isEmpty else {
            return nil
        }
        let directory = try adapterDirectory(from: path)
        let config = try loadConfig(in: directory)

        if let fineTuneType = config["fine_tune_type"]?.stringValue {
            return try ModelAdapterFactory.shared.registry.createAdapter(
                directory: directory,
                adapterType: fineTuneType
            )
        }

        let converted = try convertedPythonMLXVLMAdapterDirectory(
            source: directory,
            config: config
        )
        defer {
            try? FileManager.default.removeItem(at: converted)
        }
        return try ModelAdapterFactory.shared.registry.createAdapter(directory: converted, adapterType: "lora")
    }

    private static func adapterDirectory(from path: String) throws -> URL {
        let expanded = NSString(string: path).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw MLXVLMUpstreamBackendError.unsupportedAdapterPath(path)
        }
        if isDirectory.boolValue {
            return url
        }
        guard url.lastPathComponent == "adapters.safetensors" else {
            throw MLXVLMUpstreamBackendError.unsupportedAdapterPath(path)
        }
        return url.deletingLastPathComponent()
    }

    private static func loadConfig(in directory: URL) throws -> [String: MLXVLMCore.JSONValue] {
        let configURL = directory.appendingPathComponent("adapter_config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw MLXVLMUpstreamBackendError.invalidAdapterConfig("missing adapter_config.json in \(directory.path)")
        }
        let data = try Data(contentsOf: configURL)
        let json = try JSONDecoder().decode(MLXVLMCore.JSONValue.self, from: data)
        guard let object = json.objectValue else {
            throw MLXVLMUpstreamBackendError.invalidAdapterConfig("adapter_config.json must contain an object")
        }
        return object
    }

    private static func convertedPythonMLXVLMAdapterDirectory(
        source: URL,
        config: [String: MLXVLMCore.JSONValue]
    ) throws -> URL {
        guard let rank = config["rank"]?.intValue, rank > 0 else {
            throw MLXVLMUpstreamBackendError.invalidAdapterConfig("missing positive Python mlx-vlm rank")
        }
        let alpha = config["alpha"]?.doubleValue ?? Double(rank)
        let scale = alpha / Double(rank)
        let numLayers = config["num_layers"]?.intValue ?? config["numLayers"]?.intValue ?? 10_000
        var loraParameters: [String: MLXVLMCore.JSONValue] = [
            "rank": .number(Double(rank)),
            "scale": .number(scale),
        ]
        if let keys = config["keys"] ?? config["target_modules"] {
            loraParameters["keys"] = keys
        }
        let upstreamConfig: MLXVLMCore.JSONValue = .object([
            "fine_tune_type": .string("lora"),
            "num_layers": .number(Double(numLayers)),
            "lora_parameters": .object(loraParameters),
        ])

        let destination = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("mlx-vlm-swift-python-adapters", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        try JSONEncoder().encode(upstreamConfig).write(
            to: destination.appendingPathComponent("adapter_config.json"),
            options: [.atomic]
        )
        try writeTransformedSafetensors(
            source: source.appendingPathComponent("adapters.safetensors"),
            destination: destination.appendingPathComponent("adapters.safetensors")
        )
        return destination
    }

    private static func writeTransformedSafetensors(source: URL, destination: URL) throws {
        guard FileManager.default.fileExists(atPath: source.path) else {
            throw MLXVLMUpstreamBackendError.invalidAdapterWeights("missing adapters.safetensors in \(source.deletingLastPathComponent().path)")
        }
        let data = try Data(contentsOf: source)
        guard data.count >= 8 else {
            throw MLXVLMUpstreamBackendError.invalidAdapterWeights("file is too small")
        }
        let headerLength = Int(readLittleEndianUInt64(data.prefix(8)))
        let dataStart = 8 + headerLength
        guard headerLength > 0, data.count >= dataStart else {
            throw MLXVLMUpstreamBackendError.invalidAdapterWeights("invalid safetensors header length")
        }
        let headerData = Data(data[8..<dataStart])
        guard let object = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw MLXVLMUpstreamBackendError.invalidAdapterWeights("safetensors header must contain an object")
        }
        let transformed = object.reduce(into: [String: Any]()) { result, element in
            result[transformedWeightKey(element.key)] = element.value
        }
        let transformedHeader = try JSONSerialization.data(withJSONObject: transformed, options: [])
        var output = Data()
        output.append(littleEndianBytes(UInt64(transformedHeader.count)))
        output.append(transformedHeader)
        output.append(data[dataStart...])
        try output.write(to: destination, options: [.atomic])
    }

    private static func transformedWeightKey(_ key: String) -> String {
        if key == "__metadata__" {
            return key
        }
        if key == "A" {
            return "lora_a"
        }
        if key == "B" {
            return "lora_b"
        }
        if key.hasSuffix(".A") {
            return String(key.dropLast(2)) + ".lora_a"
        }
        if key.hasSuffix(".B") {
            return String(key.dropLast(2)) + ".lora_b"
        }
        return key
    }

    private static func readLittleEndianUInt64(_ bytes: Data.SubSequence) -> UInt64 {
        bytes.enumerated().reduce(UInt64(0)) { result, element in
            result | (UInt64(element.element) << UInt64(element.offset * 8))
        }
    }

    private static func littleEndianBytes(_ value: UInt64) -> Data {
        var result = Data()
        for shift in stride(from: 0, through: 56, by: 8) {
            result.append(UInt8((value >> UInt64(shift)) & 0xff))
        }
        return result
    }
}

public struct MLXVLMUpstreamBackend: VLMBackend, TokenizationBackend {
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

    public func tokenize(text: String, addSpecialTokens: Bool = true) async throws -> BackendTokenizationResult {
        let tokenizer = await container.upstream.tokenizer
        let tokenIDs = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        let tokens = tokenIDs.map { tokenizer.convertIdToToken($0) ?? "" }
        return BackendTokenizationResult(
            backend: status.activeBackend,
            tokens: tokens,
            tokenIDs: tokenIDs,
            text: text
        )
    }

    public func detokenize(tokenIDs: [Int], skipSpecialTokens: Bool = true) async throws -> BackendDetokenizationResult {
        let tokenizer = await container.upstream.tokenizer
        let text = tokenizer.decode(tokenIds: tokenIDs, skipSpecialTokens: skipSpecialTokens)
        let tokenPairs = tokenIDs.map { id in (id, tokenizer.convertIdToToken(id)) }
        return BackendDetokenizationResult(
            backend: status.activeBackend,
            text: text,
            tokenIDs: tokenIDs,
            tokens: tokenPairs.map { $0.1 ?? "" },
            unknownTokenIDs: tokenPairs.compactMap { $0.1 == nil ? $0.0 : nil }
        )
    }
}

struct MLXVLMResolvedDraftModel {
    let container: MLXLMCommon.ModelContainer
    let numDraftTokens: Int

    static func loadIfRequested(metadata: GenerationRequestMetadata) async throws -> MLXVLMResolvedDraftModel? {
        guard let model = metadata.draftModel?.trimmingCharacters(in: .whitespacesAndNewlines),
              !model.isEmpty
        else {
            return nil
        }
        let descriptor = try await MLXRemoteModelResolver().resolveDescriptor(pathOrIdentifier: model)
        let directory = URL(fileURLWithPath: descriptor.path)
        let tokenizers = MLXLMTokenizers.TokenizersLoader()
        let container: MLXLMCommon.ModelContainer
        do {
            container = try await MLXVLM.VLMModelFactory.shared.loadContainer(from: directory, using: tokenizers)
        } catch {
            do {
                container = try await MLXLLM.LLMModelFactory.shared.loadContainer(from: directory, using: tokenizers)
            } catch {
                let diagnostic = Self.draftModelDiagnostic(
                    requestedModel: model,
                    directory: directory,
                    draftKind: metadata.draftKind
                )
                throw MLXVLMUpstreamBackendError.invalidDraftModel(diagnostic)
            }
        }
        return MLXVLMResolvedDraftModel(
            container: container,
            numDraftTokens: max(1, metadata.draftBlockSize ?? 2)
        )
    }

    private static func draftModelDiagnostic(
        requestedModel: String,
        directory: URL,
        draftKind: String?
    ) -> String {
        guard let config = readDraftConfig(from: directory) else {
            return "\(requestedModel) could not be loaded as VLM or LLM"
        }

        let modelType = config.string("model_type")
        let architectures = config.stringArray("architectures")
        let textModelType = config.dictionary("text_config")?.string("model_type")

        if modelType == "gemma4_assistant" || architectures.contains("Gemma4AssistantForCausalLM") {
            let kind = draftKind.map { " --draft-kind \($0)" } ?? ""
            return "\(requestedModel)\(kind) uses Gemma4 MTP assistant architecture (model_type gemma4_assistant, architecture Gemma4AssistantForCausalLM). The bundled official mlx-swift-lm registry currently exposes Gemma4 VLM/LLM loaders for gemma4 and gemma4_text, but not gemma4_assistant; native Swift Gemma4Assistant draft-model support is required before this Python mlx-vlm MTP path can be replaced."
        }

        let architectureSummary = architectures.isEmpty ? "unknown" : architectures.joined(separator: ",")
        let typeSummary = [modelType, textModelType.map { "text_config.\($0)" }]
            .compactMap { $0 }
            .joined(separator: ", ")
        if typeSummary.isEmpty {
            return "\(requestedModel) could not be loaded as VLM or LLM; config architecture=\(architectureSummary)"
        }
        return "\(requestedModel) could not be loaded as VLM or LLM; config model_type=\(typeSummary), architecture=\(architectureSummary)"
    }

    private static func readDraftConfig(from directory: URL) -> [String: Any]? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        return object
    }
}

private extension Dictionary where Key == String, Value == Any {
    func string(_ key: String) -> String? {
        self[key] as? String
    }

    func stringArray(_ key: String) -> [String] {
        self[key] as? [String] ?? []
    }

    func dictionary(_ key: String) -> [String: Any]? {
        self[key] as? [String: Any]
    }
}

struct MLXVLMStructuredLogitProcessorFactory {
    static func processor(
        request: GenerationRequest,
        container: MLXLMCommon.ModelContainer,
        parameters: GenerateParameters
    ) async throws -> (any LogitProcessor)? {
        let plan = ResponseFormatPlanner().plan(metadata: request.metadata, stream: request.stream)
        guard plan.requiresJSONMode else {
            return nil
        }
        let tokenizer = await container.tokenizer
        let guidance = JSONSchemaGuidance(responseFormat: request.metadata.responseFormat)
        if let grammar = guidance.openScalarGrammar(
            defaultMaxStringLength: max(8, min(256, request.parameters.maxTokens / 2))
        ) {
            return CompositeLogitProcessor(
                first: parameters.processor(),
                second: JSONOpenScalarGrammarProcessor(
                    grammar: grammar,
                    tokenizer: tokenizer,
                    terminalTokenID: tokenizer.eosTokenId
                )
            )
        }
        if let tokenSequences = guidance.finiteDFATokenSequences(tokenizer: tokenizer),
           tokenSequences.count > 1
        {
            return CompositeLogitProcessor(
                first: parameters.processor(),
                second: JSONTokenTrieProcessor(
                    tokenSequences: tokenSequences,
                    terminalTokenID: tokenizer.eosTokenId
                )
            )
        }
        if let literal = guidance.forcedJSONLiteral(),
           let forcedTokenIDs = guidance.forcedLiteralTokenIDs(tokenizer: tokenizer, literal: literal),
           !forcedTokenIDs.isEmpty
        {
            return CompositeLogitProcessor(
                first: parameters.processor(),
                second: JSONForcedTokenProcessor(
                    expectedTokenIDs: forcedTokenIDs,
                    terminalTokenID: tokenizer.eosTokenId
                )
            )
        }
        if let prefixTokenIDs = guidance.prefixTokenIDs(tokenizer: tokenizer), !prefixTokenIDs.isEmpty {
            return CompositeLogitProcessor(
                first: parameters.processor(),
                second: JSONPrefixTokenProcessor(expectedTokenIDs: prefixTokenIDs)
            )
        }
        let startTokenIDs = JSONStartTokenProcessor.startTokenIDs(tokenizer: tokenizer, rootStart: guidance.rootStart)
        guard !startTokenIDs.isEmpty else {
            return parameters.processor()
        }
        return CompositeLogitProcessor(
            first: parameters.processor(),
            second: JSONStartTokenProcessor(allowedTokenIDs: startTokenIDs)
        )
    }
}

struct CompositeLogitProcessor: LogitProcessor {
    private var first: (any LogitProcessor)?
    private var second: any LogitProcessor

    init(first: (any LogitProcessor)?, second: any LogitProcessor) {
        self.first = first
        self.second = second
    }

    mutating func prompt(_ prompt: MLXArray) {
        first?.prompt(prompt)
        second.prompt(prompt)
    }

    func process(logits: MLXArray) -> MLXArray {
        second.process(logits: first?.process(logits: logits) ?? logits)
    }

    mutating func didSample(token: MLXArray) {
        first?.didSample(token: token)
        second.didSample(token: token)
    }
}

struct JSONSchemaGuidance {
    let rootStart: JSONRootStart
    let requiredProperties: [String]
    let schema: [String: MLXVLMCore.JSONValue]?

    init(responseFormat: MLXVLMCore.JSONValue?) {
        guard let schema = Self.schema(from: responseFormat) else {
            self.rootStart = .either
            self.requiredProperties = []
            self.schema = nil
            return
        }
        let type = schema["type"]?.stringValue?.lowercased()
        self.rootStart = type == "array" ? .array : .object
        self.requiredProperties = schema["required"]?.arrayValue?.compactMap(\.stringValue) ?? []
        self.schema = schema
    }

    func prefixTokenIDs(tokenizer: Tokenizer) -> [Int]? {
        guard rootStart == .object,
              let firstRequired = requiredProperties.first,
              !firstRequired.isEmpty
        else {
            return nil
        }
        let escapedKey = Self.escapedJSONString(firstRequired)
        let candidates = [
            "{\"\(escapedKey)\":",
            "{ \"\(escapedKey)\":",
        ]
        return candidates.lazy
            .map { tokenizer.encode(text: $0, addSpecialTokens: false) }
            .first { !$0.isEmpty }
    }

    func forcedJSONLiteral() -> String? {
        guard let schema,
              let literal = JSONSchemaDeterministicValueBuilder().literal(schema: .object(schema))
        else { return nil }
        return literal
    }

    func openScalarGrammar(defaultMaxStringLength: Int) -> JSONOpenScalarGrammar? {
        guard let schema else {
            return nil
        }
        let plan = JSONSchemaGrammarPlanner().plan(schema: .object(schema))
        guard plan?.requiresTokenizerPrefixFiltering == true else {
            return nil
        }
        return JSONOpenScalarGrammar(schema: schema, defaultMaxStringLength: defaultMaxStringLength)
    }

    func finiteDFATokenSequences(tokenizer: Tokenizer) -> [[Int]]? {
        guard let schema,
              let literals = JSONSchemaFiniteLiteralBuilder(maxAlternatives: 128).literals(schema: .object(schema))
        else { return nil }

        var seen: Set<[Int]> = []
        let sequences = literals.compactMap { literal -> [Int]? in
            let tokenIDs = tokenizer.encode(text: literal, addSpecialTokens: false)
            guard !tokenIDs.isEmpty,
                  tokenizer.decode(tokenIds: tokenIDs, skipSpecialTokens: true) == literal,
                  seen.insert(tokenIDs).inserted
            else {
                return nil
            }
            return tokenIDs
        }
        return sequences.isEmpty ? nil : sequences
    }

    func forcedLiteralTokenIDs(tokenizer: Tokenizer, literal: String) -> [Int]? {
        let tokenIDs = tokenizer.encode(text: literal, addSpecialTokens: false)
        guard !tokenIDs.isEmpty,
              tokenizer.decode(tokenIds: tokenIDs, skipSpecialTokens: true) == literal
        else {
            return nil
        }
        return tokenIDs
    }

    private static func schema(from responseFormat: MLXVLMCore.JSONValue?) -> [String: MLXVLMCore.JSONValue]? {
        guard let object = responseFormat?.objectValue else {
            return nil
        }
        if let schema = object["schema"]?.objectValue {
            return schema
        }
        if let schema = object["json_schema"]?["schema"]?.objectValue {
            return schema
        }
        return nil
    }

    private static func escapedJSONString(_ string: String) -> String {
        var result = ""
        for scalar in string.unicodeScalars {
            switch scalar {
            case "\"":
                result += "\\\""
            case "\\":
                result += "\\\\"
            case "\u{08}":
                result += "\\b"
            case "\u{0C}":
                result += "\\f"
            case "\n":
                result += "\\n"
            case "\r":
                result += "\\r"
            case "\t":
                result += "\\t"
            default:
                if scalar.value < 0x20 {
                    result += String(format: "\\u%04X", scalar.value)
                } else {
                    result.unicodeScalars.append(scalar)
                }
            }
        }
        return result
    }
}

enum JSONRootStart: Equatable {
    case object
    case array
    case either
}

struct JSONStartTokenProcessor: LogitProcessor {
    private let allowedTokenIDs: [Int]
    private var generatedTokenCount = 0

    init(allowedTokenIDs: [Int]) {
        self.allowedTokenIDs = allowedTokenIDs
    }

    static func startTokenIDs(tokenizer: Tokenizer, rootStart: JSONRootStart = .either) -> [Int] {
        var result: [Int] = []
        let candidates: [String]
        switch rootStart {
        case .object:
            candidates = ["{", " {"]
        case .array:
            candidates = ["[", " ["]
        case .either:
            candidates = ["{", "[", " {", " ["]
        }
        for candidate in candidates {
            let ids = tokenizer.encode(text: candidate, addSpecialTokens: false)
            if ids.count == 1 {
                result.append(ids[0])
            }
        }
        for candidate in candidates.map({ $0.trimmingCharacters(in: .whitespaces) }) {
            if let id = tokenizer.convertTokenToId(candidate) {
                result.append(id)
            }
        }
        var seen: Set<Int> = []
        return result.filter { seen.insert($0).inserted }
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard generatedTokenCount == 0, !allowedTokenIDs.isEmpty else {
            return logits
        }
        let masked = MLXArray.zeros(logits.shape, dtype: logits.dtype) + MLXArray(-Float.infinity)
        let indices = MLXArray(allowedTokenIDs).asType(.uint32)
        masked[0..., indices] = logits[0..., indices]
        return masked
    }

    mutating func didSample(token: MLXArray) {
        generatedTokenCount += token.size
    }
}

struct JSONForcedTokenProcessor: LogitProcessor {
    private let expectedTokenIDs: [Int]
    private let terminalTokenID: Int?
    private var generatedTokenCount = 0

    init(expectedTokenIDs: [Int], terminalTokenID: Int?) {
        self.expectedTokenIDs = expectedTokenIDs
        self.terminalTokenID = terminalTokenID
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let tokenID: Int?
        if generatedTokenCount < expectedTokenIDs.count {
            tokenID = expectedTokenIDs[generatedTokenCount]
        } else {
            tokenID = terminalTokenID
        }
        guard let tokenID else {
            return logits
        }
        let masked = MLXArray.zeros(logits.shape, dtype: logits.dtype) + MLXArray(-Float.infinity)
        let indices = MLXArray([tokenID]).asType(.uint32)
        masked[0..., indices] = logits[0..., indices]
        return masked
    }

    mutating func didSample(token: MLXArray) {
        generatedTokenCount += token.size
    }
}

struct JSONPrefixTokenProcessor: LogitProcessor {
    private let expectedTokenIDs: [Int]
    private var generatedTokenCount = 0

    init(expectedTokenIDs: [Int]) {
        self.expectedTokenIDs = expectedTokenIDs
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard generatedTokenCount < expectedTokenIDs.count else {
            return logits
        }
        let expectedTokenID = expectedTokenIDs[generatedTokenCount]
        let masked = MLXArray.zeros(logits.shape, dtype: logits.dtype) + MLXArray(-Float.infinity)
        let indices = MLXArray([expectedTokenID]).asType(.uint32)
        masked[0..., indices] = logits[0..., indices]
        return masked
    }

    mutating func didSample(token: MLXArray) {
        generatedTokenCount += token.size
    }
}

private final class JSONTokenTrieNode {
    var children: [Int: JSONTokenTrieNode] = [:]
    var terminal = false
}

struct JSONTokenTrieProcessor: LogitProcessor {
    private let root: JSONTokenTrieNode
    private let terminalTokenID: Int?
    private var current: JSONTokenTrieNode?

    init(tokenSequences: [[Int]], terminalTokenID: Int?) {
        let root = JSONTokenTrieNode()
        for sequence in tokenSequences where !sequence.isEmpty {
            var node = root
            for tokenID in sequence {
                if let child = node.children[tokenID] {
                    node = child
                } else {
                    let child = JSONTokenTrieNode()
                    node.children[tokenID] = child
                    node = child
                }
            }
            node.terminal = true
        }
        self.root = root
        self.terminalTokenID = terminalTokenID
        self.current = root
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard let current else {
            return logits
        }
        var allowedTokenIDs = Array(current.children.keys)
        if current.terminal, let terminalTokenID {
            allowedTokenIDs.append(terminalTokenID)
        }
        guard !allowedTokenIDs.isEmpty else {
            return logits
        }
        let masked = MLXArray.zeros(logits.shape, dtype: logits.dtype) + MLXArray(-Float.infinity)
        let indices = MLXArray(allowedTokenIDs).asType(.uint32)
        masked[0..., indices] = logits[0..., indices]
        return masked
    }

    mutating func didSample(token: MLXArray) {
        guard let tokenID = token.asArray(Int.self).last,
              let current
        else {
            self.current = nil
            return
        }
        if let next = current.children[tokenID] {
            self.current = next
        } else if current.terminal, tokenID == terminalTokenID {
            self.current = nil
        } else {
            self.current = nil
        }
    }
}

private enum JSONGrammarElement {
    case literal(String)
    case choice([[JSONGrammarElement]])
    case openString(maxLength: Int)
    case openNumber
}

struct JSONOpenScalarGrammar {
    private let elements: [JSONGrammarElement]
    private let defaultMaxStringLength: Int

    init?(schema: [String: MLXVLMCore.JSONValue], defaultMaxStringLength: Int) {
        self.defaultMaxStringLength = max(1, defaultMaxStringLength)
        guard let elements = Self.elements(
            for: schema,
            defaultMaxStringLength: self.defaultMaxStringLength
        ) else {
            return nil
        }
        self.elements = elements
    }

    func canComplete(prefix: String) -> Bool {
        Self.match(elements, index: 0, text: prefix, position: prefix.startIndex, requireComplete: false)
    }

    func isComplete(_ text: String) -> Bool {
        Self.match(elements, index: 0, text: text, position: text.startIndex, requireComplete: true)
    }

    private static func elements(
        for schema: [String: MLXVLMCore.JSONValue],
        defaultMaxStringLength: Int
    ) -> [JSONGrammarElement]? {
        if let value = schema["const"] {
            return [.literal(MLXVLMCore.JSONSchemaDeterministicValueBuilder.literal(value))]
        }
        if let values = schema["enum"]?.arrayValue, !values.isEmpty {
            return [.choice(values.map { [.literal(MLXVLMCore.JSONSchemaDeterministicValueBuilder.literal($0))] })]
        }
        if let value = schema["default"] {
            return [.literal(MLXVLMCore.JSONSchemaDeterministicValueBuilder.literal(value))]
        }

        let currentTypes = types(from: schema)
        if currentTypes.contains("object") || schema["properties"] != nil {
            return objectElements(for: schema, defaultMaxStringLength: defaultMaxStringLength)
        }
        if currentTypes.contains("array") || schema["items"] != nil {
            return arrayElements(for: schema, defaultMaxStringLength: defaultMaxStringLength)
        }
        if currentTypes.contains("string") {
            return [.openString(maxLength: max(0, schema["maxLength"]?.intValue ?? defaultMaxStringLength))]
        }
        if currentTypes.contains("integer") || currentTypes.contains("number") {
            return [.openNumber]
        }
        if currentTypes.contains("boolean") {
            return [.choice([[.literal("false")], [.literal("true")]])]
        }
        if currentTypes.contains("null") {
            return [.literal("null")]
        }
        return nil
    }

    private static func objectElements(
        for schema: [String: MLXVLMCore.JSONValue],
        defaultMaxStringLength: Int
    ) -> [JSONGrammarElement]? {
        let required = schema["required"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let properties = schema["properties"]?.objectValue ?? [:]
        let optionalDefaults = properties.keys
            .filter { !required.contains($0) }
            .filter { properties[$0]?.objectValue?["default"] != nil }
            .sorted()
        let keys = required + optionalDefaults

        var result: [JSONGrammarElement] = [.literal("{")]
        for (offset, key) in keys.enumerated() {
            guard let propertySchema = properties[key]?.objectValue,
                  let propertyElements = elements(
                    for: propertySchema,
                    defaultMaxStringLength: defaultMaxStringLength
                  )
            else {
                return nil
            }
            if offset > 0 {
                result.append(.literal(","))
            }
            result.append(.literal("\"\(escapedJSONString(key))\":"))
            result.append(contentsOf: propertyElements)
        }
        result.append(.literal("}"))
        return result
    }

    private static func arrayElements(
        for schema: [String: MLXVLMCore.JSONValue],
        defaultMaxStringLength: Int
    ) -> [JSONGrammarElement]? {
        let minItems = max(0, schema["minItems"]?.intValue ?? 0)
        guard minItems > 0 else {
            return [.literal("[]")]
        }
        guard let itemSchema = schema["items"]?.objectValue,
              let itemElements = elements(for: itemSchema, defaultMaxStringLength: defaultMaxStringLength)
        else {
            return nil
        }
        var result: [JSONGrammarElement] = [.literal("[")]
        for index in 0..<minItems {
            if index > 0 {
                result.append(.literal(","))
            }
            result.append(contentsOf: itemElements)
        }
        result.append(.literal("]"))
        return result
    }

    private static func match(
        _ elements: [JSONGrammarElement],
        index: Int,
        text: String,
        position: String.Index,
        requireComplete: Bool
    ) -> Bool {
        if index == elements.count {
            return position == text.endIndex
        }
        if position == text.endIndex {
            return !requireComplete
        }

        switch elements[index] {
        case .literal(let literal):
            return matchLiteral(
                literal,
                elements: elements,
                index: index,
                text: text,
                position: position,
                requireComplete: requireComplete
            )
        case .choice(let choices):
            return choices.contains { choice in
                var combined = choice
                combined.append(contentsOf: elements[(index + 1)...])
                return match(combined, index: 0, text: text, position: position, requireComplete: requireComplete)
            }
        case .openString(let maxLength):
            return matchOpenString(
                elements,
                index: index,
                text: text,
                position: position,
                maxLength: maxLength,
                requireComplete: requireComplete
            )
        case .openNumber:
            return matchOpenNumber(
                elements,
                index: index,
                text: text,
                position: position,
                requireComplete: requireComplete
            )
        }
    }

    private static func matchLiteral(
        _ literal: String,
        elements: [JSONGrammarElement],
        index: Int,
        text: String,
        position: String.Index,
        requireComplete: Bool
    ) -> Bool {
        var literalIndex = literal.startIndex
        var textIndex = position
        while literalIndex < literal.endIndex && textIndex < text.endIndex {
            guard literal[literalIndex] == text[textIndex] else {
                return false
            }
            literal.formIndex(after: &literalIndex)
            text.formIndex(after: &textIndex)
        }
        if literalIndex == literal.endIndex {
            return match(elements, index: index + 1, text: text, position: textIndex, requireComplete: requireComplete)
        }
        return textIndex == text.endIndex && !requireComplete
    }

    private static func matchOpenString(
        _ elements: [JSONGrammarElement],
        index: Int,
        text: String,
        position: String.Index,
        maxLength: Int,
        requireComplete: Bool
    ) -> Bool {
        var cursor = position
        guard cursor < text.endIndex else {
            return !requireComplete
        }
        guard text[cursor] == "\"" else {
            return false
        }
        text.formIndex(after: &cursor)
        var escaping = false
        var unicodeEscapeDigitsRemaining = 0
        var contentLength = 0
        while cursor < text.endIndex {
            let character = text[cursor]
            if unicodeEscapeDigitsRemaining > 0 {
                guard character.isHexDigit else {
                    return false
                }
                unicodeEscapeDigitsRemaining -= 1
                if unicodeEscapeDigitsRemaining == 0 {
                    contentLength += 1
                    if contentLength > maxLength {
                        return false
                    }
                }
                text.formIndex(after: &cursor)
                continue
            }
            if escaping {
                if character == "u" {
                    unicodeEscapeDigitsRemaining = 4
                } else if !"\"\\/bfnrt".contains(character) {
                    return false
                } else {
                    contentLength += 1
                    if contentLength > maxLength {
                        return false
                    }
                }
                escaping = false
                text.formIndex(after: &cursor)
                continue
            }
            if character == "\\" {
                escaping = true
                text.formIndex(after: &cursor)
                continue
            }
            if character == "\"" {
                text.formIndex(after: &cursor)
                return match(elements, index: index + 1, text: text, position: cursor, requireComplete: requireComplete)
            }
            guard character.unicodeScalars.allSatisfy({ $0.value >= 0x20 }) else {
                return false
            }
            contentLength += 1
            if contentLength > maxLength {
                return false
            }
            text.formIndex(after: &cursor)
        }
        return !requireComplete
    }

    private static func matchOpenNumber(
        _ elements: [JSONGrammarElement],
        index: Int,
        text: String,
        position: String.Index,
        requireComplete: Bool
    ) -> Bool {
        var cursor = position
        while cursor < text.endIndex {
            let prefix = String(text[position..<cursor])
            if isCompleteJSONNumber(prefix),
               match(elements, index: index + 1, text: text, position: cursor, requireComplete: requireComplete)
            {
                return true
            }
            guard isJSONNumberCharacter(text[cursor]) else {
                return false
            }
            text.formIndex(after: &cursor)
        }
        let number = String(text[position..<text.endIndex])
        if requireComplete {
            return isCompleteJSONNumber(number) &&
                match(elements, index: index + 1, text: text, position: text.endIndex, requireComplete: true)
        }
        return isJSONNumberPrefix(number)
    }

    private static func isJSONNumberCharacter(_ character: Character) -> Bool {
        character.isNumber || character == "-" || character == "+" || character == "." || character == "e" || character == "E"
    }

    private static func isJSONNumberPrefix(_ text: String) -> Bool {
        guard !text.isEmpty else {
            return true
        }
        let pattern = #"^-?(?:0|[1-9][0-9]*)?(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?$"#
        return text.range(of: pattern, options: .regularExpression) != nil
    }

    private static func isCompleteJSONNumber(_ text: String) -> Bool {
        guard !text.isEmpty else {
            return false
        }
        let pattern = #"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"#
        return text.range(of: pattern, options: .regularExpression) != nil
    }

    private static func types(from object: [String: MLXVLMCore.JSONValue]) -> [String] {
        if let type = object["type"]?.stringValue?.lowercased() {
            return [type]
        }
        if let types = object["type"]?.arrayValue?.compactMap({ $0.stringValue?.lowercased() }), !types.isEmpty {
            return types
        }
        if object["properties"] != nil {
            return ["object"]
        }
        if object["items"] != nil {
            return ["array"]
        }
        return []
    }

    private static func escapedJSONString(_ string: String) -> String {
        var result = ""
        for scalar in string.unicodeScalars {
            switch scalar {
            case "\"":
                result += "\\\""
            case "\\":
                result += "\\\\"
            case "\u{08}":
                result += "\\b"
            case "\u{0C}":
                result += "\\f"
            case "\n":
                result += "\\n"
            case "\r":
                result += "\\r"
            case "\t":
                result += "\\t"
            default:
                if scalar.value < 0x20 {
                    result += String(format: "\\u%04X", scalar.value)
                } else {
                    result.unicodeScalars.append(scalar)
                }
            }
        }
        return result
    }
}

private final class JSONTokenTextCache {
    var values: [Int: String] = [:]
}

struct JSONOpenScalarGrammarProcessor: LogitProcessor {
    private let grammar: JSONOpenScalarGrammar
    private let tokenizer: Tokenizer
    private let terminalTokenID: Int?
    private let tokenTextCache = JSONTokenTextCache()
    private var generatedText = ""

    init(grammar: JSONOpenScalarGrammar, tokenizer: Tokenizer, terminalTokenID: Int?) {
        self.grammar = grammar
        self.tokenizer = tokenizer
        self.terminalTokenID = terminalTokenID
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let complete = grammar.isComplete(generatedText)
        let vocabSize = logits.shape.last ?? 0
        guard vocabSize > 0 else {
            return logits
        }
        var allowedTokenIDs: [Int] = []
        if complete, let terminalTokenID {
            allowedTokenIDs.append(terminalTokenID)
        }
        if !complete {
            for tokenID in 0..<vocabSize {
                let text = decodedText(for: tokenID)
                guard !text.isEmpty else {
                    continue
                }
                if grammar.canComplete(prefix: generatedText + text) {
                    allowedTokenIDs.append(tokenID)
                }
            }
        }
        guard !allowedTokenIDs.isEmpty else {
            return logits
        }
        let masked = MLXArray.zeros(logits.shape, dtype: logits.dtype) + MLXArray(-Float.infinity)
        let indices = MLXArray(allowedTokenIDs).asType(.uint32)
        masked[0..., indices] = logits[0..., indices]
        return masked
    }

    mutating func didSample(token: MLXArray) {
        guard let tokenID = token.asArray(Int.self).last else {
            return
        }
        if tokenID == terminalTokenID, grammar.isComplete(generatedText) {
            return
        }
        generatedText += decodedText(for: tokenID)
    }

    private func decodedText(for tokenID: Int) -> String {
        if let cached = tokenTextCache.values[tokenID] {
            return cached
        }
        let text = tokenizer.decode(tokenIds: [tokenID], skipSpecialTokens: true)
        tokenTextCache.values[tokenID] = text
        return text
    }
}

public enum MLXVLMUserInputBridge {
    public static func userInput(from request: GenerationRequest) throws -> UserInput {
        if MLXVLMUserInputBridgePolicy.shouldUseRawMessages(request.messages) {
            var input = UserInput(
                messages: try request.messages.map(rawMessage(from:)),
                images: try request.messages.flatMap(\.content).compactMap(image(from:)),
                videos: try request.messages.flatMap(\.content).compactMap(video(from:)),
                tools: try toolSpecs(from: request.metadata.tools),
                additionalContext: try additionalContext(from: request)
            )
            input.processing = processing(from: request.metadata)
            return input
        }

        return UserInput(
            chat: try request.messages.map(chatMessage(from:)),
            processing: processing(from: request.metadata),
            tools: try toolSpecs(from: request.metadata.tools),
            additionalContext: try additionalContext(from: request)
        )
    }

    private static func processing(from metadata: GenerationRequestMetadata) -> UserInput.Processing {
        guard let resizeShape = metadata.resizeShape, resizeShape.count == 2 else {
            return .init()
        }
        return .init(
            resize: CGSize(
                width: resizeShape[1],
                height: resizeShape[0]
            )
        )
    }

    private static func additionalContext(from request: GenerationRequest) throws -> [String: any Sendable]? {
        var context: [String: any Sendable] = [:]

        if let enableThinking = request.parameters.enableThinking {
            context["enable_thinking"] = enableThinking
        }
        if let thinkingBudget = request.parameters.thinkingBudget {
            context["thinking_budget"] = thinkingBudget
        }
        if let thinkingStartToken = request.metadata.thinkingStartToken {
            context["thinking_start_token"] = thinkingStartToken
        }
        if request.metadata.rawPrompt {
            context["raw"] = true
            context["raw_prompt"] = true
        }
        if let template = request.metadata.template {
            context["template"] = template
        }
        if let suffix = request.metadata.suffix {
            context["suffix"] = suffix
        }
        if let legacyContext = request.metadata.legacyContext {
            context["context"] = legacyContext
        }
        if let rawOptions = request.metadata.rawOptions {
            let sendableOptions = try sendableValue(from: .object(rawOptions))
            context["options"] = sendableOptions
            context["raw_options"] = sendableOptions
        }
        if let responseFormat = request.metadata.responseFormat {
            context["response_format"] = try sendableValue(from: responseFormat)
        }
        if let logitBias = request.metadata.logitBias {
            context["logit_bias"] = try sendableLogitBias(from: logitBias)
        }
        if let logprobs = request.metadata.logprobs {
            context["logprobs"] = logprobs
        }
        if let topLogprobs = request.metadata.topLogprobs {
            context["top_logprobs"] = topLogprobs
        }
        if let toolChoice = request.metadata.toolChoice {
            context["tool_choice"] = try sendableValue(from: toolChoice)
        }
        if let user = request.metadata.user {
            context["user"] = user
        }
        if let tenantID = request.metadata.tenantID {
            context["apc_tenant"] = tenantID
            context["tenant_id"] = tenantID
        }
        if let responseInstructions = request.metadata.responseInstructions {
            context["instructions"] = responseInstructions
        }
        if let responseTruncation = request.metadata.responseTruncation {
            context["truncation"] = responseTruncation
        }
        if let responseMetadata = request.metadata.responseMetadata {
            context["metadata"] = try sendableValue(from: responseMetadata)
        }
        if let n = request.metadata.n {
            context["n"] = n
        }
        if let streamOptions = request.metadata.streamOptions {
            context["stream_options"] = try sendableValue(from: streamOptions)
        }
        if let modalities = request.metadata.modalities {
            context["modalities"] = try modalities.map { try sendableValue(from: $0) }
        }
        if let audio = request.metadata.audio {
            context["audio"] = try sendableValue(from: audio)
        }
        if let prediction = request.metadata.prediction {
            context["prediction"] = try sendableValue(from: prediction)
        }
        if let previousResponseID = request.metadata.previousResponseID {
            context["previous_response_id"] = previousResponseID
        }
        if let include = request.metadata.include {
            context["include"] = try include.map { try sendableValue(from: $0) }
        }
        if let parallelToolCalls = request.metadata.parallelToolCalls {
            context["parallel_tool_calls"] = parallelToolCalls
        }
        if let store = request.metadata.store {
            context["store"] = store
        }
        if let serviceTier = request.metadata.serviceTier {
            context["service_tier"] = serviceTier
        }
        if let responseReasoning = request.metadata.responseReasoning {
            context["reasoning"] = try sendableValue(from: responseReasoning)
        }
        if let adapterPath = request.metadata.adapterPath {
            context["adapter_path"] = adapterPath
        }
        if let draftModel = request.metadata.draftModel {
            context["draft_model"] = draftModel
        }
        if let draftKind = request.metadata.draftKind {
            context["draft_kind"] = draftKind
        }
        if let draftBlockSize = request.metadata.draftBlockSize {
            context["draft_block_size"] = draftBlockSize
        }
        if let resizeShape = request.metadata.resizeShape {
            context["resize_shape"] = resizeShape
        }
        if let prefillStepSize = request.parameters.prefillStepSize {
            context["prefill_step_size"] = prefillStepSize
        }
        if let quantizedKVStart = request.parameters.quantizedKVStart {
            context["quantized_kv_start"] = quantizedKVStart
        }
        if let typicalP = request.parameters.typicalP {
            context["typical_p"] = typicalP
        }
        if let tfsZ = request.parameters.tfsZ {
            context["tfs_z"] = tfsZ
        }
        if let mirostat = request.parameters.mirostat {
            context["mirostat"] = mirostat
        }
        if let mirostatTau = request.parameters.mirostatTau {
            context["mirostat_tau"] = mirostatTau
        }
        if let mirostatEta = request.parameters.mirostatEta {
            context["mirostat_eta"] = mirostatEta
        }
        if let penalizeNewline = request.parameters.penalizeNewline {
            context["penalize_newline"] = penalizeNewline
        }

        let audioReferences = request.messages.flatMap(\.content).compactMap { part -> String? in
            guard case .audioURL(let reference) = part else {
                return nil
            }
            return reference.data
        }
        if !audioReferences.isEmpty {
            context["audio"] = audioReferences.count == 1 ? audioReferences[0] : audioReferences
            context["audios"] = audioReferences
        }
        let audioFormats = request.messages.flatMap(\.content).compactMap { part -> String? in
            guard case .audioURL(let reference) = part else {
                return nil
            }
            return reference.format
        }
        if !audioFormats.isEmpty {
            context["audio_format"] = audioFormats.count == 1 ? audioFormats[0] : audioFormats
        }

        let videos = request.messages.flatMap(\.content).compactMap { part -> VideoReference? in
            guard case .videoURL(let reference) = part else {
                return nil
            }
            return reference
        }
        let videoFPS = videos.compactMap(\.fps)
        let videoMinPixels = videos.compactMap(\.minPixels)
        let videoMaxPixels = videos.compactMap(\.maxPixels)
        let videoNFrames = videos.compactMap(\.nframes)
        let videoMinFrames = videos.compactMap(\.minFrames)
        let videoMaxFrames = videos.compactMap(\.maxFrames)
        if !videoFPS.isEmpty {
            context["fps"] = videoFPS.count == 1 ? videoFPS[0] : videoFPS
        }
        if !videoMinPixels.isEmpty {
            context["min_pixels"] = videoMinPixels.count == 1 ? videoMinPixels[0] : videoMinPixels
        }
        if !videoMaxPixels.isEmpty {
            context["max_pixels"] = videoMaxPixels.count == 1 ? videoMaxPixels[0] : videoMaxPixels
        }
        if !videoNFrames.isEmpty {
            context["nframes"] = videoNFrames.count == 1 ? videoNFrames[0] : videoNFrames
        }
        if !videoMinFrames.isEmpty {
            context["min_frames"] = videoMinFrames.count == 1 ? videoMinFrames[0] : videoMinFrames
        }
        if !videoMaxFrames.isEmpty {
            context["max_frames"] = videoMaxFrames.count == 1 ? videoMaxFrames[0] : videoMaxFrames
        }

        return context.isEmpty ? nil : context
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

    private static func rawMessage(from message: ChatMessage) throws -> MLXLMCommon.Message {
        var raw: MLXLMCommon.Message = [
            "role": rawRoleName(message.role),
            "content": try rawContent(from: message.content),
        ]
        if let reasoning = message.reasoning, !reasoning.isEmpty {
            raw["reasoning"] = reasoning
        }
        if let toolCalls = message.toolCalls {
            raw["tool_calls"] = try toolCalls.map { try sendableValue(from: $0.normalizingToolCallFunctionArguments()) }
        }
        if let toolCallID = message.toolCallID, !toolCallID.isEmpty {
            raw["tool_call_id"] = toolCallID
        }
        if let name = message.name, !name.isEmpty {
            raw["name"] = name
        }
        return raw
    }

    private static func rawContent(from parts: [ContentPart]) throws -> any Sendable {
        let hasMedia = parts.contains { part in
            switch part {
            case .text:
                return false
            case .imagePlaceholder, .imageURL, .audioPlaceholder, .audioURL, .videoURL:
                return true
            }
        }
        if !hasMedia {
            return parts.compactMap { part -> String? in
                guard case .text(let value) = part else {
                    return nil
                }
                return value
            }.joined(separator: " ")
        }
        return try parts.map(rawContentPart(from:)) as [any Sendable]
    }

    private static func rawContentPart(from part: ContentPart) throws -> any Sendable {
        switch part {
        case .text(let text):
            return [
                "type": "text",
                "text": text,
                "content": text,
            ] as [String: any Sendable]
        case .imagePlaceholder:
            return ["type": "image"] as [String: any Sendable]
        case .imageURL(let reference):
            var raw: [String: any Sendable] = [
                "type": "image",
                "image": reference.url,
            ]
            if let detail = reference.detail {
                raw["detail"] = detail
            }
            if let resizedHeight = reference.resizedHeight {
                raw["resized_height"] = resizedHeight
            }
            if let resizedWidth = reference.resizedWidth {
                raw["resized_width"] = resizedWidth
            }
            if let minPixels = reference.minPixels {
                raw["min_pixels"] = minPixels
            }
            if let maxPixels = reference.maxPixels {
                raw["max_pixels"] = maxPixels
            }
            return raw
        case .audioPlaceholder:
            return ["type": "audio"] as [String: any Sendable]
        case .audioURL(let reference):
            var raw: [String: any Sendable] = [
                "type": "audio",
                "input_audio": [
                    "data": reference.data,
                ] as [String: any Sendable],
            ]
            if let format = reference.format {
                raw["input_audio"] = [
                    "data": reference.data,
                    "format": format,
                ] as [String: any Sendable]
            }
            return raw
        case .videoURL(let reference):
            var raw: [String: any Sendable] = [
                "type": "video",
                "video": reference.url,
            ]
            if let minPixels = reference.minPixels {
                raw["min_pixels"] = minPixels
            }
            if let maxPixels = reference.maxPixels {
                raw["max_pixels"] = maxPixels
            }
            if let fps = reference.fps {
                raw["fps"] = fps
            }
            if let nframes = reference.nframes {
                raw["nframes"] = nframes
            }
            if let minFrames = reference.minFrames {
                raw["min_frames"] = minFrames
            }
            if let maxFrames = reference.maxFrames {
                raw["max_frames"] = maxFrames
            }
            return raw
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

    private static func sendableLogitBias(from value: MLXVLMCore.JSONValue) throws -> any Sendable {
        guard case .object(let object) = value else {
            return try sendableValue(from: value)
        }

        var result: [Int: Double] = [:]
        for (key, value) in object {
            guard let tokenID = Int(key) else {
                continue
            }
            if let number = value.doubleValue {
                result[tokenID] = number
            }
        }
        return result
    }

    private static func chatMessage(from message: ChatMessage) throws -> MLXLMCommon.Chat.Message {
        let text = message.content.compactMap { part -> String? in
            if case .text(let value) = part {
                return value
            }
            return nil
        }.joined(separator: " ")
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

    private static func rawRoleName(_ role: MessageRole) -> String {
        switch role {
        case .developer:
            return "system"
        default:
            return role.rawValue
        }
    }

    private static func image(from part: ContentPart) throws -> UserInput.Image? {
        guard case .imageURL(let reference) = part else {
            return nil
        }
        return .url(try mediaURL(from: reference.url, kind: .image))
    }

    private static func video(from part: ContentPart) throws -> UserInput.Video? {
        guard case .videoURL(let reference) = part else {
            return nil
        }
        return .url(try mediaURL(from: reference.url, kind: .video))
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
        let relativeURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
            .appendingPathComponent(reference)
            .standardizedFileURL
        if FileManager.default.fileExists(atPath: relativeURL.path) {
            return relativeURL
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
            if URL(string: normalized)?.scheme != nil ||
                normalized.hasPrefix("/") ||
                normalized.hasPrefix("~") ||
                normalized.hasPrefix(".") ||
                normalized.contains(":")
            {
                return nil
            }
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
        case "video/x-msvideo":
            return "avi"
        case "video/webm":
            return "webm"
        case "video/x-matroska":
            return "mkv"
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
