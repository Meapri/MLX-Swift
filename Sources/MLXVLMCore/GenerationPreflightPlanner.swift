import Foundation

public enum PromptRenderStyle: String, Codable, Equatable, Sendable {
    case plain
    case qwenChat
    case llama3Chat
    case mistralInstruct
}

public struct GenerationPreflightPlan: Codable, Equatable, Sendable {
    public let requestModel: String
    public let canonicalModelType: String
    public let promptStyle: PromptRenderStyle
    public let prompt: String
    public let plainPrompt: String
    public let messageCount: Int
    public let messageMetadata: [ChatMessageMetadataPlan]
    public let parameters: GenerationParameters
    public let metadata: GenerationRequestMetadata
    public let capabilities: ModelCapabilityPlan
    public let promptRender: OllamaPromptRenderResult
    public let responseFormatPlan: ResponseFormatPlan
    public let toolCallPlan: ToolCallPlan
    public let runtime: GenerationRuntimePlan
    public let stream: Bool
    public let media: MediaResolutionReport
    public let imageInputs: QwenVLImageInputReport
    public let imagePixels: QwenVLImagePixelPreflightReport
    public let tokenization: TokenizationPreflightPlan?
    public let compatibilityChecks: [CompatibilityCheck]
    public let metadataReady: Bool
    public let generationReady: Bool
    public let canAttemptGeneration: Bool
    public let blockingReasons: [String]

    public init(
        requestModel: String,
        canonicalModelType: String,
        promptStyle: PromptRenderStyle,
        prompt: String,
        plainPrompt: String,
        messageCount: Int,
        messageMetadata: [ChatMessageMetadataPlan],
        parameters: GenerationParameters,
        metadata: GenerationRequestMetadata,
        capabilities: ModelCapabilityPlan,
        promptRender: OllamaPromptRenderResult,
        responseFormatPlan: ResponseFormatPlan,
        toolCallPlan: ToolCallPlan,
        runtime: GenerationRuntimePlan,
        stream: Bool,
        media: MediaResolutionReport,
        imageInputs: QwenVLImageInputReport,
        imagePixels: QwenVLImagePixelPreflightReport,
        tokenization: TokenizationPreflightPlan?,
        compatibilityChecks: [CompatibilityCheck],
        metadataReady: Bool,
        generationReady: Bool,
        canAttemptGeneration: Bool,
        blockingReasons: [String]
    ) {
        self.requestModel = requestModel
        self.canonicalModelType = canonicalModelType
        self.promptStyle = promptStyle
        self.prompt = prompt
        self.plainPrompt = plainPrompt
        self.messageCount = messageCount
        self.messageMetadata = messageMetadata
        self.parameters = parameters
        self.metadata = metadata
        self.capabilities = capabilities
        self.promptRender = promptRender
        self.responseFormatPlan = responseFormatPlan
        self.toolCallPlan = toolCallPlan
        self.runtime = runtime
        self.stream = stream
        self.media = media
        self.imageInputs = imageInputs
        self.imagePixels = imagePixels
        self.tokenization = tokenization
        self.compatibilityChecks = compatibilityChecks
        self.metadataReady = metadataReady
        self.generationReady = generationReady
        self.canAttemptGeneration = canAttemptGeneration
        self.blockingReasons = blockingReasons
    }
}

public struct ChatMessageMetadataPlan: Codable, Equatable, Sendable {
    public let index: Int
    public let role: MessageRole
    public let contentTypes: [String]
    public let name: String?
    public let reasoningPresent: Bool
    public let reasoning: String?
    public let toolCallID: String?
    public let toolCallCount: Int
    public let toolCalls: [JSONValue]?

    enum CodingKeys: String, CodingKey {
        case index
        case role
        case contentTypes = "content_types"
        case name
        case reasoningPresent = "reasoning_present"
        case reasoning
        case toolCallID = "tool_call_id"
        case toolCallCount = "tool_call_count"
        case toolCalls = "tool_calls"
    }

    public init(index: Int, message: ChatMessage) {
        self.index = index
        self.role = message.role
        self.contentTypes = message.content.map(Self.contentType)
        self.name = message.name
        self.reasoningPresent = message.reasoning?.isEmpty == false
        self.reasoning = message.reasoning
        self.toolCallID = message.toolCallID
        self.toolCallCount = message.toolCalls?.count ?? 0
        self.toolCalls = message.toolCalls
    }

    private static func contentType(_ part: ContentPart) -> String {
        switch part {
        case .text:
            return "text"
        case .imageURL:
            return "image"
        case .audioURL:
            return "audio"
        case .videoURL:
            return "video"
        }
    }
}

public struct GenerationPreflightPlanner {
    public let descriptor: ModelDescriptor
    public let backend: BackendStatus
    public let mediaResolver: MediaReferenceResolver
    public let imageGridConfig: QwenVLImageGridConfig?

    public init(
        descriptor: ModelDescriptor,
        backend: BackendStatus = .compatibilityShell,
        mediaResolver: MediaReferenceResolver = MediaReferenceResolver(),
        imageGridConfig: QwenVLImageGridConfig? = nil
    ) {
        self.descriptor = descriptor
        self.backend = backend
        self.mediaResolver = mediaResolver
        self.imageGridConfig = imageGridConfig
    }

    public func plan(request: GenerationRequest) -> GenerationPreflightPlan {
        let builder = QwenVLPromptBuilder(
            imageToken: descriptor.tokenizerMetadata.imageToken ?? "<|image_pad|>",
            videoToken: descriptor.tokenizerMetadata.videoToken ?? "<|video_pad|>"
        )
        let style = promptStyle(for: descriptor)
        let modelPrompt = switch style {
        case .plain:
            builder.plainPrompt(messages: request.messages)
        case .qwenChat:
            builder.qwenChatPrompt(messages: request.messages)
        case .llama3Chat:
            builder.llama3ChatPrompt(messages: request.messages)
        case .mistralInstruct:
            builder.mistralInstructPrompt(messages: request.messages)
        }
        let plainPrompt = builder.plainPrompt(messages: request.messages)
        let promptRender = OllamaPromptRenderer(builder: builder).render(
            request: request,
            defaultPrompt: modelPrompt
        )
        let prompt = promptRender.prompt
        let media = mediaResolver.report(for: request)
        let gridConfig = imageGridConfig ?? descriptor.processorMetadata.qwenImageGridConfig(defaultPatchSize: qwenDefaultPatchSize())
        let imageInputs = QwenVLImageInputPlanner(
            resolver: mediaResolver,
            config: gridConfig
        ).plan(request: request)
        let imagePixels = QwenVLImagePixelPreflightPlanner(
            resolver: mediaResolver,
            config: gridConfig,
            normalization: descriptor.processorMetadata.qwenPatchNormalization()
        ).plan(request: request)
        let tokenization = tokenizationPlan(prompt: prompt)
        let qwenConfig = try? QwenVLModelConfig.load(fromModelDirectory: descriptor.path)
        let runtime = GenerationRuntimePlan(
            request: request,
            qwenVLConfig: qwenConfig,
            tokenIDs: tokenization?.tokenIDs
        )
        let responseFormatPlan = ResponseFormatPlanner().plan(
            metadata: request.metadata,
            stream: request.stream
        )
        let toolCallPlan = ToolCallPlanner().plan(
            metadata: request.metadata,
            descriptor: descriptor
        )
        let compatibility = ModelCompatibilityValidator.validate(descriptor: descriptor, backend: backend)
        let capabilities = ModelCapabilityPlanner().plan(descriptor: descriptor)
        let blockingReasons = blockingReasons(
            capabilities: capabilities,
            compatibility: compatibility,
            media: media,
            imageInputs: imageInputs,
            imagePixels: imagePixels
        )

        return GenerationPreflightPlan(
            requestModel: request.model,
            canonicalModelType: descriptor.canonicalModelType,
            promptStyle: style,
            prompt: prompt,
            plainPrompt: plainPrompt,
            messageCount: request.messages.count,
            messageMetadata: request.messages.enumerated().map {
                ChatMessageMetadataPlan(index: $0.offset, message: $0.element)
            },
            parameters: request.parameters,
            metadata: request.metadata,
            capabilities: capabilities,
            promptRender: promptRender,
            responseFormatPlan: responseFormatPlan,
            toolCallPlan: toolCallPlan,
            runtime: runtime,
            stream: request.stream,
            media: media,
            imageInputs: imageInputs,
            imagePixels: imagePixels,
            tokenization: tokenization,
            compatibilityChecks: compatibility.checks,
            metadataReady: compatibility.metadataReady,
            generationReady: compatibility.generationReady,
            canAttemptGeneration: compatibility.generationReady && blockingReasons.isEmpty,
            blockingReasons: blockingReasons
        )
    }

    private func promptStyle(for descriptor: ModelDescriptor) -> PromptRenderStyle {
        switch descriptor.canonicalModelType {
        case "qwen2_vl", "qwen2_5_vl":
            return .qwenChat
        default:
            let template = descriptor.tokenizerMetadata.chatTemplate ?? ""
            if template.contains("<|start_header_id|>"),
               template.contains("<|end_header_id|>"),
               template.contains("<|eot_id|>")
            {
                return .llama3Chat
            }
            if template.contains("[INST]"), template.contains("[/INST]") {
                return .mistralInstruct
            }
            return .plain
        }
    }

    private func qwenDefaultPatchSize() -> Int {
        switch descriptor.canonicalModelType {
        case "qwen2_vl", "qwen2_5_vl":
            return 14
        default:
            return 14
        }
    }

    private func tokenizationPlan(prompt: String) -> TokenizationPreflightPlan? {
        guard let catalog = TokenizerCatalogBuilder().catalog(for: descriptor) else {
            return nil
        }
        let plan = TokenizerImplementationPlanner().plan(descriptor: descriptor, catalog: catalog)
        let simpleResult = SimpleTokenizer(catalog: catalog, plan: plan).tokenize(prompt)
        if simpleResult.supported {
            return TokenizationPreflightPlan(
                tokenizerModelType: catalog.modelType,
                tokenizerResult: simpleResult
            )
        }
        return TokenizationPreflightPlanner(catalog: catalog).plan(prompt: prompt)
    }

    private func blockingReasons(
        capabilities: ModelCapabilityPlan,
        compatibility: ModelCompatibilityReport,
        media: MediaResolutionReport,
        imageInputs: QwenVLImageInputReport,
        imagePixels: QwenVLImagePixelPreflightReport
    ) -> [String] {
        var reasons = compatibility.checks
            .filter { $0.severity == .error && !$0.passed }
            .map(\.message)

        if !capabilities.supportsOllamaGenerationAPI {
            reasons.append("Model type \(descriptor.canonicalModelType) uses \(capabilities.primaryTask) and is not compatible with text generation endpoints.")
        }
        if media.errorCount > 0 {
            reasons.append("\(media.errorCount) media reference(s) are not loadable.")
        }
        if imageInputs.errorCount > 0 {
            reasons.append("\(imageInputs.errorCount) image input(s) could not be planned.")
        }
        if imagePixels.errorCount > imageInputs.errorCount {
            reasons.append("\(imagePixels.errorCount) image input(s) could not be decoded and resized.")
        }
        return reasons
    }
}
