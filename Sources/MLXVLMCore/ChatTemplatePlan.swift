import Foundation

public struct ChatTemplatePlan: Codable, Equatable, Sendable {
    public let hasTemplate: Bool
    public let source: String?
    public let templateCharacterCount: Int
    public let containsJinjaSyntax: Bool
    public let containsQwenMarkers: Bool
    public let containsGenerationPromptVariable: Bool
    public let requiredRenderer: String
    public let canRenderNatively: Bool
    public let fallbackRenderer: String
    public let warnings: [String]

    public init(
        hasTemplate: Bool,
        source: String?,
        templateCharacterCount: Int,
        containsJinjaSyntax: Bool,
        containsQwenMarkers: Bool,
        containsGenerationPromptVariable: Bool,
        requiredRenderer: String,
        canRenderNatively: Bool,
        fallbackRenderer: String,
        warnings: [String]
    ) {
        self.hasTemplate = hasTemplate
        self.source = source
        self.templateCharacterCount = templateCharacterCount
        self.containsJinjaSyntax = containsJinjaSyntax
        self.containsQwenMarkers = containsQwenMarkers
        self.containsGenerationPromptVariable = containsGenerationPromptVariable
        self.requiredRenderer = requiredRenderer
        self.canRenderNatively = canRenderNatively
        self.fallbackRenderer = fallbackRenderer
        self.warnings = warnings
    }
}

public struct ChatTemplatePlanner {
    public init() {}

    public func plan(descriptor: ModelDescriptor) -> ChatTemplatePlan {
        let fallbackRenderer = isQwenVL(descriptor) ? "qwen-chat" : "plain"
        guard let template = descriptor.tokenizerMetadata.chatTemplate else {
            return ChatTemplatePlan(
                hasTemplate: false,
                source: nil,
                templateCharacterCount: 0,
                containsJinjaSyntax: false,
                containsQwenMarkers: false,
                containsGenerationPromptVariable: false,
                requiredRenderer: "plain-fallback",
                canRenderNatively: true,
                fallbackRenderer: fallbackRenderer,
                warnings: ["No chat template was found; Swift will use the \(fallbackRenderer) prompt fallback."]
            )
        }

        let containsJinjaSyntax = template.contains("{%") ||
            template.contains("{{") ||
            template.contains("{#")
        let containsQwenMarkers = template.contains("<|im_start|>") &&
            template.contains("<|im_end|>")
        let containsLlama3Markers = template.contains("<|start_header_id|>") &&
            template.contains("<|end_header_id|>") &&
            template.contains("<|eot_id|>")
        let containsMistralMarkers = template.contains("[INST]") &&
            template.contains("[/INST]")
        let containsGemma4Markers = template.contains("<|turn>") &&
            template.contains("<turn|>")
        let containsGenerationPromptVariable = template.contains("add_generation_prompt")
        let qwen = isQwenVL(descriptor)
        let requiredRenderer: String
        let canRenderNatively: Bool
        var warnings: [String] = []

        if qwen, containsQwenMarkers {
            requiredRenderer = "qwen-chat-builtin"
            canRenderNatively = true
            if containsJinjaSyntax {
                warnings.append("Swift can render the Qwen chat marker format, but does not execute every Jinja branch in the source template.")
            }
        } else if containsLlama3Markers {
            requiredRenderer = "llama3-chat-builtin"
            canRenderNatively = true
            if containsJinjaSyntax {
                warnings.append("Swift can render the Llama 3 header/eot marker format, but does not execute every Jinja branch in the source template.")
            }
        } else if containsMistralMarkers {
            requiredRenderer = "mistral-instruct-builtin"
            canRenderNatively = true
            if containsJinjaSyntax {
                warnings.append("Swift can render the Mistral [INST] marker format, but does not execute every Jinja branch in the source template.")
            }
        } else if descriptor.canonicalModelType == "gemma4" || containsGemma4Markers {
            requiredRenderer = "gemma4-chat-builtin"
            canRenderNatively = true
            if containsJinjaSyntax {
                warnings.append("Swift can render the Gemma4 turn/channel/tool-call marker format, but does not execute every Jinja branch in the source template.")
            }
        } else if containsJinjaSyntax {
            requiredRenderer = "jinja-template"
            canRenderNatively = false
            warnings.append("The chat template uses Jinja syntax and needs a Jinja-compatible renderer for exact formatting.")
        } else {
            requiredRenderer = "custom-template"
            canRenderNatively = false
            warnings.append("The chat template is custom and does not match a native Swift prompt renderer.")
        }

        return ChatTemplatePlan(
            hasTemplate: true,
            source: descriptor.tokenizerMetadata.chatTemplateSource,
            templateCharacterCount: template.count,
            containsJinjaSyntax: containsJinjaSyntax,
            containsQwenMarkers: containsQwenMarkers,
            containsGenerationPromptVariable: containsGenerationPromptVariable,
            requiredRenderer: requiredRenderer,
            canRenderNatively: canRenderNatively,
            fallbackRenderer: fallbackRenderer,
            warnings: warnings
        )
    }

    private func isQwenVL(_ descriptor: ModelDescriptor) -> Bool {
        switch descriptor.canonicalModelType {
        case "qwen2_vl",
             "qwen2_5_vl",
             "qwen3_vl",
             "qwen3_vl_moe",
             "qwen3_5",
             "qwen3_5_moe",
             "qwen3_omni_moe":
            return true
        default:
            return false
        }
    }
}
