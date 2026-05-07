import Foundation

public struct OllamaPromptRenderResult: Codable, Equatable, Sendable {
    public let prompt: String
    public let source: String
    public let appliedTemplate: Bool
    public let appliedRawPrompt: Bool
    public let appendedSuffix: Bool
    public let warnings: [String]

    public init(
        prompt: String,
        source: String,
        appliedTemplate: Bool,
        appliedRawPrompt: Bool,
        appendedSuffix: Bool,
        warnings: [String]
    ) {
        self.prompt = prompt
        self.source = source
        self.appliedTemplate = appliedTemplate
        self.appliedRawPrompt = appliedRawPrompt
        self.appendedSuffix = appendedSuffix
        self.warnings = warnings
    }
}

public struct OllamaPromptRenderer {
    public let builder: QwenVLPromptBuilder

    public init(builder: QwenVLPromptBuilder = QwenVLPromptBuilder()) {
        self.builder = builder
    }

    public func render(request: GenerationRequest, defaultPrompt: String) -> OllamaPromptRenderResult {
        guard request.metadata.rawPrompt ||
            request.metadata.template != nil ||
            request.metadata.suffix != nil
        else {
            return OllamaPromptRenderResult(
                prompt: defaultPrompt,
                source: "model-prompt-renderer",
                appliedTemplate: false,
                appliedRawPrompt: false,
                appendedSuffix: false,
                warnings: []
            )
        }

        let system = joinedContent(for: [.system, .developer], in: request.messages)
        let prompt = joinedContent(for: [.user], in: request.messages)
        let suffix = request.metadata.suffix ?? ""

        if let template = request.metadata.template, !template.isEmpty {
            var rendered = template
            let replacements = [
                ".System": system,
                ".Prompt": prompt,
                ".Suffix": suffix,
                ".Response": "",
            ]
            for (name, value) in replacements {
                rendered = replaceGoTemplateVariable(name, with: value, in: rendered)
            }
            return OllamaPromptRenderResult(
                prompt: rendered.trimmingCharacters(in: .whitespacesAndNewlines),
                source: "ollama-template",
                appliedTemplate: true,
                appliedRawPrompt: request.metadata.rawPrompt,
                appendedSuffix: templateContainsSuffix(template),
                warnings: templateWarnings(template)
            )
        }

        var rendered = request.metadata.rawPrompt ? prompt : defaultPrompt
        var appendedSuffix = false
        if !suffix.isEmpty {
            rendered += suffix
            appendedSuffix = true
        }

        return OllamaPromptRenderResult(
            prompt: rendered.trimmingCharacters(in: .whitespacesAndNewlines),
            source: request.metadata.rawPrompt ? "ollama-raw" : "model-prompt-renderer+suffix",
            appliedTemplate: false,
            appliedRawPrompt: request.metadata.rawPrompt,
            appendedSuffix: appendedSuffix,
            warnings: []
        )
    }

    private func joinedContent(for roles: Set<MessageRole>, in messages: [ChatMessage]) -> String {
        messages
            .filter { roles.contains($0.role) }
            .map { builder.flatten(content: $0.content) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private func replaceGoTemplateVariable(_ name: String, with value: String, in template: String) -> String {
        var rendered = template
        let compact = "{{\(name)}}"
        let spaced = "{{ \(name) }}"
        rendered = rendered.replacingOccurrences(of: compact, with: value)
        rendered = rendered.replacingOccurrences(of: spaced, with: value)
        return rendered
    }

    private func templateContainsSuffix(_ template: String) -> Bool {
        template.contains("{{.Suffix}}") || template.contains("{{ .Suffix }}")
    }

    private func templateWarnings(_ template: String) -> [String] {
        var warnings: [String] = []
        if template.contains("{{if") || template.contains("{{ if") ||
            template.contains("{{range") || template.contains("{{ range")
        {
            warnings.append("Only direct Ollama template variables are rendered; Go template conditionals and ranges require a full template engine.")
        }
        return warnings
    }
}
