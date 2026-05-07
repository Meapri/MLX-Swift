import Foundation

public struct ToolCallPlan: Codable, Equatable, Sendable {
    public let hasTools: Bool
    public let toolCount: Int
    public let toolNames: [String]
    public let toolChoiceMode: String
    public let forcedFunctionName: String?
    public let parserHint: String?
    public let requiresToolPrompt: Bool
    public let requiresToolParser: Bool
    public let backendMinimumFeatures: [String]
    public let warnings: [String]

    public init(
        hasTools: Bool,
        toolCount: Int,
        toolNames: [String],
        toolChoiceMode: String,
        forcedFunctionName: String?,
        parserHint: String?,
        requiresToolPrompt: Bool,
        requiresToolParser: Bool,
        backendMinimumFeatures: [String],
        warnings: [String]
    ) {
        self.hasTools = hasTools
        self.toolCount = toolCount
        self.toolNames = toolNames
        self.toolChoiceMode = toolChoiceMode
        self.forcedFunctionName = forcedFunctionName
        self.parserHint = parserHint
        self.requiresToolPrompt = requiresToolPrompt
        self.requiresToolParser = requiresToolParser
        self.backendMinimumFeatures = backendMinimumFeatures
        self.warnings = warnings
    }
}

public struct ToolCallPlanner {
    public init() {}

    public func plan(metadata: GenerationRequestMetadata, descriptor: ModelDescriptor) -> ToolCallPlan {
        let tools = metadata.tools ?? []
        let names = tools.compactMap(toolName).sorted()
        let choice = toolChoice(metadata.toolChoice)
        let parserHint = parserHint(descriptor: descriptor)
        let hasTools = !tools.isEmpty
        let requiresToolParser = hasTools || choice.mode != "auto"
        var features: [String] = []
        var warnings: [String] = []

        if hasTools {
            features.append("tool-schema-prompting")
        }
        if requiresToolParser {
            features.append("tool-call-output-parser")
        }
        if choice.forcedName != nil {
            features.append("forced-tool-choice")
        }
        if hasTools, parserHint == nil {
            warnings.append("No model-specific tool parser was inferred from the chat template.")
        }
        if let forcedName = choice.forcedName, !names.isEmpty, !names.contains(forcedName) {
            warnings.append("tool_choice forces '\(forcedName)' but that function is not present in tools.")
        }

        return ToolCallPlan(
            hasTools: hasTools,
            toolCount: tools.count,
            toolNames: names,
            toolChoiceMode: choice.mode,
            forcedFunctionName: choice.forcedName,
            parserHint: parserHint,
            requiresToolPrompt: hasTools,
            requiresToolParser: requiresToolParser,
            backendMinimumFeatures: features,
            warnings: warnings
        )
    }

    private func toolName(_ value: JSONValue) -> String? {
        if let function = value["function"]?.objectValue {
            return function["name"]?.stringValue
        }
        return value["name"]?.stringValue
    }

    private func toolChoice(_ value: JSONValue?) -> (mode: String, forcedName: String?) {
        guard let value else {
            return ("auto", nil)
        }
        if let string = value.stringValue {
            switch string {
            case "none", "auto", "required":
                return (string, nil)
            default:
                return ("named", string)
            }
        }
        if let functionName = value["function"]?["name"]?.stringValue {
            return ("function", functionName)
        }
        if let name = value["name"]?.stringValue {
            return ("function", name)
        }
        return ("unknown", nil)
    }

    private func parserHint(descriptor: ModelDescriptor) -> String? {
        if ToolCallOutputParser.usesGemma4ToolCalls(descriptor: descriptor) {
            return "gemma4"
        }
        let modelTypes = [
            descriptor.rawModelType.lowercased(),
            descriptor.canonicalModelType.lowercased(),
        ]
        if modelTypes.contains(where: { $0.hasPrefix("qwen3_5") || $0.hasPrefix("qwen3_next") || $0.hasPrefix("nemotron") }) {
            return "xml_function"
        }
        if modelTypes.contains(where: { $0.hasPrefix("glm4") }) {
            return "glm4"
        }
        if modelTypes.contains(where: { $0.hasPrefix("lfm2") }) {
            return "lfm2"
        }
        if modelTypes.contains(where: { $0.hasPrefix("mistral3") }) {
            return "mistral"
        }
        let chatTemplate = descriptor.tokenizerMetadata.chatTemplate
        guard let chatTemplate else {
            return nil
        }
        if chatTemplate.contains("tool_call") {
            return "template-tool-call"
        }
        return nil
    }
}
