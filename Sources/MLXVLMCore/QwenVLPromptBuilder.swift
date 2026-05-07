import Foundation

public struct QwenVLPromptBuilder: Sendable {
    public let imageToken: String
    public let videoToken: String
    public let audioToken: String

    public init(
        imageToken: String = "<|image_pad|>",
        videoToken: String = "<|video_pad|>",
        audioToken: String = "<audio>"
    ) {
        self.imageToken = imageToken
        self.videoToken = videoToken
        self.audioToken = audioToken
    }

    public func plainPrompt(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true
    ) -> String {
        let normalized = messages.map { message in
            (role: message.role, content: flatten(content: message.content))
        }

        guard !normalized.isEmpty else {
            return ""
        }

        if normalized.count == 1, normalized[0].role == .user {
            return normalized[0].content
        }

        var lines: [String] = []
        for message in normalized {
            let prefix = rolePrefix(message.role)
            if message.content.isEmpty {
                lines.append("\(prefix):")
            } else {
                lines.append("\(prefix): \(message.content)")
            }
        }

        if addGenerationPrompt {
            lines.append("Assistant:")
        }

        return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func qwenChatPrompt(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true
    ) -> String {
        var output = ""
        for message in messages {
            output += "<|im_start|>\(qwenRoleName(message.role))\n"
            output += flatten(content: message.content)
            output += "<|im_end|>\n"
        }
        if addGenerationPrompt {
            output += "<|im_start|>assistant\n"
        }
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func llama3ChatPrompt(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true
    ) -> String {
        var output = ""
        for message in messages {
            output += "<|start_header_id|>\(llamaRoleName(message.role))<|end_header_id|>\n\n"
            output += flatten(content: message.content)
            output += "<|eot_id|>"
        }
        if addGenerationPrompt {
            output += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        }
        return output
    }

    public func mistralInstructPrompt(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true
    ) -> String {
        var output = ""
        var pendingSystem = ""

        for message in messages {
            let content = flatten(content: message.content)
            switch message.role {
            case .system, .developer:
                if !content.isEmpty {
                    pendingSystem += pendingSystem.isEmpty ? content : "\n\(content)"
                }
            case .user, .tool:
                let userContent = [pendingSystem, content]
                    .filter { !$0.isEmpty }
                    .joined(separator: "\n\n")
                pendingSystem.removeAll()
                output += "[INST] \(userContent) [/INST]"
            case .assistant:
                if !content.isEmpty {
                    output += " \(content)</s>"
                }
            }
        }

        if addGenerationPrompt, output.isEmpty, !pendingSystem.isEmpty {
            output += "[INST] \(pendingSystem) [/INST]"
        }
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func gemma4ChatPrompt(
        messages: [ChatMessage],
        tools: [JSONValue]? = nil,
        enableThinking: Bool = false,
        addGenerationPrompt: Bool = true
    ) -> String {
        guard !messages.isEmpty else {
            return "<bos>"
        }

        var output = "<bos>"
        var startIndex = 0
        var previousMessageType: String?
        let first = messages[0]
        let hasSystemLead = first.role == .system || first.role == .developer

        if enableThinking || tools?.isEmpty == false || hasSystemLead {
            output += "<|turn>system\n"
            if enableThinking {
                output += "<|think|>\n"
                previousMessageType = "think"
            }
            if hasSystemLead {
                output += flatten(content: first.content).trimmingCharacters(in: .whitespacesAndNewlines)
                startIndex = 1
            }
            if let tools, !tools.isEmpty {
                for tool in tools {
                    output += gemma4ToolDeclaration(tool)
                }
                previousMessageType = "tool"
            }
            output += "<turn|>\n"
        }

        let lastUserIndex = messages.lastIndex { message in
            message.role == .user
        } ?? -1

        for index in startIndex..<messages.count {
            let message = messages[index]
            guard message.role != .tool else {
                continue
            }
            previousMessageType = nil
            let role = gemma4RoleName(message.role)
            let previousNonToolRole = previousNonToolRole(before: index, in: messages)
            let continueSameModelTurn = role == "model" && previousNonToolRole == .assistant
            if !continueSameModelTurn {
                output += "<|turn>\(role)\n"
            }

            if let reasoning = message.reasoning,
               !reasoning.isEmpty,
               index > lastUserIndex,
               message.toolCalls?.isEmpty == false
            {
                output += "<|channel>thought\n\(reasoning)\n<channel|>"
            }

            if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
                for toolCall in toolCalls {
                    output += gemma4ToolCall(toolCall)
                }
                previousMessageType = "tool_call"
            }

            var renderedToolResponse = false
            if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
                for follow in followingToolMessages(after: index, in: messages) {
                    let toolName = toolName(for: follow, resolvingAgainst: toolCalls)
                    output += gemma4ToolResponse(toolName: toolName, content: flatten(content: follow.content))
                    renderedToolResponse = true
                    previousMessageType = "tool_response"
                }
            }

            let content = flatten(content: message.content).trimmingCharacters(in: .whitespacesAndNewlines)
            if !content.isEmpty {
                output += role == "model" ? stripGemma4Thinking(from: content) : content
            }

            if previousMessageType == "tool_call", !renderedToolResponse {
                output += "<|tool_response>"
            } else if !(renderedToolResponse && content.isEmpty) {
                output += "<turn|>\n"
            }
        }

        if addGenerationPrompt,
           previousMessageType != "tool_response",
           previousMessageType != "tool_call"
        {
            output += "<|turn>model\n"
        }

        return output
    }

    public func flatten(content: [ContentPart]) -> String {
        var parts: [String] = []
        for part in content {
            switch part {
            case .text(let text):
                if !text.isEmpty {
                    parts.append(text)
                }
            case .imagePlaceholder, .imageURL:
                parts.append(imageToken)
            case .audioPlaceholder, .audioURL:
                parts.append(audioToken)
            case .videoURL:
                parts.append(videoToken)
            }
        }
        return stitch(parts: parts)
    }

    private func stitch(parts: [String]) -> String {
        let markers: Set<String> = [imageToken, videoToken, audioToken, "<image>", "<video>", "<audio>"]
        var output = ""
        var previousWasMarker = false

        for part in parts where !part.isEmpty {
            let currentIsMarker = markers.contains(part)
            if previousWasMarker,
               !currentIsMarker,
               let first = part.first,
               !first.isWhitespace
            {
                output += " "
            }
            output += part
            previousWasMarker = currentIsMarker
        }

        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func rolePrefix(_ role: MessageRole) -> String {
        switch role {
        case .system:
            return "System"
        case .developer:
            return "Developer"
        case .user:
            return "User"
        case .assistant:
            return "Assistant"
        case .tool:
            return "Tool"
        }
    }

    private func qwenRoleName(_ role: MessageRole) -> String {
        switch role {
        case .developer:
            return "system"
        case .tool:
            return "user"
        default:
            return role.rawValue
        }
    }

    private func llamaRoleName(_ role: MessageRole) -> String {
        switch role {
        case .developer:
            return "system"
        default:
            return role.rawValue
        }
    }

    private func gemma4RoleName(_ role: MessageRole) -> String {
        switch role {
        case .assistant:
            return "model"
        case .developer:
            return "system"
        default:
            return role.rawValue
        }
    }

    private func previousNonToolRole(before index: Int, in messages: [ChatMessage]) -> MessageRole? {
        guard index > 0 else {
            return nil
        }
        for candidate in stride(from: index - 1, through: 0, by: -1) {
            if messages[candidate].role != .tool {
                return messages[candidate].role
            }
        }
        return nil
    }

    private func followingToolMessages(after index: Int, in messages: [ChatMessage]) -> [ChatMessage] {
        guard index + 1 < messages.count else {
            return []
        }
        var result: [ChatMessage] = []
        for candidate in messages[(index + 1)...] {
            guard candidate.role == .tool else {
                break
            }
            result.append(candidate)
        }
        return result
    }

    private func toolName(for message: ChatMessage, resolvingAgainst toolCalls: [JSONValue]) -> String {
        if let name = message.name, !name.isEmpty {
            return name
        }
        guard let toolCallID = message.toolCallID else {
            return "unknown"
        }
        for toolCall in toolCalls {
            guard let object = toolCall.objectValue,
                  object["id"]?.stringValue == toolCallID,
                  let name = object["function"]?.objectValue?["name"]?.stringValue,
                  !name.isEmpty
            else {
                continue
            }
            return name
        }
        return "unknown"
    }

    private func gemma4ToolCall(_ value: JSONValue) -> String {
        guard let object = value.objectValue,
              let function = object["function"]?.objectValue,
              let name = function["name"]?.stringValue
        else {
            return ""
        }
        let arguments = function["arguments"] ?? .object([:])
        return "<|tool_call>call:\(name)\(gemma4Argument(arguments, escapeKeys: false))<tool_call|>"
    }

    private func gemma4ToolResponse(toolName: String, content: String) -> String {
        "<|tool_response>response:\(toolName){value:\(gemma4Argument(.string(content), escapeKeys: false))}<tool_response|>"
    }

    private func gemma4ToolDeclaration(_ value: JSONValue) -> String {
        guard let object = value.objectValue else {
            return ""
        }
        let function = object["function"]?.objectValue ?? object
        guard let name = function["name"]?.stringValue else {
            return ""
        }
        var fields: [String] = []
        if let description = function["description"]?.stringValue, !description.isEmpty {
            fields.append("description:\(gemma4Argument(.string(description)))")
        }
        if let parameters = function["parameters"] {
            fields.append("parameters:\(gemma4Argument(parameters, escapeKeys: false))")
        }
        if fields.isEmpty {
            fields.append("type:<|\"|>FUNCTION<|\"|>")
        }
        return "<|tool>declaration:\(name){\(fields.joined(separator: ","))}<tool|>"
    }

    private func gemma4Argument(_ value: JSONValue, escapeKeys: Bool = true) -> String {
        switch value {
        case .string(let string):
            return "<|\"|>\(string)<|\"|>"
        case .number(let number):
            if number.rounded() == number {
                return String(Int(number))
            }
            return String(number)
        case .bool(let bool):
            return bool ? "true" : "false"
        case .null:
            return "null"
        case .array(let values):
            return "[" + values.map { gemma4Argument($0, escapeKeys: escapeKeys) }.joined(separator: ",") + "]"
        case .object(let object):
            let fields = object.keys.sorted().map { key in
                let renderedKey = escapeKeys ? "<|\"|>\(key)<|\"|>" : key
                return "\(renderedKey):\(gemma4Argument(object[key] ?? .null, escapeKeys: escapeKeys))"
            }
            return "{\(fields.joined(separator: ","))}"
        }
    }

    private func stripGemma4Thinking(from text: String) -> String {
        var result = text
        while let start = result.range(of: "<|channel>"),
              let end = result.range(of: "<channel|>", range: start.upperBound..<result.endIndex)
        {
            result.removeSubrange(start.lowerBound..<end.upperBound)
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
