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

    public func flatten(content: [ContentPart]) -> String {
        var parts: [String] = []
        for part in content {
            switch part {
            case .text(let text):
                if !text.isEmpty {
                    parts.append(text)
                }
            case .imageURL:
                parts.append(imageToken)
            case .audioURL:
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
}
