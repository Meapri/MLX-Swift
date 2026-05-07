import Foundation

public struct ThinkingOutputSplit: Codable, Equatable, Sendable {
    public let reasoning: String?
    public let content: String

    public init(reasoning: String?, content: String) {
        self.reasoning = reasoning
        self.content = content
    }
}

public struct ThinkingOutputSplitter: Sendable {
    public init() {}

    public func split(_ text: String) -> ThinkingOutputSplit {
        if text.contains("<|channel>thought") || (text.contains("<channel|>") && text.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("thought")) {
            let parts = text.components(separatedBy: "<channel|>")
            let rawReasoning = parts.first?
                .replacingOccurrences(of: "<|channel>thought", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let reasoning = rawReasoning?
                .strippingLeadingThoughtMarker()
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let content = parts.dropFirst().joined(separator: "<channel|>")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ThinkingOutputSplit(
                reasoning: reasoning?.isEmpty == false ? reasoning : nil,
                content: content
            )
        }

        if text.contains("<think>") || text.contains("</think>") {
            let parts = text.components(separatedBy: "</think>")
            let reasoning = parts.first?
                .replacingOccurrences(of: "<think>", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let content = parts.dropFirst().joined(separator: "</think>")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ThinkingOutputSplit(
                reasoning: reasoning?.isEmpty == false ? reasoning : nil,
                content: content
            )
        }

        return ThinkingOutputSplit(reasoning: nil, content: text)
    }
}

public struct StreamingThinkingOutputSplitter: Sendable {
    private enum State: Sendable {
        case content
        case reasoning(endTag: String)
    }

    private let startTags: [(start: String, end: String)] = [
        ("<|channel>thought", "<channel|>"),
        ("<think>", "</think>"),
    ]
    private var state: State = .content
    private var buffer = ""

    public init() {}

    public mutating func process(_ chunk: GenerationChunk) -> [GenerationChunk] {
        guard !chunk.text.isEmpty else {
            return [chunk]
        }
        buffer += chunk.text
        var output = drain(flush: chunk.isFinished, sourceLogprob: chunk.logprob)
        if chunk.isFinished {
            output.append(
                GenerationChunk(
                    text: "",
                    isFinished: true,
                    finishReason: chunk.finishReason,
                    promptTokenCount: chunk.promptTokenCount,
                    completionTokenCount: chunk.completionTokenCount,
                    toolCalls: chunk.toolCalls
                )
            )
        } else if !chunk.toolCalls.isEmpty {
            output.append(GenerationChunk(text: "", toolCalls: chunk.toolCalls))
        }
        return output
    }

    public mutating func finish() -> [GenerationChunk] {
        drain(flush: true, sourceLogprob: nil)
    }

    private mutating func drain(flush: Bool, sourceLogprob: GenerationTokenLogprob?) -> [GenerationChunk] {
        var output: [GenerationChunk] = []
        while !buffer.isEmpty {
            switch state {
            case .content:
                if let match = earliestStartTag(in: buffer) {
                    let before = String(buffer[..<match.range.lowerBound])
                    if !before.isEmpty {
                        output.append(GenerationChunk(text: before, logprob: sourceLogprob))
                    }
                    buffer.removeSubrange(..<match.range.upperBound)
                    state = .reasoning(endTag: match.end)
                    continue
                }
                let keep = flush ? 0 : longestSuffixPrefixLength(buffer, prefixes: startTags.map(\.start))
                let emitEnd = buffer.index(buffer.endIndex, offsetBy: -keep)
                let text = String(buffer[..<emitEnd])
                if !text.isEmpty {
                    output.append(GenerationChunk(text: text, logprob: sourceLogprob))
                    buffer.removeSubrange(..<emitEnd)
                }
                return output
            case .reasoning(let endTag):
                if let endRange = buffer.range(of: endTag) {
                    let reasoning = String(buffer[..<endRange.lowerBound])
                    if !reasoning.isEmpty {
                        output.append(GenerationChunk(text: "", reasoning: reasoning))
                    }
                    buffer.removeSubrange(..<endRange.upperBound)
                    state = .content
                    continue
                }
                let keep = flush ? 0 : longestSuffixPrefixLength(buffer, prefixes: [endTag])
                let emitEnd = buffer.index(buffer.endIndex, offsetBy: -keep)
                let reasoning = String(buffer[..<emitEnd])
                if !reasoning.isEmpty {
                    output.append(GenerationChunk(text: "", reasoning: reasoning))
                    buffer.removeSubrange(..<emitEnd)
                }
                return output
            }
        }
        return output
    }

    private func earliestStartTag(in text: String) -> (range: Range<String.Index>, end: String)? {
        startTags.compactMap { tag -> (range: Range<String.Index>, end: String)? in
            guard let range = text.range(of: tag.start) else {
                return nil
            }
            return (range, tag.end)
        }
        .min { $0.range.lowerBound < $1.range.lowerBound }
    }

    private func longestSuffixPrefixLength(_ text: String, prefixes: [String]) -> Int {
        let maxLength = min(text.count, prefixes.map(\.count).max() ?? 0)
        guard maxLength > 0 else {
            return 0
        }
        for length in stride(from: maxLength, through: 1, by: -1) {
            let suffix = String(text.suffix(length))
            if prefixes.contains(where: { $0.hasPrefix(suffix) }) {
                return length
            }
        }
        return 0
    }
}

private extension String {
    func strippingLeadingThoughtMarker() -> String {
        var value = self
        if value.hasPrefix("thought") {
            value.removeFirst("thought".count)
        }
        return value
    }
}
