import Foundation

public struct ToolCallTextParseResult: Codable, Equatable, Sendable {
    public let text: String
    public let toolCalls: [GenerationToolCall]

    public init(text: String, toolCalls: [GenerationToolCall]) {
        self.text = text
        self.toolCalls = toolCalls
    }
}

public struct ToolCallOutputParser: Sendable {
    public init() {}

    public static func usesGemma4ToolCalls(descriptor: ModelDescriptor) -> Bool {
        let types = [
            descriptor.rawModelType.lowercased(),
            descriptor.canonicalModelType.lowercased(),
        ]
        if types.contains(where: { $0 == "gemma4" || $0.hasPrefix("gemma4_") }) {
            return true
        }
        let template = descriptor.tokenizerMetadata.chatTemplate ?? ""
        return template.contains("<|tool_call>") || template.contains("gemma4-tool-call")
    }

    public func parseGemma4ToolCalls(in text: String) -> ToolCallTextParseResult {
        let parsed = parseGemmaCalls(in: text)
        guard !parsed.toolCalls.isEmpty else {
            return ToolCallTextParseResult(text: text, toolCalls: [])
        }
        let cleaned = removeRanges(parsed.ranges, from: text)
            .replacingOccurrences(of: "<|tool_call>", with: "")
            .replacingOccurrences(of: "<tool_call|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ToolCallTextParseResult(text: cleaned, toolCalls: parsed.toolCalls)
    }

    private func parseGemmaCalls(in text: String) -> (toolCalls: [GenerationToolCall], ranges: [Range<String.Index>]) {
        var toolCalls: [GenerationToolCall] = []
        var ranges: [Range<String.Index>] = []
        var searchStart = text.startIndex

        while let callRange = text.range(of: "call:", range: searchStart..<text.endIndex) {
            var cursor = callRange.upperBound
            let nameStart = cursor
            while cursor < text.endIndex, isFunctionNameCharacter(text[cursor]) {
                cursor = text.index(after: cursor)
            }
            let name = String(text[nameStart..<cursor])
            guard !name.isEmpty, cursor < text.endIndex, text[cursor] == "{" else {
                searchStart = callRange.upperBound
                continue
            }
            guard let argsEnd = balancedBraceEnd(in: text, startingAt: cursor) else {
                break
            }

            let argsText = String(text[cursor..<argsEnd])
            if let arguments = gemma4Arguments(from: argsText) {
                toolCalls.append(
                    GenerationToolCall(
                        id: "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased())",
                        function: GenerationToolCallFunction(name: name, arguments: arguments)
                    )
                )
                ranges.append(expandedToolCallRange(callRange.lowerBound..<argsEnd, in: text))
            }
            searchStart = argsEnd
        }

        return (toolCalls, ranges)
    }

    private func balancedBraceEnd(in text: String, startingAt start: String.Index) -> String.Index? {
        var cursor = start
        var depth = 0
        var inGemmaString = false
        let stringMarker = #"<|"|>"#

        while cursor < text.endIndex {
            if text[cursor...].hasPrefix(stringMarker) {
                inGemmaString.toggle()
                cursor = text.index(cursor, offsetBy: stringMarker.count)
                continue
            }
            if !inGemmaString {
                if text[cursor] == "{" {
                    depth += 1
                } else if text[cursor] == "}" {
                    depth -= 1
                    if depth == 0 {
                        return text.index(after: cursor)
                    }
                }
            }
            cursor = text.index(after: cursor)
        }
        return nil
    }

    private func gemma4Arguments(from text: String) -> [String: JSONValue]? {
        var parser = Gemma4ArgumentParser(text)
        return parser.parseRootObject()
    }

    private func expandedToolCallRange(_ range: Range<String.Index>, in text: String) -> Range<String.Index> {
        let startTag = "<|tool_call>"
        let endTag = "<tool_call|>"
        var lower = range.lowerBound
        var upper = range.upperBound
        if text[..<lower].hasSuffix(startTag) {
            lower = text.index(lower, offsetBy: -startTag.count)
        }
        if text[upper...].hasPrefix(endTag) {
            upper = text.index(upper, offsetBy: endTag.count)
        }
        return lower..<upper
    }

    private func removeRanges(_ ranges: [Range<String.Index>], from text: String) -> String {
        var output = text
        for range in ranges.reversed() {
            output.removeSubrange(range)
        }
        return output
    }

    private func isFunctionNameCharacter(_ character: Character) -> Bool {
        character == "_" || character == "-" || character.isLetter || character.isNumber
    }
}

private struct Gemma4ArgumentParser {
    private let text: String
    private var cursor: String.Index
    private let marker = #"<|"|>"#

    init(_ text: String) {
        self.text = text
        self.cursor = text.startIndex
    }

    mutating func parseRootObject() -> [String: JSONValue]? {
        skipWhitespace()
        guard consume("{") else {
            return nil
        }
        return parseObjectBody()
    }

    private mutating func parseObjectBody() -> [String: JSONValue]? {
        var object: [String: JSONValue] = [:]
        while cursor < text.endIndex {
            skipWhitespace()
            if consume("}") {
                return object
            }
            guard let key = parseKey(), consume(":") else {
                return nil
            }
            object[key] = parseValue()
            skipWhitespace()
            _ = consume(",")
        }
        return object
    }

    private mutating func parseArrayBody() -> [JSONValue] {
        var array: [JSONValue] = []
        while cursor < text.endIndex {
            skipWhitespace()
            if consume("]") {
                return array
            }
            array.append(parseValue())
            skipWhitespace()
            _ = consume(",")
        }
        return array
    }

    private mutating func parseKey() -> String? {
        skipWhitespace()
        if hasPrefix(marker) {
            return parseGemmaString()
        }
        let start = cursor
        while cursor < text.endIndex, text[cursor] != ":" {
            cursor = text.index(after: cursor)
        }
        guard cursor > start else {
            return nil
        }
        return String(text[start..<cursor]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private mutating func parseValue() -> JSONValue {
        skipWhitespace()
        if hasPrefix(marker), let value = parseGemmaString() {
            return .string(value)
        }
        if consume("{") {
            return .object(parseObjectBody() ?? [:])
        }
        if consume("[") {
            return .array(parseArrayBody())
        }
        let literal = parseBareLiteral()
        switch literal {
        case "true":
            return .bool(true)
        case "false":
            return .bool(false)
        case "null":
            return .null
        default:
            if let number = Double(literal), literal.rangeOfCharacter(from: .letters) == nil {
                return .number(number)
            }
            return .string(literal)
        }
    }

    private mutating func parseGemmaString() -> String? {
        guard consume(marker) else {
            return nil
        }
        let start = cursor
        guard let end = text.range(of: marker, range: cursor..<text.endIndex) else {
            cursor = text.endIndex
            return String(text[start..<text.endIndex])
        }
        cursor = end.upperBound
        return String(text[start..<end.lowerBound])
    }

    private mutating func parseBareLiteral() -> String {
        let start = cursor
        var depth = 0
        while cursor < text.endIndex {
            if hasPrefix(marker) {
                guard let end = text.range(
                    of: marker,
                    range: text.index(cursor, offsetBy: marker.count)..<text.endIndex
                ) else {
                    cursor = text.endIndex
                    break
                }
                cursor = end.upperBound
                continue
            }

            let character = text[cursor]
            if character == "{" || character == "[" {
                depth += 1
            } else if character == "}" || character == "]" {
                if depth == 0 {
                    break
                }
                depth -= 1
            } else if character == "," && depth == 0 {
                break
            }
            cursor = text.index(after: cursor)
        }
        return String(text[start..<cursor]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private mutating func skipWhitespace() {
        while cursor < text.endIndex, text[cursor].isWhitespace {
            cursor = text.index(after: cursor)
        }
    }

    private mutating func consume(_ token: String) -> Bool {
        guard hasPrefix(token) else {
            return false
        }
        cursor = text.index(cursor, offsetBy: token.count)
        return true
    }

    private func hasPrefix(_ token: String) -> Bool {
        text[cursor...].hasPrefix(token)
    }
}
