import Foundation

public struct StopSequenceMatchResult: Codable, Equatable, Sendable {
    public let text: String
    public let finishReason: String
    public let matchedStopSequence: String?
    public let matchedRange: [Int]?

    public init(
        text: String,
        finishReason: String,
        matchedStopSequence: String? = nil,
        matchedRange: [Int]? = nil
    ) {
        self.text = text
        self.finishReason = finishReason
        self.matchedStopSequence = matchedStopSequence
        self.matchedRange = matchedRange
    }
}

public struct StopSequenceMatcher: Sendable {
    public let stopSequences: [String]

    public init(stopSequences: [String]) {
        self.stopSequences = stopSequences.filter { !$0.isEmpty }
    }

    public var maxStopSequenceCharacterCount: Int {
        stopSequences.map(\.count).max() ?? 0
    }

    public func truncate(
        _ text: String,
        defaultFinishReason: String = "length"
    ) -> StopSequenceMatchResult {
        guard !stopSequences.isEmpty else {
            return StopSequenceMatchResult(text: text, finishReason: defaultFinishReason)
        }

        var best: (range: Range<String.Index>, sequence: String, order: Int)?
        for (order, sequence) in stopSequences.enumerated() {
            guard let range = text.range(of: sequence) else {
                continue
            }
            if let current = best {
                let currentStart = text.distance(from: text.startIndex, to: current.range.lowerBound)
                let candidateStart = text.distance(from: text.startIndex, to: range.lowerBound)
                if candidateStart < currentStart || (candidateStart == currentStart && order < current.order) {
                    best = (range, sequence, order)
                }
            } else {
                best = (range, sequence, order)
            }
        }

        guard let match = best else {
            return StopSequenceMatchResult(text: text, finishReason: defaultFinishReason)
        }

        let start = text.distance(from: text.startIndex, to: match.range.lowerBound)
        let end = text.distance(from: text.startIndex, to: match.range.upperBound)
        return StopSequenceMatchResult(
            text: String(text[..<match.range.lowerBound]),
            finishReason: "stop",
            matchedStopSequence: match.sequence,
            matchedRange: [start, end]
        )
    }
}

public struct StopSequenceStreamFilter: Sendable {
    private let matcher: StopSequenceMatcher
    private var pendingText = ""
    private var didFinish = false

    public init(stopSequences: [String]) {
        self.matcher = StopSequenceMatcher(stopSequences: stopSequences)
    }

    public mutating func append(_ chunk: GenerationChunk) -> [GenerationChunk] {
        guard !didFinish else {
            return []
        }
        guard chunk.toolCalls.isEmpty else {
            return [chunk]
        }

        pendingText += chunk.text
        let result = matcher.truncate(pendingText)
        if result.matchedStopSequence != nil {
            didFinish = true
            pendingText.removeAll()
            return [GenerationChunk(text: result.text, tokenID: nil, isFinished: true, finishReason: "stop")]
        }

        if chunk.isFinished {
            didFinish = true
            let text = pendingText
            pendingText.removeAll()
            return [
                GenerationChunk(
                    text: text,
                    tokenID: chunk.tokenID,
                    logprob: chunk.logprob,
                    isFinished: true,
                    finishReason: chunk.finishReason,
                    promptTokenCount: chunk.promptTokenCount,
                    completionTokenCount: chunk.completionTokenCount,
                    toolCalls: chunk.toolCalls
                )
            ]
        }

        let withheldCharacters = max(0, matcher.maxStopSequenceCharacterCount - 1)
        guard pendingText.count > withheldCharacters else {
            return []
        }

        let emitCount = pendingText.count - withheldCharacters
        let split = pendingText.index(pendingText.startIndex, offsetBy: emitCount)
        let emitted = String(pendingText[..<split])
        pendingText = String(pendingText[split...])
        return [GenerationChunk(text: emitted, tokenID: chunk.tokenID, logprob: chunk.logprob, isFinished: false)]
    }

    public mutating func finish() -> GenerationChunk? {
        guard !didFinish else {
            return nil
        }
        didFinish = true
        let text = pendingText
        pendingText.removeAll()
        return GenerationChunk(text: text, tokenID: nil, isFinished: true, finishReason: "stop")
    }
}

public extension CompletedGeneration {
    func applyingStopSequences(_ stopSequences: [String]) -> CompletedGeneration {
        let result = StopSequenceMatcher(stopSequences: stopSequences)
            .truncate(text, defaultFinishReason: finishReason)
        return CompletedGeneration(
            model: model,
            text: result.text,
            finishReason: result.finishReason,
            usage: usage,
            toolCalls: toolCalls,
            logprobs: logprobs
        )
    }
}
