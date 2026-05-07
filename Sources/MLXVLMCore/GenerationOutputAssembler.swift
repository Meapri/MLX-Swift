import Foundation

public struct GenerationOutputSnapshot: Codable, Equatable, Sendable {
    public let text: String
    public let emittedChunks: [GenerationChunk]
    public let usage: GenerationUsage
    public let isFinished: Bool
    public let finishReason: String?

    public init(
        text: String,
        emittedChunks: [GenerationChunk],
        usage: GenerationUsage,
        isFinished: Bool,
        finishReason: String?
    ) {
        self.text = text
        self.emittedChunks = emittedChunks
        self.usage = usage
        self.isFinished = isFinished
        self.finishReason = finishReason
    }
}

public struct GenerationOutputAssembler: Sendable {
    public let model: String
    public let promptTokenCount: Int
    public let stopSequences: [String]

    private var stopFilter: StopSequenceStreamFilter?
    private var text = ""
    private var emittedChunks: [GenerationChunk] = []
    private var completionTokenCount = 0
    private var promptTokenCountOverride: Int?
    private var completionTokenCountOverride: Int?
    private var toolCalls: [GenerationToolCall] = []
    private var isFinished = false
    private var finishReason: String?

    public init(
        model: String,
        promptTokenCount: Int,
        stopSequences: [String] = []
    ) {
        self.model = model
        self.promptTokenCount = promptTokenCount
        self.stopSequences = stopSequences
        self.stopFilter = stopSequences.isEmpty ? nil : StopSequenceStreamFilter(stopSequences: stopSequences)
    }

    public mutating func append(_ chunk: GenerationChunk) -> [GenerationChunk] {
        guard !isFinished else {
            return []
        }
        if chunk.tokenID != nil {
            completionTokenCount += 1
        }

        let filtered: [GenerationChunk]
        if var filter = stopFilter {
            filtered = filter.append(chunk)
            stopFilter = filter
        } else {
            filtered = [chunk]
        }

        return record(filtered, sourceFinished: chunk.isFinished, sourceFinishReason: chunk.finishReason)
    }

    public mutating func finish(defaultFinishReason: String = "stop") -> GenerationChunk? {
        guard !isFinished else {
            return nil
        }

        let chunk: GenerationChunk?
        if var filter = stopFilter {
            chunk = filter.finish()
            stopFilter = filter
        } else {
            chunk = GenerationChunk(text: "", tokenID: nil, isFinished: true, finishReason: defaultFinishReason)
        }

        guard let chunk else {
            isFinished = true
            finishReason = defaultFinishReason
            return nil
        }
        return record([chunk], sourceFinished: true, sourceFinishReason: defaultFinishReason).last
    }

    public var snapshot: GenerationOutputSnapshot {
        GenerationOutputSnapshot(
            text: text,
            emittedChunks: emittedChunks,
            usage: GenerationUsage(
                promptTokens: promptTokenCountOverride ?? promptTokenCount,
                completionTokens: completionTokenCountOverride ?? completionTokenCount
            ),
            isFinished: isFinished,
            finishReason: finishReason
        )
    }

    public var completedGeneration: CompletedGeneration {
        CompletedGeneration(
            model: model,
            text: text,
            finishReason: finishReason ?? (isFinished ? "stop" : "length"),
            usage: GenerationUsage(
                promptTokens: promptTokenCountOverride ?? promptTokenCount,
                completionTokens: completionTokenCountOverride ?? completionTokenCount
            ),
            toolCalls: toolCalls
        )
    }

    private mutating func record(
        _ chunks: [GenerationChunk],
        sourceFinished: Bool,
        sourceFinishReason: String?
    ) -> [GenerationChunk] {
        var output: [GenerationChunk] = []
        for chunk in chunks {
            text += chunk.text
            if let promptTokenCount = chunk.promptTokenCount {
                promptTokenCountOverride = promptTokenCount
            }
            if let completionTokenCount = chunk.completionTokenCount {
                completionTokenCountOverride = completionTokenCount
            }
            toolCalls.append(contentsOf: chunk.toolCalls)
            if chunk.isFinished {
                isFinished = true
                finishReason = chunk.finishReason ?? sourceFinishReason ?? (sourceFinished ? "stop" : "stop")
            }
            emittedChunks.append(chunk)
            output.append(chunk)
        }
        return output
    }
}
