import Foundation

public struct GenerationDecodeToken: Codable, Equatable, Sendable {
    public let tokenID: Int
    public let text: String
    public let probability: Double?
    public let logProbability: Double?
    public let rank: Int?

    public init(
        tokenID: Int,
        text: String,
        probability: Double? = nil,
        logProbability: Double? = nil,
        rank: Int? = nil
    ) {
        self.tokenID = tokenID
        self.text = text
        self.probability = probability
        self.logProbability = logProbability
        self.rank = rank
    }

    public init(sampledToken: SampledToken, text: String) {
        self.init(
            tokenID: sampledToken.tokenID,
            text: text,
            probability: sampledToken.probability,
            logProbability: sampledToken.logProbability,
            rank: sampledToken.rank
        )
    }
}

public struct GenerationDecodeStep: Codable, Equatable, Sendable {
    public let token: GenerationDecodeToken
    public let emittedChunks: [GenerationChunk]
    public let snapshot: GenerationOutputSnapshot
    public let shouldContinue: Bool

    public init(
        token: GenerationDecodeToken,
        emittedChunks: [GenerationChunk],
        snapshot: GenerationOutputSnapshot,
        shouldContinue: Bool
    ) {
        self.token = token
        self.emittedChunks = emittedChunks
        self.snapshot = snapshot
        self.shouldContinue = shouldContinue
    }
}

public struct GenerationDecodeLoopReport: Codable, Equatable, Sendable {
    public let model: String
    public let maxCompletionTokens: Int
    public let eosTokenIDs: [Int]
    public let stopSequences: [String]
    public let steps: [GenerationDecodeStep]
    public let completed: CompletedGeneration
    public let finishReason: String

    public init(
        model: String,
        maxCompletionTokens: Int,
        eosTokenIDs: [Int],
        stopSequences: [String],
        steps: [GenerationDecodeStep],
        completed: CompletedGeneration
    ) {
        self.model = model
        self.maxCompletionTokens = maxCompletionTokens
        self.eosTokenIDs = eosTokenIDs
        self.stopSequences = stopSequences
        self.steps = steps
        self.completed = completed
        self.finishReason = completed.finishReason
    }
}

public struct GenerationDecodeLoop: Sendable {
    public let model: String
    public let maxCompletionTokens: Int
    public let eosTokenIDs: Set<Int>
    public let stopSequences: [String]

    private var assembler: GenerationOutputAssembler
    private var steps: [GenerationDecodeStep] = []

    public init(
        model: String,
        promptTokenCount: Int,
        maxCompletionTokens: Int,
        eosTokenIDs: [Int] = [],
        stopSequences: [String] = []
    ) {
        self.model = model
        self.maxCompletionTokens = maxCompletionTokens
        self.eosTokenIDs = Set(eosTokenIDs)
        self.stopSequences = stopSequences
        self.assembler = GenerationOutputAssembler(
            model: model,
            promptTokenCount: promptTokenCount,
            stopSequences: stopSequences
        )
    }

    public var snapshot: GenerationOutputSnapshot {
        assembler.snapshot
    }

    public var completedGeneration: CompletedGeneration {
        assembler.completedGeneration
    }

    public var report: GenerationDecodeLoopReport {
        GenerationDecodeLoopReport(
            model: model,
            maxCompletionTokens: maxCompletionTokens,
            eosTokenIDs: Array(eosTokenIDs).sorted(),
            stopSequences: stopSequences,
            steps: steps,
            completed: completedGeneration
        )
    }

    public mutating func append(_ token: GenerationDecodeToken) -> GenerationDecodeStep {
        if assembler.snapshot.isFinished {
            return record(token: token, emitted: [], shouldContinue: false)
        }

        if eosTokenIDs.contains(token.tokenID) {
            let emitted = assembler.append(
                GenerationChunk(text: "", tokenID: token.tokenID, isFinished: true, finishReason: "stop")
            )
            return record(token: token, emitted: emitted, shouldContinue: false)
        }

        var emitted = assembler.append(GenerationChunk(text: token.text, tokenID: token.tokenID))
        if !assembler.snapshot.isFinished,
           assembler.snapshot.usage.completionTokens >= maxCompletionTokens
        {
            if let final = assembler.finish(defaultFinishReason: "length") {
                emitted.append(final)
            }
        }
        return record(token: token, emitted: emitted, shouldContinue: !assembler.snapshot.isFinished)
    }

    public mutating func finish(defaultFinishReason: String = "stop") -> GenerationChunk? {
        assembler.finish(defaultFinishReason: defaultFinishReason)
    }

    public mutating func run(_ tokens: [GenerationDecodeToken]) -> GenerationDecodeLoopReport {
        for token in tokens {
            let step = append(token)
            if !step.shouldContinue {
                break
            }
        }
        if !assembler.snapshot.isFinished {
            _ = finish(defaultFinishReason: "stop")
        }
        return report
    }

    private mutating func record(
        token: GenerationDecodeToken,
        emitted: [GenerationChunk],
        shouldContinue: Bool
    ) -> GenerationDecodeStep {
        let step = GenerationDecodeStep(
            token: token,
            emittedChunks: emitted,
            snapshot: assembler.snapshot,
            shouldContinue: shouldContinue
        )
        steps.append(step)
        return step
    }
}
