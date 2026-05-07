import Foundation

public struct GenerationLogitsDecodeStep: Codable, Equatable, Sendable {
    public let logitsShape: [Int]
    public let sampledToken: SampledToken
    public let textDecode: GenerationTokenTextDecodeStep
    public let decodeStep: GenerationDecodeStep

    public init(
        logitsShape: [Int],
        sampledToken: SampledToken,
        textDecode: GenerationTokenTextDecodeStep,
        decodeStep: GenerationDecodeStep
    ) {
        self.logitsShape = logitsShape
        self.sampledToken = sampledToken
        self.textDecode = textDecode
        self.decodeStep = decodeStep
    }
}

public struct GenerationLogitsDecodeReport: Codable, Equatable, Sendable {
    public let model: String
    public let promptTokenCount: Int
    public let processedLogitRows: Int
    public let steps: [GenerationLogitsDecodeStep]
    public let decodeLoop: GenerationDecodeLoopReport
    public let completed: CompletedGeneration

    public init(
        model: String,
        promptTokenCount: Int,
        processedLogitRows: Int,
        steps: [GenerationLogitsDecodeStep],
        decodeLoop: GenerationDecodeLoopReport
    ) {
        self.model = model
        self.promptTokenCount = promptTokenCount
        self.processedLogitRows = processedLogitRows
        self.steps = steps
        self.decodeLoop = decodeLoop
        self.completed = decodeLoop.completed
    }
}

public struct GenerationLogitsDecodeExecutor: Sendable {
    public let model: String
    public let promptTokenCount: Int
    public let maxCompletionTokens: Int
    public let eosTokenIDs: [Int]
    public let stopSequences: [String]
    public let samplingPlan: GenerationSamplingPlan
    public let tokenizer: SimpleTokenizer
    public let skipSpecialTokens: Bool
    public let newlineTokenID: Int?
    public let logitBias: [Int: Double]
    public let topLogprobs: Int

    public init(
        model: String,
        promptTokenCount: Int,
        maxCompletionTokens: Int,
        eosTokenIDs: [Int] = [],
        stopSequences: [String] = [],
        samplingPlan: GenerationSamplingPlan,
        tokenizer: SimpleTokenizer,
        skipSpecialTokens: Bool = true,
        newlineTokenID: Int? = nil,
        logitBias: [Int: Double] = [:],
        topLogprobs: Int = 0
    ) {
        self.model = model
        self.promptTokenCount = promptTokenCount
        self.maxCompletionTokens = maxCompletionTokens
        self.eosTokenIDs = eosTokenIDs
        self.stopSequences = stopSequences
        self.samplingPlan = samplingPlan
        self.tokenizer = tokenizer
        self.skipSpecialTokens = skipSpecialTokens
        self.newlineTokenID = newlineTokenID
        self.logitBias = logitBias
        self.topLogprobs = max(0, topLogprobs)
    }

    public func run(
        logitsRows: [[Double]],
        recentTokenIDs: [Int] = []
    ) throws -> GenerationLogitsDecodeReport {
        var samplerGenerator = SeededLogitsRandomNumberGenerator(seed: samplingPlan.seed)
        let sampler = GenerationLogitsSampler(plan: samplingPlan)
        var textDecoder = GenerationTokenTextDecoder(
            tokenizer: tokenizer,
            skipSpecialTokens: skipSpecialTokens
        )
        var decodeLoop = GenerationDecodeLoop(
            model: model,
            promptTokenCount: promptTokenCount,
            maxCompletionTokens: maxCompletionTokens,
            eosTokenIDs: eosTokenIDs,
            stopSequences: stopSequences
        )
        var recent = recentTokenIDs
        var steps: [GenerationLogitsDecodeStep] = []

        for logits in logitsRows {
            let sampled = try sampler.sample(
                logits: logits,
                recentTokenIDs: recent,
                newlineTokenID: newlineTokenID,
                logitBias: logitBias,
                generator: &samplerGenerator
            )
            recent.append(sampled.tokenID)
            let textStep = textDecoder.append(sampled.tokenID)
            let logprob = try logprobPayload(
                logits: logits,
                sampled: sampled,
                emittedText: textStep.textDelta,
                recentTokenIDs: Array(recent.dropLast()),
                logitBias: logitBias
            )
            let decodeStep = decodeLoop.append(
                GenerationDecodeToken(
                    tokenID: sampled.tokenID,
                    text: textStep.textDelta,
                    probability: sampled.probability,
                    logProbability: sampled.logProbability,
                    rank: sampled.rank,
                    logprob: logprob
                )
            )
            steps.append(
                GenerationLogitsDecodeStep(
                    logitsShape: [logits.count],
                    sampledToken: sampled,
                    textDecode: textStep,
                    decodeStep: decodeStep
                )
            )
            if !decodeStep.shouldContinue {
                break
            }
        }
        if !decodeLoop.snapshot.isFinished {
            _ = decodeLoop.finish(defaultFinishReason: "stop")
        }

        return GenerationLogitsDecodeReport(
            model: model,
            promptTokenCount: promptTokenCount,
            processedLogitRows: steps.count,
            steps: steps,
            decodeLoop: decodeLoop.report
        )
    }

    private func logprobPayload(
        logits: [Double],
        sampled: SampledToken,
        emittedText: String,
        recentTokenIDs: [Int],
        logitBias: [Int: Double]
    ) throws -> GenerationTokenLogprob {
        let ranked = try GenerationLogitsSampler(plan: samplingPlan).rankedTokenProbabilities(
            logits: logits,
            recentTokenIDs: recentTokenIDs,
            newlineTokenID: newlineTokenID,
            logitBias: logitBias
        )
        let sampledFromFullDistribution = ranked.first { $0.tokenID == sampled.tokenID } ?? sampled
        let top = ranked.prefix(topLogprobs).map { candidate in
            GenerationTopLogprob(
                token: tokenText(for: candidate.tokenID),
                logprob: candidate.logProbability,
                bytes: bytes(for: candidate.tokenID)
            )
        }
        return GenerationTokenLogprob(
            token: emittedText,
            logprob: sampledFromFullDistribution.logProbability,
            bytes: Array(emittedText.utf8).map(Int.init),
            topLogprobs: top
        )
    }

    private func tokenText(for tokenID: Int) -> String {
        tokenizer.detokenize([tokenID], skipSpecialTokens: skipSpecialTokens).text
    }

    private func bytes(for tokenID: Int) -> [Int]? {
        Array(tokenText(for: tokenID).utf8).map(Int.init)
    }
}
