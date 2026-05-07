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

    public init(
        model: String,
        promptTokenCount: Int,
        maxCompletionTokens: Int,
        eosTokenIDs: [Int] = [],
        stopSequences: [String] = [],
        samplingPlan: GenerationSamplingPlan,
        tokenizer: SimpleTokenizer,
        skipSpecialTokens: Bool = true,
        newlineTokenID: Int? = nil
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
                generator: &samplerGenerator
            )
            recent.append(sampled.tokenID)
            let textStep = textDecoder.append(sampled.tokenID)
            let decodeStep = decodeLoop.append(
                GenerationDecodeToken(
                    tokenID: sampled.tokenID,
                    text: textStep.textDelta,
                    probability: sampled.probability,
                    logProbability: sampled.logProbability,
                    rank: sampled.rank
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
}
