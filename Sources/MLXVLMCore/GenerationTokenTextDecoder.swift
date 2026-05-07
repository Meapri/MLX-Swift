import Foundation

public struct GenerationTokenTextDecodeStep: Codable, Equatable, Sendable {
    public let tokenID: Int
    public let textDelta: String
    public let decodedText: String
    public let tokenIDs: [Int]
    public let skippedSpecialToken: Bool
    public let unknownTokenIDs: [Int]
    public let error: String?

    public init(
        tokenID: Int,
        textDelta: String,
        decodedText: String,
        tokenIDs: [Int],
        skippedSpecialToken: Bool,
        unknownTokenIDs: [Int],
        error: String?
    ) {
        self.tokenID = tokenID
        self.textDelta = textDelta
        self.decodedText = decodedText
        self.tokenIDs = tokenIDs
        self.skippedSpecialToken = skippedSpecialToken
        self.unknownTokenIDs = unknownTokenIDs
        self.error = error
    }

    public var decodeToken: GenerationDecodeToken {
        GenerationDecodeToken(tokenID: tokenID, text: textDelta)
    }
}

public struct GenerationTokenTextDecoder: Sendable {
    public let tokenizer: SimpleTokenizer
    public let skipSpecialTokens: Bool

    private var tokenIDs: [Int] = []
    private var decodedText = ""

    public init(
        tokenizer: SimpleTokenizer,
        skipSpecialTokens: Bool = true
    ) {
        self.tokenizer = tokenizer
        self.skipSpecialTokens = skipSpecialTokens
    }

    public mutating func append(_ tokenID: Int) -> GenerationTokenTextDecodeStep {
        tokenIDs.append(tokenID)
        let result = tokenizer.detokenize(tokenIDs, skipSpecialTokens: skipSpecialTokens)
        let nextText = result.text
        let delta: String
        if nextText.hasPrefix(decodedText) {
            delta = String(nextText.dropFirst(decodedText.count))
        } else {
            delta = nextText
        }
        decodedText = nextText

        return GenerationTokenTextDecodeStep(
            tokenID: tokenID,
            textDelta: delta,
            decodedText: decodedText,
            tokenIDs: tokenIDs,
            skippedSpecialToken: skipSpecialTokens && tokenizer.catalog.isSpecialToken(id: tokenID),
            unknownTokenIDs: result.unknownTokenIDs,
            error: result.error
        )
    }

    public mutating func append(_ sampledToken: SampledToken) -> GenerationDecodeToken {
        let step = append(sampledToken.tokenID)
        return GenerationDecodeToken(
            tokenID: sampledToken.tokenID,
            text: step.textDelta,
            probability: sampledToken.probability,
            logProbability: sampledToken.logProbability,
            rank: sampledToken.rank
        )
    }
}
