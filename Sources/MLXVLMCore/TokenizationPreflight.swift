import Foundation

public struct TokenizationPreflightFragment: Codable, Equatable, Sendable {
    public let text: String
    public let tokenID: Int?
    public let source: String?

    public init(text: String, tokenID: Int?, source: String?) {
        self.text = text
        self.tokenID = tokenID
        self.source = source
    }

    public var isKnownToken: Bool {
        tokenID != nil
    }
}

public struct TokenizationPreflightPlan: Codable, Equatable, Sendable {
    public let tokenizerModelType: String?
    public let fragments: [TokenizationPreflightFragment]
    public let tokenIDs: [Int]
    public let knownTokenCount: Int
    public let unknownFragmentCount: Int
    public let requiresTokenizerImplementation: Bool

    public init(tokenizerModelType: String?, fragments: [TokenizationPreflightFragment]) {
        self.tokenizerModelType = tokenizerModelType
        self.fragments = fragments
        self.tokenIDs = fragments.compactMap(\.tokenID)
        self.knownTokenCount = fragments.filter(\.isKnownToken).count
        self.unknownFragmentCount = fragments.filter { !$0.isKnownToken }.count
        self.requiresTokenizerImplementation = unknownFragmentCount > 0
    }

    public init(tokenizerModelType: String?, tokenizerResult: SimpleTokenizerResult) {
        self.tokenizerModelType = tokenizerModelType
        self.fragments = zip(tokenizerResult.tokens, tokenizerResult.tokenIDs).map { token, id in
            TokenizationPreflightFragment(
                text: token,
                tokenID: id,
                source: tokenizerResult.requiredBackend
            )
        }
        self.tokenIDs = tokenizerResult.tokenIDs
        self.knownTokenCount = max(0, tokenizerResult.tokenIDs.count - tokenizerResult.unknownTokens.count)
        self.unknownFragmentCount = tokenizerResult.unknownTokens.count
        self.requiresTokenizerImplementation = !tokenizerResult.supported || !tokenizerResult.unknownTokens.isEmpty
    }
}

public struct TokenizationPreflightPlanner {
    public let catalog: TokenizerCatalog

    public init(catalog: TokenizerCatalog) {
        self.catalog = catalog
    }

    public func plan(prompt: String) -> TokenizationPreflightPlan {
        let candidates = catalog.tokens
            .filter { !$0.content.isEmpty }
            .sorted { lhs, rhs in
                if lhs.content.count == rhs.content.count {
                    return lhs.id < rhs.id
                }
                return lhs.content.count > rhs.content.count
            }

        var fragments: [TokenizationPreflightFragment] = []
        var index = prompt.startIndex
        while index < prompt.endIndex {
            if let match = candidates.first(where: { prompt[index...].hasPrefix($0.content) }) {
                fragments.append(
                    TokenizationPreflightFragment(
                        text: match.content,
                        tokenID: match.id,
                        source: match.source
                    )
                )
                index = prompt.index(index, offsetBy: match.content.count)
            } else {
                let start = index
                index = prompt.index(after: index)
                while index < prompt.endIndex,
                      candidates.first(where: { prompt[index...].hasPrefix($0.content) }) == nil
                {
                    index = prompt.index(after: index)
                }
                fragments.append(
                    TokenizationPreflightFragment(
                        text: String(prompt[start..<index]),
                        tokenID: nil,
                        source: nil
                    )
                )
            }
        }

        return TokenizationPreflightPlan(
            tokenizerModelType: catalog.modelType,
            fragments: fragments
        )
    }
}
