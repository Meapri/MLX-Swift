import Foundation

public struct TokenizerCatalogToken: Codable, Equatable, Sendable {
    public let content: String
    public let id: Int
    public let source: String
    public let special: Bool

    public init(content: String, id: Int, source: String, special: Bool) {
        self.content = content
        self.id = id
        self.source = source
        self.special = special
    }
}

public struct TokenizerCatalogMerge: Codable, Equatable, Sendable {
    public let left: String
    public let right: String
    public let rank: Int

    public init(left: String, right: String, rank: Int) {
        self.left = left
        self.right = right
        self.rank = rank
    }
}

public struct TokenizerCatalog: Codable, Equatable, Sendable {
    public let modelType: String?
    public let unknownToken: String?
    public let unknownTokenID: Int?
    public let tokens: [TokenizerCatalogToken]
    public let merges: [TokenizerCatalogMerge]
    public let duplicateContents: [String]
    public let duplicateIDs: [Int]
    public let error: String?

    public init(
        modelType: String?,
        unknownToken: String? = nil,
        unknownTokenID: Int? = nil,
        tokens: [TokenizerCatalogToken],
        merges: [TokenizerCatalogMerge] = [],
        duplicateContents: [String],
        duplicateIDs: [Int],
        error: String?
    ) {
        self.modelType = modelType
        self.unknownToken = unknownToken
        self.unknownTokenID = unknownTokenID
        self.tokens = tokens.sorted { lhs, rhs in
            if lhs.id == rhs.id {
                return lhs.content < rhs.content
            }
            return lhs.id < rhs.id
        }
        self.merges = merges.sorted { lhs, rhs in
            if lhs.rank == rhs.rank {
                if lhs.left == rhs.left {
                    return lhs.right < rhs.right
                }
                return lhs.left < rhs.left
            }
            return lhs.rank < rhs.rank
        }
        self.duplicateContents = duplicateContents.sorted()
        self.duplicateIDs = duplicateIDs.sorted()
        self.error = error
    }

    public var tokenCount: Int {
        tokens.count
    }

    public var specialTokenCount: Int {
        tokens.filter(\.special).count
    }

    public func id(for content: String) -> Int? {
        tokens.first { $0.content == content }?.id
    }

    public func token(for id: Int) -> String? {
        tokens.first { $0.id == id }?.content
    }

    public func isSpecialToken(content: String) -> Bool {
        tokens.contains { $0.content == content && $0.special }
    }

    public func isSpecialToken(id: Int) -> Bool {
        tokens.contains { $0.id == id && $0.special }
    }
}

public struct TokenizerCatalogBuilder {
    public let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    public func catalog(for descriptor: ModelDescriptor) -> TokenizerCatalog? {
        catalog(modelDirectory: URL(fileURLWithPath: descriptor.path, isDirectory: true))
    }

    public func catalog(modelDirectory: URL) -> TokenizerCatalog? {
        let tokenizerURL = modelDirectory.appendingPathComponent("tokenizer.json")
        if fileManager.fileExists(atPath: tokenizerURL.path) {
            return tokenizerJSONCatalog(tokenizerURL: tokenizerURL)
        }
        return sidecarBPECatalog(modelDirectory: modelDirectory)
    }

    private func tokenizerJSONCatalog(tokenizerURL: URL) -> TokenizerCatalog {
        do {
            let json = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: tokenizerURL))
            guard let object = json.objectValue else {
                throw ModelStoreError.invalidConfig("tokenizer.json must contain a JSON object")
            }
            let model = object["model"]?.objectValue
            let modelType = model?["type"]?.stringValue
            let unknownToken = model?["unk_token"]?.stringValue
            var tokens: [TokenizerCatalogToken] = []
            var merges: [TokenizerCatalogMerge] = []

            for (content, idValue) in model?["vocab"]?.objectValue ?? [:] {
                guard let id = idValue.intValue else {
                    continue
                }
                tokens.append(
                    TokenizerCatalogToken(
                        content: content,
                        id: id,
                        source: "model.vocab",
                        special: false
                    )
                )
            }

            for tokenValue in object["added_tokens"]?.arrayValue ?? [] {
                guard let tokenObject = tokenValue.objectValue,
                      let id = tokenObject["id"]?.intValue,
                      let content = tokenObject["content"]?.stringValue
                else {
                    continue
                }
                tokens.append(
                    TokenizerCatalogToken(
                        content: content,
                        id: id,
                        source: "added_tokens",
                        special: tokenObject["special"]?.boolValue ?? true
                    )
                )
            }

            for (rank, mergeValue) in (model?["merges"]?.arrayValue ?? []).enumerated() {
                if let raw = mergeValue.stringValue {
                    let parts = raw.split(separator: " ", maxSplits: 1).map(String.init)
                    guard parts.count == 2 else {
                        continue
                    }
                    merges.append(TokenizerCatalogMerge(left: parts[0], right: parts[1], rank: rank))
                } else if let pair = mergeValue.arrayValue, pair.count == 2,
                          let left = pair[0].stringValue,
                          let right = pair[1].stringValue
                {
                    merges.append(TokenizerCatalogMerge(left: left, right: right, rank: rank))
                }
            }

            return TokenizerCatalog(
                modelType: modelType,
                unknownToken: unknownToken,
                unknownTokenID: unknownToken.flatMap { token in tokens.first { $0.content == token }?.id },
                tokens: tokens,
                merges: merges,
                duplicateContents: duplicates(tokens.map(\.content)),
                duplicateIDs: duplicates(tokens.map(\.id)),
                error: nil
            )
        } catch {
            return TokenizerCatalog(
                modelType: nil,
                unknownToken: nil,
                unknownTokenID: nil,
                tokens: [],
                merges: [],
                duplicateContents: [],
                duplicateIDs: [],
                error: String(describing: error)
            )
        }
    }

    private func sidecarBPECatalog(modelDirectory: URL) -> TokenizerCatalog? {
        let vocabURL = modelDirectory.appendingPathComponent("vocab.json")
        let mergesURL = modelDirectory.appendingPathComponent("merges.txt")
        guard fileManager.fileExists(atPath: vocabURL.path),
              fileManager.fileExists(atPath: mergesURL.path)
        else {
            return nil
        }

        do {
            let json = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: vocabURL))
            guard let vocab = json.objectValue else {
                throw ModelStoreError.invalidConfig("vocab.json must contain a JSON object")
            }

            var tokens: [TokenizerCatalogToken] = []
            for (content, idValue) in vocab {
                guard let id = idValue.intValue else {
                    continue
                }
                tokens.append(
                    TokenizerCatalogToken(
                        content: content,
                        id: id,
                        source: "vocab.json",
                        special: false
                    )
                )
            }

            let tokenizerConfig = try loadJSONObjectIfPresent(modelDirectory.appendingPathComponent("tokenizer_config.json"))
            markSpecialTokens(&tokens, fromTokenizerConfig: tokenizerConfig, vocab: vocab)
            let unknownToken = tokenizerConfig?["unk_token"]?.stringValue
            let merges = try loadMergesTXT(mergesURL)

            return TokenizerCatalog(
                modelType: "BPE",
                unknownToken: unknownToken,
                unknownTokenID: unknownToken.flatMap { token in tokens.first { $0.content == token }?.id },
                tokens: tokens,
                merges: merges,
                duplicateContents: duplicates(tokens.map(\.content)),
                duplicateIDs: duplicates(tokens.map(\.id)),
                error: nil
            )
        } catch {
            return TokenizerCatalog(
                modelType: "BPE",
                unknownToken: nil,
                unknownTokenID: nil,
                tokens: [],
                merges: [],
                duplicateContents: [],
                duplicateIDs: [],
                error: String(describing: error)
            )
        }
    }

    private func loadMergesTXT(_ url: URL) throws -> [TokenizerCatalogMerge] {
        let text = try String(contentsOf: url, encoding: .utf8)
        var merges: [TokenizerCatalogMerge] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else {
                continue
            }
            let parts = trimmed.split(separator: " ", maxSplits: 1).map(String.init)
            guard parts.count == 2 else {
                continue
            }
            merges.append(TokenizerCatalogMerge(left: parts[0], right: parts[1], rank: merges.count))
        }
        return merges
    }

    private func markSpecialTokens(
        _ tokens: inout [TokenizerCatalogToken],
        fromTokenizerConfig config: [String: JSONValue]?,
        vocab: [String: JSONValue]
    ) {
        guard let config else {
            return
        }

        var seen = Set<Int>()
        for value in config["added_tokens_decoder"]?.objectValue ?? [:] {
            guard let id = Int(value.key),
                  let content = specialTokenContent(value.value)
            else {
                continue
            }
            seen.insert(id)
            markOrAppendSpecialToken(
                content: content,
                id: id,
                source: "tokenizer_config.added_tokens_decoder",
                tokens: &tokens
            )
        }

        for (key, value) in config {
            guard key.hasSuffix("_token"),
                  let content = value.stringValue,
                  let id = vocab[content]?.intValue,
                  !seen.contains(id)
            else {
                continue
            }
            seen.insert(id)
            markOrAppendSpecialToken(
                content: content,
                id: id,
                source: "tokenizer_config.\(key)",
                tokens: &tokens
            )
        }
    }

    private func markOrAppendSpecialToken(
        content: String,
        id: Int,
        source: String,
        tokens: inout [TokenizerCatalogToken]
    ) {
        if let index = tokens.firstIndex(where: { $0.id == id && $0.content == content }) {
            tokens[index] = TokenizerCatalogToken(
                content: content,
                id: id,
                source: "\(tokens[index].source)+\(source)",
                special: true
            )
        } else {
            tokens.append(
                TokenizerCatalogToken(
                    content: content,
                    id: id,
                    source: source,
                    special: true
                )
            )
        }
    }

    private func specialTokenContent(_ value: JSONValue) -> String? {
        if let string = value.stringValue {
            return string
        }
        return value.objectValue?["content"]?.stringValue
    }

    private func loadJSONObjectIfPresent(_ url: URL) throws -> [String: JSONValue]? {
        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }
        let json = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))
        guard let object = json.objectValue else {
            throw ModelStoreError.invalidConfig("\(url.lastPathComponent) must contain a JSON object")
        }
        return object
    }

    private func duplicates<T: Hashable>(_ values: [T]) -> [T] {
        var seen = Set<T>()
        var duplicated = Set<T>()
        for value in values {
            if !seen.insert(value).inserted {
                duplicated.insert(value)
            }
        }
        return Array(duplicated)
    }
}
