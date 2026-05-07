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
        if let catalog = sentencePieceModelCatalog(modelDirectory: modelDirectory) {
            return catalog
        }
        if let catalog = sidecarTiktokenCatalog(modelDirectory: modelDirectory) {
            return catalog
        }
        if let catalog = sidecarBPECatalog(modelDirectory: modelDirectory) {
            return catalog
        }
        return sidecarWordPieceCatalog(modelDirectory: modelDirectory)
    }

    private func tokenizerJSONCatalog(tokenizerURL: URL) -> TokenizerCatalog {
        do {
            let json = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: tokenizerURL))
            guard let object = json.objectValue else {
                throw ModelStoreError.invalidConfig("tokenizer.json must contain a JSON object")
            }
            let model = object["model"]?.objectValue
            let modelType = model?["type"]?.stringValue
            var unknownToken = model?["unk_token"]?.stringValue
            var tokens: [TokenizerCatalogToken] = []
            var merges: [TokenizerCatalogMerge] = []

            if let vocab = model?["vocab"]?.objectValue {
                for (content, idValue) in vocab {
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
            } else if let vocab = model?["vocab"]?.arrayValue {
                for (index, tokenValue) in vocab.enumerated() {
                    let content: String?
                    if let token = tokenValue.stringValue {
                        content = token
                    } else {
                        content = tokenValue.arrayValue?.first?.stringValue
                    }
                    guard let content else {
                        continue
                    }
                    tokens.append(
                        TokenizerCatalogToken(
                            content: content,
                            id: index,
                            source: "model.vocab",
                            special: false
                        )
                    )
                }
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

            if unknownToken == nil,
               let unknownTokenID = model?["unk_id"]?.intValue
            {
                unknownToken = tokens.first { $0.id == unknownTokenID }?.content
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

    private func sentencePieceModelCatalog(modelDirectory: URL) -> TokenizerCatalog? {
        let modelURL = modelDirectory.appendingPathComponent("tokenizer.model")
        guard fileManager.fileExists(atPath: modelURL.path) else {
            return nil
        }

        do {
            let parsed = try SentencePieceModelProtoParser(data: Data(contentsOf: modelURL)).parse()
            guard !parsed.tokens.isEmpty else {
                throw ModelStoreError.invalidConfig("tokenizer.model did not contain SentencePiece pieces")
            }

            let tokenizerConfig = try loadJSONObjectIfPresent(modelDirectory.appendingPathComponent("tokenizer_config.json"))
            var tokens = parsed.tokens
            var vocab: [String: JSONValue] = [:]
            for token in tokens {
                vocab[token.content] = .number(Double(token.id))
            }
            markSpecialTokens(&tokens, fromTokenizerConfig: tokenizerConfig, vocab: vocab)
            let unknownToken = tokenizerConfig?["unk_token"]?.stringValue ?? parsed.unknownToken

            return TokenizerCatalog(
                modelType: "Unigram",
                unknownToken: unknownToken,
                unknownTokenID: unknownToken.flatMap { token in tokens.first { $0.content == token }?.id },
                tokens: tokens,
                merges: [],
                duplicateContents: duplicates(tokens.map(\.content)),
                duplicateIDs: duplicates(tokens.map(\.id)),
                error: nil
            )
        } catch {
            return TokenizerCatalog(
                modelType: "SentencePiece",
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

    private func sidecarWordPieceCatalog(modelDirectory: URL) -> TokenizerCatalog? {
        let vocabURL = modelDirectory.appendingPathComponent("vocab.txt")
        guard fileManager.fileExists(atPath: vocabURL.path) else {
            return nil
        }

        do {
            let text = try String(contentsOf: vocabURL, encoding: .utf8)
            var tokens: [TokenizerCatalogToken] = []
            var vocab: [String: JSONValue] = [:]
            for (id, rawLine) in text.split(whereSeparator: \.isNewline).enumerated() {
                let content = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !content.isEmpty else {
                    continue
                }
                tokens.append(
                    TokenizerCatalogToken(
                        content: content,
                        id: id,
                        source: "vocab.txt",
                        special: Self.looksLikeSpecialToken(content)
                    )
                )
                vocab[content] = .number(Double(id))
            }

            let tokenizerConfig = try loadJSONObjectIfPresent(modelDirectory.appendingPathComponent("tokenizer_config.json"))
            markSpecialTokens(&tokens, fromTokenizerConfig: tokenizerConfig, vocab: vocab)
            let unknownToken = tokenizerConfig?["unk_token"]?.stringValue
                ?? (vocab["[UNK]"] == nil ? nil : "[UNK]")

            return TokenizerCatalog(
                modelType: "WordPiece",
                unknownToken: unknownToken,
                unknownTokenID: unknownToken.flatMap { token in tokens.first { $0.content == token }?.id },
                tokens: tokens,
                merges: [],
                duplicateContents: duplicates(tokens.map(\.content)),
                duplicateIDs: duplicates(tokens.map(\.id)),
                error: nil
            )
        } catch {
            return TokenizerCatalog(
                modelType: "WordPiece",
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

    private func sidecarTiktokenCatalog(modelDirectory: URL) -> TokenizerCatalog? {
        let tiktokenURL = modelDirectory.appendingPathComponent("tokenizer.tiktoken")
        guard fileManager.fileExists(atPath: tiktokenURL.path) else {
            return nil
        }

        do {
            let text = try String(contentsOf: tiktokenURL, encoding: .utf8)
            var tokens: [TokenizerCatalogToken] = []
            var vocab: [String: JSONValue] = [:]
            for line in text.split(whereSeparator: \.isNewline) {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else {
                    continue
                }
                let parts = trimmed.split(whereSeparator: \.isWhitespace).map(String.init)
                guard parts.count >= 2,
                      let id = Int(parts[1]),
                      let content = Self.tiktokenContent(parts[0])
                else {
                    continue
                }
                tokens.append(
                    TokenizerCatalogToken(
                        content: content,
                        id: id,
                        source: "tokenizer.tiktoken",
                        special: Self.looksLikeSpecialToken(content)
                    )
                )
                vocab[content] = .number(Double(id))
            }

            let tokenizerConfig = try loadJSONObjectIfPresent(modelDirectory.appendingPathComponent("tokenizer_config.json"))
            markSpecialTokens(&tokens, fromTokenizerConfig: tokenizerConfig, vocab: vocab)
            let unknownToken = tokenizerConfig?["unk_token"]?.stringValue
                ?? tokenizerConfig?["oov_token"]?.stringValue

            return TokenizerCatalog(
                modelType: "Tiktoken",
                unknownToken: unknownToken,
                unknownTokenID: unknownToken.flatMap { token in tokens.first { $0.content == token }?.id },
                tokens: tokens,
                merges: [],
                duplicateContents: duplicates(tokens.map(\.content)),
                duplicateIDs: duplicates(tokens.map(\.id)),
                error: nil
            )
        } catch {
            return TokenizerCatalog(
                modelType: "Tiktoken",
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

    private static func looksLikeSpecialToken(_ token: String) -> Bool {
        token.count >= 3 &&
            ((token.hasPrefix("[") && token.hasSuffix("]")) ||
                (token.hasPrefix("<") && token.hasSuffix(">")))
    }

    private static func tiktokenContent(_ raw: String) -> String? {
        if let data = Data(base64Encoded: raw),
           let decoded = String(data: data, encoding: .utf8)
        {
            return decoded
        }
        return raw
    }
}

private struct SentencePieceModelProtoParser {
    private let data: Data

    init(data: Data) {
        self.data = data
    }

    func parse() throws -> (tokens: [TokenizerCatalogToken], unknownToken: String?) {
        var reader = ProtoReader(data: data)
        var tokens: [TokenizerCatalogToken] = []
        var unknownToken: String?
        while !reader.isAtEnd {
            let key = try reader.readVarint()
            let field = Int(key >> 3)
            let wireType = Int(key & 0x07)
            if field == 1, wireType == 2 {
                let pieceData = try reader.readLengthDelimitedData()
                if let piece = try Self.parsePiece(pieceData, id: tokens.count) {
                    tokens.append(piece.token)
                    if piece.isUnknown {
                        unknownToken = piece.token.content
                    }
                }
            } else {
                try reader.skip(wireType: wireType)
            }
        }
        return (tokens, unknownToken)
    }

    private static func parsePiece(
        _ data: Data,
        id: Int
    ) throws -> (token: TokenizerCatalogToken, isUnknown: Bool)? {
        var reader = ProtoReader(data: data)
        var content: String?
        var pieceType = 1
        while !reader.isAtEnd {
            let key = try reader.readVarint()
            let field = Int(key >> 3)
            let wireType = Int(key & 0x07)
            switch (field, wireType) {
            case (1, 2):
                content = try reader.readLengthDelimitedString()
            case (3, 0):
                pieceType = Int(try reader.readVarint())
            default:
                try reader.skip(wireType: wireType)
            }
        }
        guard let content else {
            return nil
        }
        return (
            TokenizerCatalogToken(
                content: content,
                id: id,
                source: "tokenizer.model",
                special: pieceType == 2 || pieceType == 3 || pieceType == 4 || pieceType == 5
            ),
            pieceType == 2
        )
    }
}

private struct ProtoReader {
    private let bytes: [UInt8]
    private var offset = 0

    init(data: Data) {
        self.bytes = Array(data)
    }

    var isAtEnd: Bool {
        offset >= bytes.count
    }

    mutating func readVarint() throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        while shift < 64 {
            guard offset < bytes.count else {
                throw ModelStoreError.invalidConfig("Unexpected end of tokenizer.model varint")
            }
            let byte = bytes[offset]
            offset += 1
            result |= UInt64(byte & 0x7F) << shift
            if byte & 0x80 == 0 {
                return result
            }
            shift += 7
        }
        throw ModelStoreError.invalidConfig("Invalid tokenizer.model varint")
    }

    mutating func readLengthDelimitedData() throws -> Data {
        let length = Int(try readVarint())
        guard length >= 0, offset + length <= bytes.count else {
            throw ModelStoreError.invalidConfig("Invalid tokenizer.model length-delimited field")
        }
        let start = offset
        offset += length
        return Data(bytes[start..<offset])
    }

    mutating func readLengthDelimitedString() throws -> String {
        let data = try readLengthDelimitedData()
        guard let string = String(data: data, encoding: .utf8) else {
            throw ModelStoreError.invalidConfig("Invalid UTF-8 string in tokenizer.model")
        }
        return string
    }

    mutating func skip(wireType: Int) throws {
        switch wireType {
        case 0:
            _ = try readVarint()
        case 1:
            try skipBytes(8)
        case 2:
            _ = try readLengthDelimitedData()
        case 5:
            try skipBytes(4)
        default:
            throw ModelStoreError.invalidConfig("Unsupported tokenizer.model protobuf wire type: \(wireType)")
        }
    }

    private mutating func skipBytes(_ count: Int) throws {
        guard count >= 0, offset + count <= bytes.count else {
            throw ModelStoreError.invalidConfig("Unexpected end of tokenizer.model fixed-width field")
        }
        offset += count
    }
}
