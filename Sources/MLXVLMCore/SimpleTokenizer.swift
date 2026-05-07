import Foundation

public struct SimpleTokenizerResult: Codable, Equatable, Sendable {
    public let supported: Bool
    public let requiredBackend: String
    public let tokens: [String]
    public let tokenIDs: [Int]
    public let unknownTokens: [String]
    public let usedUnknownTokenID: Int?
    public let error: String?

    public init(
        supported: Bool,
        requiredBackend: String,
        tokens: [String],
        tokenIDs: [Int],
        unknownTokens: [String],
        usedUnknownTokenID: Int?,
        error: String?
    ) {
        self.supported = supported
        self.requiredBackend = requiredBackend
        self.tokens = tokens
        self.tokenIDs = tokenIDs
        self.unknownTokens = unknownTokens
        self.usedUnknownTokenID = usedUnknownTokenID
        self.error = error
    }
}

public struct SimpleDetokenizerResult: Codable, Equatable, Sendable {
    public let supported: Bool
    public let requiredBackend: String
    public let text: String
    public let tokenIDs: [Int]
    public let unknownTokenIDs: [Int]
    public let error: String?

    public init(
        supported: Bool,
        requiredBackend: String,
        text: String,
        tokenIDs: [Int],
        unknownTokenIDs: [Int],
        error: String?
    ) {
        self.supported = supported
        self.requiredBackend = requiredBackend
        self.text = text
        self.tokenIDs = tokenIDs
        self.unknownTokenIDs = unknownTokenIDs
        self.error = error
    }
}

public struct SimpleTokenizer: Sendable {
    public let catalog: TokenizerCatalog
    public let plan: TokenizerImplementationPlan

    public init(catalog: TokenizerCatalog, plan: TokenizerImplementationPlan) {
        self.catalog = catalog
        self.plan = plan
    }

    public func tokenize(_ text: String) -> SimpleTokenizerResult {
        guard plan.requiredBackend == "tokenizers-json-wordlevel" ||
            plan.requiredBackend == "tokenizers-json-bpe" ||
            plan.requiredBackend == "bpe-vocab-json-merges-txt"
        else {
            return SimpleTokenizerResult(
                supported: false,
                requiredBackend: plan.requiredBackend,
                tokens: [],
                tokenIDs: [],
                unknownTokens: [],
                usedUnknownTokenID: nil,
                error: "Simple Swift tokenizer currently supports WordLevel tokenizer.json, tokenizer.json ByteLevel BPE, and vocab.json plus merges.txt ByteLevel BPE."
            )
        }

        let result: ([String], [Int], [String])
        if isBPEBackend, usesByteLevelVocabulary {
            result = tokenizeByteLevelBPE(text)
        } else {
            result = tokenizeCatalogPieces(splitWordLevel(text), allowBPE: isBPEBackend)
        }

        return SimpleTokenizerResult(
            supported: true,
            requiredBackend: plan.requiredBackend,
            tokens: result.0,
            tokenIDs: result.1,
            unknownTokens: result.2,
            usedUnknownTokenID: result.2.isEmpty ? nil : catalog.unknownTokenID,
            error: nil
        )
    }

    public func detokenize(_ tokenIDs: [Int], skipSpecialTokens: Bool = false) -> SimpleDetokenizerResult {
        guard plan.requiredBackend == "tokenizers-json-wordlevel" ||
            plan.requiredBackend == "tokenizers-json-bpe" ||
            plan.requiredBackend == "bpe-vocab-json-merges-txt"
        else {
            return SimpleDetokenizerResult(
                supported: false,
                requiredBackend: plan.requiredBackend,
                text: "",
                tokenIDs: tokenIDs,
                unknownTokenIDs: [],
                error: "Simple Swift detokenizer currently supports WordLevel tokenizer.json, tokenizer.json ByteLevel BPE, and vocab.json plus merges.txt ByteLevel BPE."
            )
        }

        var tokenTexts: [String] = []
        var unknownIDs: [Int] = []
        for id in tokenIDs {
            guard let token = catalog.token(for: id) else {
                unknownIDs.append(id)
                continue
            }
            if skipSpecialTokens, catalog.isSpecialToken(id: id) {
                continue
            }
            tokenTexts.append(token)
        }

        let text = if isBPEBackend, usesByteLevelVocabulary {
            decodeByteLevel(tokenTexts.joined())
        } else {
            tokenTexts.joined(separator: " ")
        }

        return SimpleDetokenizerResult(
            supported: true,
            requiredBackend: plan.requiredBackend,
            text: text,
            tokenIDs: tokenIDs,
            unknownTokenIDs: unknownIDs,
            error: nil
        )
    }

    private var isBPEBackend: Bool {
        plan.requiredBackend == "tokenizers-json-bpe" ||
            plan.requiredBackend == "bpe-vocab-json-merges-txt"
    }

    private var usesByteLevelVocabulary: Bool {
        (plan.preTokenizerType == "ByteLevel" || plan.requiredBackend == "bpe-vocab-json-merges-txt") &&
            catalog.tokens.contains { token in
                token.content.contains("\u{0120}") || token.content.contains("\u{010A}")
            }
    }

    private func tokenizeCatalogPieces(_ pieces: [String], allowBPE: Bool) -> ([String], [Int], [String]) {
        var tokens: [String] = []
        var ids: [Int] = []
        var unknown: [String] = []
        for piece in pieces {
            if let id = catalog.id(for: piece) {
                tokens.append(piece)
                ids.append(id)
            } else if allowBPE,
                      let bpe = tokenizeBPEPiece(piece)
            {
                tokens.append(contentsOf: bpe.map(\.token))
                ids.append(contentsOf: bpe.map(\.id))
            } else if let unknownID = catalog.unknownTokenID {
                unknown.append(piece)
                tokens.append(piece)
                ids.append(unknownID)
            } else {
                unknown.append(piece)
            }
        }
        return (tokens, ids, unknown)
    }

    private func splitWordLevel(_ text: String) -> [String] {
        let specialTokens = catalog.tokens
            .filter(\.special)
            .map(\.content)
            .filter { !$0.isEmpty }
            .sorted { lhs, rhs in
                if lhs.count == rhs.count {
                    return lhs < rhs
                }
                return lhs.count > rhs.count
            }

        var pieces: [String] = []
        var buffer = ""
        var index = text.startIndex
        while index < text.endIndex {
            if let special = specialTokens.first(where: { text[index...].hasPrefix($0) }) {
                appendWhitespacePieces(buffer, to: &pieces)
                buffer.removeAll()
                pieces.append(special)
                index = text.index(index, offsetBy: special.count)
            } else {
                buffer.append(text[index])
                index = text.index(after: index)
            }
        }
        appendWhitespacePieces(buffer, to: &pieces)
        return pieces
    }

    private func appendWhitespacePieces(_ text: String, to pieces: inout [String]) {
        for piece in text.split(whereSeparator: \.isWhitespace) {
            pieces.append(String(piece))
        }
    }

    private func tokenizeByteLevelBPE(_ text: String) -> ([String], [Int], [String]) {
        var tokens: [String] = []
        var ids: [Int] = []
        var unknown: [String] = []
        for piece in splitByteLevel(text) {
            if catalog.isSpecialToken(content: piece), let id = catalog.id(for: piece) {
                tokens.append(piece)
                ids.append(id)
                continue
            }

            let encoded = encodeByteLevel(piece)
            if let id = catalog.id(for: encoded) {
                tokens.append(encoded)
                ids.append(id)
            } else if let bpe = tokenizeBPEPiece(encoded) {
                tokens.append(contentsOf: bpe.map(\.token))
                ids.append(contentsOf: bpe.map(\.id))
            } else if let unknownID = catalog.unknownTokenID {
                unknown.append(piece)
                tokens.append(piece)
                ids.append(unknownID)
            } else {
                unknown.append(piece)
            }
        }
        return (tokens, ids, unknown)
    }

    private func splitByteLevel(_ text: String) -> [String] {
        let specialTokens = catalog.tokens
            .filter(\.special)
            .map(\.content)
            .filter { !$0.isEmpty }
            .sorted { lhs, rhs in
                if lhs.count == rhs.count {
                    return lhs < rhs
                }
                return lhs.count > rhs.count
            }

        var pieces: [String] = []
        var index = text.startIndex
        while index < text.endIndex {
            if let special = specialTokens.first(where: { text[index...].hasPrefix($0) }) {
                pieces.append(special)
                index = text.index(index, offsetBy: special.count)
                continue
            }

            let start = index
            var leadingWhitespace = ""
            while index < text.endIndex,
                  text[index].isWhitespace,
                  specialTokens.first(where: { text[index...].hasPrefix($0) }) == nil
            {
                leadingWhitespace.append(text[index])
                index = text.index(after: index)
            }
            guard index < text.endIndex else {
                pieces.append(leadingWhitespace)
                break
            }
            if let special = specialTokens.first(where: { text[index...].hasPrefix($0) }) {
                if !leadingWhitespace.isEmpty {
                    pieces.append(leadingWhitespace)
                }
                pieces.append(special)
                index = text.index(index, offsetBy: special.count)
                continue
            }

            let mode = characterMode(text[index])
            index = text.index(after: index)
            while index < text.endIndex,
                  !text[index].isWhitespace,
                  specialTokens.first(where: { text[index...].hasPrefix($0) }) == nil,
                  characterMode(text[index]) == mode
            {
                index = text.index(after: index)
            }
            pieces.append(String(text[start..<index]))
        }
        return pieces.filter { !$0.isEmpty }
    }

    private enum ByteLevelCharacterMode: Equatable {
        case letter
        case number
        case other
    }

    private func characterMode(_ character: Character) -> ByteLevelCharacterMode {
        if character.isLetter {
            return .letter
        }
        if character.isNumber {
            return .number
        }
        return .other
    }

    private func tokenizeBPEPiece(_ piece: String) -> [(token: String, id: Int)]? {
        guard !catalog.merges.isEmpty else {
            return nil
        }

        var symbols = piece.map(String.init)
        let ranks = Dictionary(
            catalog.merges.map { ("\($0.left)\u{0}\($0.right)", $0.rank) },
            uniquingKeysWith: min
        )

        while symbols.count > 1 {
            var bestIndex: Int?
            var bestRank = Int.max
            for index in 0..<(symbols.count - 1) {
                let key = "\(symbols[index])\u{0}\(symbols[index + 1])"
                guard let rank = ranks[key], rank < bestRank else {
                    continue
                }
                bestRank = rank
                bestIndex = index
            }
            guard let index = bestIndex else {
                break
            }
            symbols[index] += symbols[index + 1]
            symbols.remove(at: index + 1)
        }

        var output: [(token: String, id: Int)] = []
        for symbol in symbols {
            guard let id = catalog.id(for: symbol) else {
                return nil
            }
            output.append((symbol, id))
        }
        return output
    }

    private func encodeByteLevel(_ text: String) -> String {
        let mapping = Self.bytesToUnicode
        var scalars = String.UnicodeScalarView()
        for byte in text.utf8 {
            if let codePoint = mapping[Int(byte)],
               let scalar = UnicodeScalar(codePoint)
            {
                scalars.append(scalar)
            }
        }
        return String(scalars)
    }

    private func decodeByteLevel(_ text: String) -> String {
        let inverse = Self.unicodeToBytes
        var bytes: [UInt8] = []
        for scalar in text.unicodeScalars {
            if let byte = inverse[Int(scalar.value)] {
                bytes.append(UInt8(byte))
            } else {
                bytes.append(contentsOf: String(scalar).utf8)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private static let bytesToUnicode: [Int: UInt32] = {
        var byteValues = Array(33...126) + Array(161...172) + Array(174...255)
        var codePoints = byteValues
        var next = 0
        for byte in 0...255 where !byteValues.contains(byte) {
            byteValues.append(byte)
            codePoints.append(256 + next)
            next += 1
        }

        var mapping: [Int: UInt32] = [:]
        for (byte, codePoint) in zip(byteValues, codePoints) {
            mapping[byte] = UInt32(codePoint)
        }
        return mapping
    }()

    private static let unicodeToBytes: [Int: Int] = {
        var inverse: [Int: Int] = [:]
        for (byte, codePoint) in bytesToUnicode {
            inverse[Int(codePoint)] = byte
        }
        return inverse
    }()
}
