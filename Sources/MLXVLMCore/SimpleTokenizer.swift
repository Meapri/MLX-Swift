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
            plan.requiredBackend == "tokenizers-json-wordpiece" ||
            plan.requiredBackend == "tokenizers-json-bpe" ||
            plan.requiredBackend == "bpe-vocab-json-merges-txt" ||
            plan.requiredBackend == "wordpiece-vocab-txt" ||
            plan.requiredBackend == "tiktoken-file"
        else {
            return SimpleTokenizerResult(
                supported: false,
                requiredBackend: plan.requiredBackend,
                tokens: [],
                tokenIDs: [],
                unknownTokens: [],
                usedUnknownTokenID: nil,
                error: "Simple Swift tokenizer currently supports WordLevel tokenizer.json, WordPiece tokenizer.json/vocab.txt, tokenizer.tiktoken exact-piece catalogs, tokenizer.json ByteLevel BPE, and vocab.json plus merges.txt ByteLevel BPE."
            )
        }

        let result: ([String], [Int], [String])
        if isBPEBackend, usesByteLevelVocabulary {
            result = tokenizeByteLevelBPE(text)
        } else if isWordPieceBackend {
            result = tokenizeWordPiece(text)
        } else if isTiktokenBackend {
            result = tokenizeGreedyKnownText(text)
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
            plan.requiredBackend == "tokenizers-json-wordpiece" ||
            plan.requiredBackend == "tokenizers-json-bpe" ||
            plan.requiredBackend == "bpe-vocab-json-merges-txt" ||
            plan.requiredBackend == "wordpiece-vocab-txt" ||
            plan.requiredBackend == "tiktoken-file"
        else {
            return SimpleDetokenizerResult(
                supported: false,
                requiredBackend: plan.requiredBackend,
                text: "",
                tokenIDs: tokenIDs,
                unknownTokenIDs: [],
                error: "Simple Swift detokenizer currently supports WordLevel tokenizer.json, WordPiece tokenizer.json/vocab.txt, tokenizer.tiktoken exact-piece catalogs, tokenizer.json ByteLevel BPE, and vocab.json plus merges.txt ByteLevel BPE."
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
        } else if isWordPieceBackend {
            detokenizeWordPiece(tokenTexts)
        } else if isTiktokenBackend {
            tokenTexts.joined()
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

    private var isWordPieceBackend: Bool {
        plan.requiredBackend == "tokenizers-json-wordpiece" ||
            plan.requiredBackend == "wordpiece-vocab-txt"
    }

    private var isTiktokenBackend: Bool {
        plan.requiredBackend == "tiktoken-file"
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

    private func tokenizeGreedyKnownText(_ text: String) -> ([String], [Int], [String]) {
        let rankedTokens = catalog.tokens
            .map(\.content)
            .filter { !$0.isEmpty }
            .sorted { lhs, rhs in
                if lhs.count == rhs.count {
                    return lhs < rhs
                }
                return lhs.count > rhs.count
            }

        var tokens: [String] = []
        var ids: [Int] = []
        var unknown: [String] = []
        var index = text.startIndex
        while index < text.endIndex {
            if let token = rankedTokens.first(where: { text[index...].hasPrefix($0) }),
               let id = catalog.id(for: token)
            {
                tokens.append(token)
                ids.append(id)
                index = text.index(index, offsetBy: token.count)
                continue
            }

            let nextIndex = text.index(after: index)
            let piece = String(text[index..<nextIndex])
            if !piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                unknown.append(piece)
            }
            if let unknownID = catalog.unknownTokenID {
                tokens.append(piece)
                ids.append(unknownID)
            }
            index = nextIndex
        }
        return (tokens, ids, unknown)
    }

    private func tokenizeWordPiece(_ text: String) -> ([String], [Int], [String]) {
        var tokens: [String] = []
        var ids: [Int] = []
        var unknown: [String] = []
        for piece in splitWordPieceInput(text) {
            if catalog.isSpecialToken(content: piece), let id = catalog.id(for: piece) {
                tokens.append(piece)
                ids.append(id)
                continue
            }
            if let id = catalog.id(for: piece) {
                tokens.append(piece)
                ids.append(id)
                continue
            }
            guard let wordPieces = greedyWordPiece(piece) else {
                if let unknownID = catalog.unknownTokenID {
                    tokens.append(piece)
                    ids.append(unknownID)
                }
                unknown.append(piece)
                continue
            }
            tokens.append(contentsOf: wordPieces.map(\.token))
            ids.append(contentsOf: wordPieces.map(\.id))
        }
        return (tokens, ids, unknown)
    }

    private func splitWordPieceInput(_ text: String) -> [String] {
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
        func flushBuffer() {
            if !buffer.isEmpty {
                pieces.append(buffer)
                buffer.removeAll()
            }
        }

        var index = text.startIndex
        while index < text.endIndex {
            if let special = specialTokens.first(where: { text[index...].hasPrefix($0) }) {
                flushBuffer()
                pieces.append(special)
                index = text.index(index, offsetBy: special.count)
                continue
            }

            let character = text[index]
            if character.isWhitespace {
                flushBuffer()
            } else if isWordPiecePunctuation(character) {
                flushBuffer()
                pieces.append(String(character))
            } else {
                buffer.append(character)
            }
            index = text.index(after: index)
        }
        flushBuffer()
        return pieces
    }

    private func greedyWordPiece(_ text: String) -> [(token: String, id: Int)]? {
        let variants = wordPieceCaseVariants(text)
        for candidate in variants {
            if let pieces = greedyWordPieceCandidate(candidate) {
                return pieces
            }
        }
        return nil
    }

    private func wordPieceCaseVariants(_ text: String) -> [String] {
        let lowercased = text.lowercased()
        return lowercased == text ? [text] : [text, lowercased]
    }

    private func greedyWordPieceCandidate(_ text: String) -> [(token: String, id: Int)]? {
        guard !text.isEmpty else {
            return []
        }
        let characters = Array(text)
        var output: [(token: String, id: Int)] = []
        var start = 0
        while start < characters.count {
            var end = characters.count
            var matched: (token: String, id: Int, end: Int)?
            while start < end {
                let fragment = String(characters[start..<end])
                let token = start == 0 ? fragment : "##\(fragment)"
                if let id = catalog.id(for: token) {
                    matched = (token, id, end)
                    break
                }
                end -= 1
            }
            guard let matched else {
                return nil
            }
            output.append((matched.token, matched.id))
            start = matched.end
        }
        return output
    }

    private func detokenizeWordPiece(_ tokens: [String]) -> String {
        var text = ""
        for token in tokens {
            if token.hasPrefix("##") {
                text += String(token.dropFirst(2))
            } else if text.isEmpty || catalog.isSpecialToken(content: token) || isStandalonePunctuationToken(token) {
                text += token
            } else {
                text += " \(token)"
            }
        }
        return text
    }

    private func isWordPiecePunctuation(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy { scalar in
            switch scalar.properties.generalCategory {
            case .connectorPunctuation,
                 .dashPunctuation,
                 .closePunctuation,
                 .finalPunctuation,
                 .initialPunctuation,
                 .otherPunctuation,
                 .openPunctuation:
                return true
            default:
                return false
            }
        }
    }

    private func isStandalonePunctuationToken(_ token: String) -> Bool {
        token.count == 1 && token.unicodeScalars.allSatisfy { scalar in
            switch scalar.properties.generalCategory {
            case .connectorPunctuation,
                 .dashPunctuation,
                 .closePunctuation,
                 .finalPunctuation,
                 .initialPunctuation,
                 .otherPunctuation,
                 .openPunctuation:
                return true
            default:
                return false
            }
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
