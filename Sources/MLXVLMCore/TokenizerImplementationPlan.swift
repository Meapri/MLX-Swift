import Foundation

public struct TokenizerImplementationPlan: Codable, Equatable, Sendable {
    public let hasTokenizerJSON: Bool
    public let hasTokenizerModel: Bool
    public let hasTiktoken: Bool
    public let hasVocabJSON: Bool
    public let hasMergesTXT: Bool
    public let hasVocabTXT: Bool
    public let modelType: String?
    public let normalizerType: String?
    public let preTokenizerType: String?
    public let decoderType: String?
    public let requiredBackend: String
    public let catalogTokenCount: Int
    public let specialTokenCount: Int
    public let mergeCount: Int
    public let canUseCatalogPreflight: Bool
    public let swiftExecutionSupported: Bool
    public let swiftExecutionMode: String?
    public let requiresFullTokenizerImplementation: Bool
    public let blockingReasons: [String]

    public init(
        hasTokenizerJSON: Bool,
        hasTokenizerModel: Bool,
        hasTiktoken: Bool,
        hasVocabJSON: Bool,
        hasMergesTXT: Bool,
        hasVocabTXT: Bool,
        modelType: String?,
        normalizerType: String?,
        preTokenizerType: String?,
        decoderType: String?,
        requiredBackend: String,
        catalogTokenCount: Int,
        specialTokenCount: Int,
        mergeCount: Int,
        canUseCatalogPreflight: Bool,
        swiftExecutionSupported: Bool,
        swiftExecutionMode: String?,
        requiresFullTokenizerImplementation: Bool,
        blockingReasons: [String]
    ) {
        self.hasTokenizerJSON = hasTokenizerJSON
        self.hasTokenizerModel = hasTokenizerModel
        self.hasTiktoken = hasTiktoken
        self.hasVocabJSON = hasVocabJSON
        self.hasMergesTXT = hasMergesTXT
        self.hasVocabTXT = hasVocabTXT
        self.modelType = modelType
        self.normalizerType = normalizerType
        self.preTokenizerType = preTokenizerType
        self.decoderType = decoderType
        self.requiredBackend = requiredBackend
        self.catalogTokenCount = catalogTokenCount
        self.specialTokenCount = specialTokenCount
        self.mergeCount = mergeCount
        self.canUseCatalogPreflight = canUseCatalogPreflight
        self.swiftExecutionSupported = swiftExecutionSupported
        self.swiftExecutionMode = swiftExecutionMode
        self.requiresFullTokenizerImplementation = requiresFullTokenizerImplementation
        self.blockingReasons = blockingReasons
    }
}

public struct TokenizerImplementationPlanner {
    public init() {}

    public func plan(descriptor: ModelDescriptor, catalog: TokenizerCatalog?) -> TokenizerImplementationPlan {
        let metadata = descriptor.tokenizerMetadata
        let jsonMetadata = metadata.tokenizerJSONMetadata
        let modelType = jsonMetadata?.modelType ?? catalog?.modelType
        let requiredBackend = requiredBackend(
            modelType: jsonMetadata?.modelType,
            hasTokenizerJSON: metadata.hasTokenizerJSON,
            hasTokenizerModel: metadata.hasTokenizerModel,
            hasTiktoken: metadata.hasTiktoken,
            hasVocabJSON: metadata.hasVocabJSON,
            hasMergesTXT: metadata.hasMergesTXT,
            hasVocabTXT: metadata.hasVocabTXT
        )
        let canUseCatalogPreflight = catalog?.error == nil && (catalog?.tokenCount ?? 0) > 0
        let swiftExecutionMode = swiftExecutionMode(
            requiredBackend: requiredBackend,
            jsonMetadata: jsonMetadata,
            catalog: catalog
        )
        let swiftExecutionSupported = swiftExecutionMode != nil
        let blockingReasons = blockingReasons(
            metadata: metadata,
            jsonMetadata: jsonMetadata,
            catalog: catalog,
            requiredBackend: requiredBackend,
            swiftExecutionMode: swiftExecutionMode
        )

        return TokenizerImplementationPlan(
            hasTokenizerJSON: metadata.hasTokenizerJSON,
            hasTokenizerModel: metadata.hasTokenizerModel,
            hasTiktoken: metadata.hasTiktoken,
            hasVocabJSON: metadata.hasVocabJSON,
            hasMergesTXT: metadata.hasMergesTXT,
            hasVocabTXT: metadata.hasVocabTXT,
            modelType: modelType,
            normalizerType: jsonMetadata?.normalizerType,
            preTokenizerType: jsonMetadata?.preTokenizerType,
            decoderType: jsonMetadata?.decoderType,
            requiredBackend: requiredBackend,
            catalogTokenCount: catalog?.tokenCount ?? 0,
            specialTokenCount: catalog?.specialTokenCount ?? 0,
            mergeCount: catalog?.merges.count ?? 0,
            canUseCatalogPreflight: canUseCatalogPreflight,
            swiftExecutionSupported: swiftExecutionSupported,
            swiftExecutionMode: swiftExecutionMode,
            requiresFullTokenizerImplementation: !swiftExecutionSupported,
            blockingReasons: blockingReasons
        )
    }

    private func requiredBackend(
        modelType: String?,
        hasTokenizerJSON: Bool,
        hasTokenizerModel: Bool,
        hasTiktoken: Bool,
        hasVocabJSON: Bool,
        hasMergesTXT: Bool,
        hasVocabTXT: Bool
    ) -> String {
        switch modelType?.lowercased() {
        case "bpe":
            return "tokenizers-json-bpe"
        case "unigram":
            return hasTokenizerModel ? "sentencepiece-unigram-or-tokenizers-json" : "tokenizers-json-unigram"
        case "wordpiece":
            return "tokenizers-json-wordpiece"
        case "wordlevel":
            return "tokenizers-json-wordlevel"
        case .some(let value):
            return "tokenizers-json-\(value)"
        case nil:
            if hasTokenizerModel {
                return "sentencepiece-model"
            }
            if hasTiktoken {
                return "tiktoken-file"
            }
            if hasVocabJSON && hasMergesTXT {
                return "bpe-vocab-json-merges-txt"
            }
            if hasVocabTXT {
                return "wordpiece-vocab-txt"
            }
            if hasTokenizerJSON {
                return "tokenizers-json-unknown"
            }
            return "missing-tokenizer-files"
        }
    }

    private func blockingReasons(
        metadata: TokenizerMetadata,
        jsonMetadata: TokenizerJSONMetadata?,
        catalog: TokenizerCatalog?,
        requiredBackend: String,
        swiftExecutionMode: String?
    ) -> [String] {
        var reasons: [String] = []
        if !metadata.hasTokenizerJSON &&
            !metadata.hasTokenizerModel &&
            !metadata.hasTiktoken &&
            !metadata.hasVocabJSON &&
            !metadata.hasVocabTXT
        {
            reasons.append("No tokenizer.json, tokenizer.model, tokenizer.tiktoken, vocab.json, or vocab.txt file is present.")
        }
        if let jsonMetadata, !jsonMetadata.isReadable {
            reasons.append("tokenizer.json is present but unreadable: \(jsonMetadata.error ?? "unknown error").")
        }
        if let catalog, catalog.error != nil {
            reasons.append("Tokenizer catalog could not be built: \(catalog.error ?? "unknown error").")
        }
        if requiredBackend == "tokenizers-json-unknown" {
            reasons.append("tokenizer.json model type is unavailable.")
        }
        if swiftExecutionMode != nil {
            return reasons
        }
        if requiredBackend == "tokenizers-json-wordlevel" {
            reasons.append("WordLevel tokenizer execution is available for whitespace-separated text and exact special-token matches.")
        } else if requiredBackend == "tokenizers-json-bpe",
                  jsonMetadata?.preTokenizerType == "ByteLevel"
        {
            reasons.append("ByteLevel BPE tokenizer.json is present, but the vocab does not expose ByteLevel token strings needed by the dependency-free Swift tokenizer.")
        } else if requiredBackend == "tiktoken-file" ||
            requiredBackend == "bpe-vocab-json-merges-txt" ||
            requiredBackend == "wordpiece-vocab-txt"
        {
            reasons.append("Tokenizer sidecar files are present, but Swift execution for \(requiredBackend) is not implemented yet.")
        } else {
            reasons.append("Full Swift tokenizer execution is not implemented yet; catalog preflight only recognizes exact known token strings.")
        }
        return reasons
    }

    private func swiftExecutionMode(
        requiredBackend: String,
        jsonMetadata: TokenizerJSONMetadata?,
        catalog: TokenizerCatalog?
    ) -> String? {
        guard catalog?.error == nil, (catalog?.tokenCount ?? 0) > 0 else {
            return nil
        }
        if requiredBackend == "tokenizers-json-wordlevel" {
            return "wordlevel-whitespace"
        }
        if requiredBackend == "tokenizers-json-bpe",
           jsonMetadata?.preTokenizerType == "ByteLevel",
           jsonMetadata?.decoderType == "ByteLevel",
           hasByteLevelVocabulary(catalog)
        {
            return "bytelevel-bpe"
        }
        if requiredBackend == "bpe-vocab-json-merges-txt",
           hasByteLevelVocabulary(catalog)
        {
            return "bytelevel-bpe-sidecar"
        }
        return nil
    }

    private func hasByteLevelVocabulary(_ catalog: TokenizerCatalog?) -> Bool {
        catalog?.tokens.contains { token in
            token.content.contains("\u{0120}") || token.content.contains("\u{010A}")
        } ?? false
    }
}
