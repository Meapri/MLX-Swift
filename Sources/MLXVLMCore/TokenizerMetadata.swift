import Foundation

public struct TokenizerJSONMetadata: Codable, Equatable, Sendable {
    public let isReadable: Bool
    public let modelType: String?
    public let vocabCount: Int?
    public let mergeCount: Int?
    public let addedTokenCount: Int?
    public let normalizerType: String?
    public let preTokenizerType: String?
    public let decoderType: String?
    public let error: String?

    public init(
        isReadable: Bool,
        modelType: String?,
        vocabCount: Int?,
        mergeCount: Int?,
        addedTokenCount: Int?,
        normalizerType: String?,
        preTokenizerType: String?,
        decoderType: String?,
        error: String?
    ) {
        self.isReadable = isReadable
        self.modelType = modelType
        self.vocabCount = vocabCount
        self.mergeCount = mergeCount
        self.addedTokenCount = addedTokenCount
        self.normalizerType = normalizerType
        self.preTokenizerType = preTokenizerType
        self.decoderType = decoderType
        self.error = error
    }
}

public struct TokenizerMetadata: Codable, Equatable, Sendable {
    public let hasTokenizerJSON: Bool
    public let hasTokenizerModel: Bool
    public let hasTiktoken: Bool
    public let hasVocabJSON: Bool
    public let hasMergesTXT: Bool
    public let hasVocabTXT: Bool
    public let hasTokenizerConfig: Bool
    public let hasSpecialTokensMap: Bool
    public let tokenizerJSONMetadata: TokenizerJSONMetadata?
    public let chatTemplateSource: String?
    public let chatTemplate: String?
    public let imageToken: String?
    public let imageTokenID: Int?
    public let videoToken: String?
    public let videoTokenID: Int?
    public let visionStartToken: String?
    public let visionStartTokenID: Int?
    public let visionEndToken: String?
    public let visionEndTokenID: Int?

    public init(
        hasTokenizerJSON: Bool,
        hasTokenizerModel: Bool,
        hasTiktoken: Bool,
        hasVocabJSON: Bool,
        hasMergesTXT: Bool,
        hasVocabTXT: Bool,
        hasTokenizerConfig: Bool,
        hasSpecialTokensMap: Bool,
        tokenizerJSONMetadata: TokenizerJSONMetadata?,
        chatTemplateSource: String?,
        chatTemplate: String?,
        imageToken: String?,
        imageTokenID: Int?,
        videoToken: String?,
        videoTokenID: Int?,
        visionStartToken: String?,
        visionStartTokenID: Int?,
        visionEndToken: String?,
        visionEndTokenID: Int?
    ) {
        self.hasTokenizerJSON = hasTokenizerJSON
        self.hasTokenizerModel = hasTokenizerModel
        self.hasTiktoken = hasTiktoken
        self.hasVocabJSON = hasVocabJSON
        self.hasMergesTXT = hasMergesTXT
        self.hasVocabTXT = hasVocabTXT
        self.hasTokenizerConfig = hasTokenizerConfig
        self.hasSpecialTokensMap = hasSpecialTokensMap
        self.tokenizerJSONMetadata = tokenizerJSONMetadata
        self.chatTemplateSource = chatTemplateSource
        self.chatTemplate = chatTemplate
        self.imageToken = imageToken
        self.imageTokenID = imageTokenID
        self.videoToken = videoToken
        self.videoTokenID = videoTokenID
        self.visionStartToken = visionStartToken
        self.visionStartTokenID = visionStartTokenID
        self.visionEndToken = visionEndToken
        self.visionEndTokenID = visionEndTokenID
    }
}

public struct TokenizerMetadataLoader {
    public let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    public func load(from modelURL: URL) throws -> TokenizerMetadata {
        let tokenizerConfigURL = modelURL.appendingPathComponent("tokenizer_config.json")
        let specialTokensMapURL = modelURL.appendingPathComponent("special_tokens_map.json")
        let tokenizerJSONURL = modelURL.appendingPathComponent("tokenizer.json")
        let tokenizerModelURL = modelURL.appendingPathComponent("tokenizer.model")
        let tiktokenURL = modelURL.appendingPathComponent("tokenizer.tiktoken")
        let vocabJSONURL = modelURL.appendingPathComponent("vocab.json")
        let mergesTXTURL = modelURL.appendingPathComponent("merges.txt")
        let vocabTXTURL = modelURL.appendingPathComponent("vocab.txt")

        let tokenizerConfig = try loadJSONObjectIfPresent(tokenizerConfigURL)
        let specialTokensMap = try loadJSONObjectIfPresent(specialTokensMapURL)
        let addedTokens = tokenizerConfig?["added_tokens_decoder"]?.objectValue
        let tokenizerJSONMetadata = loadTokenizerJSONMetadata(tokenizerJSONURL)

        let chatTemplate = try loadChatTemplate(modelURL: modelURL, tokenizerConfig: tokenizerConfig)

        return TokenizerMetadata(
            hasTokenizerJSON: fileManager.fileExists(atPath: tokenizerJSONURL.path),
            hasTokenizerModel: fileManager.fileExists(atPath: tokenizerModelURL.path),
            hasTiktoken: fileManager.fileExists(atPath: tiktokenURL.path),
            hasVocabJSON: fileManager.fileExists(atPath: vocabJSONURL.path),
            hasMergesTXT: fileManager.fileExists(atPath: mergesTXTURL.path),
            hasVocabTXT: fileManager.fileExists(atPath: vocabTXTURL.path),
            hasTokenizerConfig: tokenizerConfig != nil,
            hasSpecialTokensMap: specialTokensMap != nil,
            tokenizerJSONMetadata: tokenizerJSONMetadata,
            chatTemplateSource: chatTemplate.source,
            chatTemplate: chatTemplate.template,
            imageToken: tokenString(
                explicit: tokenizerConfig?["image_token"],
                specialTokensMap: specialTokensMap,
                key: "image_token",
                defaultValue: "<|image_pad|>"
            ),
            imageTokenID: tokenID(tokenizerConfig: tokenizerConfig, addedTokens: addedTokens, key: "image_token_id", token: "<|image_pad|>"),
            videoToken: tokenString(
                explicit: tokenizerConfig?["video_token"],
                specialTokensMap: specialTokensMap,
                key: "video_token",
                defaultValue: "<|video_pad|>"
            ),
            videoTokenID: tokenID(tokenizerConfig: tokenizerConfig, addedTokens: addedTokens, key: "video_token_id", token: "<|video_pad|>"),
            visionStartToken: tokenString(
                explicit: tokenizerConfig?["vision_start_token"],
                specialTokensMap: specialTokensMap,
                key: "vision_start_token",
                defaultValue: "<|vision_start|>"
            ),
            visionStartTokenID: tokenID(tokenizerConfig: tokenizerConfig, addedTokens: addedTokens, key: "vision_start_token_id", token: "<|vision_start|>"),
            visionEndToken: tokenString(
                explicit: tokenizerConfig?["vision_end_token"],
                specialTokensMap: specialTokensMap,
                key: "vision_end_token",
                defaultValue: "<|vision_end|>"
            ),
            visionEndTokenID: tokenID(tokenizerConfig: tokenizerConfig, addedTokens: addedTokens, key: "vision_end_token_id", token: "<|vision_end|>")
        )
    }

    private func loadChatTemplate(
        modelURL: URL,
        tokenizerConfig: [String: JSONValue]?
    ) throws -> (source: String?, template: String?) {
        let jsonURL = modelURL.appendingPathComponent("chat_template.json")
        if let object = try loadJSONObjectIfPresent(jsonURL),
           let template = object["chat_template"]?.stringValue
        {
            return ("chat_template.json", template)
        }

        let jinjaURL = modelURL.appendingPathComponent("chat_template.jinja")
        if fileManager.fileExists(atPath: jinjaURL.path) {
            return ("chat_template.jinja", try String(contentsOf: jinjaURL, encoding: .utf8))
        }

        if let template = tokenizerConfig?["chat_template"]?.stringValue {
            return ("tokenizer_config.json", template)
        }

        return (nil, nil)
    }

    private func loadJSONObjectIfPresent(_ url: URL) throws -> [String: JSONValue]? {
        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }
        let data = try Data(contentsOf: url)
        let json = try JSONDecoder().decode(JSONValue.self, from: data)
        guard let object = json.objectValue else {
            throw ModelStoreError.invalidConfig("\(url.lastPathComponent) must contain a JSON object")
        }
        return object
    }

    private func loadTokenizerJSONMetadata(_ url: URL) -> TokenizerJSONMetadata? {
        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }

        do {
            let data = try Data(contentsOf: url)
            let json = try JSONDecoder().decode(JSONValue.self, from: data)
            guard let object = json.objectValue else {
                throw ModelStoreError.invalidConfig("tokenizer.json must contain a JSON object")
            }

            let model = object["model"]?.objectValue
            let normalizer = object["normalizer"]?.objectValue
            let preTokenizer = object["pre_tokenizer"]?.objectValue
            let decoder = object["decoder"]?.objectValue

            return TokenizerJSONMetadata(
                isReadable: true,
                modelType: model?["type"]?.stringValue,
                vocabCount: model?["vocab"]?.objectValue?.count,
                mergeCount: model?["merges"]?.arrayValue?.count,
                addedTokenCount: object["added_tokens"]?.arrayValue?.count,
                normalizerType: normalizer?["type"]?.stringValue,
                preTokenizerType: preTokenizer?["type"]?.stringValue,
                decoderType: decoder?["type"]?.stringValue,
                error: nil
            )
        } catch {
            return TokenizerJSONMetadata(
                isReadable: false,
                modelType: nil,
                vocabCount: nil,
                mergeCount: nil,
                addedTokenCount: nil,
                normalizerType: nil,
                preTokenizerType: nil,
                decoderType: nil,
                error: String(describing: error)
            )
        }
    }

    private func tokenString(
        explicit: JSONValue?,
        specialTokensMap: [String: JSONValue]?,
        key: String,
        defaultValue: String
    ) -> String? {
        if let string = explicit?.stringValue {
            return string
        }
        if let special = specialTokensMap?[key] {
            if let string = special.stringValue {
                return string
            }
            if let content = special.objectValue?["content"]?.stringValue {
                return content
            }
        }
        return defaultValue
    }

    private func tokenID(
        tokenizerConfig: [String: JSONValue]?,
        addedTokens: [String: JSONValue]?,
        key: String,
        token: String
    ) -> Int? {
        if let id = tokenizerConfig?[key]?.intValue {
            return id
        }

        for (id, tokenValue) in addedTokens ?? [:] {
            guard let intID = Int(id) else {
                continue
            }
            if tokenValue.stringValue == token {
                return intID
            }
            if tokenValue.objectValue?["content"]?.stringValue == token {
                return intID
            }
        }
        return nil
    }
}
