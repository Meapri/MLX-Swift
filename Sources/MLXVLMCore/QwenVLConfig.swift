import Foundation

public enum QwenVLConfigError: Error, CustomStringConvertible, Equatable {
    case unsupportedModelType(String)
    case invalidRopeScaling(String)

    public var description: String {
        switch self {
        case .unsupportedModelType(let modelType):
            return "Qwen VL Swift config supports qwen2_vl and qwen2_5_vl, got: \(modelType)"
        case .invalidRopeScaling(let reason):
            return "Invalid Qwen VL rope_scaling: \(reason)"
        }
    }
}

public enum QwenVLFamily: String, Codable, Equatable, Sendable {
    case qwen2VL = "qwen2_vl"
    case qwen25VL = "qwen2_5_vl"
}

public struct QwenVLVisionConfig: Codable, Equatable, Sendable {
    public let modelType: String
    public let depth: Int
    public let embedDim: Int?
    public let hiddenSize: Int
    public let intermediateSize: Int?
    public let outHiddenSize: Int?
    public let numHeads: Int
    public let imageSize: Int
    public let patchSize: Int
    public let vocabSize: Int
    public let mlpRatio: Double
    public let inChannels: Int
    public let layerNormEps: Double
    public let spatialPatchSize: Int
    public let spatialMergeSize: Int
    public let temporalPatchSize: Int
    public let tokensPerSecond: Int?
    public let windowSize: Int?
    public let fullattBlockIndexes: [Int]
}

public struct QwenVLTextConfig: Codable, Equatable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let rmsNormEps: Double
    public let vocabSize: Int
    public let numKeyValueHeads: Int
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Double
    public let ropeTraditional: Bool
    public let ropeScaling: JSONValue?
    public let tieWordEmbeddings: Bool
    public let slidingWindow: Int?
    public let useSlidingWindow: Bool?
    public let useCache: Bool?
}

public struct QwenVLModelConfig: Codable, Equatable, Sendable {
    public let family: QwenVLFamily
    public let modelType: String
    public let textConfig: QwenVLTextConfig
    public let visionConfig: QwenVLVisionConfig
    public let ignoreIndex: Int
    public let imageTokenID: Int
    public let videoTokenID: Int
    public let visionStartTokenID: Int
    public let visionEndTokenID: Int?
    public let visionTokenID: Int?
    public let visionFeatureSelectStrategy: String
    public let visionFeatureLayer: Int
    public let vocabSize: Int
    public let eosTokenIDs: [Int]?

    public static func load(fromModelDirectory path: String) throws -> QwenVLModelConfig {
        let config = try ModelStore().loadConfig(pathOrIdentifier: path)
        return try QwenVLModelConfig(json: config)
    }

    public init(json: JSONValue) throws {
        guard let object = json.objectValue else {
            throw ModelStoreError.invalidConfig("top-level config must be a JSON object")
        }

        let modelType = try object.requiredString("model_type").lowercased()
        guard let family = QwenVLFamily(rawValue: modelType) else {
            throw QwenVLConfigError.unsupportedModelType(modelType)
        }

        guard let visionObject = object["vision_config"]?.objectValue else {
            throw ModelStoreError.invalidConfig("missing object field 'vision_config'")
        }

        self.family = family
        self.modelType = modelType
        self.textConfig = try Self.parseTextConfig(object, family: family)
        self.visionConfig = Self.parseVisionConfig(visionObject, family: family)
        self.ignoreIndex = object.int("ignore_index", default: -100)
        self.imageTokenID = object.int("image_token_id", default: 151655)
        self.videoTokenID = object.int("video_token_id", default: 151656)
        self.visionStartTokenID = object.int("vision_start_token_id", default: 151652)
        self.visionEndTokenID = object["vision_end_token_id"]?.intValue
        self.visionTokenID = object["vision_token_id"]?.intValue
        self.visionFeatureSelectStrategy = object.string("vision_feature_select_strategy", default: "default") ?? "default"
        self.visionFeatureLayer = object.int("vision_feature_layer", default: -2)
        self.vocabSize = object.int("vocab_size", default: 32000)
        self.eosTokenIDs = object.intArray("eos_token_id")
    }

    private static func parseTextConfig(
        _ object: [String: JSONValue],
        family: QwenVLFamily
    ) throws -> QwenVLTextConfig {
        let numAttentionHeads = try object.requiredInt("num_attention_heads")
        let rawKVHeads = object["num_key_value_heads"]?.intValue
        let numKeyValueHeads: Int
        switch family {
        case .qwen2VL:
            numKeyValueHeads = rawKVHeads ?? 8
        case .qwen25VL:
            numKeyValueHeads = rawKVHeads ?? numAttentionHeads
        }

        let ropeScaling = object["rope_scaling"]
        try validateRopeScaling(ropeScaling)

        return QwenVLTextConfig(
            modelType: try object.requiredString("model_type"),
            hiddenSize: try object.requiredInt("hidden_size"),
            numHiddenLayers: try object.requiredInt("num_hidden_layers"),
            intermediateSize: try object.requiredInt("intermediate_size"),
            numAttentionHeads: numAttentionHeads,
            rmsNormEps: object.double("rms_norm_eps", default: 1e-6),
            vocabSize: try object.requiredInt("vocab_size"),
            numKeyValueHeads: numKeyValueHeads,
            maxPositionEmbeddings: object.int(
                "max_position_embeddings",
                default: family == .qwen2VL ? 40960 : 128000
            ),
            ropeTheta: object.double("rope_theta", default: 1_000_000.0),
            ropeTraditional: object.bool("rope_traditional", default: false),
            ropeScaling: ropeScaling,
            tieWordEmbeddings: object.bool("tie_word_embeddings", default: family == .qwen25VL),
            slidingWindow: family == .qwen2VL ? object.int("sliding_window", default: 32768) : object["sliding_window"]?.intValue,
            useSlidingWindow: family == .qwen2VL ? object.bool("use_sliding_window", default: false) : object["use_sliding_window"]?.boolValue,
            useCache: family == .qwen2VL ? object.bool("use_cache", default: true) : object["use_cache"]?.boolValue
        )
    }

    private static func parseVisionConfig(
        _ object: [String: JSONValue],
        family: QwenVLFamily
    ) -> QwenVLVisionConfig {
        QwenVLVisionConfig(
            modelType: object.string("model_type", default: family.rawValue) ?? family.rawValue,
            depth: object.int("depth", default: 32),
            embedDim: family == .qwen2VL ? object.int("embed_dim", default: 1280) : object["embed_dim"]?.intValue,
            hiddenSize: object.int("hidden_size", default: family == .qwen2VL ? 1536 : 1280),
            intermediateSize: family == .qwen25VL ? object.int("intermediate_size", default: 3420) : object["intermediate_size"]?.intValue,
            outHiddenSize: family == .qwen25VL ? object.int("out_hidden_size", default: 1536) : object["out_hidden_size"]?.intValue,
            numHeads: object.int("num_heads", default: 16),
            imageSize: object.int("image_size", default: 384),
            patchSize: object.int("patch_size", default: 14),
            vocabSize: object.int("vocab_size", default: 32000),
            mlpRatio: object.double("mlp_ratio", default: 4.0),
            inChannels: object.int("in_channels", default: 3),
            layerNormEps: object.double("layer_norm_eps", default: 1e-6),
            spatialPatchSize: object.int("spatial_patch_size", default: 14),
            spatialMergeSize: object.int("spatial_merge_size", default: 2),
            temporalPatchSize: object.int("temporal_patch_size", default: 2),
            tokensPerSecond: family == .qwen25VL ? object.int("tokens_per_second", default: 2) : object["tokens_per_second"]?.intValue,
            windowSize: family == .qwen25VL ? object.int("window_size", default: 112) : object["window_size"]?.intValue,
            fullattBlockIndexes: object.intArray("fullatt_block_indexes") ?? (family == .qwen25VL ? [7, 15, 23, 31] : [])
        )
    }

    private static func validateRopeScaling(_ value: JSONValue?) throws {
        guard let value, value != .null else {
            return
        }
        guard let object = value.objectValue else {
            throw QwenVLConfigError.invalidRopeScaling("expected object")
        }
        guard object["mrope_section"] != nil, let type = object["type"]?.stringValue else {
            throw QwenVLConfigError.invalidRopeScaling("must contain mrope_section and type")
        }
        guard type == "mrope" || type == "default" else {
            throw QwenVLConfigError.invalidRopeScaling("type must be 'mrope' or 'default'")
        }
    }
}
