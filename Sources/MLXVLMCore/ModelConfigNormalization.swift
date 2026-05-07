import Foundation

public struct ModelConfigNormalizationPlan: Codable, Equatable, Sendable {
    public let rawModelType: String
    public let canonicalModelType: String
    public let usedLLMConfigAsTextConfig: Bool
    public let hasTextConfig: Bool
    public let hasVisionConfig: Bool
    public let hasAudioConfig: Bool
    public let insertedEmptyTextConfig: Bool
    public let insertedEmptyVisionConfig: Bool
    public let insertedEmptyAudioConfig: Bool
    public let textConfigSource: String?
    public let visionConfigSource: String?
    public let audioConfigSource: String?
    public let normalizedTopLevelKeys: [String]
    public let warnings: [String]
}

public struct ModelConfigNormalizer {
    public init() {}

    public func plan(
        config: JSONValue,
        rawModelType: String,
        canonicalModelType: String
    ) -> ModelConfigNormalizationPlan {
        let object = config.objectValue ?? [:]
        let hasTextConfig = object["text_config"] != nil
        let hasLLMConfig = object["llm_config"] != nil
        let hasVisionConfig = object["vision_config"] != nil
        let hasAudioConfig = object["audio_config"] != nil
        let usedLLMConfigAsTextConfig = !hasTextConfig && hasLLMConfig

        var normalizedKeys = Set(object.keys)
        if hasLLMConfig {
            normalizedKeys.remove("llm_config")
        }
        if usedLLMConfigAsTextConfig {
            normalizedKeys.insert("text_config")
        } else if !hasTextConfig {
            normalizedKeys.insert("text_config")
        }
        if !hasVisionConfig {
            normalizedKeys.insert("vision_config")
        }
        if !hasAudioConfig {
            normalizedKeys.insert("audio_config")
        }

        var warnings: [String] = []
        if usedLLMConfigAsTextConfig {
            warnings.append("llm_config will be normalized into text_config for backend loading.")
        }
        if !hasTextConfig && !hasLLMConfig {
            warnings.append("Missing text_config/llm_config; backend loading will supply an empty text_config dictionary.")
        }
        if !hasVisionConfig {
            warnings.append("Missing vision_config; backend loading will supply an empty vision_config dictionary.")
        }
        if !hasAudioConfig {
            warnings.append("Missing audio_config; backend loading will supply an empty audio_config dictionary.")
        }

        return ModelConfigNormalizationPlan(
            rawModelType: rawModelType,
            canonicalModelType: canonicalModelType,
            usedLLMConfigAsTextConfig: usedLLMConfigAsTextConfig,
            hasTextConfig: hasTextConfig,
            hasVisionConfig: hasVisionConfig,
            hasAudioConfig: hasAudioConfig,
            insertedEmptyTextConfig: !hasTextConfig && !hasLLMConfig,
            insertedEmptyVisionConfig: !hasVisionConfig,
            insertedEmptyAudioConfig: !hasAudioConfig,
            textConfigSource: hasTextConfig ? "text_config" : (hasLLMConfig ? "llm_config" : "empty"),
            visionConfigSource: hasVisionConfig ? "vision_config" : "empty",
            audioConfigSource: hasAudioConfig ? "audio_config" : "empty",
            normalizedTopLevelKeys: normalizedKeys.sorted(),
            warnings: warnings
        )
    }

    public func normalizedConfig(config: JSONValue) -> JSONValue {
        guard var object = config.objectValue else {
            return config
        }

        let llmConfig = object.removeValue(forKey: "llm_config")
        if object["text_config"] == nil {
            object["text_config"] = llmConfig ?? .object([:])
        }
        if object["vision_config"] == nil {
            object["vision_config"] = .object([:])
        }
        if object["audio_config"] == nil {
            object["audio_config"] = .object([:])
        }

        return .object(object)
    }
}
