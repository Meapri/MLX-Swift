import Foundation

public enum QwenVLWeightRole: String, Codable, Equatable, Sendable {
    case languageModel
    case languageHead
    case visionTower
    case unknown
}

public struct QwenVLWeightEntry: Codable, Equatable, Sendable {
    public let originalKey: String
    public let sanitizedKey: String
    public let shardName: String?
    public let role: QwenVLWeightRole

    public init(originalKey: String, sanitizedKey: String, shardName: String? = nil, role: QwenVLWeightRole) {
        self.originalKey = originalKey
        self.sanitizedKey = sanitizedKey
        self.shardName = shardName
        self.role = role
    }
}

public struct QwenVLWeightKeySanitization: Codable, Equatable, Sendable {
    public let originalKey: String
    public let sanitizedKey: String
    public let role: QwenVLWeightRole

    public init(originalKey: String, sanitizedKey: String, role: QwenVLWeightRole) {
        self.originalKey = originalKey
        self.sanitizedKey = sanitizedKey
        self.role = role
    }
}

public struct QwenVLWeightReport: Codable, Equatable, Sendable {
    public let entries: [QwenVLWeightEntry]

    public init(entries: [QwenVLWeightEntry]) {
        self.entries = entries.sorted { $0.sanitizedKey < $1.sanitizedKey }
    }

    public var languageModelCount: Int {
        entries.filter { $0.role == .languageModel }.count
    }

    public var languageHeadCount: Int {
        entries.filter { $0.role == .languageHead }.count
    }

    public var visionTowerCount: Int {
        entries.filter { $0.role == .visionTower }.count
    }

    public var unknownCount: Int {
        entries.filter { $0.role == .unknown }.count
    }
}

public enum QwenVLWeightSanitizer {
    public static func sanitize(_ key: String) -> QwenVLWeightKeySanitization {
        let sanitized = sanitizeKey(key)
        return QwenVLWeightKeySanitization(
            originalKey: key,
            sanitizedKey: sanitized,
            role: classify(sanitized)
        )
    }

    public static func sanitizeKey(_ key: String) -> String {
        var sanitized = key
        if !sanitized.contains("vision_tower") {
            sanitized = sanitized.replacingOccurrences(of: "visual", with: "vision_tower")
        }
        if !sanitized.contains("language_model") {
            if sanitized.contains("model") {
                sanitized = sanitized.replacingOccurrences(of: "model", with: "language_model.model")
            } else if sanitized.contains("lm_head") {
                sanitized = sanitized.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }
        }
        return sanitized
    }

    public static func classify(_ sanitizedKey: String) -> QwenVLWeightRole {
        if sanitizedKey.hasPrefix("language_model.model.") {
            return .languageModel
        }
        if sanitizedKey.hasPrefix("language_model.lm_head") {
            return .languageHead
        }
        if sanitizedKey.hasPrefix("vision_tower.") {
            return .visionTower
        }
        return .unknown
    }

    public static func report(weightMap: [String: String]) -> QwenVLWeightReport {
        QwenVLWeightReport(
            entries: weightMap.map { key, shard in
                let sanitized = sanitizeKey(key)
                return QwenVLWeightEntry(
                    originalKey: key,
                    sanitizedKey: sanitized,
                    shardName: shard,
                    role: classify(sanitized)
                )
            }
        )
    }

    public static func report(descriptor: ModelDescriptor) -> QwenVLWeightReport? {
        guard let weightMap = descriptor.weightIndex?.weightMap else {
            return nil
        }
        return report(weightMap: weightMap)
    }
}
