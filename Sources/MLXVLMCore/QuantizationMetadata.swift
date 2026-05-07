import Foundation

public struct QuantizationMetadata: Codable, Equatable, Sendable {
    public let source: String
    public let raw: [String: JSONValue]
    public let mode: String?
    public let bits: Double?
    public let groupSize: Int?
    public let strategy: String?
    public let format: String?
    public let isSymmetric: Bool?

    public init(
        source: String,
        raw: [String: JSONValue],
        mode: String?,
        bits: Double?,
        groupSize: Int?,
        strategy: String?,
        format: String?,
        isSymmetric: Bool?
    ) {
        self.source = source
        self.raw = raw
        self.mode = mode
        self.bits = bits
        self.groupSize = groupSize
        self.strategy = strategy
        self.format = format
        self.isSymmetric = isSymmetric
    }

    public var summary: String {
        var parts: [String] = []
        if let mode {
            parts.append(mode)
        }
        if let bits {
            parts.append("\(bits)-bit")
        }
        if let groupSize {
            parts.append("group=\(groupSize)")
        }
        if let strategy {
            parts.append(strategy)
        }
        if parts.isEmpty {
            return source
        }
        return parts.joined(separator: " ")
    }
}

public struct QuantizationMetadataLoader {
    public init() {}

    public func load(config: JSONValue) -> QuantizationMetadata? {
        if let object = config["quantization"]?.objectValue {
            return metadata(source: "quantization", object: object)
        }
        if let object = config["quantization_config"]?.objectValue {
            return metadata(source: "quantization_config", object: object)
        }
        return nil
    }

    private func metadata(source: String, object: [String: JSONValue]) -> QuantizationMetadata {
        QuantizationMetadata(
            source: source,
            raw: object,
            mode: firstString(object, keys: ["mode", "quant_method", "quantization_method", "type"]),
            bits: firstDouble(object, keys: ["bits", "num_bits", "weight_bits"]),
            groupSize: firstInt(object, keys: ["group_size", "groupSize", "q_group_size"]),
            strategy: firstString(object, keys: ["strategy", "quantization_strategy"]),
            format: firstString(object, keys: ["format", "checkpoint_format"]),
            isSymmetric: firstBool(object, keys: ["sym", "symmetric", "is_symmetric"])
        )
    }

    private func firstString(_ object: [String: JSONValue], keys: [String]) -> String? {
        for key in keys {
            if let value = object[key]?.stringValue {
                return value
            }
        }
        return nil
    }

    private func firstDouble(_ object: [String: JSONValue], keys: [String]) -> Double? {
        for key in keys {
            if let value = object[key]?.doubleValue {
                return value
            }
        }
        return nil
    }

    private func firstInt(_ object: [String: JSONValue], keys: [String]) -> Int? {
        for key in keys {
            if let value = object[key]?.intValue {
                return value
            }
        }
        return nil
    }

    private func firstBool(_ object: [String: JSONValue], keys: [String]) -> Bool? {
        for key in keys {
            if let value = object[key]?.boolValue {
                return value
            }
        }
        return nil
    }
}
