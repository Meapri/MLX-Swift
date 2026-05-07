import Foundation

public enum JSONValue: Codable, Equatable, Sendable {
    case object([String: JSONValue])
    case array([JSONValue])
    case string(String)
    case number(Double)
    case bool(Bool)
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .object(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }

    public subscript(key: String) -> JSONValue? {
        if case .object(let object) = self {
            return object[key]
        }
        return nil
    }

    public var stringValue: String? {
        if case .string(let value) = self {
            return value
        }
        return nil
    }

    public var intValue: Int? {
        if case .number(let value) = self {
            return Int(value)
        }
        return nil
    }

    public var doubleValue: Double? {
        if case .number(let value) = self {
            return value
        }
        return nil
    }

    public var boolValue: Bool? {
        if case .bool(let value) = self {
            return value
        }
        return nil
    }

    public var objectValue: [String: JSONValue]? {
        if case .object(let value) = self {
            return value
        }
        return nil
    }

    public var arrayValue: [JSONValue]? {
        if case .array(let value) = self {
            return value
        }
        return nil
    }
}

public extension Dictionary where Key == String, Value == JSONValue {
    func string(_ key: String, default defaultValue: String? = nil) -> String? {
        self[key]?.stringValue ?? defaultValue
    }

    func requiredString(_ key: String) throws -> String {
        guard let value = self[key]?.stringValue else {
            throw ModelStoreError.invalidConfig("missing string field '\(key)'")
        }
        return value
    }

    func int(_ key: String, default defaultValue: Int) -> Int {
        self[key]?.intValue ?? defaultValue
    }

    func requiredInt(_ key: String) throws -> Int {
        guard let value = self[key]?.intValue else {
            throw ModelStoreError.invalidConfig("missing integer field '\(key)'")
        }
        return value
    }

    func double(_ key: String, default defaultValue: Double) -> Double {
        self[key]?.doubleValue ?? defaultValue
    }

    func bool(_ key: String, default defaultValue: Bool) -> Bool {
        self[key]?.boolValue ?? defaultValue
    }

    func intArray(_ key: String) -> [Int]? {
        self[key]?.arrayValue?.compactMap(\.intValue)
    }
}
