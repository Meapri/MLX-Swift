import Foundation

public struct ResponseFormatPlan: Codable, Equatable, Sendable {
    public let requested: Bool
    public let source: String?
    public let formatType: String
    public let requiresJSONMode: Bool
    public let requiresSchemaGuidance: Bool
    public let schemaPresent: Bool
    public let schemaConstraints: JSONSchemaConstraintPlan?
    public let streamFraming: String
    public let backendMinimumFeatures: [String]
    public let warnings: [String]

    public init(
        requested: Bool,
        source: String?,
        formatType: String,
        requiresJSONMode: Bool,
        requiresSchemaGuidance: Bool,
        schemaPresent: Bool,
        schemaConstraints: JSONSchemaConstraintPlan? = nil,
        streamFraming: String,
        backendMinimumFeatures: [String],
        warnings: [String]
    ) {
        self.requested = requested
        self.source = source
        self.formatType = formatType
        self.requiresJSONMode = requiresJSONMode
        self.requiresSchemaGuidance = requiresSchemaGuidance
        self.schemaPresent = schemaPresent
        self.schemaConstraints = schemaConstraints
        self.streamFraming = streamFraming
        self.backendMinimumFeatures = backendMinimumFeatures
        self.warnings = warnings
    }
}

public struct JSONSchemaConstraintPlan: Codable, Equatable, Sendable {
    public let rootTypes: [String]
    public let requiredProperties: [String]
    public let properties: [JSONSchemaPropertyConstraint]
    public let enumPaths: [String]
    public let arrayPaths: [String]
    public let maxDepth: Int
    public let supportedKeywords: [String]
    public let unsupportedKeywords: [String]
    public let canConstrainRootToken: Bool
    public let canConstrainRequiredPropertyPrefix: Bool
    public let canUseDeterministicKeyOrder: Bool
    public let requiresFullGrammarDFA: Bool

    public init(
        rootTypes: [String],
        requiredProperties: [String],
        properties: [JSONSchemaPropertyConstraint],
        enumPaths: [String],
        arrayPaths: [String],
        maxDepth: Int,
        supportedKeywords: [String],
        unsupportedKeywords: [String],
        canConstrainRootToken: Bool,
        canConstrainRequiredPropertyPrefix: Bool,
        canUseDeterministicKeyOrder: Bool,
        requiresFullGrammarDFA: Bool
    ) {
        self.rootTypes = rootTypes
        self.requiredProperties = requiredProperties
        self.properties = properties
        self.enumPaths = enumPaths
        self.arrayPaths = arrayPaths
        self.maxDepth = maxDepth
        self.supportedKeywords = supportedKeywords
        self.unsupportedKeywords = unsupportedKeywords
        self.canConstrainRootToken = canConstrainRootToken
        self.canConstrainRequiredPropertyPrefix = canConstrainRequiredPropertyPrefix
        self.canUseDeterministicKeyOrder = canUseDeterministicKeyOrder
        self.requiresFullGrammarDFA = requiresFullGrammarDFA
    }
}

public struct JSONSchemaPropertyConstraint: Codable, Equatable, Sendable {
    public let path: String
    public let types: [String]
    public let required: Bool
    public let enumValues: [String]
    public let additionalPropertiesAllowed: Bool?

    public init(
        path: String,
        types: [String],
        required: Bool,
        enumValues: [String],
        additionalPropertiesAllowed: Bool?
    ) {
        self.path = path
        self.types = types
        self.required = required
        self.enumValues = enumValues
        self.additionalPropertiesAllowed = additionalPropertiesAllowed
    }
}

public struct ResponseFormatPlanner {
    public init() {}

    public func plan(metadata: GenerationRequestMetadata, stream: Bool) -> ResponseFormatPlan {
        let streamFraming = stream ? "api-native-stream" : "single-json-response"
        guard let responseFormat = metadata.responseFormat else {
            return ResponseFormatPlan(
                requested: false,
                source: nil,
                formatType: "text",
                requiresJSONMode: false,
                requiresSchemaGuidance: false,
                schemaPresent: false,
                streamFraming: streamFraming,
                backendMinimumFeatures: stream ? ["stream-framing"] : [],
                warnings: []
            )
        }

        if let string = responseFormat.stringValue {
            let normalized = string.lowercased()
            if normalized == "json" {
                return ResponseFormatPlan(
                    requested: true,
                    source: "ollama-format",
                    formatType: "json_object",
                    requiresJSONMode: true,
                    requiresSchemaGuidance: false,
                    schemaPresent: false,
                    streamFraming: streamFraming,
                    backendMinimumFeatures: features(jsonMode: true, schema: false, stream: stream),
                    warnings: []
                )
            }
            return unsupported(source: "string-format", type: normalized, streamFraming: streamFraming, stream: stream)
        }

        guard let object = responseFormat.objectValue else {
            return unsupported(source: "unknown-format", type: "unknown", streamFraming: streamFraming, stream: stream)
        }

        let type = object["type"]?.stringValue ?? (object["schema"] == nil ? "json_object" : "json_schema")
        switch type {
        case "text":
            return ResponseFormatPlan(
                requested: true,
                source: "response-format",
                formatType: "text",
                requiresJSONMode: false,
                requiresSchemaGuidance: false,
                schemaPresent: false,
                streamFraming: streamFraming,
                backendMinimumFeatures: stream ? ["stream-framing"] : [],
                warnings: []
            )
        case "json_object", "json":
            return ResponseFormatPlan(
                requested: true,
                source: "response-format",
                formatType: "json_object",
                requiresJSONMode: true,
                requiresSchemaGuidance: false,
                schemaPresent: false,
                streamFraming: streamFraming,
                backendMinimumFeatures: features(jsonMode: true, schema: false, stream: stream),
                warnings: []
            )
        case "json_schema":
            let schema = Self.schemaValue(from: object)
            let schemaPresent = schema != nil
            let constraints = schema.flatMap { JSONSchemaConstraintPlanner().plan(schema: $0) }
            return ResponseFormatPlan(
                requested: true,
                source: object["json_schema"] == nil ? "responses-text-format" : "openai-response-format",
                formatType: "json_schema",
                requiresJSONMode: true,
                requiresSchemaGuidance: true,
                schemaPresent: schemaPresent,
                schemaConstraints: constraints,
                streamFraming: streamFraming,
                backendMinimumFeatures: features(jsonMode: true, schema: true, stream: stream),
                warnings: schemaPresent ? [] : ["json_schema response_format is missing a schema field."]
            )
        default:
            return unsupported(source: "response-format", type: type, streamFraming: streamFraming, stream: stream)
        }
    }

    private func unsupported(
        source: String,
        type: String,
        streamFraming: String,
        stream: Bool
    ) -> ResponseFormatPlan {
        ResponseFormatPlan(
            requested: true,
            source: source,
            formatType: type,
            requiresJSONMode: false,
            requiresSchemaGuidance: false,
            schemaPresent: false,
            streamFraming: streamFraming,
            backendMinimumFeatures: stream ? ["stream-framing"] : [],
            warnings: ["Unsupported response format type: \(type)."]
        )
    }

    private func features(jsonMode: Bool, schema: Bool, stream: Bool) -> [String] {
        var result: [String] = []
        if jsonMode {
            result.append("json-output-validation")
        }
        if schema {
            result.append("json-schema-logits-processor")
        }
        if stream {
            result.append("stream-framing")
        }
        return result
    }

    private static func schemaValue(from object: [String: JSONValue]) -> JSONValue? {
        if let schema = object["schema"] {
            return schema
        }
        return object["json_schema"]?["schema"]
    }
}

public struct JSONSchemaConstraintPlanner {
    private static let supportedKeywords: Set<String> = [
        "type",
        "properties",
        "required",
        "enum",
        "items",
        "additionalProperties",
        "description",
        "title",
        "default",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
    ]

    public init() {}

    public func plan(schema: JSONValue) -> JSONSchemaConstraintPlan? {
        guard let object = schema.objectValue else {
            return nil
        }
        var properties: [JSONSchemaPropertyConstraint] = []
        var enumPaths: Set<String> = []
        var arrayPaths: Set<String> = []
        var unsupportedKeywords: Set<String> = []
        let rootTypes = types(from: object)
        let required = object["required"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let maxDepth = walk(
            object,
            path: "$",
            requiredProperties: Set(required),
            properties: &properties,
            enumPaths: &enumPaths,
            arrayPaths: &arrayPaths,
            unsupportedKeywords: &unsupportedKeywords,
            depth: 0
        )
        let canConstrainRootToken = rootTypes.contains("object") || rootTypes.contains("array")
        return JSONSchemaConstraintPlan(
            rootTypes: rootTypes,
            requiredProperties: required,
            properties: properties.sorted { $0.path < $1.path },
            enumPaths: Array(enumPaths).sorted(),
            arrayPaths: Array(arrayPaths).sorted(),
            maxDepth: maxDepth,
            supportedKeywords: Self.supportedKeywords.sorted(),
            unsupportedKeywords: Array(unsupportedKeywords).sorted(),
            canConstrainRootToken: canConstrainRootToken,
            canConstrainRequiredPropertyPrefix: rootTypes.contains("object") && !required.isEmpty,
            canUseDeterministicKeyOrder: rootTypes.contains("object") && !required.isEmpty,
            requiresFullGrammarDFA: true
        )
    }

    private func walk(
        _ object: [String: JSONValue],
        path: String,
        requiredProperties: Set<String>,
        properties: inout [JSONSchemaPropertyConstraint],
        enumPaths: inout Set<String>,
        arrayPaths: inout Set<String>,
        unsupportedKeywords: inout Set<String>,
        depth: Int
    ) -> Int {
        for key in object.keys where !Self.supportedKeywords.contains(key) {
            unsupportedKeywords.insert(key)
        }
        if object["enum"]?.arrayValue?.isEmpty == false {
            enumPaths.insert(path)
        }

        var maxDepth = depth
        if types(from: object).contains("array") {
            arrayPaths.insert(path)
        }

        if let childProperties = object["properties"]?.objectValue {
            for (name, value) in childProperties {
                guard let child = value.objectValue else {
                    continue
                }
                let childPath = "\(path).\(name)"
                let childTypes = types(from: child)
                let enumValues = child["enum"]?.arrayValue?.compactMap(Self.enumString) ?? []
                properties.append(
                    JSONSchemaPropertyConstraint(
                        path: childPath,
                        types: childTypes,
                        required: requiredProperties.contains(name),
                        enumValues: enumValues,
                        additionalPropertiesAllowed: child["additionalProperties"]?.boolValue
                    )
                )
                let childRequired = Set(child["required"]?.arrayValue?.compactMap(\.stringValue) ?? [])
                maxDepth = max(
                    maxDepth,
                    walk(
                        child,
                        path: childPath,
                        requiredProperties: childRequired,
                        properties: &properties,
                        enumPaths: &enumPaths,
                        arrayPaths: &arrayPaths,
                        unsupportedKeywords: &unsupportedKeywords,
                        depth: depth + 1
                    )
                )
            }
        }

        if let items = object["items"]?.objectValue {
            maxDepth = max(
                maxDepth,
                walk(
                    items,
                    path: "\(path)[]",
                    requiredProperties: Set(items["required"]?.arrayValue?.compactMap(\.stringValue) ?? []),
                    properties: &properties,
                    enumPaths: &enumPaths,
                    arrayPaths: &arrayPaths,
                    unsupportedKeywords: &unsupportedKeywords,
                    depth: depth + 1
                )
            )
        }

        return maxDepth
    }

    private func types(from object: [String: JSONValue]) -> [String] {
        if let type = object["type"]?.stringValue {
            return [type]
        }
        if let types = object["type"]?.arrayValue?.compactMap(\.stringValue), !types.isEmpty {
            return types
        }
        return object["properties"] != nil ? ["object"] : []
    }

    private static func enumString(_ value: JSONValue) -> String? {
        switch value {
        case .string(let string):
            return string
        case .number(let number):
            return String(number)
        case .bool(let bool):
            return String(bool)
        case .null:
            return "null"
        case .object, .array:
            return nil
        }
    }
}
