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
    public let grammarPlan: JSONSchemaGrammarPlan?
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
        grammarPlan: JSONSchemaGrammarPlan? = nil,
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
        self.grammarPlan = grammarPlan
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

public struct JSONSchemaGrammarPlan: Codable, Equatable, Sendable {
    public let rootSymbol: String
    public let deterministicPropertyOrder: [String]
    public let structuralStateCount: Int
    public let maxContainerDepth: Int
    public let constrainedStringPaths: [String]
    public let constrainedNumberPaths: [String]
    public let constrainedBooleanPaths: [String]
    public let constrainedNullPaths: [String]
    public let constrainedArrayPaths: [String]
    public let constrainedObjectPaths: [String]
    public let enumLiteralPaths: [String]
    public let additionalPropertiesFalsePaths: [String]
    public let unsupportedReasons: [String]
    public let canCompileStructuralGrammar: Bool
    public let requiresBackendTokenDFA: Bool

    public init(
        rootSymbol: String,
        deterministicPropertyOrder: [String],
        structuralStateCount: Int,
        maxContainerDepth: Int,
        constrainedStringPaths: [String],
        constrainedNumberPaths: [String],
        constrainedBooleanPaths: [String],
        constrainedNullPaths: [String],
        constrainedArrayPaths: [String],
        constrainedObjectPaths: [String],
        enumLiteralPaths: [String],
        additionalPropertiesFalsePaths: [String],
        unsupportedReasons: [String],
        canCompileStructuralGrammar: Bool,
        requiresBackendTokenDFA: Bool
    ) {
        self.rootSymbol = rootSymbol
        self.deterministicPropertyOrder = deterministicPropertyOrder
        self.structuralStateCount = structuralStateCount
        self.maxContainerDepth = maxContainerDepth
        self.constrainedStringPaths = constrainedStringPaths
        self.constrainedNumberPaths = constrainedNumberPaths
        self.constrainedBooleanPaths = constrainedBooleanPaths
        self.constrainedNullPaths = constrainedNullPaths
        self.constrainedArrayPaths = constrainedArrayPaths
        self.constrainedObjectPaths = constrainedObjectPaths
        self.enumLiteralPaths = enumLiteralPaths
        self.additionalPropertiesFalsePaths = additionalPropertiesFalsePaths
        self.unsupportedReasons = unsupportedReasons
        self.canCompileStructuralGrammar = canCompileStructuralGrammar
        self.requiresBackendTokenDFA = requiresBackendTokenDFA
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

public struct StructuredOutputValidationReport: Codable, Equatable, Sendable {
    public let requested: Bool
    public let formatType: String
    public let validJSON: Bool
    public let schemaPresent: Bool
    public let schemaValid: Bool
    public let errors: [String]
    public let warnings: [String]

    public init(
        requested: Bool,
        formatType: String,
        validJSON: Bool,
        schemaPresent: Bool,
        schemaValid: Bool,
        errors: [String],
        warnings: [String]
    ) {
        self.requested = requested
        self.formatType = formatType
        self.validJSON = validJSON
        self.schemaPresent = schemaPresent
        self.schemaValid = schemaValid
        self.errors = errors
        self.warnings = warnings
    }
}

public struct StructuredOutputValidator {
    public init() {}

    public func validate(text: String, responseFormat: JSONValue?) -> StructuredOutputValidationReport? {
        guard let responseFormat else {
            return nil
        }
        let metadata = GenerationRequestMetadata(responseFormat: responseFormat)
        let plan = ResponseFormatPlanner().plan(metadata: metadata, stream: false)
        guard plan.requiresJSONMode else {
            return nil
        }
        let decoded: JSONValue
        do {
            decoded = try JSONDecoder().decode(JSONValue.self, from: Data(text.utf8))
        } catch {
            return StructuredOutputValidationReport(
                requested: plan.requested,
                formatType: plan.formatType,
                validJSON: false,
                schemaPresent: plan.schemaPresent,
                schemaValid: false,
                errors: ["Generated text is not valid JSON: \(error)"],
                warnings: plan.warnings
            )
        }

        var errors: [String] = []
        if plan.formatType == "json_object", decoded.objectValue == nil {
            errors.append("JSON object response_format requires an object root.")
        }
        if let schema = Self.schemaValue(from: responseFormat) {
            errors.append(contentsOf: validate(value: decoded, schema: schema, path: "$"))
        }
        return StructuredOutputValidationReport(
            requested: plan.requested,
            formatType: plan.formatType,
            validJSON: true,
            schemaPresent: plan.schemaPresent,
            schemaValid: errors.isEmpty,
            errors: errors,
            warnings: plan.warnings + (plan.schemaConstraints?.requiresFullGrammarDFA == true
                ? ["Validation is post-generation; full token-level grammar/DFA constraints are still required for strict decoding."]
                : [])
        )
    }

    private func validate(value: JSONValue, schema: JSONValue, path: String) -> [String] {
        guard let object = schema.objectValue else {
            return []
        }
        var errors: [String] = []
        let types = schemaTypes(object)
        if !types.isEmpty, !types.contains(jsonType(value)) {
            errors.append("\(path) expected \(types.joined(separator: "|")) but got \(jsonType(value)).")
            return errors
        }
        if let enumValues = object["enum"]?.arrayValue, !enumValues.isEmpty, !enumValues.contains(value) {
            errors.append("\(path) does not match allowed enum values.")
        }
        if let properties = object["properties"]?.objectValue {
            guard let valueObject = value.objectValue else {
                return errors
            }
            let required = object["required"]?.arrayValue?.compactMap(\.stringValue) ?? []
            for name in required where valueObject[name] == nil {
                errors.append("\(path).\(name) is required.")
            }
            for (name, propertySchema) in properties {
                if let propertyValue = valueObject[name] {
                    errors.append(contentsOf: validate(value: propertyValue, schema: propertySchema, path: "\(path).\(name)"))
                }
            }
            if object["additionalProperties"]?.boolValue == false {
                let allowed = Set(properties.keys)
                for name in valueObject.keys where !allowed.contains(name) {
                    errors.append("\(path).\(name) is not allowed by additionalProperties=false.")
                }
            }
        }
        if let itemSchema = object["items"], let array = value.arrayValue {
            for (index, item) in array.enumerated() {
                errors.append(contentsOf: validate(value: item, schema: itemSchema, path: "\(path)[\(index)]"))
            }
        }
        return errors
    }

    private static func schemaValue(from responseFormat: JSONValue) -> JSONValue? {
        guard let object = responseFormat.objectValue else {
            return nil
        }
        if let schema = object["schema"] {
            return schema
        }
        return object["json_schema"]?["schema"]
    }

    private func schemaTypes(_ object: [String: JSONValue]) -> [String] {
        if let type = object["type"]?.stringValue {
            return [type]
        }
        return object["type"]?.arrayValue?.compactMap(\.stringValue) ?? []
    }

    private func jsonType(_ value: JSONValue) -> String {
        switch value {
        case .object:
            return "object"
        case .array:
            return "array"
        case .string:
            return "string"
        case .number:
            return "number"
        case .bool:
            return "boolean"
        case .null:
            return "null"
        }
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
            grammarPlan: JSONSchemaGrammarPlanner().plan(schema: .object(object)),
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

public struct JSONSchemaGrammarPlanner {
    private struct Accumulator {
        var structuralStateCount = 0
        var maxContainerDepth = 0
        var constrainedStringPaths: Set<String> = []
        var constrainedNumberPaths: Set<String> = []
        var constrainedBooleanPaths: Set<String> = []
        var constrainedNullPaths: Set<String> = []
        var constrainedArrayPaths: Set<String> = []
        var constrainedObjectPaths: Set<String> = []
        var enumLiteralPaths: Set<String> = []
        var additionalPropertiesFalsePaths: Set<String> = []
        var unsupportedReasons: Set<String> = []
    }

    public init() {}

    public func plan(schema: JSONValue) -> JSONSchemaGrammarPlan? {
        guard let object = schema.objectValue else {
            return nil
        }
        var accumulator = Accumulator()
        walk(object, path: "$", depth: 0, accumulator: &accumulator)
        let rootTypes = types(from: object)
        let rootSymbol: String
        if rootTypes.contains("object") || object["properties"] != nil {
            rootSymbol = "object"
        } else if rootTypes.contains("array") {
            rootSymbol = "array"
        } else if let first = rootTypes.first {
            rootSymbol = first
        } else {
            rootSymbol = "json-value"
        }
        let deterministicPropertyOrder = Self.propertyOrder(from: object)
        return JSONSchemaGrammarPlan(
            rootSymbol: rootSymbol,
            deterministicPropertyOrder: deterministicPropertyOrder,
            structuralStateCount: accumulator.structuralStateCount,
            maxContainerDepth: accumulator.maxContainerDepth,
            constrainedStringPaths: Array(accumulator.constrainedStringPaths).sorted(),
            constrainedNumberPaths: Array(accumulator.constrainedNumberPaths).sorted(),
            constrainedBooleanPaths: Array(accumulator.constrainedBooleanPaths).sorted(),
            constrainedNullPaths: Array(accumulator.constrainedNullPaths).sorted(),
            constrainedArrayPaths: Array(accumulator.constrainedArrayPaths).sorted(),
            constrainedObjectPaths: Array(accumulator.constrainedObjectPaths).sorted(),
            enumLiteralPaths: Array(accumulator.enumLiteralPaths).sorted(),
            additionalPropertiesFalsePaths: Array(accumulator.additionalPropertiesFalsePaths).sorted(),
            unsupportedReasons: Array(accumulator.unsupportedReasons).sorted(),
            canCompileStructuralGrammar: accumulator.unsupportedReasons.isEmpty,
            requiresBackendTokenDFA: true
        )
    }

    private func walk(
        _ object: [String: JSONValue],
        path: String,
        depth: Int,
        accumulator: inout Accumulator
    ) {
        accumulator.structuralStateCount += 1
        accumulator.maxContainerDepth = max(accumulator.maxContainerDepth, depth)
        let currentTypes = types(from: object)
        if object["enum"]?.arrayValue?.isEmpty == false {
            accumulator.enumLiteralPaths.insert(path)
        }
        if object["additionalProperties"]?.boolValue == false {
            accumulator.additionalPropertiesFalsePaths.insert(path)
        }
        if currentTypes.isEmpty, object["properties"] == nil, object["items"] == nil, object["enum"] == nil {
            accumulator.unsupportedReasons.insert("\(path) has no explicit type, enum, properties, or items.")
        }
        for type in currentTypes {
            switch type {
            case "object":
                accumulator.constrainedObjectPaths.insert(path)
            case "array":
                accumulator.constrainedArrayPaths.insert(path)
            case "string":
                accumulator.constrainedStringPaths.insert(path)
            case "integer", "number":
                accumulator.constrainedNumberPaths.insert(path)
            case "boolean":
                accumulator.constrainedBooleanPaths.insert(path)
            case "null":
                accumulator.constrainedNullPaths.insert(path)
            default:
                accumulator.unsupportedReasons.insert("\(path) uses unsupported JSON Schema type \(type).")
            }
        }

        if let properties = object["properties"]?.objectValue {
            accumulator.constrainedObjectPaths.insert(path)
            for key in Self.propertyOrder(from: object) {
                guard let child = properties[key]?.objectValue else {
                    accumulator.unsupportedReasons.insert("\(path).\(key) property schema must be an object.")
                    continue
                }
                walk(child, path: "\(path).\(key)", depth: depth + 1, accumulator: &accumulator)
            }
        }

        if let items = object["items"]?.objectValue {
            accumulator.constrainedArrayPaths.insert(path)
            walk(items, path: "\(path)[]", depth: depth + 1, accumulator: &accumulator)
        } else if currentTypes.contains("array") {
            accumulator.unsupportedReasons.insert("\(path) array schema is missing an object items schema.")
        }
    }

    private func types(from object: [String: JSONValue]) -> [String] {
        if let type = object["type"]?.stringValue {
            return [type]
        }
        if let types = object["type"]?.arrayValue?.compactMap(\.stringValue), !types.isEmpty {
            return types
        }
        if object["properties"] != nil {
            return ["object"]
        }
        if object["items"] != nil {
            return ["array"]
        }
        return []
    }

    private static func propertyOrder(from object: [String: JSONValue]) -> [String] {
        let required = object["required"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let properties = object["properties"]?.objectValue ?? [:]
        let optional = properties.keys.filter { !required.contains($0) }.sorted()
        return required + optional
    }
}
