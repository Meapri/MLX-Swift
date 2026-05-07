import Foundation

public struct ResponseFormatPlan: Codable, Equatable, Sendable {
    public let requested: Bool
    public let source: String?
    public let formatType: String
    public let requiresJSONMode: Bool
    public let requiresSchemaGuidance: Bool
    public let schemaPresent: Bool
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
        self.streamFraming = streamFraming
        self.backendMinimumFeatures = backendMinimumFeatures
        self.warnings = warnings
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
            let schemaPresent = object["schema"] != nil ||
                object["json_schema"]?["schema"] != nil
            return ResponseFormatPlan(
                requested: true,
                source: object["json_schema"] == nil ? "responses-text-format" : "openai-response-format",
                formatType: "json_schema",
                requiresJSONMode: true,
                requiresSchemaGuidance: true,
                schemaPresent: schemaPresent,
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
}
