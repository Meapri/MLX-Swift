import Foundation

public enum ModelStoreError: Error, CustomStringConvertible, Equatable {
    case pathNotFound(String)
    case configNotFound(String)
    case invalidConfig(String)
    case unsupportedRemoteIdentifier(String)

    public var description: String {
        switch self {
        case .pathNotFound(let path):
            return "Model path does not exist: \(path)"
        case .configNotFound(let path):
            return "config.json was not found in model path: \(path)"
        case .invalidConfig(let reason):
            return "Invalid model config: \(reason)"
        case .unsupportedRemoteIdentifier(let id):
            return "Remote Hugging Face identifiers require an existing local Hugging Face cache entry in this Swift target: \(id)"
        }
    }
}

public struct WeightFile: Codable, Equatable, Sendable {
    public let name: String
    public let bytes: Int64
}

public struct WeightIndex: Codable, Equatable, Sendable {
    public let metadataTotalSize: Int64?
    public let tensorCount: Int
    public let shardNames: [String]
    public let weightMap: [String: String]
}

public struct ModelDescriptor: Codable, Equatable, Sendable {
    public let id: String
    public let path: String
    public let rawModelType: String
    public let canonicalModelType: String
    public let isKnownModelType: Bool
    public let hasDFlashConfig: Bool
    public let hasTextConfig: Bool
    public let hasVisionConfig: Bool
    public let hasAudioConfig: Bool
    public let configNormalization: ModelConfigNormalizationPlan
    public let hasTokenizerConfig: Bool
    public let hasChatTemplate: Bool
    public let hasQuantization: Bool
    public let quantizationMetadata: QuantizationMetadata?
    public let configVocabSize: Int?
    public let weightFiles: [WeightFile]
    public let weightIndex: WeightIndex?
    public let safetensorsMetadata: [SafetensorsFileMetadata]
    public let tokenizerMetadata: TokenizerMetadata
    public let processorMetadata: ProcessorMetadata
    public let adapterMetadata: AdapterMetadata

    public var totalWeightBytes: Int64 {
        weightFiles.reduce(0) { $0 + $1.bytes }
    }
}

public struct ModelStore {
    public let registry: ModelRegistry
    public let fileManager: FileManager
    public let cacheResolver: HuggingFaceCacheResolver

    public init(
        registry: ModelRegistry = .shared,
        fileManager: FileManager = .default,
        cacheResolver: HuggingFaceCacheResolver = HuggingFaceCacheResolver()
    ) {
        self.registry = registry
        self.fileManager = fileManager
        self.cacheResolver = cacheResolver
    }

    public func loadDescriptor(pathOrIdentifier: String) throws -> ModelDescriptor {
        let resolved = try resolveModelURL(pathOrIdentifier: pathOrIdentifier)
        return try loadDescriptor(modelURL: resolved.url, identifier: resolved.identifier)
    }

    public func loadDescriptor(modelURL: URL, identifier: String? = nil) throws -> ModelDescriptor {
        let config = try loadConfig(in: modelURL)
        guard let rawModelType = config["model_type"]?.stringValue else {
            throw ModelStoreError.invalidConfig("missing string field 'model_type'")
        }

        let hasDFlashConfig = config["dflash_config"] != nil
        let canonicalType = registry.canonicalModelType(
            for: rawModelType,
            dflashConfigPresent: hasDFlashConfig
        )

        let tokenizerConfigURL = modelURL.appendingPathComponent("tokenizer_config.json")
        let weightFiles = try safetensorFiles(in: modelURL)
        let weightIndex = try loadWeightIndex(in: modelURL)
        let safetensorsMetadata = weightFiles.map {
            SafetensorsMetadataReader.readFileMetadata(at: modelURL.appendingPathComponent($0.name))
        }
        let tokenizerMetadata = try TokenizerMetadataLoader(fileManager: fileManager)
            .load(from: modelURL)
            .mergingTokenIDs(from: config)
        let processorMetadata = ProcessorMetadataLoader(fileManager: fileManager).load(from: modelURL)
        let adapterMetadata = try AdapterMetadataLoader(fileManager: fileManager).load(from: modelURL)
        let quantizationMetadata = QuantizationMetadataLoader().load(config: config)
        let configNormalization = ModelConfigNormalizer().plan(
            config: config,
            rawModelType: rawModelType,
            canonicalModelType: canonicalType
        )

        return ModelDescriptor(
            id: identifier ?? modelURL.lastPathComponent,
            path: modelURL.path,
            rawModelType: rawModelType,
            canonicalModelType: canonicalType,
            isKnownModelType: registry.isSupported(canonicalType),
            hasDFlashConfig: hasDFlashConfig,
            hasTextConfig: config["text_config"] != nil || config["llm_config"] != nil,
            hasVisionConfig: config["vision_config"] != nil,
            hasAudioConfig: config["audio_config"] != nil,
            configNormalization: configNormalization,
            hasTokenizerConfig: fileManager.fileExists(atPath: tokenizerConfigURL.path),
            hasChatTemplate: Self.hasChatTemplate(config: config, tokenizerConfigURL: tokenizerConfigURL),
            hasQuantization: quantizationMetadata != nil,
            quantizationMetadata: quantizationMetadata,
            configVocabSize: Self.vocabSize(config: config),
            weightFiles: weightFiles,
            weightIndex: weightIndex,
            safetensorsMetadata: safetensorsMetadata,
            tokenizerMetadata: tokenizerMetadata,
            processorMetadata: processorMetadata,
            adapterMetadata: adapterMetadata
        )
    }

    public func loadConfig(in modelURL: URL) throws -> JSONValue {
        let configURL = modelURL.appendingPathComponent("config.json")
        guard fileManager.fileExists(atPath: configURL.path) else {
            throw ModelStoreError.configNotFound(modelURL.path)
        }

        let data = try Data(contentsOf: configURL)
        var config = try JSONDecoder().decode(JSONValue.self, from: data)
        try mergeGenerationConfig(in: modelURL, into: &config)
        return config
    }

    public func loadConfig(pathOrIdentifier: String) throws -> JSONValue {
        try loadConfig(in: resolveModelURL(pathOrIdentifier: pathOrIdentifier).url)
    }

    public func loadNormalizedConfig(in modelURL: URL) throws -> JSONValue {
        try ModelConfigNormalizer().normalizedConfig(config: loadConfig(in: modelURL))
    }

    public func loadNormalizedConfig(pathOrIdentifier: String) throws -> JSONValue {
        try loadNormalizedConfig(in: resolveModelURL(pathOrIdentifier: pathOrIdentifier).url)
    }

    public func descriptors(in directory: String) throws -> [ModelDescriptor] {
        let rootURL = URL(fileURLWithPath: directory).standardizedFileURL
        guard fileManager.fileExists(atPath: rootURL.path) else {
            throw ModelStoreError.pathNotFound(directory)
        }

        let children = try fileManager.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        return try children.compactMap { child in
            let values = try child.resourceValues(forKeys: [.isDirectoryKey])
            guard values.isDirectory == true else {
                return nil
            }
            return try? loadDescriptor(pathOrIdentifier: child.path)
        }.sorted { $0.id < $1.id }
    }

    private func safetensorFiles(in modelURL: URL) throws -> [WeightFile] {
        let children = try fileManager.contentsOfDirectory(
            at: modelURL,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey],
            options: [.skipsHiddenFiles]
        )

        return try children.compactMap { child in
            guard child.pathExtension == "safetensors",
                  child.lastPathComponent != "consolidated.safetensors",
                  child.lastPathComponent != "adapters.safetensors"
            else {
                return nil
            }
            let values = try child.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey])
            if values.isRegularFile == true {
                return WeightFile(name: child.lastPathComponent, bytes: Int64(values.fileSize ?? 0))
            }
            guard values.isSymbolicLink == true else {
                return nil
            }
            let resolvedValues = try child.resolvingSymlinksInPath().resourceValues(
                forKeys: [.fileSizeKey, .isRegularFileKey]
            )
            guard resolvedValues.isRegularFile == true else {
                return nil
            }
            return WeightFile(name: child.lastPathComponent, bytes: Int64(resolvedValues.fileSize ?? 0))
        }.sorted { $0.name < $1.name }
    }

    private func resolveModelURL(pathOrIdentifier: String) throws -> (url: URL, identifier: String?) {
        let modelURL = URL(fileURLWithPath: pathOrIdentifier).standardizedFileURL
        if fileManager.fileExists(atPath: modelURL.path) {
            return (modelURL, nil)
        }
        guard Self.looksLikeRemoteIdentifier(pathOrIdentifier) else {
            throw ModelStoreError.pathNotFound(pathOrIdentifier)
        }
        guard let cachedURL = cacheResolver.resolveModelDirectory(identifier: pathOrIdentifier) else {
            throw ModelStoreError.unsupportedRemoteIdentifier(pathOrIdentifier)
        }
        return (cachedURL, pathOrIdentifier)
    }

    private static func looksLikeRemoteIdentifier(_ value: String) -> Bool {
        value.contains("/") && !value.hasPrefix("/") && !value.hasPrefix("./") && !value.hasPrefix("../")
    }

    private func loadWeightIndex(in modelURL: URL) throws -> WeightIndex? {
        let indexURL = modelURL.appendingPathComponent("model.safetensors.index.json")
        guard fileManager.fileExists(atPath: indexURL.path) else {
            return nil
        }

        let data = try Data(contentsOf: indexURL)
        let json = try JSONDecoder().decode(JSONValue.self, from: data)
        guard let object = json.objectValue,
              let weightMapJSON = object["weight_map"]?.objectValue
        else {
            throw ModelStoreError.invalidConfig("model.safetensors.index.json must contain weight_map")
        }

        var weightMap: [String: String] = [:]
        for (tensorName, shardValue) in weightMapJSON {
            guard let shardName = shardValue.stringValue else {
                throw ModelStoreError.invalidConfig("weight_map entry for '\(tensorName)' must be a string")
            }
            weightMap[tensorName] = shardName
        }

        let totalSize = object["metadata"]?.objectValue?["total_size"]?.intValue
        let shardNames = Array(Set(weightMap.values)).sorted()
        return WeightIndex(
            metadataTotalSize: totalSize.map(Int64.init),
            tensorCount: weightMap.count,
            shardNames: shardNames,
            weightMap: weightMap
        )
    }

    private func mergeGenerationConfig(in modelURL: URL, into config: inout JSONValue) throws {
        let generationConfigURL = modelURL.appendingPathComponent("generation_config.json")
        guard fileManager.fileExists(atPath: generationConfigURL.path),
              case .object(var configObject) = config
        else {
            return
        }

        let data = try Data(contentsOf: generationConfigURL)
        guard let generationConfig = try? JSONDecoder().decode(JSONValue.self, from: data),
              let generationObject = generationConfig.objectValue,
              let eosTokenID = generationObject["eos_token_id"]
        else {
            return
        }

        configObject["eos_token_id"] = eosTokenID
        config = .object(configObject)
    }

    private static func hasChatTemplate(config: JSONValue, tokenizerConfigURL: URL) -> Bool {
        if config["chat_template"]?.stringValue != nil {
            return true
        }
        let modelURL = tokenizerConfigURL.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: modelURL.appendingPathComponent("chat_template.json").path) ||
            FileManager.default.fileExists(atPath: modelURL.appendingPathComponent("chat_template.jinja").path)
        {
            return true
        }
        guard let data = try? Data(contentsOf: tokenizerConfigURL),
              let tokenizerConfig = try? JSONDecoder().decode(JSONValue.self, from: data)
        else {
            return false
        }
        return tokenizerConfig["chat_template"]?.stringValue != nil
    }

    private static func vocabSize(config: JSONValue) -> Int? {
        config["vocab_size"]?.intValue
            ?? config["text_config"]?.objectValue?["vocab_size"]?.intValue
            ?? config["llm_config"]?.objectValue?["vocab_size"]?.intValue
    }
}
