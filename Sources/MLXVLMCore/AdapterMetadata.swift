import Foundation

public struct AdapterMetadata: Codable, Equatable, Sendable {
    public let hasAdapterConfig: Bool
    public let hasAdapterWeights: Bool
    public let configSource: String?
    public let weightsSource: String?
    public let rank: Int?
    public let alpha: Double?
    public let dropout: Double?
    public let rawConfig: [String: JSONValue]?
    public let weightMetadata: SafetensorsFileMetadata?
    public let isLoRA: Bool
    public let isLoadable: Bool
    public let warnings: [String]

    public init(
        hasAdapterConfig: Bool,
        hasAdapterWeights: Bool,
        configSource: String?,
        weightsSource: String?,
        rank: Int?,
        alpha: Double?,
        dropout: Double?,
        rawConfig: [String: JSONValue]?,
        weightMetadata: SafetensorsFileMetadata?,
        isLoRA: Bool,
        isLoadable: Bool,
        warnings: [String]
    ) {
        self.hasAdapterConfig = hasAdapterConfig
        self.hasAdapterWeights = hasAdapterWeights
        self.configSource = configSource
        self.weightsSource = weightsSource
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.rawConfig = rawConfig
        self.weightMetadata = weightMetadata
        self.isLoRA = isLoRA
        self.isLoadable = isLoadable
        self.warnings = warnings
    }
}

public struct AdapterMetadataLoader {
    public let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    public func load(from modelURL: URL) throws -> AdapterMetadata {
        let configURL = modelURL.appendingPathComponent("adapter_config.json")
        let weightsURL = modelURL.appendingPathComponent("adapters.safetensors")
        let hasConfig = fileManager.fileExists(atPath: configURL.path)
        let hasWeights = fileManager.fileExists(atPath: weightsURL.path)
        let config = try loadJSONObjectIfPresent(configURL)
        let weightMetadata = hasWeights ? SafetensorsMetadataReader.readFileMetadata(at: weightsURL) : nil
        let rank = config?["rank"]?.intValue
        let alpha = config?["alpha"]?.doubleValue
        let dropout = config?["dropout"]?.doubleValue
        var warnings: [String] = []

        if hasConfig, rank == nil {
            warnings.append("adapter_config.json is present but does not contain the Python mlx-vlm LoRA rank field.")
        }
        if hasConfig, !hasWeights {
            warnings.append("adapter_config.json is present but adapters.safetensors is missing.")
        }
        if hasWeights, !hasConfig {
            warnings.append("adapters.safetensors is present but adapter_config.json is missing.")
        }
        if let weightMetadata, !weightMetadata.isReadable {
            warnings.append("adapters.safetensors is not readable: \(weightMetadata.error ?? "unknown error").")
        }

        let isLoRA = rank != nil
        let isLoadable = hasConfig && hasWeights && isLoRA && (weightMetadata?.isReadable ?? false)

        return AdapterMetadata(
            hasAdapterConfig: hasConfig,
            hasAdapterWeights: hasWeights,
            configSource: hasConfig ? "adapter_config.json" : nil,
            weightsSource: hasWeights ? "adapters.safetensors" : nil,
            rank: rank,
            alpha: alpha,
            dropout: dropout,
            rawConfig: config,
            weightMetadata: weightMetadata,
            isLoRA: isLoRA,
            isLoadable: isLoadable,
            warnings: warnings
        )
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
}
