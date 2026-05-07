import Foundation

public struct WeightCatalogTensor: Codable, Equatable, Sendable {
    public let originalKey: String
    public let sanitizedKey: String
    public let shardName: String
    public let dtype: String
    public let shape: [Int]
    public let dataOffsets: [Int64]
    public let role: QwenVLWeightRole?
    public let indexed: Bool

    public init(
        originalKey: String,
        sanitizedKey: String,
        shardName: String,
        dtype: String,
        shape: [Int],
        dataOffsets: [Int64],
        role: QwenVLWeightRole?,
        indexed: Bool
    ) {
        self.originalKey = originalKey
        self.sanitizedKey = sanitizedKey
        self.shardName = shardName
        self.dtype = dtype
        self.shape = shape
        self.dataOffsets = dataOffsets
        self.role = role
        self.indexed = indexed
    }

    public var elementCount: Int {
        shape.reduce(1, *)
    }
}

public struct WeightCatalogShardError: Codable, Equatable, Sendable {
    public let shardName: String
    public let error: String
}

public struct WeightCatalog: Codable, Equatable, Sendable {
    public let tensors: [WeightCatalogTensor]
    public let missingIndexEntries: [String]
    public let unreadableShards: [WeightCatalogShardError]
    public let duplicateOriginalKeys: [String]
    public let duplicateSanitizedKeys: [String]

    public init(
        tensors: [WeightCatalogTensor],
        missingIndexEntries: [String],
        unreadableShards: [WeightCatalogShardError]
    ) {
        self.tensors = tensors.sorted { $0.sanitizedKey < $1.sanitizedKey }
        self.missingIndexEntries = missingIndexEntries.sorted()
        self.unreadableShards = unreadableShards.sorted { $0.shardName < $1.shardName }
        self.duplicateOriginalKeys = Self.duplicates(tensors.map(\.originalKey))
        self.duplicateSanitizedKeys = Self.duplicates(tensors.map(\.sanitizedKey))
    }

    public var tensorCount: Int {
        tensors.count
    }

    public var indexedTensorCount: Int {
        tensors.filter(\.indexed).count
    }

    public var unindexedTensorCount: Int {
        tensors.filter { !$0.indexed }.count
    }

    public var dtypeCounts: [String: Int] {
        Dictionary(grouping: tensors, by: \.dtype)
            .mapValues(\.count)
    }

    public var roleCounts: [String: Int] {
        Dictionary(grouping: tensors.compactMap(\.role), by: \.rawValue)
            .mapValues(\.count)
    }

    private static func duplicates<T: Hashable & Comparable>(_ values: [T]) -> [T] {
        var seen = Set<T>()
        var duplicated = Set<T>()
        for value in values {
            if !seen.insert(value).inserted {
                duplicated.insert(value)
            }
        }
        return duplicated.sorted()
    }
}

public struct WeightCatalogBuilder {
    public let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    public func catalog(for descriptor: ModelDescriptor) -> WeightCatalog {
        let modelURL = URL(fileURLWithPath: descriptor.path, isDirectory: true)
        let indexedShardByTensor = descriptor.weightIndex?.weightMap ?? [:]
        var remainingIndexEntries = Set(indexedShardByTensor.keys)
        var tensors: [WeightCatalogTensor] = []
        var unreadableShards: [WeightCatalogShardError] = []

        for weightFile in descriptor.weightFiles {
            let shardURL = modelURL.appendingPathComponent(weightFile.name)
            do {
                for tensor in try SafetensorsMetadataReader.readTensorMetadata(at: shardURL) {
                    let indexedShardName = indexedShardByTensor[tensor.name]
                    let indexed = indexedShardName == nil || indexedShardName == weightFile.name
                    remainingIndexEntries.remove(tensor.name)
                    let sanitized = sanitize(key: tensor.name, descriptor: descriptor)
                    tensors.append(
                        WeightCatalogTensor(
                            originalKey: tensor.name,
                            sanitizedKey: sanitized,
                            shardName: weightFile.name,
                            dtype: tensor.dtype,
                            shape: tensor.shape,
                            dataOffsets: tensor.dataOffsets,
                            role: role(for: sanitized, descriptor: descriptor),
                            indexed: indexed
                        )
                    )
                }
            } catch {
                unreadableShards.append(
                    WeightCatalogShardError(
                        shardName: weightFile.name,
                        error: String(describing: error)
                    )
                )
            }
        }

        return WeightCatalog(
            tensors: tensors,
            missingIndexEntries: Array(remainingIndexEntries),
            unreadableShards: unreadableShards
        )
    }

    private func sanitize(key: String, descriptor: ModelDescriptor) -> String {
        if descriptor.canonicalModelType == "qwen2_vl" || descriptor.canonicalModelType == "qwen2_5_vl" {
            return QwenVLWeightSanitizer.sanitizeKey(key)
        }
        return key
    }

    private func role(for sanitizedKey: String, descriptor: ModelDescriptor) -> QwenVLWeightRole? {
        if descriptor.canonicalModelType == "qwen2_vl" || descriptor.canonicalModelType == "qwen2_5_vl" {
            return QwenVLWeightSanitizer.classify(sanitizedKey)
        }
        return nil
    }
}
