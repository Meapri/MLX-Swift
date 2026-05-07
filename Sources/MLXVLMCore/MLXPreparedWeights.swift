import Foundation

public struct MLXWeightPreparationOptions: Codable, Equatable, Sendable {
    public let tensorNames: [String]
    public let maxTensorCount: Int?
    public let maxTotalBytes: Int64?
    public let skipTensorPayloads: Bool

    public init(
        tensorNames: [String] = [],
        maxTensorCount: Int? = nil,
        maxTotalBytes: Int64? = nil,
        skipTensorPayloads: Bool = false
    ) {
        self.tensorNames = tensorNames
        self.maxTensorCount = maxTensorCount
        self.maxTotalBytes = maxTotalBytes
        self.skipTensorPayloads = skipTensorPayloads
    }
}

public struct MLXPreparedWeightTensor: Equatable, Sendable {
    public let load: MLXWeightTensorLoad
    public let payload: WeightTensorPayload

    public init(load: MLXWeightTensorLoad, payload: WeightTensorPayload) {
        self.load = load
        self.payload = payload
    }
}

public struct MLXPreparedWeightTensorSummary: Codable, Equatable, Sendable {
    public let sanitizedKey: String
    public let originalKey: String
    public let role: QwenVLWeightRole?
    public let shardName: String
    public let safetensorsDType: String
    public let mlxDType: String
    public let shape: [Int]
    public let byteCount: Int
    public let checksum: Int64
    public let requiresDTypeConversion: Bool

    public init(tensor: MLXPreparedWeightTensor) {
        self.sanitizedKey = tensor.load.sanitizedKey
        self.originalKey = tensor.load.originalKey
        self.role = tensor.load.role
        self.shardName = tensor.load.shardName
        self.safetensorsDType = tensor.load.safetensorsDType
        self.mlxDType = tensor.load.mlxDType ?? "unknown"
        self.shape = tensor.load.shape
        self.byteCount = tensor.payload.byteCount
        self.checksum = tensor.payload.checksum
        self.requiresDTypeConversion = tensor.load.requiresDTypeConversion
    }
}

public struct MLXPreparedWeightBundle: Equatable, Sendable {
    public let modelID: String
    public let canonicalModelType: String
    public let tensors: [MLXPreparedWeightTensor]

    public init(
        modelID: String,
        canonicalModelType: String,
        tensors: [MLXPreparedWeightTensor]
    ) {
        self.modelID = modelID
        self.canonicalModelType = canonicalModelType
        self.tensors = tensors
    }

    public var summary: MLXPreparedWeightBundleSummary {
        MLXPreparedWeightBundleSummary(bundle: self)
    }
}

public struct MLXPreparedWeightBundleSummary: Codable, Equatable, Sendable {
    public let modelID: String
    public let canonicalModelType: String
    public let tensorCount: Int
    public let totalByteCount: Int64
    public let safetensorsDTypeCounts: [String: Int]
    public let mlxDTypeCounts: [String: Int]
    public let tensors: [MLXPreparedWeightTensorSummary]

    public init(bundle: MLXPreparedWeightBundle) {
        self.modelID = bundle.modelID
        self.canonicalModelType = bundle.canonicalModelType
        self.tensorCount = bundle.tensors.count
        self.totalByteCount = bundle.tensors.reduce(Int64(0)) { $0 + Int64($1.payload.byteCount) }
        self.safetensorsDTypeCounts = Dictionary(grouping: bundle.tensors.map(\.load.safetensorsDType), by: { $0 })
            .mapValues(\.count)
        self.mlxDTypeCounts = Dictionary(grouping: bundle.tensors.compactMap(\.load.mlxDType), by: { $0 })
            .mapValues(\.count)
        self.tensors = bundle.tensors.map(MLXPreparedWeightTensorSummary.init(tensor:))
    }
}

public enum MLXWeightPreparationError: Error, CustomStringConvertible, Equatable {
    case requestedTensorNotFound(String)
    case tensorNotLoadable(String, String)
    case tensorHandleNotFound(String)
    case maximumTensorCountExceeded(limit: Int, actual: Int)
    case maximumTotalBytesExceeded(limit: Int64, actual: Int64)

    public var description: String {
        switch self {
        case .requestedTensorNotFound(let name):
            return "Requested tensor was not found in the MLX weight load plan: \(name)"
        case .tensorNotLoadable(let key, let reason):
            return "Tensor cannot be prepared for MLX loading: \(key). \(reason)"
        case .tensorHandleNotFound(let key):
            return "Tensor data handle was not found for MLX load plan entry: \(key)"
        case .maximumTensorCountExceeded(let limit, let actual):
            return "Preparing \(actual) tensors would exceed max tensor count \(limit)."
        case .maximumTotalBytesExceeded(let limit, let actual):
            return "Preparing \(actual) bytes would exceed max total bytes \(limit)."
        }
    }
}

public struct MLXWeightPreparer {
    public let dataCatalogBuilder: WeightDataCatalogBuilder

    public init(dataCatalogBuilder: WeightDataCatalogBuilder = WeightDataCatalogBuilder()) {
        self.dataCatalogBuilder = dataCatalogBuilder
    }

    public func prepare(
        descriptor: ModelDescriptor,
        options: MLXWeightPreparationOptions = MLXWeightPreparationOptions()
    ) throws -> MLXPreparedWeightBundle {
        let loadPlan = MLXWeightLoadPlan(descriptor: descriptor)
        if options.skipTensorPayloads {
            return MLXPreparedWeightBundle(
                modelID: descriptor.id,
                canonicalModelType: descriptor.canonicalModelType,
                tensors: []
            )
        }
        let dataCatalog = dataCatalogBuilder.catalog(for: descriptor)
        let selectedLoads = try selectLoads(from: loadPlan, names: options.tensorNames)
        if let maxTensorCount = options.maxTensorCount, selectedLoads.count > maxTensorCount {
            throw MLXWeightPreparationError.maximumTensorCountExceeded(
                limit: maxTensorCount,
                actual: selectedLoads.count
            )
        }

        var prepared: [MLXPreparedWeightTensor] = []
        var runningBytes: Int64 = 0
        for load in selectedLoads {
            guard load.loadableAsMLXArray else {
                throw MLXWeightPreparationError.tensorNotLoadable(
                    load.sanitizedKey,
                    load.error ?? "No loadable MLX dtype or byte range is available."
                )
            }
            guard let handle = dataCatalog.tensors.first(where: {
                $0.sanitizedKey == load.sanitizedKey &&
                    $0.originalKey == load.originalKey &&
                    $0.shardName == load.shardName
            }) else {
                throw MLXWeightPreparationError.tensorHandleNotFound(load.sanitizedKey)
            }
            let nextBytes = runningBytes + (load.byteCount ?? 0)
            if let maxTotalBytes = options.maxTotalBytes, nextBytes > maxTotalBytes {
                throw MLXWeightPreparationError.maximumTotalBytesExceeded(
                    limit: maxTotalBytes,
                    actual: nextBytes
                )
            }
            let payload = try dataCatalogBuilder.readTensorPayload(handle: handle, descriptor: descriptor)
            runningBytes += Int64(payload.byteCount)
            prepared.append(MLXPreparedWeightTensor(load: load, payload: payload))
        }

        return MLXPreparedWeightBundle(
            modelID: descriptor.id,
            canonicalModelType: descriptor.canonicalModelType,
            tensors: prepared
        )
    }

    private func selectLoads(
        from loadPlan: MLXWeightLoadPlan,
        names: [String]
    ) throws -> [MLXWeightTensorLoad] {
        guard !names.isEmpty else {
            return loadPlan.tensors
        }

        return try names.map { name in
            guard let load = loadPlan.tensors.first(where: {
                $0.sanitizedKey == name || $0.originalKey == name
            }) else {
                throw MLXWeightPreparationError.requestedTensorNotFound(name)
            }
            return load
        }
    }
}
