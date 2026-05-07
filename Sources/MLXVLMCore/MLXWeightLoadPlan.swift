import Foundation

public struct MLXWeightTensorLoad: Codable, Equatable, Sendable {
    public let sanitizedKey: String
    public let originalKey: String
    public let role: QwenVLWeightRole?
    public let shardName: String
    public let safetensorsDType: String
    public let mlxDType: String?
    public let shape: [Int]
    public let elementCount: Int
    public let byteCount: Int64?
    public let absoluteDataOffsets: [Int64]?
    public let loadableAsMLXArray: Bool
    public let requiresDTypeConversion: Bool
    public let error: String?

    public init(handle: WeightTensorDataHandle, role: QwenVLWeightRole?) {
        self.sanitizedKey = handle.sanitizedKey
        self.originalKey = handle.originalKey
        self.role = role
        self.shardName = handle.shardName
        self.safetensorsDType = handle.dtype
        self.mlxDType = Self.mlxDType(forSafetensorsDType: handle.dtype)
        self.shape = handle.shape
        self.elementCount = handle.elementCount
        self.byteCount = handle.byteCount
        self.absoluteDataOffsets = handle.absoluteDataOffsets
        self.loadableAsMLXArray = handle.isReadable && self.mlxDType != nil
        self.requiresDTypeConversion = Self.requiresDTypeConversion(handle.dtype)
        if !handle.isReadable {
            self.error = handle.error
        } else if self.mlxDType == nil {
            self.error = "No MLX dtype mapping is defined for safetensors dtype \(handle.dtype)."
        } else {
            self.error = nil
        }
    }

    public static func mlxDType(forSafetensorsDType dtype: String) -> String? {
        switch dtype.uppercased() {
        case "BOOL":
            return "bool"
        case "I8":
            return "int8"
        case "U8":
            return "uint8"
        case "I16":
            return "int16"
        case "U16":
            return "uint16"
        case "I32":
            return "int32"
        case "U32":
            return "uint32"
        case "I64":
            return "int64"
        case "U64":
            return "uint64"
        case "F16":
            return "float16"
        case "BF16":
            return "bfloat16"
        case "F32":
            return "float32"
        case "F64":
            return "float64"
        default:
            return nil
        }
    }

    private static func requiresDTypeConversion(_ dtype: String) -> Bool {
        switch dtype.uppercased() {
        case "F8_E5M2", "F8_E4M3":
            return true
        default:
            return false
        }
    }
}

public struct MLXWeightLoadPlan: Codable, Equatable, Sendable {
    public let modelID: String
    public let canonicalModelType: String
    public let tensors: [MLXWeightTensorLoad]
    public let tensorCount: Int
    public let loadableTensorCount: Int
    public let unreadableTensorCount: Int
    public let unsupportedDTypeKeys: [String]
    public let dtypeCounts: [String: Int]
    public let mlxDTypeCounts: [String: Int]
    public let totalLoadableBytes: Int64
    public let canLoadAllTensorsAsMLXArrays: Bool

    public init(descriptor: ModelDescriptor) {
        let weightCatalog = WeightCatalogBuilder().catalog(for: descriptor)
        var rolesByKey: [String: QwenVLWeightRole?] = [:]
        for tensor in weightCatalog.tensors {
            if rolesByKey[tensor.sanitizedKey] == nil {
                rolesByKey[tensor.sanitizedKey] = tensor.role
            }
        }
        let dataCatalog = WeightDataCatalogBuilder().catalog(for: descriptor)
        self.modelID = descriptor.id
        self.canonicalModelType = descriptor.canonicalModelType
        self.tensors = dataCatalog.tensors
            .map { MLXWeightTensorLoad(handle: $0, role: rolesByKey[$0.sanitizedKey] ?? nil) }
            .sorted { $0.sanitizedKey < $1.sanitizedKey }
        self.tensorCount = tensors.count
        self.loadableTensorCount = tensors.filter(\.loadableAsMLXArray).count
        self.unreadableTensorCount = dataCatalog.unreadableTensorCount
        self.unsupportedDTypeKeys = tensors
            .filter { $0.mlxDType == nil }
            .map(\.sanitizedKey)
            .sorted()
        self.dtypeCounts = Dictionary(grouping: tensors, by: \.safetensorsDType)
            .mapValues(\.count)
        self.mlxDTypeCounts = Dictionary(grouping: tensors.compactMap(\.mlxDType), by: { $0 })
            .mapValues(\.count)
        self.totalLoadableBytes = tensors.reduce(0) { total, tensor in
            total + (tensor.loadableAsMLXArray ? (tensor.byteCount ?? 0) : 0)
        }
        self.canLoadAllTensorsAsMLXArrays = tensorCount > 0 &&
            loadableTensorCount == tensorCount &&
            unsupportedDTypeKeys.isEmpty
    }
}
