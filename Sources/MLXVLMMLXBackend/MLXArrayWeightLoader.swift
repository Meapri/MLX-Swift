import Foundation
import MLXVLMCore

public struct MLXArrayWeightLoadReport: Codable, Equatable, Sendable {
    public let realMLXAPIImplementationCompiled: Bool
    public let attemptedTensorCount: Int
    public let loadedArrayCount: Int
    public let loadedKeys: [String]
    public let error: String?

    public init(
        realMLXAPIImplementationCompiled: Bool,
        attemptedTensorCount: Int,
        loadedArrayCount: Int,
        loadedKeys: [String],
        error: String?
    ) {
        self.realMLXAPIImplementationCompiled = realMLXAPIImplementationCompiled
        self.attemptedTensorCount = attemptedTensorCount
        self.loadedArrayCount = loadedArrayCount
        self.loadedKeys = loadedKeys
        self.error = error
    }

    public var loadedAllRequestedArrays: Bool {
        attemptedTensorCount > 0 && error == nil && attemptedTensorCount == loadedArrayCount
    }
}

public enum MLXArrayWeightLoading {
    public static func loadReport(bundle: MLXPreparedWeightBundle) -> MLXArrayWeightLoadReport {
        #if MLXVLM_REAL_MLX_API && canImport(MLX)
        do {
            let loaded = try MLXArrayWeightLoader().load(bundle: bundle)
            return MLXArrayWeightLoadReport(
                realMLXAPIImplementationCompiled: true,
                attemptedTensorCount: bundle.tensors.count,
                loadedArrayCount: loaded.arrays.count,
                loadedKeys: loaded.arrays.keys.sorted(),
                error: nil
            )
        } catch {
            return MLXArrayWeightLoadReport(
                realMLXAPIImplementationCompiled: true,
                attemptedTensorCount: bundle.tensors.count,
                loadedArrayCount: 0,
                loadedKeys: [],
                error: String(describing: error)
            )
        }
        #else
        return MLXArrayWeightLoadReport(
            realMLXAPIImplementationCompiled: false,
            attemptedTensorCount: bundle.tensors.count,
            loadedArrayCount: 0,
            loadedKeys: [],
            error: "MLXArray weight loading was not compiled. Build with MLXVLM_ENABLE_REAL_MLX_API=1 and real MLX Swift dependencies."
        )
        #endif
    }
}

#if MLXVLM_REAL_MLX_API && canImport(MLX)
import MLX

public struct LoadedMLXWeightArrays {
    public let arrays: [String: MLXArray]

    public init(arrays: [String: MLXArray]) {
        self.arrays = arrays
    }
}

public enum MLXArrayWeightLoaderError: Error, CustomStringConvertible, Sendable {
    case unsupportedDType(String)

    public var description: String {
        switch self {
        case .unsupportedDType(let dtype):
            return "Unsupported MLX dtype for array creation: \(dtype)"
        }
    }
}

public struct MLXArrayWeightLoader {
    public init() {}

    public func load(bundle: MLXPreparedWeightBundle) throws -> LoadedMLXWeightArrays {
        var arrays: [String: MLXArray] = [:]
        arrays.reserveCapacity(bundle.tensors.count)
        for tensor in bundle.tensors {
            guard let mlxDTypeName = tensor.load.mlxDType,
                  let dtype = dtype(named: mlxDTypeName)
            else {
                throw MLXArrayWeightLoaderError.unsupportedDType(tensor.load.mlxDType ?? tensor.load.safetensorsDType)
            }
            arrays[tensor.load.sanitizedKey] = MLXArray(
                tensor.payload.data,
                tensor.load.shape,
                dtype: dtype
            )
        }
        return LoadedMLXWeightArrays(arrays: arrays)
    }

    private func dtype(named name: String) -> DType? {
        switch name {
        case "bool":
            return .bool
        case "uint8":
            return .uint8
        case "uint16":
            return .uint16
        case "uint32":
            return .uint32
        case "uint64":
            return .uint64
        case "int8":
            return .int8
        case "int16":
            return .int16
        case "int32":
            return .int32
        case "int64":
            return .int64
        case "float16":
            return .float16
        case "bfloat16":
            return .bfloat16
        case "float32":
            return .float32
        case "float64":
            return .float64
        default:
            return nil
        }
    }
}
#endif
