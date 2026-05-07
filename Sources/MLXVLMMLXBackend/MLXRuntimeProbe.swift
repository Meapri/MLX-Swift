import Foundation

#if canImport(MLX)
import MLX
#endif

#if canImport(MLXLLM)
import MLXLLM
#endif

#if canImport(MLXLMCommon)
import MLXLMCommon
#endif

#if canImport(MLXVLM)
import MLXVLM
#endif

#if canImport(MLXLMTokenizers)
import MLXLMTokenizers
#endif

public struct MLXRuntimeProbe: Codable, Equatable, Sendable {
    public let canImportMLX: Bool
    public let canImportMLXLMCommon: Bool
    public let canImportMLXLLM: Bool
    public let canImportMLXVLM: Bool
    public let canImportMLXLMTokenizers: Bool
    public let realMLXAPIImplementationCompiled: Bool
    public let backendImplementationReady: Bool
    public let missingModules: [String]

    public init(
        canImportMLX: Bool = MLXRuntimeProbe.detectCanImportMLX(),
        canImportMLXLMCommon: Bool = MLXRuntimeProbe.detectCanImportMLXLMCommon(),
        canImportMLXLLM: Bool = MLXRuntimeProbe.detectCanImportMLXLLM(),
        canImportMLXVLM: Bool = MLXRuntimeProbe.detectCanImportMLXVLM(),
        canImportMLXLMTokenizers: Bool = MLXRuntimeProbe.detectCanImportMLXLMTokenizers(),
        realMLXAPIImplementationCompiled: Bool = MLXRuntimeProbe.detectRealMLXAPIImplementationCompiled(),
        backendImplementationReady: Bool = MLXRuntimeProbe.detectBackendImplementationReady()
    ) {
        self.canImportMLX = canImportMLX
        self.canImportMLXLMCommon = canImportMLXLMCommon
        self.canImportMLXLLM = canImportMLXLLM
        self.canImportMLXVLM = canImportMLXVLM
        self.canImportMLXLMTokenizers = canImportMLXLMTokenizers
        self.realMLXAPIImplementationCompiled = realMLXAPIImplementationCompiled
        self.backendImplementationReady = backendImplementationReady
        self.missingModules = [
            canImportMLX ? nil : "MLX",
            canImportMLXLMCommon ? nil : "MLXLMCommon",
            canImportMLXLLM ? nil : "MLXLLM",
            canImportMLXVLM ? nil : "MLXVLM",
            canImportMLXLMTokenizers ? nil : "MLXLMTokenizers",
        ].compactMap { $0 }
    }

    public var canImportRequiredModules: Bool {
        canImportMLX && canImportMLXLMCommon && canImportMLXLLM && canImportMLXVLM && canImportMLXLMTokenizers
    }

    public static func detectCanImportMLX() -> Bool {
        #if canImport(MLX)
        return true
        #else
        return false
        #endif
    }

    public static func detectCanImportMLXLLM() -> Bool {
        #if canImport(MLXLLM)
        return true
        #else
        return false
        #endif
    }

    public static func detectCanImportMLXLMCommon() -> Bool {
        #if canImport(MLXLMCommon)
        return true
        #else
        return false
        #endif
    }

    public static func detectCanImportMLXVLM() -> Bool {
        #if canImport(MLXVLM)
        return true
        #else
        return false
        #endif
    }

    public static func detectCanImportMLXLMTokenizers() -> Bool {
        #if canImport(MLXLMTokenizers)
        return true
        #else
        return false
        #endif
    }

    public static func detectRealMLXAPIImplementationCompiled() -> Bool {
        #if MLXVLM_REAL_MLX_API && canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXVLM) && canImport(MLXLMTokenizers)
        return true
        #else
        return false
        #endif
    }

    public static func detectBackendImplementationReady() -> Bool {
        #if MLXVLM_REAL_MLX_API && canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXVLM) && canImport(MLXLMTokenizers)
        return true
        #else
        return false
        #endif
    }
}
