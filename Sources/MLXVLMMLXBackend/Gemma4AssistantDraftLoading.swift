import Foundation
import MLXVLMCore

public struct Gemma4AssistantDraftLoadReport: Codable, Equatable, Sendable {
    public let realMLXAPIImplementationCompiled: Bool
    public let assistantRuntimeCompiled: Bool
    public let modelID: String?
    public let modelPath: String?
    public let rawModelType: String?
    public let canonicalModelType: String?
    public let loadedTensorCount: Int
    public let loadedKeys: [String]
    public let error: String?

    public init(
        realMLXAPIImplementationCompiled: Bool,
        assistantRuntimeCompiled: Bool,
        modelID: String?,
        modelPath: String?,
        rawModelType: String?,
        canonicalModelType: String?,
        loadedTensorCount: Int,
        loadedKeys: [String],
        error: String?
    ) {
        self.realMLXAPIImplementationCompiled = realMLXAPIImplementationCompiled
        self.assistantRuntimeCompiled = assistantRuntimeCompiled
        self.modelID = modelID
        self.modelPath = modelPath
        self.rawModelType = rawModelType
        self.canonicalModelType = canonicalModelType
        self.loadedTensorCount = loadedTensorCount
        self.loadedKeys = loadedKeys
        self.error = error
    }

    public var loaded: Bool {
        error == nil && loadedTensorCount > 0
    }
}

public struct Gemma4AssistantDraftSmokeReport: Codable, Equatable, Sendable {
    public let realMLXAPIImplementationCompiled: Bool
    public let assistantRuntimeCompiled: Bool
    public let loadedTensorCount: Int
    public let blockSize: Int
    public let outputShape: [Int]
    public let outputDType: String?
    public let error: String?

    public init(
        realMLXAPIImplementationCompiled: Bool,
        assistantRuntimeCompiled: Bool,
        loadedTensorCount: Int,
        blockSize: Int,
        outputShape: [Int],
        outputDType: String?,
        error: String?
    ) {
        self.realMLXAPIImplementationCompiled = realMLXAPIImplementationCompiled
        self.assistantRuntimeCompiled = assistantRuntimeCompiled
        self.loadedTensorCount = loadedTensorCount
        self.blockSize = blockSize
        self.outputShape = outputShape
        self.outputDType = outputDType
        self.error = error
    }

    public var passed: Bool {
        error == nil && outputShape == [1, max(0, blockSize - 1)]
    }
}

public enum Gemma4AssistantDraftLoading {
    public static func loadReport(pathOrIdentifier: String) -> Gemma4AssistantDraftLoadReport {
        #if MLXVLM_REAL_MLX_API && MLXVLM_GEMMA4_ASSISTANT_RUNTIME && canImport(MLX) && canImport(MLXNN) && canImport(MLXLMCommon)
        do {
            let result = try Gemma4AssistantDraftLoader().load(pathOrIdentifier: pathOrIdentifier)
            return Gemma4AssistantDraftLoadReport(
                realMLXAPIImplementationCompiled: true,
                assistantRuntimeCompiled: true,
                modelID: result.descriptor.id,
                modelPath: result.descriptor.path,
                rawModelType: result.descriptor.rawModelType,
                canonicalModelType: result.descriptor.canonicalModelType,
                loadedTensorCount: result.loadedTensorCount,
                loadedKeys: result.loadedKeys,
                error: nil
            )
        } catch {
            return Gemma4AssistantDraftLoadReport(
                realMLXAPIImplementationCompiled: true,
                assistantRuntimeCompiled: true,
                modelID: nil,
                modelPath: nil,
                rawModelType: nil,
                canonicalModelType: nil,
                loadedTensorCount: 0,
                loadedKeys: [],
                error: String(describing: error)
            )
        }
        #else
        return Gemma4AssistantDraftLoadReport(
            realMLXAPIImplementationCompiled: Self.realMLXAPIImplementationCompiled,
            assistantRuntimeCompiled: false,
            modelID: nil,
            modelPath: nil,
            rawModelType: nil,
            canonicalModelType: nil,
            loadedTensorCount: 0,
            loadedKeys: [],
            error: "Gemma4 assistant draft loading was not compiled. Build with MLXVLM_ENABLE_REAL_MLX_API=1, MLXVLM_ENABLE_GEMMA4_ASSISTANT_RUNTIME=1, and real MLX Swift dependencies."
        )
        #endif
    }

    public static func smokeDraftBlockReport(
        pathOrIdentifier: String,
        blockSize: Int = 4
    ) -> Gemma4AssistantDraftSmokeReport {
        #if MLXVLM_REAL_MLX_API && MLXVLM_GEMMA4_ASSISTANT_RUNTIME && canImport(MLX) && canImport(MLXNN) && canImport(MLXLMCommon)
        do {
            let result = try Gemma4AssistantDraftLoader().load(pathOrIdentifier: pathOrIdentifier)
            let drafted = result.model.smokeDraftBlock(blockSize: blockSize)
            return Gemma4AssistantDraftSmokeReport(
                realMLXAPIImplementationCompiled: true,
                assistantRuntimeCompiled: true,
                loadedTensorCount: result.loadedTensorCount,
                blockSize: blockSize,
                outputShape: drafted.shape,
                outputDType: String(describing: drafted.dtype),
                error: nil
            )
        } catch {
            return Gemma4AssistantDraftSmokeReport(
                realMLXAPIImplementationCompiled: true,
                assistantRuntimeCompiled: true,
                loadedTensorCount: 0,
                blockSize: blockSize,
                outputShape: [],
                outputDType: nil,
                error: String(describing: error)
            )
        }
        #else
        return Gemma4AssistantDraftSmokeReport(
            realMLXAPIImplementationCompiled: Self.realMLXAPIImplementationCompiled,
            assistantRuntimeCompiled: false,
            loadedTensorCount: 0,
            blockSize: blockSize,
            outputShape: [],
            outputDType: nil,
            error: "Gemma4 assistant draftBlock smoke was not compiled. Build with MLXVLM_ENABLE_REAL_MLX_API=1, MLXVLM_ENABLE_GEMMA4_ASSISTANT_RUNTIME=1, and real MLX Swift dependencies."
        )
        #endif
    }

    private static var realMLXAPIImplementationCompiled: Bool {
        #if MLXVLM_REAL_MLX_API && canImport(MLX)
        return true
        #else
        return false
        #endif
    }
}
