import Foundation

public struct Gemma4MTPTargetTextLoadReport: Codable, Equatable, Sendable {
    public let realMLXAPIImplementationCompiled: Bool
    public let assistantRuntimeCompiled: Bool
    public let targetTextRuntimeCompiled: Bool
    public let modelID: String?
    public let hiddenSize: Int?
    public let hiddenLayers: Int?
    public let kvSharedLayers: Int?
    public let firstKVSharedLayerIndex: Int?
    public let loaded: Bool
    public let quantized: Bool
    public let error: String?

    public init(
        realMLXAPIImplementationCompiled: Bool,
        assistantRuntimeCompiled: Bool,
        targetTextRuntimeCompiled: Bool,
        modelID: String?,
        hiddenSize: Int?,
        hiddenLayers: Int?,
        kvSharedLayers: Int?,
        firstKVSharedLayerIndex: Int?,
        loaded: Bool,
        quantized: Bool,
        error: String?
    ) {
        self.realMLXAPIImplementationCompiled = realMLXAPIImplementationCompiled
        self.assistantRuntimeCompiled = assistantRuntimeCompiled
        self.targetTextRuntimeCompiled = targetTextRuntimeCompiled
        self.modelID = modelID
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.kvSharedLayers = kvSharedLayers
        self.firstKVSharedLayerIndex = firstKVSharedLayerIndex
        self.loaded = loaded
        self.quantized = quantized
        self.error = error
    }
}

public struct Gemma4MTPTargetTextSmokeReport: Codable, Equatable, Sendable {
    public let realMLXAPIImplementationCompiled: Bool
    public let assistantRuntimeCompiled: Bool
    public let targetTextRuntimeCompiled: Bool
    public let modelID: String?
    public let inputShape: [Int]
    public let logitsShape: [Int]
    public let hiddenShape: [Int]
    public let sharedLayerTypes: [String]
    public let cacheCount: Int
    public let cacheOffsets: [Int]
    public let passed: Bool
    public let error: String?

    public init(
        realMLXAPIImplementationCompiled: Bool,
        assistantRuntimeCompiled: Bool,
        targetTextRuntimeCompiled: Bool,
        modelID: String?,
        inputShape: [Int],
        logitsShape: [Int],
        hiddenShape: [Int],
        sharedLayerTypes: [String],
        cacheCount: Int,
        cacheOffsets: [Int],
        passed: Bool,
        error: String?
    ) {
        self.realMLXAPIImplementationCompiled = realMLXAPIImplementationCompiled
        self.assistantRuntimeCompiled = assistantRuntimeCompiled
        self.targetTextRuntimeCompiled = targetTextRuntimeCompiled
        self.modelID = modelID
        self.inputShape = inputShape
        self.logitsShape = logitsShape
        self.hiddenShape = hiddenShape
        self.sharedLayerTypes = sharedLayerTypes
        self.cacheCount = cacheCount
        self.cacheOffsets = cacheOffsets
        self.passed = passed
        self.error = error
    }
}

public enum Gemma4MTPTargetTextLoading {
    public static func loadReport(pathOrIdentifier: String) -> Gemma4MTPTargetTextLoadReport {
        #if MLXVLM_REAL_MLX_API && MLXVLM_GEMMA4_ASSISTANT_RUNTIME && canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXNN)
        do {
            let result = try Gemma4MTPTargetTextLoader().load(pathOrIdentifier: pathOrIdentifier)
            return Gemma4MTPTargetTextLoadReport(
                realMLXAPIImplementationCompiled: true,
                assistantRuntimeCompiled: true,
                targetTextRuntimeCompiled: true,
                modelID: result.descriptor.id,
                hiddenSize: result.model.config.hiddenSize,
                hiddenLayers: result.model.config.hiddenLayers,
                kvSharedLayers: result.model.config.numKVSharedLayers,
                firstKVSharedLayerIndex: result.model.model.firstKVSharedLayerIdx,
                loaded: true,
                quantized: result.quantized,
                error: nil
            )
        } catch {
            return Gemma4MTPTargetTextLoadReport(
                realMLXAPIImplementationCompiled: true,
                assistantRuntimeCompiled: true,
                targetTextRuntimeCompiled: true,
                modelID: nil,
                hiddenSize: nil,
                hiddenLayers: nil,
                kvSharedLayers: nil,
                firstKVSharedLayerIndex: nil,
                loaded: false,
                quantized: false,
                error: String(describing: error)
            )
        }
        #else
        return Gemma4MTPTargetTextLoadReport(
            realMLXAPIImplementationCompiled: Self.realMLXAPIImplementationCompiled,
            assistantRuntimeCompiled: false,
            targetTextRuntimeCompiled: false,
            modelID: nil,
            hiddenSize: nil,
            hiddenLayers: nil,
            kvSharedLayers: nil,
            firstKVSharedLayerIndex: nil,
            loaded: false,
            quantized: false,
            error: "Gemma4 MTP target text runtime was not compiled. Build with MLXVLM_ENABLE_REAL_MLX_API=1, MLXVLM_ENABLE_GEMMA4_ASSISTANT_RUNTIME=1, and real MLX Swift dependencies."
        )
        #endif
    }

    public static func smokeReport(
        pathOrIdentifier: String,
        tokenIDs: [Int] = [2, 106, 107]
    ) -> Gemma4MTPTargetTextSmokeReport {
        #if MLXVLM_REAL_MLX_API && MLXVLM_GEMMA4_ASSISTANT_RUNTIME && canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXNN)
        do {
            let result = try Gemma4MTPTargetTextLoader().load(pathOrIdentifier: pathOrIdentifier)
            return try result.model.smokeReport(
                modelID: result.descriptor.id,
                tokenIDs: tokenIDs
            )
        } catch {
            return Gemma4MTPTargetTextSmokeReport(
                realMLXAPIImplementationCompiled: true,
                assistantRuntimeCompiled: true,
                targetTextRuntimeCompiled: true,
                modelID: nil,
                inputShape: [],
                logitsShape: [],
                hiddenShape: [],
                sharedLayerTypes: [],
                cacheCount: 0,
                cacheOffsets: [],
                passed: false,
                error: String(describing: error)
            )
        }
        #else
        return Gemma4MTPTargetTextSmokeReport(
            realMLXAPIImplementationCompiled: Self.realMLXAPIImplementationCompiled,
            assistantRuntimeCompiled: false,
            targetTextRuntimeCompiled: false,
            modelID: nil,
            inputShape: [],
            logitsShape: [],
            hiddenShape: [],
            sharedLayerTypes: [],
            cacheCount: 0,
            cacheOffsets: [],
            passed: false,
            error: "Gemma4 MTP target text smoke was not compiled. Build with MLXVLM_ENABLE_REAL_MLX_API=1, MLXVLM_ENABLE_GEMMA4_ASSISTANT_RUNTIME=1, and real MLX Swift dependencies."
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
