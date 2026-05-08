import Foundation
import MLXVLMCore

public struct Gemma4MTPTargetAdapterReport: Codable, Equatable, Sendable {
    public let realMLXAPIImplementationCompiled: Bool
    public let assistantRuntimeCompiled: Bool
    public let targetAdapterCompiled: Bool
    public let supportsHiddenSlotSelection: Bool
    public let supportsSharedKVSnapshot: Bool
    public let supportsScalarCacheRollback: Bool
    public let supportsDraftBinding: Bool
    public let missingRuntimeHooks: [String]

    public init(
        realMLXAPIImplementationCompiled: Bool,
        assistantRuntimeCompiled: Bool,
        targetAdapterCompiled: Bool,
        supportsHiddenSlotSelection: Bool,
        supportsSharedKVSnapshot: Bool,
        supportsScalarCacheRollback: Bool,
        supportsDraftBinding: Bool,
        missingRuntimeHooks: [String]
    ) {
        self.realMLXAPIImplementationCompiled = realMLXAPIImplementationCompiled
        self.assistantRuntimeCompiled = assistantRuntimeCompiled
        self.targetAdapterCompiled = targetAdapterCompiled
        self.supportsHiddenSlotSelection = supportsHiddenSlotSelection
        self.supportsSharedKVSnapshot = supportsSharedKVSnapshot
        self.supportsScalarCacheRollback = supportsScalarCacheRollback
        self.supportsDraftBinding = supportsDraftBinding
        self.missingRuntimeHooks = missingRuntimeHooks
    }
}

public struct Gemma4MTPTargetAdapterSmokeReport: Codable, Equatable, Sendable {
    public let realMLXAPIImplementationCompiled: Bool
    public let assistantRuntimeCompiled: Bool
    public let targetAdapterCompiled: Bool
    public let hiddenShape: [Int]
    public let fullAttentionKVLength: Int?
    public let slidingAttentionKVLength: Int?
    public let cacheOffsetBeforeRollback: Int?
    public let cacheOffsetAfterRollback: Int?
    public let trimmedCacheTokenCount: Int
    public let passed: Bool
    public let error: String?

    public init(
        realMLXAPIImplementationCompiled: Bool,
        assistantRuntimeCompiled: Bool,
        targetAdapterCompiled: Bool,
        hiddenShape: [Int],
        fullAttentionKVLength: Int?,
        slidingAttentionKVLength: Int?,
        cacheOffsetBeforeRollback: Int?,
        cacheOffsetAfterRollback: Int?,
        trimmedCacheTokenCount: Int,
        passed: Bool,
        error: String?
    ) {
        self.realMLXAPIImplementationCompiled = realMLXAPIImplementationCompiled
        self.assistantRuntimeCompiled = assistantRuntimeCompiled
        self.targetAdapterCompiled = targetAdapterCompiled
        self.hiddenShape = hiddenShape
        self.fullAttentionKVLength = fullAttentionKVLength
        self.slidingAttentionKVLength = slidingAttentionKVLength
        self.cacheOffsetBeforeRollback = cacheOffsetBeforeRollback
        self.cacheOffsetAfterRollback = cacheOffsetAfterRollback
        self.trimmedCacheTokenCount = trimmedCacheTokenCount
        self.passed = passed
        self.error = error
    }
}

public enum Gemma4MTPTargetRuntime {
    public static var adapterReport: Gemma4MTPTargetAdapterReport {
        #if MLXVLM_REAL_MLX_API && MLXVLM_GEMMA4_ASSISTANT_RUNTIME && canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXNN)
        return Gemma4MTPTargetAdapterReport(
            realMLXAPIImplementationCompiled: true,
            assistantRuntimeCompiled: true,
            targetAdapterCompiled: true,
            supportsHiddenSlotSelection: true,
            supportsSharedKVSnapshot: true,
            supportsScalarCacheRollback: true,
            supportsDraftBinding: true,
            missingRuntimeHooks: [
                "Gemma4 --draft-kind mtp generation still must route requests through the Swift target text runtime instead of the upstream private Gemma4 container",
                "multimodal Gemma4 prefill still must pass image/audio-derived input embeddings into the Swift target text runtime before MTP verification"
            ]
        )
        #else
        return Gemma4MTPTargetAdapterReport(
            realMLXAPIImplementationCompiled: Self.realMLXAPIImplementationCompiled,
            assistantRuntimeCompiled: false,
            targetAdapterCompiled: false,
            supportsHiddenSlotSelection: false,
            supportsSharedKVSnapshot: false,
            supportsScalarCacheRollback: false,
            supportsDraftBinding: false,
            missingRuntimeHooks: [
                "Build with MLXVLM_ENABLE_REAL_MLX_API=1, MLXVLM_ENABLE_GEMMA4_ASSISTANT_RUNTIME=1, and real MLX Swift dependencies."
            ]
        )
        #endif
    }

    public static func smokeAdapterReport() -> Gemma4MTPTargetAdapterSmokeReport {
        #if MLXVLM_REAL_MLX_API && MLXVLM_GEMMA4_ASSISTANT_RUNTIME && canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXNN)
        do {
            return try Gemma4MTPTargetAdapter.smokeReport()
        } catch {
            return Gemma4MTPTargetAdapterSmokeReport(
                realMLXAPIImplementationCompiled: true,
                assistantRuntimeCompiled: true,
                targetAdapterCompiled: true,
                hiddenShape: [],
                fullAttentionKVLength: nil,
                slidingAttentionKVLength: nil,
                cacheOffsetBeforeRollback: nil,
                cacheOffsetAfterRollback: nil,
                trimmedCacheTokenCount: 0,
                passed: false,
                error: String(describing: error)
            )
        }
        #else
        return Gemma4MTPTargetAdapterSmokeReport(
            realMLXAPIImplementationCompiled: Self.realMLXAPIImplementationCompiled,
            assistantRuntimeCompiled: false,
            targetAdapterCompiled: false,
            hiddenShape: [],
            fullAttentionKVLength: nil,
            slidingAttentionKVLength: nil,
            cacheOffsetBeforeRollback: nil,
            cacheOffsetAfterRollback: nil,
            trimmedCacheTokenCount: 0,
            passed: false,
            error: "Gemma4 MTP target adapter smoke was not compiled. Build with MLXVLM_ENABLE_REAL_MLX_API=1, MLXVLM_ENABLE_GEMMA4_ASSISTANT_RUNTIME=1, and real MLX Swift dependencies."
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

#if MLXVLM_REAL_MLX_API && MLXVLM_GEMMA4_ASSISTANT_RUNTIME && canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXNN)
import MLX
import MLXLMCommon
import MLXNN

struct Gemma4MTPTargetLayerKV {
    let layerIndex: Int
    let layerType: String
    let keys: MLXArray
    let values: MLXArray

    init(layerIndex: Int, layerType: String, keys: MLXArray, values: MLXArray) {
        self.layerIndex = layerIndex
        self.layerType = layerType
        self.keys = keys
        self.values = values
    }
}

struct Gemma4MTPTargetVerifyOutput {
    let logits: MLXArray
    let hiddenStates: MLXArray
    let sharedKVStates: [String: Gemma4AssistantSharedKVState]
    let cacheOffset: Int

    init(
        logits: MLXArray,
        hiddenStates: MLXArray,
        sharedKVStates: [String: Gemma4AssistantSharedKVState],
        cacheOffset: Int
    ) {
        self.logits = logits
        self.hiddenStates = hiddenStates
        self.sharedKVStates = sharedKVStates
        self.cacheOffset = cacheOffset
    }
}

protocol Gemma4MTPTargetVerifier {
    func embedTargetTokens(_ tokens: MLXArray) -> MLXArray
    func prefill(inputIDs: MLXArray, cache: [any KVCache]?) -> Gemma4MTPTargetVerifyOutput
    func verify(inputIDs: MLXArray, cache: [any KVCache]?) -> Gemma4MTPTargetVerifyOutput
    func rollback(cache: [any KVCache], accepted: Int, blockSize: Int)
}

enum Gemma4MTPTargetAdapterError: Error, CustomStringConvertible {
    case missingSharedKVLayerType(String)
    case invalidHiddenSlot(slot: Int, hiddenLength: Int)

    var description: String {
        switch self {
        case .missingSharedKVLayerType(let layerType):
            return "Gemma4 MTP target did not provide shared K/V for layer type \(layerType)"
        case .invalidHiddenSlot(let slot, let hiddenLength):
            return "Gemma4 MTP hidden slot \(slot) is outside hidden length \(hiddenLength)"
        }
    }
}

struct Gemma4MTPTargetAdapter {
    let layerTypes: [String]

    init(layerTypes: [String]) {
        self.layerTypes = layerTypes
    }

    var requiredSharedLayerTypes: [String] {
        Array(Set(layerTypes)).sorted()
    }

    func bindDraftModel(
        _ draftModel: Gemma4AssistantDraftModel,
        targetEmbedding: @escaping (MLXArray) -> MLXArray,
        targetEmbeddingScale: Float = 1.0
    ) {
        draftModel.bindTargetEmbedding(targetEmbedding, scale: targetEmbeddingScale)
    }

    func hiddenSlot(from hiddenStates: MLXArray, acceptedCount: Int) throws -> MLXArray {
        let length = hiddenStates.dim(1)
        guard acceptedCount >= 0, acceptedCount < length else {
            throw Gemma4MTPTargetAdapterError.invalidHiddenSlot(
                slot: acceptedCount,
                hiddenLength: length
            )
        }
        return hiddenStates[0..., acceptedCount ..< acceptedCount + 1, 0...]
    }

    func sharedKVStates(from layerKVs: [Gemma4MTPTargetLayerKV]) throws -> [String: Gemma4AssistantSharedKVState] {
        var latestByType: [String: Gemma4MTPTargetLayerKV] = [:]
        for layerKV in layerKVs {
            let current = latestByType[layerKV.layerType]
            if current == nil || layerKV.layerIndex > current!.layerIndex {
                latestByType[layerKV.layerType] = layerKV
            }
        }

        var output: [String: Gemma4AssistantSharedKVState] = [:]
        for layerType in requiredSharedLayerTypes {
            guard let layerKV = latestByType[layerType] else {
                throw Gemma4MTPTargetAdapterError.missingSharedKVLayerType(layerType)
            }
            output[layerType] = Gemma4AssistantSharedKVState(
                keys: layerKV.keys,
                values: layerKV.values
            )
        }
        return output
    }

    func slicedSharedKVStates(
        _ sharedKVStates: [String: Gemma4AssistantSharedKVState],
        validLength: Int
    ) -> [String: Gemma4AssistantSharedKVState] {
        sharedKVStates.mapValues { state in
            let length = state.sequenceLength
            let clamped = min(max(1, validLength), length)
            guard clamped < length else {
                return state
            }
            return Gemma4AssistantSharedKVState(
                keys: state.keys[.ellipsis, ..<clamped, 0...],
                values: state.values[.ellipsis, ..<clamped, 0...]
            )
        }
    }

    func rollbackScalarCache(_ cache: [any KVCache], acceptedCount: Int, blockSize: Int) -> Int {
        let rejected = max(0, blockSize - (acceptedCount + 1))
        guard rejected > 0 else {
            return 0
        }
        return cache.reduce(0) { total, entry in
            guard entry.isTrimmable else {
                return total
            }
            return total + entry.trim(rejected)
        }
    }

    func applyRoundPlan(
        _ plan: MTPSpeculativeRoundPlan,
        to draftModel: Gemma4AssistantDraftModel,
        verifiedHiddenStates: MLXArray,
        verifiedSharedKVStates: [String: Gemma4AssistantSharedKVState],
        targetCache: [any KVCache]
    ) throws -> MLXArray {
        let hidden = try hiddenSlot(
            from: verifiedHiddenStates,
            acceptedCount: plan.hiddenSlotIndex
        )
        _ = rollbackScalarCache(
            targetCache,
            acceptedCount: plan.walk.acceptedCount,
            blockSize: plan.blockSize
        )
        let slicedSharedKV = slicedSharedKVStates(
            verifiedSharedKVStates,
            validLength: plan.sharedKVValidLength
        )
        draftModel.setSharedKV(
            slicedSharedKV,
            position: MLXArray(plan.positionAfterRound, dtype: .int32).reshaped(1)
        )
        return hidden
    }

    static func smokeReport() throws -> Gemma4MTPTargetAdapterSmokeReport {
        let textConfig = Gemma4AssistantTextRuntimeConfiguration(
            hiddenSize: 4,
            intermediateSize: 8,
            vocabularySize: 16,
            hiddenLayers: 2,
            attentionHeads: 1,
            kvHeads: 1,
            headDim: 2,
            globalHeadDim: 2,
            slidingWindow: 4,
            layerTypes: [
                Gemma4AssistantLayerType.slidingAttention.rawValue,
                Gemma4AssistantLayerType.fullAttention.rawValue,
            ]
        )
        let draftModel = Gemma4AssistantDraftModel(
            config: Gemma4AssistantRuntimeConfiguration(
                backboneHiddenSize: 8,
                text: textConfig,
                useOrderedEmbeddings: false,
                blockSize: 4
            )
        )
        let adapter = Gemma4MTPTargetAdapter(layerTypes: textConfig.layerTypes)
        let dtype = DType.float32
        let keyLength = 6
        let fullKV = Gemma4MTPTargetLayerKV(
            layerIndex: 1,
            layerType: Gemma4AssistantLayerType.fullAttention.rawValue,
            keys: MLXArray.zeros([1, 1, keyLength, 2], dtype: dtype),
            values: MLXArray.zeros([1, 1, keyLength, 2], dtype: dtype)
        )
        let slidingKV = Gemma4MTPTargetLayerKV(
            layerIndex: 0,
            layerType: Gemma4AssistantLayerType.slidingAttention.rawValue,
            keys: MLXArray.zeros([1, 1, keyLength, 2], dtype: dtype),
            values: MLXArray.zeros([1, 1, keyLength, 2], dtype: dtype)
        )
        let shared = try adapter.sharedKVStates(from: [slidingKV, fullKV])
        let cache = StandardKVCache()
        _ = cache.update(
            keys: MLXArray.zeros([1, 1, keyLength, 2], dtype: dtype),
            values: MLXArray.zeros([1, 1, keyLength, 2], dtype: dtype)
        )
        let before = cache.offset
        let plan = MTPSpeculativeRoundPlan(
            walk: SpeculativeWalkResult(acceptedCount: 1, newTokens: [10, 99]),
            blockSize: 4,
            emittedBeforeRound: 1,
            emittedAfterRound: 3,
            positionBeforeRound: 40,
            positionAfterRound: 42,
            hiddenSlotIndex: 1,
            rejectedTokenCount: 2,
            rollbackRequired: true,
            sharedKVValidLength: 4,
            nextBonusToken: 99,
            finished: false
        )
        let hidden = try adapter.applyRoundPlan(
            plan,
            to: draftModel,
            verifiedHiddenStates: MLXArray.zeros([1, 4, 8], dtype: dtype),
            verifiedSharedKVStates: shared,
            targetCache: [cache]
        )
        eval(hidden)
        let after = cache.offset
        let sliced = adapter.slicedSharedKVStates(shared, validLength: plan.sharedKVValidLength)
        let fullLength = sliced[Gemma4AssistantLayerType.fullAttention.rawValue]?.sequenceLength
        let slidingLength = sliced[Gemma4AssistantLayerType.slidingAttention.rawValue]?.sequenceLength
        let passed = hidden.shape == [1, 1, 8] &&
            before == 6 &&
            after == 4 &&
            fullLength == 4 &&
            slidingLength == 4

        return Gemma4MTPTargetAdapterSmokeReport(
            realMLXAPIImplementationCompiled: true,
            assistantRuntimeCompiled: true,
            targetAdapterCompiled: true,
            hiddenShape: hidden.shape,
            fullAttentionKVLength: fullLength,
            slidingAttentionKVLength: slidingLength,
            cacheOffsetBeforeRollback: before,
            cacheOffsetAfterRollback: after,
            trimmedCacheTokenCount: before - after,
            passed: passed,
            error: passed ? nil : "Gemma4 MTP target adapter smoke assertions failed."
        )
    }
}
#endif
