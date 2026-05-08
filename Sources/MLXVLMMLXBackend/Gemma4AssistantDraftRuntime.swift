#if MLXVLM_REAL_MLX_API && MLXVLM_GEMMA4_ASSISTANT_RUNTIME && canImport(MLX) && canImport(MLXNN) && canImport(MLXLMCommon)
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLMCore

struct Gemma4AssistantSharedKVState {
    let keys: MLXArray
    let values: MLXArray

    var sequenceLength: Int {
        keys.dim(-2)
    }
}

enum Gemma4AssistantLayerType: String {
    case fullAttention = "full_attention"
    case slidingAttention = "sliding_attention"
}

struct Gemma4AssistantDraftMasks {
    let fullAttention: MLXFast.ScaledDotProductAttentionMaskMode
    let slidingAttention: MLXFast.ScaledDotProductAttentionMaskMode

    func mask(for layerType: String) -> MLXFast.ScaledDotProductAttentionMaskMode {
        layerType == Gemma4AssistantLayerType.slidingAttention.rawValue
            ? slidingAttention
            : fullAttention
    }
}

enum Gemma4AssistantMaskBuilder {
    static func makeDrafterMasks(
        sharedKVStates: [String: Gemma4AssistantSharedKVState],
        queryLength: Int,
        queryOffset: MLXArray,
        slidingWindow: Int,
        dtype: DType = .float32
    ) -> Gemma4AssistantDraftMasks {
        let fullKVLength = sharedKVStates[Gemma4AssistantLayerType.fullAttention.rawValue]?
            .sequenceLength ?? 0
        let slidingKVLength = sharedKVStates[Gemma4AssistantLayerType.slidingAttention.rawValue]?
            .sequenceLength ?? fullKVLength

        let fullValidLength = minimum(queryOffset.asType(.int32), MLXArray(fullKVLength, dtype: .int32))
        let slidingValidLength = minimum(queryOffset.asType(.int32), MLXArray(slidingKVLength, dtype: .int32))

        return Gemma4AssistantDraftMasks(
            fullAttention: bidirectionalFullMask(
                queryLength: queryLength,
                kvLength: fullKVLength,
                kvValidLength: fullValidLength,
                dtype: dtype
            ),
            slidingAttention: bidirectionalSWAMask(
                queryLength: queryLength,
                queryOffset: slidingValidLength,
                kvLength: slidingKVLength,
                window: slidingWindow,
                kvValidLength: slidingValidLength,
                dtype: dtype
            )
        )
    }

    static func makeDrafterMasks(
        sharedKVStates: [String: Gemma4AssistantSharedKVState],
        queryLength: Int,
        queryOffset: Int,
        slidingWindow: Int,
        dtype: DType = .float32
    ) -> Gemma4AssistantDraftMasks {
        let fullKVLength = sharedKVStates[Gemma4AssistantLayerType.fullAttention.rawValue]?
            .sequenceLength ?? 0
        let slidingKVLength = sharedKVStates[Gemma4AssistantLayerType.slidingAttention.rawValue]?
            .sequenceLength ?? fullKVLength

        let fullValidLength = min(queryOffset, fullKVLength)
        let slidingValidLength = min(queryOffset, slidingKVLength)

        return Gemma4AssistantDraftMasks(
            fullAttention: bidirectionalFullMask(
                queryLength: queryLength,
                kvLength: fullKVLength,
                kvValidLength: fullValidLength,
                dtype: dtype
            ),
            slidingAttention: bidirectionalSWAMask(
                queryLength: queryLength,
                queryOffset: slidingValidLength,
                kvLength: slidingKVLength,
                window: slidingWindow,
                kvValidLength: slidingValidLength,
                dtype: dtype
            )
        )
    }

    static func bidirectionalFullMask(
        queryLength: Int,
        kvLength: Int,
        kvValidLength: Int? = nil,
        dtype: DType = .float32
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard kvLength > 0 else {
            return .none
        }
        guard let kvValidLength, kvValidLength < kvLength else {
            return .none
        }

        let keyIndices = MLXArray(0 ..< kvLength)
        let inside = keyIndices .< kvValidLength
        let bias = MLX.where(
            inside,
            MLXArray(0.0, dtype: dtype),
            MLXArray(-Float.infinity, dtype: dtype)
        )
        return .array(bias.reshaped(1, 1, 1, kvLength))
    }

    static func bidirectionalFullMask(
        queryLength: Int,
        kvLength: Int,
        kvValidLength: MLXArray,
        dtype: DType = .float32
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard kvLength > 0 else {
            return .none
        }

        let keyIndices = MLXArray(0 ..< kvLength).reshaped(1, kvLength)
        let valid = broadcastBatchVector(kvValidLength, limit: kvLength)
        let inside = keyIndices .< valid.reshaped(-1, 1)
        let bias = MLX.where(
            inside,
            MLXArray(0.0, dtype: dtype),
            MLXArray(-Float.infinity, dtype: dtype)
        )
        return .array(expandedDimensions(bias, axes: [1, 2]))
    }

    static func bidirectionalSWAMask(
        queryLength: Int,
        queryOffset: Int,
        kvLength: Int,
        window: Int,
        kvValidLength: Int? = nil,
        dtype: DType = .float32
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard kvLength > 0, queryLength > 0 else {
            return .none
        }
        if (kvValidLength == nil || kvValidLength! >= kvLength)
            && kvLength <= window
            && queryOffset + queryLength <= kvLength + window
        {
            return .none
        }

        let queryIndices = MLXArray(queryOffset ..< (queryOffset + queryLength))
            .reshaped(queryLength, 1)
        let keyIndices = MLXArray(0 ..< kvLength).reshaped(1, kvLength)
        let distance = queryIndices - keyIndices
        var inside = (distance .> -window) .&& (distance .< window)
        if let kvValidLength {
            inside = inside .&& (keyIndices .< kvValidLength)
        }

        let bias = MLX.where(
            inside,
            MLXArray(0.0, dtype: dtype),
            MLXArray(-Float.infinity, dtype: dtype)
        )
        return .array(bias.reshaped(1, 1, queryLength, kvLength))
    }

    static func bidirectionalSWAMask(
        queryLength: Int,
        queryOffset: MLXArray,
        kvLength: Int,
        window: Int,
        kvValidLength: MLXArray? = nil,
        dtype: DType = .float32
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard kvLength > 0, queryLength > 0 else {
            return .none
        }

        let offset = broadcastBatchVector(queryOffset, limit: Int.max)
        let querySteps = MLXArray(0 ..< queryLength).reshaped(1, queryLength)
        let queryIndices = offset.reshaped(-1, 1) + querySteps
        let keyIndices = MLXArray(0 ..< kvLength).reshaped(1, 1, kvLength)
        let distance = queryIndices.reshaped(queryIndices.dim(0), queryLength, 1) - keyIndices
        var inside = (distance .> -window) .&& (distance .< window)
        if let kvValidLength {
            let valid = broadcastBatchVector(kvValidLength, limit: kvLength)
            inside = inside .&& (keyIndices .< valid.reshaped(-1, 1, 1))
        }

        let bias = MLX.where(
            inside,
            MLXArray(0.0, dtype: dtype),
            MLXArray(-Float.infinity, dtype: dtype)
        )
        return .array(expandedDimensions(bias, axis: 1))
    }

    static func normalizeBatchedSharedKVStates(
        _ sharedKVStates: [String: Gemma4AssistantSharedKVState],
        kvValidLength: MLXArray,
        leftPadding: MLXArray?
    ) -> [String: Gemma4AssistantSharedKVState] {
        guard let leftPadding else {
            return sharedKVStates
        }

        return sharedKVStates.mapValues { state in
            Gemma4AssistantSharedKVState(
                keys: normalizeSharedKVTensor(
                    state.keys,
                    kvValidLength: kvValidLength,
                    leftPadding: leftPadding
                ),
                values: normalizeSharedKVTensor(
                    state.values,
                    kvValidLength: kvValidLength,
                    leftPadding: leftPadding
                )
            )
        }
    }

    private static func normalizeSharedKVTensor(
        _ tensor: MLXArray,
        kvValidLength: MLXArray,
        leftPadding: MLXArray
    ) -> MLXArray {
        guard tensor.ndim == 4 else {
            return tensor
        }

        let batch = tensor.dim(0)
        let sequenceLength = tensor.dim(-2)
        let valid = broadcastBatchVector(kvValidLength, batch: batch, limit: sequenceLength)
        let left = broadcastBatchVector(leftPadding, batch: batch, limit: sequenceLength)
        let rolled = dynamicRoll(tensor, shifts: -left, axis: 2)
        let keep = (MLXArray(0 ..< sequenceLength).reshaped(1, sequenceLength) .< valid.reshaped(-1, 1))
            .asType(tensor.dtype)
            .reshaped(batch, 1, sequenceLength, 1)
        return rolled * keep
    }

    private static func dynamicRoll(_ tensor: MLXArray, shifts: MLXArray, axis: Int) -> MLXArray {
        let length = tensor.dim(axis)
        let indices = (MLXArray(0 ..< length).reshaped(1, length) - shifts.reshaped(-1, 1))
        let wrapped = ((indices % length) + length) % length
        return take(tensor, wrapped.asType(.int32), axis: axis)
    }

    private static func broadcastBatchVector(
        _ value: MLXArray,
        batch: Int? = nil,
        limit: Int
    ) -> MLXArray {
        var vector = value.asType(.int32)
        if vector.ndim == 0 {
            vector = vector.reshaped(1)
        } else if vector.ndim > 1 {
            vector = vector.reshaped(-1)
        }

        if let batch, vector.dim(0) == 1, batch != 1 {
            vector = repeated(vector, count: batch, axis: 0)
        }
        if limit == Int.max {
            return maximum(vector, MLXArray(0, dtype: .int32))
        }
        return clip(vector, min: MLXArray(0, dtype: .int32), max: MLXArray(limit, dtype: .int32))
    }
}

final class Gemma4AssistantMaskedEmbedder: Module {
    let hiddenSize: Int
    let vocabularySize: Int
    let numCentroids: Int
    let topK: Int
    let vocabularySizePerCentroid: Int

    @ModuleInfo(key: "centroids") var centroids: Linear
    @ParameterInfo(key: "token_ordering") var tokenOrdering: MLXArray

    init(
        hiddenSize: Int,
        vocabularySize: Int,
        numCentroids: Int,
        topK: Int
    ) {
        self.hiddenSize = hiddenSize
        self.vocabularySize = vocabularySize
        self.numCentroids = numCentroids
        self.topK = topK
        self.vocabularySizePerCentroid = vocabularySize / numCentroids
        self._centroids.wrappedValue = Linear(hiddenSize, numCentroids, bias: false)
        self._tokenOrdering.wrappedValue = MLXArray.zeros([vocabularySize], dtype: .int32)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, lmHeadWeight: MLXArray) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let sequenceLength = hiddenStates.dim(1)
        let centroidLogits = centroids(hiddenStates)
        let topKIndices = argPartition(centroidLogits, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
        let ordering = tokenOrdering.asType(.int32).reshaped(numCentroids, vocabularySizePerCentroid)
        let selectedCanonical = ordering[topKIndices]

        let selectedCount = topK * vocabularySizePerCentroid
        let flatIndices = selectedCanonical.reshaped(-1).asType(.int32)
        let selectedEmbeddings = lmHeadWeight[flatIndices]
            .reshaped(batch, sequenceLength, selectedCount, hiddenSize)

        let selectedLogits = matmul(
            hiddenStates[.ellipsis, .newAxis, 0...],
            selectedEmbeddings.swappedAxes(-1, -2)
        ).squeezed(axis: -2)

        let maskValue = selectedLogits.min() - MLXArray(1.0, dtype: hiddenStates.dtype)
        let output = MLXArray.zeros([batch, sequenceLength, vocabularySize], dtype: hiddenStates.dtype)
            + maskValue
        let scatterIndices = selectedCanonical.reshaped(batch, sequenceLength, selectedCount)
            .asType(.int32)
        return putAlong(output, scatterIndices, values: selectedLogits, axis: -1)
    }
}

struct Gemma4AssistantTextRuntimeConfiguration {
    let hiddenSize: Int
    let intermediateSize: Int
    let vocabularySize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let globalHeadDim: Int?
    let slidingWindow: Int
    let rmsNormEps: Float
    let maxPositionEmbeddings: Int
    let ropeTraditional: Bool
    let layerTypes: [String]
    let ropeParameters: [String: [String: StringOrNumber]]

    init(
        hiddenSize: Int,
        intermediateSize: Int,
        vocabularySize: Int,
        hiddenLayers: Int,
        attentionHeads: Int,
        kvHeads: Int,
        headDim: Int,
        globalHeadDim: Int? = nil,
        slidingWindow: Int,
        rmsNormEps: Float = 1e-6,
        maxPositionEmbeddings: Int = 131_072,
        ropeTraditional: Bool = false,
        layerTypes: [String],
        ropeParameters: [String: [String: StringOrNumber]] = [:]
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.globalHeadDim = globalHeadDim
        self.slidingWindow = slidingWindow
        self.rmsNormEps = rmsNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTraditional = ropeTraditional
        self.layerTypes = layerTypes
        self.ropeParameters = ropeParameters
    }

    func attentionHeadDim(layerType: String) -> Int {
        layerType == Gemma4AssistantLayerType.fullAttention.rawValue
            ? (globalHeadDim ?? headDim)
            : headDim
    }
}

struct Gemma4AssistantRuntimeConfiguration {
    let backboneHiddenSize: Int
    let text: Gemma4AssistantTextRuntimeConfiguration
    let useOrderedEmbeddings: Bool
    let numCentroids: Int
    let centroidIntermediateTopK: Int
    let tieWordEmbeddings: Bool
    let blockSize: Int

    init(
        backboneHiddenSize: Int,
        text: Gemma4AssistantTextRuntimeConfiguration,
        useOrderedEmbeddings: Bool,
        numCentroids: Int = 2048,
        centroidIntermediateTopK: Int = 32,
        tieWordEmbeddings: Bool = true,
        blockSize: Int = 4
    ) {
        self.backboneHiddenSize = backboneHiddenSize
        self.text = text
        self.useOrderedEmbeddings = useOrderedEmbeddings
        self.numCentroids = numCentroids
        self.centroidIntermediateTopK = centroidIntermediateTopK
        self.tieWordEmbeddings = tieWordEmbeddings
        self.blockSize = blockSize
    }
}

final class Gemma4AssistantRMSNormZeroShift: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

final class Gemma4AssistantMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(config: Gemma4AssistantTextRuntimeConfiguration) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class Gemma4AssistantAttention: Module {
    let layerType: String
    let headDim: Int
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4AssistantRMSNormZeroShift
    @ModuleInfo var rope: RoPELayer

    init(config: Gemma4AssistantTextRuntimeConfiguration, layerIndex: Int) {
        self.layerType = config.layerTypes[layerIndex]
        self.headDim = layerType == Gemma4AssistantLayerType.fullAttention.rawValue
            ? (config.globalHeadDim ?? config.headDim)
            : config.headDim
        self.attentionHeads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.scale = 1.0

        self._qProj.wrappedValue = Linear(config.hiddenSize, config.attentionHeads * self.headDim, bias: false)
        self._oProj.wrappedValue = Linear(config.attentionHeads * self.headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = Gemma4AssistantRMSNormZeroShift(
            dimensions: self.headDim,
            eps: config.rmsNormEps
        )

        let ropeKey = layerType == Gemma4AssistantLayerType.slidingAttention.rawValue
            ? Gemma4AssistantLayerType.slidingAttention.rawValue
            : Gemma4AssistantLayerType.fullAttention.rawValue
        let ropeConfig = config.ropeParameters[ropeKey]
        let ropeTheta = ropeConfig?["rope_theta"]?.asFloat() ??
            (ropeKey == Gemma4AssistantLayerType.slidingAttention.rawValue ? 10_000 : 1_000_000)
        self._rope.wrappedValue = initializeRope(
            dims: self.headDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        sharedKV: Gemma4AssistantSharedKVState,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        offset: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        var queries = qProj(x).reshaped(batch, length, attentionHeads, headDim)
        queries = qNorm(queries).transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)

        let localMask = adjustAttentionMask(mask, keyLength: sharedKV.sequenceLength)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: sharedKV.keys,
            values: sharedKV.values,
            scale: scale,
            mask: localMask
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }

    private func adjustAttentionMask(
        _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
        keyLength: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        switch mask {
        case .array(let maskArray):
            let maskLength = maskArray.dim(-1)
            guard maskLength > keyLength else {
                return mask
            }
            let start = maskLength - keyLength
            return .array(maskArray[.ellipsis, start...])
        case .arrays, .causal, .none:
            return mask
        }
    }
}

final class Gemma4AssistantDecoderLayer: Module {
    let layerType: String

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4AssistantAttention
    @ModuleInfo var mlp: Gemma4AssistantMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4AssistantRMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4AssistantRMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        Gemma4AssistantRMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4AssistantRMSNormZeroShift
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(config: Gemma4AssistantTextRuntimeConfiguration, layerIndex: Int) {
        self.layerType = config.layerTypes[layerIndex]
        self._selfAttention.wrappedValue = Gemma4AssistantAttention(
            config: config,
            layerIndex: layerIndex
        )
        self._mlp.wrappedValue = Gemma4AssistantMLP(config: config)
        self._inputLayerNorm.wrappedValue = Gemma4AssistantRMSNormZeroShift(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = Gemma4AssistantRMSNormZeroShift(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4AssistantRMSNormZeroShift(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4AssistantRMSNormZeroShift(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        sharedKV: Gemma4AssistantSharedKVState,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        offset: MLXArray
    ) -> MLXArray {
        var residual = x
        var h = inputLayerNorm(x)
        h = selfAttention(h, sharedKV: sharedKV, mask: mask, offset: offset)
        h = postAttentionLayerNorm(h)
        h = residual + h

        residual = h
        h = preFeedforwardLayerNorm(h)
        h = mlp(h)
        h = postFeedforwardLayerNorm(h)
        h = residual + h
        return h * layerScalar
    }
}

final class Gemma4AssistantDraftInner: Module {
    let config: Gemma4AssistantTextRuntimeConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4AssistantDecoderLayer]
    @ModuleInfo var norm: Gemma4AssistantRMSNormZeroShift

    init(config: Gemma4AssistantTextRuntimeConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Gemma4AssistantDecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = Gemma4AssistantRMSNormZeroShift(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }
}

final class Gemma4AssistantDraftModel: Module {
    let config: Gemma4AssistantRuntimeConfiguration

    @ModuleInfo(key: "model") var model: Gemma4AssistantDraftInner
    @ModuleInfo(key: "pre_projection") var preProjection: Linear
    @ModuleInfo(key: "post_projection") var postProjection: Linear
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    @ModuleInfo(key: "masked_embedding") var maskedEmbedding: Gemma4AssistantMaskedEmbedder?

    private var targetEmbedding: ((MLXArray) -> MLXArray)?
    private var targetEmbeddingScale: Float = 1.0
    private var sharedKVStates: [String: Gemma4AssistantSharedKVState]?
    private var position: MLXArray?

    init(config: Gemma4AssistantRuntimeConfiguration) {
        self.config = config
        self._model.wrappedValue = Gemma4AssistantDraftInner(config: config.text)
        self._preProjection.wrappedValue = Linear(
            2 * config.backboneHiddenSize,
            config.text.hiddenSize,
            bias: false
        )
        self._postProjection.wrappedValue = Linear(
            config.text.hiddenSize,
            config.backboneHiddenSize,
            bias: false
        )
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.text.hiddenSize,
                config.text.vocabularySize,
                bias: false
            )
        }
        if config.useOrderedEmbeddings {
            self._maskedEmbedding.wrappedValue = Gemma4AssistantMaskedEmbedder(
                hiddenSize: config.text.hiddenSize,
                vocabularySize: config.text.vocabularySize,
                numCentroids: config.numCentroids,
                topK: config.centroidIntermediateTopK
            )
        }
        super.init()
    }

    func bindTargetEmbedding(_ embed: @escaping (MLXArray) -> MLXArray, scale: Float = 1.0) {
        self.targetEmbedding = embed
        self.targetEmbeddingScale = scale
    }

    func setSharedKV(
        _ sharedKVStates: [String: Gemma4AssistantSharedKVState],
        position: MLXArray,
        leftPadding: MLXArray? = nil
    ) {
        self.position = position
        self.sharedKVStates = Gemma4AssistantMaskBuilder.normalizeBatchedSharedKVStates(
            sharedKVStates,
            kvValidLength: position,
            leftPadding: leftPadding
        )
    }

    func callAsFunction(
        inputsEmbeds: MLXArray,
        sharedKVStates: [String: Gemma4AssistantSharedKVState],
        positionIDs: MLXArray
    ) -> (lastHidden: MLXArray, logits: MLXArray) {
        var h = preProjection(inputsEmbeds)
        let queryLength = h.dim(1)
        let queryOffset = positionIDs[0..., 0].asType(.int32)
        let masks = Gemma4AssistantMaskBuilder.makeDrafterMasks(
            sharedKVStates: sharedKVStates,
            queryLength: queryLength,
            queryOffset: queryOffset,
            slidingWindow: config.text.slidingWindow,
            dtype: h.dtype
        )

        for layer in model.layers {
            guard let sharedKV = sharedKVStates[layer.layerType] else {
                fatalError("Missing Gemma4 assistant shared K/V for layer type \(layer.layerType)")
            }
            h = layer(
                h,
                sharedKV: sharedKV,
                mask: masks.mask(for: layer.layerType),
                offset: queryOffset
            )
        }

        h = model.norm(h)
        let lastHidden = postProjection(h)
        let logits: MLXArray
        if let maskedEmbedding {
            logits = maskedEmbedding(h, lmHeadWeight: model.embedTokens.weight)
        } else if config.tieWordEmbeddings {
            logits = model.embedTokens.asLinear(h)
        } else if let lmHead {
            logits = lmHead(h)
        } else {
            logits = model.embedTokens.asLinear(h)
        }
        return (lastHidden, logits)
    }

    func draftBlock(
        lastBonus: MLXArray,
        hidden: MLXArray,
        blockSize: Int,
        sampler: (MLXArray) -> MLXArray
    ) -> MLXArray {
        guard let targetEmbedding else {
            fatalError("Gemma4 assistant draftBlock requires bindTargetEmbedding before use")
        }
        guard let sharedKVStates, let position else {
            fatalError("Gemma4 assistant draftBlock requires setSharedKV before use")
        }

        var token = lastBonus.ndim == 1 ? expandedDimensions(lastBonus.asType(.int32), axis: 1) : lastBonus.asType(.int32)
        var previousHidden = hidden
        var tokens: [MLXArray] = []
        let positionIDs = position.ndim == 1 ? expandedDimensions(position.asType(.int32), axis: 1) : position.asType(.int32)

        for _ in 0 ..< max(0, blockSize - 1) {
            let tokenEmbedding = targetEmbedding(token) * MLXArray(targetEmbeddingScale, dtype: previousHidden.dtype)
            let inputsEmbeds = concatenated([tokenEmbedding, previousHidden], axis: -1)
            let output = self(
                inputsEmbeds: inputsEmbeds,
                sharedKVStates: sharedKVStates,
                positionIDs: positionIDs
            )
            previousHidden = output.lastHidden
            token = sampler(output.logits).asType(.int32)
            tokens.append(token)
        }

        if tokens.isEmpty {
            return MLXArray.zeros([lastBonus.dim(0), 0], dtype: .int32)
        }
        return concatenated(tokens, axis: 1)
    }

    func smokeDraftBlock(blockSize: Int = 4, kvLength: Int = 4) -> MLXArray {
        let batch = 1
        let dtype = model.embedTokens.weight.dtype
        bindTargetEmbedding { [config] tokens in
            MLXArray.zeros([tokens.dim(0), tokens.dim(1), config.backboneHiddenSize], dtype: dtype)
        }

        var sharedKVStates: [String: Gemma4AssistantSharedKVState] = [:]
        for layerType in Set(config.text.layerTypes) {
            let dim = config.text.attentionHeadDim(layerType: layerType)
            sharedKVStates[layerType] = Gemma4AssistantSharedKVState(
                keys: MLXArray.zeros([batch, config.text.kvHeads, kvLength, dim], dtype: dtype),
                values: MLXArray.zeros([batch, config.text.kvHeads, kvLength, dim], dtype: dtype)
            )
        }
        setSharedKV(
            sharedKVStates,
            position: MLXArray(kvLength, dtype: .int32).reshaped(1)
        )

        let hidden = MLXArray.zeros([batch, 1, config.backboneHiddenSize], dtype: dtype)
        let tokens = draftBlock(
            lastBonus: MLXArray.zeros([batch], dtype: .int32),
            hidden: hidden,
            blockSize: blockSize
        ) { logits in
            argMax(logits, axis: -1).asType(.int32)
        }
        eval(tokens)
        return tokens
    }
}

struct Gemma4AssistantDraftLoadResult {
    let descriptor: ModelDescriptor
    let model: Gemma4AssistantDraftModel
    let loadedTensorCount: Int
    let loadedKeys: [String]
}

enum Gemma4AssistantDraftLoaderError: Error, CustomStringConvertible {
    case invalidConfig(String)
    case notGemma4Assistant(String)

    var description: String {
        switch self {
        case .invalidConfig(let message):
            return "Invalid Gemma4 assistant draft config: \(message)"
        case .notGemma4Assistant(let modelType):
            return "Draft model is not Gemma4AssistantForCausalLM: \(modelType)"
        }
    }
}

struct Gemma4AssistantDraftLoader {
    let store: ModelStore
    let preparer: MLXWeightPreparer
    let arrayLoader: MLXArrayWeightLoader

    init(
        store: ModelStore = ModelStore(),
        preparer: MLXWeightPreparer = MLXWeightPreparer(),
        arrayLoader: MLXArrayWeightLoader = MLXArrayWeightLoader()
    ) {
        self.store = store
        self.preparer = preparer
        self.arrayLoader = arrayLoader
    }

    func load(pathOrIdentifier: String) throws -> Gemma4AssistantDraftLoadResult {
        let descriptor = try store.loadDescriptor(pathOrIdentifier: pathOrIdentifier)
        let configJSON = try store.loadNormalizedConfig(pathOrIdentifier: pathOrIdentifier)
        let runtimeConfig = try makeRuntimeConfiguration(configJSON)
        let bundle = try preparer.prepare(descriptor: descriptor)
        let arrays = try arrayLoader.load(bundle: bundle).arrays
        let model = Gemma4AssistantDraftModel(config: runtimeConfig)
        let sanitized = sanitize(weights: arrays, config: runtimeConfig)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: Module.VerifyUpdate.all)
        eval(model)

        return Gemma4AssistantDraftLoadResult(
            descriptor: descriptor,
            model: model,
            loadedTensorCount: sanitized.count,
            loadedKeys: sanitized.keys.sorted()
        )
    }

    private func sanitize(
        weights: [String: MLXArray],
        config: Gemma4AssistantRuntimeConfiguration
    ) -> [String: MLXArray] {
        var output: [String: MLXArray] = [:]
        output.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key == "lm_head.weight", config.tieWordEmbeddings {
                continue
            }
            if key == "masked_embedding.token_ordering" {
                output[key] = value.asType(.int32)
            } else {
                output[key] = value
            }
        }
        return output
    }

    private func makeRuntimeConfiguration(_ value: MLXVLMCore.JSONValue) throws -> Gemma4AssistantRuntimeConfiguration {
        guard let object = value.objectValue else {
            throw Gemma4AssistantDraftLoaderError.invalidConfig("top-level config must be an object")
        }
        let modelType = object.string("model_type", default: "")
        let architectures = object["architectures"]?.arrayValue?.compactMap { $0.stringValue } ?? []
        guard modelType == "gemma4_assistant" || architectures.contains("Gemma4AssistantForCausalLM") else {
            throw Gemma4AssistantDraftLoaderError.notGemma4Assistant(modelType ?? "unknown")
        }
        guard let textObject = object["text_config"]?.objectValue else {
            throw Gemma4AssistantDraftLoaderError.invalidConfig("missing object field 'text_config'")
        }

        let hiddenLayers = try textObject.requiredInt("num_hidden_layers")
        let layerTypes = textObject["layer_types"]?.arrayValue?.compactMap { $0.stringValue }
        let resolvedLayerTypes = layerTypes?.isEmpty == false
            ? layerTypes!
            : Array(repeating: Gemma4AssistantLayerType.slidingAttention.rawValue, count: hiddenLayers)
        guard resolvedLayerTypes.count == hiddenLayers else {
            throw Gemma4AssistantDraftLoaderError.invalidConfig(
                "layer_types count \(resolvedLayerTypes.count) does not match num_hidden_layers \(hiddenLayers)"
            )
        }

        let textConfig = Gemma4AssistantTextRuntimeConfiguration(
            hiddenSize: try textObject.requiredInt("hidden_size"),
            intermediateSize: try textObject.requiredInt("intermediate_size"),
            vocabularySize: try textObject.requiredInt("vocab_size"),
            hiddenLayers: hiddenLayers,
            attentionHeads: try textObject.requiredInt("num_attention_heads"),
            kvHeads: try textObject.requiredInt("num_key_value_heads"),
            headDim: try textObject.requiredInt("head_dim"),
            globalHeadDim: textObject["global_head_dim"]?.intValue,
            slidingWindow: textObject.int("sliding_window", default: 512),
            rmsNormEps: Float(textObject.double("rms_norm_eps", default: 1e-6)),
            maxPositionEmbeddings: textObject.int("max_position_embeddings", default: 131_072),
            ropeTraditional: textObject.bool("rope_traditional", default: false),
            layerTypes: resolvedLayerTypes,
            ropeParameters: Self.ropeParameters(from: textObject["rope_parameters"])
        )

        return Gemma4AssistantRuntimeConfiguration(
            backboneHiddenSize: try object.requiredInt("backbone_hidden_size"),
            text: textConfig,
            useOrderedEmbeddings: object.bool("use_ordered_embeddings", default: false),
            numCentroids: object.int("num_centroids", default: 2048),
            centroidIntermediateTopK: object.int("centroid_intermediate_top_k", default: 32),
            tieWordEmbeddings: object.bool(
                "tie_word_embeddings",
                default: textObject.bool("tie_word_embeddings", default: true)
            ),
            blockSize: object.int("block_size", default: 4)
        )
    }

    private static func ropeParameters(from value: MLXVLMCore.JSONValue?) -> [String: [String: StringOrNumber]] {
        guard let object = value?.objectValue else {
            return [:]
        }
        var output: [String: [String: StringOrNumber]] = [:]
        for (key, value) in object {
            guard let nested = value.objectValue else {
                continue
            }
            output[key] = nested.compactMapValues(Self.stringOrNumber)
        }
        return output
    }

    private static func stringOrNumber(from value: MLXVLMCore.JSONValue) -> StringOrNumber? {
        switch value {
        case .string(let string):
            return .string(string)
        case .number(let number):
            if number.rounded() == number {
                return .int(Int(number))
            }
            return .float(Float(number))
        case .bool(let bool):
            return .bool(bool)
        case .array(let values):
            let ints = values.compactMap { $0.intValue }
            if ints.count == values.count {
                return .ints(ints)
            }
            let floats = values.compactMap { value -> Float? in
                guard let double = value.doubleValue else {
                    return nil
                }
                return Float(double)
            }
            return floats.count == values.count ? .floats(floats) : nil
        case .object, .null:
            return nil
        }
    }
}
#endif
