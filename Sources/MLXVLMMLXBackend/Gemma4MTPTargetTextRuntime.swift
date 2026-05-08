#if MLXVLM_REAL_MLX_API && MLXVLM_GEMMA4_ASSISTANT_RUNTIME && canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXNN)
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLMCore

private func gemma4MTPBuildLayerTypes(hiddenLayers: Int, slidingWindowPattern: Int) -> [String] {
    let pattern =
        Array(repeating: Gemma4AssistantLayerType.slidingAttention.rawValue, count: max(slidingWindowPattern - 1, 0))
        + [Gemma4AssistantLayerType.fullAttention.rawValue]
    guard !pattern.isEmpty else {
        return Array(repeating: Gemma4AssistantLayerType.fullAttention.rawValue, count: hiddenLayers)
    }
    var result: [String] = []
    result.reserveCapacity(hiddenLayers)
    while result.count < hiddenLayers {
        result.append(contentsOf: pattern)
    }
    return Array(result.prefix(hiddenLayers))
}

private func gemma4MTPDefaultRopeParameters() -> [String: [String: StringOrNumber]] {
    [
        Gemma4AssistantLayerType.fullAttention.rawValue: [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(1_000_000.0),
            "rope_type": .string("proportional"),
        ],
        Gemma4AssistantLayerType.slidingAttention.rawValue: [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(10_000.0),
            "rope_type": .string("default"),
        ],
    ]
}

private func gemma4MTPAdjustAttentionMask(
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

struct Gemma4MTPTargetTextConfiguration: Codable, Sendable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let kvHeads: Int
    let globalKVHeads: Int?
    let headDim: Int
    let globalHeadDim: Int
    let vocabularySize: Int
    let vocabularySizePerLayerInput: Int
    let numKVSharedLayers: Int
    let hiddenSizePerLayerInput: Int
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let maxPositionEmbeddings: Int
    let rmsNormEps: Float
    let ropeTraditional: Bool
    let finalLogitSoftcapping: Float?
    let useDoubleWideMLP: Bool
    let enableMoEBlock: Bool
    let attentionKEqV: Bool
    let layerTypes: [String]
    let ropeParameters: [String: [String: StringOrNumber]]
    let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case vocabularySize = "vocab_size"
        case vocabularySizePerLayerInput = "vocab_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTraditional = "rope_traditional"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMLP = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case attentionKEqV = "attention_k_eq_v"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_text"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 35
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        globalKVHeads = try c.decodeIfPresent(Int.self, forKey: .globalKVHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262_144
        vocabularySizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySizePerLayerInput) ?? vocabularySize
        numKVSharedLayers = try c.decodeIfPresent(Int.self, forKey: .numKVSharedLayers) ?? 20
        hiddenSizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 256
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern = try c.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        useDoubleWideMLP = try c.decodeIfPresent(Bool.self, forKey: .useDoubleWideMLP) ?? true
        enableMoEBlock = try c.decodeIfPresent(Bool.self, forKey: .enableMoEBlock) ?? false
        attentionKEqV = try c.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        ropeParameters =
            try c.decodeIfPresent([String: [String: StringOrNumber]].self, forKey: .ropeParameters)
            ?? gemma4MTPDefaultRopeParameters()
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? gemma4MTPBuildLayerTypes(
                hiddenLayers: hiddenLayers,
                slidingWindowPattern: slidingWindowPattern
            )
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
}

private struct Gemma4MTPTargetTopConfiguration: Codable, Sendable {
    let textConfiguration: Gemma4MTPTargetTextConfiguration
    let modelType: String
    let quantization: BaseConfiguration.Quantization?
    let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case modelType = "model_type"
        case quantization
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(Gemma4MTPTargetTextConfiguration.self, forKey: .textConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4"
        quantization = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? textConfiguration.tieWordEmbeddings
    }
}

final class Gemma4MTPTargetRMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

final class Gemma4MTPTargetScaledLinear: Module, UnaryLayer {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.scalar = scalar
        self._weight.wrappedValue = MLXArray.zeros([outFeatures, inFeatures])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x.matmul(weight.transposed()) * scalar
    }
}

final class Gemma4MTPTargetMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(config: Gemma4MTPTargetTextConfiguration, layerIndex: Int) {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        let isKVSharedLayer = layerIndex >= firstKVSharedLayer && firstKVSharedLayer > 0
        let useDoubleWide = config.useDoubleWideMLP && isKVSharedLayer
        let intermediateSize = config.intermediateSize * (useDoubleWide ? 2 : 1)
        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class Gemma4MTPTargetAttention: Module {
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float
    let useKEqV: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4AssistantRMSNormZeroShift
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4AssistantRMSNormZeroShift
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4MTPTargetRMSNormNoScale
    @ModuleInfo var rope: OffsetLayer

    init(config: Gemma4MTPTargetTextConfiguration, layerIndex: Int) {
        self.layerType = config.layerTypes[layerIndex]
        self.isSliding = layerType == Gemma4AssistantLayerType.slidingAttention.rawValue
        self.headDim =
            layerType == Gemma4AssistantLayerType.fullAttention.rawValue && config.globalHeadDim > 0
            ? config.globalHeadDim
            : config.headDim
        self.numHeads = config.attentionHeads
        self.useKEqV = config.attentionKEqV && !isSliding
        self.numKVHeads = useKEqV ? (config.globalKVHeads ?? config.kvHeads) : config.kvHeads
        self.scale = 1.0

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        if !useKEqV {
            self._vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        }
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = Gemma4AssistantRMSNormZeroShift(
            dimensions: headDim,
            eps: config.rmsNormEps
        )
        self._kNorm.wrappedValue = Gemma4AssistantRMSNormZeroShift(
            dimensions: headDim,
            eps: config.rmsNormEps
        )
        self._vNorm.wrappedValue = Gemma4MTPTargetRMSNormNoScale(eps: config.rmsNormEps)

        let ropeKey = isSliding
            ? Gemma4AssistantLayerType.slidingAttention.rawValue
            : Gemma4AssistantLayerType.fullAttention.rawValue
        let ropeConfig = config.ropeParameters[ropeKey]
        let ropeTheta = ropeConfig?["rope_theta"]?.asFloat() ?? (isSliding ? 10_000 : 1_000_000)
        self._rope.wrappedValue = initializeRope(
            dims: headDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        sharedKV: Gemma4AssistantSharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4AssistantSharedKVState, Int) {
        let batch = x.dim(0)
        let length = x.dim(1)
        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        queries = qNorm(queries)

        let currentOffset: Int
        let kvState: Gemma4AssistantSharedKVState
        if let sharedKV {
            currentOffset = offset ?? 0
            kvState = sharedKV
        } else {
            currentOffset = cache?.offset ?? 0
            var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
            var values = useKEqV
                ? keys
                : vProj!(x).reshaped(batch, length, numKVHeads, headDim)
            keys = kNorm(keys).transposed(0, 2, 1, 3)
            values = vNorm(values).transposed(0, 2, 1, 3)
            keys = rope(keys, offset: currentOffset)
            if let cache {
                let updated = cache.update(keys: keys, values: values)
                keys = updated.0
                values = updated.1
            }
            kvState = Gemma4AssistantSharedKVState(keys: keys, values: values)
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: currentOffset)
        let localMask = gemma4MTPAdjustAttentionMask(mask, keyLength: kvState.sequenceLength)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: kvState.keys,
            values: kvState.values,
            scale: scale,
            mask: localMask
        )
        return (
            oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)),
            kvState,
            currentOffset
        )
    }
}

final class Gemma4MTPTargetDecoderLayer: Module {
    let layerType: String

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4MTPTargetAttention
    @ModuleInfo var mlp: Gemma4MTPTargetMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4AssistantRMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4AssistantRMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        Gemma4AssistantRMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4AssistantRMSNormZeroShift
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4AssistantRMSNormZeroShift?
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(config: Gemma4MTPTargetTextConfiguration, layerIndex: Int) {
        self.layerType = config.layerTypes[layerIndex]
        self._selfAttention.wrappedValue = Gemma4MTPTargetAttention(config: config, layerIndex: layerIndex)
        self._mlp.wrappedValue = Gemma4MTPTargetMLP(config: config, layerIndex: layerIndex)
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
        if config.hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSizePerLayerInput,
                bias: false
            )
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput,
                config.hiddenSize,
                bias: false
            )
            self._postPerLayerInputNorm.wrappedValue = Gemma4AssistantRMSNormZeroShift(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps
            )
        }
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: Gemma4AssistantSharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4AssistantSharedKVState, Int) {
        var residual = x
        var h = inputLayerNorm(x)
        let attention = selfAttention(h, mask: mask, cache: cache, sharedKV: sharedKV, offset: offset)
        h = postAttentionLayerNorm(attention.0)
        h = residual + h

        residual = h
        h = preFeedforwardLayerNorm(h)
        h = mlp(h)
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        if let perLayerInputGate,
           let perLayerProjection,
           let postPerLayerInputNorm,
           let perLayerInput
        {
            residual = h
            var gated = perLayerInputGate(h)
            gated = geluApproximate(gated)
            gated = gated * perLayerInput
            gated = perLayerProjection(gated)
            gated = postPerLayerInputNorm(gated)
            h = residual + gated
        }

        return (h * layerScalar, attention.1, attention.2)
    }
}

struct Gemma4MTPTargetBackboneOutput {
    let normalizedHidden: MLXArray
    let preNormHidden: MLXArray
    let layerKVs: [Gemma4MTPTargetLayerKV]
}

final class Gemma4MTPTargetTextBackbone: Module {
    let config: Gemma4MTPTargetTextConfiguration
    let firstKVSharedLayerIdx: Int
    let layerIndexToCacheIndex: [Int]
    let firstFullCacheIndex: Int
    let firstSlidingCacheIndex: Int
    let embedScale: Float
    let embedTokensPerLayerScale: Float
    private let perLayerInputScale: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4MTPTargetDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma4AssistantRMSNormZeroShift
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Gemma4MTPTargetScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm:
        Gemma4AssistantRMSNormZeroShift?

    init(_ config: Gemma4MTPTargetTextConfiguration) {
        self.config = config
        self.firstKVSharedLayerIdx = max(0, config.hiddenLayers - config.numKVSharedLayers)
        self.embedScale = pow(Float(config.hiddenSize), 0.5)
        self.embedTokensPerLayerScale = pow(Float(max(config.hiddenSizePerLayerInput, 1)), 0.5)
        self.perLayerInputScale = pow(Float(2.0), -0.5)

        let concreteTypes = Array(config.layerTypes.prefix(firstKVSharedLayerIdx))
        let sharedFull = concreteTypes.lastIndex(of: Gemma4AssistantLayerType.fullAttention.rawValue) ?? 0
        let sharedSliding = concreteTypes.lastIndex(of: Gemma4AssistantLayerType.slidingAttention.rawValue) ?? 0
        var cacheMap: [Int] = []
        cacheMap.reserveCapacity(config.hiddenLayers)
        for (index, layerType) in config.layerTypes.enumerated() {
            if index < firstKVSharedLayerIdx {
                cacheMap.append(index)
            } else {
                cacheMap.append(
                    layerType == Gemma4AssistantLayerType.fullAttention.rawValue
                        ? sharedFull
                        : sharedSliding
                )
            }
        }
        self.layerIndexToCacheIndex = cacheMap
        self.firstFullCacheIndex = concreteTypes.firstIndex(of: Gemma4AssistantLayerType.fullAttention.rawValue) ?? 0
        self.firstSlidingCacheIndex =
            concreteTypes.firstIndex(of: Gemma4AssistantLayerType.slidingAttention.rawValue) ?? 0

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Gemma4MTPTargetDecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = Gemma4AssistantRMSNormZeroShift(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        if config.hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabularySizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput
            )
            self._perLayerModelProjection.wrappedValue = Gemma4MTPTargetScaledLinear(
                inFeatures: config.hiddenSize,
                outFeatures: config.hiddenLayers * config.hiddenSizePerLayerInput,
                scalar: pow(Float(config.hiddenSize), -0.5)
            )
            self._perLayerProjectionNorm.wrappedValue = Gemma4AssistantRMSNormZeroShift(
                dimensions: config.hiddenSizePerLayerInput,
                eps: config.rmsNormEps
            )
        }
        super.init()
    }

    func getPerLayerInputs(_ inputIDs: MLXArray) -> MLXArray {
        guard let embedTokensPerLayer else {
            fatalError("Gemma4 target per-layer inputs requested without embed_tokens_per_layer")
        }
        let valid = logicalAnd(
            inputIDs .>= 0,
            inputIDs .< config.vocabularySizePerLayerInput
        )
        let safeIDs = MLX.where(valid, inputIDs, MLXArray.zeros(like: inputIDs))
        var output = embedTokensPerLayer(safeIDs)
        output = (output * MLXArray(embedTokensPerLayerScale, dtype: .float32)).asType(output.dtype)
        return output.reshaped(
            Array(inputIDs.shape) + [config.hiddenLayers, config.hiddenSizePerLayerInput]
        )
    }

    func projectPerLayerInputs(
        _ inputsEmbeds: MLXArray,
        perLayerInputs: MLXArray?
    ) -> MLXArray? {
        guard let perLayerModelProjection, let perLayerProjectionNorm else {
            return nil
        }
        var projection = perLayerModelProjection(inputsEmbeds)
        projection = projection.reshaped(
            Array(inputsEmbeds.shape.dropLast()) + [
                config.hiddenLayers,
                config.hiddenSizePerLayerInput,
            ]
        )
        projection = perLayerProjectionNorm(projection)
        guard let perLayerInputs else {
            return projection
        }
        return (projection + perLayerInputs) * MLXArray(perLayerInputScale, dtype: inputsEmbeds.dtype)
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> Gemma4MTPTargetBackboneOutput {
        let h0: MLXArray
        if let inputsEmbeds {
            h0 = inputsEmbeds
        } else if let inputs {
            let embeddings = embedTokens(inputs)
            h0 = (embeddings * MLXArray(embedScale, dtype: .float32)).asType(embeddings.dtype)
        } else {
            fatalError("Gemma4 target text requires inputs or inputsEmbeds")
        }

        let rawPerLayerInputs: MLXArray?
        if config.hiddenSizePerLayerInput > 0 {
            if let perLayerInputs {
                rawPerLayerInputs = perLayerInputs
            } else if let inputs {
                rawPerLayerInputs = getPerLayerInputs(inputs)
            } else {
                rawPerLayerInputs = nil
            }
        } else {
            rawPerLayerInputs = nil
        }
        let projectedPerLayerInputs = projectPerLayerInputs(h0, perLayerInputs: rawPerLayerInputs)

        let hasExplicitCache = cache != nil
        let localCache = cache ?? Array(repeating: nil as KVCache?, count: max(firstKVSharedLayerIdx, 1))
        let fullMask: MLXFast.ScaledDotProductAttentionMaskMode
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            fullMask = mask
            slidingMask = mask
        } else {
            fullMask = createAttentionMask(
                h: h0,
                cache: firstFullCacheIndex < localCache.count ? localCache[firstFullCacheIndex] : nil
            )
            slidingMask = createAttentionMask(
                h: h0,
                cache: firstSlidingCacheIndex < localCache.count ? localCache[firstSlidingCacheIndex] : nil,
                windowSize: config.slidingWindow
            )
        }

        var h = h0
        var intermediates = [(kv: Gemma4AssistantSharedKVState?, offset: Int?)](
            repeating: (nil, nil),
            count: config.hiddenLayers
        )
        var layerKVs: [Gemma4MTPTargetLayerKV] = []
        layerKVs.reserveCapacity(config.hiddenLayers)

        for (index, layer) in layers.enumerated() {
            let sourceIndex = layerIndexToCacheIndex[index]
            let layerCache: KVCache? =
                if index < firstKVSharedLayerIdx, sourceIndex < localCache.count {
                    localCache[sourceIndex]
                } else {
                    nil
                }
            let layerMask = layer.layerType == Gemma4AssistantLayerType.fullAttention.rawValue
                ? fullMask
                : slidingMask
            let layerInput: MLXArray? =
                if let projectedPerLayerInputs {
                    projectedPerLayerInputs[0..., 0..., index, 0...]
                } else {
                    nil
                }
            let shared = hasExplicitCache && index >= firstKVSharedLayerIdx
                ? intermediates[sourceIndex].kv
                : nil
            let sharedOffset = hasExplicitCache && index >= firstKVSharedLayerIdx
                ? intermediates[sourceIndex].offset
                : nil
            let output = layer(
                h,
                mask: layerMask,
                cache: layerCache,
                perLayerInput: layerInput,
                sharedKV: shared,
                offset: sharedOffset
            )
            h = output.0
            intermediates[index] = (output.1, output.2)
            layerKVs.append(
                Gemma4MTPTargetLayerKV(
                    layerIndex: index,
                    layerType: layer.layerType,
                    keys: output.1.keys,
                    values: output.1.values
                )
            )
        }

        return Gemma4MTPTargetBackboneOutput(
            normalizedHidden: norm(h),
            preNormHidden: h,
            layerKVs: layerKVs
        )
    }
}

final class Gemma4MTPTargetTextModel: Module, BaseLanguageModel, Gemma4MTPTargetVerifier {
    let config: Gemma4MTPTargetTextConfiguration
    let finalLogitSoftcapping: Float?

    @ModuleInfo(key: "model") var model: Gemma4MTPTargetTextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: Gemma4MTPTargetTextConfiguration, tieWordEmbeddings: Bool) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self._model.wrappedValue = Gemma4MTPTargetTextBackbone(config)
        if !tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    func newCache() -> [any KVCache] {
        let slidingWindow = config.slidingWindow > 0 ? config.slidingWindow : 4096
        return config.layerTypes.prefix(model.firstKVSharedLayerIdx).map { layerType in
            if layerType == Gemma4AssistantLayerType.fullAttention.rawValue {
                return StandardKVCache()
            } else {
                return RotatingKVCache(maxSize: slidingWindow, keep: 0)
            }
        }
    }

    func embedTargetTokens(_ tokens: MLXArray) -> MLXArray {
        let embeddings = model.embedTokens(tokens)
        return (embeddings * MLXArray(model.embedScale, dtype: .float32)).asType(embeddings.dtype)
    }

    func prefill(inputIDs: MLXArray, cache: [any KVCache]?) -> Gemma4MTPTargetVerifyOutput {
        verify(inputIDs: inputIDs, cache: cache)
    }

    func verify(inputIDs: MLXArray, cache: [any KVCache]?) -> Gemma4MTPTargetVerifyOutput {
        let output = forward(inputIDs: inputIDs, cache: cache)
        return Gemma4MTPTargetVerifyOutput(
            logits: output.logits,
            hiddenStates: output.hiddenStates,
            sharedKVStates: output.sharedKVStates,
            cacheOffset: cache?.first?.offset ?? 0
        )
    }

    func rollback(cache: [any KVCache], accepted: Int, blockSize: Int) {
        _ = Gemma4MTPTargetAdapter(layerTypes: config.layerTypes)
            .rollbackScalarCache(cache, acceptedCount: accepted, blockSize: blockSize)
    }

    func forward(
        inputIDs: MLXArray,
        cache: [any KVCache]? = nil,
        inputsEmbeds: MLXArray? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> (logits: MLXArray, hiddenStates: MLXArray, sharedKVStates: [String: Gemma4AssistantSharedKVState]) {
        let typedCache = cache?.map { $0 as KVCache? }
        let output = model(
            inputsEmbeds == nil ? inputIDs : nil,
            inputsEmbeds: inputsEmbeds,
            cache: typedCache,
            perLayerInputs: perLayerInputs
        )
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(output.normalizedHidden)
        } else {
            logits = model.embedTokens.asLinear(output.normalizedHidden)
        }
        let softened: MLXArray
        if let finalLogitSoftcapping, finalLogitSoftcapping > 0 {
            let scale = MLXArray(finalLogitSoftcapping)
            softened = tanh(logits / scale) * scale
        } else {
            softened = logits
        }
        let shared = (try? Gemma4MTPTargetAdapter(layerTypes: config.layerTypes)
            .sharedKVStates(from: output.layerKVs)) ?? [:]
        return (softened, output.preNormHidden, shared)
    }

    func smokeReport(
        modelID: String,
        tokenIDs: [Int]
    ) throws -> Gemma4MTPTargetTextSmokeReport {
        let ids = tokenIDs.isEmpty ? [2] : tokenIDs
        let input = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
        let cache = newCache()
        let output = verify(inputIDs: input, cache: cache)
        eval(output.logits, output.hiddenStates)
        let sharedTypes = output.sharedKVStates.keys.sorted()
        let passed = output.logits.shape.prefix(2) == [1, ids.count] &&
            output.hiddenStates.shape == [1, ids.count, config.hiddenSize] &&
            sharedTypes.contains(Gemma4AssistantLayerType.fullAttention.rawValue) &&
            sharedTypes.contains(Gemma4AssistantLayerType.slidingAttention.rawValue)
        return Gemma4MTPTargetTextSmokeReport(
            realMLXAPIImplementationCompiled: true,
            assistantRuntimeCompiled: true,
            targetTextRuntimeCompiled: true,
            modelID: modelID,
            inputShape: input.shape,
            logitsShape: output.logits.shape,
            hiddenShape: output.hiddenStates.shape,
            sharedLayerTypes: sharedTypes,
            cacheCount: cache.count,
            cacheOffsets: cache.map(\.offset),
            passed: passed,
            error: passed ? nil : "Gemma4 target text smoke assertions failed."
        )
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            if key.contains("rotary_emb") {
                continue
            }
            if key.contains("vision_tower.") || key.contains("audio_tower.") || key.contains("multi_modal_projector.") {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("language_model.") {
                newKey.removeFirst("language_model.".count)
            }
            if newKey.hasPrefix("model.") || newKey.hasPrefix("lm_head.") {
                sanitized[newKey] = value
            }
        }

        if config.tieWordEmbeddings {
            sanitized = sanitized.filter { key, _ in
                !key.hasPrefix("lm_head.")
            }
        } else if sanitized["lm_head.weight"] == nil,
                  let embedWeight = sanitized["model.embed_tokens.weight"]
        {
            sanitized["lm_head.weight"] = embedWeight
        }
        return sanitized
    }
}

struct Gemma4MTPTargetTextLoadResult {
    let descriptor: ModelDescriptor
    let model: Gemma4MTPTargetTextModel
    let quantized: Bool
}

enum Gemma4MTPTargetTextLoaderError: Error, CustomStringConvertible {
    case invalidModelType(String)
    case unsupportedMoE

    var description: String {
        switch self {
        case .invalidModelType(let modelType):
            return "Target model is not Gemma4/Gemma4Text: \(modelType)"
        case .unsupportedMoE:
            return "Gemma4 target text runtime does not yet include the MoE branch."
        }
    }
}

struct Gemma4MTPTargetTextLoader {
    let store: ModelStore

    init(store: ModelStore = ModelStore()) {
        self.store = store
    }

    func load(pathOrIdentifier: String) throws -> Gemma4MTPTargetTextLoadResult {
        let descriptor = try store.loadDescriptor(pathOrIdentifier: pathOrIdentifier)
        let directory = URL(fileURLWithPath: descriptor.path)
        let configURL = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let topConfig = try JSONDecoder().decode(Gemma4MTPTargetTopConfiguration.self, from: configData)
        guard topConfig.modelType == "gemma4" || topConfig.modelType == "gemma4_text" else {
            throw Gemma4MTPTargetTextLoaderError.invalidModelType(topConfig.modelType)
        }
        guard !topConfig.textConfiguration.enableMoEBlock else {
            throw Gemma4MTPTargetTextLoaderError.unsupportedMoE
        }

        let model = Gemma4MTPTargetTextModel(
            topConfig.textConfiguration,
            tieWordEmbeddings: topConfig.tieWordEmbeddings
        )
        try loadWeights(
            modelDirectory: directory,
            model: model,
            quantization: topConfig.quantization
        )
        return Gemma4MTPTargetTextLoadResult(
            descriptor: descriptor,
            model: model,
            quantized: topConfig.quantization != nil
        )
    }
}
#endif
