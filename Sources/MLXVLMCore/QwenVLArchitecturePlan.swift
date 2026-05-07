import Foundation

public struct QwenVLExpectedTensor: Codable, Equatable, Sendable {
    public let key: String
    public let role: QwenVLWeightRole
    public let required: Bool
    public let expectedShape: [Int]?
    public let actualShape: [Int]?
    public let dtype: String?
    public let present: Bool
    public let shapeMatches: Bool?

    public init(
        key: String,
        role: QwenVLWeightRole,
        required: Bool = true,
        expectedShape: [Int]? = nil,
        actualShape: [Int]? = nil,
        dtype: String? = nil,
        present: Bool
    ) {
        self.key = key
        self.role = role
        self.required = required
        self.expectedShape = expectedShape
        self.actualShape = actualShape
        self.dtype = dtype
        self.present = present
        if let expectedShape, let actualShape {
            self.shapeMatches = expectedShape == actualShape
        } else {
            self.shapeMatches = nil
        }
    }
}

public struct QwenVLArchitecturePlan: Codable, Equatable, Sendable {
    public let family: QwenVLFamily
    public let textLayerCount: Int
    public let visionBlockCount: Int
    public let hiddenSize: Int
    public let visionHiddenSize: Int
    public let visionEmbedDim: Int?
    public let expectedCoreTensors: [QwenVLExpectedTensor]
    public let presentCoreTensorCount: Int
    public let missingRequiredKeys: [String]
    public let mismatchedShapeKeys: [String]
    public let coverage: Double

    public init(
        family: QwenVLFamily,
        textLayerCount: Int,
        visionBlockCount: Int,
        hiddenSize: Int,
        visionHiddenSize: Int,
        visionEmbedDim: Int?,
        expectedCoreTensors: [QwenVLExpectedTensor]
    ) {
        self.family = family
        self.textLayerCount = textLayerCount
        self.visionBlockCount = visionBlockCount
        self.hiddenSize = hiddenSize
        self.visionHiddenSize = visionHiddenSize
        self.visionEmbedDim = visionEmbedDim
        self.expectedCoreTensors = expectedCoreTensors.sorted { $0.key < $1.key }
        self.presentCoreTensorCount = expectedCoreTensors.filter(\.present).count
        self.missingRequiredKeys = expectedCoreTensors
            .filter { $0.required && !$0.present }
            .map(\.key)
            .sorted()
        self.mismatchedShapeKeys = expectedCoreTensors
            .filter { $0.shapeMatches == false }
            .map(\.key)
            .sorted()
        self.coverage = expectedCoreTensors.isEmpty
            ? 0
            : Double(self.presentCoreTensorCount) / Double(expectedCoreTensors.count)
    }
}

public struct QwenVLArchitecturePlanner {
    public init() {}

    public func plan(config: QwenVLModelConfig, weightCatalog: WeightCatalog) -> QwenVLArchitecturePlan {
        let tensorsByKey = Dictionary(uniqueKeysWithValues: weightCatalog.tensors.map { ($0.sanitizedKey, $0) })
        let expected = expectedCoreTensorSpecs(config: config).map { spec in
            let tensor = tensorsByKey[spec.key]
            return QwenVLExpectedTensor(
                key: spec.key,
                role: spec.role,
                required: spec.required,
                expectedShape: spec.expectedShape,
                actualShape: tensor?.shape,
                dtype: tensor?.dtype,
                present: tensor != nil
            )
        }
        return QwenVLArchitecturePlan(
            family: config.family,
            textLayerCount: config.textConfig.numHiddenLayers,
            visionBlockCount: config.visionConfig.depth,
            hiddenSize: config.textConfig.hiddenSize,
            visionHiddenSize: config.visionConfig.hiddenSize,
            visionEmbedDim: config.visionConfig.embedDim,
            expectedCoreTensors: expected
        )
    }

    private struct TensorSpec {
        let key: String
        let role: QwenVLWeightRole
        let required: Bool
        let expectedShape: [Int]?
    }

    private func expectedCoreTensorSpecs(config: QwenVLModelConfig) -> [TensorSpec] {
        let text = config.textConfig
        let vision = config.visionConfig
        let headDim = text.hiddenSize / text.numAttentionHeads
        let qOut = text.numAttentionHeads * headDim
        let kvOut = text.numKeyValueHeads * headDim
        var specs: [TensorSpec] = [
            TensorSpec(
                key: "language_model.model.embed_tokens.weight",
                role: .languageModel,
                required: true,
                expectedShape: [text.vocabSize, text.hiddenSize]
            ),
            TensorSpec(
                key: "language_model.model.norm.weight",
                role: .languageModel,
                required: true,
                expectedShape: [text.hiddenSize]
            ),
            TensorSpec(
                key: "language_model.lm_head.weight",
                role: .languageHead,
                required: !text.tieWordEmbeddings,
                expectedShape: [text.vocabSize, text.hiddenSize]
            ),
            TensorSpec(
                key: "vision_tower.patch_embed.proj.weight",
                role: .visionTower,
                required: true,
                expectedShape: nil
            ),
            TensorSpec(
                key: "vision_tower.merger.ln_q.weight",
                role: .visionTower,
                required: true,
                expectedShape: nil
            ),
        ]

        for layer in 0..<text.numHiddenLayers {
            let prefix = "language_model.model.layers.\(layer)"
            specs += [
                TensorSpec(key: "\(prefix).self_attn.q_proj.weight", role: .languageModel, required: true, expectedShape: [qOut, text.hiddenSize]),
                TensorSpec(key: "\(prefix).self_attn.q_proj.bias", role: .languageModel, required: true, expectedShape: [qOut]),
                TensorSpec(key: "\(prefix).self_attn.k_proj.weight", role: .languageModel, required: true, expectedShape: [kvOut, text.hiddenSize]),
                TensorSpec(key: "\(prefix).self_attn.k_proj.bias", role: .languageModel, required: true, expectedShape: [kvOut]),
                TensorSpec(key: "\(prefix).self_attn.v_proj.weight", role: .languageModel, required: true, expectedShape: [kvOut, text.hiddenSize]),
                TensorSpec(key: "\(prefix).self_attn.v_proj.bias", role: .languageModel, required: true, expectedShape: [kvOut]),
                TensorSpec(key: "\(prefix).self_attn.o_proj.weight", role: .languageModel, required: true, expectedShape: [text.hiddenSize, qOut]),
                TensorSpec(key: "\(prefix).mlp.gate_proj.weight", role: .languageModel, required: true, expectedShape: [text.intermediateSize, text.hiddenSize]),
                TensorSpec(key: "\(prefix).mlp.up_proj.weight", role: .languageModel, required: true, expectedShape: [text.intermediateSize, text.hiddenSize]),
                TensorSpec(key: "\(prefix).mlp.down_proj.weight", role: .languageModel, required: true, expectedShape: [text.hiddenSize, text.intermediateSize]),
                TensorSpec(key: "\(prefix).input_layernorm.weight", role: .languageModel, required: true, expectedShape: [text.hiddenSize]),
                TensorSpec(key: "\(prefix).post_attention_layernorm.weight", role: .languageModel, required: true, expectedShape: [text.hiddenSize]),
            ]
        }

        for block in 0..<vision.depth {
            let prefix = "vision_tower.blocks.\(block)"
            specs += [
                TensorSpec(key: "\(prefix).norm1.weight", role: .visionTower, required: true, expectedShape: nil),
                TensorSpec(key: "\(prefix).attn.qkv.weight", role: .visionTower, required: true, expectedShape: nil),
                TensorSpec(key: "\(prefix).attn.proj.weight", role: .visionTower, required: true, expectedShape: nil),
                TensorSpec(key: "\(prefix).mlp.fc1.weight", role: .visionTower, required: true, expectedShape: nil),
                TensorSpec(key: "\(prefix).mlp.fc2.weight", role: .visionTower, required: true, expectedShape: nil),
            ]
        }

        return specs
    }
}
