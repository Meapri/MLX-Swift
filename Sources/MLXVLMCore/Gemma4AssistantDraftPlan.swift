import Foundation

public struct Gemma4AssistantDraftPlan: Codable, Equatable, Sendable {
    public let modelType: String
    public let architectures: [String]
    public let isGemma4Assistant: Bool
    public let requiredDraftKind: String?
    public let canUseGenericSpeculativeDecoder: Bool
    public let nativeSwiftMTPReady: Bool
    public let blockSize: Int?
    public let backboneHiddenSize: Int?
    public let draftHiddenSize: Int?
    public let draftLayerCount: Int?
    public let draftAttentionHeads: Int?
    public let draftKVHeads: Int?
    public let draftHeadDim: Int?
    public let slidingWindow: Int?
    public let numKVSharedLayers: Int?
    public let useOrderedEmbeddings: Bool
    public let numCentroids: Int?
    public let centroidIntermediateTopK: Int?
    public let tieWordEmbeddings: Bool
    public let expectedTensorKeys: [String]
    public let presentTensorKeys: [String]
    public let missingTensorKeys: [String]
    public let unexpectedCriticalTensorKeys: [String]
    public let targetRuntimeRequirements: [String]
    public let blockingReasons: [String]

    public var tensorCoverageReady: Bool {
        missingTensorKeys.isEmpty && unexpectedCriticalTensorKeys.isEmpty
    }

    public var metadataReady: Bool {
        isGemma4Assistant && draftHiddenSize != nil && draftLayerCount != nil
    }
}

public struct Gemma4AssistantDraftPlanner {
    public init() {}

    public func plan(descriptor: ModelDescriptor, weightCatalog: WeightCatalog) -> Gemma4AssistantDraftPlan {
        let config = (try? ModelStore().loadNormalizedConfig(pathOrIdentifier: descriptor.path)) ?? .object([:])
        let modelType = config["model_type"]?.stringValue ?? descriptor.rawModelType
        let architectures = config["architectures"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let isGemma4Assistant = modelType == "gemma4_assistant" ||
            architectures.contains("Gemma4AssistantForCausalLM")
        let textConfig = config["text_config"]?.objectValue
        let draftLayerCount = textConfig?["num_hidden_layers"]?.intValue
        let useOrderedEmbeddings = config["use_ordered_embeddings"]?.boolValue ?? false
        let tieWordEmbeddings = config["tie_word_embeddings"]?.boolValue ??
            textConfig?["tie_word_embeddings"]?.boolValue ?? true
        let expectedKeys = Self.expectedTensorKeys(
            layerCount: draftLayerCount ?? 0,
            useOrderedEmbeddings: useOrderedEmbeddings,
            tieWordEmbeddings: tieWordEmbeddings
        )
        let presentSet = Set(weightCatalog.tensors.map(\.originalKey))
        let missing = expectedKeys.filter { !presentSet.contains($0) }
        let criticalUnexpected = Self.unexpectedCriticalTensorKeys(in: presentSet)

        var blockingReasons: [String] = []
        if !isGemma4Assistant {
            blockingReasons.append("Draft model config is not Gemma4AssistantForCausalLM.")
        }
        if !missing.isEmpty {
            blockingReasons.append("Missing Gemma4 assistant tensor keys: \(missing.joined(separator: ","))")
        }
        if !criticalUnexpected.isEmpty {
            blockingReasons.append("Draft checkpoint contains generic Gemma4 text attention tensors that are not part of kv-shared assistant decoding: \(criticalUnexpected.joined(separator: ","))")
        }
        blockingReasons.append("Native Swift Gemma4 MTP still requires target Gemma4 hidden-state capture, shared K/V export, assistant draft_block, and an MTP verification/rollback loop; generic mlx-swift-lm speculative decoding is insufficient.")

        return Gemma4AssistantDraftPlan(
            modelType: modelType,
            architectures: architectures,
            isGemma4Assistant: isGemma4Assistant,
            requiredDraftKind: isGemma4Assistant ? "mtp" : nil,
            canUseGenericSpeculativeDecoder: false,
            nativeSwiftMTPReady: false,
            blockSize: config["block_size"]?.intValue,
            backboneHiddenSize: config["backbone_hidden_size"]?.intValue,
            draftHiddenSize: textConfig?["hidden_size"]?.intValue,
            draftLayerCount: draftLayerCount,
            draftAttentionHeads: textConfig?["num_attention_heads"]?.intValue,
            draftKVHeads: textConfig?["num_key_value_heads"]?.intValue,
            draftHeadDim: textConfig?["head_dim"]?.intValue,
            slidingWindow: textConfig?["sliding_window"]?.intValue,
            numKVSharedLayers: textConfig?["num_kv_shared_layers"]?.intValue,
            useOrderedEmbeddings: useOrderedEmbeddings,
            numCentroids: config["num_centroids"]?.intValue,
            centroidIntermediateTopK: config["centroid_intermediate_top_k"]?.intValue,
            tieWordEmbeddings: tieWordEmbeddings,
            expectedTensorKeys: expectedKeys,
            presentTensorKeys: expectedKeys.filter { presentSet.contains($0) },
            missingTensorKeys: missing,
            unexpectedCriticalTensorKeys: criticalUnexpected,
            targetRuntimeRequirements: [
                "target Gemma4 prefill must return the last pre-norm hidden state",
                "target Gemma4 decode must return shared K/V states keyed by layer_type",
                "assistant must bind target token embeddings for concat([target_embed(last_token), last_hidden])",
                "assistant draft_block must run autoregressive MTP candidates at a constant position id",
                "target verification must roll back rejected KV-cache positions after each MTP round",
            ],
            blockingReasons: blockingReasons
        )
    }

    private static func expectedTensorKeys(
        layerCount: Int,
        useOrderedEmbeddings: Bool,
        tieWordEmbeddings: Bool
    ) -> [String] {
        var keys = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "post_projection.weight",
            "pre_projection.weight",
        ]
        if useOrderedEmbeddings {
            keys.append("masked_embedding.centroids.weight")
            keys.append("masked_embedding.token_ordering")
        }
        if !tieWordEmbeddings {
            keys.append("lm_head.weight")
        }
        for index in 0 ..< layerCount {
            let prefix = "model.layers.\(index)"
            keys.append(contentsOf: [
                "\(prefix).input_layernorm.weight",
                "\(prefix).layer_scalar",
                "\(prefix).mlp.down_proj.weight",
                "\(prefix).mlp.gate_proj.weight",
                "\(prefix).mlp.up_proj.weight",
                "\(prefix).post_attention_layernorm.weight",
                "\(prefix).post_feedforward_layernorm.weight",
                "\(prefix).pre_feedforward_layernorm.weight",
                "\(prefix).self_attn.o_proj.weight",
                "\(prefix).self_attn.q_norm.weight",
                "\(prefix).self_attn.q_proj.weight",
            ])
        }
        return keys.sorted()
    }

    private static func unexpectedCriticalTensorKeys(in keys: Set<String>) -> [String] {
        keys.filter {
            $0.contains(".self_attn.k_proj.") ||
                $0.contains(".self_attn.v_proj.") ||
                $0.contains(".self_attn.k_norm.") ||
                $0.contains(".self_attn.v_norm.")
        }.sorted()
    }
}
