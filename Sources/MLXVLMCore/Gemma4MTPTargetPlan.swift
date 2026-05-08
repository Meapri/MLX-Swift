import Foundation

public struct Gemma4MTPKVProducerPlan: Codable, Equatable, Sendable {
    public let layerType: String
    public let producerLayerIndex: Int

    public init(layerType: String, producerLayerIndex: Int) {
        self.layerType = layerType
        self.producerLayerIndex = producerLayerIndex
    }
}

public struct Gemma4MTPTargetPlan: Codable, Equatable, Sendable {
    public let targetModelType: String
    public let targetArchitectures: [String]
    public let isGemma4Target: Bool
    public let targetHiddenSize: Int?
    public let targetLayerCount: Int?
    public let targetLayerTypes: [String]
    public let targetKVSharedLayerCount: Int?
    public let firstKVSharedLayerIndex: Int?
    public let requiredSharedLayerTypes: [String]
    public let kvProducerLayers: [Gemma4MTPKVProducerPlan]
    public let draftModelType: String?
    public let draftArchitectures: [String]
    public let isGemma4AssistantDraft: Bool
    public let draftBackboneHiddenSize: Int?
    public let draftBlockSize: Int?
    public let hiddenSizeMatches: Bool
    public let draftKindRequired: String?
    public let targetForwardRequirements: [String]
    public let blockingReasons: [String]

    public init(
        targetModelType: String,
        targetArchitectures: [String],
        isGemma4Target: Bool,
        targetHiddenSize: Int?,
        targetLayerCount: Int?,
        targetLayerTypes: [String],
        targetKVSharedLayerCount: Int?,
        firstKVSharedLayerIndex: Int?,
        requiredSharedLayerTypes: [String],
        kvProducerLayers: [Gemma4MTPKVProducerPlan],
        draftModelType: String?,
        draftArchitectures: [String],
        isGemma4AssistantDraft: Bool,
        draftBackboneHiddenSize: Int?,
        draftBlockSize: Int?,
        hiddenSizeMatches: Bool,
        draftKindRequired: String?,
        targetForwardRequirements: [String],
        blockingReasons: [String]
    ) {
        self.targetModelType = targetModelType
        self.targetArchitectures = targetArchitectures
        self.isGemma4Target = isGemma4Target
        self.targetHiddenSize = targetHiddenSize
        self.targetLayerCount = targetLayerCount
        self.targetLayerTypes = targetLayerTypes
        self.targetKVSharedLayerCount = targetKVSharedLayerCount
        self.firstKVSharedLayerIndex = firstKVSharedLayerIndex
        self.requiredSharedLayerTypes = requiredSharedLayerTypes
        self.kvProducerLayers = kvProducerLayers
        self.draftModelType = draftModelType
        self.draftArchitectures = draftArchitectures
        self.isGemma4AssistantDraft = isGemma4AssistantDraft
        self.draftBackboneHiddenSize = draftBackboneHiddenSize
        self.draftBlockSize = draftBlockSize
        self.hiddenSizeMatches = hiddenSizeMatches
        self.draftKindRequired = draftKindRequired
        self.targetForwardRequirements = targetForwardRequirements
        self.blockingReasons = blockingReasons
    }

    public var metadataReady: Bool {
        isGemma4Target &&
            isGemma4AssistantDraft &&
            hiddenSizeMatches &&
            !kvProducerLayers.isEmpty
    }
}

public struct Gemma4MTPTargetPlanner {
    public init() {}

    public func plan(
        targetDescriptor: ModelDescriptor,
        draftDescriptor: ModelDescriptor? = nil
    ) -> Gemma4MTPTargetPlan {
        let store = ModelStore()
        let targetConfig = (try? store.loadNormalizedConfig(pathOrIdentifier: targetDescriptor.path)) ?? .object([:])
        let targetObject = targetConfig.objectValue ?? [:]
        let targetTextObject = targetObject["text_config"]?.objectValue ?? targetObject
        let targetModelType = targetObject.string("model_type", default: targetDescriptor.rawModelType) ??
            targetDescriptor.rawModelType
        let targetArchitectures = targetObject["architectures"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let isTarget = targetModelType == "gemma4" ||
            targetModelType == "gemma4_text" ||
            targetArchitectures.contains("Gemma4ForConditionalGeneration") ||
            targetArchitectures.contains("Gemma4Model")
        let targetHiddenSize = targetTextObject["hidden_size"]?.intValue ?? targetObject["hidden_size"]?.intValue
        let targetLayerCount = targetTextObject["num_hidden_layers"]?.intValue
        let kvSharedCount = targetTextObject["num_kv_shared_layers"]?.intValue
        let slidingPattern = targetTextObject["sliding_window_pattern"]?.intValue ?? 5
        let layerTypes = Self.layerTypes(
            explicit: targetTextObject["layer_types"]?.arrayValue?.compactMap(\.stringValue),
            hiddenLayers: targetLayerCount,
            slidingWindowPattern: slidingPattern
        )
        let firstShared = Self.firstSharedLayerIndex(
            hiddenLayers: targetLayerCount,
            kvSharedLayers: kvSharedCount
        )
        let producers = Self.kvProducerLayers(
            layerTypes: layerTypes,
            firstSharedLayerIndex: firstShared
        )
        let requiredTypes = producers.map(\.layerType).sorted()

        var draftModelType: String?
        var draftArchitectures: [String] = []
        var isDraft = false
        var draftBackboneHiddenSize: Int?
        var draftBlockSize: Int?
        if let draftDescriptor {
            let draftConfig = (try? store.loadNormalizedConfig(pathOrIdentifier: draftDescriptor.path)) ?? .object([:])
            let draftObject = draftConfig.objectValue ?? [:]
            draftModelType = draftObject.string("model_type", default: draftDescriptor.rawModelType)
            draftArchitectures = draftObject["architectures"]?.arrayValue?.compactMap(\.stringValue) ?? []
            isDraft = draftModelType == "gemma4_assistant" ||
                draftArchitectures.contains("Gemma4AssistantForCausalLM")
            draftBackboneHiddenSize = draftObject["backbone_hidden_size"]?.intValue
            draftBlockSize = draftObject["block_size"]?.intValue
        }

        let hiddenMatches =
            draftDescriptor == nil ||
            (targetHiddenSize != nil && draftBackboneHiddenSize != nil && targetHiddenSize == draftBackboneHiddenSize)

        var blockingReasons: [String] = []
        if !isTarget {
            blockingReasons.append("Target model config is not Gemma4/Gemma4Text.")
        }
        if draftDescriptor != nil && !isDraft {
            blockingReasons.append("Draft model config is not Gemma4AssistantForCausalLM.")
        }
        if !hiddenMatches {
            blockingReasons.append("Gemma4 target hidden_size does not match assistant backbone_hidden_size.")
        }
        if producers.isEmpty {
            blockingReasons.append("No target K/V producer layers were discovered for Gemma4 MTP shared-KV export.")
        }
        blockingReasons.append("Real Gemma4 target forward still must call the Swift MTP adapter with hidden states and shared K/V tensors.")

        return Gemma4MTPTargetPlan(
            targetModelType: targetModelType,
            targetArchitectures: targetArchitectures,
            isGemma4Target: isTarget,
            targetHiddenSize: targetHiddenSize,
            targetLayerCount: targetLayerCount,
            targetLayerTypes: layerTypes,
            targetKVSharedLayerCount: kvSharedCount,
            firstKVSharedLayerIndex: firstShared,
            requiredSharedLayerTypes: requiredTypes,
            kvProducerLayers: producers,
            draftModelType: draftModelType,
            draftArchitectures: draftArchitectures,
            isGemma4AssistantDraft: draftDescriptor == nil ? false : isDraft,
            draftBackboneHiddenSize: draftBackboneHiddenSize,
            draftBlockSize: draftBlockSize,
            hiddenSizeMatches: hiddenMatches,
            draftKindRequired: isDraft ? "mtp" : nil,
            targetForwardRequirements: [
                "prefill must return logits plus the full prompt hidden state so MTP can keep the last prompt slot",
                "verify must return logits plus per-token hidden states for the bonus-plus-draft block",
                "target forward must export K/V pairs from the latest non-shared producer layer of each layer_type",
                "rollback must trim rejected target KV cache positions after each MTP round",
                "target token embedding must be bound to the assistant draft model",
            ],
            blockingReasons: blockingReasons
        )
    }

    private static func layerTypes(
        explicit: [String]?,
        hiddenLayers: Int?,
        slidingWindowPattern: Int
    ) -> [String] {
        guard let hiddenLayers, hiddenLayers > 0 else {
            return []
        }
        if let explicit, explicit.count == hiddenLayers {
            return explicit
        }
        let pattern =
            Array(repeating: "sliding_attention", count: max(slidingWindowPattern - 1, 0))
            + ["full_attention"]
        guard !pattern.isEmpty else {
            return Array(repeating: "full_attention", count: hiddenLayers)
        }
        var output: [String] = []
        while output.count < hiddenLayers {
            output.append(contentsOf: pattern)
        }
        return Array(output.prefix(hiddenLayers))
    }

    private static func firstSharedLayerIndex(
        hiddenLayers: Int?,
        kvSharedLayers: Int?
    ) -> Int? {
        guard let hiddenLayers, let kvSharedLayers else {
            return nil
        }
        return max(0, hiddenLayers - kvSharedLayers)
    }

    private static func kvProducerLayers(
        layerTypes: [String],
        firstSharedLayerIndex: Int?
    ) -> [Gemma4MTPKVProducerPlan] {
        guard !layerTypes.isEmpty else {
            return []
        }
        let searchEnd = firstSharedLayerIndex.map { max(0, min($0, layerTypes.count)) } ?? layerTypes.count
        var lastByType: [String: Int] = [:]
        for index in 0 ..< searchEnd {
            lastByType[layerTypes[index]] = index
        }
        return lastByType
            .map { Gemma4MTPKVProducerPlan(layerType: $0.key, producerLayerIndex: $0.value) }
            .sorted { lhs, rhs in
                lhs.layerType == rhs.layerType
                    ? lhs.producerLayerIndex < rhs.producerLayerIndex
                    : lhs.layerType < rhs.layerType
            }
    }
}
