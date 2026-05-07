import Foundation

public enum MLXBackendBindingPhase: String, Codable, Equatable, Sendable {
    case tokenEmbedding = "token-embedding"
    case languageLayer = "language-layer"
    case languageNorm = "language-norm"
    case languageHead = "language-head"
    case visionPatchEmbedding = "vision-patch-embedding"
    case visionBlock = "vision-block"
    case visionMerger = "vision-merger"
    case unknown
}

public struct MLXBackendTensorBinding: Codable, Equatable, Sendable {
    public let sanitizedKey: String
    public let role: QwenVLWeightRole
    public let phase: MLXBackendBindingPhase
    public let required: Bool
    public let present: Bool
    public let expectedShape: [Int]?
    public let actualShape: [Int]?
    public let dtype: String?
    public let shapeMatches: Bool?

    public init(expectedTensor: QwenVLExpectedTensor) {
        self.sanitizedKey = expectedTensor.key
        self.role = expectedTensor.role
        self.phase = Self.phase(for: expectedTensor.key, role: expectedTensor.role)
        self.required = expectedTensor.required
        self.present = expectedTensor.present
        self.expectedShape = expectedTensor.expectedShape
        self.actualShape = expectedTensor.actualShape
        self.dtype = expectedTensor.dtype
        self.shapeMatches = expectedTensor.shapeMatches
    }

    private static func phase(for key: String, role: QwenVLWeightRole) -> MLXBackendBindingPhase {
        if key == "language_model.model.embed_tokens.weight" {
            return .tokenEmbedding
        }
        if key == "language_model.model.norm.weight" {
            return .languageNorm
        }
        if role == .languageHead {
            return .languageHead
        }
        if key.contains(".layers.") {
            return .languageLayer
        }
        if key == "vision_tower.patch_embed.proj.weight" {
            return .visionPatchEmbedding
        }
        if key.hasPrefix("vision_tower.blocks.") {
            return .visionBlock
        }
        if key.hasPrefix("vision_tower.merger.") {
            return .visionMerger
        }
        return .unknown
    }
}

public struct MLXBackendBindingPlan: Codable, Equatable, Sendable {
    public let modelID: String
    public let canonicalModelType: String
    public let family: QwenVLFamily?
    public let supportedBySwiftScaffold: Bool
    public let bindings: [MLXBackendTensorBinding]
    public let totalExpectedTensorBindings: Int
    public let presentTensorBindings: Int
    public let missingRequiredKeys: [String]
    public let mismatchedShapeKeys: [String]
    public let phaseCounts: [String: Int]
    public let blockingReasons: [String]

    public init(descriptor: ModelDescriptor, loadPlan: ModelLoadPlan) {
        self.modelID = descriptor.id
        self.canonicalModelType = descriptor.canonicalModelType
        self.family = loadPlan.qwenVLConfig?.family
        self.supportedBySwiftScaffold = loadPlan.qwenVLConfig != nil && loadPlan.qwenVLArchitecture != nil
        self.bindings = (loadPlan.qwenVLArchitecture?.expectedCoreTensors ?? [])
            .map(MLXBackendTensorBinding.init(expectedTensor:))
            .sorted { $0.sanitizedKey < $1.sanitizedKey }
        self.totalExpectedTensorBindings = bindings.count
        self.presentTensorBindings = bindings.filter(\.present).count
        self.missingRequiredKeys = loadPlan.qwenVLArchitecture?.missingRequiredKeys ?? []
        self.mismatchedShapeKeys = loadPlan.qwenVLArchitecture?.mismatchedShapeKeys ?? []
        self.phaseCounts = Dictionary(
            grouping: bindings,
            by: { $0.phase.rawValue }
        ).mapValues(\.count)

        var blockers: [String] = []
        if loadPlan.qwenVLConfig == nil {
            blockers.append("Qwen VL config could not be parsed for this model.")
        }
        if !missingRequiredKeys.isEmpty {
            blockers.append("Required Qwen VL tensors are missing from the safetensors catalog.")
        }
        if !mismatchedShapeKeys.isEmpty {
            blockers.append("One or more Qwen VL tensors have shapes that do not match the parsed config.")
        }
        blockers.append("MLX arrays, module construction, and generation sampling are not implemented yet.")
        self.blockingReasons = blockers
    }
}

public struct MLXBackendBindingPlanner {
    public init() {}

    public func plan(descriptor: ModelDescriptor) -> MLXBackendBindingPlan {
        let loadPlan = ModelLoadPlanner().plan(descriptor: descriptor)
        return MLXBackendBindingPlan(descriptor: descriptor, loadPlan: loadPlan)
    }
}
