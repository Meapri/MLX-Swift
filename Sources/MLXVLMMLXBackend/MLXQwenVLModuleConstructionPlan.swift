import Foundation
import MLXVLMCore

public struct MLXQwenVLModuleGroupPlan: Codable, Equatable, Sendable {
    public let phase: MLXBackendBindingPhase
    public let requiredBindingCount: Int
    public let presentBindingCount: Int
    public let shapeMatchedBindingCount: Int
    public let loadedArrayCount: Int
    public let missingRequiredKeys: [String]
    public let mismatchedShapeKeys: [String]
    public let loadedKeys: [String]
    public let constructible: Bool

    public init(
        phase: MLXBackendBindingPhase,
        bindings: [MLXBackendTensorBinding],
        loadedKeys: Set<String>,
        arrayBacked: Bool
    ) {
        let requiredBindings = bindings.filter(\.required)
        let phaseLoadedKeys = bindings
            .map(\.sanitizedKey)
            .filter { loadedKeys.contains($0) }
            .sorted()
        self.phase = phase
        self.requiredBindingCount = requiredBindings.count
        self.presentBindingCount = bindings.filter(\.present).count
        self.shapeMatchedBindingCount = bindings.filter { $0.shapeMatches == true }.count
        self.loadedArrayCount = phaseLoadedKeys.count
        self.missingRequiredKeys = requiredBindings
            .filter { !$0.present }
            .map(\.sanitizedKey)
            .sorted()
        self.mismatchedShapeKeys = bindings
            .filter { $0.shapeMatches == false }
            .map(\.sanitizedKey)
            .sorted()
        self.loadedKeys = phaseLoadedKeys
        self.constructible = arrayBacked &&
            !requiredBindings.isEmpty &&
            missingRequiredKeys.isEmpty &&
            mismatchedShapeKeys.isEmpty &&
            requiredBindings.allSatisfy { loadedKeys.contains($0.sanitizedKey) }
    }
}

public struct MLXQwenVLModuleConstructionPlan: Codable, Equatable, Sendable {
    public let modelID: String
    public let canonicalModelType: String
    public let family: QwenVLFamily?
    public let arrayBacked: Bool
    public let totalBindingCount: Int
    public let loadedArrayCount: Int
    public let constructibleGroupCount: Int
    public let moduleConstructionReady: Bool
    public let groups: [MLXQwenVLModuleGroupPlan]
    public let blockingReasons: [String]

    public init(container: MLXWeightBackedModelContainer) {
        let bindingPlan = container.preflight.bindingPlan
        let loadedKeys = Set(container.preflight.arrayLoadReport?.loadedKeys ?? [])
        let grouped = Dictionary(grouping: bindingPlan.bindings, by: \.phase)
        let phases: [MLXBackendBindingPhase] = [
            .tokenEmbedding,
            .languageLayer,
            .languageNorm,
            .languageHead,
            .visionPatchEmbedding,
            .visionBlock,
            .visionMerger,
            .unknown,
        ]
        self.modelID = container.context.descriptor.id
        self.canonicalModelType = container.context.descriptor.canonicalModelType
        self.family = bindingPlan.family
        self.arrayBacked = container.summary.arrayBacked
        self.totalBindingCount = bindingPlan.totalExpectedTensorBindings
        self.loadedArrayCount = loadedKeys.count
        self.groups = phases.compactMap { phase in
            guard let bindings = grouped[phase], !bindings.isEmpty else {
                return nil
            }
            return MLXQwenVLModuleGroupPlan(
                phase: phase,
                bindings: bindings,
                loadedKeys: loadedKeys,
                arrayBacked: container.summary.arrayBacked
            )
        }
        self.constructibleGroupCount = groups.filter(\.constructible).count
        self.moduleConstructionReady = container.summary.arrayBacked &&
            bindingPlan.missingRequiredKeys.isEmpty &&
            bindingPlan.mismatchedShapeKeys.isEmpty &&
            groups.contains { $0.phase == .tokenEmbedding && $0.constructible } &&
            groups.contains { $0.phase == .visionPatchEmbedding && $0.constructible }

        var reasons = container.summary.blockingReasons
        if !container.summary.arrayBacked {
            reasons.append("Loaded MLX arrays are required before Qwen VL modules can be constructed.")
        }
        if !bindingPlan.missingRequiredKeys.isEmpty {
            reasons.append("Required Qwen VL module tensors are missing.")
        }
        if !bindingPlan.mismatchedShapeKeys.isEmpty {
            reasons.append("One or more Qwen VL module tensors have incompatible shapes.")
        }
        if !moduleConstructionReady {
            reasons.append("Qwen VL module construction is not ready yet.")
        }
        self.blockingReasons = Self.unique(reasons)
    }

    private static func unique(_ values: [String]) -> [String] {
        var seen: Set<String> = []
        var result: [String] = []
        for value in values where !value.isEmpty && !seen.contains(value) {
            seen.insert(value)
            result.append(value)
        }
        return result
    }
}

public struct MLXQwenVLModuleConstructionPlanner {
    public init() {}

    public func plan(container: MLXWeightBackedModelContainer) -> MLXQwenVLModuleConstructionPlan {
        MLXQwenVLModuleConstructionPlan(container: container)
    }
}

