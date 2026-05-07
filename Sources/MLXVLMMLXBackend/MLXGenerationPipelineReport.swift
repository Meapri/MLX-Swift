import Foundation
import MLXVLMCore

public struct MLXGenerationPipelineReport: Codable, Equatable, Sendable {
    public let modelID: String
    public let canonicalModelType: String
    public let requestModel: String
    public let load: MLXBackendLoadPreflightReport
    public let container: MLXWeightBackedModelContainerSummary
    public let moduleConstruction: MLXQwenVLModuleConstructionPlan
    public let forward: MLXQwenVLForwardPlan
    public let generationLoop: MLXQwenVLGenerationLoopPlan
    public let generateParametersBridge: MLXGenerateParametersBridgeReport
    public let modelFactoryBridge: MLXVLMModelFactoryBridgeReport
    public let decodeState: MLXQwenVLDecodeStatePlan
    public let pipelineReady: Bool
    public let blockingReasons: [String]

    public init(
        request: GenerationRequest,
        container: MLXWeightBackedModelContainer,
        moduleConstruction: MLXQwenVLModuleConstructionPlan,
        forward: MLXQwenVLForwardPlan,
        generationLoop: MLXQwenVLGenerationLoopPlan,
        generateParametersBridge: MLXGenerateParametersBridgeReport,
        modelFactoryBridge: MLXVLMModelFactoryBridgeReport,
        decodeState: MLXQwenVLDecodeStatePlan
    ) {
        self.modelID = container.context.descriptor.id
        self.canonicalModelType = container.context.descriptor.canonicalModelType
        self.requestModel = request.model
        self.load = container.preflight
        self.container = container.summary
        self.moduleConstruction = moduleConstruction
        self.forward = forward
        self.generationLoop = generationLoop
        self.generateParametersBridge = generateParametersBridge
        self.modelFactoryBridge = modelFactoryBridge
        self.decodeState = decodeState
        self.pipelineReady = generationLoop.generationLoopReady &&
            generateParametersBridge.canBridgeToGenerateParameters &&
            modelFactoryBridge.canLoadLocalModelContainer
        self.blockingReasons = Self.unique(
            container.summary.blockingReasons +
                moduleConstruction.blockingReasons +
                forward.blockingReasons +
                generationLoop.blockingReasons +
                generateParametersBridge.blockingReasons +
                modelFactoryBridge.blockingReasons +
                decodeState.blockingReasons
        )
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

public struct MLXGenerationPipelineReporter {
    public init() {}

    public func report(
        descriptor: ModelDescriptor,
        request: GenerationRequest,
        rootPath: String = FileManager.default.currentDirectoryPath,
        weightOptions: MLXWeightPreparationOptions = MLXWeightPreparationOptions()
    ) throws -> MLXGenerationPipelineReport {
        let container = MLXBackendFactory.loadWeightBackedContainer(
            descriptor: descriptor,
            rootPath: rootPath,
            weightOptions: weightOptions
        )
        let modulePlan = MLXQwenVLModuleConstructionPlanner().plan(container: container)
        let input = try CompatibilityProcessor().process(
            request: request,
            context: container.context
        )
        let forwardPlan = MLXQwenVLForwardPlanner().plan(
            container: container,
            modulePlan: modulePlan,
            input: input
        )
        let generationLoopPlan = MLXQwenVLGenerationLoopPlanner().plan(
            forwardPlan: forwardPlan,
            input: input
        )
        let generateParametersBridge = MLXGenerateParametersBridge().report(
            for: generationLoopPlan.generateParameters
        )
        let modelFactoryBridge = MLXVLMModelFactoryBridge().report(for: descriptor)
        let decodeStatePlan = MLXQwenVLDecodeStatePlanner().plan(
            container: container,
            forwardPlan: forwardPlan,
            generationLoopPlan: generationLoopPlan,
            input: input
        )
        return MLXGenerationPipelineReport(
            request: request,
            container: container,
            moduleConstruction: modulePlan,
            forward: forwardPlan,
            generationLoop: generationLoopPlan,
            generateParametersBridge: generateParametersBridge,
            modelFactoryBridge: modelFactoryBridge,
            decodeState: decodeStatePlan
        )
    }
}
