import Foundation

public struct ModelCapabilityPlan: Codable, Equatable, Sendable {
    public let canonicalModelType: String
    public let primaryTask: String
    public let supportsTextGeneration: Bool
    public let supportsVisionInputs: Bool
    public let supportsVideoInputs: Bool
    public let supportsAudioInputs: Bool
    public let supportsOllamaGenerationAPI: Bool
    public let preferredSwiftEntryPoint: String
    public let warnings: [String]

    public init(
        canonicalModelType: String,
        primaryTask: String,
        supportsTextGeneration: Bool,
        supportsVisionInputs: Bool,
        supportsVideoInputs: Bool,
        supportsAudioInputs: Bool,
        supportsOllamaGenerationAPI: Bool,
        preferredSwiftEntryPoint: String,
        warnings: [String]
    ) {
        self.canonicalModelType = canonicalModelType
        self.primaryTask = primaryTask
        self.supportsTextGeneration = supportsTextGeneration
        self.supportsVisionInputs = supportsVisionInputs
        self.supportsVideoInputs = supportsVideoInputs
        self.supportsAudioInputs = supportsAudioInputs
        self.supportsOllamaGenerationAPI = supportsOllamaGenerationAPI
        self.preferredSwiftEntryPoint = preferredSwiftEntryPoint
        self.warnings = warnings
    }
}

public struct ModelCapabilityPlanner {
    public init() {}

    public func plan(descriptor: ModelDescriptor) -> ModelCapabilityPlan {
        let type = descriptor.canonicalModelType
        if nonGenerativeVisionModels.contains(type) {
            return ModelCapabilityPlan(
                canonicalModelType: type,
                primaryTask: nonGenerativeTask(for: type),
                supportsTextGeneration: false,
                supportsVisionInputs: true,
                supportsVideoInputs: type == "sam3" || type == "sam3_1",
                supportsAudioInputs: false,
                supportsOllamaGenerationAPI: false,
                preferredSwiftEntryPoint: "\(type)-predictor",
                warnings: [
                    "\(type) is registered in mlx-vlm but uses a custom predictor pipeline rather than mlx_vlm.generate text generation.",
                ]
            )
        }

        return ModelCapabilityPlan(
            canonicalModelType: type,
            primaryTask: descriptor.hasVisionConfig ? "vision-language-generation" : "text-generation",
            supportsTextGeneration: true,
            supportsVisionInputs: descriptor.hasVisionConfig,
            supportsVideoInputs: supportsVideo(type),
            supportsAudioInputs: descriptor.hasAudioConfig || supportsAudio(type),
            supportsOllamaGenerationAPI: true,
            preferredSwiftEntryPoint: "generate",
            warnings: []
        )
    }

    private let nonGenerativeVisionModels: Set<String> = [
        "rfdetr",
        "sam3",
        "sam3_1",
        "sam3d_body",
    ]

    private func nonGenerativeTask(for type: String) -> String {
        switch type {
        case "rfdetr":
            return "object-detection-or-segmentation"
        case "sam3", "sam3_1":
            return "open-vocabulary-detection-segmentation-or-tracking"
        case "sam3d_body":
            return "3d-body-estimation"
        default:
            return "custom-vision-prediction"
        }
    }

    private func supportsVideo(_ type: String) -> Bool {
        type.contains("video") ||
            type == "qwen2_vl" ||
            type == "qwen2_5_vl" ||
            type == "qwen3_vl" ||
            type == "qwen3_vl_moe"
    }

    private func supportsAudio(_ type: String) -> Bool {
        type == "phi4mm" ||
            type == "qwen3_omni_moe" ||
            type == "nemotron_h_nano_omni" ||
            type == "minicpmo"
    }
}
