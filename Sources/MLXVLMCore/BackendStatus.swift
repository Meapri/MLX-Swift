import Foundation

public enum BackendCapability: String, Codable, Equatable, Sendable {
    case compatibilityOnly
    case mlxSwiftUnavailable
    case generationUnavailable
    case mlxSwiftGeneration
}

public struct BackendStatus: Codable, Equatable, Sendable {
    public let ready: Bool
    public let activeBackend: String
    public let capabilities: [BackendCapability]
    public let message: String
    public let nextSteps: [String]

    public init(
        ready: Bool,
        activeBackend: String,
        capabilities: [BackendCapability],
        message: String,
        nextSteps: [String]
    ) {
        self.ready = ready
        self.activeBackend = activeBackend
        self.capabilities = capabilities
        self.message = message
        self.nextSteps = nextSteps
    }

    public static let compatibilityShell = BackendStatus(
        ready: false,
        activeBackend: "compatibility-shell",
        capabilities: [
            .compatibilityOnly,
            .mlxSwiftUnavailable,
            .generationUnavailable,
        ],
        message: "MLX Swift inference backend is not linked in this target. Metadata, prompt, tokenizer, Ollama/OpenAI API compatibility, and Qwen VL planning are available; token generation is intentionally unavailable.",
        nextSteps: [
            "Vendor or fetch mlx-swift 0.31.3, mlx-swift-lm 3.31.3, and swift-tokenizers-mlx.",
            "Add a backend target that depends on MLXVLMCore plus MLX, MLXLMCommon, MLXLLM, MLXVLM, and MLXLMTokenizers modules.",
            "Replace UnavailableVLMGenerator with an MLX-backed VLMGenerator and route server generation endpoints through CompletedGeneration/APIResponses.",
            "Run text-only and image Qwen2-VL smoke tests before replacing 501 responses.",
        ]
    )

    public static let mlxSwiftVLM = BackendStatus(
        ready: true,
        activeBackend: "mlx-swift-vlm",
        capabilities: [
            .mlxSwiftGeneration,
        ],
        message: "MLX Swift VLM backend is linked and routes generation through mlx-swift-lm ModelContainer.",
        nextSteps: []
    )
}
