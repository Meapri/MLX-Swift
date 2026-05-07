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
            "Build with MLXVLM_ENABLE_MLX_BACKEND=1, MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1, and MLXVLM_ENABLE_REAL_MLX_API=1 to route generation through the upstream mlx-swift-lm VLM engine.",
            "Run scripts/verify_real_gemma4_smoke.sh, or an equivalent local model smoke test, before using the real backend path.",
            "Keep native Swift model-module work scoped to Python mlx-vlm behaviors that upstream mlx-swift-lm does not cover.",
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
