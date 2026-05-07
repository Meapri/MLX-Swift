import Foundation
import MLXVLMCore

#if MLXVLM_REAL_MLX_API
import MLX
import MLXEmbedders
import MLXLMCommon
import MLXLMTokenizers
#endif

public enum MLXVLMUpstreamEmbeddingBackendError: Error, CustomStringConvertible, Sendable {
    case unavailable
    case emptyInput

    public var description: String {
        switch self {
        case .unavailable:
            "MLXEmbedders is not available in this build."
        case .emptyInput:
            "Embedding input is empty."
        }
    }
}

#if MLXVLM_REAL_MLX_API
public struct MLXVLMUpstreamEmbeddingBackend: EmbeddingBackend {
    public let descriptor: ModelDescriptor
    public let status: BackendStatus
    private let container: EmbedderModelContainer

    public init(descriptor: ModelDescriptor, container: EmbedderModelContainer) {
        self.descriptor = descriptor
        self.status = .mlxSwiftVLM
        self.container = container
    }

    public static func load(descriptor: ModelDescriptor) async throws -> MLXVLMUpstreamEmbeddingBackend {
        let modelURL = URL(fileURLWithPath: descriptor.path, isDirectory: true)
        let container = try await EmbedderModelFactory.shared.loadContainer(
            from: modelURL,
            using: TokenizersLoader()
        )
        return MLXVLMUpstreamEmbeddingBackend(descriptor: descriptor, container: container)
    }

    public func embed(_ request: EmbeddingRequest) async throws -> CompletedEmbedding {
        let model = request.model.isEmpty ? descriptor.id : request.model
        let result = try await container.perform { context in
            let tokenRows = try tokenRows(for: request, tokenizer: context.tokenizer)
            guard !tokenRows.isEmpty else {
                throw MLXVLMUpstreamEmbeddingBackendError.emptyInput
            }

            let paddingToken = context.tokenizer.eosTokenId ?? 0
            let maxLength = tokenRows.map(\.count).max() ?? 0
            let padded = stacked(
                tokenRows.map { row in
                    MLXArray(row + Array(repeating: paddingToken, count: maxLength - row.count))
                }
            )
            let mask = padded .!= paddingToken
            let tokenTypes = MLXArray.zeros(like: padded)
            let output = context.model(
                padded,
                positionIds: nil,
                tokenTypeIds: tokenTypes,
                attentionMask: mask
            )
            let pooled = context.pooling(
                output,
                mask: mask,
                normalize: true,
                applyLayerNorm: true
            )
            pooled.eval()
            return (
                embeddings: pooled.map { $0.asArray(Float.self) },
                promptTokenCount: tokenRows.reduce(0) { $0 + $1.count }
            )
        }

        return CompletedEmbedding(
            model: model,
            embeddings: result.embeddings,
            promptTokenCount: result.promptTokenCount
        )
    }

    private func tokenRows(for request: EmbeddingRequest, tokenizer: any Tokenizer) throws -> [[Int]] {
        if let tokenIDInputs = request.tokenIDInputs {
            return tokenIDInputs
        }
        return request.texts.map {
            tokenizer.encode(text: $0, addSpecialTokens: true)
        }
    }
}
#endif
