import Foundation
import MLXVLMCore

#if MLXVLM_REAL_MLX_API && MLXVLM_HUGGINGFACE_DOWNLOADER && canImport(MLXLMCommon) && canImport(MLXVLM) && canImport(MLXHuggingFace) && canImport(HuggingFace)
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
#endif

public enum MLXRemoteModelResolverError: Error, CustomStringConvertible, Sendable {
    case downloaderUnavailable(String)

    public var description: String {
        switch self {
        case .downloaderUnavailable(let identifier):
            return """
            Remote Hugging Face model download is not available for \(identifier). Build with MLXVLM_ENABLE_MLX_BACKEND=1, MLXVLM_ENABLE_REAL_MLX_API=1, MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1, and MLXVLM_ENABLE_HUGGINGFACE_DOWNLOADER=1, then provide the Hugging Face downloader dependencies.
            """
        }
    }
}

public struct MLXRemoteModelResolver {
    public let store: ModelStore

    public init(store: ModelStore = ModelStore()) {
        self.store = store
    }

    public func resolveDescriptor(
        pathOrIdentifier: String,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelDescriptor {
        do {
            return try store.loadDescriptor(pathOrIdentifier: pathOrIdentifier)
        } catch ModelStoreError.unsupportedRemoteIdentifier(let identifier) {
            return try await downloadDescriptor(
                identifier: identifier,
                useLatest: useLatest,
                progressHandler: progressHandler
            )
        }
    }

    private func downloadDescriptor(
        identifier: String,
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelDescriptor {
        #if MLXVLM_REAL_MLX_API && MLXVLM_HUGGINGFACE_DOWNLOADER && canImport(MLXLMCommon) && canImport(MLXVLM) && canImport(MLXHuggingFace) && canImport(HuggingFace)
        let downloader = #hubDownloader(HubClient())
        let configuration = VLMModelFactory.shared.configuration(id: identifier)
        let resolved = try await MLXLMCommon.resolve(
            configuration: configuration,
            from: downloader,
            useLatest: useLatest,
            progressHandler: progressHandler
        )
        return try store.loadDescriptor(modelURL: resolved.modelDirectory, identifier: identifier)
        #else
        throw MLXRemoteModelResolverError.downloaderUnavailable(identifier)
        #endif
    }
}
