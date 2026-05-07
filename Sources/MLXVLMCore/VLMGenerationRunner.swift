import Foundation

public struct VLMGenerationRunner: Sendable {
    public let backend: any VLMBackend

    public init(backend: any VLMBackend) {
        self.backend = backend
    }

    public func completedGeneration(for request: GenerationRequest) async throws -> CompletedGeneration {
        let processed = try backend.process(request)
        var assembler = GenerationOutputAssembler(
            model: request.model,
            promptTokenCount: processed.tokenIDs?.count ?? 0,
            stopSequences: request.parameters.stopSequences
        )

        let stream = try await backend.generate(request)
        for try await chunk in stream {
            _ = assembler.append(chunk)
            if assembler.snapshot.isFinished {
                break
            }
        }
        _ = assembler.finish()
        return assembler.completedGeneration
    }

    public func renderedResponse(
        for request: GenerationRequest,
        api: GenerationResponseAPI
    ) async throws -> GenerationEndpointRenderReport {
        let processed = try backend.process(request)
        let stream = try await backend.generate(request)
        var chunks: [GenerationChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
            if chunk.isFinished {
                break
            }
        }
        if !chunks.contains(where: \.isFinished) {
            chunks.append(GenerationChunk(text: "", isFinished: true, finishReason: "stop"))
        }
        return GenerationEndpointRenderer.render(
            model: request.model,
            promptTokenCount: processed.tokenIDs?.count ?? 0,
            api: api,
            stream: request.stream,
            chunks: chunks,
            request: request
        )
    }
}
