import Foundation

public struct GenerationChunkCollectionReport: Codable, Equatable, Sendable {
    public let model: String
    public let text: String
    public let finishReason: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int
    public let chunkCount: Int
    public let tokenIDChunkCount: Int
    public let finishedChunkCount: Int
    public let tokenAccountingSource: String
    public let toolCalls: [GenerationToolCall]

    public init(
        model: String,
        text: String,
        finishReason: String,
        promptTokens: Int,
        completionTokens: Int,
        chunkCount: Int,
        tokenIDChunkCount: Int,
        finishedChunkCount: Int,
        tokenAccountingSource: String,
        toolCalls: [GenerationToolCall] = []
    ) {
        self.model = model
        self.text = text
        self.finishReason = finishReason
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = promptTokens + completionTokens
        self.chunkCount = chunkCount
        self.tokenIDChunkCount = tokenIDChunkCount
        self.finishedChunkCount = finishedChunkCount
        self.tokenAccountingSource = tokenAccountingSource
        self.toolCalls = toolCalls
    }

    public var completedGeneration: CompletedGeneration {
        CompletedGeneration(
            model: model,
            text: text,
            finishReason: finishReason,
            usage: GenerationUsage(promptTokens: promptTokens, completionTokens: completionTokens),
            toolCalls: toolCalls
        )
    }
}

public enum GenerationChunkCollector {
    public static func collect(
        model: String,
        promptTokenCount: Int,
        chunks: [GenerationChunk]
    ) -> GenerationChunkCollectionReport {
        let text = chunks.map(\.text).joined()
        let toolCalls = chunks.flatMap(\.toolCalls)
        let promptTokens = chunks.reversed().compactMap(\.promptTokenCount).first ?? promptTokenCount
        let completionTokenOverride = chunks.reversed().compactMap(\.completionTokenCount).first
        let tokenIDChunkCount = chunks.filter { $0.tokenID != nil && !$0.isFinished }.count
        let nonEmptyDeltaCount = chunks.filter { !$0.isFinished && !$0.text.isEmpty }.count
        let completionTokens: Int
        let tokenAccountingSource: String
        if let completionTokenOverride {
            completionTokens = completionTokenOverride
            tokenAccountingSource = "backend-completion-info"
        } else if tokenIDChunkCount > 0 {
            completionTokens = tokenIDChunkCount
            tokenAccountingSource = "token-id-chunks"
        } else if nonEmptyDeltaCount > 0 {
            completionTokens = nonEmptyDeltaCount
            tokenAccountingSource = "non-empty-delta-chunks"
        } else if !text.isEmpty {
            completionTokens = 1
            tokenAccountingSource = "completed-text-fallback"
        } else {
            completionTokens = 0
            tokenAccountingSource = "empty-output"
        }

        return GenerationChunkCollectionReport(
            model: model,
            text: text,
            finishReason: chunks.last(where: { $0.isFinished })?.finishReason ?? "stop",
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            chunkCount: chunks.count,
            tokenIDChunkCount: tokenIDChunkCount,
            finishedChunkCount: chunks.filter(\.isFinished).count,
            tokenAccountingSource: tokenAccountingSource,
            toolCalls: toolCalls
        )
    }
}

public struct GenerationEndpointRenderReport: Codable, Equatable, Sendable {
    public let collection: GenerationChunkCollectionReport
    public let response: GenerationAPIResponseRenderReport

    public init(
        collection: GenerationChunkCollectionReport,
        response: GenerationAPIResponseRenderReport
    ) {
        self.collection = collection
        self.response = response
    }
}

public enum GenerationEndpointRenderer {
    public static func render(
        model: String,
        promptTokenCount: Int,
        api: GenerationResponseAPI,
        stream: Bool,
        chunks: [GenerationChunk],
        request: GenerationRequest? = nil
    ) -> GenerationEndpointRenderReport {
        let collection = GenerationChunkCollector.collect(
            model: model,
            promptTokenCount: promptTokenCount,
            chunks: chunks
        )
        let response = GenerationAPIResponseRenderer.renderCompleted(
            collection.completedGeneration,
            api: api,
            stream: stream,
            chunks: chunks,
            request: request
        )
        return GenerationEndpointRenderReport(collection: collection, response: response)
    }
}
