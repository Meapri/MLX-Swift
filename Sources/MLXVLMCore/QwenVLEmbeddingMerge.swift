import Foundation

public enum QwenVLEmbeddingMergeError: Error, CustomStringConvertible, Equatable {
    case featureCountMismatch(batch: Int, tokenPositions: Int, availableFeatures: Int)

    public var description: String {
        switch self {
        case .featureCountMismatch(let batch, let tokenPositions, let availableFeatures):
            return "Batch \(batch) has \(tokenPositions) image/video token positions but \(availableFeatures) feature rows are available"
        }
    }
}

public struct QwenVLFeaturePlacement: Codable, Equatable, Sendable {
    public let batchIndex: Int
    public let tokenIndex: Int
    public let featureIndex: Int

    public init(batchIndex: Int, tokenIndex: Int, featureIndex: Int) {
        self.batchIndex = batchIndex
        self.tokenIndex = tokenIndex
        self.featureIndex = featureIndex
    }
}

public enum QwenVLEmbeddingMergePlanner {
    public static func plan(
        inputIDs: [[Int]],
        imageTokenID: Int,
        videoTokenID: Int,
        featureCount: Int
    ) throws -> [QwenVLFeaturePlacement] {
        let imagePositionsCount = inputIDs.flatMap { $0 }.filter { $0 == imageTokenID }.count
        let tokenID = imagePositionsCount == 0 ? videoTokenID : imageTokenID

        var featureStartIndex = 0
        var placements: [QwenVLFeaturePlacement] = []

        for (batchIndex, row) in inputIDs.enumerated() {
            let tokenPositions = row.indices.filter { row[$0] == tokenID }
            guard featureStartIndex + tokenPositions.count <= featureCount else {
                throw QwenVLEmbeddingMergeError.featureCountMismatch(
                    batch: batchIndex,
                    tokenPositions: tokenPositions.count,
                    availableFeatures: max(0, featureCount - featureStartIndex)
                )
            }

            for (offset, tokenIndex) in tokenPositions.enumerated() {
                placements.append(
                    QwenVLFeaturePlacement(
                        batchIndex: batchIndex,
                        tokenIndex: tokenIndex,
                        featureIndex: featureStartIndex + offset
                    )
                )
            }
            featureStartIndex += tokenPositions.count
        }

        return placements
    }
}
