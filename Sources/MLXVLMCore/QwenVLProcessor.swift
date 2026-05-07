import Foundation

public enum QwenVLProcessorError: Error, CustomStringConvertible, Equatable {
    case missingGrid(kind: String, index: Int)
    case invalidGrid(String)

    public var description: String {
        switch self {
        case .missingGrid(let kind, let index):
            return "Missing \(kind) grid for placeholder index \(index)"
        case .invalidGrid(let value):
            return "Invalid grid '\(value)'; expected t,h,w"
        }
    }
}

public struct QwenVLGridTHW: Codable, Equatable, Sendable {
    public let temporal: Int
    public let height: Int
    public let width: Int

    public init(temporal: Int, height: Int, width: Int) {
        self.temporal = temporal
        self.height = height
        self.width = width
    }

    public init(csv: String) throws {
        let parts = csv.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard parts.count == 3,
              let temporal = parts[0],
              let height = parts[1],
              let width = parts[2]
        else {
            throw QwenVLProcessorError.invalidGrid(csv)
        }
        self.init(temporal: temporal, height: height, width: width)
    }

    public var product: Int {
        temporal * height * width
    }
}

public struct QwenVLPlaceholderExpander: Sendable {
    public let imageToken: String
    public let videoToken: String
    public let imageMergeSize: Int
    public let videoMergeSize: Int

    public init(
        imageToken: String = "<|image_pad|>",
        videoToken: String = "<|video_pad|>",
        imageMergeSize: Int = 2,
        videoMergeSize: Int = 2
    ) {
        self.imageToken = imageToken
        self.videoToken = videoToken
        self.imageMergeSize = imageMergeSize
        self.videoMergeSize = videoMergeSize
    }

    public func expand(
        texts: [String],
        imageGrids: [QwenVLGridTHW] = [],
        videoGrids: [QwenVLGridTHW] = []
    ) throws -> [String] {
        var imageIndex = 0
        let imageExpanded = try texts.map { text in
            try expand(
                text: text,
                token: imageToken,
                grids: imageGrids,
                mergeSize: imageMergeSize,
                index: &imageIndex,
                kind: "image"
            )
        }

        var videoIndex = 0
        return try imageExpanded.map { text in
            try expand(
                text: text,
                token: videoToken,
                grids: videoGrids,
                mergeSize: videoMergeSize,
                index: &videoIndex,
                kind: "video"
            )
        }
    }

    private func expand(
        text: String,
        token: String,
        grids: [QwenVLGridTHW],
        mergeSize: Int,
        index: inout Int,
        kind: String
    ) throws -> String {
        var result = ""
        var remaining = text[...]
        while let range = remaining.range(of: token) {
            guard grids.indices.contains(index) else {
                throw QwenVLProcessorError.missingGrid(kind: kind, index: index)
            }
            let mergeLength = mergeSize * mergeSize
            let repeatCount = grids[index].product / mergeLength
            result += String(remaining[..<range.lowerBound])
            result += String(repeating: token, count: repeatCount)
            remaining = remaining[range.upperBound...]
            index += 1
        }
        result += String(remaining)
        return result
    }
}
