import Foundation

public enum QwenVLImageGridError: Error, CustomStringConvertible, Equatable {
    case invalidAspectRatio(Double)
    case dimensionsBelowFactor(height: Int, width: Int, factor: Int)

    public var description: String {
        switch self {
        case .invalidAspectRatio(let ratio):
            return "absolute aspect ratio must be smaller than 200, got \(ratio)"
        case .dimensionsBelowFactor(let height, let width, let factor):
            return "height:\(height) or width:\(width) must be larger than factor:\(factor)"
        }
    }
}

public struct QwenVLImageGridConfig: Codable, Equatable, Sendable {
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let mergeSize: Int
    public let minPixels: Int
    public let maxPixels: Int

    public init(
        patchSize: Int = 14,
        temporalPatchSize: Int = 2,
        mergeSize: Int = 2,
        minPixels: Int = 56 * 56,
        maxPixels: Int = 14 * 14 * 4 * 1280
    ) {
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.mergeSize = mergeSize
        self.minPixels = minPixels
        self.maxPixels = maxPixels
    }

    public var factor: Int {
        patchSize * mergeSize
    }
}

public struct QwenVLImageGridPlan: Codable, Equatable, Sendable {
    public let originalHeight: Int
    public let originalWidth: Int
    public let resizedHeight: Int
    public let resizedWidth: Int
    public let grid: QwenVLGridTHW
    public let placeholderTokenCount: Int

    public init(
        originalHeight: Int,
        originalWidth: Int,
        resizedHeight: Int,
        resizedWidth: Int,
        grid: QwenVLGridTHW,
        placeholderTokenCount: Int
    ) {
        self.originalHeight = originalHeight
        self.originalWidth = originalWidth
        self.resizedHeight = resizedHeight
        self.resizedWidth = resizedWidth
        self.grid = grid
        self.placeholderTokenCount = placeholderTokenCount
    }
}

public enum QwenVLImageGridPlanner {
    public static func smartResizeImage(
        height: Int,
        width: Int,
        config: QwenVLImageGridConfig = QwenVLImageGridConfig()
    ) throws -> (height: Int, width: Int) {
        let ratio = Double(max(height, width)) / Double(min(height, width))
        if ratio > 200 {
            throw QwenVLImageGridError.invalidAspectRatio(ratio)
        }

        var hBar = roundByFactor(height, factor: config.factor)
        var wBar = roundByFactor(width, factor: config.factor)

        if hBar * wBar > config.maxPixels {
            let beta = sqrt(Double(height * width) / Double(config.maxPixels))
            hBar = max(config.factor, floorByFactor(Int(Double(height) / beta), factor: config.factor))
            wBar = max(config.factor, floorByFactor(Int(Double(width) / beta), factor: config.factor))
        } else if hBar * wBar < config.minPixels {
            let beta = sqrt(Double(config.minPixels) / Double(height * width))
            hBar = ceilByFactor(Int(ceil(Double(height) * beta)), factor: config.factor)
            wBar = ceilByFactor(Int(ceil(Double(width) * beta)), factor: config.factor)
        }

        return (hBar, wBar)
    }

    public static func imagePlan(
        height: Int,
        width: Int,
        config: QwenVLImageGridConfig = QwenVLImageGridConfig()
    ) throws -> QwenVLImageGridPlan {
        let resized = try smartResizeImage(height: height, width: width, config: config)
        let grid = QwenVLGridTHW(
            temporal: 1,
            height: resized.height / config.patchSize,
            width: resized.width / config.patchSize
        )
        return QwenVLImageGridPlan(
            originalHeight: height,
            originalWidth: width,
            resizedHeight: resized.height,
            resizedWidth: resized.width,
            grid: grid,
            placeholderTokenCount: grid.product / (config.mergeSize * config.mergeSize)
        )
    }

    public static func smartResizeVideo(
        frames: Int,
        height: Int,
        width: Int,
        config: QwenVLImageGridConfig = QwenVLImageGridConfig(
            patchSize: 14,
            temporalPatchSize: 2,
            mergeSize: 2,
            minPixels: 128 * 32 * 32,
            maxPixels: 32 * 32 * 768
        )
    ) throws -> (height: Int, width: Int) {
        if height < config.factor || width < config.factor {
            throw QwenVLImageGridError.dimensionsBelowFactor(height: height, width: width, factor: config.factor)
        }
        let ratio = Double(max(height, width)) / Double(min(height, width))
        if ratio > 200 {
            throw QwenVLImageGridError.invalidAspectRatio(ratio)
        }

        var hBar = roundByFactor(height, factor: config.factor)
        var wBar = roundByFactor(width, factor: config.factor)
        let tBar = ceilByFactor(frames, factor: config.temporalPatchSize)

        if tBar * hBar * wBar > config.maxPixels {
            let beta = sqrt(Double(frames * height * width) / Double(config.maxPixels))
            hBar = max(config.factor, floorByFactor(Int(Double(height) / beta), factor: config.factor))
            wBar = max(config.factor, floorByFactor(Int(Double(width) / beta), factor: config.factor))
        } else if tBar * hBar * wBar < config.minPixels {
            let beta = sqrt(Double(config.minPixels) / Double(frames * height * width))
            hBar = ceilByFactor(Int(ceil(Double(height) * beta)), factor: config.factor)
            wBar = ceilByFactor(Int(ceil(Double(width) * beta)), factor: config.factor)
        }

        return (hBar, wBar)
    }

    public static func roundByFactor(_ value: Int, factor: Int) -> Int {
        Int(round(Double(value) / Double(factor))) * factor
    }

    public static func ceilByFactor(_ value: Int, factor: Int) -> Int {
        Int(ceil(Double(value) / Double(factor))) * factor
    }

    public static func floorByFactor(_ value: Int, factor: Int) -> Int {
        Int(floor(Double(value) / Double(factor))) * factor
    }
}
