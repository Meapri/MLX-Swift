import CoreGraphics
import Foundation
import ImageIO

public struct QwenVLPatchNormalization: Codable, Equatable, Sendable {
    public let imageMean: [Double]
    public let imageStd: [Double]
    public let rescaleFactor: Double
    public let doRescale: Bool
    public let doNormalize: Bool

    public init(
        imageMean: [Double] = [0.5, 0.5, 0.5],
        imageStd: [Double] = [0.5, 0.5, 0.5],
        rescaleFactor: Double = 1.0 / 255.0,
        doRescale: Bool = true,
        doNormalize: Bool = true
    ) {
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.rescaleFactor = rescaleFactor
        self.doRescale = doRescale
        self.doNormalize = doNormalize
    }

    public func normalizedValue(_ byte: UInt8, channel: Int) -> Float {
        var value = Double(byte)
        if doRescale {
            value *= rescaleFactor
        }
        if doNormalize {
            let mean = imageMean.indices.contains(channel) ? imageMean[channel] : 0.5
            let std = imageStd.indices.contains(channel) ? imageStd[channel] : 0.5
            value = (value - mean) / std
        }
        return Float(value)
    }
}

public struct QwenVLPatchFloat32Stats: Codable, Equatable, Sendable {
    public let count: Int
    public let min: Float
    public let max: Float
    public let checksum: Double

    public init(count: Int, min: Float, max: Float, checksum: Double) {
        self.count = count
        self.min = min
        self.max = max
        self.checksum = checksum
    }
}

public struct QwenVLImagePixelPreflightImage: Codable, Equatable, Sendable {
    public let media: MediaReferenceSummary
    public let gridPlan: QwenVLImageGridPlan?
    public let pixelShape: [Int]?
    public let rgbByteCount: Int?
    public let layout: String?
    public let patchShape: [Int]?
    public let patchFloat32ByteCount: Int?
    public let patchLayout: String?
    public let patchStats: QwenVLPatchFloat32Stats?
    public let error: String?

    public init(
        media: MediaReferenceSummary,
        gridPlan: QwenVLImageGridPlan?,
        pixelShape: [Int]?,
        rgbByteCount: Int?,
        layout: String?,
        patchShape: [Int]?,
        patchFloat32ByteCount: Int?,
        patchLayout: String?,
        patchStats: QwenVLPatchFloat32Stats?,
        error: String?
    ) {
        self.media = media
        self.gridPlan = gridPlan
        self.pixelShape = pixelShape
        self.rgbByteCount = rgbByteCount
        self.layout = layout
        self.patchShape = patchShape
        self.patchFloat32ByteCount = patchFloat32ByteCount
        self.patchLayout = patchLayout
        self.patchStats = patchStats
        self.error = error
    }
}

public struct QwenVLImagePixelPreflightReport: Codable, Equatable, Sendable {
    public let images: [QwenVLImagePixelPreflightImage]

    public init(images: [QwenVLImagePixelPreflightImage]) {
        self.images = images
    }

    enum CodingKeys: String, CodingKey {
        case images
        case imageCount
        case preparedCount
        case errorCount
        case totalRGBByteCount
        case totalPatchFloat32ByteCount
        case imageGrids
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.images = try container.decode([QwenVLImagePixelPreflightImage].self, forKey: .images)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(images, forKey: .images)
        try container.encode(imageCount, forKey: .imageCount)
        try container.encode(preparedCount, forKey: .preparedCount)
        try container.encode(errorCount, forKey: .errorCount)
        try container.encode(totalRGBByteCount, forKey: .totalRGBByteCount)
        try container.encode(totalPatchFloat32ByteCount, forKey: .totalPatchFloat32ByteCount)
        try container.encode(imageGrids, forKey: .imageGrids)
    }

    public var imageCount: Int {
        images.count
    }

    public var preparedCount: Int {
        images.filter { $0.rgbByteCount != nil }.count
    }

    public var errorCount: Int {
        images.filter { $0.error != nil }.count
    }

    public var totalRGBByteCount: Int {
        images.reduce(0) { $0 + ($1.rgbByteCount ?? 0) }
    }

    public var totalPatchFloat32ByteCount: Int {
        images.reduce(0) { $0 + ($1.patchFloat32ByteCount ?? 0) }
    }

    public var imageGrids: [QwenVLGridTHW] {
        images.compactMap(\.gridPlan?.grid)
    }
}

public struct QwenVLImagePixelPreflightPlanner {
    public let resolver: MediaReferenceResolver
    public let config: QwenVLImageGridConfig
    public let normalization: QwenVLPatchNormalization

    public init(
        resolver: MediaReferenceResolver = MediaReferenceResolver(),
        config: QwenVLImageGridConfig = QwenVLImageGridConfig(),
        normalization: QwenVLPatchNormalization = QwenVLPatchNormalization()
    ) {
        self.resolver = resolver
        self.config = config
        self.normalization = normalization
    }

    public func plan(request: GenerationRequest) -> QwenVLImagePixelPreflightReport {
        QwenVLImagePixelPreflightReport(
            images: request.messages.flatMap { message in
                message.content.compactMap { part in
                    guard case .imageURL(let reference) = part else {
                        return nil
                    }
                    return plan(reference: reference)
                }
            }
        )
    }

    public func plan(reference: String) -> QwenVLImagePixelPreflightImage {
        let media = resolver.resolve(reference: reference, kind: .image)
        guard media.isLoadable else {
            return QwenVLImagePixelPreflightImage(
                media: media,
                gridPlan: nil,
                pixelShape: nil,
                rgbByteCount: nil,
                layout: nil,
                patchShape: nil,
                patchFloat32ByteCount: nil,
                patchLayout: nil,
                patchStats: nil,
                error: media.error
            )
        }

        do {
            let image = try loadImage(reference: reference, media: media)
            let gridPlan = try QwenVLImageGridPlanner.imagePlan(
                height: image.height,
                width: image.width,
                config: config
            )
            let rgb = try prepareRGBTensor(image: image, gridPlan: gridPlan)
            let patchShape = patchTensorShape(gridPlan: gridPlan)
            let patchFloat32ByteCount = patchShape.reduce(1, *) * MemoryLayout<Float>.size
            let patchStats = patchFloat32Stats(rgb: rgb, gridPlan: gridPlan)
            return QwenVLImagePixelPreflightImage(
                media: media,
                gridPlan: gridPlan,
                pixelShape: [gridPlan.resizedHeight, gridPlan.resizedWidth, 3],
                rgbByteCount: rgb.count,
                layout: "rgb_interleaved_uint8",
                patchShape: patchShape,
                patchFloat32ByteCount: patchFloat32ByteCount,
                patchLayout: "qwen_vl_patch_float32",
                patchStats: patchStats,
                error: nil
            )
        } catch {
            return QwenVLImagePixelPreflightImage(
                media: media,
                gridPlan: nil,
                pixelShape: nil,
                rgbByteCount: nil,
                layout: nil,
                patchShape: nil,
                patchFloat32ByteCount: nil,
                patchLayout: nil,
                patchStats: nil,
                error: String(describing: error)
            )
        }
    }

    private func loadImage(reference: String, media: MediaReferenceSummary) throws -> CGImage {
        switch media.source {
        case .rawBase64:
            guard let data = Data(base64Encoded: reference, options: [.ignoreUnknownCharacters]) else {
                throw ImageInputError.unreadableImage("base64 payload could not be decoded")
            }
            return try loadImage(data: data)
        case .dataURI:
            guard let data = Self.data(fromDataURI: reference) else {
                throw ImageInputError.unreadableImage("data URI payload could not be decoded")
            }
            return try loadImage(data: data)
        case .filePath, .fileURL:
            guard let location = media.location else {
                throw ImageInputError.unreadableImage("file media is missing a path")
            }
            return try loadImage(url: URL(fileURLWithPath: location))
        case .remoteURL:
            throw ImageInputError.unreadableImage("remote images are not downloaded by the compatibility shell")
        case .unresolved:
            throw ImageInputError.unreadableImage(media.error ?? "unresolved image reference")
        }
    }

    private func loadImage(data: Data) throws -> CGImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw ImageInputError.unreadableImage("image data could not be decoded")
        }
        return image
    }

    private func loadImage(url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw ImageInputError.unreadableImage("image file could not be decoded")
        }
        return image
    }

    private func prepareRGBTensor(image: CGImage, gridPlan: QwenVLImageGridPlan) throws -> [UInt8] {
        let width = gridPlan.resizedWidth
        let height = gridPlan.resizedHeight
        let rgbaBytesPerPixel = 4
        let rgbaBytesPerRow = width * rgbaBytesPerPixel
        var rgba = [UInt8](repeating: 0, count: height * rgbaBytesPerRow)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

        try rgba.withUnsafeMutableBytes { buffer in
            guard let baseAddress = buffer.baseAddress,
                  let context = CGContext(
                    data: baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: rgbaBytesPerRow,
                    space: colorSpace,
                    bitmapInfo: bitmapInfo.rawValue
                  )
            else {
                throw ImageInputError.unreadableImage("resized image buffer could not be created")
            }

            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        var sourceIndex = 0
        var targetIndex = 0
        while targetIndex < rgb.count {
            rgb[targetIndex] = rgba[sourceIndex]
            rgb[targetIndex + 1] = rgba[sourceIndex + 1]
            rgb[targetIndex + 2] = rgba[sourceIndex + 2]
            sourceIndex += rgbaBytesPerPixel
            targetIndex += 3
        }
        return rgb
    }

    private func patchTensorShape(gridPlan: QwenVLImageGridPlan) -> [Int] {
        [
            gridPlan.grid.product,
            3 * config.temporalPatchSize * config.patchSize * config.patchSize,
        ]
    }

    private func patchFloat32Stats(rgb: [UInt8], gridPlan: QwenVLImageGridPlan) -> QwenVLPatchFloat32Stats {
        let patchSize = config.patchSize
        let temporalPatchSize = config.temporalPatchSize
        let mergeSize = config.mergeSize
        let blockRows = gridPlan.grid.height / mergeSize
        let blockCols = gridPlan.grid.width / mergeSize
        var count = 0
        var minimum = Float.greatestFiniteMagnitude
        var maximum = -Float.greatestFiniteMagnitude
        var checksum = 0.0

        for blockRow in 0..<blockRows {
            for blockCol in 0..<blockCols {
                for mergeRow in 0..<mergeSize {
                    for mergeCol in 0..<mergeSize {
                        for channel in 0..<3 {
                            for _ in 0..<temporalPatchSize {
                                for patchRow in 0..<patchSize {
                                    for patchCol in 0..<patchSize {
                                        let y = (blockRow * mergeSize + mergeRow) * patchSize + patchRow
                                        let x = (blockCol * mergeSize + mergeCol) * patchSize + patchCol
                                        let rgbIndex = (y * gridPlan.resizedWidth + x) * 3 + channel
                                        let value = normalization.normalizedValue(rgb[rgbIndex], channel: channel)
                                        minimum = min(minimum, value)
                                        maximum = max(maximum, value)
                                        checksum += Double(value)
                                        count += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return QwenVLPatchFloat32Stats(
            count: count,
            min: minimum,
            max: maximum,
            checksum: checksum
        )
    }

    private static func data(fromDataURI reference: String) -> Data? {
        guard let comma = reference.firstIndex(of: ",") else {
            return nil
        }
        let metadata = reference[reference.index(reference.startIndex, offsetBy: 5)..<comma]
        guard metadata.split(separator: ";").contains("base64") else {
            return nil
        }
        return Data(
            base64Encoded: String(reference[reference.index(after: comma)...]),
            options: [.ignoreUnknownCharacters]
        )
    }
}
