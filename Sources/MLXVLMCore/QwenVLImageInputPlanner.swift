import Foundation
import ImageIO

public struct QwenVLResolvedImageInput: Codable, Equatable, Sendable {
    public let media: MediaReferenceSummary
    public let gridPlan: QwenVLImageGridPlan?
    public let error: String?

    public init(media: MediaReferenceSummary, gridPlan: QwenVLImageGridPlan?, error: String?) {
        self.media = media
        self.gridPlan = gridPlan
        self.error = error
    }
}

public struct QwenVLImageInputReport: Codable, Equatable, Sendable {
    public let images: [QwenVLResolvedImageInput]

    public init(images: [QwenVLResolvedImageInput]) {
        self.images = images
    }

    public var imageCount: Int {
        images.count
    }

    public var plannedCount: Int {
        images.filter { $0.gridPlan != nil }.count
    }

    public var errorCount: Int {
        images.filter { $0.error != nil }.count
    }

    public var imageGrids: [QwenVLGridTHW] {
        images.compactMap(\.gridPlan?.grid)
    }

    public var placeholderTokenCount: Int {
        images.reduce(0) { $0 + ($1.gridPlan?.placeholderTokenCount ?? 0) }
    }
}

public struct QwenVLImageInputPlanner {
    public let resolver: MediaReferenceResolver
    public let config: QwenVLImageGridConfig

    public init(
        resolver: MediaReferenceResolver = MediaReferenceResolver(),
        config: QwenVLImageGridConfig = QwenVLImageGridConfig()
    ) {
        self.resolver = resolver
        self.config = config
    }

    public func plan(request: GenerationRequest) -> QwenVLImageInputReport {
        QwenVLImageInputReport(
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

    public func plan(reference: ImageReference) -> QwenVLResolvedImageInput {
        let media = resolver.resolve(reference: reference.url, kind: .image)
            .withImageOptions(reference)
        guard media.isLoadable else {
            return QwenVLResolvedImageInput(media: media, gridPlan: nil, error: media.error)
        }

        do {
            let dimensions = try imageDimensions(reference: reference.url, media: media)
            let gridPlan = try imagePlan(dimensions: dimensions, reference: reference)
            return QwenVLResolvedImageInput(media: media, gridPlan: gridPlan, error: nil)
        } catch {
            return QwenVLResolvedImageInput(
                media: media,
                gridPlan: nil,
                error: String(describing: error)
            )
        }
    }

    public func plan(reference: String) -> QwenVLResolvedImageInput {
        plan(reference: ImageReference(url: reference))
    }

    private func imageGridConfig(for reference: ImageReference) -> QwenVLImageGridConfig {
        QwenVLImageGridConfig(
            patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize,
            mergeSize: config.mergeSize,
            minPixels: reference.minPixels ?? config.minPixels,
            maxPixels: reference.maxPixels ?? config.maxPixels
        )
    }

    private func imagePlan(
        dimensions: (height: Int, width: Int),
        reference: ImageReference
    ) throws -> QwenVLImageGridPlan {
        let imageConfig = imageGridConfig(for: reference)
        if let resizedHeight = reference.resizedHeight,
           let resizedWidth = reference.resizedWidth
        {
            let resized = try QwenVLImageGridPlanner.smartResizeImage(
                height: resizedHeight,
                width: resizedWidth,
                config: imageConfig
            )
            let grid = QwenVLGridTHW(
                temporal: 1,
                height: resized.height / imageConfig.patchSize,
                width: resized.width / imageConfig.patchSize
            )
            return QwenVLImageGridPlan(
                originalHeight: dimensions.height,
                originalWidth: dimensions.width,
                resizedHeight: resized.height,
                resizedWidth: resized.width,
                grid: grid,
                placeholderTokenCount: grid.product / (imageConfig.mergeSize * imageConfig.mergeSize)
            )
        }
        return try QwenVLImageGridPlanner.imagePlan(
            height: dimensions.height,
            width: dimensions.width,
            config: imageConfig
        )
    }

    private func imageDimensions(reference: String, media: MediaReferenceSummary) throws -> (height: Int, width: Int) {
        switch media.source {
        case .rawBase64:
            guard let data = Data(base64Encoded: reference, options: [.ignoreUnknownCharacters]) else {
                throw ImageInputError.unreadableImage("base64 payload could not be decoded")
            }
            return try imageDimensions(data: data)
        case .dataURI:
            guard let data = Self.data(fromDataURI: reference) else {
                throw ImageInputError.unreadableImage("data URI payload could not be decoded")
            }
            return try imageDimensions(data: data)
        case .filePath, .fileURL:
            guard let location = media.location else {
                throw ImageInputError.unreadableImage("file media is missing a path")
            }
            return try imageDimensions(url: URL(fileURLWithPath: location))
        case .remoteURL:
            throw ImageInputError.unreadableImage("remote images are not downloaded by the compatibility shell")
        case .unresolved:
            throw ImageInputError.unreadableImage(media.error ?? "unresolved image reference")
        }
    }

    private func imageDimensions(data: Data) throws -> (height: Int, width: Int) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw ImageInputError.unreadableImage("image data could not be decoded")
        }
        return try imageDimensions(source: source)
    }

    private func imageDimensions(url: URL) throws -> (height: Int, width: Int) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw ImageInputError.unreadableImage("image file could not be decoded")
        }
        return try imageDimensions(source: source)
    }

    private func imageDimensions(source: CGImageSource) throws -> (height: Int, width: Int) {
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int
        else {
            throw ImageInputError.unreadableImage("image dimensions were not present")
        }
        return (height, width)
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

public enum ImageInputError: Error, CustomStringConvertible, Equatable {
    case unreadableImage(String)

    public var description: String {
        switch self {
        case .unreadableImage(let reason):
            return "Unreadable image input: \(reason)"
        }
    }
}
