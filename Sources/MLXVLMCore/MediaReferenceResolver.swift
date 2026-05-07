import Foundation

public enum MediaKind: String, Codable, Equatable, Sendable {
    case image
    case audio
    case video
}

public enum MediaSourceKind: String, Codable, Equatable, Sendable {
    case rawBase64
    case dataURI
    case fileURL
    case filePath
    case remoteURL
    case unresolved
}

public struct MediaReferenceSummary: Codable, Equatable, Sendable {
    public let kind: MediaKind
    public let source: MediaSourceKind
    public let mimeType: String?
    public let byteCount: Int?
    public let location: String?
    public let isLoadable: Bool
    public let error: String?
    public let imageDetail: String?
    public let imageResizedHeight: Int?
    public let imageResizedWidth: Int?
    public let imageMinPixels: Int?
    public let imageMaxPixels: Int?
    public let audioFormat: String?
    public let videoMinPixels: Int?
    public let videoMaxPixels: Int?
    public let videoFPS: Double?
    public let videoNFrames: Int?
    public let videoMinFrames: Int?
    public let videoMaxFrames: Int?

    public init(
        kind: MediaKind,
        source: MediaSourceKind,
        mimeType: String?,
        byteCount: Int?,
        location: String?,
        isLoadable: Bool,
        error: String?,
        imageDetail: String? = nil,
        imageResizedHeight: Int? = nil,
        imageResizedWidth: Int? = nil,
        imageMinPixels: Int? = nil,
        imageMaxPixels: Int? = nil,
        audioFormat: String? = nil,
        videoMinPixels: Int? = nil,
        videoMaxPixels: Int? = nil,
        videoFPS: Double? = nil,
        videoNFrames: Int? = nil,
        videoMinFrames: Int? = nil,
        videoMaxFrames: Int? = nil
    ) {
        self.kind = kind
        self.source = source
        self.mimeType = mimeType
        self.byteCount = byteCount
        self.location = location
        self.isLoadable = isLoadable
        self.error = error
        self.imageDetail = imageDetail
        self.imageResizedHeight = imageResizedHeight
        self.imageResizedWidth = imageResizedWidth
        self.imageMinPixels = imageMinPixels
        self.imageMaxPixels = imageMaxPixels
        self.audioFormat = audioFormat
        self.videoMinPixels = videoMinPixels
        self.videoMaxPixels = videoMaxPixels
        self.videoFPS = videoFPS
        self.videoNFrames = videoNFrames
        self.videoMinFrames = videoMinFrames
        self.videoMaxFrames = videoMaxFrames
    }
}

public struct MediaResolutionReport: Codable, Equatable, Sendable {
    public let references: [MediaReferenceSummary]

    public init(references: [MediaReferenceSummary]) {
        self.references = references
    }

    public var imageCount: Int {
        references.filter { $0.kind == .image }.count
    }

    public var audioCount: Int {
        references.filter { $0.kind == .audio }.count
    }

    public var videoCount: Int {
        references.filter { $0.kind == .video }.count
    }

    public var loadableCount: Int {
        references.filter(\.isLoadable).count
    }

    public var errorCount: Int {
        references.filter { $0.error != nil }.count
    }
}

public struct MediaReferenceResolver: @unchecked Sendable {
    public let fileManager: FileManager
    public let baseDirectory: URL

    public init(
        fileManager: FileManager = .default,
        baseDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    ) {
        self.fileManager = fileManager
        self.baseDirectory = baseDirectory
    }

    public func report(for request: GenerationRequest) -> MediaResolutionReport {
        MediaResolutionReport(
            references: request.messages.flatMap { message in
                message.content.compactMap(resolve)
            }
        )
    }

    public func resolve(_ part: ContentPart) -> MediaReferenceSummary? {
        switch part {
        case .text, .imagePlaceholder, .audioPlaceholder:
            return nil
        case .imageURL(let reference):
            return resolve(reference: reference.url, kind: .image)
                .withImageOptions(reference)
        case .audioURL(let reference):
            return resolve(reference: reference.data, kind: .audio)
                .withAudioFormat(reference.format)
        case .videoURL(let reference):
            return resolve(reference: reference.url, kind: .video)
                .withVideoOptions(
                    minPixels: reference.minPixels,
                    maxPixels: reference.maxPixels,
                    fps: reference.fps,
                    nframes: reference.nframes,
                    minFrames: reference.minFrames,
                    maxFrames: reference.maxFrames
                )
        }
    }

    public func resolve(reference: String, kind: MediaKind) -> MediaReferenceSummary {
        if reference.hasPrefix("data:") {
            return resolveDataURI(reference, kind: kind)
        }
        if let url = URL(string: reference), let scheme = url.scheme?.lowercased() {
            switch scheme {
            case "file":
                return resolveFile(url: url, kind: kind, source: .fileURL, location: url.path)
            case "http", "https":
                return MediaReferenceSummary(
                    kind: kind,
                    source: .remoteURL,
                    mimeType: nil,
                    byteCount: nil,
                    location: reference,
                    isLoadable: false,
                    error: "Remote media downloads are not performed by the compatibility shell."
                )
            default:
                return unresolved(kind: kind, location: reference, error: "Unsupported media URL scheme: \(scheme)")
            }
        }

        let pathURL = reference.hasPrefix("/")
            ? URL(fileURLWithPath: reference).standardizedFileURL
            : baseDirectory.appendingPathComponent(reference).standardizedFileURL
        if fileManager.fileExists(atPath: pathURL.path) {
            return resolveFile(url: pathURL, kind: kind, source: .filePath, location: pathURL.path)
        }

        let normalizedBase64 = reference
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\n", with: "")
            .replacingOccurrences(of: "\r", with: "")
        if let data = Data(base64Encoded: normalizedBase64, options: [.ignoreUnknownCharacters]) {
            return MediaReferenceSummary(
                kind: kind,
                source: .rawBase64,
                mimeType: nil,
                byteCount: data.count,
                location: nil,
                isLoadable: true,
                error: nil
            )
        }

        return unresolved(kind: kind, location: reference, error: "Media reference is not a readable file, data URI, remote URL, or base64 payload.")
    }

    private func resolveDataURI(_ reference: String, kind: MediaKind) -> MediaReferenceSummary {
        guard let comma = reference.firstIndex(of: ",") else {
            return unresolved(kind: kind, location: nil, error: "Data URI is missing a comma separator.")
        }
        let metadata = String(reference[reference.index(reference.startIndex, offsetBy: 5)..<comma])
        let payload = String(reference[reference.index(after: comma)...])
        let metadataParts = metadata.split(separator: ";").map(String.init)
        let mimeType = metadataParts.first?.isEmpty == false ? metadataParts.first : nil
        guard metadataParts.contains("base64") else {
            return unresolved(kind: kind, location: nil, error: "Only base64 data URIs are supported.")
        }
        guard let data = Data(base64Encoded: payload, options: [.ignoreUnknownCharacters]) else {
            return unresolved(kind: kind, location: nil, error: "Data URI base64 payload could not be decoded.")
        }
        return MediaReferenceSummary(
            kind: kind,
            source: .dataURI,
            mimeType: mimeType,
            byteCount: data.count,
            location: nil,
            isLoadable: true,
            error: nil
        )
    }

    private func resolveFile(
        url: URL,
        kind: MediaKind,
        source: MediaSourceKind,
        location: String
    ) -> MediaReferenceSummary {
        do {
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            guard values.isRegularFile == true else {
                return unresolved(kind: kind, location: location, error: "Media path is not a regular file.")
            }
            return MediaReferenceSummary(
                kind: kind,
                source: source,
                mimeType: Self.mimeType(forPathExtension: url.pathExtension),
                byteCount: values.fileSize,
                location: location,
                isLoadable: true,
                error: nil
            )
        } catch {
            return unresolved(kind: kind, location: location, error: String(describing: error))
        }
    }

    private func unresolved(kind: MediaKind, location: String?, error: String) -> MediaReferenceSummary {
        MediaReferenceSummary(
            kind: kind,
            source: .unresolved,
            mimeType: nil,
            byteCount: nil,
            location: location,
            isLoadable: false,
            error: error
        )
    }

    private static func mimeType(forPathExtension pathExtension: String) -> String? {
        switch pathExtension.lowercased() {
        case "jpg", "jpeg":
            return "image/jpeg"
        case "png":
            return "image/png"
        case "gif":
            return "image/gif"
        case "webp":
            return "image/webp"
        case "wav":
            return "audio/wav"
        case "mp3":
            return "audio/mpeg"
        case "m4a":
            return "audio/mp4"
        case "mp4":
            return "video/mp4"
        case "mov":
            return "video/quicktime"
        case "avi":
            return "video/x-msvideo"
        case "webm":
            return "video/webm"
        case "mkv":
            return "video/x-matroska"
        default:
            return nil
        }
    }
}

extension MediaReferenceSummary {
    func withImageOptions(_ reference: ImageReference) -> MediaReferenceSummary {
        MediaReferenceSummary(
            kind: kind,
            source: source,
            mimeType: mimeType,
            byteCount: byteCount,
            location: location,
            isLoadable: isLoadable,
            error: error,
            imageDetail: reference.detail,
            imageResizedHeight: reference.resizedHeight,
            imageResizedWidth: reference.resizedWidth,
            imageMinPixels: reference.minPixels,
            imageMaxPixels: reference.maxPixels,
            audioFormat: audioFormat,
            videoMinPixels: videoMinPixels,
            videoMaxPixels: videoMaxPixels,
            videoFPS: videoFPS,
            videoNFrames: videoNFrames,
            videoMinFrames: videoMinFrames,
            videoMaxFrames: videoMaxFrames
        )
    }

    func withAudioFormat(_ format: String?) -> MediaReferenceSummary {
        MediaReferenceSummary(
            kind: kind,
            source: source,
            mimeType: mimeType,
            byteCount: byteCount,
            location: location,
            isLoadable: isLoadable,
            error: error,
            imageDetail: imageDetail,
            imageResizedHeight: imageResizedHeight,
            imageResizedWidth: imageResizedWidth,
            imageMinPixels: imageMinPixels,
            imageMaxPixels: imageMaxPixels,
            audioFormat: format,
            videoMinPixels: videoMinPixels,
            videoMaxPixels: videoMaxPixels,
            videoFPS: videoFPS,
            videoNFrames: videoNFrames,
            videoMinFrames: videoMinFrames,
            videoMaxFrames: videoMaxFrames
        )
    }

    func withVideoOptions(
        minPixels: Int?,
        maxPixels: Int?,
        fps: Double?,
        nframes: Int?,
        minFrames: Int?,
        maxFrames: Int?
    ) -> MediaReferenceSummary {
        MediaReferenceSummary(
            kind: kind,
            source: source,
            mimeType: mimeType,
            byteCount: byteCount,
            location: location,
            isLoadable: isLoadable,
            error: error,
            imageDetail: imageDetail,
            imageResizedHeight: imageResizedHeight,
            imageResizedWidth: imageResizedWidth,
            imageMinPixels: imageMinPixels,
            imageMaxPixels: imageMaxPixels,
            audioFormat: audioFormat,
            videoMinPixels: minPixels,
            videoMaxPixels: maxPixels,
            videoFPS: fps,
            videoNFrames: nframes,
            videoMinFrames: minFrames,
            videoMaxFrames: maxFrames
        )
    }
}
