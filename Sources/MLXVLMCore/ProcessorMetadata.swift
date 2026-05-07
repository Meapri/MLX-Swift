import Foundation

public struct ProcessorVisionMetadata: Codable, Equatable, Sendable {
    public let source: String?
    public let processorType: String?
    public let patchSize: Int?
    public let temporalPatchSize: Int?
    public let mergeSize: Int?
    public let minPixels: Int?
    public let maxPixels: Int?
    public let imageMean: [Double]?
    public let imageStd: [Double]?
    public let rescaleFactor: Double?
    public let doRescale: Bool?
    public let doNormalize: Bool?
    public let doConvertRGB: Bool?

    public init(
        source: String?,
        processorType: String?,
        patchSize: Int?,
        temporalPatchSize: Int?,
        mergeSize: Int?,
        minPixels: Int?,
        maxPixels: Int?,
        imageMean: [Double]?,
        imageStd: [Double]?,
        rescaleFactor: Double?,
        doRescale: Bool?,
        doNormalize: Bool?,
        doConvertRGB: Bool?
    ) {
        self.source = source
        self.processorType = processorType
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.mergeSize = mergeSize
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.rescaleFactor = rescaleFactor
        self.doRescale = doRescale
        self.doNormalize = doNormalize
        self.doConvertRGB = doConvertRGB
    }
}

public struct ProcessorVideoMetadata: Codable, Equatable, Sendable {
    public let source: String?
    public let processorType: String?
    public let patchSize: Int?
    public let temporalPatchSize: Int?
    public let mergeSize: Int?
    public let minPixels: Int?
    public let maxPixels: Int?
    public let minFrames: Int?
    public let maxFrames: Int?
    public let fps: Double?

    public init(
        source: String?,
        processorType: String?,
        patchSize: Int?,
        temporalPatchSize: Int?,
        mergeSize: Int?,
        minPixels: Int?,
        maxPixels: Int?,
        minFrames: Int?,
        maxFrames: Int?,
        fps: Double?
    ) {
        self.source = source
        self.processorType = processorType
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.mergeSize = mergeSize
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        self.minFrames = minFrames
        self.maxFrames = maxFrames
        self.fps = fps
    }
}

public struct ProcessorMetadata: Codable, Equatable, Sendable {
    public let hasProcessorConfig: Bool
    public let hasPreprocessorConfig: Bool
    public let hasVideoPreprocessorConfig: Bool
    public let chatTemplate: String?
    public let image: ProcessorVisionMetadata?
    public let video: ProcessorVideoMetadata?
    public let error: String?

    public init(
        hasProcessorConfig: Bool,
        hasPreprocessorConfig: Bool,
        hasVideoPreprocessorConfig: Bool,
        chatTemplate: String?,
        image: ProcessorVisionMetadata?,
        video: ProcessorVideoMetadata?,
        error: String?
    ) {
        self.hasProcessorConfig = hasProcessorConfig
        self.hasPreprocessorConfig = hasPreprocessorConfig
        self.hasVideoPreprocessorConfig = hasVideoPreprocessorConfig
        self.chatTemplate = chatTemplate
        self.image = image
        self.video = video
        self.error = error
    }

    public func qwenImageGridConfig(defaultPatchSize: Int = 14) -> QwenVLImageGridConfig {
        QwenVLImageGridConfig(
            patchSize: image?.patchSize ?? defaultPatchSize,
            temporalPatchSize: image?.temporalPatchSize ?? 2,
            mergeSize: image?.mergeSize ?? 2,
            minPixels: image?.minPixels ?? 56 * 56,
            maxPixels: image?.maxPixels ?? 14 * 14 * 4 * 1280
        )
    }

    public func qwenPatchNormalization() -> QwenVLPatchNormalization {
        QwenVLPatchNormalization(
            imageMean: image?.imageMean ?? [0.5, 0.5, 0.5],
            imageStd: image?.imageStd ?? [0.5, 0.5, 0.5],
            rescaleFactor: image?.rescaleFactor ?? 1.0 / 255.0,
            doRescale: image?.doRescale ?? true,
            doNormalize: image?.doNormalize ?? true
        )
    }
}

public struct ProcessorMetadataLoader {
    public let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    public func load(from modelURL: URL) -> ProcessorMetadata {
        do {
            let processor = try loadJSONObjectIfPresent(modelURL.appendingPathComponent("processor_config.json"))
            let preprocessor = try loadJSONObjectIfPresent(modelURL.appendingPathComponent("preprocessor_config.json"))
            let videoPreprocessor = try loadJSONObjectIfPresent(modelURL.appendingPathComponent("video_preprocessor_config.json"))

            return ProcessorMetadata(
                hasProcessorConfig: processor != nil,
                hasPreprocessorConfig: preprocessor != nil,
                hasVideoPreprocessorConfig: videoPreprocessor != nil,
                chatTemplate: processor?["chat_template"]?.stringValue,
                image: imageMetadata(processor: processor, preprocessor: preprocessor),
                video: videoMetadata(videoPreprocessor: videoPreprocessor, preprocessor: preprocessor),
                error: nil
            )
        } catch {
            return ProcessorMetadata(
                hasProcessorConfig: fileManager.fileExists(atPath: modelURL.appendingPathComponent("processor_config.json").path),
                hasPreprocessorConfig: fileManager.fileExists(atPath: modelURL.appendingPathComponent("preprocessor_config.json").path),
                hasVideoPreprocessorConfig: fileManager.fileExists(atPath: modelURL.appendingPathComponent("video_preprocessor_config.json").path),
                chatTemplate: nil,
                image: nil,
                video: nil,
                error: String(describing: error)
            )
        }
    }

    private func imageMetadata(
        processor: [String: JSONValue]?,
        preprocessor: [String: JSONValue]?
    ) -> ProcessorVisionMetadata? {
        var raw = preprocessor ?? [:]
        let imageProcessor = processor?["image_processor"]?.objectValue ?? [:]
        raw.merge(imageProcessor) { _, new in new }
        guard !raw.isEmpty else {
            return nil
        }
        return ProcessorVisionMetadata(
            source: source(preprocessor: preprocessor, override: imageProcessor, overrideName: "processor_config.json:image_processor"),
            processorType: raw["image_processor_type"]?.stringValue,
            patchSize: int(raw["patch_size"]),
            temporalPatchSize: int(raw["temporal_patch_size"]),
            mergeSize: int(raw["merge_size"]),
            minPixels: raw["min_pixels"]?.intValue ?? raw["size"]?.objectValue?["shortest_edge"]?.intValue,
            maxPixels: raw["max_pixels"]?.intValue ?? raw["size"]?.objectValue?["longest_edge"]?.intValue,
            imageMean: doubleArray(raw["image_mean"]),
            imageStd: doubleArray(raw["image_std"]),
            rescaleFactor: raw["rescale_factor"]?.doubleValue,
            doRescale: raw["do_rescale"]?.boolValue,
            doNormalize: raw["do_normalize"]?.boolValue,
            doConvertRGB: raw["do_convert_rgb"]?.boolValue
        )
    }

    private func videoMetadata(
        videoPreprocessor: [String: JSONValue]?,
        preprocessor: [String: JSONValue]?
    ) -> ProcessorVideoMetadata? {
        let raw = videoPreprocessor ?? preprocessor ?? [:]
        guard !raw.isEmpty else {
            return nil
        }
        return ProcessorVideoMetadata(
            source: videoPreprocessor != nil ? "video_preprocessor_config.json" : "preprocessor_config.json",
            processorType: raw["video_processor_type"]?.stringValue ?? raw["image_processor_type"]?.stringValue,
            patchSize: int(raw["patch_size"]),
            temporalPatchSize: int(raw["temporal_patch_size"]),
            mergeSize: int(raw["merge_size"]),
            minPixels: raw["min_pixels"]?.intValue ?? raw["size"]?.objectValue?["shortest_edge"]?.intValue,
            maxPixels: raw["max_pixels"]?.intValue ?? raw["size"]?.objectValue?["longest_edge"]?.intValue,
            minFrames: raw["min_frames"]?.intValue,
            maxFrames: raw["max_frames"]?.intValue,
            fps: raw["fps"]?.doubleValue
        )
    }

    private func source(
        preprocessor: [String: JSONValue]?,
        override: [String: JSONValue],
        overrideName: String
    ) -> String? {
        if !override.isEmpty {
            return preprocessor == nil ? overrideName : "preprocessor_config.json+\(overrideName)"
        }
        return preprocessor == nil ? nil : "preprocessor_config.json"
    }

    private func loadJSONObjectIfPresent(_ url: URL) throws -> [String: JSONValue]? {
        guard fileManager.fileExists(atPath: url.path) else {
            return nil
        }
        let json = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))
        guard let object = json.objectValue else {
            throw ModelStoreError.invalidConfig("\(url.lastPathComponent) must contain a JSON object")
        }
        return object
    }

    private func int(_ value: JSONValue?) -> Int? {
        if let int = value?.intValue {
            return int
        }
        if let object = value?.objectValue {
            return object["height"]?.intValue ?? object["width"]?.intValue
        }
        return nil
    }

    private func doubleArray(_ value: JSONValue?) -> [Double]? {
        value?.arrayValue?.compactMap(\.doubleValue)
    }
}
