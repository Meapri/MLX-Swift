import Foundation

public struct WeightTensorDataHandle: Codable, Equatable, Sendable {
    public let originalKey: String
    public let sanitizedKey: String
    public let shardName: String
    public let dtype: String
    public let shape: [Int]
    public let elementCount: Int
    public let dtypeByteWidth: Int?
    public let relativeDataOffsets: [Int64]
    public let absoluteDataOffsets: [Int64]?
    public let byteCount: Int64?
    public let expectedByteCount: Int64?
    public let byteCountMatchesShape: Bool?
    public let isReadable: Bool
    public let error: String?

    public init(
        originalKey: String,
        sanitizedKey: String,
        shardName: String,
        dtype: String,
        shape: [Int],
        elementCount: Int,
        dtypeByteWidth: Int?,
        relativeDataOffsets: [Int64],
        absoluteDataOffsets: [Int64]?,
        byteCount: Int64?,
        expectedByteCount: Int64?,
        byteCountMatchesShape: Bool?,
        isReadable: Bool,
        error: String?
    ) {
        self.originalKey = originalKey
        self.sanitizedKey = sanitizedKey
        self.shardName = shardName
        self.dtype = dtype
        self.shape = shape
        self.elementCount = elementCount
        self.dtypeByteWidth = dtypeByteWidth
        self.relativeDataOffsets = relativeDataOffsets
        self.absoluteDataOffsets = absoluteDataOffsets
        self.byteCount = byteCount
        self.expectedByteCount = expectedByteCount
        self.byteCountMatchesShape = byteCountMatchesShape
        self.isReadable = isReadable
        self.error = error
    }
}

public struct WeightDataCatalog: Codable, Equatable, Sendable {
    public let tensors: [WeightTensorDataHandle]
    public let unreadableTensorCount: Int
    public let byteMismatchCount: Int
    public let totalReadableBytes: Int64

    public init(tensors: [WeightTensorDataHandle]) {
        self.tensors = tensors.sorted { $0.sanitizedKey < $1.sanitizedKey }
        self.unreadableTensorCount = tensors.filter { !$0.isReadable }.count
        self.byteMismatchCount = tensors.filter { $0.byteCountMatchesShape == false }.count
        self.totalReadableBytes = tensors.reduce(0) { $0 + ($1.isReadable ? ($1.byteCount ?? 0) : 0) }
    }

    public var tensorCount: Int {
        tensors.count
    }

    public func tensor(named name: String) -> WeightTensorDataHandle? {
        tensors.first { $0.originalKey == name || $0.sanitizedKey == name }
    }
}

public struct WeightTensorDataPreview: Codable, Equatable, Sendable {
    public let tensor: WeightTensorDataHandle
    public let requestedByteCount: Int
    public let returnedByteCount: Int
    public let hexPrefix: String
    public let numericValues: [Double]?

    public init(
        tensor: WeightTensorDataHandle,
        requestedByteCount: Int,
        returnedByteCount: Int,
        hexPrefix: String,
        numericValues: [Double]?
    ) {
        self.tensor = tensor
        self.requestedByteCount = requestedByteCount
        self.returnedByteCount = returnedByteCount
        self.hexPrefix = hexPrefix
        self.numericValues = numericValues
    }
}

public struct WeightTensorPayload: Equatable, Sendable {
    public let tensor: WeightTensorDataHandle
    public let data: Data
    public let checksum: Int64

    public init(tensor: WeightTensorDataHandle, data: Data) {
        self.tensor = tensor
        self.data = data
        self.checksum = data.reduce(Int64(0)) { $0 + Int64($1) }
    }

    public var byteCount: Int {
        data.count
    }
}

public struct WeightTensorPayloadSummary: Codable, Equatable, Sendable {
    public let tensor: WeightTensorDataHandle
    public let byteCount: Int
    public let checksum: Int64

    public init(payload: WeightTensorPayload) {
        self.tensor = payload.tensor
        self.byteCount = payload.byteCount
        self.checksum = payload.checksum
    }
}

public enum WeightDataCatalogError: Error, CustomStringConvertible, Equatable {
    case tensorNotFound(String)
    case tensorNotReadable(String)
    case missingAbsoluteOffsets(String)
    case invalidReadLength(Int)

    public var description: String {
        switch self {
        case .tensorNotFound(let name):
            return "Tensor was not found in weight data catalog: \(name)"
        case .tensorNotReadable(let name):
            return "Tensor is not readable: \(name)"
        case .missingAbsoluteOffsets(let name):
            return "Tensor is missing absolute data offsets: \(name)"
        case .invalidReadLength(let length):
            return "Preview byte count must be positive, got \(length)"
        }
    }
}

public struct WeightDataCatalogBuilder {
    public let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    public func catalog(for descriptor: ModelDescriptor) -> WeightDataCatalog {
        let modelURL = URL(fileURLWithPath: descriptor.path, isDirectory: true)
        let weightCatalog = WeightCatalogBuilder(fileManager: fileManager).catalog(for: descriptor)
        let shardInfo = Dictionary(uniqueKeysWithValues: descriptor.safetensorsMetadata.map { ($0.name, $0) })
        let tensors = weightCatalog.tensors.map { tensor in
            handle(for: tensor, modelURL: modelURL, shardInfo: shardInfo[tensor.shardName])
        }
        return WeightDataCatalog(tensors: tensors)
    }

    public func previewTensor(
        named name: String,
        descriptor: ModelDescriptor,
        maxBytes: Int = 32
    ) throws -> WeightTensorDataPreview {
        guard maxBytes > 0 else {
            throw WeightDataCatalogError.invalidReadLength(maxBytes)
        }
        let catalog = catalog(for: descriptor)
        guard let tensor = catalog.tensor(named: name) else {
            throw WeightDataCatalogError.tensorNotFound(name)
        }
        guard tensor.isReadable else {
            throw WeightDataCatalogError.tensorNotReadable(name)
        }
        guard let absoluteDataOffsets = tensor.absoluteDataOffsets,
              absoluteDataOffsets.count == 2,
              let byteCount = tensor.byteCount
        else {
            throw WeightDataCatalogError.missingAbsoluteOffsets(name)
        }

        let modelURL = URL(fileURLWithPath: descriptor.path, isDirectory: true)
        let shardURL = modelURL.appendingPathComponent(tensor.shardName)
        let length = min(maxBytes, Int(byteCount))
        let data = try readBytes(at: shardURL, offset: UInt64(absoluteDataOffsets[0]), count: length)
        return WeightTensorDataPreview(
            tensor: tensor,
            requestedByteCount: maxBytes,
            returnedByteCount: data.count,
            hexPrefix: data.map { String(format: "%02x", $0) }.joined(),
            numericValues: numericValues(data: data, dtype: tensor.dtype)
        )
    }

    public func readTensorPayload(
        named name: String,
        descriptor: ModelDescriptor
    ) throws -> WeightTensorPayload {
        let catalog = catalog(for: descriptor)
        guard let tensor = catalog.tensor(named: name) else {
            throw WeightDataCatalogError.tensorNotFound(name)
        }
        return try readTensorPayload(handle: tensor, descriptor: descriptor)
    }

    public func readTensorPayload(
        handle tensor: WeightTensorDataHandle,
        descriptor: ModelDescriptor
    ) throws -> WeightTensorPayload {
        guard tensor.isReadable else {
            throw WeightDataCatalogError.tensorNotReadable(tensor.sanitizedKey)
        }
        guard let absoluteDataOffsets = tensor.absoluteDataOffsets,
              absoluteDataOffsets.count == 2,
              let byteCount = tensor.byteCount
        else {
            throw WeightDataCatalogError.missingAbsoluteOffsets(tensor.sanitizedKey)
        }
        let modelURL = URL(fileURLWithPath: descriptor.path, isDirectory: true)
        let shardURL = modelURL.appendingPathComponent(tensor.shardName)
        let data = try readBytes(at: shardURL, offset: UInt64(absoluteDataOffsets[0]), count: Int(byteCount))
        return WeightTensorPayload(tensor: tensor, data: data)
    }

    private func handle(
        for tensor: WeightCatalogTensor,
        modelURL: URL,
        shardInfo: SafetensorsFileMetadata?
    ) -> WeightTensorDataHandle {
        let dtypeByteWidth = Self.byteWidth(dtype: tensor.dtype)
        let byteCount = tensor.dataOffsets.count == 2 ? tensor.dataOffsets[1] - tensor.dataOffsets[0] : nil
        let expectedByteCount = dtypeByteWidth.map { Int64(tensor.elementCount * $0) }
        let byteCountMatchesShape = byteCount.flatMap { count in
            expectedByteCount.map { count == $0 }
        }

        guard tensor.dataOffsets.count == 2 else {
            return unreadableHandle(
                tensor: tensor,
                dtypeByteWidth: dtypeByteWidth,
                byteCount: byteCount,
                expectedByteCount: expectedByteCount,
                byteCountMatchesShape: byteCountMatchesShape,
                error: "Tensor data_offsets must contain exactly two entries."
            )
        }
        guard tensor.dataOffsets[0] >= 0, tensor.dataOffsets[1] >= tensor.dataOffsets[0] else {
            return unreadableHandle(
                tensor: tensor,
                dtypeByteWidth: dtypeByteWidth,
                byteCount: byteCount,
                expectedByteCount: expectedByteCount,
                byteCountMatchesShape: byteCountMatchesShape,
                error: "Tensor data_offsets are not a valid non-negative range."
            )
        }
        guard let headerLength = shardInfo?.headerLength else {
            return unreadableHandle(
                tensor: tensor,
                dtypeByteWidth: dtypeByteWidth,
                byteCount: byteCount,
                expectedByteCount: expectedByteCount,
                byteCountMatchesShape: byteCountMatchesShape,
                error: "Shard safetensors header length is unavailable."
            )
        }

        let shardURL = modelURL.appendingPathComponent(tensor.shardName)
        let fileByteCount = fileSize(at: shardURL)
        let dataStart = 8 + headerLength
        let absoluteOffsets = [
            dataStart + tensor.dataOffsets[0],
            dataStart + tensor.dataOffsets[1],
        ]
        if let fileByteCount, absoluteOffsets[1] > fileByteCount {
            return unreadableHandle(
                tensor: tensor,
                dtypeByteWidth: dtypeByteWidth,
                byteCount: byteCount,
                expectedByteCount: expectedByteCount,
                byteCountMatchesShape: byteCountMatchesShape,
                absoluteDataOffsets: absoluteOffsets,
                error: "Tensor byte range extends beyond shard file size."
            )
        }
        if dtypeByteWidth == nil {
            return unreadableHandle(
                tensor: tensor,
                dtypeByteWidth: dtypeByteWidth,
                byteCount: byteCount,
                expectedByteCount: expectedByteCount,
                byteCountMatchesShape: byteCountMatchesShape,
                absoluteDataOffsets: absoluteOffsets,
                error: "Unsupported safetensors dtype byte width: \(tensor.dtype)."
            )
        }
        if byteCountMatchesShape == false {
            return unreadableHandle(
                tensor: tensor,
                dtypeByteWidth: dtypeByteWidth,
                byteCount: byteCount,
                expectedByteCount: expectedByteCount,
                byteCountMatchesShape: byteCountMatchesShape,
                absoluteDataOffsets: absoluteOffsets,
                error: "Tensor byte range does not match dtype byte width multiplied by shape."
            )
        }

        return WeightTensorDataHandle(
            originalKey: tensor.originalKey,
            sanitizedKey: tensor.sanitizedKey,
            shardName: tensor.shardName,
            dtype: tensor.dtype,
            shape: tensor.shape,
            elementCount: tensor.elementCount,
            dtypeByteWidth: dtypeByteWidth,
            relativeDataOffsets: tensor.dataOffsets,
            absoluteDataOffsets: absoluteOffsets,
            byteCount: byteCount,
            expectedByteCount: expectedByteCount,
            byteCountMatchesShape: byteCountMatchesShape,
            isReadable: true,
            error: nil
        )
    }

    private func unreadableHandle(
        tensor: WeightCatalogTensor,
        dtypeByteWidth: Int?,
        byteCount: Int64?,
        expectedByteCount: Int64?,
        byteCountMatchesShape: Bool?,
        absoluteDataOffsets: [Int64]? = nil,
        error: String
    ) -> WeightTensorDataHandle {
        WeightTensorDataHandle(
            originalKey: tensor.originalKey,
            sanitizedKey: tensor.sanitizedKey,
            shardName: tensor.shardName,
            dtype: tensor.dtype,
            shape: tensor.shape,
            elementCount: tensor.elementCount,
            dtypeByteWidth: dtypeByteWidth,
            relativeDataOffsets: tensor.dataOffsets,
            absoluteDataOffsets: absoluteDataOffsets,
            byteCount: byteCount,
            expectedByteCount: expectedByteCount,
            byteCountMatchesShape: byteCountMatchesShape,
            isReadable: false,
            error: error
        )
    }

    private func fileSize(at url: URL) -> Int64? {
        let resolvedURL = url.resolvingSymlinksInPath()
        guard let attributes = try? fileManager.attributesOfItem(atPath: resolvedURL.path),
              let size = attributes[.size] as? NSNumber
        else {
            return nil
        }
        return size.int64Value
    }

    private func readBytes(at url: URL, offset: UInt64, count: Int) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
        }
        try handle.seek(toOffset: offset)
        return try handle.read(upToCount: count) ?? Data()
    }

    public static func byteWidth(dtype: String) -> Int? {
        switch dtype.uppercased() {
        case "BOOL", "I8", "U8", "F8_E5M2", "F8_E4M3":
            return 1
        case "I16", "U16", "F16", "BF16":
            return 2
        case "I32", "U32", "F32":
            return 4
        case "I64", "U64", "F64":
            return 8
        default:
            return nil
        }
    }

    private func numericValues(data: Data, dtype: String) -> [Double]? {
        guard let byteWidth = Self.byteWidth(dtype: dtype), byteWidth > 0 else {
            return nil
        }
        let bytes = Array(data)
        let elementCount = bytes.count / byteWidth
        guard elementCount > 0 else {
            return []
        }
        return (0..<elementCount).compactMap { index in
            let start = index * byteWidth
            return numericValue(bytes: Array(bytes[start..<(start + byteWidth)]), dtype: dtype)
        }
    }

    private func numericValue(bytes: [UInt8], dtype: String) -> Double? {
        switch dtype.uppercased() {
        case "BOOL":
            return bytes[0] == 0 ? 0 : 1
        case "U8":
            return Double(bytes[0])
        case "I8":
            return Double(Int8(bitPattern: bytes[0]))
        case "U16":
            return Double(UInt16(littleEndianBytes: bytes))
        case "I16":
            return Double(Int16(bitPattern: UInt16(littleEndianBytes: bytes)))
        case "U32":
            return Double(UInt32(littleEndianBytes: bytes))
        case "I32":
            return Double(Int32(bitPattern: UInt32(littleEndianBytes: bytes)))
        case "U64":
            return Double(UInt64(littleEndianBytes: bytes))
        case "I64":
            return Double(Int64(bitPattern: UInt64(littleEndianBytes: bytes)))
        case "F16":
            return Double(Self.float32(fromIEEEFloat16Bits: UInt16(littleEndianBytes: bytes)))
        case "BF16":
            return Double(Self.float32(fromBFloat16Bits: UInt16(littleEndianBytes: bytes)))
        case "F32":
            return Double(Float(bitPattern: UInt32(littleEndianBytes: bytes)))
        case "F64":
            return Double(bitPattern: UInt64(littleEndianBytes: bytes))
        default:
            return nil
        }
    }

    private static func float32(fromBFloat16Bits bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    private static func float32(fromIEEEFloat16Bits bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        let exponent = Int((bits & 0x7c00) >> 10)
        let fraction = UInt32(bits & 0x03ff)

        if exponent == 0 {
            if fraction == 0 {
                return Float(bitPattern: sign)
            }
            var mantissa = fraction
            var adjustedExponent = -14
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1
                adjustedExponent -= 1
            }
            mantissa &= 0x03ff
            let exponentBits = UInt32(adjustedExponent + 127) << 23
            return Float(bitPattern: sign | exponentBits | (mantissa << 13))
        }

        if exponent == 31 {
            let exponentBits: UInt32 = 0xff << 23
            return Float(bitPattern: sign | exponentBits | (fraction << 13))
        }

        let exponentBits = UInt32(exponent - 15 + 127) << 23
        return Float(bitPattern: sign | exponentBits | (fraction << 13))
    }
}

private extension UInt16 {
    init(littleEndianBytes bytes: [UInt8]) {
        self = UInt16(bytes[0]) | (UInt16(bytes[1]) << 8)
    }
}

private extension UInt32 {
    init(littleEndianBytes bytes: [UInt8]) {
        self = UInt32(bytes[0]) |
            (UInt32(bytes[1]) << 8) |
            (UInt32(bytes[2]) << 16) |
            (UInt32(bytes[3]) << 24)
    }
}

private extension UInt64 {
    init(littleEndianBytes bytes: [UInt8]) {
        self = UInt64(bytes[0]) |
            (UInt64(bytes[1]) << 8) |
            (UInt64(bytes[2]) << 16) |
            (UInt64(bytes[3]) << 24) |
            (UInt64(bytes[4]) << 32) |
            (UInt64(bytes[5]) << 40) |
            (UInt64(bytes[6]) << 48) |
            (UInt64(bytes[7]) << 56)
    }
}
