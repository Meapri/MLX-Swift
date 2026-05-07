import Foundation

public struct SafetensorsTensorMetadata: Codable, Equatable, Sendable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let dataOffsets: [Int64]

    public init(name: String, dtype: String, shape: [Int], dataOffsets: [Int64]) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.dataOffsets = dataOffsets
    }
}

public struct SafetensorsFileMetadata: Codable, Equatable, Sendable {
    public let name: String
    public let headerLength: Int64?
    public let tensorCount: Int
    public let dtypes: [String]
    public let declaredDataBytes: Int64?
    public let metadata: [String: String]
    public let isReadable: Bool
    public let error: String?

    public init(
        name: String,
        headerLength: Int64?,
        tensorCount: Int,
        dtypes: [String],
        declaredDataBytes: Int64?,
        metadata: [String: String],
        isReadable: Bool,
        error: String?
    ) {
        self.name = name
        self.headerLength = headerLength
        self.tensorCount = tensorCount
        self.dtypes = dtypes
        self.declaredDataBytes = declaredDataBytes
        self.metadata = metadata
        self.isReadable = isReadable
        self.error = error
    }
}

public enum SafetensorsMetadataError: Error, CustomStringConvertible, Equatable {
    case fileTooSmall(String)
    case headerTooLarge(String, UInt64)
    case invalidUTF8(String)
    case invalidHeaderJSON(String)
    case invalidTensorEntry(String)

    public var description: String {
        switch self {
        case .fileTooSmall(let name):
            return "\(name) is too small to contain a safetensors header"
        case .headerTooLarge(let name, let length):
            return "\(name) declares an invalid safetensors header length: \(length)"
        case .invalidUTF8(let name):
            return "\(name) safetensors header is not valid UTF-8"
        case .invalidHeaderJSON(let name):
            return "\(name) safetensors header is not a JSON object"
        case .invalidTensorEntry(let name):
            return "\(name) contains an invalid tensor metadata entry"
        }
    }
}

public enum SafetensorsMetadataReader {
    public static func readFileMetadata(at url: URL) -> SafetensorsFileMetadata {
        do {
            return try readFileMetadataThrowing(at: url)
        } catch {
            return SafetensorsFileMetadata(
                name: url.lastPathComponent,
                headerLength: nil,
                tensorCount: 0,
                dtypes: [],
                declaredDataBytes: nil,
                metadata: [:],
                isReadable: false,
                error: String(describing: error)
            )
        }
    }

    public static func readFileMetadataThrowing(at url: URL) throws -> SafetensorsFileMetadata {
        let parsed = try parseHeader(at: url)

        return SafetensorsFileMetadata(
            name: url.lastPathComponent,
            headerLength: Int64(parsed.headerLength),
            tensorCount: parsed.tensors.count,
            dtypes: Array(Set(parsed.tensors.map(\.dtype))).sorted(),
            declaredDataBytes: parsed.tensors.map { $0.dataOffsets.max() ?? 0 }.max(),
            metadata: parsed.metadata,
            isReadable: true,
            error: nil
        )
    }

    public static func readTensorMetadata(at url: URL) throws -> [SafetensorsTensorMetadata] {
        try parseHeader(at: url).tensors.sorted { $0.name < $1.name }
    }

    private static func parseHeader(
        at url: URL
    ) throws -> (headerLength: UInt64, metadata: [String: String], tensors: [SafetensorsTensorMetadata]) {
        let name = url.lastPathComponent
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
        }

        let lengthData = try handle.read(upToCount: 8) ?? Data()
        guard lengthData.count == 8 else {
            throw SafetensorsMetadataError.fileTooSmall(name)
        }

        let headerLength = UInt64(littleEndianBytes: lengthData.prefix(8))
        guard headerLength <= UInt64(Int.max) else {
            throw SafetensorsMetadataError.headerTooLarge(name, headerLength)
        }

        let headerData = try handle.read(upToCount: Int(headerLength)) ?? Data()
        guard headerData.count == Int(headerLength) else {
            throw SafetensorsMetadataError.headerTooLarge(name, headerLength)
        }
        guard let headerString = String(data: headerData, encoding: .utf8) else {
            throw SafetensorsMetadataError.invalidUTF8(name)
        }
        guard let headerJSONData = headerString.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
              let header = try JSONDecoder().decode(JSONValue.self, from: headerJSONData).objectValue
        else {
            throw SafetensorsMetadataError.invalidHeaderJSON(name)
        }

        var metadata: [String: String] = [:]
        var tensors: [SafetensorsTensorMetadata] = []
        for (key, value) in header {
            if key == "__metadata__" {
                metadata = value.objectValue?.compactMapValues(\.stringValue) ?? [:]
                continue
            }
            guard let object = value.objectValue,
                  let dtype = object["dtype"]?.stringValue,
                  let shape = object["shape"]?.arrayValue?.compactMap(\.intValue),
                  let offsets = object["data_offsets"]?.arrayValue?.compactMap(\.intValue),
                  offsets.count == 2
            else {
                throw SafetensorsMetadataError.invalidTensorEntry(key)
            }
            tensors.append(
                SafetensorsTensorMetadata(
                    name: key,
                    dtype: dtype,
                    shape: shape,
                    dataOffsets: offsets.map(Int64.init)
                )
            )
        }
        return (headerLength, metadata, tensors)
    }
}

private extension UInt64 {
    init(littleEndianBytes bytes: Data.SubSequence) {
        var value: UInt64 = 0
        for (index, byte) in bytes.enumerated() {
            value |= UInt64(byte) << UInt64(index * 8)
        }
        self = value
    }
}
