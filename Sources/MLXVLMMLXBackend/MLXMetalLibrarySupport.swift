import Foundation

public struct MLXMetalLibraryReport: Codable, Equatable, Sendable {
    public let requiredFileName: String
    public let executableDirectory: String?
    public let workingDirectory: String
    public let alreadyAvailable: Bool
    public let installed: Bool
    public let sourcePath: String?
    public let checkedPaths: [String]
    public let destinationPaths: [String]
    public let errors: [String]
    public let ready: Bool
    public let instructions: [String]

    public init(
        requiredFileName: String,
        executableDirectory: String?,
        workingDirectory: String,
        alreadyAvailable: Bool,
        installed: Bool,
        sourcePath: String?,
        checkedPaths: [String],
        destinationPaths: [String],
        errors: [String],
        instructions: [String]
    ) {
        self.requiredFileName = requiredFileName
        self.executableDirectory = executableDirectory
        self.workingDirectory = workingDirectory
        self.alreadyAvailable = alreadyAvailable
        self.installed = installed
        self.sourcePath = sourcePath
        self.checkedPaths = checkedPaths
        self.destinationPaths = destinationPaths
        self.errors = errors
        self.ready = alreadyAvailable || installed || destinationPaths.contains {
            FileManager.default.fileExists(atPath: $0)
        }
        self.instructions = instructions
    }
}

public enum MLXMetalLibrarySupport {
    public static let defaultMetallibName = "default.metallib"
    public static let upstreamMetallibName = "mlx.metallib"
    public static let swiftPMBundleDirectory = "mlx-swift_Cmlx.bundle"

    public static func ensureDefaultLibraryAvailable(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        currentDirectory: String = FileManager.default.currentDirectoryPath,
        executablePath: String? = CommandLine.arguments.first,
        install: Bool = true
    ) -> MLXMetalLibraryReport {
        let executableDirectory = executablePath.map {
            URL(fileURLWithPath: $0).deletingLastPathComponent().standardizedFileURL.path
        }
        let workingDirectory = URL(fileURLWithPath: currentDirectory).standardizedFileURL.path
        let destinations = destinationPaths(
            executableDirectory: executableDirectory,
            workingDirectory: workingDirectory
        )
        let existingDestination = destinations.first { fileManager.fileExists(atPath: $0) }
        let candidates = sourceCandidates(
            environment: environment,
            executableDirectory: executableDirectory,
            workingDirectory: workingDirectory
        )
        if let existingDestination, !install {
            return MLXMetalLibraryReport(
                requiredFileName: defaultMetallibName,
                executableDirectory: executableDirectory,
                workingDirectory: workingDirectory,
                alreadyAvailable: true,
                installed: false,
                sourcePath: existingDestination,
                checkedPaths: candidates,
                destinationPaths: destinations,
                errors: [],
                instructions: []
            )
        }

        guard let source = existingDestination ?? candidates.first(where: { fileManager.fileExists(atPath: $0) }) else {
            return MLXMetalLibraryReport(
                requiredFileName: defaultMetallibName,
                executableDirectory: executableDirectory,
                workingDirectory: workingDirectory,
                alreadyAvailable: false,
                installed: false,
                sourcePath: nil,
                checkedPaths: candidates,
                destinationPaths: destinations,
                errors: [],
                instructions: [
                    "Set MLXVLM_MLX_METALLIB_PATH=/path/to/mlx.metallib or default.metallib.",
                    "For SwiftPM debug runs, copy mlx.metallib to default.metallib next to the mlx-vlm-swift executable or into mlx-swift_Cmlx.bundle/default.metallib.",
                ]
            )
        }

        guard install else {
            return MLXMetalLibraryReport(
                requiredFileName: defaultMetallibName,
                executableDirectory: executableDirectory,
                workingDirectory: workingDirectory,
                alreadyAvailable: false,
                installed: false,
                sourcePath: source,
                checkedPaths: candidates,
                destinationPaths: destinations,
                errors: [],
                instructions: [
                    "Run the server to install \(source) as \(defaultMetallibName), or copy it manually to one destination path.",
                ]
            )
        }

        var installed = false
        var errors: [String] = []
        for destination in destinations {
            do {
                if try installLibrary(source: source, destination: destination, fileManager: fileManager) {
                    installed = true
                }
            } catch {
                errors.append("\(destination): \(error)")
            }
        }
        let ready = existingDestination != nil || installed

        return MLXMetalLibraryReport(
            requiredFileName: defaultMetallibName,
            executableDirectory: executableDirectory,
            workingDirectory: workingDirectory,
            alreadyAvailable: existingDestination != nil,
            installed: installed,
            sourcePath: source,
            checkedPaths: candidates,
            destinationPaths: destinations,
            errors: errors,
            instructions: ready ? [] : [
                "Copy \(source) to one of the destination paths as \(defaultMetallibName).",
            ]
        )
    }

    private static func destinationPaths(
        executableDirectory: String?,
        workingDirectory: String
    ) -> [String] {
        unique([
            executableDirectory.map { URL(fileURLWithPath: $0).appendingPathComponent(defaultMetallibName).path },
            executableDirectory.map {
                URL(fileURLWithPath: $0)
                    .appendingPathComponent(swiftPMBundleDirectory)
                    .appendingPathComponent(defaultMetallibName)
                    .path
            },
            URL(fileURLWithPath: workingDirectory).appendingPathComponent(defaultMetallibName).path,
            URL(fileURLWithPath: workingDirectory)
                .appendingPathComponent(swiftPMBundleDirectory)
                .appendingPathComponent(defaultMetallibName)
                .path,
        ].compactMap { $0 })
    }

    private static func sourceCandidates(
        environment: [String: String],
        executableDirectory: String?,
        workingDirectory: String
    ) -> [String] {
        var candidates: [String] = []
        if let explicit = environment["MLXVLM_MLX_METALLIB_PATH"], !explicit.isEmpty {
            candidates.append(expandTilde(explicit))
        }

        let bases = [executableDirectory, workingDirectory].compactMap { $0 }
        for base in bases {
            candidates.append(URL(fileURLWithPath: base).appendingPathComponent(defaultMetallibName).path)
            candidates.append(URL(fileURLWithPath: base).appendingPathComponent(upstreamMetallibName).path)
        }
        candidates.append(contentsOf: pythonMLXMetallibCandidates(workingDirectory: workingDirectory))
        if let home = environment["HOME"], !home.isEmpty {
            candidates.append(contentsOf: pythonMLXMetallibCandidates(workingDirectory: home))
        }
        return unique(candidates)
    }

    private static func pythonMLXMetallibCandidates(workingDirectory: String) -> [String] {
        let root = URL(fileURLWithPath: workingDirectory)
        return [
            root.appendingPathComponent(".venv/lib/python3.12/site-packages/mlx/lib/\(upstreamMetallibName)").path,
            root.appendingPathComponent(".venv/lib/python3.11/site-packages/mlx/lib/\(upstreamMetallibName)").path,
            root.appendingPathComponent("Library/Python/3.12/lib/python/site-packages/mlx/lib/\(upstreamMetallibName)").path,
            root.appendingPathComponent("Library/Python/3.11/lib/python/site-packages/mlx/lib/\(upstreamMetallibName)").path,
            root.appendingPathComponent("Library/Python/3.9/lib/python/site-packages/mlx/lib/\(upstreamMetallibName)").path,
        ]
    }

    private static func installLibrary(
        source: String,
        destination: String,
        fileManager: FileManager
    ) throws -> Bool {
        if fileManager.fileExists(atPath: destination) {
            return false
        }
        let destinationURL = URL(fileURLWithPath: destination)
        try fileManager.createDirectory(
            at: destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try fileManager.copyItem(
            at: URL(fileURLWithPath: source),
            to: destinationURL
        )
        return true
    }

    private static func expandTilde(_ path: String) -> String {
        NSString(string: path).expandingTildeInPath
    }

    private static func unique(_ values: [String]) -> [String] {
        var seen: Set<String> = []
        var result: [String] = []
        for value in values where !value.isEmpty && !seen.contains(value) {
            seen.insert(value)
            result.append(value)
        }
        return result
    }
}
