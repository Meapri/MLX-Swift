// swift-tools-version: 6.0

import Foundation
import PackageDescription

let environment = ProcessInfo.processInfo.environment
let enableMLXBackendDependencies = environment["MLXVLM_ENABLE_MLX_BACKEND"] == "1"
let enableRealMLXAPIImplementation = environment["MLXVLM_ENABLE_REAL_MLX_API"] == "1"
let enableTokenizerIntegrationDependencies = environment["MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS"] == "1"
let enableHuggingFaceDownloaderDependencies = environment["MLXVLM_ENABLE_HUGGINGFACE_DOWNLOADER"] == "1"
let useLocalMLXDependencies = environment["MLXVLM_USE_LOCAL_MLX"] == "1"
let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()

var dependencies: [Package.Dependency] = []
var mlxBackendDependencies: [Target.Dependency] = ["MLXVLMCore"]
var mlxBackendSwiftSettings: [SwiftSetting] = []

if enableRealMLXAPIImplementation {
    mlxBackendSwiftSettings.append(.define("MLXVLM_REAL_MLX_API"))
}

if enableHuggingFaceDownloaderDependencies {
    mlxBackendSwiftSettings.append(.define("MLXVLM_HUGGINGFACE_DOWNLOADER"))
}

if enableMLXBackendDependencies {
    let mlxSwiftPackageRef: String
    if let path = localDependencyPath(packageName: "mlx-swift", envPathKey: "MLXVLM_MLX_SWIFT_PATH") {
        dependencies.append(.package(name: "MLXSwift", path: path))
        mlxSwiftPackageRef = "MLXSwift"
    } else {
        dependencies.append(.package(url: "https://github.com/ml-explore/mlx-swift", "0.31.3"..<"0.32.0"))
        mlxSwiftPackageRef = "mlx-swift"
    }

    let mlxSwiftLMPackageRef: String
    if let path = localDependencyPath(packageName: "mlx-swift-lm", envPathKey: "MLXVLM_MLX_SWIFT_LM_PATH") {
        dependencies.append(.package(name: "MLXSwiftLM", path: path))
        mlxSwiftLMPackageRef = "MLXSwiftLM"
    } else {
        dependencies.append(.package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"))
        mlxSwiftLMPackageRef = "mlx-swift-lm"
    }

    mlxBackendDependencies.append(.product(name: "MLX", package: mlxSwiftPackageRef))
    mlxBackendDependencies.append(.product(name: "MLXLMCommon", package: mlxSwiftLMPackageRef))
    mlxBackendDependencies.append(.product(name: "MLXLLM", package: mlxSwiftLMPackageRef))
    mlxBackendDependencies.append(.product(name: "MLXVLM", package: mlxSwiftLMPackageRef))
    mlxBackendDependencies.append(.product(name: "MLXEmbedders", package: mlxSwiftLMPackageRef))

    if enableHuggingFaceDownloaderDependencies {
        dependencies.append(.package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"))
        mlxBackendDependencies.append(.product(name: "MLXHuggingFace", package: mlxSwiftLMPackageRef))
        mlxBackendDependencies.append(.product(name: "HuggingFace", package: "swift-huggingface"))
    }
}

if enableTokenizerIntegrationDependencies {
    let swiftTokenizersMLXPackageRef: String
    if let path = localDependencyPath(packageName: "swift-tokenizers-mlx", envPathKey: "MLXVLM_SWIFT_TOKENIZERS_MLX_PATH") {
        dependencies.append(.package(name: "SwiftTokenizersMLX", path: path))
        swiftTokenizersMLXPackageRef = "SwiftTokenizersMLX"
    } else {
        dependencies.append(.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx", from: "0.3.0"))
        swiftTokenizersMLXPackageRef = "swift-tokenizers-mlx"
    }
    mlxBackendDependencies.append(.product(name: "MLXLMTokenizers", package: swiftTokenizersMLXPackageRef))
}

func localDependencyPath(packageName: String, envPathKey: String) -> String? {
    if let explicitPath = environment[envPathKey], !explicitPath.isEmpty {
        return explicitPath
    }

    guard useLocalMLXDependencies else {
        return nil
    }

    return localDependencyCandidatePaths(packageName: packageName).first {
        FileManager.default.fileExists(atPath: $0)
    }
}

func localDependencyCandidatePaths(packageName: String) -> [String] {
    [
        packageRoot.appendingPathComponent("vendor").appendingPathComponent(aliasDirectoryName(for: packageName)).path,
        packageRoot.appendingPathComponent("Vendor").appendingPathComponent(aliasDirectoryName(for: packageName)).path,
        packageRoot.appendingPathComponent("Dependencies").appendingPathComponent(aliasDirectoryName(for: packageName)).path,
        packageRoot.deletingLastPathComponent().appendingPathComponent(aliasDirectoryName(for: packageName)).path,
        packageRoot.appendingPathComponent("vendor").appendingPathComponent(packageName).path,
        packageRoot.appendingPathComponent("Vendor").appendingPathComponent(packageName).path,
        packageRoot.appendingPathComponent("Dependencies").appendingPathComponent(packageName).path,
        packageRoot.appendingPathComponent(".build").appendingPathComponent("checkouts").appendingPathComponent(packageName).path,
        packageRoot.deletingLastPathComponent().appendingPathComponent(packageName).path,
    ].filter {
        URL(fileURLWithPath: $0).standardizedFileURL.path.caseInsensitiveCompare(packageRoot.standardizedFileURL.path) != .orderedSame
    }
}

func aliasDirectoryName(for packageName: String) -> String {
    switch packageName {
    case "mlx-swift":
        return "MLXSwift"
    case "mlx-swift-lm":
        return "MLXSwiftLM"
    case "swift-tokenizers-mlx":
        return "SwiftTokenizersMLX"
    default:
        return packageName
    }
}

let package = Package(
    name: "MLXVLMPort",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "MLXVLMCore", targets: ["MLXVLMCore"]),
        .library(name: "MLXVLMMLXBackend", targets: ["MLXVLMMLXBackend"]),
        .executable(name: "mlx-vlm-swift", targets: ["MLXVLMCli"]),
        .executable(name: "mlx-vlm-settings", targets: ["MLXVLMSettingsApp"]),
    ],
    dependencies: dependencies,
    targets: [
        .target(name: "MLXVLMCore"),
        .target(
            name: "MLXVLMMLXBackend",
            dependencies: mlxBackendDependencies,
            swiftSettings: mlxBackendSwiftSettings
        ),
        .executableTarget(
            name: "MLXVLMCli",
            dependencies: ["MLXVLMCore", "MLXVLMMLXBackend"]
        ),
        .executableTarget(
            name: "MLXVLMSettingsApp",
            dependencies: ["MLXVLMCore", "MLXVLMMLXBackend"]
        ),
    ]
)
