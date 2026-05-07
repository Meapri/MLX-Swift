import Foundation

public struct BackendDependencyRequirement: Codable, Equatable, Sendable {
    public let packageName: String
    public let productHint: String
    public let url: String
    public let versionRequirement: String
    public let localCandidatePaths: [String]
    public let localCheckoutPresent: Bool

    public init(
        packageName: String,
        productHint: String,
        url: String,
        versionRequirement: String,
        localCandidatePaths: [String],
        localCheckoutPresent: Bool
    ) {
        self.packageName = packageName
        self.productHint = productHint
        self.url = url
        self.versionRequirement = versionRequirement
        self.localCandidatePaths = localCandidatePaths
        self.localCheckoutPresent = localCheckoutPresent
    }
}

public struct BackendDependencyPlan: Codable, Equatable, Sendable {
    public let packagePath: String
    public let packageDeclaresMLXSwift: Bool
    public let packageDeclaresMLXSwiftLM: Bool
    public let packageDeclaresSwiftTokenizersMLX: Bool
    public let packageDeclaresBackendTarget: Bool
    public let manifestSupportsMLXBackendToggle: Bool
    public let manifestSupportsTokenizerIntegrationToggle: Bool
    public let manifestSupportsLocalMLXDependencies: Bool
    public let manifestSupportsExplicitMLXPaths: Bool
    public let manifestSupportsExplicitTokenizerPath: Bool
    public let compatibilityShellBuildable: Bool
    public let canEnableMLXBackend: Bool
    public let requirements: [BackendDependencyRequirement]
    public let nextSteps: [String]

    public init(
        packagePath: String,
        packageDeclaresMLXSwift: Bool,
        packageDeclaresMLXSwiftLM: Bool,
        packageDeclaresSwiftTokenizersMLX: Bool,
        packageDeclaresBackendTarget: Bool,
        manifestSupportsMLXBackendToggle: Bool,
        manifestSupportsTokenizerIntegrationToggle: Bool,
        manifestSupportsLocalMLXDependencies: Bool,
        manifestSupportsExplicitMLXPaths: Bool,
        manifestSupportsExplicitTokenizerPath: Bool,
        compatibilityShellBuildable: Bool,
        requirements: [BackendDependencyRequirement],
        nextSteps: [String]
    ) {
        self.packagePath = packagePath
        self.packageDeclaresMLXSwift = packageDeclaresMLXSwift
        self.packageDeclaresMLXSwiftLM = packageDeclaresMLXSwiftLM
        self.packageDeclaresSwiftTokenizersMLX = packageDeclaresSwiftTokenizersMLX
        self.packageDeclaresBackendTarget = packageDeclaresBackendTarget
        self.manifestSupportsMLXBackendToggle = manifestSupportsMLXBackendToggle
        self.manifestSupportsTokenizerIntegrationToggle = manifestSupportsTokenizerIntegrationToggle
        self.manifestSupportsLocalMLXDependencies = manifestSupportsLocalMLXDependencies
        self.manifestSupportsExplicitMLXPaths = manifestSupportsExplicitMLXPaths
        self.manifestSupportsExplicitTokenizerPath = manifestSupportsExplicitTokenizerPath
        self.compatibilityShellBuildable = compatibilityShellBuildable
        self.requirements = requirements
        self.canEnableMLXBackend = packageDeclaresBackendTarget &&
            requirements.allSatisfy(\.localCheckoutPresent) &&
            packageDeclaresMLXSwift &&
            packageDeclaresMLXSwiftLM &&
            packageDeclaresSwiftTokenizersMLX
        self.nextSteps = nextSteps
    }
}

public struct BackendDependencyPlanner {
    public let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    public func plan(rootPath: String) -> BackendDependencyPlan {
        let rootURL = URL(fileURLWithPath: rootPath, isDirectory: true).standardizedFileURL
        let packageURL = rootURL.appendingPathComponent("Package.swift")
        let packageText = (try? String(contentsOf: packageURL, encoding: .utf8)) ?? ""
        let requirements = [
            requirement(
                packageName: "mlx-swift",
                productHint: "MLX",
                url: "https://github.com/ml-explore/mlx-swift",
                versionRequirement: ".upToNextMinor(from: \"0.31.3\")",
                rootURL: rootURL
            ),
            requirement(
                packageName: "mlx-swift-lm",
                productHint: "MLXLMCommon/MLXLLM/MLXVLM",
                url: "https://github.com/ml-explore/mlx-swift-lm",
                versionRequirement: ".upToNextMajor(from: \"3.31.3\")",
                rootURL: rootURL
            ),
            requirement(
                packageName: "swift-tokenizers-mlx",
                productHint: "MLXLMTokenizers",
                url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx",
                versionRequirement: ".upToNextMajor(from: \"0.3.0\")",
                rootURL: rootURL
            ),
        ]
        let declaresMLXSwift = packageText.contains("mlx-swift")
        let declaresMLXSwiftLM = packageText.contains("mlx-swift-lm")
        let declaresSwiftTokenizersMLX = packageText.contains("swift-tokenizers-mlx")
        let declaresBackendTarget = packageText.contains("MLXVLMMLXBackend") ||
            packageText.contains("MLXVLMBackend")
        let supportsBackendToggle = packageText.contains("MLXVLM_ENABLE_MLX_BACKEND")
        let supportsTokenizerToggle = packageText.contains("MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS")
        let supportsLocalDependencies = packageText.contains("MLXVLM_USE_LOCAL_MLX")
        let supportsExplicitPaths = packageText.contains("MLXVLM_MLX_SWIFT_PATH") &&
            packageText.contains("MLXVLM_MLX_SWIFT_LM_PATH")
        let supportsExplicitTokenizerPath = packageText.contains("MLXVLM_SWIFT_TOKENIZERS_MLX_PATH")

        var nextSteps: [String] = []
        if !declaresMLXSwift || !declaresMLXSwiftLM {
            nextSteps.append("Add mlx-swift and mlx-swift-lm dependencies once network fetches or local vendored checkouts are available.")
        }
        if !declaresSwiftTokenizersMLX {
            nextSteps.append("Add swift-tokenizers-mlx so MLXVLM can load local tokenizer.json files through TokenizersLoader.")
        }
        if supportsBackendToggle {
            nextSteps.append("Set MLXVLM_ENABLE_MLX_BACKEND=1 when resolving/building after MLX Swift dependencies are available.")
            nextSteps.append("Set MLXVLM_ENABLE_REAL_MLX_API=1 after real MLX Swift dependencies are present to compile the guarded MLXArray weight loader.")
        }
        if supportsTokenizerToggle {
            nextSteps.append("Set MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1 when a local swift-tokenizers-mlx checkout is available.")
        }
        if supportsLocalDependencies || supportsExplicitPaths {
            nextSteps.append("For network-restricted builds, set MLXVLM_USE_LOCAL_MLX=1 or explicit MLXVLM_MLX_SWIFT_PATH/MLXVLM_MLX_SWIFT_LM_PATH/MLXVLM_SWIFT_TOKENIZERS_MLX_PATH values.")
        }
        if requirements.contains(where: { !$0.localCheckoutPresent }) {
            nextSteps.append("Place local checkouts under vendor/, Vendor/, Dependencies/, or .build/checkouts/ before enabling a dependency-free local backend target.")
        }
        if !declaresBackendTarget {
            nextSteps.append("Add a separate MLXVLMMLXBackend target so MLXVLMCore remains dependency-free for compatibility inspection builds.")
        }
        nextSteps.append("Use upstream mlx-swift-lm ModelContainer/VLMModelFactory as the primary Swift engine; reserve Swift-owned model modules for Python mlx-vlm compatibility gaps.")

        return BackendDependencyPlan(
            packagePath: packageURL.path,
            packageDeclaresMLXSwift: declaresMLXSwift,
            packageDeclaresMLXSwiftLM: declaresMLXSwiftLM,
            packageDeclaresSwiftTokenizersMLX: declaresSwiftTokenizersMLX,
            packageDeclaresBackendTarget: declaresBackendTarget,
            manifestSupportsMLXBackendToggle: supportsBackendToggle,
            manifestSupportsTokenizerIntegrationToggle: supportsTokenizerToggle,
            manifestSupportsLocalMLXDependencies: supportsLocalDependencies,
            manifestSupportsExplicitMLXPaths: supportsExplicitPaths,
            manifestSupportsExplicitTokenizerPath: supportsExplicitTokenizerPath,
            compatibilityShellBuildable: packageText.contains("MLXVLMCore") &&
                packageText.contains("MLXVLMCli"),
            requirements: requirements,
            nextSteps: nextSteps
        )
    }

    private func requirement(
        packageName: String,
        productHint: String,
        url: String,
        versionRequirement: String,
        rootURL: URL
    ) -> BackendDependencyRequirement {
        let candidates = candidateURLs(packageName: packageName, rootURL: rootURL)
        let present = candidates.contains { fileManager.fileExists(atPath: $0.path) }
        return BackendDependencyRequirement(
            packageName: packageName,
            productHint: productHint,
            url: url,
            versionRequirement: versionRequirement,
            localCandidatePaths: candidates.map(\.path),
            localCheckoutPresent: present
        )
    }

    private func candidateURLs(packageName: String, rootURL: URL) -> [URL] {
        let candidates = [
            rootURL.appendingPathComponent("vendor").appendingPathComponent(aliasDirectoryName(for: packageName)),
            rootURL.appendingPathComponent("Vendor").appendingPathComponent(aliasDirectoryName(for: packageName)),
            rootURL.appendingPathComponent("Dependencies").appendingPathComponent(aliasDirectoryName(for: packageName)),
            rootURL.deletingLastPathComponent().appendingPathComponent(aliasDirectoryName(for: packageName)),
            rootURL.appendingPathComponent("vendor").appendingPathComponent(packageName),
            rootURL.appendingPathComponent("Vendor").appendingPathComponent(packageName),
            rootURL.appendingPathComponent("Dependencies").appendingPathComponent(packageName),
            rootURL.appendingPathComponent(".build").appendingPathComponent("checkouts").appendingPathComponent(packageName),
            rootURL.deletingLastPathComponent().appendingPathComponent(packageName),
        ]
        return candidates.filter { !isSameDirectory($0, rootURL) }
    }

    private func aliasDirectoryName(for packageName: String) -> String {
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

    private func isSameDirectory(_ lhs: URL, _ rhs: URL) -> Bool {
        lhs.standardizedFileURL.path.caseInsensitiveCompare(rhs.standardizedFileURL.path) == .orderedSame
    }
}
