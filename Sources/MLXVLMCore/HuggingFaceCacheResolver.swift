import Foundation

public struct HuggingFaceCacheResolver {
    public let fileManager: FileManager
    public let environment: [String: String]

    public init(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.fileManager = fileManager
        self.environment = environment
    }

    public func resolveModelDirectory(identifier: String) -> URL? {
        let cacheDirectoryName = "models--" + identifier.replacingOccurrences(of: "/", with: "--")
        for cacheRoot in cacheRoots() {
            let modelCacheURL = cacheRoot.appendingPathComponent(cacheDirectoryName, isDirectory: true)
            guard fileManager.fileExists(atPath: modelCacheURL.path) else {
                continue
            }
            if let snapshot = resolveSnapshot(in: modelCacheURL),
               fileManager.fileExists(atPath: snapshot.appendingPathComponent("config.json").path) {
                return snapshot.standardizedFileURL
            }
            if fileManager.fileExists(atPath: modelCacheURL.appendingPathComponent("config.json").path) {
                return modelCacheURL.standardizedFileURL
            }
        }
        return nil
    }

    public func cacheRoots() -> [URL] {
        var roots: [URL] = []
        if let hubCache = environment["HUGGINGFACE_HUB_CACHE"], !hubCache.isEmpty {
            roots.append(URL(fileURLWithPath: hubCache, isDirectory: true))
        }
        if let hfHome = environment["HF_HOME"], !hfHome.isEmpty {
            roots.append(URL(fileURLWithPath: hfHome, isDirectory: true).appendingPathComponent("hub", isDirectory: true))
        }
        if let transformersCache = environment["TRANSFORMERS_CACHE"], !transformersCache.isEmpty {
            roots.append(URL(fileURLWithPath: transformersCache, isDirectory: true))
        }
        roots.append(
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache", isDirectory: true)
                .appendingPathComponent("huggingface", isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
        )

        var seen: Set<String> = []
        return roots.map(\.standardizedFileURL).filter { seen.insert($0.path).inserted }
    }

    private func resolveSnapshot(in modelCacheURL: URL) -> URL? {
        if let refSnapshot = resolveMainRef(in: modelCacheURL) {
            return refSnapshot
        }

        let snapshotsURL = modelCacheURL.appendingPathComponent("snapshots", isDirectory: true)
        guard let snapshots = try? fileManager.contentsOfDirectory(
            at: snapshotsURL,
            includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        return snapshots
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { lhs, rhs in
                let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return lhsDate > rhsDate
            }
            .first
    }

    private func resolveMainRef(in modelCacheURL: URL) -> URL? {
        let mainRefURL = modelCacheURL
            .appendingPathComponent("refs", isDirectory: true)
            .appendingPathComponent("main")
        guard let refData = try? Data(contentsOf: mainRefURL),
              let snapshotName = String(data: refData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              !snapshotName.isEmpty
        else {
            return nil
        }
        let snapshotURL = modelCacheURL
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(snapshotName, isDirectory: true)
        return fileManager.fileExists(atPath: snapshotURL.path) ? snapshotURL : nil
    }
}
