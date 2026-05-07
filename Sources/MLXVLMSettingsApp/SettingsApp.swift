import Foundation
import MLXVLMCore
import MLXVLMMLXBackend
import SwiftUI

@main
struct MLXVLMSettingsApp: App {
    @StateObject private var controller = ServerController()

    var body: some Scene {
        WindowGroup("MLX-VLM Swift") {
            SettingsView(controller: controller)
                .frame(minWidth: 560, minHeight: 420)
        }
        .windowResizability(.contentSize)
    }
}

struct SettingsView: View {
    @ObservedObject var controller: ServerController

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text("MLX-VLM Swift")
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
                Circle()
                    .fill(controller.isRunning ? Color.green : Color.secondary)
                    .frame(width: 10, height: 10)
                Text(controller.isRunning ? "Running" : "Stopped")
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("Model")
                    .font(.headline)
                TextField("Local MLX model path", text: $controller.modelPath)
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Button("Inspect") {
                        controller.inspect()
                    }
                    .disabled(controller.modelPath.isEmpty)

                    Button("Choose...") {
                        controller.chooseModelPath()
                    }
                }
            }

            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Server")
                        .font(.headline)
                    Stepper(value: $controller.port, in: 1024...65535) {
                        Text("Port \(controller.port)")
                    }
                    HStack {
                        Button(controller.isRunning ? "Stop" : "Start") {
                            controller.toggleServer()
                        }
                        .keyboardShortcut(.defaultAction)
                        .disabled(controller.modelPath.isEmpty)

                        Text("http://127.0.0.1:\(controller.port)")
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("Generation Defaults")
                    .font(.headline)
                HStack {
                    Stepper(value: $controller.maxTokens, in: 1...32768, step: 32) {
                        Text("Max tokens \(controller.maxTokens)")
                    }
                    Stepper(value: $controller.contextLength, in: 0...262144, step: 512) {
                        Text(controller.contextLength == 0 ? "Context auto" : "Context \(controller.contextLength)")
                    }
                    Stepper(value: $controller.topK, in: 0...200) {
                        Text("Top-k \(controller.topK)")
                    }
                }
                HStack {
                    Stepper(value: $controller.seed, in: 0...Int(Int32.max)) {
                        Text("Seed \(controller.seed)")
                    }
                    Toggle("Quantize activations", isOn: $controller.quantizeActivations)
                    Text("Keep-alive")
                        .frame(width: 80, alignment: .leading)
                    TextField("default", text: $controller.keepAlive)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 120)
                }
                HStack {
                    Text("Temperature")
                        .frame(width: 96, alignment: .leading)
                    Slider(value: $controller.temperature, in: 0...2, step: 0.05)
                    Text(controller.temperature.formatted(.number.precision(.fractionLength(2))))
                        .frame(width: 44, alignment: .trailing)
                    Text("Top-p")
                        .frame(width: 44, alignment: .leading)
                    Slider(value: $controller.topP, in: 0...1, step: 0.01)
                    Text(controller.topP.formatted(.number.precision(.fractionLength(2))))
                        .frame(width: 44, alignment: .trailing)
                }
                HStack {
                    Stepper(value: $controller.kvBitsTenths, in: 0...160) {
                        Text(controller.kvBitsTenths == 0 ? "KV quant off" : "KV bits \(Double(controller.kvBitsTenths) / 10.0, specifier: "%.1f")")
                    }
                    Text("KV scheme")
                        .frame(width: 72, alignment: .leading)
                    TextField("uniform", text: $controller.kvQuantScheme)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 110)
                    Stepper(value: $controller.kvGroupSize, in: 0...4096, step: 32) {
                        Text(controller.kvGroupSize == 0 ? "KV group auto" : "KV group \(controller.kvGroupSize)")
                    }
                }
                HStack {
                    Stepper(value: $controller.maxKVSize, in: 0...1_048_576, step: 1024) {
                        Text(controller.maxKVSize == 0 ? "Max KV auto" : "Max KV \(controller.maxKVSize)")
                    }
                    Stepper(value: $controller.visionCacheSize, in: 0...512) {
                        Text(controller.visionCacheSize == 0 ? "Vision cache off" : "Vision cache \(controller.visionCacheSize)")
                    }
                }
            }

            Divider()

            if let descriptor = controller.descriptor {
                VStack(alignment: .leading, spacing: 6) {
                    Text(descriptor.id)
                        .font(.headline)
                    Text("type: \(descriptor.canonicalModelType)")
                    Text("weights: \(descriptor.weightFiles.count) files, \(descriptor.totalWeightBytes) bytes")
                    Text("safetensors: \(controller.readableSafetensorsCount)/\(descriptor.safetensorsMetadata.count) readable")
                    Text("chat template: \(descriptor.hasChatTemplate ? "yes" : "no")")
                    Text("quantization: \(descriptor.quantizationMetadata?.summary ?? "none")")
                    if let compatibility = controller.compatibility {
                        Text("metadata ready: \(compatibility.metadataReady ? "yes" : "no")")
                        Text("generation ready: \(compatibility.generationReady ? "yes" : "no")")
                        Text("backend: \(compatibility.backend.activeBackend)")
                    }
                    if let availability = controller.backendAvailability {
                        Text("backend available: \(availability.canCreateBackend ? "yes" : "no")")
                        Text("real mlx api: \(availability.runtimeProbe.realMLXAPIImplementationCompiled ? "compiled" : "not compiled")")
                    }
                    if let metal = controller.metalLibraryReport {
                        Text("metal library: \(metal.ready ? "ready" : "missing")")
                        if let source = metal.sourcePath {
                            Text("metal source: \(source)")
                        } else if let instruction = metal.instructions.first {
                            Text("metal action: \(instruction)")
                                .foregroundStyle(.secondary)
                        }
                    }
                    if let pipeline = controller.pipelineReport {
                        Text("mlx pipeline: \(pipeline.pipelineReady ? "ready" : "blocked")")
                        Text("mlx api compiled: \(pipeline.load.availability.runtimeProbe.realMLXAPIImplementationCompiled ? "yes" : "no")")
                        Text("mlx deps: \(pipeline.load.availability.dependencyPlan.canEnableMLXBackend ? "found" : "missing")")
                        Text("modules: \(pipeline.moduleConstruction.moduleConstructionReady ? "ready" : "blocked")")
                        Text("forward: \(pipeline.forward.forwardReady ? "ready" : "blocked")")
                        Text("decode: \(pipeline.generationLoop.generationLoopReady ? "ready" : "blocked")")
                        if let blocker = pipeline.blockingReasons.first {
                            Text("blocker: \(blocker)")
                                .foregroundStyle(.secondary)
                        }
                    } else if let pipelineError = controller.pipelineError {
                        Text("mlx pipeline: \(pipelineError)")
                            .foregroundStyle(.red)
                    }
                }
                .font(.system(.body, design: .monospaced))
            } else {
                Text("Inspect a local MLX model directory or cached Hugging Face model ID before starting the server.")
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if !controller.statusMessage.isEmpty {
                Text(controller.statusMessage)
                    .font(.footnote)
                    .foregroundStyle(controller.statusIsError ? Color.red : Color.secondary)
            }
        }
        .padding(24)
    }
}

@MainActor
final class ServerController: ObservableObject {
    @Published var modelPath: String {
        didSet {
            UserDefaults.standard.set(modelPath, forKey: Self.modelPathKey)
        }
    }
    @Published var port: Int {
        didSet {
            UserDefaults.standard.set(port, forKey: Self.portKey)
        }
    }
    @Published var maxTokens: Int {
        didSet {
            UserDefaults.standard.set(maxTokens, forKey: Self.maxTokensKey)
        }
    }
    @Published var temperature: Double {
        didSet {
            UserDefaults.standard.set(temperature, forKey: Self.temperatureKey)
        }
    }
    @Published var topP: Double {
        didSet {
            UserDefaults.standard.set(topP, forKey: Self.topPKey)
        }
    }
    @Published var topK: Int {
        didSet {
            UserDefaults.standard.set(topK, forKey: Self.topKKey)
        }
    }
    @Published var seed: Int {
        didSet {
            UserDefaults.standard.set(seed, forKey: Self.seedKey)
        }
    }
    @Published var contextLength: Int {
        didSet {
            UserDefaults.standard.set(contextLength, forKey: Self.contextLengthKey)
        }
    }
    @Published var keepAlive: String {
        didSet {
            UserDefaults.standard.set(keepAlive, forKey: Self.keepAliveKey)
        }
    }
    @Published var kvBitsTenths: Int {
        didSet {
            UserDefaults.standard.set(kvBitsTenths, forKey: Self.kvBitsTenthsKey)
        }
    }
    @Published var kvQuantScheme: String {
        didSet {
            UserDefaults.standard.set(kvQuantScheme, forKey: Self.kvQuantSchemeKey)
        }
    }
    @Published var kvGroupSize: Int {
        didSet {
            UserDefaults.standard.set(kvGroupSize, forKey: Self.kvGroupSizeKey)
        }
    }
    @Published var maxKVSize: Int {
        didSet {
            UserDefaults.standard.set(maxKVSize, forKey: Self.maxKVSizeKey)
        }
    }
    @Published var visionCacheSize: Int {
        didSet {
            UserDefaults.standard.set(visionCacheSize, forKey: Self.visionCacheSizeKey)
        }
    }
    @Published var quantizeActivations: Bool {
        didSet {
            UserDefaults.standard.set(quantizeActivations, forKey: Self.quantizeActivationsKey)
        }
    }
    @Published var descriptor: ModelDescriptor?
    @Published var compatibility: ModelCompatibilityReport?
    @Published var backendAvailability: MLXBackendAvailability?
    @Published var metalLibraryReport: MLXMetalLibraryReport?
    @Published var pipelineReport: MLXGenerationPipelineReport?
    @Published var pipelineError: String?
    @Published var statusMessage: String = ""
    @Published var statusIsError: Bool = false
    @Published private(set) var isRunning: Bool = false

    var readableSafetensorsCount: Int {
        descriptor?.safetensorsMetadata.filter(\.isReadable).count ?? 0
    }

    private var process: Process?
    private static let modelPathKey = "MLXVLMSettings.modelPath"
    private static let portKey = "MLXVLMSettings.port"
    private static let maxTokensKey = "MLXVLMSettings.maxTokens"
    private static let temperatureKey = "MLXVLMSettings.temperature"
    private static let topPKey = "MLXVLMSettings.topP"
    private static let topKKey = "MLXVLMSettings.topK"
    private static let seedKey = "MLXVLMSettings.seed"
    private static let contextLengthKey = "MLXVLMSettings.contextLength"
    private static let keepAliveKey = "MLXVLMSettings.keepAlive"
    private static let kvBitsTenthsKey = "MLXVLMSettings.kvBitsTenths"
    private static let kvQuantSchemeKey = "MLXVLMSettings.kvQuantScheme"
    private static let kvGroupSizeKey = "MLXVLMSettings.kvGroupSize"
    private static let maxKVSizeKey = "MLXVLMSettings.maxKVSize"
    private static let visionCacheSizeKey = "MLXVLMSettings.visionCacheSize"
    private static let quantizeActivationsKey = "MLXVLMSettings.quantizeActivations"

    init() {
        modelPath = UserDefaults.standard.string(forKey: Self.modelPathKey) ?? ""
        let storedPort = UserDefaults.standard.integer(forKey: Self.portKey)
        port = storedPort == 0 ? 11434 : storedPort
        let storedMaxTokens = UserDefaults.standard.integer(forKey: Self.maxTokensKey)
        maxTokens = storedMaxTokens == 0 ? 512 : storedMaxTokens
        let storedTemperature = UserDefaults.standard.object(forKey: Self.temperatureKey) as? Double
        temperature = storedTemperature ?? 0.0
        let storedTopP = UserDefaults.standard.object(forKey: Self.topPKey) as? Double
        topP = storedTopP ?? 1.0
        topK = UserDefaults.standard.integer(forKey: Self.topKKey)
        seed = UserDefaults.standard.integer(forKey: Self.seedKey)
        contextLength = UserDefaults.standard.integer(forKey: Self.contextLengthKey)
        keepAlive = UserDefaults.standard.string(forKey: Self.keepAliveKey) ?? ""
        kvBitsTenths = UserDefaults.standard.integer(forKey: Self.kvBitsTenthsKey)
        kvQuantScheme = UserDefaults.standard.string(forKey: Self.kvQuantSchemeKey) ?? ""
        kvGroupSize = UserDefaults.standard.integer(forKey: Self.kvGroupSizeKey)
        maxKVSize = UserDefaults.standard.integer(forKey: Self.maxKVSizeKey)
        visionCacheSize = UserDefaults.standard.integer(forKey: Self.visionCacheSizeKey)
        quantizeActivations = UserDefaults.standard.bool(forKey: Self.quantizeActivationsKey)
    }

    func inspect() {
        do {
            descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: modelPath)
            if let descriptor {
                let availability = MLXBackendFactory.availability()
                backendAvailability = availability
                let backendStatus: BackendStatus = availability.canCreateBackend ? .mlxSwiftVLM : .compatibilityShell
                compatibility = ModelCompatibilityValidator.validate(descriptor: descriptor, backend: backendStatus)
                metalLibraryReport = MLXMetalLibrarySupport.ensureDefaultLibraryAvailable(install: false)
                do {
                    pipelineReport = try MLXGenerationPipelineReporter().report(
                        descriptor: descriptor,
                        request: GenerationRequest(
                            model: descriptor.id,
                            messages: [ChatMessage(role: .user, content: [.text("hello")])]
                        ),
                        weightOptions: MLXWeightPreparationOptions(skipTensorPayloads: true)
                    )
                    pipelineError = nil
                } catch {
                    pipelineReport = nil
                    pipelineError = String(describing: error)
                }
            }
            status("Loaded \(descriptor?.id ?? "model")")
        } catch {
            descriptor = nil
            compatibility = nil
            backendAvailability = nil
            metalLibraryReport = nil
            pipelineReport = nil
            pipelineError = error.localizedDescription
            status(error.localizedDescription, isError: true)
        }
    }

    func chooseModelPath() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            modelPath = url.path
            inspect()
        }
    }

    func toggleServer() {
        isRunning ? stopServer() : startServer()
    }

    private func startServer() {
        inspect()
        guard descriptor != nil else {
            return
        }

        do {
            let process = Process()
            process.executableURL = try cliExecutableURL()
            var arguments = [
                "serve",
                "--model", modelPath,
                "--port", String(port),
                "--max-tokens", String(maxTokens),
                "--temperature", String(temperature),
                "--top-p", String(topP),
                "--top-k", String(topK),
                "--seed", String(seed),
            ]
            if contextLength > 0 {
                arguments.append(contentsOf: ["--context-length", String(contextLength)])
            }
            if kvBitsTenths > 0 {
                arguments.append(contentsOf: ["--kv-bits", String(Double(kvBitsTenths) / 10.0)])
            }
            if !kvQuantScheme.isEmpty {
                arguments.append(contentsOf: ["--kv-quant-scheme", kvQuantScheme])
            }
            if kvGroupSize > 0 {
                arguments.append(contentsOf: ["--kv-group-size", String(kvGroupSize)])
            }
            if maxKVSize > 0 {
                arguments.append(contentsOf: ["--max-kv-size", String(maxKVSize)])
            }
            if visionCacheSize > 0 {
                arguments.append(contentsOf: ["--vision-cache-size", String(visionCacheSize)])
            }
            if quantizeActivations {
                arguments.append(contentsOf: ["--quantize-activations", "true"])
            }
            if !keepAlive.isEmpty {
                arguments.append(contentsOf: ["--keep-alive", keepAlive])
            }
            process.arguments = arguments
            try process.run()
            self.process = process
            isRunning = true
            status("Server started on port \(port)")
        } catch {
            status(error.localizedDescription, isError: true)
        }
    }

    private func stopServer() {
        process?.terminate()
        process = nil
        isRunning = false
        status("Server stopped")
    }

    private func cliExecutableURL() throws -> URL {
        let directory = Bundle.main.executableURL?.deletingLastPathComponent()
        if let url = directory?.appendingPathComponent("mlx-vlm-swift"),
           FileManager.default.isExecutableFile(atPath: url.path)
        {
            return url
        }
        throw SettingsError.cliNotFound
    }

    private func status(_ message: String, isError: Bool = false) {
        statusMessage = message
        statusIsError = isError
    }
}

enum SettingsError: LocalizedError {
    case cliNotFound

    var errorDescription: String? {
        switch self {
        case .cliNotFound:
            return "mlx-vlm-swift executable was not found next to the settings app."
        }
    }
}
