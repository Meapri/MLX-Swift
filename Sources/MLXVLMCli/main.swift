import Foundation
import MLXVLMCore
import MLXVLMMLXBackend
import Darwin

@main
struct MLXVLMCli {
    static func main() {
        do {
            try run()
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func run() throws {
        var arguments = Array(CommandLine.arguments.dropFirst())
        guard let command = arguments.first else {
            printUsage()
            return
        }
        arguments.removeFirst()

        switch command {
        case "inspect":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(descriptor)
        case "inspect-qwen-vl":
            let path = try value(for: "--model", in: arguments)
            let config = try QwenVLModelConfig.load(fromModelDirectory: path)
            printJSON(config)
        case "expand-qwen-vl-placeholders":
            let text = try value(for: "--text", in: arguments)
            let imageGrids = try values(for: "--image-grid", in: arguments).map(QwenVLGridTHW.init(csv:))
            let videoGrids = try values(for: "--video-grid", in: arguments).map(QwenVLGridTHW.init(csv:))
            let imageMergeSize = Int(optionalValue(for: "--image-merge-size", in: arguments) ?? "2") ?? 2
            let videoMergeSize = Int(optionalValue(for: "--video-merge-size", in: arguments) ?? "2") ?? 2
            let expander = QwenVLPlaceholderExpander(
                imageMergeSize: imageMergeSize,
                videoMergeSize: videoMergeSize
            )
            print(try expander.expand(texts: [text], imageGrids: imageGrids, videoGrids: videoGrids)[0])
        case "inspect-qwen-vl-weights":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            guard let report = QwenVLWeightSanitizer.report(descriptor: descriptor) else {
                throw CLIError.missingWeightIndex(path)
            }
            printJSON(report)
        case "sanitize-qwen-vl-key":
            let key = try value(for: "--key", in: arguments)
            printJSON(QwenVLWeightSanitizer.sanitize(key))
        case "inspect-qwen-vl-architecture":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let config = try QwenVLModelConfig.load(fromModelDirectory: descriptor.path)
            let weightCatalog = WeightCatalogBuilder().catalog(for: descriptor)
            printJSON(QwenVLArchitecturePlanner().plan(config: config, weightCatalog: weightCatalog))
        case "inspect-weight-catalog":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(WeightCatalogBuilder().catalog(for: descriptor))
        case "inspect-weight-data":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(WeightDataCatalogBuilder().catalog(for: descriptor))
        case "plan-mlx-weight-load":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(MLXWeightLoadPlan(descriptor: descriptor))
        case "prepare-mlx-weights":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let maxTensorCount = optionalValue(for: "--max-tensors", in: arguments).flatMap(Int.init)
            let maxTotalBytes = optionalValue(for: "--max-total-bytes", in: arguments).flatMap(Int64.init)
            let bundle = try MLXWeightPreparer().prepare(
                descriptor: descriptor,
                options: MLXWeightPreparationOptions(
                    tensorNames: values(for: "--tensor", in: arguments),
                    maxTensorCount: maxTensorCount,
                    maxTotalBytes: maxTotalBytes,
                    skipTensorPayloads: hasFlag("--skip-weight-payloads", in: arguments)
                )
            )
            printJSON(bundle.summary)
        case "preview-weight-tensor":
            let path = try value(for: "--model", in: arguments)
            let tensor = try value(for: "--tensor", in: arguments)
            let maxBytes = Int(optionalValue(for: "--bytes", in: arguments) ?? "32") ?? 32
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(
                try WeightDataCatalogBuilder().previewTensor(
                    named: tensor,
                    descriptor: descriptor,
                    maxBytes: maxBytes
                )
            )
        case "read-weight-tensor-payload":
            let path = try value(for: "--model", in: arguments)
            let tensor = try value(for: "--tensor", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(
                WeightTensorPayloadSummary(
                    payload: try WeightDataCatalogBuilder().readTensorPayload(
                        named: tensor,
                        descriptor: descriptor
                    )
                )
            )
        case "inspect-safetensors":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(descriptor.safetensorsMetadata)
        case "inspect-tokenizer-catalog":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            guard let catalog = TokenizerCatalogBuilder().catalog(for: descriptor) else {
                throw CLIError.missingTokenizerJSON(path)
            }
            printJSON(catalog)
        case "inspect-tokenizer-plan":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let catalog = TokenizerCatalogBuilder().catalog(for: descriptor)
            printJSON(TokenizerImplementationPlanner().plan(descriptor: descriptor, catalog: catalog))
        case "inspect-chat-template-plan":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(ChatTemplatePlanner().plan(descriptor: descriptor))
        case "inspect-capabilities":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(ModelCapabilityPlanner().plan(descriptor: descriptor))
        case "inspect-adapter":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(descriptor.adapterMetadata)
        case "inspect-processor":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(descriptor.processorMetadata)
        case "inspect-config-normalization":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(descriptor.configNormalization)
        case "inspect-normalized-config":
            let path = try value(for: "--model", in: arguments)
            printJSON(try ModelStore().loadNormalizedConfig(pathOrIdentifier: path))
        case "preflight-tokenize":
            let path = try value(for: "--model", in: arguments)
            let text = try value(for: "--text", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            guard let catalog = TokenizerCatalogBuilder().catalog(for: descriptor) else {
                throw CLIError.missingTokenizerJSON(path)
            }
            printJSON(TokenizationPreflightPlanner(catalog: catalog).plan(prompt: text))
        case "tokenize-simple":
            let path = try value(for: "--model", in: arguments)
            let text = try value(for: "--text", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            guard let catalog = TokenizerCatalogBuilder().catalog(for: descriptor) else {
                throw CLIError.missingTokenizerJSON(path)
            }
            let plan = TokenizerImplementationPlanner().plan(descriptor: descriptor, catalog: catalog)
            printJSON(SimpleTokenizer(catalog: catalog, plan: plan).tokenize(text))
        case "detokenize-simple":
            let path = try value(for: "--model", in: arguments)
            let tokenIDs = try parseIntCSV(try value(for: "--ids", in: arguments))
            let skipSpecial = optionalValue(for: "--skip-special", in: arguments) == "true"
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            guard let catalog = TokenizerCatalogBuilder().catalog(for: descriptor) else {
                throw CLIError.missingTokenizerJSON(path)
            }
            let plan = TokenizerImplementationPlanner().plan(descriptor: descriptor, catalog: catalog)
            printJSON(
                SimpleTokenizer(catalog: catalog, plan: plan)
                    .detokenize(tokenIDs, skipSpecialTokens: skipSpecial)
            )
        case "decode-token-stream":
            let path = try value(for: "--model", in: arguments)
            let tokenIDs = try parseIntCSV(try value(for: "--ids", in: arguments))
            let skipSpecial = optionalValue(for: "--skip-special", in: arguments) != "false"
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            guard let catalog = TokenizerCatalogBuilder().catalog(for: descriptor) else {
                throw CLIError.missingTokenizerJSON(path)
            }
            let plan = TokenizerImplementationPlanner().plan(descriptor: descriptor, catalog: catalog)
            var decoder = GenerationTokenTextDecoder(
                tokenizer: SimpleTokenizer(catalog: catalog, plan: plan),
                skipSpecialTokens: skipSpecial
            )
            printJSON(tokenIDs.map { decoder.append($0) })
        case "plan-qwen-vl-merge":
            let rows = try values(for: "--input-ids", in: arguments).map(parseIntCSV)
            let imageTokenID = Int(optionalValue(for: "--image-token-id", in: arguments) ?? "151655") ?? 151655
            let videoTokenID = Int(optionalValue(for: "--video-token-id", in: arguments) ?? "151656") ?? 151656
            let featureCount = Int(try value(for: "--feature-count", in: arguments)) ?? 0
            printJSON(
                try QwenVLEmbeddingMergePlanner.plan(
                    inputIDs: rows,
                    imageTokenID: imageTokenID,
                    videoTokenID: videoTokenID,
                    featureCount: featureCount
                )
            )
        case "format-qwen-vl-prompt":
            let api = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            let style = optionalValue(for: "--style", in: arguments) ?? "plain"
            let json = try value(for: "--json", in: arguments)
            let request = try normalizedGenerationRequest(api: api, json: json)
            let builder = QwenVLPromptBuilder()
            switch style {
            case "plain":
                print(builder.plainPrompt(messages: request.messages))
            case "qwen-chat":
                print(builder.qwenChatPrompt(messages: request.messages))
            default:
                throw CLIError.unsupportedPromptStyle(style)
            }
        case "inspect-media":
            let api = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            let json = try value(for: "--json", in: arguments)
            let request = try normalizedGenerationRequest(api: api, json: json)
            printJSON(MediaReferenceResolver().report(for: request))
        case "plan-qwen-vl-image-grid":
            let height = Int(try value(for: "--height", in: arguments)) ?? 0
            let width = Int(try value(for: "--width", in: arguments)) ?? 0
            let patchSize = Int(optionalValue(for: "--patch-size", in: arguments) ?? "14") ?? 14
            let mergeSize = Int(optionalValue(for: "--merge-size", in: arguments) ?? "2") ?? 2
            let config = QwenVLImageGridConfig(patchSize: patchSize, mergeSize: mergeSize)
            printJSON(try QwenVLImageGridPlanner.imagePlan(height: height, width: width, config: config))
        case "plan-qwen-vl-images":
            let api = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            let json = try value(for: "--json", in: arguments)
            let request = try normalizedGenerationRequest(api: api, json: json)
            printJSON(QwenVLImageInputPlanner().plan(request: request))
        case "plan-qwen-vl-pixels":
            let api = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            let json = try value(for: "--json", in: arguments)
            let request = try normalizedGenerationRequest(api: api, json: json)
            printJSON(QwenVLImagePixelPreflightPlanner().plan(request: request))
        case "apply-stop-sequences":
            let text = try value(for: "--text", in: arguments)
            let stops = values(for: "--stop", in: arguments)
            printJSON(StopSequenceMatcher(stopSequences: stops).truncate(text))
        case "list-supported":
            for modelType in ModelRegistry.shared.supportedModelTypes.sorted() {
                print(modelType)
            }
        case "backend-status":
            printJSON(BackendStatus.compatibilityShell)
        case "backend-plan":
            let root = optionalValue(for: "--root", in: arguments) ?? FileManager.default.currentDirectoryPath
            printJSON(BackendDependencyPlanner().plan(rootPath: root))
        case "backend-availability":
            let root = optionalValue(for: "--root", in: arguments) ?? FileManager.default.currentDirectoryPath
            printJSON(MLXBackendFactory.availability(rootPath: root))
        case "inspect-mlx-metal-library":
            printJSON(MLXMetalLibrarySupport.ensureDefaultLibraryAvailable(install: false))
        case "preflight-mlx-backend-load":
            let path = try value(for: "--model", in: arguments)
            let root = optionalValue(for: "--root", in: arguments) ?? FileManager.default.currentDirectoryPath
            let maxTensorCount = optionalValue(for: "--max-tensors", in: arguments).flatMap(Int.init) ?? 16
            let maxTotalBytes = optionalValue(for: "--max-total-bytes", in: arguments).flatMap(Int64.init) ?? 67_108_864
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(
                MLXBackendFactory.preflightLoad(
                    descriptor: descriptor,
                    rootPath: root,
                    weightOptions: MLXWeightPreparationOptions(
                        tensorNames: values(for: "--tensor", in: arguments),
                        maxTensorCount: maxTensorCount,
                        maxTotalBytes: maxTotalBytes,
                        skipTensorPayloads: hasFlag("--skip-weight-payloads", in: arguments)
                    )
                )
            )
        case "inspect-mlx-container":
            let path = try value(for: "--model", in: arguments)
            let root = optionalValue(for: "--root", in: arguments) ?? FileManager.default.currentDirectoryPath
            let maxTensorCount = optionalValue(for: "--max-tensors", in: arguments).flatMap(Int.init) ?? 16
            let maxTotalBytes = optionalValue(for: "--max-total-bytes", in: arguments).flatMap(Int64.init) ?? 67_108_864
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(
                MLXBackendFactory.loadWeightBackedContainer(
                    descriptor: descriptor,
                    rootPath: root,
                    weightOptions: MLXWeightPreparationOptions(
                        tensorNames: values(for: "--tensor", in: arguments),
                        maxTensorCount: maxTensorCount,
                        maxTotalBytes: maxTotalBytes,
                        skipTensorPayloads: hasFlag("--skip-weight-payloads", in: arguments)
                    )
                ).summary
            )
        case "inspect-mlx-module-plan":
            let path = try value(for: "--model", in: arguments)
            let root = optionalValue(for: "--root", in: arguments) ?? FileManager.default.currentDirectoryPath
            let maxTensorCount = optionalValue(for: "--max-tensors", in: arguments).flatMap(Int.init) ?? 16
            let maxTotalBytes = optionalValue(for: "--max-total-bytes", in: arguments).flatMap(Int64.init) ?? 67_108_864
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let container = MLXBackendFactory.loadWeightBackedContainer(
                descriptor: descriptor,
                rootPath: root,
                weightOptions: MLXWeightPreparationOptions(
                    tensorNames: values(for: "--tensor", in: arguments),
                    maxTensorCount: maxTensorCount,
                    maxTotalBytes: maxTotalBytes,
                    skipTensorPayloads: hasFlag("--skip-weight-payloads", in: arguments)
                )
            )
            printJSON(MLXQwenVLModuleConstructionPlanner().plan(container: container))
        case "inspect-mlx-forward-plan":
            let path = try value(for: "--model", in: arguments)
            let root = optionalValue(for: "--root", in: arguments) ?? FileManager.default.currentDirectoryPath
            let api = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            let json = try value(for: "--json", in: arguments)
            let maxTensorCount = optionalValue(for: "--max-tensors", in: arguments).flatMap(Int.init) ?? 16
            let maxTotalBytes = optionalValue(for: "--max-total-bytes", in: arguments).flatMap(Int64.init) ?? 67_108_864
            let defaultParameters = generationDefaults(from: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let request = try normalizedGenerationRequest(
                api: api,
                json: json,
                defaultParameters: defaultParameters
            )
            let container = MLXBackendFactory.loadWeightBackedContainer(
                descriptor: descriptor,
                rootPath: root,
                weightOptions: MLXWeightPreparationOptions(
                    tensorNames: values(for: "--tensor", in: arguments),
                    maxTensorCount: maxTensorCount,
                    maxTotalBytes: maxTotalBytes,
                    skipTensorPayloads: hasFlag("--skip-weight-payloads", in: arguments)
                )
            )
            let modulePlan = MLXQwenVLModuleConstructionPlanner().plan(container: container)
            let input = try CompatibilityProcessor().process(
                request: request,
                context: container.context
            )
            printJSON(
                MLXQwenVLForwardPlanner().plan(
                    container: container,
                    modulePlan: modulePlan,
                    input: input
                )
            )
        case "inspect-mlx-generation-loop":
            let path = try value(for: "--model", in: arguments)
            let root = optionalValue(for: "--root", in: arguments) ?? FileManager.default.currentDirectoryPath
            let api = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            let json = try value(for: "--json", in: arguments)
            let maxTensorCount = optionalValue(for: "--max-tensors", in: arguments).flatMap(Int.init) ?? 16
            let maxTotalBytes = optionalValue(for: "--max-total-bytes", in: arguments).flatMap(Int64.init) ?? 67_108_864
            let defaultParameters = generationDefaults(from: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let request = try normalizedGenerationRequest(
                api: api,
                json: json,
                defaultParameters: defaultParameters
            )
            let container = MLXBackendFactory.loadWeightBackedContainer(
                descriptor: descriptor,
                rootPath: root,
                weightOptions: MLXWeightPreparationOptions(
                    tensorNames: values(for: "--tensor", in: arguments),
                    maxTensorCount: maxTensorCount,
                    maxTotalBytes: maxTotalBytes,
                    skipTensorPayloads: hasFlag("--skip-weight-payloads", in: arguments)
                )
            )
            let modulePlan = MLXQwenVLModuleConstructionPlanner().plan(container: container)
            let input = try CompatibilityProcessor().process(
                request: request,
                context: container.context
            )
            let forwardPlan = MLXQwenVLForwardPlanner().plan(
                container: container,
                modulePlan: modulePlan,
                input: input
            )
            printJSON(
                MLXQwenVLGenerationLoopPlanner().plan(
                    forwardPlan: forwardPlan,
                    input: input
                )
            )
        case "inspect-mlx-decode-state":
            let path = try value(for: "--model", in: arguments)
            let root = optionalValue(for: "--root", in: arguments) ?? FileManager.default.currentDirectoryPath
            let api = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            let json = try value(for: "--json", in: arguments)
            let maxTensorCount = optionalValue(for: "--max-tensors", in: arguments).flatMap(Int.init) ?? 16
            let maxTotalBytes = optionalValue(for: "--max-total-bytes", in: arguments).flatMap(Int64.init) ?? 67_108_864
            let defaultParameters = generationDefaults(from: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let request = try normalizedGenerationRequest(
                api: api,
                json: json,
                defaultParameters: defaultParameters
            )
            let container = MLXBackendFactory.loadWeightBackedContainer(
                descriptor: descriptor,
                rootPath: root,
                weightOptions: MLXWeightPreparationOptions(
                    tensorNames: values(for: "--tensor", in: arguments),
                    maxTensorCount: maxTensorCount,
                    maxTotalBytes: maxTotalBytes,
                    skipTensorPayloads: hasFlag("--skip-weight-payloads", in: arguments)
                )
            )
            let modulePlan = MLXQwenVLModuleConstructionPlanner().plan(container: container)
            let input = try CompatibilityProcessor().process(
                request: request,
                context: container.context
            )
            let forwardPlan = MLXQwenVLForwardPlanner().plan(
                container: container,
                modulePlan: modulePlan,
                input: input
            )
            let generationLoopPlan = MLXQwenVLGenerationLoopPlanner().plan(
                forwardPlan: forwardPlan,
                input: input
            )
            printJSON(
                MLXQwenVLDecodeStatePlanner().plan(
                    container: container,
                    forwardPlan: forwardPlan,
                    generationLoopPlan: generationLoopPlan,
                    input: input
                )
            )
        case "inspect-mlx-generate-parameters":
            let path = try value(for: "--model", in: arguments)
            let api = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            let json = try value(for: "--json", in: arguments)
            let defaultParameters = generationDefaults(from: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let request = try normalizedGenerationRequest(
                api: api,
                json: json,
                defaultParameters: defaultParameters
            )
            let context = try CompatibilityVLMBackend(descriptor: descriptor).loadContext()
            let input = try CompatibilityProcessor().process(request: request, context: context)
            printJSON(MLXGenerateParametersPlan(runtime: input.runtime))
        case "inspect-mlx-generate-parameters-bridge":
            let path = try value(for: "--model", in: arguments)
            let api = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            let json = try value(for: "--json", in: arguments)
            let defaultParameters = generationDefaults(from: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let request = try normalizedGenerationRequest(
                api: api,
                json: json,
                defaultParameters: defaultParameters
            )
            let context = try CompatibilityVLMBackend(descriptor: descriptor).loadContext()
            let input = try CompatibilityProcessor().process(request: request, context: context)
            let plan = MLXGenerateParametersPlan(runtime: input.runtime)
            printJSON(MLXGenerateParametersBridge().report(for: plan))
        case "inspect-mlx-model-factory-bridge":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(MLXVLMModelFactoryBridge().report(for: descriptor))
        case "inspect-mlx-pipeline":
            let path = try value(for: "--model", in: arguments)
            let root = optionalValue(for: "--root", in: arguments) ?? FileManager.default.currentDirectoryPath
            let api = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            let json = try value(for: "--json", in: arguments)
            let maxTensorCount = optionalValue(for: "--max-tensors", in: arguments).flatMap(Int.init) ?? 16
            let maxTotalBytes = optionalValue(for: "--max-total-bytes", in: arguments).flatMap(Int64.init) ?? 67_108_864
            let defaultParameters = generationDefaults(from: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let request = try normalizedGenerationRequest(
                api: api,
                json: json,
                defaultParameters: defaultParameters
            )
            printJSON(
                try MLXGenerationPipelineReporter().report(
                    descriptor: descriptor,
                    request: request,
                    rootPath: root,
                    weightOptions: MLXWeightPreparationOptions(
                        tensorNames: values(for: "--tensor", in: arguments),
                        maxTensorCount: maxTensorCount,
                        maxTotalBytes: maxTotalBytes,
                        skipTensorPayloads: hasFlag("--skip-weight-payloads", in: arguments)
                    )
                )
            )
        case "inspect-backend-context":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(try CompatibilityVLMBackend(descriptor: descriptor).loadContext())
        case "inspect-ollama-show":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(OllamaShowResponse(descriptor: descriptor))
        case "preflight-ollama-show":
            let json = optionalValue(for: "--json", in: arguments) ?? "{}"
            let request = try JSONDecoder().decode(
                OllamaShowRequest.self,
                from: Data(json.utf8)
            )
            printJSON(request)
        case "validate-model":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(ModelCompatibilityValidator.validate(descriptor: descriptor))
        case "plan-model-load":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(ModelLoadPlanner().plan(descriptor: descriptor))
        case "plan-mlx-bindings":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            printJSON(MLXBackendBindingPlanner().plan(descriptor: descriptor))
        case "estimate-memory":
            let path = try value(for: "--model", in: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let qwenConfig = try? QwenVLModelConfig.load(fromModelDirectory: descriptor.path)
            let weightDataCatalog = WeightDataCatalogBuilder().catalog(for: descriptor)
            printJSON(
                ModelMemoryEstimator().estimate(
                    descriptor: descriptor,
                    weightDataCatalog: weightDataCatalog,
                    qwenVLConfig: qwenConfig,
                    parameters: generationDefaults(from: arguments)
                )
            )
        case "preflight-generate":
            let path = try value(for: "--model", in: arguments)
            let api = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            let json = try value(for: "--json", in: arguments)
            let defaultParameters = generationDefaults(from: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let request = try normalizedGenerationRequest(
                api: api,
                json: json,
                defaultParameters: defaultParameters
            )
            printJSON(GenerationPreflightPlanner(descriptor: descriptor).plan(request: request))
        case "preflight-embed":
            let path = try value(for: "--model", in: arguments)
            let api = optionalValue(for: "--api", in: arguments) ?? "ollama-embed"
            let json = try value(for: "--json", in: arguments)
            let defaultParameters = generationDefaults(from: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let request = try normalizedEmbeddingRequest(
                api: api,
                json: json,
                defaultParameters: defaultParameters
            )
            printJSON(CompatibilityGenerationEngine(descriptor: descriptor).unavailableEmbeddingReport(for: request))
        case "preflight-model-operation":
            let operation = try value(for: "--operation", in: arguments)
            let json = optionalValue(for: "--json", in: arguments) ?? "{}"
            let request = try JSONDecoder().decode(
                OllamaModelOperationRequest.self,
                from: Data(json.utf8)
            )
            printJSON(OllamaModelOperationReport(operation: operation, request: request))
        case "preflight-ollama-blob":
            let operation = optionalValue(for: "--operation", in: arguments) ?? "push-blob"
            let digest = try value(for: "--digest", in: arguments)
            printJSON(OllamaBlobOperationReport(operation: operation, digest: digest))
        case "preflight-ollama-residency":
            let json = try value(for: "--json", in: arguments)
            let request = try JSONDecoder().decode(
                OllamaGenerateRequest.self,
                from: Data(json.utf8)
            )
            printJSON(request.residencyAction(defaultModel: "default"))
        case "sample-logits":
            let logits = try parseDoubleCSV(try value(for: "--logits", in: arguments))
            let recentTokenIDs = try optionalValue(for: "--recent-token-ids", in: arguments)
                .map(parseIntCSV) ?? []
            let newlineTokenID = optionalValue(for: "--newline-token-id", in: arguments).flatMap(Int.init)
            let samplingPlan = GenerationSamplingPlanner().plan(parameters: generationDefaults(from: arguments))
            printJSON(
                try GenerationLogitsSampler(plan: samplingPlan).sample(
                    logits: logits,
                    recentTokenIDs: recentTokenIDs,
                    newlineTokenID: newlineTokenID
                )
            )
        case "simulate-decode-loop":
            let model = optionalValue(for: "--model", in: arguments) ?? "model"
            let promptTokens = Int(optionalValue(for: "--prompt-tokens", in: arguments) ?? "0") ?? 0
            let maxTokens = Int(optionalValue(for: "--max-tokens", in: arguments) ?? "512") ?? 512
            let eosTokenIDs = try values(for: "--eos", in: arguments).flatMap(parseIntCSV)
            let tokens = try values(for: "--token", in: arguments).map(parseDecodeToken)
            var loop = GenerationDecodeLoop(
                model: model,
                promptTokenCount: promptTokens,
                maxCompletionTokens: maxTokens,
                eosTokenIDs: eosTokenIDs,
                stopSequences: values(for: "--stop", in: arguments)
            )
            printJSON(loop.run(tokens))
        case "simulate-logits-decode":
            let path = try value(for: "--model", in: arguments)
            let promptTokens = Int(optionalValue(for: "--prompt-tokens", in: arguments) ?? "0") ?? 0
            let maxTokens = Int(optionalValue(for: "--max-tokens", in: arguments) ?? "512") ?? 512
            let eosTokenIDs = try values(for: "--eos", in: arguments).flatMap(parseIntCSV)
            let logitsRows = try values(for: "--logits", in: arguments).map(parseDoubleCSV)
            let recentTokenIDs = try optionalValue(for: "--recent-token-ids", in: arguments)
                .map(parseIntCSV) ?? []
            let newlineTokenID = optionalValue(for: "--newline-token-id", in: arguments).flatMap(Int.init)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            guard let catalog = TokenizerCatalogBuilder().catalog(for: descriptor) else {
                throw CLIError.missingTokenizerJSON(path)
            }
            let plan = TokenizerImplementationPlanner().plan(descriptor: descriptor, catalog: catalog)
            let samplingPlan = GenerationSamplingPlanner().plan(parameters: generationDefaults(from: arguments))
            printJSON(
                try GenerationLogitsDecodeExecutor(
                    model: descriptor.id,
                    promptTokenCount: promptTokens,
                    maxCompletionTokens: maxTokens,
                    eosTokenIDs: eosTokenIDs,
                    stopSequences: values(for: "--stop", in: arguments),
                    samplingPlan: samplingPlan,
                    tokenizer: SimpleTokenizer(catalog: catalog, plan: plan),
                    newlineTokenID: newlineTokenID
                ).run(
                    logitsRows: logitsRows,
                    recentTokenIDs: recentTokenIDs
                )
            )
        case "render-generation-response":
            let rawAPI = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            guard let api = GenerationResponseAPI(rawValue: rawAPI) else {
                throw CLIError.unsupportedAPI(rawAPI)
            }
            let model = optionalValue(for: "--model", in: arguments) ?? "model"
            let text = optionalValue(for: "--text", in: arguments) ?? ""
            let promptTokens = Int(optionalValue(for: "--prompt-tokens", in: arguments) ?? "0") ?? 0
            let completionTokens = Int(optionalValue(for: "--completion-tokens", in: arguments) ?? "0") ?? 0
            let finishReason = optionalValue(for: "--finish-reason", in: arguments) ?? "stop"
            let stream = optionalValue(for: "--stream", in: arguments) == "true"
            let result = CompletedGeneration(
                model: model,
                text: text,
                finishReason: finishReason,
                usage: GenerationUsage(promptTokens: promptTokens, completionTokens: completionTokens)
            )
            let chunks = values(for: "--chunk", in: arguments).map {
                GenerationChunk(text: $0, tokenID: nil)
            } + (stream ? [GenerationChunk(text: "", isFinished: true, finishReason: finishReason)] : [])
            printJSON(
                GenerationAPIResponseRenderer.renderCompleted(
                    result,
                    api: api,
                    stream: stream,
                    chunks: chunks
                )
            )
        case "render-generation-chunks":
            let rawAPI = optionalValue(for: "--api", in: arguments) ?? "openai-chat"
            guard let api = GenerationResponseAPI(rawValue: rawAPI) else {
                throw CLIError.unsupportedAPI(rawAPI)
            }
            let model = optionalValue(for: "--model", in: arguments) ?? "model"
            let promptTokens = Int(optionalValue(for: "--prompt-tokens", in: arguments) ?? "0") ?? 0
            let finishReason = optionalValue(for: "--finish-reason", in: arguments) ?? "stop"
            let stream = optionalValue(for: "--stream", in: arguments) == "true"
            let finalText = optionalValue(for: "--final-text", in: arguments) ?? ""
            let chunks = try values(for: "--chunk", in: arguments).map(parseGenerationChunk) + [
                GenerationChunk(text: finalText, isFinished: true, finishReason: finishReason),
            ]
            printJSON(
                GenerationEndpointRenderer.render(
                    model: model,
                    promptTokenCount: promptTokens,
                    api: api,
                    stream: stream,
                    chunks: chunks
                )
            )
        case "self-test":
            try SelfTest.run()
        case "serve":
            let path = try value(for: "--model", in: arguments)
            let port = UInt16(optionalValue(for: "--port", in: arguments) ?? "11434") ?? 11434
            let defaultParameters = generationDefaults(from: arguments)
            let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: path)
            let server = CompatibilityServer(
                descriptor: descriptor,
                port: port,
                defaultParameters: defaultParameters
            )
            try server.start()
        default:
            printUsage()
        }
    }

    private static func value(for flag: String, in arguments: [String]) throws -> String {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else {
            throw CLIError.missingArgument(flag)
        }
        return arguments[index + 1]
    }

    private static func optionalValue(for flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else {
            return nil
        }
        return arguments[index + 1]
    }

    private static func values(for flag: String, in arguments: [String]) -> [String] {
        arguments.indices.compactMap { index in
            guard arguments[index] == flag, arguments.indices.contains(index + 1) else {
                return nil
            }
            return arguments[index + 1]
        }
    }

    private static func hasFlag(_ flag: String, in arguments: [String]) -> Bool {
        arguments.contains(flag)
    }

    private static func printJSON<T: Encodable>(_ value: T) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try! encoder.encode(value)
        print(String(decoding: data, as: UTF8.self))
    }

    private static func parseIntCSV(_ value: String) throws -> [Int] {
        try value.split(separator: ",").map { part in
            guard let intValue = Int(part.trimmingCharacters(in: .whitespaces)) else {
                throw CLIError.invalidIntegerList(value)
            }
            return intValue
        }
    }

    private static func parseDoubleCSV(_ value: String) throws -> [Double] {
        try value.split(separator: ",").map { part in
            guard let doubleValue = Double(part.trimmingCharacters(in: .whitespaces)) else {
                throw CLIError.invalidDoubleList(value)
            }
            return doubleValue
        }
    }

    private static func parseDecodeToken(_ value: String) throws -> GenerationDecodeToken {
        let parts = value.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2,
              let tokenID = Int(parts[0].trimmingCharacters(in: .whitespaces))
        else {
            throw CLIError.invalidDecodeToken(value)
        }
        return GenerationDecodeToken(tokenID: tokenID, text: String(parts[1]))
    }

    private static func parseGenerationChunk(_ value: String) throws -> GenerationChunk {
        let token = try parseDecodeToken(value)
        return GenerationChunk(text: token.text, tokenID: token.tokenID)
    }

    private static func printUsage() {
        print("""
        Usage:
          mlx-vlm-swift inspect --model /path/to/mlx-model
          mlx-vlm-swift inspect-qwen-vl --model /path/to/qwen2-vl-model
          mlx-vlm-swift inspect-qwen-vl-weights --model /path/to/qwen2-vl-model
          mlx-vlm-swift sanitize-qwen-vl-key --key model.embed_tokens.weight
          mlx-vlm-swift inspect-qwen-vl-architecture --model /path/to/qwen2-vl-model
          mlx-vlm-swift inspect-weight-catalog --model /path/to/mlx-model
          mlx-vlm-swift inspect-weight-data --model /path/to/mlx-model
          mlx-vlm-swift plan-mlx-weight-load --model /path/to/mlx-model
          mlx-vlm-swift prepare-mlx-weights --model /path/to/mlx-model [--tensor model.embed_tokens.weight] [--max-tensors 128] [--max-total-bytes 1048576] [--skip-weight-payloads]
          mlx-vlm-swift preview-weight-tensor --model /path/to/mlx-model --tensor model.embed_tokens.weight [--bytes 32]
          mlx-vlm-swift read-weight-tensor-payload --model /path/to/mlx-model --tensor model.embed_tokens.weight
          mlx-vlm-swift inspect-safetensors --model /path/to/mlx-model
          mlx-vlm-swift inspect-tokenizer-catalog --model /path/to/mlx-model
          mlx-vlm-swift inspect-tokenizer-plan --model /path/to/mlx-model
          mlx-vlm-swift inspect-chat-template-plan --model /path/to/mlx-model
          mlx-vlm-swift inspect-capabilities --model /path/to/mlx-model
          mlx-vlm-swift inspect-adapter --model /path/to/mlx-model
          mlx-vlm-swift inspect-processor --model /path/to/mlx-model
          mlx-vlm-swift inspect-config-normalization --model /path/to/mlx-model
          mlx-vlm-swift inspect-normalized-config --model /path/to/mlx-model
          mlx-vlm-swift preflight-tokenize --model /path/to/mlx-model --text '<|image_pad|>'
          mlx-vlm-swift tokenize-simple --model /path/to/mlx-model --text 'hello world'
          mlx-vlm-swift detokenize-simple --model /path/to/mlx-model --ids 1,2,3 [--skip-special true]
          mlx-vlm-swift decode-token-stream --model /path/to/mlx-model --ids 1,2,3 [--skip-special true]
          mlx-vlm-swift expand-qwen-vl-placeholders --text '<|image_pad|>' --image-grid 1,28,28
          mlx-vlm-swift plan-qwen-vl-merge --input-ids 1,151655,2 --feature-count 1
          mlx-vlm-swift format-qwen-vl-prompt [--style plain|qwen-chat] --api openai-chat --json '{"model":"m","messages":[...]}'
          mlx-vlm-swift format-qwen-vl-prompt --api openai-responses --json '{"model":"m","input":"..."}'
          mlx-vlm-swift inspect-media --api ollama-generate --json '{"model":"m","prompt":"p","images":[...]}'
          mlx-vlm-swift plan-qwen-vl-image-grid --height 224 --width 224
          mlx-vlm-swift plan-qwen-vl-images --api openai-chat --json '{"model":"m","messages":[...]}'
          mlx-vlm-swift plan-qwen-vl-pixels --api openai-chat --json '{"model":"m","messages":[...]}'
          mlx-vlm-swift apply-stop-sequences --text 'helloENDtail' --stop END
          mlx-vlm-swift list-supported
          mlx-vlm-swift backend-status
          mlx-vlm-swift backend-plan [--root /path/to/repo]
          mlx-vlm-swift backend-availability [--root /path/to/repo]
          mlx-vlm-swift inspect-mlx-metal-library
          mlx-vlm-swift preflight-mlx-backend-load --model /path/to/mlx-model [--tensor model.embed_tokens.weight] [--max-tensors 16] [--max-total-bytes 67108864] [--skip-weight-payloads]
          mlx-vlm-swift inspect-mlx-container --model /path/to/mlx-model [--tensor model.embed_tokens.weight] [--max-tensors 16] [--max-total-bytes 67108864] [--skip-weight-payloads]
          mlx-vlm-swift inspect-mlx-module-plan --model /path/to/mlx-model [--tensor model.embed_tokens.weight] [--max-tensors 16] [--max-total-bytes 67108864] [--skip-weight-payloads]
          mlx-vlm-swift inspect-mlx-forward-plan --model /path/to/mlx-model --api openai-chat --json '{"model":"m","messages":[...]}' [--max-tensors 16] [--max-total-bytes 67108864] [--skip-weight-payloads]
          mlx-vlm-swift inspect-mlx-generation-loop --model /path/to/mlx-model --api openai-chat --json '{"model":"m","messages":[...]}' [--max-tensors 16] [--max-total-bytes 67108864] [--skip-weight-payloads]
          mlx-vlm-swift inspect-mlx-decode-state --model /path/to/mlx-model --api openai-chat --json '{"model":"m","messages":[...]}' [--max-tensors 16] [--max-total-bytes 67108864] [--skip-weight-payloads]
          mlx-vlm-swift inspect-mlx-generate-parameters --model /path/to/mlx-model --api openai-chat --json '{"model":"m","messages":[...]}'
          mlx-vlm-swift inspect-mlx-generate-parameters-bridge --model /path/to/mlx-model --api openai-chat --json '{"model":"m","messages":[...]}'
          mlx-vlm-swift inspect-mlx-model-factory-bridge --model /path/to/mlx-model
          mlx-vlm-swift inspect-mlx-pipeline --model /path/to/mlx-model --api openai-chat --json '{"model":"m","messages":[...]}' [--max-tensors 16] [--max-total-bytes 67108864] [--skip-weight-payloads]
          mlx-vlm-swift inspect-backend-context --model /path/to/mlx-model
          mlx-vlm-swift inspect-ollama-show --model /path/to/mlx-model
          mlx-vlm-swift preflight-ollama-show --json '{"model":"m","verbose":true}'
          mlx-vlm-swift validate-model --model /path/to/mlx-model
          mlx-vlm-swift plan-model-load --model /path/to/mlx-model
          mlx-vlm-swift plan-mlx-bindings --model /path/to/mlx-model
          mlx-vlm-swift estimate-memory --model /path/to/mlx-model [--context-length 4096] [--kv-bits 8] [--max-kv-size 4096] [--vision-cache-size 8]
          mlx-vlm-swift preflight-generate --model /path/to/mlx-model --api openai-chat --json '{"model":"m","messages":[...]}' [--context-length 4096] [--keep-alive 5m]
          mlx-vlm-swift preflight-embed --model /path/to/mlx-model --api ollama-embed --json '{"model":"m","input":"..."}'
          mlx-vlm-swift preflight-model-operation --operation pull --json '{"model":"m"}'
          mlx-vlm-swift preflight-ollama-blob --digest sha256:abc123
          mlx-vlm-swift preflight-ollama-residency --json '{"model":"m","prompt":"","keep_alive":0}'
          mlx-vlm-swift sample-logits --logits 0.1,0.9,0.2 [--temperature 0.7] [--top-k 40] [--top-p 0.9] [--min-p 0.05] [--seed 42] [--recent-token-ids 1,2,3]
          mlx-vlm-swift simulate-decode-loop --model m --prompt-tokens 4 --max-tokens 8 --token 1:he --token 2:llo --stop END [--eos 151643]
          mlx-vlm-swift simulate-logits-decode --model /path/to/mlx-model --prompt-tokens 4 --max-tokens 8 --logits 0.1,0.9,0.2 [--temperature 0.7] [--eos 151643]
          mlx-vlm-swift render-generation-response --api openai-chat --model m --text ok [--stream true] [--chunk o]
          mlx-vlm-swift render-generation-chunks --api openai-chat --model m --prompt-tokens 4 --chunk 1:o [--stream true]
          mlx-vlm-swift self-test
          mlx-vlm-swift serve --model /path/to/mlx-model [--port 11434] [--max-tokens 512] [--temperature 0.0] [--top-p 1.0] [--top-k 0] [--min-p 0.0] [--seed 0] [--context-length 4096] [--kv-bits 8] [--kv-quant-scheme uniform] [--keep-alive 5m]
        """)
    }
}

private func generationDefaults(from arguments: [String]) -> GenerationParameters {
    GenerationParameters(
        maxTokens: Int(optionalArgumentValue(for: "--max-tokens", in: arguments) ?? "512") ?? 512,
        temperature: Double(optionalArgumentValue(for: "--temperature", in: arguments) ?? "0.0") ?? 0.0,
        topP: Double(optionalArgumentValue(for: "--top-p", in: arguments) ?? "1.0") ?? 1.0,
        topK: Int(optionalArgumentValue(for: "--top-k", in: arguments) ?? "0") ?? 0,
        minP: optionalArgumentValue(for: "--min-p", in: arguments).flatMap(Double.init),
        typicalP: optionalArgumentValue(for: "--typical-p", in: arguments).flatMap(Double.init),
        tfsZ: optionalArgumentValue(for: "--tfs-z", in: arguments).flatMap(Double.init),
        seed: Int(optionalArgumentValue(for: "--seed", in: arguments) ?? "0") ?? 0,
        contextLength: optionalArgumentValue(for: "--context-length", in: arguments).flatMap(Int.init),
        numKeep: optionalArgumentValue(for: "--num-keep", in: arguments).flatMap(Int.init),
        kvBits: optionalArgumentValue(for: "--kv-bits", in: arguments).flatMap(Double.init),
        kvQuantizationScheme: optionalArgumentValue(for: "--kv-quant-scheme", in: arguments),
        kvGroupSize: optionalArgumentValue(for: "--kv-group-size", in: arguments).flatMap(Int.init),
        maxKVSize: optionalArgumentValue(for: "--max-kv-size", in: arguments).flatMap(Int.init),
        visionCacheSize: optionalArgumentValue(for: "--vision-cache-size", in: arguments).flatMap(Int.init),
        quantizeActivations: optionalArgumentValue(for: "--quantize-activations", in: arguments).flatMap(parseBool),
        repetitionPenalty: optionalArgumentValue(for: "--repeat-penalty", in: arguments).flatMap(Double.init),
        repeatLastN: optionalArgumentValue(for: "--repeat-last-n", in: arguments).flatMap(Int.init),
        presencePenalty: optionalArgumentValue(for: "--presence-penalty", in: arguments).flatMap(Double.init),
        frequencyPenalty: optionalArgumentValue(for: "--frequency-penalty", in: arguments).flatMap(Double.init),
        penalizeNewline: optionalArgumentValue(for: "--penalize-newline", in: arguments).flatMap(parseBool),
        mirostat: optionalArgumentValue(for: "--mirostat", in: arguments).flatMap(Int.init),
        mirostatTau: optionalArgumentValue(for: "--mirostat-tau", in: arguments).flatMap(Double.init),
        mirostatEta: optionalArgumentValue(for: "--mirostat-eta", in: arguments).flatMap(Double.init),
        keepAlive: optionalArgumentValue(for: "--keep-alive", in: arguments)
    )
}

private func parseBool(_ value: String) -> Bool? {
    switch value.lowercased() {
    case "1", "true", "yes", "on":
        return true
    case "0", "false", "no", "off":
        return false
    default:
        return nil
    }
}

private func optionalArgumentValue(for flag: String, in arguments: [String]) -> String? {
    guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else {
        return nil
    }
    return arguments[index + 1]
}

private func normalizedGenerationRequest(
    api: String,
    json: String,
    defaultParameters: GenerationParameters = GenerationParameters()
) throws -> GenerationRequest {
    let data = Data(json.utf8)
    switch api {
    case "ollama-generate":
        return try JSONDecoder()
            .decode(OllamaGenerateRequest.self, from: data)
            .generationRequest(defaultModel: "default", defaultParameters: defaultParameters)
    case "ollama-chat":
        return try JSONDecoder()
            .decode(OllamaChatRequest.self, from: data)
            .generationRequest(defaultModel: "default", defaultParameters: defaultParameters)
    case "openai-chat":
        return try JSONDecoder()
            .decode(OpenAIChatCompletionRequest.self, from: data)
            .generationRequest(defaultModel: "default", defaultParameters: defaultParameters)
    case "openai-responses":
        return try JSONDecoder()
            .decode(OpenAIResponsesRequest.self, from: data)
            .generationRequest(defaultModel: "default", defaultParameters: defaultParameters)
    default:
        throw CLIError.unsupportedAPI(api)
    }
}

private func normalizedEmbeddingRequest(
    api: String,
    json: String,
    defaultParameters: GenerationParameters = GenerationParameters()
) throws -> EmbeddingRequest {
    let data = Data(json.utf8)
    switch api {
    case "ollama-embed":
        return try JSONDecoder()
            .decode(OllamaEmbedRequest.self, from: data)
            .embeddingRequest(defaultModel: "default", defaultParameters: defaultParameters)
    case "ollama-embeddings":
        return try JSONDecoder()
            .decode(OllamaEmbeddingsRequest.self, from: data)
            .embeddingRequest(defaultModel: "default", defaultParameters: defaultParameters)
    case "openai-embeddings":
        return try JSONDecoder()
            .decode(OpenAIEmbeddingRequest.self, from: data)
            .embeddingRequest(defaultModel: "default")
    default:
        throw CLIError.unsupportedAPI(api)
    }
}

private extension MediaResolutionReport {
    var dictionary: [String: Any] {
        [
            "image_count": imageCount,
            "audio_count": audioCount,
            "video_count": videoCount,
            "loadable_count": loadableCount,
            "error_count": errorCount,
            "references": references.map(\.dictionary),
        ]
    }
}

private extension MediaReferenceSummary {
    var dictionary: [String: Any] {
        var result: [String: Any] = [
            "kind": kind.rawValue,
            "source": source.rawValue,
            "is_loadable": isLoadable,
        ]
        if let mimeType {
            result["mime_type"] = mimeType
        }
        if let byteCount {
            result["byte_count"] = byteCount
        }
        if let location {
            result["location"] = location
        }
        if let error {
            result["error"] = error
        }
        return result
    }
}

private extension QwenVLImageInputReport {
    var dictionary: [String: Any] {
        [
            "image_count": imageCount,
            "planned_count": plannedCount,
            "error_count": errorCount,
            "placeholder_token_count": placeholderTokenCount,
            "images": images.map(\.dictionary),
        ]
    }
}

private extension QwenVLResolvedImageInput {
    var dictionary: [String: Any] {
        var result: [String: Any] = ["media": media.dictionary]
        if let gridPlan {
            result["grid_plan"] = gridPlan.dictionary
        }
        if let error {
            result["error"] = error
        }
        return result
    }
}

private extension QwenVLImageGridPlan {
    var dictionary: [String: Any] {
        [
            "original_height": originalHeight,
            "original_width": originalWidth,
            "resized_height": resizedHeight,
            "resized_width": resizedWidth,
            "grid": [
                "temporal": grid.temporal,
                "height": grid.height,
                "width": grid.width,
            ],
            "placeholder_token_count": placeholderTokenCount,
        ]
    }
}

private extension TokenizationPreflightPlan {
    var dictionary: [String: Any] {
        var result: [String: Any] = [
            "token_ids": tokenIDs,
            "known_token_count": knownTokenCount,
            "unknown_fragment_count": unknownFragmentCount,
            "requires_tokenizer_implementation": requiresTokenizerImplementation,
            "fragments": fragments.map(\.dictionary),
        ]
        if let tokenizerModelType {
            result["tokenizer_model_type"] = tokenizerModelType
        }
        return result
    }
}

private extension TokenizationPreflightFragment {
    var dictionary: [String: Any] {
        var result: [String: Any] = [
            "text": text,
            "known": isKnownToken,
        ]
        if let tokenID {
            result["token_id"] = tokenID
        }
        if let source {
            result["source"] = source
        }
        return result
    }
}

enum CLIError: Error, CustomStringConvertible {
    case missingArgument(String)
    case missingWeightIndex(String)
    case missingTokenizerJSON(String)
    case invalidIntegerList(String)
    case invalidDoubleList(String)
    case invalidDecodeToken(String)
    case unsupportedAPI(String)
    case unsupportedPromptStyle(String)

    var description: String {
        switch self {
        case .missingArgument(let flag):
            return "Missing required argument: \(flag)"
        case .missingWeightIndex(let path):
            return "Missing model.safetensors.index.json for model path: \(path)"
        case .missingTokenizerJSON(let path):
            return "Missing tokenizer.json for model path: \(path)"
        case .invalidIntegerList(let value):
            return "Invalid comma-separated integer list: \(value)"
        case .invalidDoubleList(let value):
            return "Invalid comma-separated number list: \(value)"
        case .invalidDecodeToken(let value):
            return "Invalid decode token: \(value). Expected <token-id>:<text>."
        case .unsupportedAPI(let api):
            return "Unsupported API formatter: \(api)"
        case .unsupportedPromptStyle(let style):
            return "Unsupported prompt style: \(style)"
        }
    }
}

struct ServerMLXGenerationDiagnostics: Codable, Equatable, Sendable {
    let pipelineReady: Bool
    let backendImplementationReady: Bool
    let realMLXAPIImplementationCompiled: Bool
    let canPrepareWeightPayloads: Bool
    let preparedWeightTensorCount: Int
    let loadedArrayCount: Int
    let canCreateMLXArrays: Bool
    let canInstantiateModelModules: Bool
    let forwardReady: Bool
    let generationLoopReady: Bool
    let canBridgeToGenerateParameters: Bool
    let canReferenceVLMModelFactory: Bool
    let canLoadLocalModelContainer: Bool
    let blockingReasons: [String]

    init(
        descriptor: ModelDescriptor,
        availability: MLXBackendAvailability = MLXBackendFactory.availability()
    ) {
        let factoryBridge = MLXVLMModelFactoryBridge().report(for: descriptor)
        self.pipelineReady = false
        self.backendImplementationReady = availability.runtimeProbe.backendImplementationReady
        self.realMLXAPIImplementationCompiled = availability.runtimeProbe.realMLXAPIImplementationCompiled
        self.canPrepareWeightPayloads = false
        self.preparedWeightTensorCount = descriptor.safetensorsMetadata.reduce(0) { $0 + $1.tensorCount }
        self.loadedArrayCount = 0
        self.canCreateMLXArrays = false
        self.canInstantiateModelModules = false
        self.forwardReady = false
        self.generationLoopReady = false
        self.canBridgeToGenerateParameters = false
        self.canReferenceVLMModelFactory = factoryBridge.canReferenceVLMModelFactory
        self.canLoadLocalModelContainer = factoryBridge.canLoadLocalModelContainer
        self.blockingReasons = availability.blockingReasons + factoryBridge.blockingReasons
    }

    init(report: MLXGenerationPipelineReport) {
        self.pipelineReady = report.pipelineReady
        self.backendImplementationReady = report.load.availability.runtimeProbe.backendImplementationReady
        self.realMLXAPIImplementationCompiled = report.load.availability.runtimeProbe.realMLXAPIImplementationCompiled
        self.canPrepareWeightPayloads = report.load.canPrepareWeightPayloads
        self.preparedWeightTensorCount = report.container.preparedWeightTensorCount
        self.loadedArrayCount = report.container.loadedArrayCount
        self.canCreateMLXArrays = report.load.canCreateMLXArrays
        self.canInstantiateModelModules = report.load.canInstantiateModelModules
        self.forwardReady = report.forward.forwardReady
        self.generationLoopReady = report.generationLoop.generationLoopReady
        self.canBridgeToGenerateParameters = report.generateParametersBridge.canBridgeToGenerateParameters
        self.canReferenceVLMModelFactory = report.modelFactoryBridge.canReferenceVLMModelFactory
        self.canLoadLocalModelContainer = report.modelFactoryBridge.canLoadLocalModelContainer
        self.blockingReasons = report.blockingReasons
    }
}

struct ServerGenerationUnavailableReport: Codable, Equatable, Sendable {
    let error: String
    let model: String
    let canonicalModelType: String
    let backend: BackendStatus
    let preflight: GenerationPreflightPlan
    let mlxPipeline: ServerMLXGenerationDiagnostics?

    init(base: GenerationUnavailableReport, mlxPipeline: ServerMLXGenerationDiagnostics?) {
        self.error = base.error
        self.model = base.model
        self.canonicalModelType = base.canonicalModelType
        self.backend = base.backend
        self.preflight = base.preflight
        self.mlxPipeline = mlxPipeline
    }
}

final class CompatibilityServer: @unchecked Sendable {
    private let descriptor: ModelDescriptor
    private let port: UInt16
    private let defaultParameters: GenerationParameters
    private let deepDiagnostics: Bool
    private let queue = DispatchQueue(label: "mlx-vlm-swift.compatibility-server")
    private var residency = OllamaModelResidency()
    private var generationBackend: (any VLMBackend)?
    private var embeddingBackend: (any EmbeddingBackend)?

    init(
        descriptor: ModelDescriptor,
        port: UInt16,
        defaultParameters: GenerationParameters = GenerationParameters()
    ) {
        self.descriptor = descriptor
        self.port = port
        self.defaultParameters = defaultParameters
        self.deepDiagnostics = ProcessInfo.processInfo.environment["MLXVLM_SERVER_DEEP_DIAGNOSTICS"] == "1"
    }

    private enum EmbeddingResponseAPI {
        case ollamaEmbed
        case ollamaEmbeddings
        case openAIEmbeddings
    }

    func start() throws {
        let serverSocket = socket(AF_INET, SOCK_STREAM, 0)
        guard serverSocket >= 0 else {
            throw ServerError.socketFailed(errno)
        }
        var reuse = 1
        setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

        let bindResult = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                bind(serverSocket, sockaddrPointer, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            close(serverSocket)
            throw ServerError.bindFailed(errno)
        }

        guard listen(serverSocket, 16) == 0 else {
            close(serverSocket)
            throw ServerError.listenFailed(errno)
        }

        print("mlx-vlm-swift compatibility server listening on http://127.0.0.1:\(port)")
        print("Loaded model descriptor: \(descriptor.id) (\(descriptor.canonicalModelType))")
        let minP = defaultParameters.minP.map { String($0) } ?? "default"
        let contextLength = defaultParameters.contextLength.map { String($0) } ?? "auto"
        let kvBits = defaultParameters.kvBits.map { String($0) } ?? "off"
        let kvScheme = defaultParameters.kvQuantizationScheme ?? "default"
        let keepAlive = defaultParameters.keepAlive ?? "default"
        print("Default generation parameters: max_tokens=\(defaultParameters.maxTokens), temperature=\(defaultParameters.temperature), top_p=\(defaultParameters.topP), top_k=\(defaultParameters.topK), min_p=\(minP), seed=\(defaultParameters.seed), context_length=\(contextLength), kv_bits=\(kvBits), kv_quant_scheme=\(kvScheme), keep_alive=\(keepAlive)")

        if MLXBackendFactory.availability().canCreateBackend {
            print("Loading MLX Swift generation backend...")
            do {
                generationBackend = try waitForAsync {
                    try await MLXBackendFactory.makeBackendAsync(descriptor: self.descriptor)
                }
                print("MLX Swift generation backend ready.")
            } catch {
                print("MLX Swift generation backend failed to load; serving compatibility diagnostics: \(error)")
            }

            print("Loading MLX Swift embedding backend...")
            do {
                embeddingBackend = try waitForAsync {
                    try await MLXBackendFactory.makeEmbeddingBackendAsync(descriptor: self.descriptor)
                }
                print("MLX Swift embedding backend ready.")
            } catch {
                print("MLX Swift embedding backend failed to load; serving embedding diagnostics: \(error)")
            }
        }

        while true {
            let clientSocket = accept(serverSocket, nil, nil)
            guard clientSocket >= 0 else {
                continue
            }
            queue.async { [self] in
                self.handle(clientSocket)
            }
        }
    }

    private func handle(_ clientSocket: Int32) {
        guard let request = readHTTPRequest(from: clientSocket) else {
            close(clientSocket)
            return
        }
        if streamGenerationResponseIfNeeded(for: request, clientSocket: clientSocket) {
            close(clientSocket)
            return
        }
        let response = route(request: request)
        writeData(response, to: clientSocket)
        close(clientSocket)
    }

    private func streamGenerationResponseIfNeeded(for rawRequest: String, clientSocket: Int32) -> Bool {
        guard let generationBackend,
              let (request, api) = streamableGenerationRequest(from: rawRequest)
        else {
            return false
        }

        do {
            try waitForAsync {
                try await self.writeStreamingGenerationResponse(
                    request: request,
                    api: api,
                    backend: generationBackend,
                    clientSocket: clientSocket
                )
            }
        } catch {
            let response = jsonResponse(
                [
                    "error": String(describing: error),
                    "backend": generationBackend.status.activeBackend,
                ],
                status: "500 Internal Server Error"
            )
            writeData(response, to: clientSocket)
        }
        return true
    }

    private func streamableGenerationRequest(from rawRequest: String) -> (GenerationRequest, GenerationResponseAPI)? {
        let firstLine = rawRequest.split(separator: "\r\n", maxSplits: 1).first ?? ""
        let parts = firstLine.split(separator: " ")
        let path = normalizedRoutePath(parts.count >= 2 ? String(parts[1]) : "/")
        guard let body = httpBody(from: rawRequest), !body.isEmpty else {
            return nil
        }
        let data = Data(body.utf8)
        do {
            switch path {
            case "/api/generate":
                let decoded = try JSONDecoder().decode(OllamaGenerateRequest.self, from: data)
                guard decoded.residencyAction(defaultModel: descriptor.id) == nil else {
                    return nil
                }
                let request = try decoded.generationRequest(
                    defaultModel: descriptor.id,
                    defaultParameters: defaultParameters
                )
                guard request.stream else {
                    return nil
                }
                residency.markLoaded()
                return (request, .ollamaGenerate)
            case "/api/chat":
                let decoded = try JSONDecoder().decode(OllamaChatRequest.self, from: data)
                let request = try decoded.generationRequest(
                    defaultModel: descriptor.id,
                    defaultParameters: defaultParameters
                )
                guard request.stream else {
                    return nil
                }
                residency.markLoaded()
                return (request, .ollamaChat)
            case "/generate", "/chat/completions", "/v1/chat/completions":
                let decoded = try JSONDecoder().decode(OpenAIChatCompletionRequest.self, from: data)
                let request = try decoded.generationRequest(
                    defaultModel: descriptor.id,
                    defaultParameters: defaultParameters
                )
                guard request.stream else {
                    return nil
                }
                residency.markLoaded()
                return (request, .openAIChat)
            default:
                return nil
            }
        } catch {
            return nil
        }
    }

    private func writeStreamingGenerationResponse(
        request: GenerationRequest,
        api: GenerationResponseAPI,
        backend: any VLMBackend,
        clientSocket: Int32
    ) async throws {
        let contentType: String = switch api {
        case .ollamaGenerate, .ollamaChat:
            "application/x-ndjson"
        case .openAIChat, .openAIResponses:
            "text/event-stream"
        }
        let header = streamingHTTPHeader(status: "200 OK", contentType: contentType)
        writeData(Data(header.utf8), to: clientSocket)

        let processed = try backend.process(request)
        var assembler = GenerationOutputAssembler(
            model: request.model,
            promptTokenCount: processed.tokenIDs?.count ?? 0,
            stopSequences: request.parameters.stopSequences
        )
        let stream = try await backend.generate(request)
        var wroteFinishedChunk = false
        for try await chunk in stream {
            let emitted = assembler.append(chunk)
            for output in emitted {
                writeStreamFrame(
                    output,
                    model: request.model,
                    api: api,
                    usage: output.isFinished ? assembler.snapshot.usage : nil,
                    to: clientSocket
                )
                if output.isFinished {
                    wroteFinishedChunk = true
                }
            }
            if assembler.snapshot.isFinished {
                break
            }
        }

        if !wroteFinishedChunk, let finalChunk = assembler.finish() {
            writeStreamFrame(
                finalChunk,
                model: request.model,
                api: api,
                usage: assembler.snapshot.usage,
                to: clientSocket
            )
        }

        if api == .openAIChat {
            writeData(Data(ResponseStreamFramer.doneServerSentEvent().utf8), to: clientSocket)
        }
    }

    private func writeStreamFrame(
        _ chunk: GenerationChunk,
        model: String,
        api: GenerationResponseAPI,
        usage: GenerationUsage?,
        to clientSocket: Int32
    ) {
        let frame: String
        switch api {
        case .ollamaGenerate:
            frame = ResponseStreamFramer.jsonLine(
                OllamaGenerateStreamChunk(model: model, chunk: chunk, usage: usage)
            )
        case .ollamaChat:
            frame = ResponseStreamFramer.jsonLine(
                OllamaChatStreamChunk(model: model, chunk: chunk, usage: usage)
            )
        case .openAIChat:
            frame = ResponseStreamFramer.serverSentEvent(
                OpenAIChatCompletionStreamResponse(model: model, chunk: chunk, usage: usage)
            )
        case .openAIResponses:
            return
        }
        writeData(Data(frame.utf8), to: clientSocket)
    }

    private func streamingHTTPHeader(status: String, contentType: String) -> String {
        "HTTP/1.1 \(status)\r\n" +
            "Content-Type: \(contentType)\r\n" +
            "Cache-Control: no-cache\r\n" +
            "Access-Control-Allow-Origin: *\r\n" +
            "Access-Control-Allow-Methods: GET, POST, HEAD, OPTIONS\r\n" +
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" +
            "Connection: close\r\n" +
            "\r\n"
    }

    private func writeData(_ data: Data, to clientSocket: Int32) {
        data.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else {
                return
            }
            var sent = 0
            while sent < data.count {
                let result = Darwin.write(
                    clientSocket,
                    baseAddress.advanced(by: sent),
                    data.count - sent
                )
                guard result > 0 else {
                    break
                }
                sent += result
            }
        }
    }

    private func readHTTPRequest(from clientSocket: Int32) -> String? {
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 16 * 1024)
        var expectedLength: Int?

        while true {
            let count = read(clientSocket, &buffer, buffer.count)
            guard count > 0 else {
                break
            }
            data.append(buffer, count: count)

            if expectedLength == nil,
               let headerRange = data.range(of: Data("\r\n\r\n".utf8))
            {
                let headerData = data[..<headerRange.lowerBound]
                let header = String(decoding: headerData, as: UTF8.self)
                let contentLength = parseContentLength(from: header)
                expectedLength = headerRange.upperBound + contentLength
            }

            if let expectedLength, data.count >= expectedLength {
                break
            }
        }

        guard !data.isEmpty else {
            return nil
        }
        return String(decoding: data, as: UTF8.self)
    }

    private func parseContentLength(from header: String) -> Int {
        for line in header.split(separator: "\r\n") {
            let parts = line.split(separator: ":", maxSplits: 1)
            guard parts.count == 2,
                  parts[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "content-length"
            else {
                continue
            }
            return Int(parts[1].trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
        }
        return 0
    }

    private func route(request: String) -> Data {
        let firstLine = request.split(separator: "\r\n", maxSplits: 1).first ?? ""
        let parts = firstLine.split(separator: " ")
        let method = parts.first.map(String.init) ?? "GET"
        let path = normalizedRoutePath(parts.count >= 2 ? String(parts[1]) : "/")

        if method == "OPTIONS" {
            return httpResponse(status: "204 No Content", body: Data())
        }

        switch path {
        case "/health":
            return jsonResponse(
                [
                    "status": "ok",
                    "model": descriptor.id,
                    "model_loaded": residency.isLoaded,
                    "backend_ready": currentBackendStatus.ready,
                    "backend": currentBackendStatus.activeBackend,
                    "embedding_backend_ready": embeddingBackend != nil,
                    "embedding_backend": embeddingBackend?.status.activeBackend ?? "unavailable",
                ]
            )
        case "/backend/status":
            return encodedResponse(currentBackendStatus)
        case "/api/version":
            return encodedResponse(OllamaVersionResponse(version: "mlx-vlm-swift-compat"))
        case "/api/tags":
            return encodedResponse(OllamaTagsResponse(models: [OllamaModelTag(descriptor: descriptor)]))
        case "/api/ps":
            return encodedResponse(residency.runningModelsResponse(descriptor: descriptor))
        case "/models", "/v1/models":
            return encodedResponse(
                OpenAIModelListResponse(models: [OpenAIModelResponse(id: descriptor.id)])
            )
        case let modelPath where modelPath.hasPrefix("/v1/models/") || modelPath.hasPrefix("/models/"):
            return openAIModelResponse(for: modelPath)
        case "/api/show":
            return parseOllamaShowRequest(from: request)
        case "/unload", "/api/unload":
            return encodedResponse(residency.unload(model: descriptor.id))
        case "/api/create":
            return parseModelOperationRequest(from: request, operation: "create")
        case "/api/pull":
            return parseModelOperationRequest(from: request, operation: "pull")
        case "/api/push":
            return parseModelOperationRequest(from: request, operation: "push")
        case "/api/copy":
            return parseModelOperationRequest(from: request, operation: "copy")
        case "/api/delete":
            return parseModelOperationRequest(from: request, operation: "delete")
        case let blobPath where blobPath.hasPrefix("/api/blobs/"):
            return parseBlobOperationRequest(path: blobPath, method: method)
        case "/api/generate":
            return parseOllamaGenerateRequest(from: request)
        case "/api/chat":
            return parseOllamaChatRequest(from: request)
        case "/generate", "/chat/completions", "/v1/chat/completions":
            return parseOpenAIChatRequest(from: request)
        case "/responses", "/v1/responses":
            return parseOpenAIResponsesRequest(from: request)
        case "/api/embed":
            return parseOllamaEmbedRequest(from: request)
        case "/api/embeddings":
            return parseOllamaEmbeddingsRequest(from: request)
        case "/embeddings", "/v1/embeddings":
            return parseOpenAIEmbeddingRequest(from: request)
        default:
            return jsonResponse(["error": "Not found", "path": path], status: "404 Not Found")
        }
    }

    private func openAIModelResponse(for path: String) -> Data {
        let prefix = path.hasPrefix("/v1/models/") ? "/v1/models/" : "/models/"
        let requestedID = String(path.dropFirst(prefix.count))
        guard requestedID == descriptor.id else {
            return jsonResponse(
                [
                    "error": [
                        "message": "Model '\(requestedID)' was not found.",
                        "type": "invalid_request_error",
                        "param": "model",
                        "code": "model_not_found",
                    ]
                ],
                status: "404 Not Found"
            )
        }
        return encodedResponse(OpenAIModelResponse(id: descriptor.id))
    }

    private func parseOllamaShowRequest(from request: String) -> Data {
        let body = httpBody(from: request) ?? ""
        guard !body.isEmpty else {
            return encodedResponse(OllamaShowResponse(descriptor: descriptor))
        }

        do {
            let decoded = try JSONDecoder().decode(OllamaShowRequest.self, from: Data(body.utf8))
            if let requestedModel = decoded.modelName, requestedModel != descriptor.id {
                return jsonResponse(
                    [
                        "error": [
                            "message": "Model '\(requestedModel)' was not found.",
                            "type": "invalid_request_error",
                            "param": "model",
                            "code": "model_not_found",
                        ]
                    ],
                    status: "404 Not Found"
                )
            }
            return encodedResponse(OllamaShowResponse(descriptor: descriptor))
        } catch {
            return jsonResponse(["parse_error": String(describing: error)], status: "400 Bad Request")
        }
    }

    private func parseOllamaGenerateRequest(from request: String) -> Data {
        guard let body = httpBody(from: request), !body.isEmpty else {
            return jsonResponse(["parse_error": "empty request body"], status: "400 Bad Request")
        }

        do {
            let decoded = try JSONDecoder().decode(OllamaGenerateRequest.self, from: Data(body.utf8))
            if let action = decoded.residencyAction(defaultModel: descriptor.id) {
                return encodedResponse(applyResidencyAction(action))
            }
            let normalized = try decoded.generationRequest(
                defaultModel: descriptor.id,
                defaultParameters: defaultParameters
            )
            residency.markLoaded()
            return generationResponse(for: normalized, api: .ollamaGenerate)
        } catch {
            return jsonResponse(["parse_error": String(describing: error)], status: "400 Bad Request")
        }
    }

    private func applyResidencyAction(_ action: OllamaResidencyAction) -> OllamaGenerateResponse {
        switch action.action {
        case .load:
            residency.markLoaded()
        case .unload:
            _ = residency.unload(model: action.model)
        }
        return OllamaGenerateResponse(
            result: CompletedGeneration(model: action.model, text: ""),
            doneReason: action.action.rawValue
        )
    }

    private func parseOllamaChatRequest(from request: String) -> Data {
        parseGenerationRequest(from: request, decode: OllamaChatRequest.self, api: .ollamaChat) {
            try $0.generationRequest(
                defaultModel: descriptor.id,
                defaultParameters: defaultParameters
            )
        }
    }

    private func parseOpenAIChatRequest(from request: String) -> Data {
        parseGenerationRequest(from: request, decode: OpenAIChatCompletionRequest.self, api: .openAIChat) {
            try $0.generationRequest(
                defaultModel: descriptor.id,
                defaultParameters: defaultParameters
            )
        }
    }

    private func parseOpenAIResponsesRequest(from request: String) -> Data {
        parseGenerationRequest(from: request, decode: OpenAIResponsesRequest.self, api: .openAIResponses) {
            try $0.generationRequest(
                defaultModel: descriptor.id,
                defaultParameters: defaultParameters
            )
        }
    }

    private func parseOllamaEmbedRequest(from request: String) -> Data {
        parseEmbeddingRequest(from: request, decode: OllamaEmbedRequest.self, api: .ollamaEmbed) {
            $0.embeddingRequest(
                defaultModel: descriptor.id,
                defaultParameters: defaultParameters
            )
        }
    }

    private func parseOllamaEmbeddingsRequest(from request: String) -> Data {
        parseEmbeddingRequest(from: request, decode: OllamaEmbeddingsRequest.self, api: .ollamaEmbeddings) {
            $0.embeddingRequest(
                defaultModel: descriptor.id,
                defaultParameters: defaultParameters
            )
        }
    }

    private func parseOpenAIEmbeddingRequest(from request: String) -> Data {
        parseEmbeddingRequest(from: request, decode: OpenAIEmbeddingRequest.self, api: .openAIEmbeddings) {
            $0.embeddingRequest(defaultModel: descriptor.id)
        }
    }

    private func parseModelOperationRequest(from request: String, operation: String) -> Data {
        let body = httpBody(from: request) ?? "{}"
        let data = body.isEmpty ? Data("{}".utf8) : Data(body.utf8)
        do {
            let decoded = try JSONDecoder().decode(OllamaModelOperationRequest.self, from: data)
            return encodedResponse(
                OllamaModelOperationReport(operation: operation, request: decoded),
                status: "501 Not Implemented"
            )
        } catch {
            return jsonResponse(["parse_error": String(describing: error)], status: "400 Bad Request")
        }
    }

    private func parseBlobOperationRequest(path: String, method: String) -> Data {
        let digest = String(path.dropFirst("/api/blobs/".count))
        guard digest.hasPrefix("sha256:"), digest.count > "sha256:".count else {
            return jsonResponse(
                [
                    "error": [
                        "message": "Invalid Ollama blob digest '\(digest)'. Expected sha256:<hex>.",
                        "type": "invalid_request_error",
                        "param": "digest",
                        "code": "invalid_blob_digest",
                    ]
                ],
                status: "400 Bad Request"
            )
        }

        switch method.uppercased() {
        case "HEAD":
            return httpResponse(status: "404 Not Found", body: Data())
        case "POST":
            return encodedResponse(
                OllamaBlobOperationReport(operation: "push-blob", digest: digest),
                status: "501 Not Implemented"
            )
        default:
            return encodedResponse(
                OllamaBlobOperationReport(operation: "blob-\(method.lowercased())", digest: digest),
                status: "405 Method Not Allowed"
            )
        }
    }

    private func parseGenerationRequest<T: Decodable>(
        from request: String,
        decode type: T.Type,
        api: GenerationResponseAPI,
        normalize: (T) throws -> GenerationRequest
    ) -> Data {
        guard let body = httpBody(from: request), !body.isEmpty else {
            return jsonResponse(["parse_error": "empty request body"], status: "400 Bad Request")
        }

        do {
            let decoded = try JSONDecoder().decode(type, from: Data(body.utf8))
            let normalized = try normalize(decoded)
            residency.markLoaded()
            return generationResponse(for: normalized, api: api)
        } catch {
            return jsonResponse(["parse_error": String(describing: error)], status: "400 Bad Request")
        }
    }

    private var currentBackendStatus: BackendStatus {
        generationBackend?.status ?? .compatibilityShell
    }

    private func generationResponse(for request: GenerationRequest, api: GenerationResponseAPI) -> Data {
        guard let generationBackend else {
            return encodedResponse(generationUnavailableReport(for: request), status: "501 Not Implemented")
        }

        do {
            let report = try waitForAsync {
                try await VLMGenerationRunner(backend: generationBackend).renderedResponse(
                    for: request,
                    api: api
                )
            }
            return renderedGenerationResponse(report.response)
        } catch {
            return jsonResponse(
                [
                    "error": String(describing: error),
                    "backend": generationBackend.status.activeBackend,
                ],
                status: "500 Internal Server Error"
            )
        }
    }

    private func generationUnavailableReport(for request: GenerationRequest) -> ServerGenerationUnavailableReport {
        let base = CompatibilityGenerationEngine(descriptor: descriptor).unavailableReport(for: request)
        let diagnostics: ServerMLXGenerationDiagnostics
        if deepDiagnostics,
           let pipeline = try? MLXGenerationPipelineReporter().report(
                descriptor: descriptor,
                request: request,
                weightOptions: MLXWeightPreparationOptions(skipTensorPayloads: true)
           ) {
            diagnostics = ServerMLXGenerationDiagnostics(report: pipeline)
        } else {
            diagnostics = ServerMLXGenerationDiagnostics(descriptor: descriptor)
        }
        return ServerGenerationUnavailableReport(
            base: base,
            mlxPipeline: diagnostics
        )
    }

    private func parseEmbeddingRequest<T: Decodable>(
        from request: String,
        decode type: T.Type,
        api: EmbeddingResponseAPI,
        normalize: (T) throws -> EmbeddingRequest
    ) -> Data {
        guard let body = httpBody(from: request), !body.isEmpty else {
            return jsonResponse(["parse_error": "empty request body"], status: "400 Bad Request")
        }

        do {
            let decoded = try JSONDecoder().decode(type, from: Data(body.utf8))
            let normalized = try normalize(decoded)
            return embeddingResponse(for: normalized, api: api)
        } catch {
            return jsonResponse(["parse_error": String(describing: error)], status: "400 Bad Request")
        }
    }

    private func embeddingResponse(for request: EmbeddingRequest, api: EmbeddingResponseAPI) -> Data {
        guard let embeddingBackend else {
            let report = CompatibilityGenerationEngine(
                descriptor: descriptor,
                backend: currentBackendStatus
            ).unavailableEmbeddingReport(for: request)
            return encodedResponse(report, status: "501 Not Implemented")
        }

        do {
            let result = try waitForAsync {
                try await embeddingBackend.embed(request)
            }
            switch api {
            case .ollamaEmbed:
                return encodedResponse(OllamaEmbedResponse(result: result))
            case .ollamaEmbeddings:
                return encodedResponse(OllamaEmbeddingsResponse(result: result))
            case .openAIEmbeddings:
                return encodedResponse(OpenAIEmbeddingResponse(result: result))
            }
        } catch {
            return jsonResponse(
                [
                    "error": String(describing: error),
                    "backend": embeddingBackend.status.activeBackend,
                ],
                status: "500 Internal Server Error"
            )
        }
    }

    private func httpBody(from request: String) -> String? {
        guard let range = request.range(of: "\r\n\r\n") else {
            return nil
        }
        return String(request[range.upperBound...])
    }

    private func encodedResponse<T: Encodable>(_ value: T, status: String = "200 OK") -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.keyEncodingStrategy = .useDefaultKeys
        let data = (try? encoder.encode(value)) ?? Data("{}".utf8)
        return httpResponse(status: status, body: data)
    }

    private func jsonResponse(_ object: Any, status: String = "200 OK") -> Data {
        let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8)
        return httpResponse(status: status, body: data)
    }

    private func renderedGenerationResponse(
        _ report: GenerationAPIResponseRenderReport,
        status: String = "200 OK"
    ) -> Data {
        httpResponse(status: status, body: Data(report.body.utf8), contentType: report.contentType)
    }

    private func httpResponse(
        status: String,
        body: Data,
        contentType: String = "application/json"
    ) -> Data {
        var response = Data()
        let header = "HTTP/1.1 \(status)\r\n" +
            "Content-Type: \(contentType)\r\n" +
            "Access-Control-Allow-Origin: *\r\n" +
            "Access-Control-Allow-Methods: GET, POST, HEAD, OPTIONS\r\n" +
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" +
            "Content-Length: \(body.count)\r\n" +
            "Connection: close\r\n" +
            "\r\n"
        response.append(Data(header.utf8))
        response.append(body)
        return response
    }
}

enum ServerError: Error, CustomStringConvertible {
    case socketFailed(Int32)
    case bindFailed(Int32)
    case listenFailed(Int32)

    var description: String {
        switch self {
        case .socketFailed(let code):
            return "socket() failed with errno \(code)"
        case .bindFailed(let code):
            return "bind() failed with errno \(code)"
        case .listenFailed(let code):
            return "listen() failed with errno \(code)"
        }
    }
}

enum SelfTest {
    static func run() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-vlm-swift-self-test")
            .appendingPathComponent(UUID().uuidString)
        let model = root.appendingPathComponent("qwen2-vl")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data(qwen2VLConfig.utf8).write(to: model.appendingPathComponent("config.json"))
        try Data("{\"eos_token_id\":[151645,151643]}".utf8)
            .write(to: model.appendingPathComponent("generation_config.json"))
        try Data(tokenizerConfig.utf8)
            .write(to: model.appendingPathComponent("tokenizer_config.json"))
        try Data(tokenizerJSON.utf8)
            .write(to: model.appendingPathComponent("tokenizer.json"))
        try Data(processorConfig.utf8)
            .write(to: model.appendingPathComponent("processor_config.json"))
        try Data(preprocessorConfig.utf8)
            .write(to: model.appendingPathComponent("preprocessor_config.json"))
        try Data("{\"chat_template\":\"override-template\"}".utf8)
            .write(to: model.appendingPathComponent("chat_template.json"))
        try Data("{\"rank\":8,\"alpha\":16,\"dropout\":0.05}".utf8)
            .write(to: model.appendingPathComponent("adapter_config.json"))
        try makeSafetensorsFixture().write(to: model.appendingPathComponent("model.safetensors"))
        try makeSafetensorsFixture().write(to: model.appendingPathComponent("adapters.safetensors"))
        try Data(weightIndex.utf8).write(to: model.appendingPathComponent("model.safetensors.index.json"))

        let descriptor = try ModelStore().loadDescriptor(pathOrIdentifier: model.path)
        precondition(descriptor.canonicalModelType == "qwen2_vl")
        precondition(descriptor.hasChatTemplate)
        precondition(descriptor.tokenizerMetadata.chatTemplateSource == "chat_template.json")
        precondition(descriptor.tokenizerMetadata.chatTemplate == "override-template")
        precondition(descriptor.configNormalization.insertedEmptyTextConfig)
        precondition(!descriptor.configNormalization.insertedEmptyVisionConfig)
        precondition(descriptor.configNormalization.insertedEmptyAudioConfig)
        precondition(descriptor.configNormalization.visionConfigSource == "vision_config")
        precondition(descriptor.configNormalization.textConfigSource == "empty")
        precondition(descriptor.configNormalization.normalizedTopLevelKeys.contains("text_config"))
        precondition(descriptor.configNormalization.normalizedTopLevelKeys.contains("audio_config"))
        let normalizedQwenConfig = try ModelStore().loadNormalizedConfig(pathOrIdentifier: model.path)
        precondition(normalizedQwenConfig["text_config"]?.objectValue != nil)
        precondition(normalizedQwenConfig["audio_config"]?.objectValue != nil)
        precondition(normalizedQwenConfig["vision_config"]?.objectValue != nil)
        let chatTemplatePlan = ChatTemplatePlanner().plan(descriptor: descriptor)
        precondition(chatTemplatePlan.hasTemplate)
        precondition(chatTemplatePlan.source == "chat_template.json")
        precondition(chatTemplatePlan.requiredRenderer == "custom-template")
        precondition(!chatTemplatePlan.canRenderNatively)
        precondition(!chatTemplatePlan.warnings.isEmpty)
        let llamaTemplateModel = root.appendingPathComponent("llama-template")
        try FileManager.default.createDirectory(at: llamaTemplateModel, withIntermediateDirectories: true)
        try Data("""
        {"model_type":"llava","vocab_size":10}
        """.utf8).write(to: llamaTemplateModel.appendingPathComponent("config.json"))
        try Data("""
        {"chat_template":"{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\\n\\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\\n\\n{% endif %}"}
        """.utf8).write(to: llamaTemplateModel.appendingPathComponent("tokenizer_config.json"))
        let llamaTemplateDescriptor = try ModelStore().loadDescriptor(pathOrIdentifier: llamaTemplateModel.path)
        let llamaTemplatePlan = ChatTemplatePlanner().plan(descriptor: llamaTemplateDescriptor)
        precondition(llamaTemplatePlan.requiredRenderer == "llama3-chat-builtin")
        precondition(llamaTemplatePlan.canRenderNatively)
        let llamaPreflight = GenerationPreflightPlanner(descriptor: llamaTemplateDescriptor).plan(
            request: GenerationRequest(
                model: "llama-template",
                messages: [
                    ChatMessage(role: .system, content: [.text("sys")]),
                    ChatMessage(role: .user, content: [.text("hello")]),
                ]
            )
        )
        precondition(llamaPreflight.promptStyle == .llama3Chat)
        precondition(llamaPreflight.prompt.contains("<|start_header_id|>system<|end_header_id|>"))
        precondition(llamaPreflight.prompt.contains("<|eot_id|><|start_header_id|>user<|end_header_id|>"))
        precondition(llamaPreflight.prompt.hasSuffix("<|start_header_id|>assistant<|end_header_id|>\n\n"))
        let mistralTemplateModel = root.appendingPathComponent("mistral-template")
        try FileManager.default.createDirectory(at: mistralTemplateModel, withIntermediateDirectories: true)
        try Data("""
        {"model_type":"mistral3","vocab_size":10}
        """.utf8).write(to: mistralTemplateModel.appendingPathComponent("config.json"))
        try Data("""
        {"chat_template":"{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %} {{ message['content'] }}</s>{% endif %}{% endfor %}"}
        """.utf8).write(to: mistralTemplateModel.appendingPathComponent("tokenizer_config.json"))
        let mistralTemplateDescriptor = try ModelStore().loadDescriptor(pathOrIdentifier: mistralTemplateModel.path)
        let mistralTemplatePlan = ChatTemplatePlanner().plan(descriptor: mistralTemplateDescriptor)
        precondition(mistralTemplatePlan.requiredRenderer == "mistral-instruct-builtin")
        precondition(mistralTemplatePlan.canRenderNatively)
        let mistralPreflight = GenerationPreflightPlanner(descriptor: mistralTemplateDescriptor).plan(
            request: GenerationRequest(
                model: "mistral-template",
                messages: [
                    ChatMessage(role: .system, content: [.text("sys")]),
                    ChatMessage(role: .user, content: [.text("hello")]),
                    ChatMessage(role: .assistant, content: [.text("ok")]),
                ]
            )
        )
        precondition(mistralPreflight.promptStyle == .mistralInstruct)
        precondition(mistralPreflight.prompt == "[INST] sys\n\nhello [/INST] ok</s>")
        precondition(descriptor.configVocabSize == 151936)
        precondition(descriptor.tokenizerMetadata.tokenizerJSONMetadata?.isReadable == true)
        precondition(descriptor.tokenizerMetadata.tokenizerJSONMetadata?.modelType == "BPE")
        precondition(descriptor.tokenizerMetadata.tokenizerJSONMetadata?.vocabCount == 3)
        precondition(descriptor.tokenizerMetadata.tokenizerJSONMetadata?.mergeCount == 1)
        precondition(descriptor.hasQuantization)
        precondition(descriptor.quantizationMetadata?.source == "quantization")
        precondition(descriptor.quantizationMetadata?.bits == 4)
        precondition(descriptor.quantizationMetadata?.groupSize == 64)
        precondition(descriptor.quantizationMetadata?.mode == "mlx")
        precondition(descriptor.adapterMetadata.hasAdapterConfig)
        precondition(descriptor.adapterMetadata.hasAdapterWeights)
        precondition(descriptor.adapterMetadata.rank == 8)
        precondition(descriptor.adapterMetadata.alpha == 16)
        precondition(descriptor.adapterMetadata.dropout == 0.05)
        precondition(descriptor.adapterMetadata.isLoRA)
        precondition(descriptor.adapterMetadata.isLoadable)
        precondition(descriptor.adapterMetadata.weightMetadata?.isReadable == true)
        precondition(descriptor.processorMetadata.hasProcessorConfig)
        precondition(descriptor.processorMetadata.hasPreprocessorConfig)
        precondition(descriptor.processorMetadata.image?.source == "preprocessor_config.json+processor_config.json:image_processor")
        precondition(descriptor.processorMetadata.image?.patchSize == 14)
        precondition(descriptor.processorMetadata.image?.mergeSize == 2)
        precondition(descriptor.processorMetadata.image?.minPixels == 56 * 56)
        precondition(descriptor.processorMetadata.image?.maxPixels == 14 * 14 * 4 * 1280)
        guard let tokenizerCatalog = TokenizerCatalogBuilder().catalog(for: descriptor) else {
            fatalError("Expected tokenizer catalog")
        }
        precondition(tokenizerCatalog.modelType == "BPE")
        precondition(tokenizerCatalog.tokenCount == 7)
        precondition(tokenizerCatalog.specialTokenCount == 4)
        precondition(tokenizerCatalog.merges == [TokenizerCatalogMerge(left: "hello", right: "world", rank: 0)])
        precondition(tokenizerCatalog.id(for: "hello") == 0)
        precondition(tokenizerCatalog.id(for: "<|image_pad|>") == 151655)
        precondition(tokenizerCatalog.token(for: 151656) == "<|video_pad|>")
        precondition(tokenizerCatalog.duplicateIDs.isEmpty)
        let tokenizerPlan = TokenizerImplementationPlanner().plan(
            descriptor: descriptor,
            catalog: tokenizerCatalog
        )
        precondition(tokenizerPlan.requiredBackend == "tokenizers-json-bpe")
        precondition(tokenizerPlan.canUseCatalogPreflight)
        precondition(!tokenizerPlan.swiftExecutionSupported)
        precondition(tokenizerPlan.swiftExecutionMode == nil)
        precondition(tokenizerPlan.requiresFullTokenizerImplementation)
        precondition(tokenizerPlan.catalogTokenCount == 7)
        precondition(tokenizerPlan.mergeCount == 1)
        precondition(tokenizerPlan.preTokenizerType == "ByteLevel")
        let bpeSimpleTokenization = SimpleTokenizer(
            catalog: tokenizerCatalog,
            plan: tokenizerPlan
        ).tokenize("hello world <|image_pad|>")
        precondition(bpeSimpleTokenization.supported)
        precondition(bpeSimpleTokenization.tokenIDs == [0, 1, 151655])
        precondition(bpeSimpleTokenization.unknownTokens.isEmpty)
        let tokenizationPreflight = TokenizationPreflightPlanner(catalog: tokenizerCatalog)
            .plan(prompt: "<|im_start|>user\nhello<|image_pad|><|im_end|>")
        precondition(tokenizationPreflight.tokenIDs == [151644, 2, 0, 151655, 151645])
        precondition(tokenizationPreflight.unknownFragmentCount == 1)
        precondition(tokenizationPreflight.requiresTokenizerImplementation)
        precondition(descriptor.tokenizerMetadata.imageTokenID == 151655)
        precondition(descriptor.tokenizerMetadata.videoTokenID == 151656)
        precondition(descriptor.weightFiles.count == 1)
        precondition(!descriptor.weightFiles.contains { $0.name == "adapters.safetensors" })
        precondition(descriptor.weightIndex?.tensorCount == 2)
        precondition(descriptor.weightIndex?.shardNames == ["model.safetensors"])
        precondition(descriptor.safetensorsMetadata.count == 1)
        precondition(descriptor.safetensorsMetadata[0].isReadable)
        precondition(descriptor.safetensorsMetadata[0].tensorCount == 2)
        precondition(descriptor.safetensorsMetadata[0].dtypes == ["F16"])
        guard let weightReport = QwenVLWeightSanitizer.report(descriptor: descriptor) else {
            fatalError("Expected weight report")
        }
        precondition(weightReport.languageModelCount == 1)
        precondition(weightReport.visionTowerCount == 1)
        precondition(weightReport.unknownCount == 0)
        let weightCatalog = WeightCatalogBuilder().catalog(for: descriptor)
        precondition(weightCatalog.tensorCount == 2)
        precondition(weightCatalog.indexedTensorCount == 2)
        precondition(weightCatalog.missingIndexEntries.isEmpty)
        precondition(weightCatalog.unreadableShards.isEmpty)
        precondition(weightCatalog.duplicateOriginalKeys.isEmpty)
        precondition(weightCatalog.duplicateSanitizedKeys.isEmpty)
        precondition(weightCatalog.roleCounts[QwenVLWeightRole.languageModel.rawValue] == 1)
        precondition(weightCatalog.roleCounts[QwenVLWeightRole.visionTower.rawValue] == 1)
        precondition(weightCatalog.tensors.contains { $0.sanitizedKey == "language_model.model.embed_tokens.weight" && $0.dtype == "F16" && $0.shape == [2, 1] })
        let duplicateWeightModel = root.appendingPathComponent("duplicate-weights")
        try FileManager.default.createDirectory(at: duplicateWeightModel, withIntermediateDirectories: true)
        try Data("""
        {"model_type":"qwen2_vl","vision_config":{},"vocab_size":10}
        """.utf8).write(to: duplicateWeightModel.appendingPathComponent("config.json"))
        try makeSafetensorsFixture().write(to: duplicateWeightModel.appendingPathComponent("model-00001-of-00002.safetensors"))
        try makeSafetensorsFixture().write(to: duplicateWeightModel.appendingPathComponent("model-00002-of-00002.safetensors"))
        let duplicateWeightDescriptor = try ModelStore().loadDescriptor(pathOrIdentifier: duplicateWeightModel.path)
        let duplicateWeightCatalog = WeightCatalogBuilder().catalog(for: duplicateWeightDescriptor)
        precondition(duplicateWeightCatalog.duplicateOriginalKeys.contains("model.embed_tokens.weight"))
        precondition(duplicateWeightCatalog.duplicateSanitizedKeys.contains("language_model.model.embed_tokens.weight"))
        let duplicateWeightCompatibility = ModelCompatibilityValidator.validate(descriptor: duplicateWeightDescriptor)
        precondition(duplicateWeightCompatibility.checks.contains { $0.id == "weight-catalog-unique" && !$0.passed })
        let qwenConfig = try QwenVLModelConfig.load(fromModelDirectory: model.path)
        precondition(QwenVLWeightSanitizer.sanitize("visual.patch_embed.proj.weight").sanitizedKey == "vision_tower.patch_embed.proj.weight")
        precondition(QwenVLWeightSanitizer.sanitize("model.embed_tokens.weight").sanitizedKey == "language_model.model.embed_tokens.weight")
        precondition(QwenVLWeightSanitizer.sanitize("lm_head.weight").sanitizedKey == "language_model.lm_head.weight")
        precondition(QwenVLWeightSanitizer.sanitize("vision_tower.blocks.0.attn.qkv.weight").role == .visionTower)
        let architecturePlan = QwenVLArchitecturePlanner().plan(config: qwenConfig, weightCatalog: weightCatalog)
        precondition(architecturePlan.family == .qwen2VL)
        precondition(architecturePlan.textLayerCount == 28)
        precondition(architecturePlan.visionBlockCount == 32)
        precondition(architecturePlan.presentCoreTensorCount == 2)
        precondition(architecturePlan.missingRequiredKeys.contains("language_model.model.norm.weight"))
        precondition(architecturePlan.mismatchedShapeKeys.contains("language_model.model.embed_tokens.weight"))
        let mlxBindingPlan = MLXBackendBindingPlanner().plan(descriptor: descriptor)
        precondition(mlxBindingPlan.supportedBySwiftScaffold)
        precondition(mlxBindingPlan.totalExpectedTensorBindings == architecturePlan.expectedCoreTensors.count)
        precondition(mlxBindingPlan.presentTensorBindings == 2)
        precondition(mlxBindingPlan.phaseCounts["token-embedding"] == 1)
        precondition(mlxBindingPlan.phaseCounts["vision-patch-embedding"] == 1)
        precondition(mlxBindingPlan.missingRequiredKeys.contains("language_model.model.norm.weight"))
        precondition(mlxBindingPlan.blockingReasons.contains("MLX arrays, module construction, and generation sampling are not implemented yet."))
        let weightDataCatalog = WeightDataCatalogBuilder().catalog(for: descriptor)
        precondition(weightDataCatalog.tensorCount == 2)
        precondition(weightDataCatalog.unreadableTensorCount == 0)
        precondition(weightDataCatalog.byteMismatchCount == 0)
        precondition(weightDataCatalog.totalReadableBytes == 8)
        precondition(weightDataCatalog.tensors.allSatisfy { $0.dtypeByteWidth == 2 && $0.byteCount == 4 && $0.byteCountMatchesShape == true })
        let mlxWeightLoadPlan = MLXWeightLoadPlan(descriptor: descriptor)
        precondition(mlxWeightLoadPlan.tensorCount == 2)
        precondition(mlxWeightLoadPlan.loadableTensorCount == 2)
        precondition(mlxWeightLoadPlan.totalLoadableBytes == 8)
        precondition(mlxWeightLoadPlan.canLoadAllTensorsAsMLXArrays)
        precondition(mlxWeightLoadPlan.mlxDTypeCounts["float16"] == 2)
        let weightPreview = try WeightDataCatalogBuilder().previewTensor(
            named: "language_model.model.embed_tokens.weight",
            descriptor: descriptor,
            maxBytes: 4
        )
        precondition(weightPreview.hexPrefix == "003c0040")
        precondition(weightPreview.returnedByteCount == 4)
        precondition(weightPreview.numericValues == [1.0, 2.0])
        let weightPayload = try WeightDataCatalogBuilder().readTensorPayload(
            named: "language_model.model.embed_tokens.weight",
            descriptor: descriptor
        )
        precondition(weightPayload.byteCount == 4)
        precondition(weightPayload.checksum == 124)
        let preparedOne = try MLXWeightPreparer().prepare(
            descriptor: descriptor,
            options: MLXWeightPreparationOptions(
                tensorNames: ["language_model.model.embed_tokens.weight"],
                maxTensorCount: 1,
                maxTotalBytes: 4
            )
        )
        precondition(preparedOne.tensors.count == 1)
        precondition(preparedOne.summary.tensorCount == 1)
        precondition(preparedOne.summary.totalByteCount == 4)
        precondition(preparedOne.summary.tensors[0].checksum == 124)
        precondition(preparedOne.summary.tensors[0].mlxDType == "float16")
        let preparedAll = try MLXWeightPreparer().prepare(
            descriptor: descriptor,
            options: MLXWeightPreparationOptions(maxTensorCount: 2, maxTotalBytes: 8)
        )
        precondition(preparedAll.tensors.count == 2)
        precondition(preparedAll.summary.totalByteCount == 8)
        precondition(preparedAll.summary.safetensorsDTypeCounts["F16"] == 2)
        precondition(preparedAll.summary.mlxDTypeCounts["float16"] == 2)
        let skippedWeights = try MLXWeightPreparer().prepare(
            descriptor: descriptor,
            options: MLXWeightPreparationOptions(skipTensorPayloads: true)
        )
        precondition(skippedWeights.tensors.isEmpty)
        precondition(skippedWeights.summary.totalByteCount == 0)
        let mlxBackendLoadPreflight = MLXBackendFactory.preflightLoad(
            descriptor: descriptor,
            weightOptions: MLXWeightPreparationOptions(maxTensorCount: 2, maxTotalBytes: 8)
        )
        precondition(mlxBackendLoadPreflight.metadataReady)
        precondition(mlxBackendLoadPreflight.preparedWeights?.tensorCount == 2)
        precondition(mlxBackendLoadPreflight.preparedWeights?.totalByteCount == 8)
        precondition(mlxBackendLoadPreflight.canPrepareWeightPayloads)
        let realMLXAPICompiled = MLXRuntimeProbe.detectRealMLXAPIImplementationCompiled()
        precondition(mlxBackendLoadPreflight.arrayLoadReport?.realMLXAPIImplementationCompiled == realMLXAPICompiled)
        precondition(mlxBackendLoadPreflight.arrayLoadReport?.attemptedTensorCount == 2)
        if realMLXAPICompiled {
            precondition(mlxBackendLoadPreflight.arrayLoadReport?.loadedArrayCount == 2)
            precondition(mlxBackendLoadPreflight.arrayLoadReport?.loadedKeys.contains("language_model.model.embed_tokens.weight") == true)
            precondition(mlxBackendLoadPreflight.canCreateMLXArrays)
        } else {
            precondition(mlxBackendLoadPreflight.arrayLoadReport?.loadedArrayCount == 0)
            precondition(!mlxBackendLoadPreflight.canCreateMLXArrays)
            precondition(mlxBackendLoadPreflight.blockingReasons.contains("MLXArray creation from safetensors payload bytes is not available yet."))
        }
        precondition(!mlxBackendLoadPreflight.canInstantiateModelModules)
        precondition(!mlxBackendLoadPreflight.canRunGeneration)
        let mlxContainer = MLXBackendFactory.loadWeightBackedContainer(
            descriptor: descriptor,
            weightOptions: MLXWeightPreparationOptions(maxTensorCount: 2, maxTotalBytes: 8)
        )
        precondition(mlxContainer.context.descriptor.id == descriptor.id)
        precondition(mlxContainer.summary.metadataReady)
        precondition(mlxContainer.summary.preparedWeightTensorCount == 2)
        precondition(mlxContainer.summary.loadedArrayCount == (realMLXAPICompiled ? 2 : 0))
        precondition(mlxContainer.summary.arrayBacked == realMLXAPICompiled)
        precondition(!mlxContainer.summary.moduleInstantiationReady)
        precondition(!mlxContainer.summary.generationReady)
        let mlxModulePlan = MLXQwenVLModuleConstructionPlanner().plan(container: mlxContainer)
        precondition(mlxModulePlan.totalBindingCount == mlxBindingPlan.totalExpectedTensorBindings)
        precondition(mlxModulePlan.loadedArrayCount == (realMLXAPICompiled ? 2 : 0))
        precondition(mlxModulePlan.groups.contains { $0.phase == .tokenEmbedding && $0.requiredBindingCount == 1 })
        precondition(mlxModulePlan.groups.contains { $0.phase == .visionPatchEmbedding && $0.requiredBindingCount == 1 })
        precondition(!mlxModulePlan.moduleConstructionReady)
        precondition(mlxModulePlan.blockingReasons.contains("Qwen VL module construction is not ready yet."))
        let forwardRequest = GenerationRequest(
            model: "verify",
            messages: [ChatMessage(role: .user, content: [.text("hello")])]
        )
        let forwardInput = try CompatibilityProcessor().process(
            request: forwardRequest,
            context: mlxContainer.context
        )
        let forwardPlan = MLXQwenVLForwardPlanner().plan(
            container: mlxContainer,
            modulePlan: mlxModulePlan,
            input: forwardInput
        )
        precondition(forwardPlan.vocabSize == 151936)
        precondition(forwardPlan.promptTokenCount > 0)
        precondition(forwardPlan.inputIDsShape == [1, forwardPlan.promptTokenCount])
        precondition(forwardPlan.logitsShape == [1, forwardPlan.promptTokenCount, 151936])
        precondition(!forwardPlan.forwardReady)
        precondition(!forwardPlan.nextTokenSelectionReady)
        precondition(forwardPlan.blockingReasons.contains("Qwen VL modules must be constructed before forward can run."))
        let generationLoopPlan = MLXQwenVLGenerationLoopPlanner().plan(
            forwardPlan: forwardPlan,
            input: forwardInput
        )
        precondition(generationLoopPlan.promptTokenCount == forwardPlan.promptTokenCount)
        precondition(generationLoopPlan.maxCompletionTokens == forwardInput.runtime.maxCompletionTokens)
        precondition(generationLoopPlan.sampler == forwardInput.runtime.sampling.sampler)
        precondition(generationLoopPlan.generateParameters.sourceAPI == "mlx-swift-lm GenerateParameters")
        precondition(generationLoopPlan.generateParameters.maxTokens == forwardInput.runtime.maxCompletionTokens)
        precondition(generationLoopPlan.generateParameters.prefillStepSize == 512)
        precondition(generationLoopPlan.generateParameters.kvGroupSize == 64)
        precondition(generationLoopPlan.generateParameters.canInstantiateGenerateParameters)
        let generateParametersBridge = MLXGenerateParametersBridge().report(
            for: generationLoopPlan.generateParameters
        )
        precondition(generateParametersBridge.sourceAPI == "mlx-swift-lm GenerateParameters")
        if MLXRuntimeProbe().realMLXAPIImplementationCompiled {
            precondition(generateParametersBridge.canBridgeToGenerateParameters)
        } else {
            precondition(!generateParametersBridge.canBridgeToGenerateParameters)
        }
        precondition(generationLoopPlan.assemblerHandoffReady)
        precondition(!generationLoopPlan.generationLoopReady)
        precondition(generationLoopPlan.blockingReasons.contains("MLX-backed decode loop is not implemented yet."))
        let decodeStatePlan = MLXQwenVLDecodeStatePlanner().plan(
            container: mlxContainer,
            forwardPlan: forwardPlan,
            generationLoopPlan: generationLoopPlan,
            input: forwardInput
        )
        precondition(decodeStatePlan.promptTokenCount == forwardPlan.promptTokenCount)
        precondition(decodeStatePlan.initialPosition == forwardPlan.promptTokenCount)
        precondition(decodeStatePlan.nextTokenInputShape == [1, 1])
        precondition(decodeStatePlan.eosTokenIDs == [151645, 151643])
        precondition(decodeStatePlan.stopConditionOrder.contains("eos-token"))
        precondition(decodeStatePlan.stopConditionOrder.last == "max-completion-tokens")
        precondition(decodeStatePlan.canInitializeDecodeState)
        let pipelineReport = try MLXGenerationPipelineReporter().report(
            descriptor: descriptor,
            request: forwardRequest,
            weightOptions: MLXWeightPreparationOptions(maxTensorCount: 2, maxTotalBytes: 8)
        )
        precondition(pipelineReport.requestModel == "verify")
        precondition(pipelineReport.container.preparedWeightTensorCount == 2)
        precondition(pipelineReport.forward.logitsShape == [1, forwardPlan.promptTokenCount, 151936])
        precondition(pipelineReport.decodeState.initialPosition == forwardPlan.promptTokenCount)
        precondition(!pipelineReport.pipelineReady)
        precondition(pipelineReport.blockingReasons.contains("MLX-backed decode loop is not implemented yet."))
        let serverPipelineDiagnostics = ServerMLXGenerationDiagnostics(report: pipelineReport)
        precondition(!serverPipelineDiagnostics.pipelineReady)
        precondition(serverPipelineDiagnostics.preparedWeightTensorCount == 2)
        precondition(serverPipelineDiagnostics.loadedArrayCount == (realMLXAPICompiled ? 2 : 0))
        precondition(serverPipelineDiagnostics.canBridgeToGenerateParameters == realMLXAPICompiled)
        precondition(serverPipelineDiagnostics.canReferenceVLMModelFactory == realMLXAPICompiled)
        precondition(serverPipelineDiagnostics.canLoadLocalModelContainer == realMLXAPICompiled)
        let serverUnavailableReport = ServerGenerationUnavailableReport(
            base: CompatibilityGenerationEngine(descriptor: descriptor).unavailableReport(for: forwardRequest),
            mlxPipeline: serverPipelineDiagnostics
        )
        precondition(serverUnavailableReport.mlxPipeline?.generationLoopReady == false)
        precondition(serverUnavailableReport.preflight.plainPrompt.contains("hello"))
        let lightweightServerPipeline = try MLXGenerationPipelineReporter().report(
            descriptor: descriptor,
            request: forwardRequest,
            weightOptions: MLXWeightPreparationOptions(skipTensorPayloads: true)
        )
        precondition(lightweightServerPipeline.container.preparedWeightTensorCount == 0)
        precondition(!lightweightServerPipeline.pipelineReady)
        precondition(lightweightServerPipeline.forward.promptTokenCount == forwardPlan.promptTokenCount)

        precondition(qwenConfig.family == .qwen2VL)
        precondition(qwenConfig.eosTokenIDs == [151645, 151643])
        precondition(qwenConfig.textConfig.numKeyValueHeads == 8)
        precondition(qwenConfig.textConfig.maxPositionEmbeddings == 40960)
        precondition(qwenConfig.visionConfig.spatialMergeSize == 2)

        let qwen25 = try QwenVLModelConfig(json: try JSONDecoder().decode(JSONValue.self, from: Data(qwen25VLConfig.utf8)))
        precondition(qwen25.family == .qwen25VL)
        precondition(qwen25.textConfig.numKeyValueHeads == qwen25.textConfig.numAttentionHeads)
        precondition(qwen25.visionConfig.fullattBlockIndexes == [7, 15, 23, 31])

        let wordModel = root.appendingPathComponent("wordlevel")
        try FileManager.default.createDirectory(at: wordModel, withIntermediateDirectories: true)
        try Data("""
        {"model_type":"wordlevel_test","vocab_size":5}
        """.utf8).write(to: wordModel.appendingPathComponent("config.json"))
        try Data("""
        {
          "model": {
            "type": "WordLevel",
            "unk_token": "[UNK]",
            "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "<image>": 3}
          },
          "added_tokens": [
            {"id": 3, "content": "<image>", "special": true}
          ],
          "pre_tokenizer": {"type": "Whitespace"}
        }
        """.utf8).write(to: wordModel.appendingPathComponent("tokenizer.json"))
        let wordDescriptor = try ModelStore().loadDescriptor(pathOrIdentifier: wordModel.path)
        guard let wordCatalog = TokenizerCatalogBuilder().catalog(for: wordDescriptor) else {
            fatalError("Expected WordLevel tokenizer catalog")
        }
        let wordPlan = TokenizerImplementationPlanner().plan(descriptor: wordDescriptor, catalog: wordCatalog)
        precondition(wordPlan.requiredBackend == "tokenizers-json-wordlevel")
        precondition(!wordPlan.requiresFullTokenizerImplementation)
        precondition(wordCatalog.unknownToken == "[UNK]")
        precondition(wordCatalog.unknownTokenID == 0)
        let wordTokens = SimpleTokenizer(catalog: wordCatalog, plan: wordPlan).tokenize("hello missing <image> world")
        precondition(wordTokens.supported)
        precondition(wordTokens.tokens == ["hello", "missing", "<image>", "world"])
        precondition(wordTokens.tokenIDs == [1, 0, 3, 2])
        precondition(wordTokens.unknownTokens == ["missing"])

        let byteBPEModel = root.appendingPathComponent("bytelevel-bpe")
        try FileManager.default.createDirectory(at: byteBPEModel, withIntermediateDirectories: true)
        try Data("""
        {"model_type":"bytelevel_bpe_test","vocab_size":5}
        """.utf8).write(to: byteBPEModel.appendingPathComponent("config.json"))
        try Data("""
        {
          "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "hello": 10, "\\u0120world": 11, "!": 12, "\\u0120": 13}
          },
          "added_tokens": [
            {"id": 14, "content": "<|end|>", "special": true}
          ],
          "pre_tokenizer": {"type": "ByteLevel"},
          "decoder": {"type": "ByteLevel"}
        }
        """.utf8).write(to: byteBPEModel.appendingPathComponent("tokenizer.json"))
        let byteBPEDescriptor = try ModelStore().loadDescriptor(pathOrIdentifier: byteBPEModel.path)
        guard let byteBPECatalog = TokenizerCatalogBuilder().catalog(for: byteBPEDescriptor) else {
            fatalError("Expected ByteLevel BPE tokenizer catalog")
        }
        let byteBPEPlan = TokenizerImplementationPlanner().plan(descriptor: byteBPEDescriptor, catalog: byteBPECatalog)
        precondition(byteBPEPlan.requiredBackend == "tokenizers-json-bpe")
        precondition(byteBPEPlan.preTokenizerType == "ByteLevel")
        precondition(byteBPEPlan.swiftExecutionSupported)
        precondition(byteBPEPlan.swiftExecutionMode == "bytelevel-bpe")
        precondition(!byteBPEPlan.requiresFullTokenizerImplementation)
        precondition(byteBPEPlan.blockingReasons.isEmpty)
        let byteTokenizer = SimpleTokenizer(catalog: byteBPECatalog, plan: byteBPEPlan)
        let byteTokens = byteTokenizer.tokenize("hello world! <|end|>")
        precondition(byteTokens.supported)
        precondition(byteTokens.tokens == ["hello", "\u{0120}world", "!", "\u{0120}", "<|end|>"])
        precondition(byteTokens.tokenIDs == [10, 11, 12, 13, 14])
        precondition(byteTokens.unknownTokens.isEmpty)
        let byteText = byteTokenizer.detokenize([10, 11, 12, 13, 14])
        precondition(byteText.supported)
        precondition(byteText.text == "hello world! <|end|>")
        let byteTextWithoutSpecial = byteTokenizer.detokenize([10, 11, 12, 13, 14], skipSpecialTokens: true)
        precondition(byteTextWithoutSpecial.text == "hello world! ")
        var tokenTextDecoder = GenerationTokenTextDecoder(tokenizer: byteTokenizer)
        let decodedSteps = [10, 11, 12, 13, 14].map { tokenTextDecoder.append($0) }
        precondition(decodedSteps.map(\.textDelta) == ["hello", " world", "!", " ", ""])
        precondition(decodedSteps.last?.decodedText == "hello world! ")
        precondition(decodedSteps.last?.skippedSpecialToken == true)
        precondition(decodedSteps.last?.decodeToken.text == "")
        var logitsForHello = Array(repeating: 0.0, count: 15)
        logitsForHello[10] = 5.0
        var logitsForWorld = Array(repeating: 0.0, count: 15)
        logitsForWorld[11] = 5.0
        var logitsForBang = Array(repeating: 0.0, count: 15)
        logitsForBang[12] = 5.0
        let logitsDecodeReport = try GenerationLogitsDecodeExecutor(
            model: "bytelevel",
            promptTokenCount: 2,
            maxCompletionTokens: 3,
            samplingPlan: GenerationSamplingPlanner().plan(parameters: GenerationParameters()),
            tokenizer: byteTokenizer
        ).run(logitsRows: [logitsForHello, logitsForWorld, logitsForBang])
        precondition(logitsDecodeReport.completed.text == "hello world!")
        precondition(logitsDecodeReport.completed.finishReason == "length")
        precondition(logitsDecodeReport.steps.map(\.sampledToken.tokenID) == [10, 11, 12])
        precondition(logitsDecodeReport.processedLogitRows == 3)
        let bytePreflight = GenerationPreflightPlanner(descriptor: byteBPEDescriptor).plan(
            request: GenerationRequest(
                model: "bytelevel",
                messages: [ChatMessage(role: .user, content: [.text("hello world!")])]
            )
        )
        precondition(bytePreflight.tokenization?.tokenIDs == [10, 11, 12])
        precondition(bytePreflight.tokenization?.requiresTokenizerImplementation == false)
        let runnerResult = try waitForAsync {
            try await VLMGenerationRunner(
                backend: SelfTestStreamingBackend(
                    descriptor: byteBPEDescriptor,
                    chunks: [
                        GenerationChunk(text: "he", tokenID: 21),
                        GenerationChunk(text: "llo", tokenID: 22),
                        GenerationChunk(text: "ENDtail", tokenID: 23),
                    ]
                )
            ).completedGeneration(
                for: GenerationRequest(
                    model: "bytelevel",
                    messages: [ChatMessage(role: .user, content: [.text("hello world!")])],
                    parameters: GenerationParameters(stopSequences: ["END"])
                )
            )
        }
        precondition(runnerResult.text == "hello")
        precondition(runnerResult.finishReason == "stop")
        precondition(runnerResult.usage.promptTokens == 3)
        precondition(runnerResult.usage.completionTokens == 3)
        let runnerRendered = try waitForAsync {
            try await VLMGenerationRunner(
                backend: SelfTestStreamingBackend(
                    descriptor: byteBPEDescriptor,
                    chunks: [
                        GenerationChunk(text: "he", tokenID: 21),
                        GenerationChunk(text: "llo", tokenID: 22),
                    ]
                )
            ).renderedResponse(
                for: GenerationRequest(
                    model: "bytelevel",
                    messages: [ChatMessage(role: .user, content: [.text("hello world!")])],
                    stream: true
                ),
                api: .openAIChat
            )
        }
        precondition(runnerRendered.collection.text == "hello")
        precondition(runnerRendered.collection.promptTokens == 3)
        precondition(runnerRendered.collection.completionTokens == 2)
        precondition(runnerRendered.response.contentType == "text/event-stream")
        precondition(runnerRendered.response.body.contains("data: [DONE]"))

        let sidecarBPEModel = root.appendingPathComponent("sidecar-bpe")
        try FileManager.default.createDirectory(at: sidecarBPEModel, withIntermediateDirectories: true)
        try Data("""
        {"model_type":"sidecar_bpe_test","vocab_size":6}
        """.utf8).write(to: sidecarBPEModel.appendingPathComponent("config.json"))
        try Data("""
        {"<unk>":0,"hello":10,"\\u0120world":11,"!":12,"\\u0120":13}
        """.utf8).write(to: sidecarBPEModel.appendingPathComponent("vocab.json"))
        try Data("""
        #version: 0.2
        \\u0120 world
        """.utf8).write(to: sidecarBPEModel.appendingPathComponent("merges.txt"))
        try Data("""
        {
          "unk_token": "<unk>",
          "added_tokens_decoder": {
            "14": {"content": "<|end|>", "special": true}
          }
        }
        """.utf8).write(to: sidecarBPEModel.appendingPathComponent("tokenizer_config.json"))
        let sidecarBPEDescriptor = try ModelStore().loadDescriptor(pathOrIdentifier: sidecarBPEModel.path)
        guard let sidecarBPECatalog = TokenizerCatalogBuilder().catalog(for: sidecarBPEDescriptor) else {
            fatalError("Expected sidecar BPE tokenizer catalog")
        }
        precondition(sidecarBPECatalog.modelType == "BPE")
        precondition(sidecarBPECatalog.id(for: "hello") == 10)
        precondition(sidecarBPECatalog.merges == [TokenizerCatalogMerge(left: "\\u0120", right: "world", rank: 0)])
        precondition(sidecarBPECatalog.unknownToken == "<unk>")
        precondition(sidecarBPECatalog.unknownTokenID == 0)
        precondition(sidecarBPECatalog.specialTokenCount == 2)
        precondition(sidecarBPECatalog.duplicateIDs.isEmpty)
        let sidecarBPEPlan = TokenizerImplementationPlanner().plan(descriptor: sidecarBPEDescriptor, catalog: sidecarBPECatalog)
        precondition(sidecarBPEPlan.requiredBackend == "bpe-vocab-json-merges-txt")
        precondition(sidecarBPEPlan.swiftExecutionSupported)
        precondition(sidecarBPEPlan.swiftExecutionMode == "bytelevel-bpe-sidecar")
        precondition(!sidecarBPEPlan.requiresFullTokenizerImplementation)
        let sidecarBPETokens = SimpleTokenizer(catalog: sidecarBPECatalog, plan: sidecarBPEPlan)
            .tokenize("hello world! <|end|>")
        precondition(sidecarBPETokens.tokenIDs == [10, 11, 12, 13, 14])
        let sidecarBPEText = SimpleTokenizer(catalog: sidecarBPECatalog, plan: sidecarBPEPlan)
            .detokenize([10, 11, 12])
        precondition(sidecarBPEText.text == "hello world!")

        let sidecarTokenizerModel = root.appendingPathComponent("sidecar-tokenizer")
        try FileManager.default.createDirectory(at: sidecarTokenizerModel, withIntermediateDirectories: true)
        try Data("""
        {"model_type":"llava","vocab_size":10}
        """.utf8).write(to: sidecarTokenizerModel.appendingPathComponent("config.json"))
        try Data("token 0\n".utf8).write(to: sidecarTokenizerModel.appendingPathComponent("tokenizer.tiktoken"))
        try Data("{\"hello\":0}".utf8).write(to: sidecarTokenizerModel.appendingPathComponent("vocab.json"))
        try Data("#version: 0.2\nh e\n".utf8).write(to: sidecarTokenizerModel.appendingPathComponent("merges.txt"))
        try Data("[UNK]\nhello\n".utf8).write(to: sidecarTokenizerModel.appendingPathComponent("vocab.txt"))
        let sidecarDescriptor = try ModelStore().loadDescriptor(pathOrIdentifier: sidecarTokenizerModel.path)
        precondition(sidecarDescriptor.tokenizerMetadata.hasTiktoken)
        precondition(sidecarDescriptor.tokenizerMetadata.hasVocabJSON)
        precondition(sidecarDescriptor.tokenizerMetadata.hasMergesTXT)
        precondition(sidecarDescriptor.tokenizerMetadata.hasVocabTXT)
        let sidecarPlan = TokenizerImplementationPlanner().plan(descriptor: sidecarDescriptor, catalog: nil)
        precondition(sidecarPlan.requiredBackend == "tiktoken-file")
        precondition(sidecarPlan.hasTiktoken)
        precondition(sidecarPlan.requiresFullTokenizerImplementation)
        let sidecarCompatibility = ModelCompatibilityValidator.validate(descriptor: sidecarDescriptor)
        precondition(sidecarCompatibility.checks.contains { $0.id == "tokenizer-present" && $0.passed })

        let llmConfigModel = root.appendingPathComponent("llm-config")
        try FileManager.default.createDirectory(at: llmConfigModel, withIntermediateDirectories: true)
        try Data("""
        {"model_type":"llava","llm_config":{"vocab_size":9}}
        """.utf8).write(to: llmConfigModel.appendingPathComponent("config.json"))
        let llmConfigDescriptor = try ModelStore().loadDescriptor(pathOrIdentifier: llmConfigModel.path)
        precondition(llmConfigDescriptor.configNormalization.usedLLMConfigAsTextConfig)
        precondition(llmConfigDescriptor.configNormalization.textConfigSource == "llm_config")
        precondition(llmConfigDescriptor.configNormalization.normalizedTopLevelKeys.contains("text_config"))
        precondition(!llmConfigDescriptor.configNormalization.normalizedTopLevelKeys.contains("llm_config"))
        precondition(llmConfigDescriptor.configNormalization.insertedEmptyVisionConfig)
        precondition(llmConfigDescriptor.configNormalization.insertedEmptyAudioConfig)
        let normalizedLLMConfig = try ModelStore().loadNormalizedConfig(pathOrIdentifier: llmConfigModel.path)
        precondition(normalizedLLMConfig["llm_config"] == nil)
        precondition(normalizedLLMConfig["text_config"]?["vocab_size"]?.intValue == 9)
        precondition(normalizedLLMConfig["vision_config"]?.objectValue != nil)
        precondition(normalizedLLMConfig["audio_config"]?.objectValue != nil)

        let dualConfigModel = root.appendingPathComponent("dual-config")
        try FileManager.default.createDirectory(at: dualConfigModel, withIntermediateDirectories: true)
        try Data("""
        {"model_type":"llava","text_config":{"vocab_size":7},"llm_config":{"vocab_size":9}}
        """.utf8).write(to: dualConfigModel.appendingPathComponent("config.json"))
        let normalizedDualConfig = try ModelStore().loadNormalizedConfig(pathOrIdentifier: dualConfigModel.path)
        precondition(normalizedDualConfig["llm_config"] == nil)
        precondition(normalizedDualConfig["text_config"]?["vocab_size"]?.intValue == 7)

        let expanded = try QwenVLPlaceholderExpander().expand(
            texts: ["A <|image_pad|> B <|video_pad|>"],
            imageGrids: [QwenVLGridTHW(temporal: 1, height: 28, width: 28)],
            videoGrids: [QwenVLGridTHW(temporal: 2, height: 14, width: 14)]
        )[0]
        precondition(expanded.filterImageTokens() == 196)
        precondition(expanded.filterVideoTokens() == 98)

        let placements = try QwenVLEmbeddingMergePlanner.plan(
            inputIDs: [[1, 151655, 2, 151655], [3, 4, 151655]],
            imageTokenID: 151655,
            videoTokenID: 151656,
            featureCount: 3
        )
        precondition(placements == [
            QwenVLFeaturePlacement(batchIndex: 0, tokenIndex: 1, featureIndex: 0),
            QwenVLFeaturePlacement(batchIndex: 0, tokenIndex: 3, featureIndex: 1),
            QwenVLFeaturePlacement(batchIndex: 1, tokenIndex: 2, featureIndex: 2),
        ])

        let videoPlacements = try QwenVLEmbeddingMergePlanner.plan(
            inputIDs: [[1, 151656, 2]],
            imageTokenID: 151655,
            videoTokenID: 151656,
            featureCount: 1
        )
        precondition(videoPlacements == [
            QwenVLFeaturePlacement(batchIndex: 0, tokenIndex: 1, featureIndex: 0)
        ])

        precondition(QwenVLImageGridPlanner.roundByFactor(100, factor: 28) == 112)
        precondition(QwenVLImageGridPlanner.ceilByFactor(57, factor: 28) == 84)
        precondition(QwenVLImageGridPlanner.floorByFactor(55, factor: 28) == 28)
        let imagePlan = try QwenVLImageGridPlanner.imagePlan(height: 224, width: 224)
        precondition(imagePlan.resizedHeight % 28 == 0)
        precondition(imagePlan.resizedWidth % 28 == 0)
        precondition(imagePlan.grid.height == imagePlan.resizedHeight / 14)
        precondition(imagePlan.placeholderTokenCount == imagePlan.grid.product / 4)

        let ollamaGenerate = try JSONDecoder().decode(
            OllamaGenerateRequest.self,
            from: Data("""
            {"model":"qwen","system":"You are concise.","prompt":"Describe this","suffix":"done","images":["base64-image"],"stream":false,"format":"json","template":"{{ .Prompt }}","raw":true,"context":[1,2,3],"keep_alive":"10m","options":{"num_predict":42,"temperature":0.2,"top_k":40,"min_p":0.05,"typical_p":0.95,"tfs_z":1.2,"seed":7,"num_ctx":4096,"num_keep":8,"stop":["</s>"],"repeat_penalty":1.1,"repeat_last_n":64,"presence_penalty":0.2,"frequency_penalty":0.3,"penalize_newline":false,"mirostat":2,"mirostat_tau":5.0,"mirostat_eta":0.1,"num_gpu":1,"main_gpu":0,"low_vram":true,"use_mmap":false}}
            """.utf8)
        )
        let normalizedGenerate = try ollamaGenerate.generationRequest(defaultModel: "default")
        precondition(normalizedGenerate.messages.count == 2)
        precondition(normalizedGenerate.messages[0].role == .system)
        precondition(normalizedGenerate.messages[1].role == .user)
        precondition(normalizedGenerate.metadata.responseFormat == .string("json"))
        precondition(normalizedGenerate.metadata.rawPrompt)
        precondition(normalizedGenerate.metadata.template == "{{ .Prompt }}")
        precondition(normalizedGenerate.metadata.suffix == "done")
        precondition(normalizedGenerate.metadata.legacyContext == [1, 2, 3])
        precondition(normalizedGenerate.metadata.rawOptions?["num_gpu"]?.intValue == 1)
        precondition(normalizedGenerate.metadata.rawOptions?["main_gpu"]?.intValue == 0)
        precondition(normalizedGenerate.metadata.rawOptions?["low_vram"]?.boolValue == true)
        precondition(normalizedGenerate.metadata.rawOptions?["use_mmap"]?.boolValue == false)
        let ollamaRendered = OllamaPromptRenderer().render(
            request: normalizedGenerate,
            defaultPrompt: "unused"
        )
        precondition(ollamaRendered.source == "ollama-template")
        precondition(ollamaRendered.appliedTemplate)
        precondition(ollamaRendered.appliedRawPrompt)
        precondition(ollamaRendered.prompt == "Describe this<|image_pad|>")
        let ollamaResponseFormatPlan = ResponseFormatPlanner().plan(
            metadata: normalizedGenerate.metadata,
            stream: normalizedGenerate.stream
        )
        precondition(ollamaResponseFormatPlan.source == "ollama-format")
        precondition(ollamaResponseFormatPlan.formatType == "json_object")
        precondition(ollamaResponseFormatPlan.requiresJSONMode)
        precondition(!ollamaResponseFormatPlan.requiresSchemaGuidance)
        let rawRendered = OllamaPromptRenderer().render(
            request: try OllamaGenerateRequest(
                model: "qwen",
                prompt: "raw text",
                suffix: "tail",
                raw: true
            ).generationRequest(defaultModel: "default"),
            defaultPrompt: "unused"
        )
        precondition(rawRendered.source == "ollama-raw")
        precondition(rawRendered.prompt == "raw texttail")
        precondition(rawRendered.appendedSuffix)
        let unloadAction = OllamaGenerateRequest(
            model: "qwen",
            prompt: "",
            keepAlive: .number(0)
        ).residencyAction(defaultModel: "default")
        precondition(unloadAction == OllamaResidencyAction(model: "qwen", action: .unload, keepAlive: "0"))
        let loadAction = OllamaGenerateRequest(
            model: "qwen",
            prompt: "",
            keepAlive: .string("5m")
        ).residencyAction(defaultModel: "default")
        precondition(loadAction == OllamaResidencyAction(model: "qwen", action: .load, keepAlive: "5m"))
        precondition(
            OllamaGenerateRequest(model: "qwen", prompt: "", images: ["image"], keepAlive: .number(0))
                .residencyAction(defaultModel: "default") == nil
        )
        precondition(normalizedGenerate.parameters.maxTokens == 42)
        precondition(normalizedGenerate.parameters.temperature == 0.2)
        precondition(normalizedGenerate.parameters.topK == 40)
        precondition(normalizedGenerate.parameters.minP == 0.05)
        precondition(normalizedGenerate.parameters.typicalP == 0.95)
        precondition(normalizedGenerate.parameters.tfsZ == 1.2)
        precondition(normalizedGenerate.parameters.seed == 7)
        precondition(normalizedGenerate.parameters.contextLength == 4096)
        precondition(normalizedGenerate.parameters.numKeep == 8)
        precondition(normalizedGenerate.parameters.keepAlive == "10m")
        precondition(normalizedGenerate.parameters.stopSequences == ["</s>"])
        precondition(normalizedGenerate.parameters.repetitionPenalty == 1.1)
        precondition(normalizedGenerate.parameters.repeatLastN == 64)
        precondition(normalizedGenerate.parameters.presencePenalty == 0.2)
        precondition(normalizedGenerate.parameters.frequencyPenalty == 0.3)
        precondition(normalizedGenerate.parameters.penalizeNewline == false)
        precondition(normalizedGenerate.parameters.mirostat == 2)
        precondition(normalizedGenerate.parameters.mirostatTau == 5.0)
        precondition(normalizedGenerate.parameters.mirostatEta == 0.1)
        let stopResult = StopSequenceMatcher(stopSequences: ["</s>", "END"])
            .truncate("helloENDtail")
        precondition(stopResult.text == "hello")
        precondition(stopResult.finishReason == "stop")
        precondition(stopResult.matchedStopSequence == "END")
        precondition(stopResult.matchedRange == [5, 8])
        let stopCompleted = CompletedGeneration(model: "qwen", text: "helloENDtail", finishReason: "length")
            .applyingStopSequences(["END"])
        precondition(stopCompleted.text == "hello")
        precondition(stopCompleted.finishReason == "stop")
        let noStopCompleted = CompletedGeneration(model: "qwen", text: "hello", finishReason: "length")
            .applyingStopSequences(["END"])
        precondition(noStopCompleted.text == "hello")
        precondition(noStopCompleted.finishReason == "length")
        var stopStream = StopSequenceStreamFilter(stopSequences: ["END"])
        let streamedStopText = [
            stopStream.append(GenerationChunk(text: "he")),
            stopStream.append(GenerationChunk(text: "lloE")),
            stopStream.append(GenerationChunk(text: "NDtail")),
        ].flatMap { $0 }
        precondition(streamedStopText.map(\.text).joined() == "hello")
        precondition(streamedStopText.last?.isFinished == true)
        precondition(streamedStopText.last?.finishReason == "stop")
        var assembler = GenerationOutputAssembler(
            model: "qwen",
            promptTokenCount: 4,
            stopSequences: ["END"]
        )
        precondition(assembler.append(GenerationChunk(text: "he", tokenID: 10)).isEmpty)
        let assembledMiddle = assembler.append(GenerationChunk(text: "llo", tokenID: 11))
        precondition(assembledMiddle.map(\.text).joined() == "hel")
        let assembledFinal = assembler.append(GenerationChunk(text: "ENDtail", tokenID: 12))
        precondition(assembledFinal.map(\.text).joined() == "lo")
        precondition(assembledFinal.last?.isFinished == true)
        let assembled = assembler.completedGeneration
        precondition(assembled.text == "hello")
        precondition(assembled.finishReason == "stop")
        precondition(assembled.usage.promptTokens == 4)
        precondition(assembled.usage.completionTokens == 3)
        var lengthAssembler = GenerationOutputAssembler(model: "qwen", promptTokenCount: 2)
        _ = lengthAssembler.append(GenerationChunk(text: "abc", tokenID: 3))
        _ = lengthAssembler.append(GenerationChunk(text: "", isFinished: true, finishReason: "length"))
        precondition(lengthAssembler.completedGeneration.text == "abc")
        precondition(lengthAssembler.completedGeneration.finishReason == "length")
        var decodeLoop = GenerationDecodeLoop(
            model: "qwen",
            promptTokenCount: 4,
            maxCompletionTokens: 4,
            eosTokenIDs: [99],
            stopSequences: ["END"]
        )
        let decodeReport = decodeLoop.run([
            GenerationDecodeToken(tokenID: 10, text: "he"),
            GenerationDecodeToken(tokenID: 11, text: "llo"),
            GenerationDecodeToken(tokenID: 12, text: "ENDtail"),
        ])
        precondition(decodeReport.completed.text == "hello")
        precondition(decodeReport.completed.finishReason == "stop")
        precondition(decodeReport.completed.usage.completionTokens == 3)
        precondition(decodeReport.steps.count == 3)
        var eosDecodeLoop = GenerationDecodeLoop(
            model: "qwen",
            promptTokenCount: 1,
            maxCompletionTokens: 4,
            eosTokenIDs: [2]
        )
        let eosReport = eosDecodeLoop.run([
            GenerationDecodeToken(tokenID: 1, text: "a"),
            GenerationDecodeToken(tokenID: 2, text: "</s>"),
            GenerationDecodeToken(tokenID: 3, text: "b"),
        ])
        precondition(eosReport.completed.text == "a")
        precondition(eosReport.completed.finishReason == "stop")
        precondition(eosReport.completed.usage.completionTokens == 2)
        precondition(eosReport.steps.count == 2)
        var lengthDecodeLoop = GenerationDecodeLoop(model: "qwen", promptTokenCount: 1, maxCompletionTokens: 2)
        let lengthReport = lengthDecodeLoop.run([
            GenerationDecodeToken(tokenID: 1, text: "a"),
            GenerationDecodeToken(tokenID: 2, text: "b"),
            GenerationDecodeToken(tokenID: 3, text: "c"),
        ])
        precondition(lengthReport.completed.text == "ab")
        precondition(lengthReport.completed.finishReason == "length")
        precondition(lengthReport.completed.usage.completionTokens == 2)
        let defaultParameters = GenerationParameters(
            maxTokens: 128,
            temperature: 0.7,
            topP: 0.8,
            topK: 32,
            minP: 0.03,
            seed: 123,
            contextLength: 8192,
            kvBits: 8,
            kvQuantizationScheme: "uniform",
            kvGroupSize: 64,
            maxKVSize: 16_384,
            visionCacheSize: 20,
            quantizeActivations: true,
            repeatLastN: 128,
            keepAlive: "1h"
        )
        let defaultedGenerate = try OllamaGenerateRequest(
            model: "qwen",
            prompt: "defaults"
        ).generationRequest(defaultModel: "default", defaultParameters: defaultParameters)
        precondition(defaultedGenerate.parameters.maxTokens == 128)
        precondition(defaultedGenerate.parameters.temperature == 0.7)
        precondition(defaultedGenerate.parameters.topP == 0.8)
        precondition(defaultedGenerate.parameters.topK == 32)
        precondition(defaultedGenerate.parameters.minP == 0.03)
        precondition(defaultedGenerate.parameters.seed == 123)
        precondition(defaultedGenerate.parameters.contextLength == 8192)
        precondition(defaultedGenerate.parameters.kvBits == 8)
        precondition(defaultedGenerate.parameters.kvQuantizationScheme == "uniform")
        precondition(defaultedGenerate.parameters.kvGroupSize == 64)
        precondition(defaultedGenerate.parameters.maxKVSize == 16_384)
        precondition(defaultedGenerate.parameters.visionCacheSize == 20)
        precondition(defaultedGenerate.parameters.quantizeActivations == true)
        precondition(defaultedGenerate.parameters.repeatLastN == 128)
        precondition(defaultedGenerate.parameters.keepAlive == "1h")

        let numericKeepAlive = try JSONDecoder().decode(
            OllamaChatRequest.self,
            from: Data("""
            {"model":"qwen","messages":[{"role":"user","content":"hi"}],"keep_alive":30,"tools":[{"type":"function","function":{"name":"lookup"}}],"options":{"context_length":2048,"kv_bits":3.5,"kv_quant_scheme":"turboquant","kv_group_size":128,"max_kv_size":4096,"vision_cache_size":8,"quantize_activations":true}}
            """.utf8)
        ).generationRequest(defaultModel: "default")
        precondition(numericKeepAlive.parameters.contextLength == 2048)
        precondition(numericKeepAlive.parameters.keepAlive == "30")
        precondition(numericKeepAlive.parameters.kvBits == 3.5)
        precondition(numericKeepAlive.parameters.kvQuantizationScheme == "turboquant")
        precondition(numericKeepAlive.parameters.kvGroupSize == 128)
        precondition(numericKeepAlive.parameters.maxKVSize == 4096)
        precondition(numericKeepAlive.parameters.visionCacheSize == 8)
        precondition(numericKeepAlive.parameters.quantizeActivations == true)
        precondition(numericKeepAlive.metadata.tools?.count == 1)
        precondition(numericKeepAlive.metadata.tools?.first?["function"]?["name"]?.stringValue == "lookup")
        precondition(numericKeepAlive.metadata.rawOptions?["context_length"]?.intValue == 2048)

        let ollamaEmbed = try JSONDecoder().decode(
            OllamaEmbedRequest.self,
            from: Data("""
            {"model":"qwen","input":["alpha","beta"],"truncate":false,"keep_alive":"5m","options":{"num_ctx":1024}}
            """.utf8)
        ).embeddingRequest(defaultModel: "default")
        precondition(ollamaEmbed.model == "qwen")
        precondition(ollamaEmbed.texts == ["alpha", "beta"])
        precondition(ollamaEmbed.truncate == false)
        precondition(ollamaEmbed.parameters.contextLength == 1024)
        precondition(ollamaEmbed.parameters.keepAlive == "5m")
        let legacyEmbedding = try JSONDecoder().decode(
            OllamaEmbeddingsRequest.self,
            from: Data("""
            {"model":"qwen","prompt":"legacy","keep_alive":60}
            """.utf8)
        ).embeddingRequest(defaultModel: "default")
        precondition(legacyEmbedding.texts == ["legacy"])
        precondition(legacyEmbedding.parameters.keepAlive == "60")
        let openAIEmbedding = try JSONDecoder().decode(
            OpenAIEmbeddingRequest.self,
            from: Data("""
            {"model":"qwen","input":[[1,2,3],[4,5]]}
            """.utf8)
        ).embeddingRequest(defaultModel: "default")
        precondition(openAIEmbedding.texts.isEmpty)
        precondition(openAIEmbedding.tokenIDInputs == [[1, 2, 3], [4, 5]])
        let modelOperation = OllamaModelOperationReport(
            operation: "pull",
            request: OllamaModelOperationRequest(fields: ["model": .string("qwen")])
        )
        precondition(modelOperation.model == "qwen")
        precondition(modelOperation.accepted == false)
        precondition(modelOperation.backend.activeBackend == "compatibility-shell")
        let blobOperation = OllamaBlobOperationReport(operation: "push-blob", digest: "sha256:abc123")
        precondition(blobOperation.digest == "sha256:abc123")
        precondition(blobOperation.accepted == false)
        precondition(blobOperation.exists == false)
        precondition(blobOperation.backend.activeBackend == "compatibility-shell")

        let openAIChat = try JSONDecoder().decode(
            OpenAIChatCompletionRequest.self,
            from: Data("""
            {"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"file:///tmp/a.png"}}]}],"max_completion_tokens":8,"temperature":0.5,"top_p":0.8,"top_k":11,"min_p":0.02,"seed":9,"repetition_penalty":1.2,"presence_penalty":0.1,"frequency_penalty":0.4,"stop":"END","logit_bias":{"1":-2},"enable_thinking":true,"thinking_budget":64,"thinking_start_token":"<think>","logprobs":true,"top_logprobs":3,"resize_shape":[224,336],"adapter_path":"/tmp/adapter","user":"user-1","response_format":{"type":"json_object"},"tools":[{"type":"function","function":{"name":"describe"}}],"tool_choice":{"type":"function","function":{"name":"describe"}}}
            """.utf8)
        )
        let normalizedChat = try openAIChat.generationRequest(defaultModel: "default")
        precondition(normalizedChat.messages[0].content.count == 2)
        precondition(normalizedChat.parameters.maxTokens == 8)
        precondition(normalizedChat.parameters.temperature == 0.5)
        precondition(normalizedChat.parameters.topP == 0.8)
        precondition(normalizedChat.parameters.topK == 11)
        precondition(normalizedChat.parameters.minP == 0.02)
        precondition(normalizedChat.parameters.seed == 9)
        precondition(normalizedChat.parameters.repetitionPenalty == 1.2)
        precondition(normalizedChat.parameters.presencePenalty == 0.1)
        precondition(normalizedChat.parameters.frequencyPenalty == 0.4)
        precondition(normalizedChat.parameters.stopSequences == ["END"])
        precondition(normalizedChat.parameters.enableThinking == true)
        precondition(normalizedChat.parameters.thinkingBudget == 64)
        precondition(normalizedChat.metadata.responseFormat?["type"]?.stringValue == "json_object")
        precondition(normalizedChat.metadata.tools?.first?["function"]?["name"]?.stringValue == "describe")
        precondition(normalizedChat.metadata.toolChoice?["function"]?["name"]?.stringValue == "describe")
        precondition(normalizedChat.metadata.logitBias?["1"]?.intValue == -2)
        precondition(normalizedChat.metadata.thinkingStartToken == "<think>")
        precondition(normalizedChat.metadata.logprobs == true)
        precondition(normalizedChat.metadata.topLogprobs == 3)
        precondition(normalizedChat.metadata.resizeShape == [224, 336])
        precondition(normalizedChat.metadata.adapterPath == "/tmp/adapter")
        precondition(normalizedChat.metadata.user == "user-1")
        let chatResponseFormatPlan = ResponseFormatPlanner().plan(
            metadata: normalizedChat.metadata,
            stream: normalizedChat.stream
        )
        precondition(chatResponseFormatPlan.formatType == "json_object")
        precondition(chatResponseFormatPlan.requiresJSONMode)
        precondition(chatResponseFormatPlan.backendMinimumFeatures.contains("json-output-validation"))
        let chatToolCallPlan = ToolCallPlanner().plan(
            metadata: normalizedChat.metadata,
            descriptor: descriptor
        )
        precondition(chatToolCallPlan.hasTools)
        precondition(chatToolCallPlan.toolCount == 1)
        precondition(chatToolCallPlan.toolNames == ["describe"])
        precondition(chatToolCallPlan.toolChoiceMode == "function")
        precondition(chatToolCallPlan.forcedFunctionName == "describe")
        precondition(chatToolCallPlan.requiresToolPrompt)
        precondition(chatToolCallPlan.requiresToolParser)
        precondition(chatToolCallPlan.backendMinimumFeatures.contains("forced-tool-choice"))
        let flattenedPrompt = QwenVLPromptBuilder().plainPrompt(messages: normalizedChat.messages)
        precondition(flattenedPrompt == "hi<|image_pad|>")
        let qwenChatPrompt = QwenVLPromptBuilder().qwenChatPrompt(messages: normalizedChat.messages)
        precondition(qwenChatPrompt == "<|im_start|>user\nhi<|image_pad|><|im_end|>\n<|im_start|>assistant")

        let messageMetadataChat = try JSONDecoder().decode(
            OpenAIChatCompletionRequest.self,
            from: Data("""
            {"model":"qwen","messages":[{"role":"assistant","content":[{"type":"output_text","text":"need tool"}],"reasoning":"checking","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","content":"tool result","tool_call_id":"call_1","name":"lookup"},{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AAECAw==","format":"wav"}}]}]}
            """.utf8)
        ).generationRequest(defaultModel: "default")
        precondition(messageMetadataChat.messages[0].content == [.text("need tool")])
        precondition(messageMetadataChat.messages[0].reasoning == "checking")
        precondition(messageMetadataChat.messages[0].toolCalls?.first?["id"]?.stringValue == "call_1")
        precondition(messageMetadataChat.messages[1].role == .tool)
        precondition(messageMetadataChat.messages[1].toolCallID == "call_1")
        precondition(messageMetadataChat.messages[1].name == "lookup")
        precondition(messageMetadataChat.messages[2].content == [.audioURL("AAECAw==")])

        let responsesRequest = try JSONDecoder().decode(
            OpenAIResponsesRequest.self,
            from: Data("""
            {"model":"qwen","instructions":"be concise","input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":"data:image/png;base64,AAECAw=="}]}],"max_output_tokens":6,"temperature":0.4,"top_p":0.7,"top_k":12,"min_p":0.03,"seed":11,"repetition_penalty":1.15,"stop":["DONE"],"logit_bias":{"2":-1},"enable_thinking":false,"thinking_budget":32,"thinking_start_token":"<reason>","user":"resp-user","text":{"format":{"type":"json_schema","name":"answer"}},"tools":[{"type":"function","name":"search"}],"tool_choice":"auto"}
            """.utf8)
        )
        let normalizedResponses = try responsesRequest.generationRequest(defaultModel: "default")
        precondition(normalizedResponses.messages.count == 2)
        precondition(normalizedResponses.parameters.maxTokens == 6)
        precondition(normalizedResponses.parameters.temperature == 0.4)
        precondition(normalizedResponses.parameters.topP == 0.7)
        precondition(normalizedResponses.parameters.topK == 12)
        precondition(normalizedResponses.parameters.minP == 0.03)
        precondition(normalizedResponses.parameters.seed == 11)
        precondition(normalizedResponses.parameters.repetitionPenalty == 1.15)
        precondition(normalizedResponses.parameters.stopSequences == ["DONE"])
        precondition(normalizedResponses.parameters.enableThinking == false)
        precondition(normalizedResponses.parameters.thinkingBudget == 32)
        precondition(normalizedResponses.metadata.responseFormat?["type"]?.stringValue == "json_schema")
        precondition(normalizedResponses.metadata.tools?.first?["name"]?.stringValue == "search")
        precondition(normalizedResponses.metadata.toolChoice == .string("auto"))
        precondition(normalizedResponses.metadata.logitBias?["2"]?.intValue == -1)
        precondition(normalizedResponses.metadata.thinkingStartToken == "<reason>")
        precondition(normalizedResponses.metadata.user == "resp-user")
        let responsesResponseFormatPlan = ResponseFormatPlanner().plan(
            metadata: normalizedResponses.metadata,
            stream: true
        )
        precondition(responsesResponseFormatPlan.formatType == "json_schema")
        precondition(responsesResponseFormatPlan.requiresSchemaGuidance)
        precondition(!responsesResponseFormatPlan.schemaPresent)
        precondition(responsesResponseFormatPlan.streamFraming == "api-native-stream")
        precondition(responsesResponseFormatPlan.backendMinimumFeatures.contains("json-schema-logits-processor"))
        precondition(!responsesResponseFormatPlan.warnings.isEmpty)
        let responsesToolCallPlan = ToolCallPlanner().plan(
            metadata: normalizedResponses.metadata,
            descriptor: descriptor
        )
        precondition(responsesToolCallPlan.hasTools)
        precondition(responsesToolCallPlan.toolNames == ["search"])
        precondition(responsesToolCallPlan.toolChoiceMode == "auto")
        precondition(responsesToolCallPlan.forcedFunctionName == nil)
        precondition(QwenVLPromptBuilder().plainPrompt(messages: normalizedResponses.messages) == "System: be concise\nUser: hi<|image_pad|>\nAssistant:")

        let largeBase64 = "iVBOR" + String(repeating: "A", count: 5_000)
        let guardedChat = try JSONDecoder().decode(
            OpenAIChatCompletionRequest.self,
            from: Data("""
            {"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"extract"},{"type":"image_url","image_url":{"url":"data:image/png;base64,\(largeBase64)"}}]}]}
            """.utf8)
        )
        let guardedPrompt = QwenVLPromptBuilder().plainPrompt(
            messages: try guardedChat.generationRequest(defaultModel: "default").messages
        )
        precondition(guardedPrompt == "extract<|image_pad|>")
        precondition(!guardedPrompt.contains(largeBase64))

        try Data(base64Encoded: Self.onePixelPNGBase64)!
            .write(to: model.appendingPathComponent("tiny.png"))
        let mediaRequest = try OllamaGenerateRequest(
            model: "qwen",
            prompt: "media",
            images: [
                onePixelPNGBase64,
                "data:image/png;base64,\(onePixelPNGBase64)",
                "tiny.png",
                "https://example.invalid/image.png",
            ]
        ).generationRequest(defaultModel: "default")
        let mediaReport = MediaReferenceResolver(baseDirectory: model).report(for: mediaRequest)
        precondition(mediaReport.imageCount == 4)
        precondition(mediaReport.loadableCount == 3)
        precondition(mediaReport.errorCount == 1)
        precondition(mediaReport.references.map(\.source) == [.rawBase64, .dataURI, .filePath, .remoteURL])
        let imagePlanReport = QwenVLImageInputPlanner(
            resolver: MediaReferenceResolver(baseDirectory: model)
        ).plan(request: mediaRequest)
        precondition(imagePlanReport.imageCount == 4)
        precondition(imagePlanReport.plannedCount == 3)
        precondition(imagePlanReport.errorCount == 1)
        precondition(imagePlanReport.imageGrids == [
            QwenVLGridTHW(temporal: 1, height: 4, width: 4),
            QwenVLGridTHW(temporal: 1, height: 4, width: 4),
            QwenVLGridTHW(temporal: 1, height: 4, width: 4),
        ])
        precondition(imagePlanReport.placeholderTokenCount == 12)
        let pixelPreflight = QwenVLImagePixelPreflightPlanner(
            resolver: MediaReferenceResolver(baseDirectory: model)
        ).plan(request: mediaRequest)
        precondition(pixelPreflight.imageCount == 4)
        precondition(pixelPreflight.preparedCount == 3)
        precondition(pixelPreflight.errorCount == 1)
        precondition(pixelPreflight.totalRGBByteCount == 56 * 56 * 3 * 3)
        precondition(pixelPreflight.totalPatchFloat32ByteCount == 75_264 * 3)
        precondition(pixelPreflight.imageGrids == [
            QwenVLGridTHW(temporal: 1, height: 4, width: 4),
            QwenVLGridTHW(temporal: 1, height: 4, width: 4),
            QwenVLGridTHW(temporal: 1, height: 4, width: 4),
        ])
        precondition(pixelPreflight.images.compactMap(\.pixelShape) == [
            [56, 56, 3],
            [56, 56, 3],
            [56, 56, 3],
        ])
        precondition(pixelPreflight.images.compactMap(\.patchShape) == [
            [16, 1176],
            [16, 1176],
            [16, 1176],
        ])
        precondition(pixelPreflight.images.compactMap(\.patchFloat32ByteCount) == [
            75_264,
            75_264,
            75_264,
        ])
        precondition(pixelPreflight.images.compactMap(\.patchStats?.count) == [
            18_816,
            18_816,
            18_816,
        ])
        precondition(pixelPreflight.images.compactMap(\.patchStats?.min) == [-1, -1, -1])
        precondition(pixelPreflight.images.compactMap(\.patchStats?.max) == [-1, -1, -1])
        precondition(pixelPreflight.images.compactMap(\.patchStats?.checksum) == [-18_816, -18_816, -18_816])
        let preflight = GenerationPreflightPlanner(
            descriptor: descriptor,
            mediaResolver: MediaReferenceResolver(baseDirectory: model)
        ).plan(request: mediaRequest)
        precondition(preflight.promptStyle == .qwenChat)
        precondition(preflight.prompt.contains("<|im_start|>user"))
        precondition(preflight.promptRender.source == "model-prompt-renderer")
        precondition(!preflight.responseFormatPlan.requested)
        precondition(!preflight.toolCallPlan.hasTools)
        precondition(preflight.imageInputs.placeholderTokenCount == 12)
        precondition(preflight.imagePixels.preparedCount == 3)
        precondition(preflight.imagePixels.totalRGBByteCount == 56 * 56 * 3 * 3)
        precondition(preflight.imagePixels.totalPatchFloat32ByteCount == 75_264 * 3)
        precondition(preflight.imagePixels.images.compactMap(\.patchStats?.checksum) == [-18_816, -18_816, -18_816])
        precondition(preflight.runtime.modelMaxPositionEmbeddings == 40960)
        precondition(preflight.runtime.effectiveContextLength == 40960)
        precondition(preflight.runtime.maxCompletionTokens == 512)
        precondition(preflight.runtime.useCache == true)
        precondition(preflight.runtime.stream == false)
        precondition(preflight.runtime.sampling.sampler == "greedy")
        precondition(preflight.runtime.sampling.deterministic)
        precondition(preflight.runtime.sampling.backendMinimumFeatures == ["argmax-logits"])
        precondition(preflight.tokenization?.tokenIDs.contains(151655) == true)
        precondition(preflight.tokenization?.requiresTokenizerImplementation == true)
        precondition(preflight.metadataReady)
        precondition(!preflight.generationReady)
        precondition(!preflight.canAttemptGeneration)
        precondition(preflight.blockingReasons.contains { $0.contains("backend") || $0.contains("not linked") })
        let constrainedRuntime = GenerationPreflightPlanner(descriptor: descriptor).plan(
            request: GenerationRequest(
                model: "qwen",
                messages: [ChatMessage(role: .user, content: [.text("hello")])],
                parameters: GenerationParameters(maxTokens: 8, contextLength: 4, keepAlive: "15m"),
                stream: true
            )
        ).runtime
        precondition(constrainedRuntime.requestedContextLength == 4)
        precondition(constrainedRuntime.effectiveContextLength == 4)
        precondition(constrainedRuntime.keepAlive == "15m")
        precondition(constrainedRuntime.stream)
        precondition(constrainedRuntime.exceedsContextLength == true)
        let advancedRuntime = GenerationPreflightPlanner(descriptor: descriptor).plan(
            request: GenerationRequest(
                model: "qwen",
                messages: [ChatMessage(role: .user, content: [.text("hello")])],
                parameters: GenerationParameters(
                    maxTokens: 8,
                    temperature: 0.7,
                    topP: 0.9,
                    topK: 40,
                    minP: 0.05,
                    typicalP: 0.8,
                    tfsZ: 1.0,
                    seed: 42,
                    repetitionPenalty: 1.1,
                    repeatLastN: 64,
                    presencePenalty: 0.2,
                    frequencyPenalty: 0.3,
                    penalizeNewline: true,
                    mirostat: 2,
                    mirostatTau: 5.0,
                    mirostatEta: 0.1
                )
            )
        ).runtime
        precondition(advancedRuntime.sampling.sampler == "mirostat")
        precondition(!advancedRuntime.sampling.deterministic)
        precondition(advancedRuntime.sampling.enabledFilters == ["top-k", "top-p", "min-p", "typical-p", "tail-free"])
        precondition(advancedRuntime.sampling.enabledPenalties == ["repetition", "presence", "frequency", "repeat-window", "newline"])
        precondition(advancedRuntime.sampling.requiresAdvancedSampler)
        precondition(advancedRuntime.sampling.backendMinimumFeatures.contains("seeded-rng"))
        precondition(!advancedRuntime.sampling.warnings.isEmpty)
        let greedySample = try GenerationLogitsSampler(plan: preflight.runtime.sampling)
            .sample(logits: [0.1, 0.9, 0.2])
        precondition(greedySample.tokenID == 1)
        precondition(greedySample.sampler == "greedy")
        let temperaturePlan = GenerationSamplingPlanner().plan(
            parameters: GenerationParameters(
                temperature: 0.7,
                topP: 0.8,
                topK: 2,
                minP: 0.1,
                seed: 7
            )
        )
        let temperatureSampleA = try GenerationLogitsSampler(plan: temperaturePlan)
            .sample(logits: [0.1, 0.9, 0.2])
        let temperatureSampleB = try GenerationLogitsSampler(plan: temperaturePlan)
            .sample(logits: [0.1, 0.9, 0.2])
        precondition(temperatureSampleA == temperatureSampleB)
        precondition([1, 2].contains(temperatureSampleA.tokenID))
        let penaltyPlan = GenerationSamplingPlanner().plan(
            parameters: GenerationParameters(
                temperature: 0,
                repetitionPenalty: 2.0,
                repeatLastN: 1
            )
        )
        let penaltySample = try GenerationLogitsSampler(plan: penaltyPlan)
            .sample(logits: [0.1, 0.9, 0.8], recentTokenIDs: [1])
        precondition(penaltySample.tokenID == 2)
        let unsupportedSampler = GenerationLogitsSampler(plan: advancedRuntime.sampling)
        do {
            _ = try unsupportedSampler.sample(logits: [0.1, 0.9, 0.2])
            preconditionFailure("advanced sampler should require a backend-specific implementation")
        } catch GenerationLogitsSamplerError.unsupportedAdvancedSampler(_) {
        } catch {
            preconditionFailure("unexpected sampler error: \(error)")
        }
        let cacheRuntime = GenerationPreflightPlanner(descriptor: descriptor).plan(
            request: GenerationRequest(
                model: "qwen",
                messages: [ChatMessage(role: .user, content: [.text("hello")])],
                parameters: GenerationParameters(
                    kvBits: 3.5,
                    kvQuantizationScheme: "turboquant",
                    kvGroupSize: 128,
                    maxKVSize: 4096,
                    visionCacheSize: 8,
                    quantizeActivations: true
                )
            )
        ).runtime
        precondition(cacheRuntime.kvBits == 3.5)
        precondition(cacheRuntime.kvQuantizationScheme == "turboquant")
        precondition(cacheRuntime.kvGroupSize == 128)
        precondition(cacheRuntime.maxKVSize == 4096)
        precondition(cacheRuntime.visionCacheSize == 8)
        precondition(cacheRuntime.quantizeActivations == true)
        let engine = CompatibilityGenerationEngine(
            descriptor: descriptor,
            mediaResolver: MediaReferenceResolver(baseDirectory: model)
        )
        let embeddingReport = engine.unavailableEmbeddingReport(for: ollamaEmbed)
        precondition(embeddingReport.model == descriptor.id)
        precondition(embeddingReport.request.texts == ["alpha", "beta"])
        precondition(embeddingReport.inputCount == 2)
        let processed = try engine.processedInput(for: mediaRequest)
        precondition(processed.pixelShape == [3, 56, 56, 3])
        precondition(processed.runtime == preflight.runtime)
        precondition(processed.tokenIDs?.contains(151655) == true)
        let unavailable = engine.unavailableReport(for: mediaRequest)
        precondition(unavailable.model == descriptor.id)
        precondition(unavailable.canonicalModelType == "qwen2_vl")
        precondition(unavailable.backend.activeBackend == "compatibility-shell")
        precondition(unavailable.preflight.promptStyle == .qwenChat)
        precondition(unavailable.preflight.imagePixels.preparedCount == 3)
        precondition(!unavailable.preflight.canAttemptGeneration)

        let result = CompletedGeneration(
            model: "qwen",
            text: "ok",
            usage: GenerationUsage(promptTokens: 3, completionTokens: 1)
        )
        let ollamaResponseData = try JSONEncoder().encode(OllamaGenerateResponse(result: result, createdAt: "1970-01-01T00:00:00Z"))
        let ollamaResponseJSON = String(decoding: ollamaResponseData, as: UTF8.self)
        precondition(ollamaResponseJSON.contains("\"prompt_eval_count\":3"))
        precondition(ollamaResponseJSON.contains("\"eval_count\":1"))
        let residencyResponseJSON = String(
            decoding: try JSONEncoder().encode(
                OllamaGenerateResponse(
                    result: CompletedGeneration(model: "qwen", text: ""),
                    doneReason: "unload"
                )
            ),
            as: UTF8.self
        )
        precondition(residencyResponseJSON.contains("\"done_reason\":\"unload\""))
        precondition(residencyResponseJSON.contains("\"response\":\"\""))
        let tokenChunk = GenerationChunk(text: "o", tokenID: 1, isFinished: false)
        let finalChunk = GenerationChunk(text: "", tokenID: nil, isFinished: true)
        let lengthChunk = GenerationChunk(text: "", tokenID: nil, isFinished: true, finishReason: "length")
        let ollamaGenerateChunkJSON = String(
            decoding: try JSONEncoder().encode(
                OllamaGenerateStreamChunk(
                    model: "qwen",
                    chunk: tokenChunk,
                    createdAt: "1970-01-01T00:00:00Z"
                )
            ),
            as: UTF8.self
        )
        precondition(ollamaGenerateChunkJSON.contains("\"response\":\"o\""))
        precondition(ollamaGenerateChunkJSON.contains("\"done\":false"))
        let ollamaJSONLine = ResponseStreamFramer.jsonLine(
            OllamaGenerateStreamChunk(
                model: "qwen",
                chunk: tokenChunk,
                createdAt: "1970-01-01T00:00:00Z"
            )
        )
        precondition(ollamaJSONLine.hasSuffix("\n"))
        precondition(ollamaJSONLine.contains("\"response\":\"o\""))
        let ollamaChatFinalJSON = String(
            decoding: try JSONEncoder().encode(
                OllamaChatStreamChunk(
                    model: "qwen",
                    chunk: finalChunk,
                    usage: result.usage,
                    createdAt: "1970-01-01T00:00:00Z"
                )
            ),
            as: UTF8.self
        )
        precondition(ollamaChatFinalJSON.contains("\"done\":true"))
        precondition(ollamaChatFinalJSON.contains("\"eval_count\":1"))
        let ollamaLengthJSON = String(
            decoding: try JSONEncoder().encode(
                OllamaGenerateStreamChunk(
                    model: "qwen",
                    chunk: lengthChunk,
                    usage: result.usage,
                    createdAt: "1970-01-01T00:00:00Z"
                )
            ),
            as: UTF8.self
        )
        precondition(ollamaLengthJSON.contains("\"done_reason\":\"length\""))

        let openAIResponseData = try JSONEncoder().encode(OpenAIChatCompletionResponse(result: result, id: "chatcmpl-test", created: 0))
        let openAIResponseJSON = String(decoding: openAIResponseData, as: UTF8.self)
        precondition(openAIResponseJSON.contains("\"object\":\"chat.completion\""))
        precondition(openAIResponseJSON.contains("\"completion_tokens\":1"))
        let toolCall = GenerationToolCall(
            id: "call_test",
            function: GenerationToolCallFunction(
                name: "lookup",
                arguments: ["query": .string("swift")]
            )
        )
        let toolResult = CompletedGeneration(
            model: "qwen",
            text: "",
            finishReason: "tool_calls",
            usage: GenerationUsage(promptTokens: 3, completionTokens: 1),
            toolCalls: [toolCall]
        )
        let toolOpenAIJSON = String(
            decoding: try JSONEncoder().encode(
                OpenAIChatCompletionResponse(result: toolResult, id: "chatcmpl-tool", created: 0)
            ),
            as: UTF8.self
        )
        precondition(toolOpenAIJSON.contains("\"finish_reason\":\"tool_calls\""))
        precondition(toolOpenAIJSON.contains("\"tool_calls\""))
        precondition(toolOpenAIJSON.contains("\"id\":\"call_test\""))
        precondition(toolOpenAIJSON.contains("\"name\":\"lookup\""))
        precondition(toolOpenAIJSON.contains(#""arguments":"{\"query\":\"swift\"}""#))
        let toolOllamaJSON = String(
            decoding: try JSONEncoder().encode(OllamaChatResponse(result: toolResult, createdAt: "1970-01-01T00:00:00Z")),
            as: UTF8.self
        )
        precondition(toolOllamaJSON.contains("\"tool_calls\""))
        precondition(toolOllamaJSON.contains("\"arguments\":{\"query\":\"swift\"}"))
        let modelListJSON = String(
            decoding: try JSONEncoder().encode(
                OpenAIModelListResponse(models: [OpenAIModelResponse(id: descriptor.id)])
            ),
            as: UTF8.self
        )
        precondition(modelListJSON.contains("\"object\":\"list\""))
        precondition(modelListJSON.contains("\"owned_by\":\"mlx-vlm-swift\""))
        let modelResponseJSON = String(
            decoding: try JSONEncoder().encode(OpenAIModelResponse(id: descriptor.id)),
            as: UTF8.self
        )
        precondition(modelResponseJSON.contains("\"object\":\"model\""))
        precondition(normalizedRoutePath("/v1/models?limit=20") == "/v1/models")
        precondition(normalizedRoutePath("/v1/models/\(descriptor.id)?x=1") == "/v1/models/\(descriptor.id)")
        precondition(normalizedRoutePath("/v1/models/") == "/v1/models")
        let openAIStreamJSON = String(
            decoding: try JSONEncoder().encode(
                OpenAIChatCompletionStreamResponse(
                    model: "qwen",
                    chunk: finalChunk,
                    id: "chatcmpl-test",
                    created: 0,
                    usage: result.usage
                )
            ),
            as: UTF8.self
        )
        precondition(openAIStreamJSON.contains("\"object\":\"chat.completion.chunk\""))
        precondition(openAIStreamJSON.contains("\"finish_reason\":\"stop\""))
        precondition(openAIStreamJSON.contains("\"completion_tokens\":1"))
        let toolChunk = GenerationChunk(text: "", finishReason: nil, toolCalls: [toolCall])
        let toolStreamJSON = String(
            decoding: try JSONEncoder().encode(
                OpenAIChatCompletionStreamResponse(
                    model: "qwen",
                    chunk: toolChunk,
                    id: "chatcmpl-tool",
                    created: 0
                )
            ),
            as: UTF8.self
        )
        precondition(toolStreamJSON.contains("\"tool_calls\""))
        precondition(toolStreamJSON.contains("\"id\":\"call_test\""))
        let openAILengthStreamJSON = String(
            decoding: try JSONEncoder().encode(
                OpenAIChatCompletionStreamResponse(
                    model: "qwen",
                    chunk: lengthChunk,
                    id: "chatcmpl-test",
                    created: 0,
                    usage: result.usage
                )
            ),
            as: UTF8.self
        )
        precondition(openAILengthStreamJSON.contains("\"finish_reason\":\"length\""))
        let openAISSE = ResponseStreamFramer.serverSentEvent(
            OpenAIChatCompletionStreamResponse(
                model: "qwen",
                chunk: tokenChunk,
                id: "chatcmpl-test",
                created: 0
            )
        )
        precondition(openAISSE.hasPrefix("data: {"))
        precondition(openAISSE.hasSuffix("\n\n"))
        precondition(openAISSE.contains("\"object\":\"chat.completion.chunk\""))
        precondition(ResponseStreamFramer.doneServerSentEvent() == "data: [DONE]\n\n")
        let renderedOpenAI = GenerationAPIResponseRenderer.renderCompleted(result, api: .openAIChat)
        precondition(renderedOpenAI.contentType == "application/json")
        precondition(renderedOpenAI.body.contains("\"object\":\"chat.completion\""))
        precondition(renderedOpenAI.frameCount == 1)
        let renderedOllamaStream = GenerationAPIResponseRenderer.renderCompleted(
            result,
            api: .ollamaGenerate,
            stream: true,
            chunks: [tokenChunk, finalChunk]
        )
        precondition(renderedOllamaStream.contentType == "application/x-ndjson")
        precondition(renderedOllamaStream.frameCount == 2)
        precondition(renderedOllamaStream.body.contains("\"response\":\"o\""))
        let renderedOpenAIStream = GenerationAPIResponseRenderer.renderCompleted(
            result,
            api: .openAIChat,
            stream: true,
            chunks: [tokenChunk, finalChunk]
        )
        precondition(renderedOpenAIStream.contentType == "text/event-stream")
        precondition(renderedOpenAIStream.body.contains("data: [DONE]"))
        precondition(renderedOpenAIStream.frameCount == 3)
        let chunkCollection = GenerationChunkCollector.collect(
            model: "qwen",
            promptTokenCount: 2,
            chunks: [tokenChunk, finalChunk]
        )
        precondition(chunkCollection.text == "o")
        precondition(chunkCollection.completionTokens == 1)
        precondition(chunkCollection.tokenAccountingSource == "token-id-chunks")
        let toolCollection = GenerationChunkCollector.collect(
            model: "qwen",
            promptTokenCount: 2,
            chunks: [toolChunk, GenerationChunk(text: "", isFinished: true, finishReason: "tool_calls")]
        )
        precondition(toolCollection.completedGeneration.toolCalls == [toolCall])
        precondition(toolCollection.completedGeneration.finishReason == "tool_calls")
        let endpointRender = GenerationEndpointRenderer.render(
            model: "qwen",
            promptTokenCount: 2,
            api: .openAIChat,
            stream: true,
            chunks: [tokenChunk, finalChunk]
        )
        precondition(endpointRender.collection.totalTokens == 3)
        precondition(endpointRender.response.contentType == "text/event-stream")
        precondition(endpointRender.response.body.contains("data: [DONE]"))

        let responsesResponseData = try JSONEncoder().encode(
            OpenAIResponsesResponse(
                result: result,
                id: "resp-test",
                createdAt: 0
            )
        )
        let responsesResponseJSON = String(decoding: responsesResponseData, as: UTF8.self)
        precondition(responsesResponseJSON.contains("\"object\":\"response\""))
        precondition(responsesResponseJSON.contains("\"output_text\":\"ok\""))
        precondition(responsesResponseJSON.contains("\"type\":\"output_text\""))
        precondition(responsesResponseJSON.contains("\"completion_tokens\":1"))

        let running = OllamaRunningModelsResponse(
            models: [
                OllamaRunningModel(
                    descriptor: descriptor,
                    expiresAt: "1970-01-01T00:00:00Z"
                )
            ]
        )
        let runningJSON = String(decoding: try JSONEncoder().encode(running), as: UTF8.self)
        precondition(runningJSON.contains("\"size_vram\":0"))
        precondition(runningJSON.contains("\"models\""))
        var residency = OllamaModelResidency()
        precondition(residency.runningModelsResponse(descriptor: descriptor).models.count == 1)
        let unloadResponse = residency.unload(model: descriptor.id)
        precondition(unloadResponse.status == "ok")
        precondition(unloadResponse.unloaded)
        precondition(residency.unloadCount == 1)
        precondition(residency.runningModelsResponse(descriptor: descriptor).models.isEmpty)
        residency.markLoaded()
        precondition(residency.runningModelsResponse(descriptor: descriptor).models.count == 1)
        let showJSON = String(decoding: try JSONEncoder().encode(OllamaShowResponse(descriptor: descriptor)), as: UTF8.self)
        precondition(showJSON.contains("\"mlx_vlm.quantization_bits\":4"))
        precondition(showJSON.contains("\"quantization_level\":\"mlx 4.0-bit group=64\""))
        precondition(showJSON.contains("\"mlx_vlm.backend_ready\":false"))
        precondition(showJSON.contains("\"mlx_vlm.generation_ready\":false"))
        precondition(showJSON.contains("\"mlx_vlm.metadata_ready\":true"))
        precondition(showJSON.contains("\"mlx_vlm.config_text_source\":\"empty\""))
        precondition(showJSON.contains("\"mlx_vlm.normalized_config_keys\""))
        precondition(showJSON.contains("\"mlx_vlm.tokenizer_backend\":\"tokenizers-json-bpe\""))
        precondition(showJSON.contains("\"mlx_vlm.tokenizer_swift_execution_supported\":false"))
        precondition(showJSON.contains("\"mlx_vlm.primary_task\":\"vision-language-generation\""))
        let showRequest = try JSONDecoder().decode(
            OllamaShowRequest.self,
            from: Data(#"{"model":"qwen","verbose":true}"#.utf8)
        )
        precondition(showRequest.modelName == "qwen")
        precondition(showRequest.verbose)
        let backendPlan = BackendDependencyPlanner().plan(rootPath: FileManager.default.currentDirectoryPath)
        precondition(backendPlan.compatibilityShellBuildable)
        precondition(backendPlan.packageDeclaresBackendTarget)
        precondition(backendPlan.packageDeclaresMLXSwift)
        precondition(backendPlan.packageDeclaresMLXSwiftLM)
        precondition(backendPlan.manifestSupportsMLXBackendToggle)
        precondition(backendPlan.manifestSupportsLocalMLXDependencies)
        precondition(backendPlan.manifestSupportsExplicitMLXPaths)
        precondition(!backendPlan.canEnableMLXBackend)
        precondition(backendPlan.requirements.map(\.packageName).contains("mlx-swift"))
        precondition(backendPlan.requirements.map(\.packageName).contains("mlx-swift-lm"))
        let backendAvailability = MLXBackendFactory.availability(rootPath: FileManager.default.currentDirectoryPath)
        precondition(backendAvailability.canCreateBackend == MLXRuntimeProbe.detectRealMLXAPIImplementationCompiled())
        precondition(backendAvailability.runtimeProbe.backendImplementationReady == MLXRuntimeProbe.detectBackendImplementationReady())
        precondition(backendAvailability.runtimeProbe.realMLXAPIImplementationCompiled == MLXRuntimeProbe.detectRealMLXAPIImplementationCompiled())
        precondition(
            backendPlan.requirements
                .first { $0.packageName == "mlx-swift" }?
                .localCandidatePaths
                .contains { $0.caseInsensitiveCompare(FileManager.default.currentDirectoryPath) == .orderedSame } == false
        )
        precondition(!BackendStatus.compatibilityShell.ready)
        precondition(BackendStatus.compatibilityShell.capabilities.contains(.generationUnavailable))
        let capabilityPlan = ModelCapabilityPlanner().plan(descriptor: descriptor)
        precondition(capabilityPlan.primaryTask == "vision-language-generation")
        precondition(capabilityPlan.supportsTextGeneration)
        precondition(capabilityPlan.supportsOllamaGenerationAPI)
        precondition(capabilityPlan.supportsVideoInputs)
        let compatibility = ModelCompatibilityValidator.validate(descriptor: descriptor)
        precondition(compatibility.metadataReady)
        precondition(!compatibility.generationReady)
        precondition(compatibility.checks.contains { $0.id == "safetensors-readable" && $0.passed })
        precondition(compatibility.checks.contains { $0.id == "config-normalization" && $0.passed && $0.severity == .warning })
        precondition(compatibility.checks.contains { $0.id == "weight-index-shards-present" && $0.passed })
        precondition(compatibility.checks.contains { $0.id == "weight-catalog-readable" && $0.passed })
        precondition(compatibility.checks.contains { $0.id == "weight-data-readable" && $0.passed })
        precondition(compatibility.checks.contains { $0.id == "weight-data-byte-counts" && $0.passed })
        precondition(compatibility.checks.contains { $0.id == "mlx-weight-loadable" && $0.passed })
        precondition(compatibility.checks.contains { $0.id == "weight-index-total-size" && $0.passed && $0.severity == .warning })
        precondition(compatibility.checks.contains { $0.id == "weight-index-tensors-present" && $0.passed })
        precondition(compatibility.checks.contains { $0.id == "tokenizer-json-readable" && $0.passed })
        precondition(compatibility.checks.contains { $0.id == "tokenizer-vocab-compatible" && $0.passed })
        precondition(compatibility.checks.contains { $0.id == "tokenizer-catalog-readable" && $0.passed })
        precondition(compatibility.checks.contains { $0.id == "tokenizer-catalog-unique" && $0.passed })
        precondition(compatibility.checks.contains { $0.id == "chat-template-renderer" && !$0.passed && $0.severity == .warning })
        precondition(compatibility.checks.contains { $0.id == "adapter-metadata" && $0.passed && $0.severity == .warning })
        precondition(compatibility.checks.contains { $0.id == "generation-api-compatible" && $0.passed && $0.severity == .warning })
        precondition(compatibility.checks.contains { $0.id == "qwen-vl-core-tensor-coverage" && !$0.passed && $0.severity == .warning })
        precondition(compatibility.checks.contains { $0.id == "backend-ready" && !$0.passed })
        let modelLoadPlan = ModelLoadPlanner().plan(descriptor: descriptor)
        precondition(modelLoadPlan.normalizedConfig["text_config"]?.objectValue != nil)
        precondition(modelLoadPlan.normalizedConfig["audio_config"]?.objectValue != nil)
        precondition(modelLoadPlan.qwenVLConfig?.family == .qwen2VL)
        precondition(modelLoadPlan.qwenVLArchitecture?.presentCoreTensorCount == 2)
        precondition(modelLoadPlan.capabilities.supportsOllamaGenerationAPI)
        precondition(modelLoadPlan.tokenizerCatalog?.tokenCount == 7)
        precondition(modelLoadPlan.tokenizerPlan.requiredBackend == "tokenizers-json-bpe")
        precondition(modelLoadPlan.chatTemplatePlan.requiredRenderer == "custom-template")
        precondition(modelLoadPlan.configNormalization == descriptor.configNormalization)
        precondition(modelLoadPlan.adapterMetadata.isLoadable)
        precondition(modelLoadPlan.weightCatalog.tensorCount == 2)
        precondition(modelLoadPlan.weightDataCatalog.totalReadableBytes == 8)
        precondition(modelLoadPlan.memoryEstimate.readableWeightBytes == 8)
        precondition(modelLoadPlan.memoryEstimate.kvCacheTokenCapacity == 40960)
        precondition(modelLoadPlan.metadataReady)
        precondition(modelLoadPlan.canLoadMetadata)
        precondition(!modelLoadPlan.generationReady)
        precondition(!modelLoadPlan.canAttemptGeneration)
        precondition(modelLoadPlan.blockingReasons.contains { $0.contains("not linked") })
        let backendContext = try CompatibilityVLMBackend(descriptor: descriptor).loadContext()
        precondition(backendContext.normalizedConfig["text_config"]?.objectValue != nil)
        precondition(backendContext.loadPlan == modelLoadPlan)
        let memoryEstimate = ModelMemoryEstimator().estimate(
            descriptor: descriptor,
            weightDataCatalog: weightDataCatalog,
            qwenVLConfig: qwenConfig,
            parameters: GenerationParameters(
                contextLength: 4096,
                kvBits: 8,
                maxKVSize: 2048,
                visionCacheSize: 4
            )
        )
        precondition(memoryEstimate.readableWeightBytes == 8)
        precondition(memoryEstimate.kvCacheTokenCapacity == 2048)
        precondition(memoryEstimate.kvCacheElementBits == 8)
        precondition(memoryEstimate.estimatedKVCacheBytes == 28 * 2048 * 8 * 128 * 2)
        precondition(memoryEstimate.estimatedVisionCacheBytes == 4 * 1536 * 4)

        let rfdetrModel = root.appendingPathComponent("rfdetr")
        try FileManager.default.createDirectory(at: rfdetrModel, withIntermediateDirectories: true)
        try Data("""
        {"model_type":"rf-detr","vision_config":{},"vocab_size":10}
        """.utf8).write(to: rfdetrModel.appendingPathComponent("config.json"))
        try makeSafetensorsFixture().write(to: rfdetrModel.appendingPathComponent("model.safetensors"))
        let rfdetrDescriptor = try ModelStore().loadDescriptor(pathOrIdentifier: rfdetrModel.path)
        precondition(rfdetrDescriptor.canonicalModelType == "rfdetr")
        let rfdetrCapabilities = ModelCapabilityPlanner().plan(descriptor: rfdetrDescriptor)
        precondition(rfdetrCapabilities.primaryTask == "object-detection-or-segmentation")
        precondition(!rfdetrCapabilities.supportsTextGeneration)
        precondition(!rfdetrCapabilities.supportsOllamaGenerationAPI)
        let rfdetrCompatibility = ModelCompatibilityValidator.validate(descriptor: rfdetrDescriptor)
        precondition(rfdetrCompatibility.checks.contains { $0.id == "generation-api-compatible" && !$0.passed && $0.severity == .warning })
        let rfdetrPreflight = GenerationPreflightPlanner(descriptor: rfdetrDescriptor).plan(
            request: GenerationRequest(
                model: "rfdetr",
                messages: [ChatMessage(role: .user, content: [.text("detect person")])]
            )
        )
        precondition(rfdetrPreflight.capabilities.primaryTask == "object-detection-or-segmentation")
        precondition(rfdetrPreflight.blockingReasons.contains { $0.contains("not compatible with text generation endpoints") })
        let rfdetrLoadPlan = ModelLoadPlanner().plan(descriptor: rfdetrDescriptor)
        precondition(!rfdetrLoadPlan.capabilities.supportsOllamaGenerationAPI)
        precondition(rfdetrLoadPlan.blockingReasons.contains { $0.contains("not compatible with text generation endpoints") })

        let cachedModelID = "mlx-community/Qwen2-VL-Test"
        let hubRoot = root.appendingPathComponent("hf").appendingPathComponent("hub")
        let modelCache = hubRoot.appendingPathComponent("models--mlx-community--Qwen2-VL-Test")
        let snapshot = modelCache.appendingPathComponent("snapshots").appendingPathComponent("abc123")
        try FileManager.default.createDirectory(at: snapshot.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: model, to: snapshot)
        try FileManager.default.createDirectory(at: modelCache.appendingPathComponent("refs"), withIntermediateDirectories: true)
        try Data("abc123\n".utf8).write(to: modelCache.appendingPathComponent("refs").appendingPathComponent("main"))
        let cachedStore = ModelStore(
            cacheResolver: HuggingFaceCacheResolver(environment: ["HUGGINGFACE_HUB_CACHE": hubRoot.path])
        )
        let cachedDescriptor = try cachedStore.loadDescriptor(pathOrIdentifier: cachedModelID)
        precondition(cachedDescriptor.id == cachedModelID)
        precondition(cachedDescriptor.path.hasSuffix("/snapshots/abc123"))
        precondition(cachedDescriptor.safetensorsMetadata.first?.isReadable == true)

        print("self-test passed")
    }

    private static func makeSafetensorsFixture() -> Data {
        let header = #"{"model.embed_tokens.weight":{"dtype":"F16","shape":[2,1],"data_offsets":[0,4]},"visual.patch_embed.proj.weight":{"dtype":"F16","shape":[2,1],"data_offsets":[4,8]}}"#
        let headerData = Data(header.utf8)
        var length = UInt64(headerData.count).littleEndian
        var data = Data()
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(headerData)
        data.append(contentsOf: [0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44])
        return data
    }

    private static let qwen2VLConfig = """
    {
      "model_type": "qwen2_vl",
      "quantization": {"mode": "mlx", "bits": 4, "group_size": 64},
      "hidden_size": 1536,
      "num_hidden_layers": 28,
      "intermediate_size": 8960,
      "num_attention_heads": 12,
      "rms_norm_eps": 0.000001,
      "vocab_size": 151936,
      "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
      "vision_config": {
        "model_type": "qwen2_vl",
        "depth": 32,
        "embed_dim": 1280,
        "hidden_size": 1536,
        "num_heads": 16
      }
    }
    """

    private static let qwen25VLConfig = """
    {
      "model_type": "qwen2_5_vl",
      "hidden_size": 2048,
      "num_hidden_layers": 36,
      "intermediate_size": 11008,
      "num_attention_heads": 16,
      "rms_norm_eps": 0.000001,
      "vocab_size": 152064,
      "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
      "vision_config": {
        "model_type": "qwen2_5_vl"
      }
    }
    """

    private static let tokenizerConfig = """
    {
      "chat_template": "tokenizer-template",
      "added_tokens_decoder": {
        "151655": {"content": "<|image_pad|>"},
        "151656": {"content": "<|video_pad|>"},
        "151652": {"content": "<|vision_start|>"},
        "151653": {"content": "<|vision_end|>"}
      }
    }
    """

    private static let tokenizerJSON = """
    {
      "version": "1.0",
      "model": {
        "type": "BPE",
        "vocab": {"hello": 0, "world": 1, "user": 2},
        "merges": ["hello world"]
      },
      "added_tokens": [
        {"id": 151655, "content": "<|image_pad|>"},
        {"id": 151656, "content": "<|video_pad|>"},
        {"id": 151644, "content": "<|im_start|>"},
        {"id": 151645, "content": "<|im_end|>"}
      ],
      "normalizer": {"type": "Sequence"},
      "pre_tokenizer": {"type": "ByteLevel"},
      "decoder": {"type": "ByteLevel"}
    }
    """

    private static let processorConfig = """
    {
      "chat_template": "processor-template",
      "image_processor": {
        "merge_size": 2,
        "min_pixels": 3136,
        "max_pixels": 1003520
      }
    }
    """

    private static let preprocessorConfig = """
    {
      "image_processor_type": "Qwen2VLImageProcessor",
      "patch_size": {"height": 14, "width": 14},
      "temporal_patch_size": 2,
      "merge_size": 1,
      "size": {
        "shortest_edge": 1024,
        "longest_edge": 2048
      },
      "image_mean": [0.5, 0.5, 0.5],
      "image_std": [0.5, 0.5, 0.5],
      "do_rescale": true,
      "do_normalize": true,
      "do_convert_rgb": true
    }
    """

    private static let weightIndex = """
    {
      "metadata": {"total_size": 8},
      "weight_map": {
        "model.embed_tokens.weight": "model.safetensors",
        "visual.patch_embed.proj.weight": "model.safetensors"
      }
    }
    """

    private static let onePixelPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABDQottAAAAABJRU5ErkJggg=="
}

private extension String {
    func filterImageTokens() -> Int {
        components(separatedBy: "<|image_pad|>").count - 1
    }

    func filterVideoTokens() -> Int {
        components(separatedBy: "<|video_pad|>").count - 1
    }
}

private struct SelfTestStreamingBackend: VLMBackend {
    let descriptor: ModelDescriptor
    let chunks: [GenerationChunk]
    let status: BackendStatus = .compatibilityShell

    func loadContext() throws -> ModelLoadContext {
        CompatibilityModelContainer(descriptor: descriptor, backend: status).context
    }

    func process(_ request: GenerationRequest) throws -> ProcessedGenerationInput {
        try CompatibilityProcessor(backend: status).process(request: request, context: loadContext())
    }

    func generate(_ request: GenerationRequest) async throws -> AsyncThrowingStream<GenerationChunk, Error> {
        AsyncThrowingStream { continuation in
            for chunk in chunks {
                continuation.yield(chunk)
            }
            continuation.finish()
        }
    }
}

private func waitForAsync<T: Sendable>(
    _ operation: @escaping @Sendable () async throws -> T
) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = AsyncResultBox<T>()

    Task {
        do {
            box.result = Result<T, Error>.success(try await operation())
        } catch {
            box.result = Result<T, Error>.failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    return try box.result!.get()
}

private final class AsyncResultBox<T: Sendable>: @unchecked Sendable {
    var result: Result<T, Error>?
}

fileprivate func normalizedRoutePath(_ target: String) -> String {
    var path = target
    if let queryIndex = path.firstIndex(of: "?") {
        path = String(path[..<queryIndex])
    }
    if path.count > 1, path.hasSuffix("/") {
        path.removeLast()
    }
    return path.isEmpty ? "/" : path
}
