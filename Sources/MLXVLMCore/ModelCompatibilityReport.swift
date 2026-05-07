import Foundation

public enum CompatibilitySeverity: String, Codable, Equatable, Sendable {
    case info
    case warning
    case error
}

public struct CompatibilityCheck: Codable, Equatable, Sendable {
    public let id: String
    public let passed: Bool
    public let severity: CompatibilitySeverity
    public let message: String

    public init(id: String, passed: Bool, severity: CompatibilitySeverity, message: String) {
        self.id = id
        self.passed = passed
        self.severity = severity
        self.message = message
    }
}

public struct ModelCompatibilityReport: Codable, Equatable, Sendable {
    public let descriptor: ModelDescriptor
    public let backend: BackendStatus
    public let checks: [CompatibilityCheck]

    public var metadataReady: Bool {
        checks.filter { $0.severity == .error && $0.id != "backend-ready" }.allSatisfy(\.passed)
    }

    public var generationReady: Bool {
        backend.ready && metadataReady
    }

    public init(descriptor: ModelDescriptor, backend: BackendStatus, checks: [CompatibilityCheck]) {
        self.descriptor = descriptor
        self.backend = backend
        self.checks = checks
    }
}

public enum ModelCompatibilityValidator {
    public static func validate(
        descriptor: ModelDescriptor,
        backend: BackendStatus = .compatibilityShell
    ) -> ModelCompatibilityReport {
        var checks: [CompatibilityCheck] = []

        checks.append(
            CompatibilityCheck(
                id: "model-type-known",
                passed: descriptor.isKnownModelType,
                severity: .error,
                message: descriptor.isKnownModelType
                    ? "Model type \(descriptor.canonicalModelType) is known to the Swift compatibility registry."
                    : "Model type \(descriptor.canonicalModelType) is not in the Swift compatibility registry."
            )
        )

        let configNormalization = descriptor.configNormalization
        let normalizedConfigDetails = configNormalization.warnings.isEmpty
            ? "Config already exposes backend-ready text/vision/audio sections."
            : configNormalization.warnings.joined(separator: " ")
        checks.append(
            CompatibilityCheck(
                id: "config-normalization",
                passed: true,
                severity: .warning,
                message: normalizedConfigDetails
            )
        )

        checks.append(
            CompatibilityCheck(
                id: "weights-present",
                passed: !descriptor.weightFiles.isEmpty,
                severity: .error,
                message: descriptor.weightFiles.isEmpty
                    ? "No safetensors weight files were found."
                    : "Found \(descriptor.weightFiles.count) safetensors weight file(s)."
            )
        )

        let unreadableSafetensors = descriptor.safetensorsMetadata.filter { !$0.isReadable }
        checks.append(
            CompatibilityCheck(
                id: "safetensors-readable",
                passed: unreadableSafetensors.isEmpty,
                severity: .error,
                message: unreadableSafetensors.isEmpty
                    ? "All safetensors headers are readable."
                    : "Unreadable safetensors files: \(unreadableSafetensors.map(\.name).joined(separator: ", "))."
            )
        )

        if let weightIndex = descriptor.weightIndex {
            let presentShards = Set(descriptor.weightFiles.map(\.name))
            let missingShards = weightIndex.shardNames.filter { !presentShards.contains($0) }
            checks.append(
                CompatibilityCheck(
                    id: "weight-index-shards-present",
                    passed: missingShards.isEmpty,
                    severity: .error,
                    message: missingShards.isEmpty
                        ? "All model.safetensors.index.json shards are present on disk."
                        : "Missing safetensors shards referenced by index: \(missingShards.joined(separator: ", "))."
                )
            )
        }

        let catalog = WeightCatalogBuilder().catalog(for: descriptor)
        checks.append(
            CompatibilityCheck(
                id: "weight-catalog-readable",
                passed: catalog.unreadableShards.isEmpty,
                severity: .error,
                message: catalog.unreadableShards.isEmpty
                    ? "Weight catalog contains \(catalog.tensorCount) tensor metadata entr\(catalog.tensorCount == 1 ? "y" : "ies")."
                    : "Unreadable weight catalog shard(s): \(catalog.unreadableShards.map(\.shardName).joined(separator: ", "))."
            )
        )
        checks.append(
            CompatibilityCheck(
                id: "weight-catalog-unique",
                passed: catalog.duplicateOriginalKeys.isEmpty && catalog.duplicateSanitizedKeys.isEmpty,
                severity: .error,
                message: catalog.duplicateOriginalKeys.isEmpty && catalog.duplicateSanitizedKeys.isEmpty
                    ? "Weight catalog tensor keys are unique after model-specific sanitization."
                    : "Duplicate tensor key(s) found in weight catalog: original=\(catalog.duplicateOriginalKeys.prefix(5).joined(separator: ", ")), sanitized=\(catalog.duplicateSanitizedKeys.prefix(5).joined(separator: ", "))."
            )
        )

        let weightDataCatalog = WeightDataCatalogBuilder().catalog(for: descriptor)
        checks.append(
            CompatibilityCheck(
                id: "weight-data-readable",
                passed: weightDataCatalog.unreadableTensorCount == 0,
                severity: .error,
                message: weightDataCatalog.unreadableTensorCount == 0
                    ? "All safetensors tensor payload byte ranges are readable."
                    : "\(weightDataCatalog.unreadableTensorCount) tensor payload byte range(s) are not readable."
            )
        )
        checks.append(
            CompatibilityCheck(
                id: "weight-data-byte-counts",
                passed: weightDataCatalog.byteMismatchCount == 0,
                severity: .error,
                message: weightDataCatalog.byteMismatchCount == 0
                    ? "All tensor payload byte counts match dtype width multiplied by shape."
                    : "\(weightDataCatalog.byteMismatchCount) tensor payload byte count(s) do not match dtype width multiplied by shape."
            )
        )
        let mlxWeightLoadPlan = MLXWeightLoadPlan(descriptor: descriptor)
        checks.append(
            CompatibilityCheck(
                id: "mlx-weight-loadable",
                passed: mlxWeightLoadPlan.canLoadAllTensorsAsMLXArrays,
                severity: .error,
                message: mlxWeightLoadPlan.canLoadAllTensorsAsMLXArrays
                    ? "All \(mlxWeightLoadPlan.tensorCount) safetensors tensor payload(s) have MLX dtype mappings and readable byte ranges."
                    : "Only \(mlxWeightLoadPlan.loadableTensorCount)/\(mlxWeightLoadPlan.tensorCount) safetensors tensor payload(s) can be mapped to MLX arrays; unsupported_dtype=\(mlxWeightLoadPlan.unsupportedDTypeKeys.count), unreadable=\(mlxWeightLoadPlan.unreadableTensorCount)."
            )
        )

        if descriptor.weightIndex != nil {
            if let metadataTotalSize = descriptor.weightIndex?.metadataTotalSize {
                checks.append(
                    CompatibilityCheck(
                        id: "weight-index-total-size",
                        passed: metadataTotalSize == weightDataCatalog.totalReadableBytes,
                        severity: .warning,
                        message: metadataTotalSize == weightDataCatalog.totalReadableBytes
                            ? "model.safetensors.index.json metadata total_size matches readable tensor payload bytes."
                            : "model.safetensors.index.json metadata total_size \(metadataTotalSize) does not match readable tensor payload bytes \(weightDataCatalog.totalReadableBytes)."
                    )
                )
            }
            checks.append(
                CompatibilityCheck(
                    id: "weight-index-tensors-present",
                    passed: catalog.missingIndexEntries.isEmpty,
                    severity: .error,
                    message: catalog.missingIndexEntries.isEmpty
                        ? "All model.safetensors.index.json tensors are present in readable shard headers."
                        : "Tensor(s) referenced by index but missing from shard headers: \(catalog.missingIndexEntries.prefix(5).joined(separator: ", "))."
                )
            )
        }

        checks.append(
            CompatibilityCheck(
                id: "tokenizer-present",
                passed: descriptor.tokenizerMetadata.hasTokenizerConfig ||
                    descriptor.tokenizerMetadata.hasTokenizerJSON ||
                    descriptor.tokenizerMetadata.hasTokenizerModel ||
                    descriptor.tokenizerMetadata.hasTiktoken ||
                    descriptor.tokenizerMetadata.hasVocabJSON ||
                    descriptor.tokenizerMetadata.hasVocabTXT,
                severity: .warning,
                message: "Tokenizer files: config=\(descriptor.tokenizerMetadata.hasTokenizerConfig), tokenizer.json=\(descriptor.tokenizerMetadata.hasTokenizerJSON), tokenizer.model=\(descriptor.tokenizerMetadata.hasTokenizerModel), tokenizer.tiktoken=\(descriptor.tokenizerMetadata.hasTiktoken), vocab.json=\(descriptor.tokenizerMetadata.hasVocabJSON), merges.txt=\(descriptor.tokenizerMetadata.hasMergesTXT), vocab.txt=\(descriptor.tokenizerMetadata.hasVocabTXT)."
            )
        )

        if let tokenizerJSON = descriptor.tokenizerMetadata.tokenizerJSONMetadata {
            checks.append(
                CompatibilityCheck(
                    id: "tokenizer-json-readable",
                    passed: tokenizerJSON.isReadable,
                    severity: .error,
                    message: tokenizerJSON.isReadable
                        ? "tokenizer.json model=\(tokenizerJSON.modelType ?? "unknown"), vocab=\(tokenizerJSON.vocabCount.map(String.init) ?? "unknown"), merges=\(tokenizerJSON.mergeCount.map(String.init) ?? "unknown")."
                        : "tokenizer.json could not be read: \(tokenizerJSON.error ?? "unknown error")."
                )
            )

            if let configVocabSize = descriptor.configVocabSize,
               let tokenizerVocabCount = tokenizerJSON.vocabCount
            {
                checks.append(
                    CompatibilityCheck(
                        id: "tokenizer-vocab-compatible",
                        passed: tokenizerVocabCount <= configVocabSize,
                        severity: .warning,
                        message: tokenizerVocabCount <= configVocabSize
                            ? "Tokenizer vocab count \(tokenizerVocabCount) fits config vocab_size \(configVocabSize)."
                            : "Tokenizer vocab count \(tokenizerVocabCount) exceeds config vocab_size \(configVocabSize)."
                    )
                )
            }
        }

        if let tokenizerCatalog = TokenizerCatalogBuilder().catalog(for: descriptor) {
            checks.append(
                CompatibilityCheck(
                    id: "tokenizer-catalog-readable",
                    passed: tokenizerCatalog.error == nil,
                    severity: .error,
                    message: tokenizerCatalog.error == nil
                        ? "Tokenizer catalog contains \(tokenizerCatalog.tokenCount) token(s), including \(tokenizerCatalog.specialTokenCount) special token(s)."
                        : "Tokenizer catalog could not be read: \(tokenizerCatalog.error ?? "unknown error")."
                )
            )
            checks.append(
                CompatibilityCheck(
                    id: "tokenizer-catalog-unique",
                    passed: tokenizerCatalog.duplicateIDs.isEmpty,
                    severity: .warning,
                    message: tokenizerCatalog.duplicateIDs.isEmpty
                        ? "Tokenizer catalog IDs are unique."
                        : "Tokenizer catalog has duplicate ID(s): \(tokenizerCatalog.duplicateIDs.prefix(5).map(String.init).joined(separator: ", "))."
                )
            )
        }

        checks.append(
            CompatibilityCheck(
                id: "chat-template-present",
                passed: descriptor.hasChatTemplate,
                severity: .warning,
                message: descriptor.hasChatTemplate
                    ? "Chat template source: \(descriptor.tokenizerMetadata.chatTemplateSource ?? "config")."
                    : "No chat template was found; Swift will use plain prompt fallback."
            )
        )

        let chatTemplatePlan = ChatTemplatePlanner().plan(descriptor: descriptor)
        checks.append(
            CompatibilityCheck(
                id: "chat-template-renderer",
                passed: chatTemplatePlan.canRenderNatively,
                severity: .warning,
                message: chatTemplatePlan.canRenderNatively
                    ? "Chat template can use Swift renderer \(chatTemplatePlan.requiredRenderer)."
                    : "Chat template requires \(chatTemplatePlan.requiredRenderer); Swift will use \(chatTemplatePlan.fallbackRenderer) fallback until that renderer is linked."
            )
        )

        let adapterMetadata = descriptor.adapterMetadata
        let adapterReady = !adapterMetadata.hasAdapterConfig && !adapterMetadata.hasAdapterWeights ||
            adapterMetadata.isLoadable
        checks.append(
            CompatibilityCheck(
                id: "adapter-metadata",
                passed: adapterReady,
                severity: .warning,
                message: adapterReady
                    ? (adapterMetadata.isLoadable
                        ? "LoRA adapter metadata is present and adapters.safetensors is readable."
                        : "No LoRA adapter metadata was found.")
                    : "LoRA adapter metadata is incomplete: \(adapterMetadata.warnings.joined(separator: " "))"
            )
        )

        let capabilities = ModelCapabilityPlanner().plan(descriptor: descriptor)
        checks.append(
            CompatibilityCheck(
                id: "generation-api-compatible",
                passed: capabilities.supportsOllamaGenerationAPI,
                severity: .warning,
                message: capabilities.supportsOllamaGenerationAPI
                    ? "Model capability plan supports text generation through Ollama/OpenAI-compatible generation endpoints."
                    : "Model type \(descriptor.canonicalModelType) uses \(capabilities.primaryTask); use \(capabilities.preferredSwiftEntryPoint) instead of Ollama generation endpoints."
            )
        )

        if descriptor.canonicalModelType == "qwen2_vl" || descriptor.canonicalModelType == "qwen2_5_vl" {
            let hasImageToken = descriptor.tokenizerMetadata.imageTokenID != nil
            let hasVideoToken = descriptor.tokenizerMetadata.videoTokenID != nil
            checks.append(
                CompatibilityCheck(
                    id: "qwen-vl-special-tokens",
                    passed: hasImageToken && hasVideoToken,
                    severity: .warning,
                    message: "Qwen VL token IDs: image=\(descriptor.tokenizerMetadata.imageTokenID.map(String.init) ?? "missing"), video=\(descriptor.tokenizerMetadata.videoTokenID.map(String.init) ?? "missing")."
                )
            )

            if let report = QwenVLWeightSanitizer.report(descriptor: descriptor) {
                checks.append(
                    CompatibilityCheck(
                        id: "qwen-vl-weight-map",
                        passed: report.unknownCount == 0,
                        severity: .warning,
                        message: "Qwen VL weight roles: language=\(report.languageModelCount), head=\(report.languageHeadCount), vision=\(report.visionTowerCount), unknown=\(report.unknownCount)."
                    )
                )
            } else {
                checks.append(
                    CompatibilityCheck(
                        id: "qwen-vl-weight-map",
                        passed: false,
                        severity: .warning,
                        message: "No model.safetensors.index.json found; weight role validation is skipped."
                    )
                )
            }

            if let qwenConfig = try? QwenVLModelConfig.load(fromModelDirectory: descriptor.path) {
                let architecture = QwenVLArchitecturePlanner().plan(config: qwenConfig, weightCatalog: catalog)
                let passed = architecture.missingRequiredKeys.isEmpty && architecture.mismatchedShapeKeys.isEmpty
                checks.append(
                    CompatibilityCheck(
                        id: "qwen-vl-core-tensor-coverage",
                        passed: passed,
                        severity: .warning,
                        message: passed
                            ? "Qwen VL core tensor coverage is complete for \(architecture.expectedCoreTensors.count) expected tensor(s)."
                            : "Qwen VL core tensor coverage \(architecture.presentCoreTensorCount)/\(architecture.expectedCoreTensors.count); missing=\(architecture.missingRequiredKeys.count), shape_mismatch=\(architecture.mismatchedShapeKeys.count)."
                    )
                )
            } else {
                checks.append(
                    CompatibilityCheck(
                        id: "qwen-vl-core-tensor-coverage",
                        passed: false,
                        severity: .warning,
                        message: "Qwen VL config could not be parsed; core tensor coverage was skipped."
                    )
                )
            }
        }

        checks.append(
            CompatibilityCheck(
                id: "backend-ready",
                passed: backend.ready,
                severity: .error,
                message: backend.ready
                    ? "MLX Swift generation backend is linked."
                    : backend.message
            )
        )

        return ModelCompatibilityReport(
            descriptor: descriptor,
            backend: backend,
            checks: checks
        )
    }
}
