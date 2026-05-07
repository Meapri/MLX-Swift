import Foundation

public struct GenerationSamplingPlan: Codable, Equatable, Sendable {
    public let sampler: String
    public let deterministic: Bool
    public let temperature: Double
    public let topP: Double
    public let topK: Int
    public let minP: Double?
    public let typicalP: Double?
    public let tfsZ: Double?
    public let seed: Int
    public let repetitionPenalty: Double?
    public let repeatLastN: Int?
    public let presencePenalty: Double?
    public let frequencyPenalty: Double?
    public let penalizeNewline: Bool?
    public let mirostat: Int?
    public let mirostatTau: Double?
    public let mirostatEta: Double?
    public let enabledFilters: [String]
    public let enabledPenalties: [String]
    public let requiresAdvancedSampler: Bool
    public let backendMinimumFeatures: [String]
    public let warnings: [String]

    public init(
        sampler: String,
        deterministic: Bool,
        temperature: Double,
        topP: Double,
        topK: Int,
        minP: Double?,
        typicalP: Double?,
        tfsZ: Double?,
        seed: Int,
        repetitionPenalty: Double?,
        repeatLastN: Int?,
        presencePenalty: Double?,
        frequencyPenalty: Double?,
        penalizeNewline: Bool?,
        mirostat: Int?,
        mirostatTau: Double?,
        mirostatEta: Double?,
        enabledFilters: [String],
        enabledPenalties: [String],
        requiresAdvancedSampler: Bool,
        backendMinimumFeatures: [String],
        warnings: [String]
    ) {
        self.sampler = sampler
        self.deterministic = deterministic
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.typicalP = typicalP
        self.tfsZ = tfsZ
        self.seed = seed
        self.repetitionPenalty = repetitionPenalty
        self.repeatLastN = repeatLastN
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.penalizeNewline = penalizeNewline
        self.mirostat = mirostat
        self.mirostatTau = mirostatTau
        self.mirostatEta = mirostatEta
        self.enabledFilters = enabledFilters
        self.enabledPenalties = enabledPenalties
        self.requiresAdvancedSampler = requiresAdvancedSampler
        self.backendMinimumFeatures = backendMinimumFeatures
        self.warnings = warnings
    }
}

public struct GenerationSamplingPlanner {
    public init() {}

    public func plan(parameters: GenerationParameters) -> GenerationSamplingPlan {
        let deterministic = parameters.temperature <= 0
        var filters: [String] = []
        var penalties: [String] = []
        var warnings: [String] = []

        if parameters.topK > 0 {
            filters.append("top-k")
        }
        if parameters.topP < 1.0 {
            filters.append("top-p")
        }
        if let minP = parameters.minP, minP > 0 {
            filters.append("min-p")
        }
        if let typicalP = parameters.typicalP, typicalP < 1.0 {
            filters.append("typical-p")
        }
        if let tfsZ = parameters.tfsZ, tfsZ > 0 {
            filters.append("tail-free")
        }

        if let repetitionPenalty = parameters.repetitionPenalty, repetitionPenalty != 1.0 {
            penalties.append("repetition")
        }
        if let presencePenalty = parameters.presencePenalty, presencePenalty != 0 {
            penalties.append("presence")
        }
        if let frequencyPenalty = parameters.frequencyPenalty, frequencyPenalty != 0 {
            penalties.append("frequency")
        }
        if parameters.repeatLastN != nil {
            penalties.append("repeat-window")
        }
        if parameters.penalizeNewline != nil {
            penalties.append("newline")
        }

        let sampler: String
        if let mirostat = parameters.mirostat, mirostat > 0 {
            sampler = "mirostat"
        } else if deterministic {
            sampler = "greedy"
        } else {
            sampler = "temperature"
        }

        let requiresAdvancedSampler = parameters.mirostat.map { $0 > 0 } == true ||
            parameters.typicalP != nil ||
            parameters.tfsZ != nil

        if deterministic, !filters.isEmpty {
            warnings.append("Sampling filters are preserved but have no effect for greedy decoding unless the backend overrides temperature semantics.")
        }
        if parameters.mirostat.map({ $0 > 0 }) == true {
            warnings.append("Mirostat requires a sampler implementation beyond basic greedy/temperature sampling.")
        }
        if parameters.typicalP != nil {
            warnings.append("typical_p requires an advanced probability filter.")
        }
        if parameters.tfsZ != nil {
            warnings.append("tfs_z requires a tail-free probability filter.")
        }

        var minimumFeatures = deterministic ? ["argmax-logits"] : ["random-categorical", "temperature"]
        minimumFeatures += filters
        minimumFeatures += penalties
        if parameters.seed != 0 {
            minimumFeatures.append("seeded-rng")
        }

        return GenerationSamplingPlan(
            sampler: sampler,
            deterministic: deterministic,
            temperature: parameters.temperature,
            topP: parameters.topP,
            topK: parameters.topK,
            minP: parameters.minP,
            typicalP: parameters.typicalP,
            tfsZ: parameters.tfsZ,
            seed: parameters.seed,
            repetitionPenalty: parameters.repetitionPenalty,
            repeatLastN: parameters.repeatLastN,
            presencePenalty: parameters.presencePenalty,
            frequencyPenalty: parameters.frequencyPenalty,
            penalizeNewline: parameters.penalizeNewline,
            mirostat: parameters.mirostat,
            mirostatTau: parameters.mirostatTau,
            mirostatEta: parameters.mirostatEta,
            enabledFilters: filters,
            enabledPenalties: penalties,
            requiresAdvancedSampler: requiresAdvancedSampler,
            backendMinimumFeatures: Array(NSOrderedSet(array: minimumFeatures).compactMap { $0 as? String }),
            warnings: warnings
        )
    }
}
