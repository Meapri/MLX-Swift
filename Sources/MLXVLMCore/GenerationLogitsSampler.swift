import Foundation

public struct SampledToken: Codable, Equatable, Sendable {
    public let tokenID: Int
    public let probability: Double
    public let logProbability: Double
    public let rank: Int
    public let sampler: String

    public init(
        tokenID: Int,
        probability: Double,
        logProbability: Double,
        rank: Int,
        sampler: String
    ) {
        self.tokenID = tokenID
        self.probability = probability
        self.logProbability = logProbability
        self.rank = rank
        self.sampler = sampler
    }
}

public enum GenerationLogitsSamplerError: Error, CustomStringConvertible, Equatable {
    case emptyLogits
    case nonFiniteLogit(Int, Double)
    case unsupportedAdvancedSampler(String)
    case noCandidateTokens

    public var description: String {
        switch self {
        case .emptyLogits:
            return "Cannot sample from an empty logits vector."
        case .nonFiniteLogit(let index, let value):
            return "Cannot sample from non-finite logit at index \(index): \(value)."
        case .unsupportedAdvancedSampler(let sampler):
            return "Sampler '\(sampler)' requires an advanced implementation that is not available in the Swift logits sampler."
        case .noCandidateTokens:
            return "No candidate tokens remain after applying sampling filters."
        }
    }
}

public struct SeededLogitsRandomNumberGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: Int) {
        let value = UInt64(bitPattern: Int64(seed))
        self.state = value == 0 ? 0x9E37_79B9_7F4A_7C15 : value
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}

public struct GenerationLogitsSampler: Sendable {
    public let plan: GenerationSamplingPlan

    public init(plan: GenerationSamplingPlan) {
        self.plan = plan
    }

    public func sample(
        logits: [Double],
        recentTokenIDs: [Int] = [],
        newlineTokenID: Int? = nil,
        logitBias: [Int: Double] = [:]
    ) throws -> SampledToken {
        var generator = SeededLogitsRandomNumberGenerator(seed: plan.seed)
        return try sample(
            logits: logits,
            recentTokenIDs: recentTokenIDs,
            newlineTokenID: newlineTokenID,
            logitBias: logitBias,
            generator: &generator
        )
    }

    public func sample<R: RandomNumberGenerator>(
        logits: [Double],
        recentTokenIDs: [Int] = [],
        newlineTokenID: Int? = nil,
        logitBias: [Int: Double] = [:],
        generator: inout R
    ) throws -> SampledToken {
        guard !logits.isEmpty else {
            throw GenerationLogitsSamplerError.emptyLogits
        }
        for (index, value) in logits.enumerated() where !value.isFinite {
            throw GenerationLogitsSamplerError.nonFiniteLogit(index, value)
        }
        guard !plan.requiresAdvancedSampler else {
            throw GenerationLogitsSamplerError.unsupportedAdvancedSampler(plan.sampler)
        }

        let biased = applyLogitBias(logits: logits, logitBias: logitBias)
        let adjusted = applyPenalties(
            logits: biased,
            recentTokenIDs: recentTokenIDs,
            newlineTokenID: newlineTokenID
        )
        if plan.deterministic || plan.temperature <= 0 {
            let ranked = rankedProbabilities(logits: adjusted, temperature: 1.0)
            guard let winner = ranked.first else {
                throw GenerationLogitsSamplerError.noCandidateTokens
            }
            return SampledToken(
                tokenID: winner.tokenID,
                probability: winner.probability,
                logProbability: log(max(winner.probability, Double.leastNonzeroMagnitude)),
                rank: 1,
                sampler: "greedy"
            )
        }

        let temperature = max(plan.temperature, Double.leastNonzeroMagnitude)
        let ranked = applyFilters(to: rankedProbabilities(logits: adjusted, temperature: temperature))
        guard !ranked.isEmpty else {
            throw GenerationLogitsSamplerError.noCandidateTokens
        }
        let normalized = normalize(ranked)
        let draw = Double.random(in: 0..<1, using: &generator)
        var cumulative = 0.0
        for (index, candidate) in normalized.enumerated() {
            cumulative += candidate.probability
            if draw < cumulative || index == normalized.count - 1 {
                return SampledToken(
                    tokenID: candidate.tokenID,
                    probability: candidate.probability,
                    logProbability: log(max(candidate.probability, Double.leastNonzeroMagnitude)),
                    rank: index + 1,
                    sampler: plan.sampler
                )
            }
        }
        throw GenerationLogitsSamplerError.noCandidateTokens
    }

    public func rankedTokenProbabilities(
        logits: [Double],
        recentTokenIDs: [Int] = [],
        newlineTokenID: Int? = nil,
        logitBias: [Int: Double] = [:],
        applySamplingFilters: Bool = false
    ) throws -> [SampledToken] {
        guard !logits.isEmpty else {
            throw GenerationLogitsSamplerError.emptyLogits
        }
        for (index, value) in logits.enumerated() where !value.isFinite {
            throw GenerationLogitsSamplerError.nonFiniteLogit(index, value)
        }
        guard !plan.requiresAdvancedSampler else {
            throw GenerationLogitsSamplerError.unsupportedAdvancedSampler(plan.sampler)
        }

        let biased = applyLogitBias(logits: logits, logitBias: logitBias)
        let adjusted = applyPenalties(
            logits: biased,
            recentTokenIDs: recentTokenIDs,
            newlineTokenID: newlineTokenID
        )
        let temperature = plan.deterministic || plan.temperature <= 0
            ? 1.0
            : max(plan.temperature, Double.leastNonzeroMagnitude)
        let ranked = applySamplingFilters
            ? normalize(applyFilters(to: rankedProbabilities(logits: adjusted, temperature: temperature)))
            : rankedProbabilities(logits: adjusted, temperature: temperature)
        return ranked.enumerated().map { index, candidate in
            SampledToken(
                tokenID: candidate.tokenID,
                probability: candidate.probability,
                logProbability: log(max(candidate.probability, Double.leastNonzeroMagnitude)),
                rank: index + 1,
                sampler: plan.sampler
            )
        }
    }

    private func applyLogitBias(
        logits: [Double],
        logitBias: [Int: Double]
    ) -> [Double] {
        guard !logitBias.isEmpty else {
            return logits
        }
        var result = logits
        for (tokenID, bias) in logitBias where bias.isFinite && result.indices.contains(tokenID) {
            result[tokenID] += bias
        }
        return result
    }

    private func applyPenalties(
        logits: [Double],
        recentTokenIDs: [Int],
        newlineTokenID: Int?
    ) -> [Double] {
        var result = logits
        let repeatWindow = if let repeatLastN = plan.repeatLastN, repeatLastN > 0 {
            Array(recentTokenIDs.suffix(repeatLastN))
        } else {
            recentTokenIDs
        }
        let counts = Dictionary(grouping: repeatWindow, by: { $0 }).mapValues(\.count)

        if let repetitionPenalty = plan.repetitionPenalty, repetitionPenalty > 0, repetitionPenalty != 1 {
            for tokenID in counts.keys where result.indices.contains(tokenID) {
                if result[tokenID] >= 0 {
                    result[tokenID] /= repetitionPenalty
                } else {
                    result[tokenID] *= repetitionPenalty
                }
            }
        }
        if let presencePenalty = plan.presencePenalty, presencePenalty != 0 {
            for tokenID in counts.keys where result.indices.contains(tokenID) {
                result[tokenID] -= presencePenalty
            }
        }
        if let frequencyPenalty = plan.frequencyPenalty, frequencyPenalty != 0 {
            for (tokenID, count) in counts where result.indices.contains(tokenID) {
                result[tokenID] -= Double(count) * frequencyPenalty
            }
        }
        if plan.penalizeNewline == true,
           let newlineTokenID,
           result.indices.contains(newlineTokenID)
        {
            result[newlineTokenID] -= 1.0
        }
        return result
    }

    private func rankedProbabilities(
        logits: [Double],
        temperature: Double
    ) -> [CandidateToken] {
        let scaled = logits.map { $0 / temperature }
        let maxLogit = scaled.max() ?? 0
        let expValues = scaled.map { exp($0 - maxLogit) }
        let total = expValues.reduce(0, +)
        let denominator = total > 0 ? total : 1
        return expValues.enumerated()
            .map { index, value in
                CandidateToken(tokenID: index, probability: value / denominator, logit: logits[index])
            }
            .sorted {
                if $0.probability == $1.probability {
                    return $0.tokenID < $1.tokenID
                }
                return $0.probability > $1.probability
            }
    }

    private func applyFilters(to ranked: [CandidateToken]) -> [CandidateToken] {
        var result = ranked
        if plan.topK > 0, result.count > plan.topK {
            result = Array(result.prefix(plan.topK))
        }
        if let minP = plan.minP, minP > 0, let maxProbability = result.first?.probability {
            let threshold = maxProbability * minP
            result = result.filter { $0.probability >= threshold }
        }
        if plan.topP < 1.0 {
            var cumulative = 0.0
            var nucleus: [CandidateToken] = []
            for candidate in result {
                nucleus.append(candidate)
                cumulative += candidate.probability
                if cumulative >= plan.topP {
                    break
                }
            }
            result = nucleus
        }
        result = applyTypicalP(to: result)
        result = applyTailFree(to: result)
        return result
    }

    private func applyTypicalP(to ranked: [CandidateToken]) -> [CandidateToken] {
        guard let typicalP = plan.typicalP,
              typicalP > 0,
              typicalP < 1,
              ranked.count > 1
        else {
            return ranked
        }
        let entropy = ranked.reduce(0.0) { total, candidate in
            let probability = max(candidate.probability, Double.leastNonzeroMagnitude)
            return total - probability * log(probability)
        }
        let typicalOrder = ranked.sorted {
            let left = abs(-log(max($0.probability, Double.leastNonzeroMagnitude)) - entropy)
            let right = abs(-log(max($1.probability, Double.leastNonzeroMagnitude)) - entropy)
            if left == right {
                return $0.tokenID < $1.tokenID
            }
            return left < right
        }
        var kept: Set<Int> = []
        var cumulative = 0.0
        for candidate in typicalOrder {
            kept.insert(candidate.tokenID)
            cumulative += candidate.probability
            if cumulative >= typicalP {
                break
            }
        }
        return ranked.filter { kept.contains($0.tokenID) }
    }

    private func applyTailFree(to ranked: [CandidateToken]) -> [CandidateToken] {
        guard let tfsZ = plan.tfsZ,
              tfsZ > 0,
              tfsZ < 1,
              ranked.count > 2
        else {
            return ranked
        }
        let probabilities = ranked.map(\.probability)
        let firstDerivatives = zip(probabilities, probabilities.dropFirst()).map { left, right in
            max(left - right, 0)
        }
        let secondDerivatives = zip(firstDerivatives, firstDerivatives.dropFirst()).map { left, right in
            abs(left - right)
        }
        let total = secondDerivatives.reduce(0, +)
        guard total > 0 else {
            return [ranked[0]]
        }

        var keepCount = 1
        var cumulative = 0.0
        for value in secondDerivatives {
            cumulative += value / total
            keepCount += 1
            if cumulative >= tfsZ {
                break
            }
        }
        return Array(ranked.prefix(max(1, min(keepCount, ranked.count))))
    }

    private func normalize(_ candidates: [CandidateToken]) -> [CandidateToken] {
        let total = candidates.reduce(0) { $0 + $1.probability }
        guard total > 0 else {
            return candidates
        }
        return candidates.map {
            CandidateToken(
                tokenID: $0.tokenID,
                probability: $0.probability / total,
                logit: $0.logit
            )
        }
    }
}

private struct CandidateToken: Equatable {
    let tokenID: Int
    let probability: Double
    let logit: Double
}
