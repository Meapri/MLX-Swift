import Foundation

public struct SpeculativeWalkResult: Codable, Equatable, Sendable {
    public let acceptedCount: Int
    public let newTokens: [Int]

    public init(acceptedCount: Int, newTokens: [Int]) {
        self.acceptedCount = acceptedCount
        self.newTokens = newTokens
    }
}

public struct MTPSpeculativeRoundPlan: Codable, Equatable, Sendable {
    public let walk: SpeculativeWalkResult
    public let blockSize: Int
    public let emittedBeforeRound: Int
    public let emittedAfterRound: Int
    public let positionBeforeRound: Int
    public let positionAfterRound: Int
    public let hiddenSlotIndex: Int
    public let rejectedTokenCount: Int
    public let rollbackRequired: Bool
    public let sharedKVValidLength: Int
    public let nextBonusToken: Int?
    public let finished: Bool

    public init(
        walk: SpeculativeWalkResult,
        blockSize: Int,
        emittedBeforeRound: Int,
        emittedAfterRound: Int,
        positionBeforeRound: Int,
        positionAfterRound: Int,
        hiddenSlotIndex: Int,
        rejectedTokenCount: Int,
        rollbackRequired: Bool,
        sharedKVValidLength: Int,
        nextBonusToken: Int?,
        finished: Bool
    ) {
        self.walk = walk
        self.blockSize = blockSize
        self.emittedBeforeRound = emittedBeforeRound
        self.emittedAfterRound = emittedAfterRound
        self.positionBeforeRound = positionBeforeRound
        self.positionAfterRound = positionAfterRound
        self.hiddenSlotIndex = hiddenSlotIndex
        self.rejectedTokenCount = rejectedTokenCount
        self.rollbackRequired = rollbackRequired
        self.sharedKVValidLength = sharedKVValidLength
        self.nextBonusToken = nextBonusToken
        self.finished = finished
    }
}

public struct MTPBatchSpeculativeRoundPlan: Codable, Equatable, Sendable {
    public let rows: [MTPSpeculativeRoundPlan]
    public let maxAcceptedCount: Int
    public let globalRejectedTokenCount: Int
    public let rollbackRequired: Bool
    public let activeRowIndexesAfterRound: [Int]

    public init(
        rows: [MTPSpeculativeRoundPlan],
        maxAcceptedCount: Int,
        globalRejectedTokenCount: Int,
        rollbackRequired: Bool,
        activeRowIndexesAfterRound: [Int]
    ) {
        self.rows = rows
        self.maxAcceptedCount = maxAcceptedCount
        self.globalRejectedTokenCount = globalRejectedTokenCount
        self.rollbackRequired = rollbackRequired
        self.activeRowIndexesAfterRound = activeRowIndexesAfterRound
    }
}

public struct MTPSpeculativeRoundEmission: Codable, Equatable, Sendable {
    public let roundIndex: Int
    public let inputBonusToken: Int
    public let plan: MTPSpeculativeRoundPlan
    public let emittedTokens: [Int]

    public init(
        roundIndex: Int,
        inputBonusToken: Int,
        plan: MTPSpeculativeRoundPlan,
        emittedTokens: [Int]
    ) {
        self.roundIndex = roundIndex
        self.inputBonusToken = inputBonusToken
        self.plan = plan
        self.emittedTokens = emittedTokens
    }
}

public struct MTPBatchSpeculativeRoundEmission: Codable, Equatable, Sendable {
    public let roundIndex: Int
    public let inputBonusTokens: [Int]
    public let plan: MTPBatchSpeculativeRoundPlan
    public let emittedTokenColumns: [[Int?]]

    public init(
        roundIndex: Int,
        inputBonusTokens: [Int],
        plan: MTPBatchSpeculativeRoundPlan,
        emittedTokenColumns: [[Int?]]
    ) {
        self.roundIndex = roundIndex
        self.inputBonusTokens = inputBonusTokens
        self.plan = plan
        self.emittedTokenColumns = emittedTokenColumns
    }
}

public struct MTPSpeculativeSession: Codable, Equatable, Sendable {
    public let maxTokens: Int
    public let draftBlockSize: Int
    public private(set) var bonusToken: Int
    public private(set) var emittedTokenCount: Int
    public private(set) var position: Int
    public private(set) var sharedKVSequenceLength: Int
    public private(set) var roundIndex: Int
    public private(set) var finished: Bool

    public init(
        firstBonusToken: Int,
        maxTokens: Int,
        draftBlockSize: Int,
        initialPosition: Int,
        initialSharedKVSequenceLength: Int? = nil,
        firstBonusAlreadyEmitted: Bool = true
    ) {
        self.maxTokens = max(0, maxTokens)
        self.draftBlockSize = max(1, draftBlockSize)
        self.bonusToken = firstBonusToken
        self.emittedTokenCount = firstBonusAlreadyEmitted ? min(1, max(0, maxTokens)) : 0
        self.position = max(0, initialPosition)
        self.sharedKVSequenceLength = max(
            1,
            initialSharedKVSequenceLength ?? max(0, initialPosition)
        )
        self.roundIndex = 0
        self.finished = self.emittedTokenCount >= max(0, maxTokens)
    }

    public mutating func nextRound(
        draftTokens: [Int],
        targetTokens: [Int],
        sharedKVSequenceLengthAfterVerify: Int? = nil
    ) -> MTPSpeculativeRoundEmission? {
        guard !finished, emittedTokenCount < maxTokens else {
            finished = true
            return nil
        }

        let blockSize = min(draftBlockSize, maxTokens - emittedTokenCount + 1)
        guard blockSize > 1 else {
            finished = true
            return nil
        }

        let draft = Array(draftTokens.prefix(blockSize - 1))
        let target = Array(targetTokens.prefix(blockSize))
        let verifyKVLength = sharedKVSequenceLengthAfterVerify ?? (position + blockSize)
        let inputBonus = bonusToken
        let plan = SpeculativeDecoding.mtpRoundPlan(
            draftTokens: draft,
            targetTokens: target,
            emittedBeforeRound: emittedTokenCount,
            maxTokens: maxTokens,
            blockSize: blockSize,
            positionBeforeRound: position,
            sharedKVSequenceLength: verifyKVLength
        )

        emittedTokenCount = plan.emittedAfterRound
        position = plan.positionAfterRound
        sharedKVSequenceLength = plan.sharedKVValidLength
        if let nextBonus = plan.nextBonusToken {
            bonusToken = nextBonus
        }
        roundIndex += 1
        finished = plan.finished

        return MTPSpeculativeRoundEmission(
            roundIndex: roundIndex - 1,
            inputBonusToken: inputBonus,
            plan: plan,
            emittedTokens: plan.walk.newTokens
        )
    }
}

public struct MTPBatchSpeculativeSession: Codable, Equatable, Sendable {
    public let maxTokens: Int
    public let draftBlockSize: Int
    public private(set) var bonusTokens: [Int]
    public private(set) var emittedTokenCounts: [Int]
    public private(set) var positions: [Int]
    public private(set) var sharedKVSequenceLength: Int
    public private(set) var finished: [Bool]
    public private(set) var activeRowIndexes: [Int]
    public private(set) var roundIndex: Int

    public init(
        firstBonusTokens: [Int],
        maxTokens: Int,
        draftBlockSize: Int,
        initialPositions: [Int],
        initialSharedKVSequenceLength: Int? = nil,
        firstBonusAlreadyEmitted: Bool = true
    ) {
        let rowCount = min(firstBonusTokens.count, initialPositions.count)
        self.maxTokens = max(0, maxTokens)
        self.draftBlockSize = max(1, draftBlockSize)
        self.bonusTokens = Array(firstBonusTokens.prefix(rowCount))
        self.emittedTokenCounts = Array(
            repeating: firstBonusAlreadyEmitted ? min(1, max(0, maxTokens)) : 0,
            count: rowCount
        )
        self.positions = initialPositions.prefix(rowCount).map { max(0, $0) }
        self.sharedKVSequenceLength = max(
            1,
            initialSharedKVSequenceLength ?? (self.positions.max() ?? 0)
        )
        self.finished = Array(repeating: false, count: rowCount)
        self.activeRowIndexes = Array(0 ..< rowCount)
        self.roundIndex = 0

        if self.maxTokens == 0 {
            self.finished = Array(repeating: true, count: rowCount)
            self.activeRowIndexes = []
        }
    }

    public var allFinished: Bool {
        activeRowIndexes.isEmpty || activeRowIndexes.allSatisfy { finished[$0] }
    }

    public mutating func nextRound(
        draftTokens: [[Int]],
        targetTokens: [[Int]],
        eosTokenIDs: Set<Int> = [],
        stoppedRowIndexes: Set<Int> = [],
        canFilterFinishedRows: Bool = true,
        sharedKVSequenceLengthAfterVerify: Int? = nil
    ) -> MTPBatchSpeculativeRoundEmission? {
        guard !allFinished else {
            activeRowIndexes = []
            return nil
        }

        let nActive = activeRowIndexes.count
        guard nActive > 0 else {
            return nil
        }

        let remainingBlockSizes = activeRowIndexes.map { row in
            max(1, maxTokens - emittedTokenCounts[row] + 1)
        }
        let blockSize = min(draftBlockSize, remainingBlockSizes.min() ?? draftBlockSize)
        guard blockSize > 1 else {
            activeRowIndexes = []
            return nil
        }

        let rowCount = min(nActive, draftTokens.count, targetTokens.count)
        guard rowCount > 0 else {
            return nil
        }

        let activeRows = Array(activeRowIndexes.prefix(rowCount))
        let activeDraft = (0 ..< rowCount).map { Array(draftTokens[$0].prefix(blockSize - 1)) }
        let activeTarget = (0 ..< rowCount).map { Array(targetTokens[$0].prefix(blockSize)) }
        let activeEmitted = activeRows.map { emittedTokenCounts[$0] }
        let activePositions = activeRows.map { positions[$0] }
        let verifyKVLength = sharedKVSequenceLengthAfterVerify ?? ((activePositions.max() ?? 0) + blockSize)
        let inputBonus = activeRows.map { bonusTokens[$0] }

        let plan = SpeculativeDecoding.mtpBatchRoundPlan(
            draftTokens: activeDraft,
            targetTokens: activeTarget,
            emittedBeforeRound: activeEmitted,
            maxTokens: maxTokens,
            blockSize: blockSize,
            positionsBeforeRound: activePositions,
            sharedKVSequenceLength: verifyKVLength,
            activeRowIndexes: activeRows
        )

        let maxNewTokenCount = plan.rows.map { $0.walk.newTokens.count }.max() ?? 0
        var emittedColumns: [[Int?]] = []
        for tokenIndex in 0 ..< maxNewTokenCount {
            var column = Array<Int?>(repeating: nil, count: bonusTokens.count)
            for (slot, row) in activeRows.enumerated() {
                let tokens = plan.rows[slot].walk.newTokens
                guard tokenIndex < tokens.count, !finished[row] else {
                    continue
                }
                let token = tokens[tokenIndex]
                column[row] = token
                emittedTokenCounts[row] += 1
                if emittedTokenCounts[row] >= maxTokens ||
                    eosTokenIDs.contains(token) ||
                    stoppedRowIndexes.contains(row)
                {
                    finished[row] = true
                }
            }
            emittedColumns.append(column)
        }

        for (slot, row) in activeRows.enumerated() {
            let rowPlan = plan.rows[slot]
            if let nextBonus = rowPlan.nextBonusToken {
                bonusTokens[row] = nextBonus
            }
            positions[row] = rowPlan.positionAfterRound
        }
        sharedKVSequenceLength = max(1, verifyKVLength - plan.globalRejectedTokenCount)

        if activeRows.allSatisfy({ finished[$0] }) {
            activeRowIndexes = []
        } else if canFilterFinishedRows {
            activeRowIndexes = activeRows.filter { !finished[$0] }
        } else {
            activeRowIndexes = activeRows
        }
        roundIndex += 1

        return MTPBatchSpeculativeRoundEmission(
            roundIndex: roundIndex - 1,
            inputBonusTokens: inputBonus,
            plan: plan,
            emittedTokenColumns: emittedColumns
        )
    }
}

public enum SpeculativeDecoding {
    public static func walk(
        draftTokens: [Int],
        targetTokens: [Int],
        budget: Int
    ) -> SpeculativeWalkResult {
        guard budget > 0 else {
            return SpeculativeWalkResult(acceptedCount: 0, newTokens: [])
        }

        let pairCount = min(draftTokens.count, targetTokens.count)
        var accepted = 0
        while accepted < pairCount, draftTokens[accepted] == targetTokens[accepted] {
            accepted += 1
        }
        if accepted == pairCount, pairCount == draftTokens.count {
            accepted = draftTokens.count
        }

        var output = Array(draftTokens.prefix(accepted))
        if accepted < targetTokens.count {
            output.append(targetTokens[accepted])
        }
        return SpeculativeWalkResult(
            acceptedCount: accepted,
            newTokens: Array(output.prefix(budget))
        )
    }

    public static func walkBatch(
        draftTokens: [[Int]],
        targetTokens: [[Int]],
        budgets: [Int]
    ) -> [SpeculativeWalkResult] {
        let count = min(draftTokens.count, targetTokens.count, budgets.count)
        return (0 ..< count).map { index in
            walk(
                draftTokens: draftTokens[index],
                targetTokens: targetTokens[index],
                budget: budgets[index]
            )
        }
    }

    public static func mtpRoundPlan(
        draftTokens: [Int],
        targetTokens: [Int],
        emittedBeforeRound: Int,
        maxTokens: Int,
        blockSize: Int,
        positionBeforeRound: Int,
        sharedKVSequenceLength: Int
    ) -> MTPSpeculativeRoundPlan {
        let budget = max(0, maxTokens - emittedBeforeRound)
        let walk = walk(draftTokens: draftTokens, targetTokens: targetTokens, budget: budget)
        return mtpRoundPlan(
            walk: walk,
            emittedBeforeRound: emittedBeforeRound,
            maxTokens: maxTokens,
            blockSize: blockSize,
            positionBeforeRound: positionBeforeRound,
            sharedKVSequenceLength: sharedKVSequenceLength
        )
    }

    public static func mtpRoundPlan(
        walk: SpeculativeWalkResult,
        emittedBeforeRound: Int,
        maxTokens: Int,
        blockSize: Int,
        positionBeforeRound: Int,
        sharedKVSequenceLength: Int
    ) -> MTPSpeculativeRoundPlan {
        let accepted = max(0, walk.acceptedCount)
        let newTokenCount = min(walk.newTokens.count, max(0, maxTokens - emittedBeforeRound))
        let emittedAfterRound = emittedBeforeRound + newTokenCount
        let rejected = max(0, blockSize - (accepted + 1))
        let validLength = max(1, sharedKVSequenceLength - rejected)
        let nextBonus = walk.newTokens.last

        return MTPSpeculativeRoundPlan(
            walk: SpeculativeWalkResult(
                acceptedCount: accepted,
                newTokens: Array(walk.newTokens.prefix(newTokenCount))
            ),
            blockSize: blockSize,
            emittedBeforeRound: emittedBeforeRound,
            emittedAfterRound: emittedAfterRound,
            positionBeforeRound: positionBeforeRound,
            positionAfterRound: positionBeforeRound + accepted + 1,
            hiddenSlotIndex: accepted,
            rejectedTokenCount: rejected,
            rollbackRequired: rejected > 0,
            sharedKVValidLength: validLength,
            nextBonusToken: nextBonus,
            finished: emittedAfterRound >= maxTokens
        )
    }

    public static func mtpBatchRoundPlan(
        draftTokens: [[Int]],
        targetTokens: [[Int]],
        emittedBeforeRound: [Int],
        maxTokens: Int,
        blockSize: Int,
        positionsBeforeRound: [Int],
        sharedKVSequenceLength: Int,
        activeRowIndexes: [Int]? = nil
    ) -> MTPBatchSpeculativeRoundPlan {
        let rowCount = [
            draftTokens.count,
            targetTokens.count,
            emittedBeforeRound.count,
            positionsBeforeRound.count
        ].min() ?? 0
        let indexes = activeRowIndexes ?? Array(0 ..< rowCount)
        let rows = (0 ..< rowCount).map { row in
            mtpRoundPlan(
                draftTokens: draftTokens[row],
                targetTokens: targetTokens[row],
                emittedBeforeRound: emittedBeforeRound[row],
                maxTokens: maxTokens,
                blockSize: blockSize,
                positionBeforeRound: positionsBeforeRound[row],
                sharedKVSequenceLength: sharedKVSequenceLength
            )
        }
        let maxAccepted = rows.map(\.walk.acceptedCount).max() ?? 0
        let globalRejected = max(0, blockSize - (maxAccepted + 1))
        return MTPBatchSpeculativeRoundPlan(
            rows: rows,
            maxAcceptedCount: maxAccepted,
            globalRejectedTokenCount: globalRejected,
            rollbackRequired: globalRejected > 0,
            activeRowIndexesAfterRound: zip(indexes, rows).compactMap { index, row in
                row.finished ? nil : index
            }
        )
    }
}
