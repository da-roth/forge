#pragma once

#include "forge/core/computation_graph.h"
#include <vector>

namespace forge::compiler::analysis {

/**
 * Standalone stability cleaner for numerical safety transformations.
 * 
 * This is NOT an optimization but a numerical safety requirement.
 * It transforms mathematically equivalent but numerically unstable patterns
 * into more stable forms (e.g., 1.0/exp(x) → exp(-x)).
 * 
 * Separated from GraphOptimizer to avoid coupling AsmStitcher with optimization logic.
 */
class StabilityCleaner {
public:
    struct CleaningResult {
        forge::core::ComputationGraph cleanedGraph;
        std::vector<forge::core::NodeId> originalToCleanedMapping;
        int stabilityFixesApplied = 0;
        double cleaningTimeMs = 0.0;
    };
    
    /**
     * Apply stability cleaning transformations to a computation graph
     * @param graph Input computation graph
     * @param enabled Whether stability cleaning is enabled
     * @return Cleaned graph with mapping and statistics
     */
    static CleaningResult clean(const forge::core::ComputationGraph& graph, bool enabled = true);
    
private:
    /**
     * Check if a node is a specific constant value
     * @param nodeId Node ID to check
     * @param expectedValue Expected constant value
     * @param graph The graph containing the node
     * @return True if the node is a constant with the specified value
     */
    static bool isConstantValue(forge::core::NodeId nodeId, double expectedValue, 
                               const forge::core::ComputationGraph& graph);
};

} // namespace forge::compiler::analysis