#pragma once

#include "../graph.hpp"
#include "../graph_optimizer.hpp"
#include <vector>

namespace forge {
namespace optimizations {

/**
 * Constant cleanup optimization: Remove unused constants from const pool
 * 
 * This optimization identifies constants that are no longer referenced by any nodes
 * and removes them from the const pool, updating constant indices in nodes.
 * 
 * This helps reduce memory usage and improves cache locality.
 */
class ConstantCleanup {
public:
    /**
     * Apply constant cleanup to the given tape
     * @param graph The input tape to optimize
     * @param stats Reference to optimization stats to update
     * @return Optimized tape with cleaned constant pool
     */
    static forge::Graph apply(const forge::Graph& graph, 
                                         forge::GraphOptimizer::OptimizationStats& stats);

private:
    /**
     * Count references to each constant in the tape
     * @param graph The tape to analyze
     * @return Vector of reference counts for each constant
     */
    static std::vector<size_t> countConstantReferences(const forge::Graph& graph);
    
    /**
     * Build mapping from old constant indices to new constant indices
     * @param refCounts Reference counts for each constant
     * @return Mapping from old index to new index (UINT32_MAX if unused)
     */
    static std::vector<forge::NodeId> buildConstantMapping(const std::vector<size_t>& refCounts);
};

} // namespace optimizations
} // namespace forge
