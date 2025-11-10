#pragma once

#include "../graph.hpp"
#include "../graph_optimizer.hpp"
#include <vector>

namespace forge {
namespace optimizations {

/**
 * Inactive folding optimization: Evaluates and folds entire constant subgraphs
 * 
 * This optimization identifies nodes that don't depend on inputs (isActive=false)
 * and evaluates them at compile time, replacing them with constant values.
 * 
 * Example: y = 2 + 3; z = y / 5; result = x + z → result = x + 1.0
 */
class InactiveFolding {
public:
    /**
     * Apply inactive folding to the given tape
     * @param graph The input tape to optimize
     * @param stats Reference to optimization stats to update
     * @return Optimized tape with constant subgraphs folded
     */
    static forge::Graph apply(const forge::Graph& graph, 
                                         forge::GraphOptimizer::OptimizationStats& stats);

private:
    /**
     * Evaluate a constant subgraph recursively
     * @param graph The tape containing the subgraph
     * @param nodeId The root node of the subgraph to evaluate
     * @return The computed constant value
     */
    static double evaluateInactiveSubgraph(const forge::Graph& graph, 
                                          forge::NodeId nodeId);
};

} // namespace optimizations
} // namespace forge
