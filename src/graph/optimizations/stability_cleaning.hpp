#pragma once

#include "../graph.hpp"
#include "../graph_optimizer.hpp"
#include <map>
#include <vector>

namespace forge {
namespace optimizations {

/**
 * Stability cleaning optimization
 * 
 * Transforms numerically unstable patterns into stable equivalents to improve
 * numerical precision and avoid overflow/underflow issues.
 * 
 * Key patterns:
 * - 1.0 / exp(x) → exp(-x) (avoids precision issues)
 * - exp(x) / exp(y) → exp(x - y) (avoids precision issues)
 * 
 * This optimization uses a two-pass approach to avoid issues with node reallocation
 * during transformation.
 */
class StabilityCleaning {
public:
    /**
     * Apply stability cleaning to the given tape
     * @param graph The input tape to optimize
     * @param stats Reference to optimization stats to update
     * @return Optimized tape with stability improvements applied
     */
    static forge::Graph apply(const forge::Graph& graph, 
                                         forge::GraphOptimizer::OptimizationStats& stats);

private:
    // Structure to track transformations needed
    struct Transformation {
        size_t nodeIndex;           // Index of node to transform
        enum Type { DivExpToExpNeg, ExpDivExpToExpSub } type;
        forge::NodeId operand1;            // For DivExpToExpNeg: exp_input; For ExpDivExpToExpSub: x
        forge::NodeId operand2;            // For ExpDivExpToExpSub: y
        forge::NodeId oldNode1;            // Node to mark as dead
        forge::NodeId oldNode2;            // Second node to mark as dead (for ExpDivExpToExpSub)
    };
    
    /**
     * Check if a node is a specific constant value
     * @param id Node ID to check
     * @param value Expected constant value
     * @param graph The tape containing the node
     * @return True if the node is a constant with the specified value
     */
    static bool isConstantValue(forge::NodeId id, double value, 
                               const forge::Graph& graph);
};

} // namespace optimizations
} // namespace forge
