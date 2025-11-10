#pragma once

#include "../graph.hpp"
#include "../graph_optimizer.hpp"

namespace forge {
namespace optimizations {

/**
 * Algebraic simplification optimization
 * 
 * Applies simple algebraic identities and strength reduction to simplify expressions.
 * This includes patterns like:
 * - x * 1.0 → x (multiplicative identity)
 * - x + 0.0 → x (additive identity)
 * - x * x → Square(x) (square pattern recognition)
 * - x * 2.0 → x + x (strength reduction)
 * - x - x → 0.0 (self-subtraction)
 * 
 * All simplifications preserve mathematical correctness while improving performance.
 */
class AlgebraicSimplification {
public:
    /**
     * Apply algebraic simplification to the given tape
     * @param graph The input tape to optimize
     * @param stats Reference to optimization stats to update
     * @return Optimized tape with algebraic simplifications applied
     */
    static forge::Graph apply(const forge::Graph& graph, 
                                         forge::GraphOptimizer::OptimizationStats& stats);

private:
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
