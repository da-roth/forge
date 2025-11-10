#pragma once

#include "../graph.hpp"
#include "../graph_optimizer.hpp"
#include <unordered_map>
#include <vector>

namespace forge {
namespace optimizations {

/**
 * Common Subexpression Elimination (CSE) optimization
 * 
 * Identifies and eliminates duplicate computations by remapping nodes.
 * When multiple nodes perform the same computation, all but one are marked as dead
 * and all references are redirected to the canonical node.
 * 
 * Example: x = a + b; y = a + b; z = x + y → x = a + b; y = x; z = x + x
 */
class CommonSubexpressionElimination {
public:
    /**
     * Apply CSE to the given tape
     * @param graph The input tape to optimize
     * @param stats Reference to optimization stats to update
     * @return Optimized tape with duplicate computations eliminated
     */
    static forge::Graph apply(const forge::Graph& graph, 
                                         forge::GraphOptimizer::OptimizationStats& stats);

private:
    // Hash function for node signatures
    struct NodeSignature {
        forge::OpCode op;
        forge::NodeId a, b, c;
        double imm;
        
        bool operator==(const NodeSignature& other) const {
            return op == other.op && a == other.a && b == other.b && 
                   c == other.c && imm == other.imm;
        }
    };
    
    struct NodeSignatureHash {
        std::size_t operator()(const NodeSignature& sig) const {
            // Simple hash combining - can be improved
            std::size_t h1 = std::hash<int>{}(static_cast<int>(sig.op));
            std::size_t h2 = std::hash<size_t>{}(sig.a);
            std::size_t h3 = std::hash<size_t>{}(sig.b);
            std::size_t h4 = std::hash<size_t>{}(sig.c);
            std::size_t h5 = std::hash<double>{}(sig.imm);
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
        }
    };
    
    /**
     * Normalize constant references for comparison
     * @param id Node ID to normalize
     * @param graph The tape containing the node
     * @return Normalized node ID for comparison
     */
    static forge::NodeId normalizeOperand(forge::NodeId id, 
                                                       const forge::Graph& graph,
                                                       const std::vector<forge::NodeId>& oldToNew);
};

} // namespace optimizations
} // namespace forge
