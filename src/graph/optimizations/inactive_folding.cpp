#include "inactive_folding.hpp"
#include <algorithm>
#include <cmath>
#include <functional>

namespace forge {
namespace optimizations {

forge::Graph InactiveFolding::apply(const forge::Graph& graph, 
                                                forge::GraphOptimizer::OptimizationStats& stats) {
    // Inactive folding: Evaluate and fold entire constant subgraphs
    // Return a new tape with only live nodes, maintaining order by construction
    
    forge::Graph result;
    result.constPool = graph.constPool;  // Copy const pool
    
    // Mapping from old node IDs to new node IDs
    std::vector<forge::NodeId> oldToNew(graph.nodes.size(), UINT32_MAX);
    
    size_t foldedCount = 0;
    
    // Helper function to evaluate a constant subgraph
    std::function<double(forge::NodeId)> evaluateConstantSubgraph = [&](forge::NodeId nodeId) -> double {
        const auto& node = graph.nodes[nodeId];
        
        switch (node.op) {
            case forge::OpCode::Constant: {
                size_t constIndex = static_cast<size_t>(node.imm);
                if (constIndex < graph.constPool.size()) {
                    return graph.constPool[constIndex];
                }
                return 0.0;
            }
            case forge::OpCode::Add: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return a + b;
            }
            case forge::OpCode::Sub: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return a - b;
            }
            case forge::OpCode::Mul: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return a * b;
            }
            case forge::OpCode::Div: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return (b != 0.0) ? (a / b) : 0.0;
            }
            case forge::OpCode::Neg: {
                double a = evaluateConstantSubgraph(node.a);
                return -a;
            }
            case forge::OpCode::Exp: {
                double a = evaluateConstantSubgraph(node.a);
                return std::exp(a);
            }
            case forge::OpCode::Log: {
                double a = evaluateConstantSubgraph(node.a);
                return (a > 0.0) ? std::log(a) : 0.0;
            }
            case forge::OpCode::Sqrt: {
                double a = evaluateConstantSubgraph(node.a);
                return (a >= 0.0) ? std::sqrt(a) : 0.0;
            }
            case forge::OpCode::Square: {
                double a = evaluateConstantSubgraph(node.a);
                return a * a;
            }
            case forge::OpCode::Recip: {
                double a = evaluateConstantSubgraph(node.a);
                return (a != 0.0) ? (1.0 / a) : 0.0;
            }
            case forge::OpCode::Abs: {
                double a = evaluateConstantSubgraph(node.a);
                return std::abs(a);
            }
            case forge::OpCode::Sin: {
                double a = evaluateConstantSubgraph(node.a);
                return std::sin(a);
            }
            case forge::OpCode::Cos: {
                double a = evaluateConstantSubgraph(node.a);
                return std::cos(a);
            }
            case forge::OpCode::Tan: {
                double a = evaluateConstantSubgraph(node.a);
                return std::tan(a);
            }
            case forge::OpCode::Pow: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return std::pow(a, b);
            }
            case forge::OpCode::Min: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return std::min(a, b);
            }
            case forge::OpCode::Max: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return std::max(a, b);
            }
            // Comparison operations - return 1.0 for true, 0.0 for false
            case forge::OpCode::CmpLT: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return (a < b) ? 1.0 : 0.0;
            }
            case forge::OpCode::CmpLE: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return (a <= b) ? 1.0 : 0.0;
            }
            case forge::OpCode::CmpGT: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return (a > b) ? 1.0 : 0.0;
            }
            case forge::OpCode::CmpGE: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return (a >= b) ? 1.0 : 0.0;
            }
            case forge::OpCode::CmpEQ: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return (a == b) ? 1.0 : 0.0;
            }
            case forge::OpCode::CmpNE: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return (a != b) ? 1.0 : 0.0;
            }
            // Conditional operation
            case forge::OpCode::If: {
                double condition = evaluateConstantSubgraph(node.a);
                double true_val = evaluateConstantSubgraph(node.b);
                double false_val = evaluateConstantSubgraph(node.c);
                return (condition != 0.0) ? true_val : false_val;
            }
            // Boolean operations
            case forge::OpCode::BoolAnd: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return ((a != 0.0) && (b != 0.0)) ? 1.0 : 0.0;
            }
            case forge::OpCode::BoolOr: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                return ((a != 0.0) || (b != 0.0)) ? 1.0 : 0.0;
            }
            case forge::OpCode::BoolNot: {
                double a = evaluateConstantSubgraph(node.a);
                return (a == 0.0) ? 1.0 : 0.0;
            }
            case forge::OpCode::BoolEq: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                // Both true or both false
                return ((a != 0.0) == (b != 0.0)) ? 1.0 : 0.0;
            }
            case forge::OpCode::BoolNe: {
                double a = evaluateConstantSubgraph(node.a);
                double b = evaluateConstantSubgraph(node.b);
                // One true, one false
                return ((a != 0.0) != (b != 0.0)) ? 1.0 : 0.0;
            }
            default:
                return 0.0;  // Unknown operation, return 0
        }
    };
    
    // Process nodes in original order to maintain dependency order by construction
    for (forge::NodeId oldId = 0; oldId < graph.nodes.size(); ++oldId) {
        const auto& node = graph.nodes[oldId];
        
        // Skip if this node is already processed (part of a folded subgraph)
        if (oldToNew[oldId] != UINT32_MAX) {
            continue;
        }
        
        // Check if this node is part of a constant subgraph
        if (!node.isActive) {
            // This is an inactive node, check if it can be folded
            try {
                double foldedValue = evaluateConstantSubgraph(oldId);
                
                // Add as a new constant node
                forge::NodeId newConstId = result.addConstant(foldedValue);
                oldToNew[oldId] = newConstId;
                foldedCount++;
                
                // Mark all nodes in this constant subgraph as processed
                std::function<void(forge::NodeId)> markProcessed = [&](forge::NodeId id) {
                    if (id < graph.nodes.size() && oldToNew[id] == UINT32_MAX) {
                        oldToNew[id] = newConstId;  // All point to the same constant
                        const auto& subNode = graph.nodes[id];
                        if (!subNode.isActive) {
                            if (subNode.a != UINT32_MAX) markProcessed(subNode.a);
                            if (subNode.b != UINT32_MAX) markProcessed(subNode.b);
                            if (subNode.c != UINT32_MAX) markProcessed(subNode.c);
                        }
                    }
                };
                markProcessed(oldId);
                continue;
            } catch (...) {
                // If evaluation fails, treat as regular node
            }
        }
        
        // This is an active node, add it to the result
        forge::Node newNode = node;
        
        // Remap references to new node IDs
        if (node.a != UINT32_MAX) {
            if (oldToNew[node.a] != UINT32_MAX) {
                newNode.a = oldToNew[node.a];
            } else {
                // This shouldn't happen in a well-formed graph
                newNode.a = UINT32_MAX;
            }
        }
        if (node.b != UINT32_MAX) {
            if (oldToNew[node.b] != UINT32_MAX) {
                newNode.b = oldToNew[node.b];
            } else {
                newNode.b = UINT32_MAX;
            }
        }
        if (node.c != UINT32_MAX) {
            if (oldToNew[node.c] != UINT32_MAX) {
                newNode.c = oldToNew[node.c];
            } else {
                newNode.c = UINT32_MAX;
            }
        }
        
        forge::NodeId newId = result.addNode(newNode);
        oldToNew[oldId] = newId;
    }
    
    // Remap outputs
    for (forge::NodeId oldOutput : graph.outputs) {
        if (oldToNew[oldOutput] != UINT32_MAX) {
            result.markOutput(oldToNew[oldOutput]);
        }
    }
    
    // Remap diff_inputs
    for (forge::NodeId oldDiffInput : graph.diff_inputs) {
        if (oldToNew[oldDiffInput] != UINT32_MAX) {
            result.diff_inputs.push_back(oldToNew[oldDiffInput]);
        }
    }
    
    // Update stats
    stats.inactiveNodesFolded += foldedCount;
    
    return result;
}

} // namespace optimizations
} // namespace forge