#include "stability_cleaning.hpp"
#include <cmath>

namespace forge {
namespace optimizations {

forge::Graph StabilityCleaning::apply(const forge::Graph& graph, 
                                                  forge::GraphOptimizer::OptimizationStats& stats) {
    // Stability cleaning: Transform numerically unstable patterns into stable equivalents
    // Return a new tape with stable patterns, maintaining order by construction
    
    forge::Graph result;
    result.constPool = graph.constPool;  // Copy const pool
    
    // Mapping from old node IDs to new node IDs
    std::vector<forge::NodeId> oldToNew(graph.nodes.size(), UINT32_MAX);
    
    size_t stabilityFixes = 0;
    
    // Process nodes in original order to maintain dependency order by construction
    for (forge::NodeId oldId = 0; oldId < graph.nodes.size(); ++oldId) {
        const auto& node = graph.nodes[oldId];

        // Apply stability transformations
        forge::Node newNode = node;
        bool transformed = false;
        
        // Remap references to new node IDs first
        if (node.a != UINT32_MAX && oldToNew[node.a] != UINT32_MAX) {
            newNode.a = oldToNew[node.a];
        }
        if (node.b != UINT32_MAX && oldToNew[node.b] != UINT32_MAX) {
            newNode.b = oldToNew[node.b];
        }
        if (node.c != UINT32_MAX && oldToNew[node.c] != UINT32_MAX) {
            newNode.c = oldToNew[node.c];
        }
        
        // Pattern: 1.0 / exp(x) -> exp(-x)
        if (node.op == forge::OpCode::Div) {
            // Check if numerator is constant 1.0
            if (isConstantValue(node.a, 1.0, graph)) {
                // Check if denominator is exp(something)
                if (node.b < graph.nodes.size() &&
                    graph.nodes[node.b].op == forge::OpCode::Exp) {
                    
                    // Transform to exp(-x)
                    newNode.op = forge::OpCode::Exp;
                    newNode.a = oldToNew[graph.nodes[node.b].a];  // Use the input to exp
                    newNode.b = UINT32_MAX;
                    newNode.c = UINT32_MAX;
                    
                    // Create a negation node for -x
                    forge::Node negNode;
                    negNode.op = forge::OpCode::Neg;
                    negNode.a = newNode.a;
                    negNode.b = UINT32_MAX;
                    negNode.c = UINT32_MAX;
                    negNode.isActive = graph.nodes[node.b].isActive;
                    negNode.needsGradient = graph.nodes[node.b].needsGradient;
                    
                    forge::NodeId negId = result.addNode(negNode);
                    newNode.a = negId;
                    
                    transformed = true;
                }
            }
            // Pattern: exp(x) / exp(y) -> exp(x - y)
            else if (node.a < graph.nodes.size() && node.b < graph.nodes.size() &&
                     graph.nodes[node.a].op == forge::OpCode::Exp &&
                     graph.nodes[node.b].op == forge::OpCode::Exp) {
                
                // Transform to exp(x - y)
                newNode.op = forge::OpCode::Exp;
                newNode.a = UINT32_MAX;  // Will be set to subtraction result
                newNode.b = UINT32_MAX;
                newNode.c = UINT32_MAX;
                
                // Create subtraction node for x - y
                forge::Node subNode;
                subNode.op = forge::OpCode::Sub;
                subNode.a = oldToNew[graph.nodes[node.a].a];  // x
                subNode.b = oldToNew[graph.nodes[node.b].a];  // y
                subNode.c = UINT32_MAX;
                subNode.isActive = graph.nodes[node.a].isActive || graph.nodes[node.b].isActive;
                subNode.needsGradient = graph.nodes[node.a].needsGradient || graph.nodes[node.b].needsGradient;
                
                forge::NodeId subId = result.addNode(subNode);
                newNode.a = subId;
                
                transformed = true;
            }
        }
        // Pattern: log(exp(x)) -> x (but be careful about domain)
        else if (node.op == forge::OpCode::Log) {
            if (node.a < graph.nodes.size() &&
                graph.nodes[node.a].op == forge::OpCode::Exp) {

                // Map this log node directly to the (already remapped) input of exp
                forge::NodeId expInputId = graph.nodes[node.a].a;
                oldToNew[oldId] = oldToNew[expInputId];
                stabilityFixes++;
                continue;  // Skip adding a new node
            }
        }
        // Pattern: sqrt(x * x) -> abs(x) (but be careful about domain)
        else if (node.op == forge::OpCode::Sqrt) {
            if (node.a < graph.nodes.size() &&
                graph.nodes[node.a].op == forge::OpCode::Mul) {
                
                const auto& mulNode = graph.nodes[node.a];
                if (mulNode.a == mulNode.b) {  // x * x
                    // Transform to abs(x)
                    newNode.op = forge::OpCode::Abs;
                    newNode.a = oldToNew[mulNode.a];
                    newNode.b = UINT32_MAX;
                    newNode.c = UINT32_MAX;
                    
                    transformed = true;
                }
            }
        }
        
        if (transformed) {
            stabilityFixes++;
        }
        
        // Add the node to the result
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
    stats.stabilityFixes += stabilityFixes;
    
    return result;
}

bool StabilityCleaning::isConstantValue(forge::NodeId nodeId, double expectedValue, 
                                       const forge::Graph& graph) {
    if (nodeId >= graph.nodes.size()) return false;
    const auto& node = graph.nodes[nodeId];
    
    if (node.op == forge::OpCode::Constant) {
        size_t constIndex = static_cast<size_t>(node.imm);
        if (constIndex < graph.constPool.size()) {
            return std::abs(graph.constPool[constIndex] - expectedValue) < 1e-15;
        }
    }
    
    return false;
}

} // namespace optimizations
} // namespace forge