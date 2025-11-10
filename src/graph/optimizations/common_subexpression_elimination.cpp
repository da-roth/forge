#include "common_subexpression_elimination.hpp"
#include <functional>
#include <unordered_map>

namespace forge {
namespace optimizations {

forge::Graph CommonSubexpressionElimination::apply(const forge::Graph& graph, 
                                                              forge::GraphOptimizer::OptimizationStats& stats) {
    // Common Subexpression Elimination
    // Return a new tape with deduplicated nodes, maintaining order by construction
    
    forge::Graph result;
    result.constPool = graph.constPool;  // Copy const pool
    
    // Mapping from old node IDs to new node IDs
    std::vector<forge::NodeId> oldToNew(graph.nodes.size(), UINT32_MAX);
    
    // Map from node signature to first occurrence node ID in new tape
    std::unordered_map<NodeSignature, forge::NodeId, NodeSignatureHash> seenNodes;
    
    size_t duplicatesFound = 0;
    
    // Process nodes in original order to maintain dependency order by construction
    for (forge::NodeId oldId = 0; oldId < graph.nodes.size(); ++oldId) {
        const auto& node = graph.nodes[oldId];
        
        // Skip if this node is already processed (dead)
        if (oldToNew[oldId] != UINT32_MAX) {
            continue;
        }
        
        // Skip dead nodes and special nodes (inputs can't be CSE'd)
        if (node.isDead || node.op == forge::OpCode::Input) {
            // For dead nodes, we still need to add them to maintain order
            // but they won't be deduplicated
            forge::Node newNode = node;
            
            // Remap references to new node IDs
            if (node.a != UINT32_MAX && oldToNew[node.a] != UINT32_MAX) {
                newNode.a = oldToNew[node.a];
            }
            if (node.b != UINT32_MAX && oldToNew[node.b] != UINT32_MAX) {
                newNode.b = oldToNew[node.b];
            }
            if (node.c != UINT32_MAX && oldToNew[node.c] != UINT32_MAX) {
                newNode.c = oldToNew[node.c];
            }
            
            forge::NodeId newId = result.addNode(newNode);
            oldToNew[oldId] = newId;
            continue;
        }
        
        // Create signature for this node, normalizing constant operands by value
        NodeSignature sig;
        sig.op = node.op;
        sig.a = normalizeOperand(node.a, graph, oldToNew);
        sig.b = normalizeOperand(node.b, graph, oldToNew);
        sig.c = normalizeOperand(node.c, graph, oldToNew);
        sig.imm = node.imm;
        
        // Check if we've seen this exact computation before
        auto it = seenNodes.find(sig);
        if (it != seenNodes.end()) {
            // Found duplicate! This node does the same computation as the canonical node
            oldToNew[oldId] = it->second;  // Redirect all uses of oldId to canonical node
            duplicatesFound++;
        } else {
            // First time seeing this computation signature
            forge::Node newNode = node;
            
            // Remap references to new node IDs
            if (node.a != UINT32_MAX && oldToNew[node.a] != UINT32_MAX) {
                newNode.a = oldToNew[node.a];
            }
            if (node.b != UINT32_MAX && oldToNew[node.b] != UINT32_MAX) {
                newNode.b = oldToNew[node.b];
            }
            if (node.c != UINT32_MAX && oldToNew[node.c] != UINT32_MAX) {
                newNode.c = oldToNew[node.c];
            }
            
            forge::NodeId newId = result.addNode(newNode);
            oldToNew[oldId] = newId;
            seenNodes[sig] = newId;  // Store canonical node ID in new tape
        }
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
    stats.duplicatesEliminated += duplicatesFound;
    
    return result;
}

forge::NodeId CommonSubexpressionElimination::normalizeOperand(forge::NodeId id, 
                                                                           const forge::Graph& graph,
                                                                           const std::vector<forge::NodeId>& oldToNew) {
    if (id >= graph.nodes.size()) return id;
    const auto& node = graph.nodes[id];
    
    // For constants, use a special encoding based on value
    if (node.op == forge::OpCode::Constant) {
        size_t constIdx = static_cast<size_t>(node.imm);
        if (constIdx < graph.constPool.size()) {
            // Encode constant value in high bits to distinguish from regular node IDs
            // This makes constants with same value compare equal
            return static_cast<forge::NodeId>(0x80000000 | std::hash<double>{}(graph.constPool[constIdx]));
        }
    }
    
    // For other nodes, use the new node ID if it exists, otherwise use original
    if (oldToNew[id] != UINT32_MAX) {
        return oldToNew[id];
    }
    
    return id;
}

} // namespace optimizations
} // namespace forge