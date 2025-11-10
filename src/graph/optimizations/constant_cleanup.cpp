#include "constant_cleanup.hpp"
#include <algorithm>

namespace forge {
namespace optimizations {

forge::Graph ConstantCleanup::apply(const forge::Graph& graph, 
                                               forge::GraphOptimizer::OptimizationStats& stats) {
    // Constant cleanup: Remove unused constants from const pool
    // Return a new tape with cleaned constant pool, maintaining order by construction
    
    forge::Graph result;
    
    // Count references to each constant
    std::vector<size_t> refCounts = countConstantReferences(graph);
    
    // Build mapping from old constant indices to new constant indices
    std::vector<forge::NodeId> constMapping = buildConstantMapping(refCounts);
    
    // Copy only used constants to new const pool
    for (size_t i = 0; i < graph.constPool.size(); ++i) {
        if (constMapping[i] != UINT32_MAX) {
            result.constPool.push_back(graph.constPool[i]);
        }
    }
    
    // Mapping from old node IDs to new node IDs
    std::vector<forge::NodeId> oldToNew(graph.nodes.size(), UINT32_MAX);
    
    size_t constantsRemoved = 0;
    
    // Process nodes in original order to maintain dependency order by construction
    for (forge::NodeId oldId = 0; oldId < graph.nodes.size(); ++oldId) {
        const auto& node = graph.nodes[oldId];
        
        // Skip if this node is already processed (dead)
        if (oldToNew[oldId] != UINT32_MAX) {
            continue;
        }
        
        // Skip dead nodes
        if (node.isDead) {
            // For dead nodes, we still need to add them to maintain order
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
        
        // Process active nodes
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
        
        // Update constant references
        if (node.op == forge::OpCode::Constant) {
            size_t oldConstIndex = static_cast<size_t>(node.imm);
            if (oldConstIndex < constMapping.size() && constMapping[oldConstIndex] != UINT32_MAX) {
                newNode.imm = static_cast<double>(constMapping[oldConstIndex]);
            } else {
                // This constant is unused, but we still need to add the node
                // It will be marked as dead or we can replace it with 0
                newNode.imm = 0.0;  // Replace with 0 for unused constants
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
    
    // Count removed constants
    for (size_t i = 0; i < refCounts.size(); ++i) {
        if (refCounts[i] == 0) {
            constantsRemoved++;
        }
    }
    
    // Update stats
    stats.constantsRemoved += constantsRemoved;
    
    return result;
}

std::vector<size_t> ConstantCleanup::countConstantReferences(const forge::Graph& graph) {
    std::vector<size_t> refCounts(graph.constPool.size(), 0);
    
    // Count references to each constant
    for (const auto& node : graph.nodes) {
        if (node.op == forge::OpCode::Constant) {
            size_t constIndex = static_cast<size_t>(node.imm);
            if (constIndex < refCounts.size()) {
                refCounts[constIndex]++;
            }
        }
    }
    
    return refCounts;
}

std::vector<forge::NodeId> ConstantCleanup::buildConstantMapping(const std::vector<size_t>& refCounts) {
    std::vector<forge::NodeId> mapping(refCounts.size(), UINT32_MAX);
    forge::NodeId newIndex = 0;
    
    // Build mapping: old index -> new index (UINT32_MAX if unused)
    for (size_t i = 0; i < refCounts.size(); ++i) {
        if (refCounts[i] > 0) {
            mapping[i] = newIndex++;
        }
    }
    
    return mapping;
}

} // namespace optimizations
} // namespace forge
