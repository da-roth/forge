#include "stability_cleaner.h"
#include <chrono>
#include <iostream>
#include <cmath>

namespace forge::compiler::analysis {

using namespace forge::core;

StabilityCleaner::CleaningResult StabilityCleaner::clean(const ComputationGraph& graph, bool enabled) {
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    
    auto start = Clock::now();
    
    CleaningResult result;
    
    if (!enabled) {
        // When disabled, return original graph with identity mapping
        result.cleanedGraph = graph;
        result.originalToCleanedMapping.resize(graph.nodes.size());
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            result.originalToCleanedMapping[i] = static_cast<NodeId>(i);
        }
        result.stabilityFixesApplied = 0;
        result.cleaningTimeMs = 0.0;
        return result;
    }
    
    // Apply the complete stability cleaning logic (extracted from existing implementation)
    result.cleanedGraph = ComputationGraph();
    result.cleanedGraph.constPool = graph.constPool;  // Copy const pool
    
    // Mapping from old node IDs to new node IDs
    result.originalToCleanedMapping.resize(graph.nodes.size(), UINT32_MAX);
    
    size_t stabilityFixes = 0;
    
    // Process nodes in original order to maintain dependency order by construction
    for (NodeId oldId = 0; oldId < graph.nodes.size(); ++oldId) {
        const auto& node = graph.nodes[oldId];
        
        // Skip if this node is already processed
        if (result.originalToCleanedMapping[oldId] != UINT32_MAX) {
            continue;
        }
        
        // Skip dead nodes
        if (node.isDead) {
            // For dead nodes, we still need to add them to maintain order
            Node newNode = node;
            
            // Remap references to new node IDs
            if (node.a != UINT32_MAX && result.originalToCleanedMapping[node.a] != UINT32_MAX) {
                newNode.a = result.originalToCleanedMapping[node.a];
            }
            if (node.b != UINT32_MAX && result.originalToCleanedMapping[node.b] != UINT32_MAX) {
                newNode.b = result.originalToCleanedMapping[node.b];
            }
            if (node.c != UINT32_MAX && result.originalToCleanedMapping[node.c] != UINT32_MAX) {
                newNode.c = result.originalToCleanedMapping[node.c];
            }
            
            NodeId newId = result.cleanedGraph.addNode(newNode);
            result.originalToCleanedMapping[oldId] = newId;
            continue;
        }
        
        // Apply stability transformations
        Node newNode = node;
        bool transformed = false;
        
        // Remap references to new node IDs first
        if (node.a != UINT32_MAX && result.originalToCleanedMapping[node.a] != UINT32_MAX) {
            newNode.a = result.originalToCleanedMapping[node.a];
        }
        if (node.b != UINT32_MAX && result.originalToCleanedMapping[node.b] != UINT32_MAX) {
            newNode.b = result.originalToCleanedMapping[node.b];
        }
        if (node.c != UINT32_MAX && result.originalToCleanedMapping[node.c] != UINT32_MAX) {
            newNode.c = result.originalToCleanedMapping[node.c];
        }
        
        // Pattern: 1.0 / exp(x) -> exp(-x)
        if (node.op == OpCode::Div) {
            // Check if numerator is constant 1.0
            if (isConstantValue(node.a, 1.0, graph)) {
                // Check if denominator is exp(something)
                if (node.b < graph.nodes.size() && 
                    graph.nodes[node.b].op == OpCode::Exp &&
                    !graph.nodes[node.b].isDead) {
                    
                    // Transform to exp(-x)
                    newNode.op = OpCode::Exp;
                    newNode.a = result.originalToCleanedMapping[graph.nodes[node.b].a];  // Use the input to exp
                    newNode.b = UINT32_MAX;
                    newNode.c = UINT32_MAX;
                    
                    // Create a negation node for -x
                    Node negNode;
                    negNode.op = OpCode::Neg;
                    negNode.a = newNode.a;
                    negNode.b = UINT32_MAX;
                    negNode.c = UINT32_MAX;
                    negNode.isActive = graph.nodes[node.b].isActive;
                    negNode.needsGradient = graph.nodes[node.b].needsGradient;
                    
                    NodeId negId = result.cleanedGraph.addNode(negNode);
                    newNode.a = negId;
                    
                    transformed = true;
                }
            }
            // Pattern: exp(x) / exp(y) -> exp(x - y)
            else if (node.a < graph.nodes.size() && node.b < graph.nodes.size() &&
                     graph.nodes[node.a].op == OpCode::Exp &&
                     graph.nodes[node.b].op == OpCode::Exp &&
                     !graph.nodes[node.a].isDead && !graph.nodes[node.b].isDead) {
                
                // Transform to exp(x - y)
                newNode.op = OpCode::Exp;
                newNode.a = UINT32_MAX;  // Will be set to subtraction result
                newNode.b = UINT32_MAX;
                newNode.c = UINT32_MAX;
                
                // Create subtraction node for x - y
                Node subNode;
                subNode.op = OpCode::Sub;
                subNode.a = result.originalToCleanedMapping[graph.nodes[node.a].a];  // x
                subNode.b = result.originalToCleanedMapping[graph.nodes[node.b].a];  // y
                subNode.c = UINT32_MAX;
                subNode.isActive = graph.nodes[node.a].isActive || graph.nodes[node.b].isActive;
                subNode.needsGradient = graph.nodes[node.a].needsGradient || graph.nodes[node.b].needsGradient;
                
                NodeId subId = result.cleanedGraph.addNode(subNode);
                newNode.a = subId;
                
                transformed = true;
            }
        }
        // Pattern: log(exp(x)) -> x (but be careful about domain)
        else if (node.op == OpCode::Log) {
            if (node.a < graph.nodes.size() && 
                graph.nodes[node.a].op == OpCode::Exp &&
                !graph.nodes[node.a].isDead) {
                
                // Transform to just x
                newNode = graph.nodes[graph.nodes[node.a].a];
                newNode.needsGradient = node.needsGradient || newNode.needsGradient;
                newNode.isActive = node.isActive || newNode.isActive;
                
                transformed = true;
            }
        }
        // Pattern: sqrt(x * x) -> abs(x) (but be careful about domain)
        else if (node.op == OpCode::Sqrt) {
            if (node.a < graph.nodes.size() && 
                graph.nodes[node.a].op == OpCode::Mul &&
                !graph.nodes[node.a].isDead) {
                
                const auto& mulNode = graph.nodes[node.a];
                if (mulNode.a == mulNode.b) {  // x * x
                    // Transform to abs(x)
                    newNode.op = OpCode::Abs;
                    newNode.a = result.originalToCleanedMapping[mulNode.a];
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
        NodeId newId = result.cleanedGraph.addNode(newNode);
        result.originalToCleanedMapping[oldId] = newId;
    }
    
    // Remap outputs
    for (NodeId oldOutput : graph.outputs) {
        if (result.originalToCleanedMapping[oldOutput] != UINT32_MAX) {
            result.cleanedGraph.markOutput(result.originalToCleanedMapping[oldOutput]);
        }
    }
    
    // Remap diff_inputs
    for (NodeId oldDiffInput : graph.diff_inputs) {
        if (result.originalToCleanedMapping[oldDiffInput] != UINT32_MAX) {
            result.cleanedGraph.diff_inputs.push_back(result.originalToCleanedMapping[oldDiffInput]);
        }
    }
    
    result.stabilityFixesApplied = static_cast<int>(stabilityFixes);
    
    auto end = Clock::now();
    result.cleaningTimeMs = Duration(end - start).count();
    
    return result;
}

bool StabilityCleaner::isConstantValue(NodeId nodeId, double expectedValue, 
                                     const ComputationGraph& graph) {
    if (nodeId >= graph.nodes.size()) return false;
    const auto& node = graph.nodes[nodeId];
    
    if (node.op == OpCode::Constant) {
        size_t constIndex = static_cast<size_t>(node.imm);
        if (constIndex < graph.constPool.size()) {
            return std::abs(graph.constPool[constIndex] - expectedValue) < 1e-15;
        }
    }
    
    return false;
}

} // namespace forge::compiler::analysis