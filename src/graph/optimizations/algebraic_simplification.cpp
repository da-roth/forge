#include "algebraic_simplification.hpp"
#include <cmath>

namespace forge {
namespace optimizations {

forge::Graph AlgebraicSimplification::apply(const forge::Graph& graph, 
                                                        forge::GraphOptimizer::OptimizationStats& stats) {
    // Algebraic simplification
    // Return a new tape with simplified expressions, maintaining order by construction
    
    forge::Graph result;
    result.constPool = graph.constPool;  // Copy const pool
    
    // Mapping from old node IDs to new node IDs
    std::vector<forge::NodeId> oldToNew(graph.nodes.size(), UINT32_MAX);
    
    size_t simplifications = 0;
    
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
        
        // Apply algebraic simplifications
        forge::Node newNode = node;
        bool simplified = false;
        
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
        
        // Pattern matching for common simplifications
        switch (node.op) {
            case forge::OpCode::Mul:
                // SQUARE PATTERN RECOGNITION: x * x → Square(x)
                if (node.a == node.b) {
                    newNode.op = forge::OpCode::Square;
                    newNode.b = UINT32_MAX;  // Square only uses operand a
                    simplified = true;
                }
                // x * 1.0 → x (multiplicative identity)
                else if (isConstantValue(node.b, 1.0, graph)) {
                    // Only optimize if operand is NOT an Input node to avoid graph corruption
                    if (node.a < graph.nodes.size() && 
                        graph.nodes[node.a].op != forge::OpCode::Input) {
                        // Safe to replace - operand is a computed value
                        newNode = graph.nodes[node.a];
                        newNode.dst = oldId;  // Preserve destination
                        newNode.needsGradient = node.needsGradient || newNode.needsGradient;
                        newNode.isActive = node.isActive || newNode.isActive;
                        simplified = true;
                    }
                }
                // x * 0.0 or 0.0 * x → 0.0 (multiplication by zero)
                else if (isConstantValue(node.a, 0.0, graph) || isConstantValue(node.b, 0.0, graph)) {
                    // Replace with constant 0
                    forge::NodeId constId = result.addConstant(0.0);
                    oldToNew[oldId] = constId;
                    simplified = true;
                    continue;  // Skip adding the node normally
                }
                break;
                
            case forge::OpCode::Add:
                // x + 0.0 or 0.0 + x → x (additive identity)
                if (isConstantValue(node.a, 0.0, graph)) {
                    newNode = graph.nodes[node.b];
                    newNode.dst = oldId;
                    newNode.needsGradient = node.needsGradient || newNode.needsGradient;
                    newNode.isActive = node.isActive || newNode.isActive;
                    simplified = true;
                }
                else if (isConstantValue(node.b, 0.0, graph)) {
                    newNode = graph.nodes[node.a];
                    newNode.dst = oldId;
                    newNode.needsGradient = node.needsGradient || newNode.needsGradient;
                    newNode.isActive = node.isActive || newNode.isActive;
                    simplified = true;
                }
                break;
                
            case forge::OpCode::Sub:
                // x - 0.0 → x (subtractive identity)
                if (isConstantValue(node.b, 0.0, graph)) {
                    newNode = graph.nodes[node.a];
                    newNode.dst = oldId;
                    newNode.needsGradient = node.needsGradient || newNode.needsGradient;
                    newNode.isActive = node.isActive || newNode.isActive;
                    simplified = true;
                }
                // x - x → 0.0 (self-subtraction)
                else if (node.a == node.b) {
                    forge::NodeId constId = result.addConstant(0.0);
                    oldToNew[oldId] = constId;
                    simplified = true;
                    continue;  // Skip adding the node normally
                }
                break;
                
            case forge::OpCode::Div:
                // x / 1.0 → x (division by one)
                if (isConstantValue(node.b, 1.0, graph)) {
                    newNode = graph.nodes[node.a];
                    newNode.dst = oldId;
                    newNode.needsGradient = node.needsGradient || newNode.needsGradient;
                    newNode.isActive = node.isActive || newNode.isActive;
                    simplified = true;
                }
                // x / x → 1.0 (self-division, but be careful about division by zero)
                else if (node.a == node.b) {
                    forge::NodeId constId = result.addConstant(1.0);
                    oldToNew[oldId] = constId;
                    simplified = true;
                    continue;  // Skip adding the node normally
                }
                break;
                
            case forge::OpCode::Neg:
                // -(-x) → x (double negation)
                if (graph.nodes[node.a].op == forge::OpCode::Neg) {
                    newNode = graph.nodes[graph.nodes[node.a].a];
                    newNode.dst = oldId;
                    newNode.needsGradient = node.needsGradient || newNode.needsGradient;
                    newNode.isActive = node.isActive || newNode.isActive;
                    simplified = true;
                }
                break;
                
            case forge::OpCode::Square:
                // Square(0.0) → 0.0
                if (isConstantValue(node.a, 0.0, graph)) {
                    forge::NodeId constId = result.addConstant(0.0);
                    oldToNew[oldId] = constId;
                    simplified = true;
                    continue;  // Skip adding the node normally
                }
                // Square(1.0) → 1.0
                else if (isConstantValue(node.a, 1.0, graph)) {
                    forge::NodeId constId = result.addConstant(1.0);
                    oldToNew[oldId] = constId;
                    simplified = true;
                    continue;  // Skip adding the node normally
                }
                break;
                
            case forge::OpCode::Sqrt:
                // Sqrt(0.0) → 0.0
                if (isConstantValue(node.a, 0.0, graph)) {
                    forge::NodeId constId = result.addConstant(0.0);
                    oldToNew[oldId] = constId;
                    simplified = true;
                    continue;  // Skip adding the node normally
                }
                // Sqrt(1.0) → 1.0
                else if (isConstantValue(node.a, 1.0, graph)) {
                    forge::NodeId constId = result.addConstant(1.0);
                    oldToNew[oldId] = constId;
                    simplified = true;
                    continue;  // Skip adding the node normally
                }
                break;
                
            case forge::OpCode::Exp:
                // Exp(0.0) → 1.0
                if (isConstantValue(node.a, 0.0, graph)) {
                    forge::NodeId constId = result.addConstant(1.0);
                    oldToNew[oldId] = constId;
                    simplified = true;
                    continue;  // Skip adding the node normally
                }
                break;
                
            case forge::OpCode::Log:
                // Log(1.0) → 0.0
                if (isConstantValue(node.a, 1.0, graph)) {
                    forge::NodeId constId = result.addConstant(0.0);
                    oldToNew[oldId] = constId;
                    simplified = true;
                    continue;  // Skip adding the node normally
                }
                break;
                
            default:
                // No simplification for this operation
                break;
        }
        
        if (simplified) {
            simplifications++;
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
    stats.algebraicSimplifications += simplifications;
    
    return result;
}

bool AlgebraicSimplification::isConstantValue(forge::NodeId nodeId, double expectedValue, 
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