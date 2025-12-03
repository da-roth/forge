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
                // x * 1.0 → x (multiplicative identity) - REDIRECT instead of copy
                else if (isConstantValue(node.b, 1.0, graph)) {
                    // Redirect to the existing node instead of copying
                    oldToNew[oldId] = oldToNew[node.a];
                    simplifications++;
                    continue;  // Skip adding the node normally
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
                // 0.0 + x → x (additive identity) - REDIRECT instead of copy
                if (isConstantValue(node.a, 0.0, graph)) {
                    oldToNew[oldId] = oldToNew[node.b];
                    simplifications++;
                    continue;  // Skip adding the node normally
                }
                // x + 0.0 → x (additive identity) - REDIRECT instead of copy
                else if (isConstantValue(node.b, 0.0, graph)) {
                    oldToNew[oldId] = oldToNew[node.a];
                    simplifications++;
                    continue;  // Skip adding the node normally
                }
                break;

            case forge::OpCode::Sub:
                // x - 0.0 → x (subtractive identity) - REDIRECT instead of copy
                if (isConstantValue(node.b, 0.0, graph)) {
                    oldToNew[oldId] = oldToNew[node.a];
                    simplifications++;
                    continue;  // Skip adding the node normally
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
                // x / 1.0 → x (division by one) - REDIRECT instead of copy
                if (isConstantValue(node.b, 1.0, graph)) {
                    oldToNew[oldId] = oldToNew[node.a];
                    simplifications++;
                    continue;  // Skip adding the node normally
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
                // -(-x) → x (double negation) - REDIRECT instead of copy
                if (graph.nodes[node.a].op == forge::OpCode::Neg) {
                    // Redirect to the inner operand of the inner Neg
                    forge::NodeId innerOperand = graph.nodes[node.a].a;
                    oldToNew[oldId] = oldToNew[innerOperand];
                    simplifications++;
                    continue;  // Skip adding the node normally
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
