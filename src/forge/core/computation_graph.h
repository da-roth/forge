#pragma once

#include "opcodes.h"
#include <cstdint>
#include <vector>

namespace forge {
namespace core {

using NodeId = uint32_t;
using SlotId = uint32_t;

/**
 * Node structure representing a single operation in the computation graph.
 * This is the fundamental unit of the tape-based automatic differentiation system.
 */
struct Node {
    OpCode op;                    // Operation type
    NodeId dst{};                 // Destination node ID (self-reference)
    NodeId a{};                   // First input operand
    NodeId b{};                   // Second input operand
    NodeId c{};                   // Third input operand (for ternary operations)
    uint32_t flags{};             // Operation-specific flags
    double imm{};                 // Immediate value or constant pool index
    bool isActive{true};          // Whether node depends on runtime inputs (false = constant)
    bool isDead{false};           // Whether node has been optimized away
    bool needsGradient{false};    // Whether gradient computation is required for AAD
};

/**
 * ComputationGraph represents the tape structure for automatic differentiation.
 * It stores all operations as nodes and provides methods to build and manipulate the graph.
 */
class ComputationGraph {
public:
    std::vector<Node> nodes;              // All computation nodes
    std::vector<double> constPool;        // Pool of constant values
    std::vector<NodeId> outputs;          // Indices of output nodes
    std::vector<NodeId> diff_inputs;      // Indices of inputs for differentiation
    
    // Core operations
    NodeId addNode(const Node& node);
    NodeId addConstant(double value);
    NodeId addInput();
    void markOutput(NodeId node);
    
    // Graph management
    void clear();
    bool empty() const { return nodes.empty(); }
    size_t size() const { return nodes.size(); }
    
    // Utility methods
    const Node& getNode(NodeId id) const { return nodes[id]; }
    Node& getNode(NodeId id) { return nodes[id]; }
    double getConstant(size_t index) const { return constPool[index]; }
};

} // namespace core
} // namespace forge