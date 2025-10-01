#include "computation_graph.h"

namespace forge {
namespace core {

NodeId ComputationGraph::addNode(const Node& node) {
    NodeId id = static_cast<NodeId>(nodes.size());
    nodes.push_back(node);
    nodes.back().dst = id;
    return id;
}

NodeId ComputationGraph::addConstant(double value) {
    size_t constIndex = constPool.size();
    constPool.push_back(value);
    
    Node node{};
    node.op = OpCode::Constant;
    node.imm = static_cast<double>(constIndex);
    node.isActive = false;  // Constants never depend on inputs
    return addNode(node);
}

NodeId ComputationGraph::addInput() {
    Node node{};
    node.op = OpCode::Input;
    node.isActive = true;  // Inputs are always active
    return addNode(node);
}

void ComputationGraph::markOutput(NodeId node) {
    outputs.push_back(node);
}

void ComputationGraph::clear() {
    nodes.clear();
    constPool.clear();
    outputs.clear();
    diff_inputs.clear();
}

} // namespace core
} // namespace forge