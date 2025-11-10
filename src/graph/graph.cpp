#include "graph.hpp"

namespace forge {

NodeId Graph::addNode(const Node& node) {
    NodeId id = static_cast<NodeId>(nodes.size());
    nodes.push_back(node);
    nodes.back().dst = id;
    return id;
}

NodeId Graph::addConstant(double value) {
    size_t constIndex = constPool.size();
    constPool.push_back(value);
    
    Node node{};
    node.op = OpCode::Constant;
    node.imm = static_cast<double>(constIndex);
    node.isActive = false;  // Constants never depend on inputs
    return addNode(node);
}

NodeId Graph::addInput() {
    Node node{};
    node.op = OpCode::Input;
    node.isActive = true;  // Inputs are always active
    return addNode(node);
}

void Graph::markOutput(NodeId node) {
    outputs.push_back(node);
}

void Graph::clear() {
    nodes.clear();
    constPool.clear();
    outputs.clear();
    diff_inputs.clear();
}

} // namespace forge