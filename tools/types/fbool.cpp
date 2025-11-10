#include "fbool.hpp"
#include "fdouble.hpp"
#include "../../src/graph/graph.hpp"
#include "../../src/graph/graph_recorder.hpp"

namespace forge {

NodeId fbool::ensureNode() const {
    // Check if we already have a valid node (not -1/invalid)
    if (activeNode_ != static_cast<NodeId>(-1)) {
        return activeNode_;
    }
    
    // Need to add this constant bool to the graph
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return -1;
    }
    
    Node node{};
    node.op = OpCode::BoolConstant;
    node.imm = passiveValue_ ? 1.0 : 0.0;  // Store as double for compatibility
    node.isActive = false;  // Constants are never active
    node.needsGradient = false;  // Constants don't need gradients
    
    NodeId resultNode = recorder->graph().addNode(node);
    
    // Cast away const to cache the node ID (mutable-like behavior)
    const_cast<fbool*>(this)->activeNode_ = resultNode;
    
    return resultNode;
}

fbool fbool::operator&&(const fbool& other) const {
    bool result = passiveValue_ && other.passiveValue_;
    
    if (!GraphRecorder::isAnyRecording()) {
        return fbool(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fbool(result);
    }
    
    // Record BoolAnd node
    NodeId aNode = this->ensureNode();
    NodeId bNode = other.ensureNode();
    
    Node node{};
    node.op = OpCode::BoolAnd;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    node.needsGradient = needsGradient_ || other.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool fbool::operator||(const fbool& other) const {
    bool result = passiveValue_ || other.passiveValue_;
    
    if (!GraphRecorder::isAnyRecording()) {
        return fbool(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fbool(result);
    }
    
    // Record BoolOr node
    NodeId aNode = this->ensureNode();
    NodeId bNode = other.ensureNode();
    
    Node node{};
    node.op = OpCode::BoolOr;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    node.needsGradient = needsGradient_ || other.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool fbool::operator!() const {
    bool result = !passiveValue_;
    
    if (!GraphRecorder::isAnyRecording()) {
        return fbool(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fbool(result);
    }
    
    // Record BoolNot node
    NodeId aNode = this->ensureNode();
    
    Node node{};
    node.op = OpCode::BoolNot;
    node.a = aNode;
    node.isActive = isActive_;
    node.needsGradient = needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool fbool::operator==(const fbool& other) const {
    bool result = (passiveValue_ == other.passiveValue_);
    
    if (!GraphRecorder::isAnyRecording()) {
        return fbool(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fbool(result);
    }
    
    NodeId aNode = this->ensureNode();
    NodeId bNode = other.ensureNode();
    
    Node node{};
    node.op = OpCode::BoolEq;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    node.needsGradient = needsGradient_ || other.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool fbool::operator!=(const fbool& other) const {
    bool result = (passiveValue_ != other.passiveValue_);
    
    if (!GraphRecorder::isAnyRecording()) {
        return fbool(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fbool(result);
    }
    
    NodeId aNode = this->ensureNode();
    NodeId bNode = other.ensureNode();
    
    Node node{};
    node.op = OpCode::BoolNe;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    node.needsGradient = needsGradient_ || other.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

// Static If function for convenience
fdouble fbool::If(const fbool& condition, const fdouble& true_val, const fdouble& false_val) {
    return condition.If(true_val, false_val);
}

// The If operation
fdouble fbool::If(const fdouble& true_val, const fdouble& false_val) const {
    // For now, just return a simple result without using .value()
    // This is temporary to test if the rest of the code works
    
    if (!GraphRecorder::isAnyRecording()) {
        // Not recording - use the passive value to select
        double result = passiveValue_ ? true_val.value() : false_val.value();
        return fdouble(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        // No recorder - use passive value
        double result = passiveValue_ ? true_val.value() : false_val.value();
        return fdouble(result);
    }
    
    // Recording - create the If node in the graph
    NodeId condNode = this->ensureNode();
    NodeId trueNode = true_val.ensureNode();
    NodeId falseNode = false_val.ensureNode();
    
    Node node{};
    node.op = OpCode::If;
    node.a = condNode;   // Bool condition
    node.b = trueNode;   // Value if true
    node.c = falseNode;  // Value if false
    node.isActive = isActive_ || true_val.isActive() || false_val.isActive();
    node.needsGradient = needsGradient_ || true_val.needsGradient_ || false_val.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    
    // For now, just return a placeholder value during recording
    // The actual selection will happen during JIT execution
    // Use 0.0 as placeholder - the JIT will compute the real value
    return fdouble::fromNode(resultNode, 0.0, node.isActive, node.needsGradient);
}

} // namespace forge