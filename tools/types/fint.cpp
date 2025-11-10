#include "fint.hpp"
#include "fdouble.hpp"
#include "fbool.hpp"
#include "../../src/graph/graph.hpp"
#include "../../src/graph/graph_recorder.hpp"
#include <cmath>

namespace forge {

NodeId fint::ensureNode() const {
    // Check if we already have a valid node (not -1/invalid)
    if (activeNode_ != static_cast<NodeId>(-1)) {
        return activeNode_;
    }
    
    // Need to add this constant int to the graph
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return -1;
    }
    
    Node node{};
    node.op = OpCode::IntConstant;
    node.imm = static_cast<double>(passiveValue_);  // Store as double for compatibility
    node.isActive = false;  // Constants are never active
    
    NodeId resultNode = recorder->graph().addNode(node);
    
    // Cast away const to cache the node ID (mutable-like behavior)
    const_cast<fint*>(this)->activeNode_ = resultNode;
    
    return resultNode;
}

// Arithmetic operations
fint fint::operator+(const fint& other) const {
    int64_t result = passiveValue_ + other.passiveValue_;
    
    if (!GraphRecorder::isAnyRecording()) {
        return fint(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fint(result);
    }
    
    NodeId aNode = this->ensureNode();
    NodeId bNode = other.ensureNode();
    
    Node node{};
    node.op = OpCode::IntAdd;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fint::fromNode(resultNode, result, node.isActive);
}

fint fint::operator-(const fint& other) const {
    int64_t result = passiveValue_ - other.passiveValue_;
    
    if (!GraphRecorder::isAnyRecording()) {
        return fint(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fint(result);
    }
    
    NodeId aNode = this->ensureNode();
    NodeId bNode = other.ensureNode();
    
    Node node{};
    node.op = OpCode::IntSub;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fint::fromNode(resultNode, result, node.isActive);
}

fint fint::operator*(const fint& other) const {
    int64_t result = passiveValue_ * other.passiveValue_;
    
    if (!GraphRecorder::isAnyRecording()) {
        return fint(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fint(result);
    }
    
    NodeId aNode = this->ensureNode();
    NodeId bNode = other.ensureNode();
    
    Node node{};
    node.op = OpCode::IntMul;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fint::fromNode(resultNode, result, node.isActive);
}

fint fint::operator/(const fint& other) const {
    // Integer division - truncate toward zero
    int64_t result = passiveValue_ / other.passiveValue_;
    
    if (!GraphRecorder::isAnyRecording()) {
        return fint(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fint(result);
    }
    
    NodeId aNode = this->ensureNode();
    NodeId bNode = other.ensureNode();
    
    Node node{};
    node.op = OpCode::IntDiv;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fint::fromNode(resultNode, result, node.isActive);
}

fint fint::operator%(const fint& other) const {
    int64_t result = passiveValue_ % other.passiveValue_;
    
    if (!GraphRecorder::isAnyRecording()) {
        return fint(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fint(result);
    }
    
    NodeId aNode = this->ensureNode();
    NodeId bNode = other.ensureNode();
    
    Node node{};
    node.op = OpCode::IntMod;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fint::fromNode(resultNode, result, node.isActive);
}

fint fint::operator-() const {
    int64_t result = -passiveValue_;
    
    if (!GraphRecorder::isAnyRecording()) {
        return fint(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fint(result);
    }
    
    NodeId aNode = this->ensureNode();
    
    Node node{};
    node.op = OpCode::IntNeg;
    node.a = aNode;
    node.isActive = isActive_;
    node.needsGradient = needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fint::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

// Comparison operations (return fbool)
fbool fint::operator<(const fint& other) const {
    bool result = passiveValue_ < other.passiveValue_;
    
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
    node.op = OpCode::IntCmpLT;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    node.needsGradient = needsGradient_ || other.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool fint::operator<=(const fint& other) const {
    bool result = passiveValue_ <= other.passiveValue_;
    
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
    node.op = OpCode::IntCmpLE;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    node.needsGradient = needsGradient_ || other.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool fint::operator>(const fint& other) const {
    bool result = passiveValue_ > other.passiveValue_;
    
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
    node.op = OpCode::IntCmpGT;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    node.needsGradient = needsGradient_ || other.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool fint::operator>=(const fint& other) const {
    bool result = passiveValue_ >= other.passiveValue_;
    
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
    node.op = OpCode::IntCmpGE;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    node.needsGradient = needsGradient_ || other.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool fint::operator==(const fint& other) const {
    bool result = passiveValue_ == other.passiveValue_;
    
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
    node.op = OpCode::IntCmpEQ;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    node.needsGradient = needsGradient_ || other.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool fint::operator!=(const fint& other) const {
    bool result = passiveValue_ != other.passiveValue_;
    
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
    node.op = OpCode::IntCmpNE;
    node.a = aNode;
    node.b = bNode;
    node.isActive = isActive_ || other.isActive_;
    node.needsGradient = needsGradient_ || other.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

// Comparison with integer literals
fbool fint::operator<(int64_t value) const {
    return *this < fint(value);
}

fbool fint::operator<=(int64_t value) const {
    return *this <= fint(value);
}

fbool fint::operator>(int64_t value) const {
    return *this > fint(value);
}

fbool fint::operator>=(int64_t value) const {
    return *this >= fint(value);
}

fbool fint::operator==(int64_t value) const {
    return *this == fint(value);
}

fbool fint::operator!=(int64_t value) const {
    return *this != fint(value);
}

// Array indexing support - THE KEY USE CASE!
fdouble fint::index(const std::vector<fdouble>& array) const {
    // Bounds check for passive execution
    size_t idx = static_cast<size_t>(passiveValue_);
    if (idx >= array.size()) {
        throw std::runtime_error("fint array index out of bounds");
    }
    
    // Implement dynamic indexing using conditional selection
    // This creates a chain of If operations that select the correct array element
    if (array.size() == 0) {
        throw std::runtime_error("Cannot index empty array");
    }
    
    fdouble result = array[0];  // Start with first element
    
    // Chain If operations for each array element
    for (size_t i = 1; i < array.size(); ++i) {
        fbool condition = (*this == fint(static_cast<int64_t>(i)));
        result = fbool::If(condition, array[i], result);
    }
    
    return result;
}

// Static If function for convenience
fint fint::If(const fbool& condition, const fint& true_val, const fint& false_val) {
    // Compute actual result for immediate use
    int64_t result = condition.value() ? true_val.passiveValue_ : false_val.passiveValue_;
    
    if (!GraphRecorder::isAnyRecording()) {
        return fint(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fint(result);
    }
    
    // Record all three operands in the graph
    NodeId condNode = condition.ensureNode();
    NodeId trueNode = true_val.ensureNode();
    NodeId falseNode = false_val.ensureNode();
    
    Node node{};
    node.op = OpCode::IntIf;
    node.a = condNode;   // Bool condition
    node.b = trueNode;   // Value if true
    node.c = falseNode;  // Value if false
    node.isActive = condition.isActive() || true_val.isActive_ || false_val.isActive_;
    node.needsGradient = condition.needsGradient_ || true_val.needsGradient_ || false_val.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fint::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

} // namespace forge