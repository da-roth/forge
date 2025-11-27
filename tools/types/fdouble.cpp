#include "fdouble.hpp"
#include "fbool.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace forge {

NodeId fdouble::ensureNode() const {
    if (!GraphRecorder::isAnyRecording()) {
        throw std::runtime_error("Cannot ensure node when not recording");
    }
    
    // If we have a valid node, return it (whether active or inactive)
    if (activeNode_ != static_cast<NodeId>(-1)) {
        return activeNode_;
    }
    
    // Otherwise, create a constant node for the passive value
    // This handles lazy initialization of constants
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        throw std::runtime_error("No active recorder during recording");
    }
    
    // Need to cast away const to update activeNode_ for lazy initialization
    const_cast<fdouble*>(this)->activeNode_ = recorder->graph().addConstant(passiveValue_);
    return activeNode_;
}

InputHandle fdouble::markInput() {
    if (!GraphRecorder::isAnyRecording()) {
        throw std::runtime_error("Cannot mark input when not recording");
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        throw std::runtime_error("No active recorder");
    }
    
    NodeId inputNode = recorder->graph().addInput();
    // Preserve the current value when marking as input
    *this = fdouble::fromNode(inputNode, passiveValue_);
    return InputHandle(inputNode);
}

InputHandle fdouble::markInputAndDiff() {
    auto handle = markInput();  // Regular input marking
    
    // Set BOTH flags
    isActive_ = true;
    needsGradient_ = true;
    
    if (GraphRecorder::isAnyRecording()) {
        auto* recorder = GraphRecorder::active();
        auto& graph = recorder->graph();
        graph.nodes[activeNode_].isActive = true;
        graph.nodes[activeNode_].needsGradient = true;
        graph.diff_inputs.push_back(activeNode_);
    }
    return handle;
}

ResultHandle fdouble::markOutput() {
    if (!GraphRecorder::isAnyRecording()) {
        throw std::runtime_error("Cannot mark output when not recording");
    }

    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        throw std::runtime_error("No active recorder");
    }

    // Warn if marking a passive value as output - this may indicate missing
    // Forge wiring in the computation path.
    if (!isActive_) {
        static std::size_t warnCount = 0;
        if (warnCount < 10) {
            ++warnCount;
            std::cerr << "[Forge][Warning] markOutput() called on passive value "
                      << "(value=" << passiveValue_ << ") - gradients will be zero. "
                      << "This may indicate incomplete Forge wiring. "
                      << "(occurrence " << warnCount << ")\n";
        }
    }

    // Ensure we have a node in the graph even for passive values. This allows
    // outputs that are currently constant w.r.t. Forge inputs to be marked as
    // outputs; gradients will be zero in that case. This is preferable to
    // failing hard and is useful while incrementally wiring up Forge through
    // larger code paths like QuantLib.
    NodeId nodeId = ensureNode();
    recorder->graph().markOutput(nodeId);
    return ResultHandle(nodeId);
}

fdouble fdouble::binaryOp(const fdouble& a, const fdouble& b, OpCode op) {
    // Always compute the actual value
    double result = 0.0;
    switch (op) {
        case OpCode::Add: result = a.passiveValue_ + b.passiveValue_; break;
        case OpCode::Sub: result = a.passiveValue_ - b.passiveValue_; break;
        case OpCode::Mul: result = a.passiveValue_ * b.passiveValue_; break;
        case OpCode::Div: result = a.passiveValue_ / b.passiveValue_; break;
        case OpCode::Mod: result = std::fmod(a.passiveValue_, b.passiveValue_); break;
        case OpCode::Min: result = std::fmin(a.passiveValue_, b.passiveValue_); break;
        case OpCode::Max: result = std::fmax(a.passiveValue_, b.passiveValue_); break;
        case OpCode::CmpLT: result = (a.passiveValue_ < b.passiveValue_) ? 1.0 : 0.0; break;
        case OpCode::CmpLE: result = (a.passiveValue_ <= b.passiveValue_) ? 1.0 : 0.0; break;
        case OpCode::CmpGT: result = (a.passiveValue_ > b.passiveValue_) ? 1.0 : 0.0; break;
        case OpCode::CmpGE: result = (a.passiveValue_ >= b.passiveValue_) ? 1.0 : 0.0; break;
        case OpCode::CmpEQ: result = (a.passiveValue_ == b.passiveValue_) ? 1.0 : 0.0; break;
        case OpCode::CmpNE: result = (a.passiveValue_ != b.passiveValue_) ? 1.0 : 0.0; break;
        default: throw std::runtime_error("Unsupported operation");
    }
    
    if (!GraphRecorder::isAnyRecording()) {
        return fdouble(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        throw std::runtime_error("No active recorder during recording");
    }
    
    // Always record the operation in the graph
    NodeId aNode = a.ensureNode();
    NodeId bNode = b.ensureNode();
    
    Node node{};
    node.op = op;
    node.a = aNode;
    node.b = bNode;
    // Propagate both flags independently
    node.isActive = a.isActive_ || b.isActive_;
    node.needsGradient = a.needsGradient_ || b.needsGradient_;
    
    NodeId resultNode = recorder->graph().addNode(node);
    
    // Return with both flags
    return fdouble::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fdouble fdouble::operator+(const fdouble& rhs) const {
    return binaryOp(*this, rhs, OpCode::Add);
}

fdouble fdouble::operator-(const fdouble& rhs) const {
    return binaryOp(*this, rhs, OpCode::Sub);
}

fdouble fdouble::operator*(const fdouble& rhs) const {
    return binaryOp(*this, rhs, OpCode::Mul);
}

fdouble fdouble::operator/(const fdouble& rhs) const {
    return binaryOp(*this, rhs, OpCode::Div);
}


fdouble& fdouble::operator+=(const fdouble& rhs) {
    *this = *this + rhs;
    return *this;
}

fdouble& fdouble::operator-=(const fdouble& rhs) {
    *this = *this - rhs;
    return *this;
}

fdouble& fdouble::operator*=(const fdouble& rhs) {
    *this = *this * rhs;
    return *this;
}

fdouble& fdouble::operator/=(const fdouble& rhs) {
    *this = *this / rhs;
    return *this;
}

fdouble fdouble::operator-() const {
    if (!GraphRecorder::isAnyRecording()) {
        return fdouble(-passiveValue_);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fdouble(-passiveValue_);
    }
    
    // Always record the operation
    NodeId xNode = ensureNode();
    
    Node node{};
    node.op = OpCode::Neg;
    node.a = xNode;
    node.isActive = isActive_;  // Propagate active state
    node.needsGradient = needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    
    return fdouble::fromNode(resultNode, -passiveValue_, isActive_, needsGradient_);
}

fdouble operator+(double lhs, const fdouble& rhs) {
    return fdouble(lhs) + rhs;
}

fdouble operator-(double lhs, const fdouble& rhs) {
    return fdouble(lhs) - rhs;
}

fdouble operator*(double lhs, const fdouble& rhs) {
    return fdouble(lhs) * rhs;
}

fdouble operator/(double lhs, const fdouble& rhs) {
    return fdouble(lhs) / rhs;
}

fdouble abs(const fdouble& x) {
    if (!GraphRecorder::isAnyRecording()) {
        return fdouble(std::abs(x.value()));
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fdouble(std::abs(x.value()));
    }
    
    // Always record the operation
    NodeId xNode = x.ensureNode();
    
    Node node{};
    node.op = OpCode::Abs;
    node.a = xNode;
    node.isActive = x.isActive_;  // Propagate active state
    node.needsGradient = x.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    double result = std::abs(x.value());
    return fdouble::fromNode(resultNode, result, x.isActive_, x.needsGradient_);
}

fdouble square(const fdouble& x) {
    if (!GraphRecorder::isAnyRecording()) {
        double val = x.value();
        return fdouble(val * val);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        double val = x.value();
        return fdouble(val * val);
    }
    
    // Always record the operation
    NodeId xNode = x.ensureNode();
    double val = x.value();
    
    Node node{};
    node.op = OpCode::Square;
    node.a = xNode;
    node.isActive = x.isActive_;  // Propagate active state
    node.needsGradient = x.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    double result = val * val;
    return fdouble::fromNode(resultNode, result, x.isActive_, x.needsGradient_);
}

fdouble recip(const fdouble& x) {
    if (!GraphRecorder::isAnyRecording()) {
        return fdouble(1.0 / x.value());
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fdouble(1.0 / x.value());
    }
    
    // Always record the operation
    NodeId xNode = x.ensureNode();
    
    Node node{};
    node.op = OpCode::Recip;
    node.a = xNode;
    node.isActive = x.isActive_;  // Propagate active state
    node.needsGradient = x.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    double result = 1.0 / x.value();
    return fdouble::fromNode(resultNode, result, x.isActive_, x.needsGradient_);
}

fdouble mod(const fdouble& x, const fdouble& y) {
    return fdouble::binaryOp(x, y, OpCode::Mod);
}

fdouble exp(const fdouble& x) {
    // Handle non-recording case
    if (!GraphRecorder::isAnyRecording()) {
        return fdouble(std::exp(x.value()));
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fdouble(std::exp(x.value()));
    }
    
    // Record the operation
    NodeId xNode = x.ensureNode();
    
    Node node{};
    node.op = OpCode::Exp;
    node.a = xNode;
    node.isActive = x.isActive_;  // Propagate active state
    node.needsGradient = x.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    double result = std::exp(x.value());
    return fdouble::fromNode(resultNode, result, x.isActive_, x.needsGradient_);
}

fdouble log(const fdouble& x) {
    // Handle non-recording case
    if (!GraphRecorder::isAnyRecording()) {
        return fdouble(std::log(x.value()));
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fdouble(std::log(x.value()));
    }
    
    // Record the operation
    NodeId xNode = x.ensureNode();
    
    Node node{};
    node.op = OpCode::Log;
    node.a = xNode;
    node.isActive = x.isActive_;  // Propagate active state
    node.needsGradient = x.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    double result = std::log(x.value());
    return fdouble::fromNode(resultNode, result, x.isActive_, x.needsGradient_);
}

fdouble sqrt(const fdouble& x) {
    // Handle non-recording case
    if (!GraphRecorder::isAnyRecording()) {
        return fdouble(std::sqrt(x.value()));
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fdouble(std::sqrt(x.value()));
    }
    
    // Record the operation
    NodeId xNode = x.ensureNode();
    
    Node node{};
    node.op = OpCode::Sqrt;
    node.a = xNode;
    node.isActive = x.isActive_;  // Propagate active state
    node.needsGradient = x.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    double result = std::sqrt(x.value());
    return fdouble::fromNode(resultNode, result, x.isActive_, x.needsGradient_);
}

fdouble pow(const fdouble& x, const fdouble& y) {
    // Handle non-recording case
    if (!GraphRecorder::isAnyRecording()) {
        return fdouble(std::pow(x.value(), y.value()));
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fdouble(std::pow(x.value(), y.value()));
    }
    
    // Record the pow operation directly
    NodeId xNode = x.ensureNode();
    NodeId yNode = y.ensureNode();
    
    Node node{};
    node.op = OpCode::Pow;
    node.a = xNode;
    node.b = yNode;
    node.isActive = x.isActive_ || y.isActive_;
    node.needsGradient = x.needsGradient_ || y.needsGradient_;
    
    NodeId resultNode = recorder->graph().addNode(node);
    double result = std::pow(x.value(), y.value());
    return fdouble::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fdouble sin(const fdouble& x) {
    // Handle non-recording case
    if (!GraphRecorder::isAnyRecording()) {
        return fdouble(std::sin(x.value()));
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fdouble(std::sin(x.value()));
    }
    
    // Record the operation
    NodeId xNode = x.ensureNode();
    
    Node node{};
    node.op = OpCode::Sin;
    node.a = xNode;
    node.isActive = x.isActive_;  // Propagate active state
    node.needsGradient = x.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    double result = std::sin(x.value());
    return fdouble::fromNode(resultNode, result, x.isActive_, x.needsGradient_);
}

fdouble cos(const fdouble& x) {
    // Handle non-recording case
    if (!GraphRecorder::isAnyRecording()) {
        return fdouble(std::cos(x.value()));
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fdouble(std::cos(x.value()));
    }
    
    // Record the operation
    NodeId xNode = x.ensureNode();
    
    Node node{};
    node.op = OpCode::Cos;
    node.a = xNode;
    node.isActive = x.isActive_;  // Propagate active state
    node.needsGradient = x.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    double result = std::cos(x.value());
    return fdouble::fromNode(resultNode, result, x.isActive_, x.needsGradient_);
}

fdouble tan(const fdouble& x) {
    // Handle non-recording case
    if (!GraphRecorder::isAnyRecording()) {
        return fdouble(std::tan(x.value()));
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fdouble(std::tan(x.value()));
    }
    
    // Record the operation
    NodeId xNode = x.ensureNode();
    
    Node node{};
    node.op = OpCode::Tan;
    node.a = xNode;
    node.isActive = x.isActive_;  // Propagate active state
    node.needsGradient = x.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    double result = std::tan(x.value());
    return fdouble::fromNode(resultNode, result, x.isActive_, x.needsGradient_);
}

fdouble min(const fdouble& x, const fdouble& y) {
    // For binary operations, we use the binaryOp helper
    return fdouble::binaryOp(x, y, OpCode::Min);
}

fdouble max(const fdouble& x, const fdouble& y) {
    // For binary operations, we use the binaryOp helper
    return fdouble::binaryOp(x, y, OpCode::Max);
}

fbool cmpLT(const fdouble& x, const fdouble& y) {
    bool result = x.value() < y.value();
    
    if (!GraphRecorder::isAnyRecording()) {
        return fbool(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fbool(result);
    }
    
    NodeId xNode = x.ensureNode();
    NodeId yNode = y.ensureNode();
    
    Node node{};
    node.op = OpCode::CmpLT;
    node.a = xNode;
    node.b = yNode;
    node.isActive = x.isActive() || y.isActive();
    node.needsGradient = x.needsGradient_ || y.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool cmpLE(const fdouble& x, const fdouble& y) {
    bool result = x.value() <= y.value();
    
    if (!GraphRecorder::isAnyRecording()) {
        return fbool(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fbool(result);
    }
    
    NodeId xNode = x.ensureNode();
    NodeId yNode = y.ensureNode();
    
    Node node{};
    node.op = OpCode::CmpLE;
    node.a = xNode;
    node.b = yNode;
    node.isActive = x.isActive() || y.isActive();
    node.needsGradient = x.needsGradient_ || y.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool cmpGT(const fdouble& x, const fdouble& y) {
    bool result = x.value() > y.value();
    
    if (!GraphRecorder::isAnyRecording()) {
        return fbool(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fbool(result);
    }
    
    NodeId xNode = x.ensureNode();
    NodeId yNode = y.ensureNode();
    
    Node node{};
    node.op = OpCode::CmpGT;
    node.a = xNode;
    node.b = yNode;
    node.isActive = x.isActive() || y.isActive();
    node.needsGradient = x.needsGradient_ || y.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool cmpGE(const fdouble& x, const fdouble& y) {
    bool result = x.value() >= y.value();
    
    if (!GraphRecorder::isAnyRecording()) {
        return fbool(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fbool(result);
    }
    
    NodeId xNode = x.ensureNode();
    NodeId yNode = y.ensureNode();
    
    Node node{};
    node.op = OpCode::CmpGE;
    node.a = xNode;
    node.b = yNode;
    node.isActive = x.isActive() || y.isActive();
    node.needsGradient = x.needsGradient_ || y.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool cmpEQ(const fdouble& x, const fdouble& y) {
    bool result = x.value() == y.value();
    
    if (!GraphRecorder::isAnyRecording()) {
        return fbool(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fbool(result);
    }
    
    NodeId xNode = x.ensureNode();
    NodeId yNode = y.ensureNode();
    
    Node node{};
    node.op = OpCode::CmpEQ;
    node.a = xNode;
    node.b = yNode;
    node.isActive = x.isActive() || y.isActive();
    node.needsGradient = x.needsGradient_ || y.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

fbool cmpNE(const fdouble& x, const fdouble& y) {
    bool result = x.value() != y.value();
    
    if (!GraphRecorder::isAnyRecording()) {
        return fbool(result);
    }
    
    auto* recorder = GraphRecorder::active();
    if (!recorder) {
        return fbool(result);
    }
    
    NodeId xNode = x.ensureNode();
    NodeId yNode = y.ensureNode();
    
    Node node{};
    node.op = OpCode::CmpNE;
    node.a = xNode;
    node.b = yNode;
    node.isActive = x.isActive() || y.isActive();
    node.needsGradient = x.needsGradient_ || y.needsGradient_;  // Propagate gradient flag
    
    NodeId resultNode = recorder->graph().addNode(node);
    return fbool::fromNode(resultNode, result, node.isActive, node.needsGradient);
}

// Double comparison operator implementations
fbool fdouble::operator<(const fdouble& other) const { return cmpLT(*this, other); }
fbool fdouble::operator<=(const fdouble& other) const { return cmpLE(*this, other); }
fbool fdouble::operator>(const fdouble& other) const { return cmpGT(*this, other); }
fbool fdouble::operator>=(const fdouble& other) const { return cmpGE(*this, other); }
fbool fdouble::operator==(const fdouble& other) const { return cmpEQ(*this, other); }
fbool fdouble::operator!=(const fdouble& other) const { return cmpNE(*this, other); }

// Overloads for comparison with double
fbool fdouble::operator<(double other) const { return cmpLT(*this, fdouble(other)); }
fbool fdouble::operator<=(double other) const { return cmpLE(*this, fdouble(other)); }
fbool fdouble::operator>(double other) const { return cmpGT(*this, fdouble(other)); }
fbool fdouble::operator>=(double other) const { return cmpGE(*this, fdouble(other)); }
fbool fdouble::operator==(double other) const { return cmpEQ(*this, fdouble(other)); }
fbool fdouble::operator!=(double other) const { return cmpNE(*this, fdouble(other)); }

} // namespace forge