#pragma once

#include "../../src/graph/graph.hpp"
#include "../../src/graph/graph_recorder.hpp"
#include <cstdint>

namespace forge {

// Forward declarations
class fdouble;
class fbool;

// fint - Integer type for forge that participates in tape recording
class fint {
    friend class fdouble;
    friend class fbool;
    
private:
    int64_t passiveValue_;    // The actual integer value
    NodeId activeNode_;       // Node in computation graph (-1 if passive/constant)
    bool isActive_;          // Does this value depend on runtime inputs?
    bool needsGradient_;     // AAD: Propagates flag even though ints don't have gradients
    
public:
    // Constructors
    fint(int64_t value = 0) 
        : passiveValue_(value), activeNode_(-1), isActive_(false), needsGradient_(false) {}
    
    // Create fint from a graph node
    static fint fromNode(NodeId node, int64_t value, bool isActive = true, bool needsGrad = false) {
        fint i;
        i.activeNode_ = node;
        i.passiveValue_ = value;
        i.isActive_ = isActive;
        i.needsGradient_ = needsGrad;
        return i;
    }
    
    // Get the actual integer value
    int64_t value() const { return passiveValue_; }
    
    // Check if this Int depends on runtime inputs
    bool isActive() const { return isActive_; }
    
    // Get the node ID (-1 if not in graph)
    NodeId nodeId() const { return activeNode_; }
    
    // Ensure this Int has a node in the graph
    NodeId ensureNode() const;
    
    // Arithmetic operations
    fint operator+(const fint& other) const;
    fint operator-(const fint& other) const;
    fint operator*(const fint& other) const;
    fint operator/(const fint& other) const;  // Integer division (truncating)
    fint operator%(const fint& other) const;  // Modulo
    fint operator-() const;                    // Unary negation
    
    // Arithmetic with integer literals
    fint operator+(int64_t value) const { return *this + fint(value); }
    fint operator-(int64_t value) const { return *this - fint(value); }
    fint operator*(int64_t value) const { return *this * fint(value); }
    fint operator/(int64_t value) const { return *this / fint(value); }
    fint operator%(int64_t value) const { return *this % fint(value); }
    
    // Comparison operations (return fbool)
    fbool operator<(const fint& other) const;
    fbool operator<=(const fint& other) const;
    fbool operator>(const fint& other) const;
    fbool operator>=(const fint& other) const;
    fbool operator==(const fint& other) const;
    fbool operator!=(const fint& other) const;
    
    // Comparison with integer literals
    fbool operator<(int64_t value) const;
    fbool operator<=(int64_t value) const;
    fbool operator>(int64_t value) const;
    fbool operator>=(int64_t value) const;
    fbool operator==(int64_t value) const;
    fbool operator!=(int64_t value) const;
    
    // Array indexing support - THE KEY USE CASE!
    fdouble index(const std::vector<fdouble>& array) const;  // array[this]
    
    // Conditional selection (defined after fbool is complete)
    static fint If(const fbool& condition, const fint& true_val, const fint& false_val);
};

// For use as fint constructor with literals
inline fint Int(int64_t value) {
    return fint(value);
}

} // namespace forge