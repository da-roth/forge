#pragma once

#include "../../src/graph/graph.hpp"
#include "../../src/graph/graph_recorder.hpp"
#include <functional>
#include <stdexcept>

namespace forge {

// Forward declaration
class fdouble;

// fbool - Boolean type for forge that participates in tape recording
class fbool {
    friend class fint;  // Allow fint to access private members for If operation
private:
    bool passiveValue_;       // The actual boolean value
    NodeId activeNode_;       // Node in computation graph (-1 if passive/constant)
    bool isActive_;          // Does this value depend on runtime inputs?
    bool needsGradient_;     // AAD: Even though bool doesn't have gradients, it propagates the flag
    
public:
    // Constructors
    fbool(bool value = false) 
        : passiveValue_(value), activeNode_(-1), isActive_(false), needsGradient_(false) {}
    
    // Create Bool from a graph node
    static fbool fromNode(NodeId node, bool value, bool isActive = true, bool needsGrad = false) {
        fbool b;
        b.activeNode_ = node;
        b.passiveValue_ = value;
        b.isActive_ = isActive;
        b.needsGradient_ = needsGrad;
        return b;
    }
    
    // Get the actual boolean value
    bool value() const { return passiveValue_; }
    
    // Implicit conversion to bool for use in if statements (only when not recording)
    operator bool() const { 
        // This is only safe when not recording - during recording we shouldn't branch
        if (GraphRecorder::isAnyRecording() && isActive_) {
            throw std::runtime_error("Cannot use fbool in if statement during recording - use .If() instead");
        }
        return passiveValue_; 
    }
    
    // Check if this Bool depends on runtime inputs
    bool isActive() const { return isActive_; }
    
    // Get the node ID (-1 if not in graph)
    NodeId nodeId() const { return activeNode_; }
    
    // Ensure this Bool has a node in the graph
    NodeId ensureNode() const;
    
    // Conditional selection - THE KEY FEATURE
    // This will be defined after Double is fully defined
    fdouble If(const fdouble& true_val, const fdouble& false_val) const;
    
    // Static If function for convenience (defined in cpp file due to circular dependency)
    static fdouble If(const fbool& condition, const fdouble& true_val, const fdouble& false_val);
    
    // Boolean operations
    fbool operator&&(const fbool& other) const;
    fbool operator||(const fbool& other) const;
    fbool operator!() const;
    
    // Comparison with another Bool (equality)
    fbool operator==(const fbool& other) const;
    fbool operator!=(const fbool& other) const;
};

} // namespace forge