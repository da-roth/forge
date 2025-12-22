#pragma once

#include "../../src/graph/graph.hpp"
#include "../../src/graph/handles.hpp"
#include "../../src/graph/graph_recorder.hpp"
#include <stdexcept>

namespace forge {

// Forward declarations
class fbool;
class fint;

class fdouble {
    friend class fbool;  // Allow fbool to access private members
    friend class fint;   // Allow fint to access private members
private:
    double passiveValue_;
    NodeId activeNode_;
    bool isActive_;
    bool needsGradient_;  // AAD: Track gradient requirement
    
    // Private constructor with gradient flag
    fdouble(NodeId node, double val, bool active, bool needsGrad) 
        : passiveValue_(val), activeNode_(node), isActive_(active), needsGradient_(needsGrad) {}
    
public:
    fdouble() : passiveValue_(0.0), activeNode_(static_cast<NodeId>(-1)), isActive_(false), needsGradient_(false) {}
    explicit fdouble(double val) : passiveValue_(val), activeNode_(static_cast<NodeId>(-1)), isActive_(false), needsGradient_(false) {}
    fdouble(int val) : passiveValue_(static_cast<double>(val)), activeNode_(static_cast<NodeId>(-1)), isActive_(false), needsGradient_(false) {}
    
    double value() const { return passiveValue_; }
    NodeId node() const { return activeNode_; }
    bool isActive() const { return isActive_; }
    bool isRecording() const { return GraphRecorder::isAnyRecording(); }
    
    InputHandle markInput();
    InputHandle markInputAndDiff();  // AAD: Mark as input AND request gradient
    ResultHandle markOutput();
    
    // Modified factory method with gradient flag
    static fdouble fromNode(NodeId node, double val, bool active = true, bool needsGrad = false) {
        return fdouble(node, val, active, needsGrad);
    }
    
    fdouble operator+(const fdouble& rhs) const;
    fdouble operator-(const fdouble& rhs) const;
    fdouble operator*(const fdouble& rhs) const;
    fdouble operator/(const fdouble& rhs) const;
    
    fdouble& operator+=(const fdouble& rhs);
    fdouble& operator-=(const fdouble& rhs);
    fdouble& operator*=(const fdouble& rhs);
    fdouble& operator/=(const fdouble& rhs);
    
    // Overloads for double
    fdouble operator+(double rhs) const { return *this + fdouble(rhs); }
    fdouble operator-(double rhs) const { return *this - fdouble(rhs); }
    fdouble operator*(double rhs) const { return *this * fdouble(rhs); }
    fdouble operator/(double rhs) const { return *this / fdouble(rhs); }
    
    fdouble& operator+=(double rhs) { return *this += fdouble(rhs); }
    fdouble& operator-=(double rhs) { return *this -= fdouble(rhs); }
    fdouble& operator*=(double rhs) { return *this *= fdouble(rhs); }
    fdouble& operator/=(double rhs) { return *this /= fdouble(rhs); }
    
    fdouble operator-() const;
    
    operator double() const {
        if (GraphRecorder::isAnyRecording() && isActive_) {
            throw std::runtime_error("Cannot convert active Double to passive during recording");
        }
        return passiveValue_;
    }
    
    // Comparison operators that return fbool for template compatibility
    fbool operator<(const fdouble& other) const;
    fbool operator<=(const fdouble& other) const;
    fbool operator>(const fdouble& other) const;
    fbool operator>=(const fdouble& other) const;
    fbool operator==(const fdouble& other) const;
    fbool operator!=(const fdouble& other) const;
    
    // Overloads for comparison with double
    fbool operator<(double other) const;
    fbool operator<=(double other) const;
    fbool operator>(double other) const;
    fbool operator>=(double other) const;
    fbool operator==(double other) const;
    fbool operator!=(double other) const;
    
private:
    NodeId ensureNode() const;
    static fdouble binaryOp(const fdouble& a, const fdouble& b, OpCode op);
    
    // Friend functions for new operations
    friend fdouble abs(const fdouble& x);
    friend fdouble square(const fdouble& x);
    friend fdouble recip(const fdouble& x);
    friend fdouble mod(const fdouble& x, const fdouble& y);
    friend fdouble exp(const fdouble& x);
    friend fdouble log(const fdouble& x);
    friend fdouble sqrt(const fdouble& x);
    friend fdouble pow(const fdouble& x, const fdouble& y);
    friend fdouble sin(const fdouble& x);
    friend fdouble cos(const fdouble& x);
    friend fdouble tan(const fdouble& x);
    friend fdouble min(const fdouble& x, const fdouble& y);
    friend fdouble max(const fdouble& x, const fdouble& y);
    friend fbool cmpLT(const fdouble& x, const fdouble& y);
    friend fbool cmpLE(const fdouble& x, const fdouble& y);
    friend fbool cmpGT(const fdouble& x, const fdouble& y);
    friend fbool cmpGE(const fdouble& x, const fdouble& y);
    friend fbool cmpEQ(const fdouble& x, const fdouble& y);
    friend fbool cmpNE(const fdouble& x, const fdouble& y);
};

fdouble operator+(double lhs, const fdouble& rhs);
fdouble operator-(double lhs, const fdouble& rhs);
fdouble operator*(double lhs, const fdouble& rhs);
fdouble operator/(double lhs, const fdouble& rhs);

// Math functions as free functions
fdouble abs(const fdouble& x);
fdouble square(const fdouble& x);
fdouble recip(const fdouble& x);
fdouble mod(const fdouble& x, const fdouble& y);
fdouble exp(const fdouble& x);
fdouble log(const fdouble& x);
fdouble sqrt(const fdouble& x);
fdouble pow(const fdouble& x, const fdouble& y);
fdouble sin(const fdouble& x);
fdouble cos(const fdouble& x);
fdouble tan(const fdouble& x);
fdouble min(const fdouble& x, const fdouble& y);
fdouble max(const fdouble& x, const fdouble& y);

// Comparison functions (return fbool)
fbool cmpLT(const fdouble& x, const fdouble& y);  // x < y
fbool cmpLE(const fdouble& x, const fdouble& y);  // x <= y
fbool cmpGT(const fdouble& x, const fdouble& y);  // x > y
fbool cmpGE(const fdouble& x, const fdouble& y);  // x >= y
fbool cmpEQ(const fdouble& x, const fdouble& y);  // x == y
fbool cmpNE(const fdouble& x, const fdouble& y);  // x != y

} // namespace forge

// Add std namespace overloads for uniform template usage
namespace std {
    inline forge::fdouble exp(const forge::fdouble& x) { return forge::exp(x); }
    inline forge::fdouble log(const forge::fdouble& x) { return forge::log(x); }
    inline forge::fdouble sqrt(const forge::fdouble& x) { return forge::sqrt(x); }
    inline forge::fdouble sin(const forge::fdouble& x) { return forge::sin(x); }
    inline forge::fdouble cos(const forge::fdouble& x) { return forge::cos(x); }
    inline forge::fdouble tan(const forge::fdouble& x) { return forge::tan(x); }
    inline forge::fdouble abs(const forge::fdouble& x) { return forge::abs(x); }
    inline forge::fdouble fmod(const forge::fdouble& x, const forge::fdouble& y) { return forge::mod(x, y); }
    inline forge::fdouble fmin(const forge::fdouble& x, const forge::fdouble& y) { return forge::min(x, y); }
    inline forge::fdouble fmax(const forge::fdouble& x, const forge::fdouble& y) { return forge::max(x, y); }
    inline forge::fdouble pow(const forge::fdouble& x, const forge::fdouble& y) { return forge::pow(x, y); }
}