#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace forge {

using NodeId = uint32_t;
using SlotId = uint32_t;

enum class OpCode : uint16_t {
    Input,
    Constant,
    Add,
    Sub,
    Mul,
    Div,
    Neg,      // Negation: -x
    Abs,      // Absolute value: |x|
    Square,   // Square: x*x (faster than Pow)
    Recip,    // Reciprocal: 1/x
    Mod,      // Modulo: x % y
    Exp,
    Log,
    Sqrt,
    Pow,      // Power function: x^y
    Sin,      // Sine function
    Cos,      // Cosine function  
    Tan,      // Tangent function
    Min,
    Max,
    If,       // Conditional: bool ? true_val : false_val
    CmpLT,    // Comparison: returns Bool
    CmpLE,    // Comparison: returns Bool
    CmpGT,    // Comparison: returns Bool
    CmpGE,    // Comparison: returns Bool
    CmpEQ,    // Comparison: returns Bool
    CmpNE,    // Comparison: returns Bool
    
    // Boolean-specific operations
    BoolConstant,  // Boolean constant (0.0 or 1.0 in imm field)
    BoolAnd,       // Logical AND: a && b
    BoolOr,        // Logical OR: a || b
    BoolNot,       // Logical NOT: !a
    BoolEq,        // Bool equality: a == b
    BoolNe,        // Bool inequality: a != b
    
    // Integer-specific operations
    IntConstant,   // Integer constant (stored as double in imm field)
    IntAdd,        // Integer addition
    IntSub,        // Integer subtraction
    IntMul,        // Integer multiplication
    IntDiv,        // Integer division (truncating)
    IntMod,        // Integer modulo
    IntNeg,        // Integer negation
    
    // No conversions - fint is purely integer-only
    
    // Integer comparisons (return Bool)
    IntCmpLT,      // Int < Int
    IntCmpLE,      // Int <= Int
    IntCmpGT,      // Int > Int
    IntCmpGE,      // Int >= Int
    IntCmpEQ,      // Int == Int
    IntCmpNE,      // Int != Int
    
    // Integer conditional
    IntIf,         // Bool ? Int : Int
    
    // Array indexing
    ArrayIndex     // Double array[fint index] - dynamic array access
};

struct Node {
    OpCode op;
    NodeId dst{};
    NodeId a{};
    NodeId b{};
    NodeId c{};
    uint32_t flags{};
    double imm{};
    bool isActive{true};  // Phase 2.2.0: Track if node depends on inputs (false = constant)
    bool isDead{false};   // Optimization: Mark nodes that have been folded and can be skipped
    bool needsGradient{false};  // AAD: Track if this node requires gradient computation
};

struct Graph {
    std::vector<Node> nodes;
    std::vector<double> constPool;
    std::vector<NodeId> outputs;
    std::vector<NodeId> diff_inputs;  // AAD: Inputs marked for differentiation
    
    NodeId addNode(const Node& node);
    NodeId addConstant(double value);
    NodeId addInput();
    void markOutput(NodeId node);
    
    void clear();
    bool empty() const { return nodes.empty(); }
    size_t size() const { return nodes.size(); }
};

} // namespace forge