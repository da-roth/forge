#pragma once

#include <cstdint>

namespace forge {
namespace core {

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
    
    // No conversions - aint is purely integer-only
    
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
    ArrayIndex     // Double array[aint index] - dynamic array access
};

} // namespace core
} // namespace forge