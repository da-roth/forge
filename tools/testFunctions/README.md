# Test Functions Summary

## Overview
This directory contains a collection of test functions designed as templates that can work with any numeric type, particularly with native `double` and other custom numeric types used in the forge framework.

## Template Design
All test functions are implemented as C++ templates using the pattern:
```cpp
template<typename T>
T functionName(T x) { ... }
```

This template approach allows the functions to accept any numeric type that supports the required arithmetic and mathematical operations, making them compatible with both native `double` and custom types.

## Function Categories

### 1. Polynomials (`polynomials.hpp`)
- Linear, quadratic, cubic, quartic, and fifth-power functions
- General polynomials and alternating polynomials
- All operations use template type `T` for constants (e.g., `T(2.0)`)

### 2. Trigonometric (`trigonometric.hpp`)
- Basic: sine, cosine, tangent
- Combined and multi-term trigonometric expressions
- Identity tests (sin²(x) + cos²(x))
- Range-limited tangent functions to avoid singularities

### 3. Exponential & Logarithmic (`exponential.hpp`)
- Basic exponential, logarithm, and square root
- Scaled versions to avoid overflow
- Safe logarithm/sqrt with guaranteed positive inputs (using x² + 1)
- Complex transcendental compositions

### 4. Rational Functions (`rational.hpp`)
- Inverse and inverse-squared functions
- Rational expressions with quadratic numerators/denominators
- Gaussian-like and Lorentzian functions
- Small offsets added to denominators to prevent division by zero

### 5. Special Functions (`special.hpp`)
- Approximations (absolute value, step function)
- Clamping and modulo operations
- Mixed operations combining trig, exponential, and logarithmic
- CSE test function with intentional duplicate subexpressions
- Deep nesting for testing recursive evaluation

## Key Design Features

1. **Type-Generic**: All functions use template parameter `T` for maximum flexibility
2. **Safe Operations**: Functions include safeguards against:
   - Division by zero (adding small offsets)
   - Negative logarithm arguments (using x² + 1 patterns)
   - Exponential overflow (input scaling)
3. **Test Input Sets**: Each header provides appropriate input vectors for testing
4. **Namespace Organization**: Clean namespace hierarchy under `forge::tools::test_functions::one_to_one`

## Usage Example
```cpp
#include "tools/testFunctions/oneToOne/all.hpp"

// Works with native double
double result1 = forge::tools::test_functions::one_to_one::quadratic(3.14);

// Works with any custom numeric type
CustomType x(2.0);
CustomType result2 = forge::tools::test_functions::one_to_one::exponential(x);
```

## Purpose
These test functions serve as a comprehensive suite for validating automatic differentiation implementations, numerical computations, and type compatibility across the forge framework.