#pragma once

#include <cmath>
#include <vector>

namespace forge {
namespace tools {
namespace test_functions {
namespace one_to_one {

// Special and mixed test functions (R → R)

// Absolute value approximation: x^2 / (x + 0.001)
template<typename T>
T absApprox(T x) {
    return x * x / (x + T(0.001));
}

// Step function approximation: x^3 / (x^3 + 1)
template<typename T>
T stepApprox(T x) {
    T x3 = x * x * x;
    return x3 / (x3 + T(1.0));
}

// Min/Max test: clamp between -2 and 2
template<typename T>
T clamp(T x) {
    return std::fmax(T(-2.0), std::fmin(T(2.0), x));
}

// Modulo test: fmod(|x|, 3)
template<typename T>
T moduloAbs(T x) {
    return std::fmod(std::abs(x), T(3.0));
}

// Combined operations with trig and exp
template<typename T>
T mixedOperations(T x) {
    T a = std::sin(x) + std::cos(x);
    T b = std::exp(-std::abs(x));
    T c = std::log(std::abs(a) + T(1.0));
    return std::fmax(b, c);
}

// Nested composition: sin(exp(x/4))
template<typename T>
T nestedSinExp(T x) {
    T scaled = x * T(0.25);  // Scale to avoid overflow
    T expPart = std::exp(scaled);
    return std::sin(expPart);
}

// Complex transcendental combination
template<typename T>
T transcendentalMix(T x) {
    T a = x + T(2.0);
    T b = std::sqrt(a * a + T(1.0));  // sqrt(a^2 + 1) always positive
    T c = std::exp(x * T(0.5));       // exp(x/2)
    T d = std::log(b + c);            // log of sum (always positive)
    T e = d * std::sqrt(c);           // Multiply by sqrt(exp(x/2))
    return e - T(1.0);
}

// Deep nesting test: iterative function application
template<typename T>
T deepNesting(T x) {
    T y = x;
    for (int i = 0; i < 10; ++i) {
        y = y * T(1.1) + T(0.1);
    }
    return y;
}

// CSE test function - has obvious duplicate subexpressions
template<typename T>
T cseTestFunc(T x) {
    // Create obvious duplicate subexpressions
    T a = x - T(0.5);        // First (x - 0.5)
    T b = x - T(0.5);        // Second (x - 0.5) - DUPLICATE!
    T c = a * b;             // Should use same node after CSE
    
    T d = x * x;             // First x^2
    T e = x * x;             // Second x^2 - DUPLICATE!
    T f = d + e;             // Should use same node after CSE
    
    T g = x + T(2.0);        // First (x + 2.0)
    T h = x + T(2.0);        // Second (x + 2.0) - DUPLICATE!
    T i = g * h;             // Should use same node after CSE
    
    return c + f + i;
}

// Test basic arithmetic: (x^2 - 2x + 1) / (x + 1)
template<typename T>
T arithmeticTest(T x) {
    T numerator = x * x - T(2.0) * x + T(1.0);
    T denominator = x + T(1.0);
    return numerator / denominator;
}

// Compound function 1: (2x + 1)(x - 3) / (x^2 + 1)
template<typename T>
T compound1(T x) {
    return (T(2.0) * x + T(1.0)) * (x - T(3.0)) / (x * x + T(1.0));
}

// Compound function 2: ((x+1)^2 + (x-1)^2) / ((x+1)(x-1) + 1)
template<typename T>
T compound2(T x) {
    T a = x + T(1.0);
    T b = x - T(1.0);
    return (a * a + b * b) / (a * b + T(1.0));
}

// Simple negation: -x
template<typename T>
T negation(T x) {
    return -x;
}

// Squared function: x^2
template<typename T>
T squared(T x) {
    return x * x;
}

// Reciprocal: 1/x (with safety offset)
template<typename T>
T reciprocal(T x) {
    return T(1.0) / (x + T(0.001));
}

// Min test: min(x, 2) + min(x-1, 0.5)
template<typename T>
T minTest(T x) {
    T a = std::fmin(x, T(2.0));
    T b = std::fmin(x - T(1.0), T(0.5));
    return a + b;
}

// Max test: max(x, -2) * max(x+1, 0)
template<typename T>
T maxTest(T x) {
    T a = std::fmax(x, T(-2.0));
    T b = std::fmax(x + T(1.0), T(0.0));
    return a * b;
}

// Min-max combination: max(min(x, 2), -2) + min(max(x, -1), 1)
template<typename T>
T minmaxCombo(T x) {
    T a = std::fmin(x, T(2.0));
    T b = std::fmax(a, T(-2.0));
    T c = std::fmax(x, T(-1.0));
    T d = std::fmin(c, T(1.0));
    return b + d;
}

// Function with exactly 10 operations for benchmarking
template<typename T>
T ops10(T x) {
    T a = x * T(2.0);           // 1: mul
    T p = T(1.0) + T(1.0);
    T u = p * T(2.0);
    T v = u * T(2.0);
    T b = x * x + v;            // 2: mul, 3: add
    T c = a + b + u;            // 4: add, 5: add
    T d = c * T(3.0);           // 6: mul
    T e = d + x;                // 7: add
    T f = e * T(1.5);           // 8: mul
    T g = f - b;                // 9: sub
    T h = g + T(10.0);          // 10: add
    T i = h * T(0.5);           // 11: mul
    T j = x + T(1.0);           // intermediate for div
    return i / j;               // 12: div
}

// Function with exactly 50 operations for benchmarking
template<typename T>
T ops50(T x) {
    T v = x;
    for (int i = 0; i < 5; ++i) {
        T t1 = v * T(1.01);
        T t2 = t1 + T(0.01);
        T t3 = t2 * v;
        T t4 = t3 - x;
        T t5 = t4 * T(0.5);
        T t6 = t5 + v;
        T t7 = t6 / T(2.5);
        T t8 = t7 * T(0.9);
        T t9 = t8 - T(0.1);
        v = t9 * T(0.95) + x * T(0.01);
    }
    return v;
}

// Complex massive function with additional operations
template<typename T>
T massiveComplex(T x) {
    T v1 = x * x + T(1.0);
    T v2 = std::sqrt(v1);
    T v3 = v2 + x;
    T v4 = v3 * v3;
    T v5 = v4 - v2;
    T v6 = v5 * T(0.5);
    T v7 = v6 / (v5 + T(0.01));
    T v8 = v7 * v7 * v7;
    T v9 = v8 - v6 + v4;
    T v10 = v9 * x + v1;
    return v10 / (v10 * v10 + T(0.1));
}

// Ultra massive iterative function for stress testing
template<typename T>
T ultraMassiveIterative(T x, int iterations = 10) {
    T v = x;
    T sum = T(0.0);
    for (int i = 0; i < iterations; ++i) {
        T a = v * v + T(0.1);
        T b = std::sqrt(a);
        T c = b * v - T(1.0);
        T d = c / (a + T(0.01));
        T e = d * d + v;
        T f = e - b * T(0.5);
        T g = f * f * f;
        T h = g / (g + T(1.0));
        v = (h + x * T(0.1)) * T(0.95);
        sum += h;
    }
    return sum / T(iterations);
}

// Exponential stress test with negation
template<typename T>
T expNegativeStress(T x) {
    T a = -x;
    T b = std::exp(a * T(0.1));
    T c = -b;
    T d = std::exp(c * T(0.1) + T(1.0));
    return d;
}

// Absolute value
template<typename T>
T absolute(T x) {
    return std::abs(x);
}

// Test input sets for special functions
inline std::vector<double> getSpecialInputs() {
    return {-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5};
}

// Test inputs for modulo operations
inline std::vector<double> getModuloInputs() {
    return {-7.5, -3, -1.5, 0, 1.5, 3, 4.5, 7.5};
}

} // namespace one_to_one
} // namespace test_functions
} // namespace tools
} // namespace forge