#pragma once

#include <cmath>
#include <vector>

// Define PI if not available
#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif

namespace forge {
namespace tools {
namespace test_functions {
namespace one_to_one {

// Trigonometric test functions (R → R)

// Simple sine
template<typename T>
T sine(T x) {
    return std::sin(x);
}

// Simple cosine
template<typename T>
T cosine(T x) {
    return std::cos(x);
}

// Simple tangent (with range limiting to avoid singularities)
template<typename T>
T tangent(T x) {
    // Limit to (-π/3, π/3) to avoid singularities near π/2
    T limited = std::fmax(T(-M_PI/3), std::fmin(T(M_PI/3), x));
    return std::tan(limited);
}

// Combined trig: sin(x) * cos(x) + tan(x/2)
template<typename T>
T trigCombined(T x) {
    return std::sin(x) * std::cos(x) + std::tan(x * T(0.5));
}

// Sine with multiple terms: sin(x) + 2*sin(x/2) - 0.5*sin(x+1)
template<typename T>
T sineMultiTerm(T x) {
    T a = std::sin(x);
    T b = std::sin(x * T(0.5));
    T c = std::sin(x + T(1.0));
    return a + b * T(2.0) - c * T(0.5);
}

// Cosine with multiple terms: cos²(x) + cos(2x) - 1.5*cos(x-0.5)
template<typename T>
T cosineMultiTerm(T x) {
    T a = std::cos(x);
    T b = std::cos(x * T(2.0));
    T c = std::cos(x - T(0.5));
    return a * a + b - c * T(1.5);
}

// Tangent composition: (tan(x) + tan(x/2)) / (1 + |tan(x)*tan(x/2)|)
template<typename T>
T tangentComposition(T x) {
    // Use limited range to avoid singularities
    T limited = std::fmax(T(-1.5), std::fmin(T(1.5), x));
    T a = std::tan(limited);
    T b = std::tan(limited * T(0.5));
    return (a + b) / (T(1.0) + std::abs(a * b));
}

// Trigonometric identity test: sin²(x) + cos²(x) - should always be 1
template<typename T>
T trigIdentity(T x) {
    T s = std::sin(x);
    T c = std::cos(x);
    return s * s + c * c;
}

// Sine test: sin(x) + sin(2x) + sin(x/2)
template<typename T>
T sinTest(T x) {
    T a = std::sin(x);
    T b = std::sin(x * T(2.0));
    T c = std::sin(x * T(0.5));
    return a + b + c;
}

// Cosine test: cos(x) + 2*cos(2x) - cos(x+1)
template<typename T>
T cosTest(T x) {
    T a = std::cos(x);
    T b = std::cos(x * T(2.0));
    T c = std::cos(x + T(1.0));
    return a + b * T(2.0) - c;
}

// Tangent test: tan(x) + tan(x/2) - tan(x/3)
template<typename T>
T tanTest(T x) {
    T limited = std::fmax(T(-1.0), std::fmin(T(1.0), x));
    T a = std::tan(limited);
    T b = std::tan(limited * T(0.5));
    T c = std::tan(limited / T(3.0));
    return a + b - c;
}

// Trigonometric combination: sin(x)*cos(x) + tan(x/2) - cos(2x)
template<typename T>
T trigCombo(T x) {
    T a = std::sin(x);
    T b = std::cos(x);
    T c = std::tan(x * T(0.5));
    T d = std::cos(x * T(2.0));
    return a * b + c - d;
}

// Test input sets for trigonometric functions
inline std::vector<double> getTrigonometricInputs() {
    return {0, M_PI/6, M_PI/4, M_PI/3, M_PI/2, 2*M_PI/3, 3*M_PI/4, 5*M_PI/6, M_PI};
}

// Smaller range for functions involving tangent
inline std::vector<double> getTangentInputs() {
    return {-1.5, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 1.5};
}

// Smaller range for functions involving tangent
inline std::vector<double> getTangentInputsShort() {
    return {-1.5, -1.0};
}

} // namespace one_to_one
} // namespace test_functions
} // namespace tools
} // namespace forge