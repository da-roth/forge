#pragma once

#include <cmath>
#include <vector>

namespace forge {
namespace tools {
namespace test_functions {
namespace one_to_one {

// Rational and algebraic test functions (R → R)

// Simple inverse: 1/x (with small offset to avoid division by zero)
template<typename T>
T inverse(T x) {
    return T(1.0) / (x + T(0.001));
}

// Inverse squared: 1/x^2 (with small offset)
template<typename T>
T inverseSquared(T x) {
    T denom = x * x + T(0.001);  // Avoid division by zero
    return T(1.0) / denom;
}

// Simple rational: (x + 1) / (x - 1) (with offset)
template<typename T>
T simpleRational(T x) {
    return (x + T(1.0)) / (x - T(1.0) + T(0.001));
}

// Rational with quadratic: (x^2 + 1) / (x + 0.5)
template<typename T>
T quadraticRational(T x) {
    return (x * x + T(1.0)) / (x + T(0.5));
}

// Complex rational: ((x + 1)(x - 1)(x + 2)) / (x^2 + 0.1)
template<typename T>
T complexRational(T x) {
    T numerator = (x + T(1.0)) * (x - T(1.0)) * (x + T(2.0));
    T denominator = x * x + T(0.1);  // Add constant to avoid division by zero
    return numerator / denominator;
}

// Rational function: (x^2 + 2x + 1) / (x^2 + 0.1)
template<typename T>
T rationalFunction(T x) {
    T numerator = x * x + T(2.0) * x + T(1.0);
    T denominator = x * x + T(0.1);  // Add small constant to avoid division by zero
    return numerator / denominator;
}

// Gaussian-like: 1 / (1 + x^2)
template<typename T>
T gaussianLike(T x) {
    return T(1.0) / (T(1.0) + x * x);
}

// Lorentzian: 1 / (1 + 4x^2)
template<typename T>
T lorentzian(T x) {
    return T(1.0) / (T(1.0) + T(4.0) * x * x);
}

// Nested rational: ((x + 2) * 3 - 1) / (x - 0.5)
template<typename T>
T nestedRational(T x) {
    T numerator = (x + T(2.0)) * T(3.0) - T(1.0);
    T denominator = x - T(0.5) + T(0.001);  // Avoid division by zero
    return numerator / denominator;
}

// Compound rational: ((x + 1)^2 + (x - 1)^2) / ((x + 1)(x - 1) + 1)
template<typename T>
T compoundRational(T x) {
    T a = x + T(1.0);
    T b = x - T(1.0);
    T numerator = a * a + b * b;
    T denominator = a * b + T(1.0);
    return numerator / denominator;
}

// Test input sets for rational functions
inline std::vector<double> getRationalInputs() {
    return {-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5};
}

// Inputs that avoid singularities for simple rationals
inline std::vector<double> getSafeRationalInputs() {
    return {-5, -2, -0.5, 0.1, 0.5, 1.5, 2, 3, 5};
}

} // namespace one_to_one
} // namespace test_functions
} // namespace tools
} // namespace forge