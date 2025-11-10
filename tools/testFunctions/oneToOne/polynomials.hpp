#pragma once

#include <cmath>
#include <vector>

namespace forge {
namespace tools {
namespace test_functions {
namespace one_to_one {

// Polynomial test functions (R → R)

// Linear: 2x + 3
template<typename T>
T linear(T x) {
    return T(2.0) * x + T(3.0);
}

// Quadratic: x^2 + 2x + 1
template<typename T>
T quadratic(T x) {
    return x * x + T(2.0) * x + T(1.0);
}

// Cubic: x^3 - 3x^2 + 3x - 1
template<typename T>
T cubic(T x) {
    return x * x * x - T(3.0) * x * x + T(3.0) * x - T(1.0);
}

// Quartic: x^4 - 4x^3 + 6x^2 - 4x + 1
template<typename T>
T quartic(T x) {
    return x * x * x * x - T(4.0) * x * x * x + T(6.0) * x * x - T(4.0) * x + T(1.0);
}

// Fifth power: x^5
template<typename T>
T powerFive(T x) {
    return x * x * x * x * x;
}

// General polynomial: 3x^4 - 2x^3 + x^2 - 5x + 7
template<typename T>
T polynomial(T x) {
    return T(3.0) * x * x * x * x - T(2.0) * x * x * x + x * x - T(5.0) * x + T(7.0);
}

// Alternating polynomial: x - x^2 + x^3 - x^4
template<typename T>
T alternatingPoly(T x) {
    return x - x * x + x * x * x - x * x * x * x;
}

// Taylor series approximation of sin(x): x - x^3/6
template<typename T>
T sineApprox(T x) {
    return x - (x * x * x) / T(6.0);
}

// Taylor series approximation of cos(x): 1 - x^2/2
template<typename T>
T cosineApprox(T x) {
    return T(1.0) - (x * x) / T(2.0);
}

// Test input sets for polynomial functions
inline std::vector<double> getPolynomialInputs() {
    return {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2};
}

} // namespace one_to_one
} // namespace test_functions
} // namespace tools
} // namespace forge