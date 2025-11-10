#pragma once

#include <cmath>
#include <vector>

namespace forge {
namespace tools {
namespace test_functions {
namespace one_to_one {

// Exponential and logarithmic test functions (R → R)

// Simple exponential
template<typename T>
T exponential(T x) {
    return std::exp(x);
}

// Simple logarithm (with positive input guarantee)
template<typename T>
T logarithm(T x) {
    // Ensure positive input
    T safe = x * x + T(1.0);  // x^2 + 1 is always >= 1
    return std::log(safe);
}

// Simple square root (with positive input guarantee)
template<typename T>
T squareRoot(T x) {
    // Ensure positive input
    T safe = x * x + T(1.0);  // x^2 + 1 is always >= 1
    return std::sqrt(safe);
}

// Exponential with scaling to avoid overflow: 0.1 * exp(x/2 + 1) - 2
template<typename T>
T expScaled(T x) {
    T a = x * T(0.5);           // Scale input to avoid overflow
    T b = a + T(1.0);           // Shift
    T c = std::exp(b);          // Apply exponential
    T d = c * T(0.1);           // Scale result
    return d - T(2.0);          // Shift result
}

// Logarithm with conditioning: 3 * (log(2(x^2 + 1)) + 1)
template<typename T>
T logConditioned(T x) {
    T a = x * x + T(1.0);       // Ensure positive (x^2 + 1 >= 1)
    T b = a * T(2.0);           // Scale
    T c = std::log(b);          // Apply logarithm
    T d = c + T(1.0);           // Shift
    return d * T(3.0);          // Scale result
}

// Square root with operations: (2*sqrt(x^2 + 4) - 3)^2
template<typename T>
T sqrtWithOps(T x) {
    T a = x * x + T(4.0);       // Ensure positive (x^2 + 4 >= 4)
    T b = std::sqrt(a);         // Apply square root
    T c = b * T(2.0);           // Scale
    T d = c - T(3.0);           // Shift
    return d * d;               // Square the result
}

// Combined exponential and logarithm: exp(x) * log(x^2 + 1)
template<typename T>
T expLogCombo(T x) {
    T expPart = std::exp(x * T(0.2));  // Scale x to avoid overflow
    T logPart = std::log(x * x + T(1.0));
    return expPart * logPart;
}

// Nested composition: sqrt(exp(x/4) + 1)
template<typename T>
T nestedExpSqrt(T x) {
    T expPart = std::exp(x * T(0.25));  // Scale to avoid overflow
    return std::sqrt(expPart + T(1.0));
}

// Complex transcendental: (exp(x/2) * sqrt(log(x^2 + 2))) / (x^2 + 1)
template<typename T>
T transcendentalComplex(T x) {
    T a = x * x + T(2.0);              // Ensure positive for log
    T b = std::log(a);                 // log(x^2 + 2)
    T c = std::sqrt(b);                // sqrt(log(...))
    T d = std::exp(x * T(0.5));        // exp(x/2)
    T e = d * c;                       // Multiply
    T f = x * x + T(1.0);              // Denominator
    return e / f;
}

// Exponential test: exp(x/2 + 1) * 0.1 - 2
template<typename T>
T expTest(T x) {
    T a = x * T(0.5);
    T b = a + T(1.0);
    T c = std::exp(b);
    T d = c * T(0.1);
    return d - T(2.0);
}

// Logarithm test: 3 * log(2x^2 + 2) + 1
template<typename T>
T logTest(T x) {
    T a = x * x * T(2.0) + T(2.0);
    T b = std::log(a);
    T c = b + T(1.0);
    return c * T(3.0);
}

// Square root test: (2*sqrt(x^2 + 4) - 3)^2
template<typename T>
T sqrtTest(T x) {
    T a = x * x + T(4.0);
    T b = std::sqrt(a);
    T c = b * T(2.0);
    T d = c - T(3.0);
    return d * d;
}

// Transcendental combination: exp(x) * log(x^2 + 1) / (x^2 + 0.1)
template<typename T>
T transcendentalCombo(T x) {
    T expPart = std::exp(x * T(0.2));
    T logPart = std::log(x * x + T(1.0));
    T denominator = x * x + T(0.1);
    return expPart * logPart / denominator;
}

// Simple power function: x^2.5 (positive input guaranteed)
template<typename T>
T powerTest(T x) {
    T safe = x * x + T(1.0);  // Ensure positive input: x^2 + 1 >= 1
    return std::pow(safe, T(2.5));  // safe^2.5
}

// Power with integer exponent: (x^2 + 2)^3
template<typename T>
T powerIntegerTest(T x) {
    T base = x * x + T(2.0);  // Ensure positive: x^2 + 2 >= 2
    return std::pow(base, T(3.0));  // Use integer power (should use our x*x*x optimization)
}

// Fractional power: (x^2 + 1)^(1/3)
template<typename T>
T powerFractionalTest(T x) {
    T base = x * x + T(1.0);  // Ensure positive: x^2 + 1 >= 1
    return std::pow(base, T(1.0/3.0));  // Cube root
}

// Complex power expression: 2 * (x^2 + 3)^1.5 - 1
template<typename T>
T powerComplexTest(T x) {
    T base = x * x + T(3.0);  // Ensure positive: x^2 + 3 >= 3
    T powered = std::pow(base, T(1.5));
    return T(2.0) * powered - T(1.0);
}

// Power with negative base (for integer exponents): (-x-1)^3
template<typename T>
T powerNegativeBaseIntTest(T x) {
    T base = -x - T(1.0);  // Ensure negative: -x - 1 <= -1 for x >= 0
    return std::pow(base, T(3.0));  // Odd integer power preserves sign
}

// Power with negative base (even exponent): (-x-1)^2
template<typename T>
T powerNegativeBaseEvenTest(T x) {
    T base = -x - T(1.0);  // Ensure negative: -x - 1 <= -1 for x >= 0
    return std::pow(base, T(2.0));  // Even integer power makes it positive
}

// Power with very small exponent: (x^2 + 40)^0.01
template<typename T>
T powerSmallExponentTest(T x) {
    T base = x * x + T(40.0);  // Ensure positive and reasonably large
    return std::pow(base, T(0.01));  // Very small exponent (like 100th root)
}

// Power with large base and small exponent: (x + 5)^0.01
template<typename T>
T powerLargeBaseSmallExpTest(T x) {
    // Make sure base is positive and reasonably large
    T base = x + T(5.0);  // base ranges from 2 to 7 for x in [-3, 2]
    T exponent = T(0.01);
    return std::pow(base, exponent);  // Direct pow operation
}

// Power with negative result through odd root of negative: (-abs(x)-1)^(1/3)
// Note: This may not work correctly with std::pow for negative bases with fractional exponents
// We handle it by using cbrt for cube root of negative numbers
template<typename T>
T powerNegativeCubeRootTest(T x) {
    T base = std::abs(x) + T(1.0);  // Ensure positive: |x| + 1 >= 1
    // For negative cube root, we can't use pow with negative base and fractional exponent
    // Instead, we compute positive cube root and negate
    T result = std::pow(base, T(1.0/3.0));
    return -result;  // Make it negative
}

// Complex power with varying bases: (2 + sin(x))^(3 - cos(x))
template<typename T>
T powerVaryingBaseAndExpTest(T x) {
    T base = T(2.0) + std::sin(x);  // Base varies between 1 and 3
    T exponent = T(3.0) - std::cos(x);  // Exponent varies between 2 and 4
    return std::pow(base, exponent);
}

// Power tower test: x^(x^2 + 1) for small x
template<typename T>
T powerTowerTest(T x) {
    // Limit x to avoid overflow
    T safe_x = x * T(0.1) + T(1.0);  // Transform to range around 1
    T exponent = safe_x * safe_x + T(1.0);
    return std::pow(safe_x, exponent);
}

// Test input sets for exponential functions
inline std::vector<double> getExponentialInputs() {
    return {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2};
}

// Smaller range for functions that can overflow easily
inline std::vector<double> getSafeExponentialInputs() {
    return {-2, -1, -0.5, 0, 0.5, 1, 2};
}

// Special inputs for power functions with extreme values
inline std::vector<double> getPowerExtremeInputs() {
    return {-3, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 3};
}


} // namespace one_to_one
} // namespace test_functions  
} // namespace tools
} // namespace forge