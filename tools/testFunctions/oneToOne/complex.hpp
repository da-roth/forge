#pragma once

#include <cmath>
#include <vector>

namespace forge {
namespace tools {
namespace test_functions {
namespace one_to_one {

// Complex test functions with many operations (R → R)
// These functions stress-test the compiler's register allocation and instruction scheduling
// Note: ops10 and ops50 are already defined in special.hpp

// Function with nested operations - tests expression depth
template<typename T>
T opsNested(T x) {
    // Deep nesting tests the compiler's ability to handle complex expression trees
    T safe = x * x + T(1.0);
    return std::exp(
        std::log(
            std::sqrt(
                std::sin(
                    std::cos(
                        safe * T(0.5) + T(1.0)
                    ) * T(2.0)
                ) + T(1.0)
            ) * T(1.5)
        ) * T(0.8)
    );
}

// Function mixing arithmetic and transcendental operations
template<typename T>
T opsMixed(T x) {
    T safe = x * x + T(0.1);
    
    // Mix of arithmetic and transcendental functions
    T a = (safe + T(1.0)) * T(2.0);
    T b = std::exp(a * T(0.1)) + T(0.5);
    T c = std::log(b) * T(3.0);
    T d = std::sin(c * T(0.2)) + T(1.0);
    T e = std::cos(d) * T(2.5);
    T f = (e * e + T(1.0)) / T(2.0);
    T g = std::sqrt(f) - T(0.3);
    T h = (g + T(1.0)) * (g - T(0.5));
    T i = h / (h + T(1.0));
    T j = std::exp(i * T(0.5));
    
    return j;
}

// Function with repeated patterns - tests compiler optimization
template<typename T>
T opsRepeated(T x) {
    T safe = x + T(10.0); // Ensure positive
    T result = safe;
    
    // Repeat the same pattern multiple times
    for (int i = 0; i < 5; ++i) {
        result = result * T(1.1) + T(0.1);
        result = std::sqrt(result + T(0.01));
        result = result / T(1.05) - T(0.05);
        result = result * result + T(0.1);
    }
    
    return result;
}

// Function with many binary operations - tests register allocation under pressure
template<typename T>
T opsBinary(T x) {
    // Create many intermediate values that need to be kept alive
    T safe = x + T(5.0);
    
    T a = safe * T(1.1);
    T b = safe * T(1.2);  
    T c = safe * T(1.3);
    T d = safe * T(1.4);
    T e = safe * T(1.5);
    
    T f = a + b;
    T g = c + d;
    T h = e + a;
    T i = b + c;
    T j = d + e;
    
    T k = f * g;
    T l = h * i;
    T m = j * f;
    T n = g * h;
    T o = i * j;
    
    T p = k + l + m;
    T q = n + o + k;
    T r = l + m + n;
    T s = o + p + q;
    
    return (p * q + r * s) / T(4.0);
}

// Test input sets for complex functions
inline std::vector<double> getComplexInputs() {
    return {-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5};
}

// Smaller range for functions that might be sensitive to large inputs
inline std::vector<double> getSafeComplexInputs() {
    return {-2, -1, -0.5, 0, 0.5, 1, 1.5, 2};
}

// Massive expression function with 100+ operations - template version converted from test_functions_1d.hpp
template<typename T>
T massiveExpression(T x) {
    // Start with some basic transformations (10 ops)
    T a = x * T(2.0) + T(3.0);
    T b = x * x - T(1.0);
    T c = (x + T(1.0)) * (x - T(1.0));
    T d = (x - T(0.5)) * (x - T(0.5));
    T e = (x + T(0.1)) * (x + T(0.1));
    
    // Some divisions and negations (10 ops)
    T f = T(1.0) / (a + T(1.0));
    T g = -b * T(2.0);
    T h = T(1.0) / (c + T(10.0));
    T i = -d + e;
    T j = f * f + g * g;
    
    // Nested operations (20 ops)
    T k = (a + b) * (c - d) / (e + T(1.0));
    T l = (k - T(5.0)) * (k - T(5.0)) + h * h;
    T m = (T(1.0) / (i * i + T(0.1))) * j;
    T n = -(k + l) * (m - T(2.0));
    T o = x * T(0.5) * x * T(0.5) * x * T(0.5) * x * T(0.5);  // x^4 / 16
    
    // More complex combinations (20 ops)
    T p = (n + o) / (m * m + T(1.0));
    T q = T(1.0) / (p * p + T(0.01));
    T r = (q - T(0.5)) * (q + T(0.5));
    T s = r * r + (T(1.0) - r) * (T(1.0) - r);
    T t = (s * p + q * n) / (o + T(1.0));
    
    // Wave-like patterns (20 ops)
    T u = t * (T(1.0) + x * T(0.1));
    T v = u - (u * T(0.5)) * (u * T(0.5));
    T w = v + (v - T(0.5)) * (v - T(0.5)) * T(0.2);
    T wave1 = w * (T(2.0) - x * T(0.3) * x * T(0.3));
    T wave2 = wave1 + (wave1 * T(0.7)) * (wave1 * T(0.7));
    
    // Rational-like expressions (20 ops)
    T num1 = wave2 * x + T(1.0);
    T den1 = x * x + x + T(1.0);
    T rat1 = num1 / den1;
    T num2 = (rat1 - T(0.5)) * (rat1 - T(0.5)) * T(3.0);
    T den2 = T(1.0) / (rat1 + T(0.1));
    T rat2 = num2 * den2;
    
    // Final mixing operations (20+ ops)
    T mix1 = rat2 * wave2 + rat1 * rat1;
    T mix2 = (mix1 - T(1.0)) * (mix1 - T(1.0)) + T(1.0) / (mix1 + T(2.0));
    T mix3 = -mix2 * T(0.5) + (mix1 * T(0.3)) * (mix1 * T(0.3));
    T mix4 = (mix3 + mix2) / ((mix1 * mix1) + T(0.5));
    T mix5 = T(1.0) / ((mix4 * mix4) + T(0.01));
    
    // Final computation stages (20+ ops)
    T final1 = mix5 + mix4 * T(0.1);
    T final2 = (final1 * final1) - T(0.2) * final1;
    T final3 = final2 / (final2 * final2 + T(1.0));
    T final4 = final3 * T(10.0) + T(1.0);
    T final5 = final4 / (final4 + T(1.0));
    
    return final5;
}

// Ultra-massive iterative function - template version with parameterized iterations
template<typename T, int ITERATIONS>
T ultraMassiveIterative(T x) {
    // Simulate solving a PDE or iterative numerical method
    // Each iteration does ~100 operations
    
    T u = x;  // Initial condition
    T dt = T(0.001); // Time step
    
    // Simulate N time steps of a diffusion-like equation
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        // Compute spatial derivatives (fake 1D discretization)
        T u_left = u * T(0.98);  // Fake left neighbor
        T u_right = u * T(1.02); // Fake right neighbor
        
        // Second derivative approximation (diffusion term)
        T u_xx = (u_left - T(2.0) * u + u_right) / (T(0.1) * T(0.1));
        
        // Add some nonlinear reaction terms
        T reaction = u * (T(1.0) - u) * (u - T(0.5));  // Bistable reaction
        reaction = reaction * T(10.0);
        
        // Add forcing/source terms with various operations
        T forcing = (u - T(0.5)) * (u - T(0.5)) * T(0.1);
        forcing = forcing + (u * T(0.5)) * (u * T(0.5)) * T(0.05);
        forcing = forcing - T(1.0) / ((u * u) + T(1.0)) * T(0.02);
        
        // Some additional complex terms to increase operation count
        T modifier = (u_xx * u_xx) * T(0.001);
        modifier = modifier + T(1.0) / ((reaction * reaction) + T(0.1)) * T(0.01);
        modifier = modifier * (T(1.0) + forcing * forcing);
        
        // Update step with stabilization
        T delta = dt * (u_xx * T(0.1) + reaction * T(0.01) + forcing - modifier);
        u = u + delta;
    }
    
    // Final post-processing with many operations
    for (int i = 0; i < 50; ++i) {
        T post = u + T(0.01) * T(i);
        post = (post * post) - (post - T(0.5)) * (post - T(0.5));
        post = post * T(0.98) + T(0.01);
        u = post * T(0.1) + u * T(0.9);  // Blend with previous
    }
    
    return u;
}

// Convenience functions with different iteration counts for testing different scales

// Small scale - 1 iteration for testing basic functionality
template<typename T>
T massiveExpression1(T x) {
    return massiveExpression<T>(x);
}

// Medium scale - 10 iterations
template<typename T>
T ultraMassiveIterative10(T x) {
    return ultraMassiveIterative<T, 10>(x);
}

// Large scale - 100 iterations  
template<typename T>
T ultraMassiveIterative100(T x) {
    return ultraMassiveIterative<T, 100>(x);
}

// Ultra scale - 1000 iterations (matches original)
template<typename T>
T ultraMassiveIterative1000(T x) {
    return ultraMassiveIterative<T, 1000>(x);
}

// Test input sets for massive functions (smaller ranges to avoid numerical instability)
inline std::vector<double> getMassiveExpressionInputs() {
    return {0.1, 0.5, 1.0, 1.5, 2.0};  // Small positive values to avoid numerical issues
}

inline std::vector<double> getUltraMassiveInputs() {
    return {0.3, 0.5, 0.7};  // Limited inputs due to computational complexity
}

} // namespace one_to_one
} // namespace test_functions
} // namespace tools
} // namespace forge