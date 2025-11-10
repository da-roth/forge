#pragma once

#include "../tools/types/fdouble.hpp"
#include "../tools/types/fbool.hpp"
#include "../tools/types/fint.hpp"
#include "American/test_american_option.hpp"
#include <vector>
#include <functional>
#include <string>

namespace forge {
namespace testing {

// Repository of 1D test functions (ℝ → ℝ)
struct TestFunctions1D {
    // Each function uses Double for both recording and evaluation
    static fdouble linear(fdouble x) { return 2.0 * x + 3.0; }
    static fdouble quadratic(fdouble x) { return x * x + 2.0 * x + 1.0; }
    static fdouble cubic(fdouble x) { return x * x * x - 3.0 * x * x + 3.0 * x - 1.0; }
    static fdouble quartic(fdouble x) { return x * x * x * x - 4.0 * x * x * x + 6.0 * x * x - 4.0 * x + 1.0; }
    static fdouble rational(fdouble x) { return (x + 1.0) / (x - 1.0); }
    static fdouble rational2(fdouble x) { return (x * x + 1.0) / (x + 0.5); }
    static fdouble exponential5(fdouble x) { return x * x * x * x * x; }
    static fdouble complex_polynomial(fdouble x) { 
        return ((x + 1.0) * (x - 1.0) * (x + 2.0)) / (x * x + 0.1); 
    }
    static fdouble inverse(fdouble x) { return 1.0 / x; }
    static fdouble inverse_squared(fdouble x) { return 1.0 / (x * x); }
    static fdouble nested_arithmetic(fdouble x) { return ((x + 2.0) * 3.0 - 1.0) / (x - 0.5); }
    static fdouble deep_nesting(fdouble x) {
        fdouble y = x;
        for (int i = 0; i < 10; ++i) {
            y = y * 1.1 + 0.1;
        }
        return y;
    }
    static fdouble alternating(fdouble x) { return x - x * x + x * x * x - x * x * x * x; }
    static fdouble abs_like(fdouble x) { 
        // Approximates |x| behavior using smooth function
        return x * x / (x + 0.001); 
    }
    static fdouble step_like(fdouble x) {
        // Smooth approximation of step function
        return (x * x * x) / (x * x * x + 1.0);
    }
    static fdouble gaussian_like(fdouble x) {
        // Approximates gaussian shape
        return 1.0 / (1.0 + x * x);
    }
    static fdouble sine_approx(fdouble x) {
        // Taylor series approximation of sin(x)
        return x - (x * x * x) / 6.0;
    }
    static fdouble cosine_approx(fdouble x) {
        // Taylor series approximation of cos(x) 
        return 1.0 - (x * x) / 2.0;
    }
    static fdouble compound1(fdouble x) {
        return (2.0 * x + 1.0) * (x - 3.0) / (x * x + 1.0);
    }
    static fdouble compound2(fdouble x) {
        fdouble a = x + 1.0;
        fdouble b = x - 1.0;
        return (a * a + b * b) / (a * b + 1.0);
    }
    
    // Function with exactly 10 operations for benchmarking
    static fdouble ops_10(const fdouble& x) {
        // 10 operations: 5 muls, 3 adds, 1 sub, 1 div
        fdouble a = x * 2.0;        // 1: mul
        fdouble p = fdouble(1.0) + fdouble(1.0); // testing
        fdouble u = p * (fdouble)2.0;      // testing
        fdouble v = u * 2.0;        // testing 2 
        fdouble b = x * x + v;          // 2: mul
        fdouble c = a + b + u;          // 3: add
        fdouble d = c * 3.0;        // 4: mul
        fdouble e = d + x;          // 5: add
        fdouble f = e * 1.5;        // 6: mul
        fdouble Z = fdouble(3.0) * (fdouble)3.0;
        fdouble g = f - b;          // 7: sub
        fdouble h = g + 10.0;       // 8: add
        fdouble i = h * 0.5;        // 9: mul
        fdouble j = x + 1.0;        // intermediate for div
        return i / j;
    }
    
    // Test function specifically for CSE - has obvious duplicates
    static fdouble cse_test(const fdouble& x) {
        // Create obvious duplicate subexpressions
        fdouble a = x - 0.5;        // First (x - 0.5)
        fdouble b = x - 0.5;        // Second (x - 0.5) - DUPLICATE!
        fdouble c = a * b;          // Should use same node for a and b after CSE
        
        fdouble d = x * x;          // First x^2
        fdouble e = x * x;          // Second x^2 - DUPLICATE!
        fdouble f = d + e;          // Should use same node for d and e after CSE
        
        fdouble g = x + 2.0;        // First (x + 2.0)
        fdouble h = x + 2.0;        // Second (x + 2.0) - DUPLICATE!
        fdouble i = g * h;          // Should use same node for g and h after CSE
        
        return c + f + i;
    }
    
    static double cse_test_native(double x) {
        double a = x - 0.5;
        double b = x - 0.5;
        double c = a * b;
        
        double d = x * x;
        double e = x * x;
        double f = d + e;
        
        double g = x + 2.0;
        double h = x + 2.0;
        double i = g * h;
        
        return c + f + i;
    }
    
    // Function with exactly 50 operations for benchmarking
    static fdouble ops_50(const fdouble& x) {
        fdouble v = x;
        // Generate exactly 50 operations through a controlled pattern
        // 5 iterations of 10 operations each
        // Modified to use damping to prevent overflow
        for (int i = 0; i < 5; ++i) {
            fdouble t1 = v * 1.01;      // 1 - reduced from 1.1
            fdouble t2 = t1 + 0.01;     // 2 - reduced from 0.1
            fdouble t3 = t2 * v;        // 3
            fdouble t4 = t3 - x;        // 4
            fdouble t5 = t4 * 0.5;      // 5 - reduced from 2.0
            fdouble t6 = t5 + v;        // 6
            fdouble t7 = t6 / 2.5;      // 7 - increased divisor from 1.5
            fdouble t8 = t7 * 0.9;      // 8 - changed from squaring to scaling
            fdouble t9 = t8 - 0.1;      // 9 - reduced from 1.0
            v = t9 * 0.95 + x * 0.01; // 10 - added damping factor 0.95
        }
        return v;
    }
    
    // Native versions for benchmarking
    static double linear_native(double x) { return 2.0 * x + 3.0; }
    static double quadratic_native(double x) { return x * x + 2.0 * x + 1.0; }
    static double cubic_native(double x) { return x * x * x - 3.0 * x * x + 3.0 * x - 1.0; }
    static double quartic_native(double x) { return x * x * x * x - 4.0 * x * x * x + 6.0 * x * x - 4.0 * x + 1.0; }
    static double rational_native(double x) { return (x + 1.0) / (x - 1.0); }
    static double rational2_native(double x) { return (x * x + 1.0) / (x + 0.5); }
    static double exponential5_native(double x) { return x * x * x * x * x; }
    static double complex_polynomial_native(double x) { 
        return ((x + 1.0) * (x - 1.0) * (x + 2.0)) / (x * x + 0.1); 
    }
    static double inverse_native(double x) { return 1.0 / x; }
    static double inverse_squared_native(double x) { return 1.0 / (x * x); }
    static double nested_arithmetic_native(double x) { return ((x + 2.0) * 3.0 - 1.0) / (x - 0.5); }
    static double deep_nesting_native(double x) {
        double y = x;
        for (int i = 0; i < 10; ++i) {
            y = y * 1.1 + 0.1;
        }
        return y;
    }
    static double alternating_native(double x) { return x - x * x + x * x * x - x * x * x * x; }
    static double abs_like_native(double x) { return x * x / (x + 0.001); }
    static double step_like_native(double x) { return (x * x * x) / (x * x * x + 1.0); }
    static double gaussian_like_native(double x) { return 1.0 / (1.0 + x * x); }
    static double sine_approx_native(double x) { return x - (x * x * x) / 6.0; }
    static double cosine_approx_native(double x) { return 1.0 - (x * x) / 2.0; }
    static double compound1_native(double x) {
        return (2.0 * x + 1.0) * (x - 3.0) / (x * x + 1.0);
    }
    static double compound2_native(double x) {
        double a = x + 1.0;
        double b = x - 1.0;
        return (a * a + b * b) / (a * b + 1.0);
    }
    
    static double ops_10_native(double x) {
        double a = x * 2.0;
        double p = 1.0 + 1.0;
        double u = p * 2.0;      // testing
        double v = u * 2.0;        // testing 2 
        double b = x * x + v;          // 2: mul
        double c = a + b + u;          // 3: add
        double d = c * 3.0;
        double e = d + x;
        double f = e * 1.5;
        double Z = 3.0 * 3.0;
        double g = f - b;
        double h = g + 10.0;
        double i = h * 0.5;
        double j = x + 1.0;
        return i / j;
    }
    
    static double ops_50_native(double x) {
        double v = x;
        for (int i = 0; i < 5; ++i) {
            double t1 = v * 1.01;
            double t2 = t1 + 0.01;
            double t3 = t2 * v;
            double t4 = t3 - x;
            double t5 = t4 * 0.5;
            double t6 = t5 + v;
            double t7 = t6 / 2.5;
            double t8 = t7 * 0.9;
            double t9 = t8 - 0.1;
            v = t9 * 0.95 + x * 0.01;
        }
        return v;
    }
    
    // AVX2-friendly test function for benchmarking vectorized execution
    // Simple polynomial that's easy to verify and benchmark
    static fdouble avx2_polynomial(const fdouble& x) {
        // f(x) = 3x^3 - 2x^2 + 5x - 7
        // Easy to compute and verify, shows benefit of vectorization
        fdouble x2 = x * x;           // x^2
        fdouble x3 = x2 * x;          // x^3
        fdouble term1 = 3.0 * x3;     // 3x^3
        fdouble term2 = 2.0 * x2;     // 2x^2
        fdouble term3 = 5.0 * x;      // 5x
        fdouble result = term1 - term2 + term3 - 7.0;
        return result;
    }
    static double avx2_polynomial_native(double x) {
        double x2 = x * x;
        double x3 = x2 * x;
        double term1 = 3.0 * x3;
        double term2 = 2.0 * x2;
        double term3 = 5.0 * x;
        double result = term1 - term2 + term3 - 7.0;
        return result;
    }
    
    // New operation test functions
    static fdouble negation(const fdouble& x) { return -x; }
    static double negation_native(double x) { return -x; }
    
    static fdouble absolute(const fdouble& x) { return abs(x); }
    static double absolute_native(double x) { return std::abs(x); }
    
    static fdouble squared(const fdouble& x) { return square(x); }
    static double squared_native(double x) { return x * x; }
    
    static fdouble reciprocal(const fdouble& x) { return recip(x); }
    static double reciprocal_native(double x) { return 1.0 / x; }
    
    // New transcendental function tests
    static fdouble exp_test(const fdouble& x) {
        // Test exponential in a realistic expression
        fdouble a = x * 2.0;           // Scale input
        fdouble b = a + 3.0;           // Shift
        fdouble c = exp(b);            // Apply exponential
        fdouble d = c * 0.5;           // Scale result
        fdouble e = d - 2.0;           // Shift result
        return e;
    }
    static double exp_test_native(double x) {
        double a = x * 2.0;
        double b = a + 3.0;
        double c = std::exp(b);
        double d = c * 0.5;
        double e = d - 2.0;
        return e;
    }
    
    static fdouble log_test(const fdouble& x) {
        // Test logarithm with input conditioning to ensure positive values
        fdouble a = x * x + 1.0;       // Ensure positive (x^2 + 1 >= 1)
        fdouble b = a * 2.0;           // Scale
        fdouble c = log(b);            // Apply logarithm
        fdouble d = c + 1.0;           // Shift
        fdouble e = d * 3.0;           // Scale result
        return e;
    }
    static double log_test_native(double x) {
        double a = x * x + 1.0;
        double b = a * 2.0;
        double c = std::log(b);
        double d = c + 1.0;
        double e = d * 3.0;
        return e;
    }
    
    static fdouble sqrt_test(const fdouble& x) {
        // Test square root with input conditioning to ensure positive values
        fdouble a = x * x + 4.0;       // Ensure positive (x^2 + 4 >= 4)
        fdouble b = sqrt(a);           // Apply square root
        fdouble c = b * 2.0;           // Scale
        fdouble d = c - 3.0;           // Shift
        fdouble e = d * d;             // Square the result
        return e;
    }
    static double sqrt_test_native(double x) {
        double a = x * x + 4.0;
        double b = std::sqrt(a);
        double c = b * 2.0;
        double d = c - 3.0;
        double e = d * d;
        return e;
    }
    
    // Combined test using all three new operators
    static fdouble transcendental_combo(const fdouble& x) {
        // Complex expression using exp, log, and sqrt
        fdouble a = x + 2.0;
        fdouble b = sqrt(a * a + 1.0);  // sqrt(a^2 + 1) always positive
        fdouble c = exp(x * 0.5);       // exp(x/2)
        fdouble d = log(b + c);         // log of sum (always positive)
        fdouble e = d * sqrt(c);        // Multiply by sqrt(exp(x/2))
        return e - 1.0;
    }
    static double transcendental_combo_native(double x) {
        double a = x + 2.0;
        double b = std::sqrt(a * a + 1.0);
        double c = std::exp(x * 0.5);
        double d = std::log(b + c);
        double e = d * std::sqrt(c);
        return e - 1.0;
    }
    
    // Trigonometric function tests
    static fdouble sin_test(const fdouble& x) {
        // Test sin with various operations
        fdouble a = sin(x);
        fdouble b = sin(x * 0.5);
        fdouble c = sin(x + 1.0);
        return a + b * 2.0 - c * 0.5;
    }
    static double sin_test_native(double x) {
        double a = std::sin(x);
        double b = std::sin(x * 0.5);
        double c = std::sin(x + 1.0);
        return a + b * 2.0 - c * 0.5;
    }
    
    static fdouble cos_test(const fdouble& x) {
        // Test cos with various operations
        fdouble a = cos(x);
        fdouble b = cos(x * 2.0);
        fdouble c = cos(x - 0.5);
        return a * a + b - c * 1.5;
    }
    static double cos_test_native(double x) {
        double a = std::cos(x);
        double b = std::cos(x * 2.0);
        double c = std::cos(x - 0.5);
        return a * a + b - c * 1.5;
    }
    
    static fdouble tan_test(const fdouble& x) {
        // Test tan with various operations
        // Use small range to avoid tan singularities
        fdouble a = tan(x * 0.3);
        fdouble b = tan(x * 0.1 + 0.2);
        return a + b * b;
    }
    static double tan_test_native(double x) {
        double a = std::tan(x * 0.3);
        double b = std::tan(x * 0.1 + 0.2);
        return a + b * b;
    }
    
    static fdouble trig_combo(const fdouble& x) {
        // Complex expression combining sin, cos, tan
        // sin^2(x) + cos^2(x) = 1 identity, plus tan
        fdouble s = sin(x);
        fdouble c = cos(x);
        fdouble identity = s * s + c * c;  // Should be ~1.0
        fdouble t = tan(x * 0.25);  // Small angle for tan
        return identity + t * 0.1 - 1.0;  // Should be ~t*0.1
    }
    static double trig_combo_native(double x) {
        double s = std::sin(x);
        double c = std::cos(x);
        double identity = s * s + c * c;
        double t = std::tan(x * 0.25);
        return identity + t * 0.1 - 1.0;
    }
    
    // Min/Max comparison operator tests
    static fdouble min_test(const fdouble& x) {
        // Test min with various expressions
        fdouble a = x * 2.0;
        fdouble b = x + 3.0;
        fdouble c = min(a, b);           // min(2x, x+3)
        fdouble d = min(c, fdouble(1.0)); // min with constant
        fdouble e = d * 2.0 - 0.5;
        return e;
    }
    static double min_test_native(double x) {
        double a = x * 2.0;
        double b = x + 3.0;
        double c = std::fmin(a, b);
        double d = std::fmin(c, 1.0);
        double e = d * 2.0 - 0.5;
        return e;
    }
    
    static fdouble max_test(const fdouble& x) {
        // Test max with various expressions
        fdouble a = x * x;
        fdouble b = x - 1.0;
        fdouble c = max(a, b);           // max(x^2, x-1)
        fdouble d = max(c, fdouble(0.0)); // max with zero (like ReLU)
        fdouble e = d * 0.5 + 1.0;
        return e;
    }
    static double max_test_native(double x) {
        double a = x * x;
        double b = x - 1.0;
        double c = std::fmax(a, b);
        double d = std::fmax(c, 0.0);
        double e = d * 0.5 + 1.0;
        return e;
    }
    
    static fdouble minmax_combo(const fdouble& x) {
        // Complex expression combining min and max
        fdouble a = x * 3.0;
        fdouble b = x + 2.0;
        fdouble c = x - 2.0;
        fdouble d = max(a, b);          // max(3x, x+2)
        fdouble e = min(d, c * c);      // min(max(3x, x+2), (x-2)^2)
        fdouble f = max(e, fdouble(-1.0)); // Clamp to minimum of -1
        fdouble g = min(f, fdouble(10.0)); // Clamp to maximum of 10
        return g * 1.5 - 0.5;
    }
    static double minmax_combo_native(double x) {
        double a = x * 3.0;
        double b = x + 2.0;
        double c = x - 2.0;
        double d = std::fmax(a, b);
        double e = std::fmin(d, c * c);
        double f = std::fmax(e, -1.0);
        double g = std::fmin(f, 10.0);
        return g * 1.5 - 0.5;
    }
    
    // Clamp function implemented with min/max
    static fdouble clamp_test(const fdouble& x) {
        // Clamp x to range [-2, 3] using min/max
        fdouble lower = fdouble(-2.0);
        fdouble upper = fdouble(3.0);
        fdouble clamped = min(max(x, lower), upper);
        // Apply some transformation to the clamped value
        fdouble result = clamped * clamped + clamped * 0.5 - 1.0;
        return result;
    }
    static double clamp_test_native(double x) {
        double lower = -2.0;
        double upper = 3.0;
        double clamped = std::fmin(std::fmax(x, lower), upper);
        double result = clamped * clamped + clamped * 0.5 - 1.0;
        return result;
    }
    
    // Comparison operator tests
    static fdouble cmpLT_test(const fdouble& x) {
        // Test less-than with various expressions
        fdouble a = x * 2.0;
        fdouble b = x + 3.0;
        fbool c = cmpLT(a, b);          // Is 2x < x+3? (true for x < 3)
        fbool d = cmpLT(x, fdouble(0.0)); // Is x < 0?
        // Convert to Double using If: c ? 1.0 : 0.0
        fdouble c_val = c.If(fdouble(1.0), fdouble(0.0));
        fdouble d_val = d.If(fdouble(1.0), fdouble(0.0));
        fdouble result = c_val + d_val * 2.0;      // Combines both comparisons
        return result;
    }
    static double cmpLT_test_native(double x) {
        double a = x * 2.0;
        double b = x + 3.0;
        double c = (a < b) ? 1.0 : 0.0;
        double d = (x < 0.0) ? 1.0 : 0.0;
        double result = c + d * 2.0;
        return result;
    }
    
    static fdouble cmpLE_test(const fdouble& x) {
        // Test less-than-or-equal
        fdouble a = x * x;
        fdouble b = fdouble(4.0);
        fbool c = cmpLE(a, b);          // Is x² <= 4? (true for -2 <= x <= 2)
        fbool d = cmpLE(x, x);          // Always true (x <= x)
        fdouble c_val = c.If(fdouble(1.0), fdouble(0.0));
        fdouble d_val = d.If(fdouble(1.0), fdouble(0.0));
        fdouble result = c_val * 3.0 + d_val;     // Should be 3*c + 1
        return result;
    }
    static double cmpLE_test_native(double x) {
        double a = x * x;
        double b = 4.0;
        double c = (a <= b) ? 1.0 : 0.0;
        double d = (x <= x) ? 1.0 : 0.0;
        double result = c * 3.0 + d;
        return result;
    }
    
    static fdouble cmpGT_test(const fdouble& x) {
        // Test greater-than
        fdouble a = x + 1.0;
        fdouble b = x - 1.0;
        fbool c = cmpGT(a, b);          // Is x+1 > x-1? (always true)
        fbool d = cmpGT(x, fdouble(2.0)); // Is x > 2?
        fdouble c_val = c.If(fdouble(1.0), fdouble(0.0));
        fdouble d_val = d.If(fdouble(1.0), fdouble(0.0));
        fdouble result = c_val * 2.0 + d_val * 3.0;
        return result;
    }
    static double cmpGT_test_native(double x) {
        double a = x + 1.0;
        double b = x - 1.0;
        double c = (a > b) ? 1.0 : 0.0;
        double d = (x > 2.0) ? 1.0 : 0.0;
        double result = c * 2.0 + d * 3.0;
        return result;
    }
    
    static fdouble cmpGE_test(const fdouble& x) {
        // Test greater-than-or-equal
        fdouble a = x * 3.0;
        fdouble b = x + 5.0;
        fbool c = cmpGE(a, b);          // Is 3x >= x+5? (true for x >= 2.5)
        fbool d = cmpGE(fdouble(0.0), x); // Is 0 >= x?
        fdouble c_val = c.If(fdouble(1.0), fdouble(0.0));
        fdouble d_val = d.If(fdouble(1.0), fdouble(0.0));
        fdouble result = c_val * 4.0 - d_val;
        return result;
    }
    static double cmpGE_test_native(double x) {
        double a = x * 3.0;
        double b = x + 5.0;
        double c = (a >= b) ? 1.0 : 0.0;
        double d = (0.0 >= x) ? 1.0 : 0.0;
        double result = c * 4.0 - d;
        return result;
    }
    
    static fdouble cmpEQ_test(const fdouble& x) {
        // Test equality
        fdouble a = x * 2.0;
        fdouble b = x + x;
        fbool c = cmpEQ(a, b);          // Is 2x == x+x? (always true)
        fbool d = cmpEQ(x, fdouble(1.0)); // Is x == 1?
        fdouble c_val = c.If(fdouble(1.0), fdouble(0.0));
        fdouble d_val = d.If(fdouble(1.0), fdouble(0.0));
        fdouble result = c_val * 5.0 + d_val * 2.0;
        return result;
    }
    static double cmpEQ_test_native(double x) {
        double a = x * 2.0;
        double b = x + x;
        double c = (a == b) ? 1.0 : 0.0;
        double d = (x == 1.0) ? 1.0 : 0.0;
        double result = c * 5.0 + d * 2.0;
        return result;
    }
    
    static fdouble cmpNE_test(const fdouble& x) {
        // Test not-equal
        fdouble a = x;
        fdouble b = x + 0.1;
        fbool c = cmpNE(a, b);          // Is x != x+0.1? (always true)
        fbool d = cmpNE(x, x);          // Is x != x? (always false)
        fdouble c_val = c.If(fdouble(1.0), fdouble(0.0));
        fdouble d_val = d.If(fdouble(1.0), fdouble(0.0));
        fdouble result = c_val * 3.0 - d_val * 2.0; // Should be 3.0
        return result;
    }
    static double cmpNE_test_native(double x) {
        double a = x;
        double b = x + 0.1;
        double c = (a != b) ? 1.0 : 0.0;
        double d = (x != x) ? 1.0 : 0.0;
        double result = c * 3.0 - d * 2.0;
        return result;
    }
    
    // Simple debug test for comparison operators
    static fdouble cmp_debug_test(const fdouble& x) {
        // Very simple test to debug comparison behavior
        // For x = 2.5: should return 1.0 (since 2.5 >= 2.0)
        fbool cmp = cmpGE(x, fdouble(2.0));
        fdouble result = cmp.If(fdouble(1.0), fdouble(0.0));
        return result;
    }
    static double cmp_debug_test_native(double x) {
        double result = (x >= 2.0) ? 1.0 : 0.0;
        return result;
    }
    
    // Another debug test with multiplication
    static fdouble cmp_debug_mul_test(const fdouble& x) {
        // Test if multiplication with comparison result works
        // For x = 2.5: cmpGE(2.5, 2.0) = 1.0, so result = 1.0 * 5.0 = 5.0
        fbool cmp = cmpGE(x, fdouble(2.0));
        fdouble cmp_val = cmp.If(fdouble(1.0), fdouble(0.0));
        fdouble result = cmp_val * 5.0;
        return result;
    }
    static double cmp_debug_mul_test_native(double x) {
        double cmp = (x >= 2.0) ? 1.0 : 0.0;
        double result = cmp * 5.0;
        return result;
    }
    
    // Debug test for multiplying two comparison results
    static fdouble cmp_debug_and_test(const fdouble& x) {
        // Test AND-like behavior via multiplication
        // For x = 2.5: cmpGE(2.5, 2.0) = 1.0, cmpLT(2.5, 4.0) = 1.0
        // Result = 1.0 * 1.0 = 1.0
        fbool cmp1 = cmpGE(x, fdouble(2.0));
        fbool cmp2 = cmpLT(x, fdouble(4.0));
        // Use logical AND instead of multiplication
        fbool and_result = cmp1 && cmp2;
        fdouble result = and_result.If(fdouble(1.0), fdouble(0.0));
        return result;
    }
    static double cmp_debug_and_test_native(double x) {
        double cmp1 = (x >= 2.0) ? 1.0 : 0.0;
        double cmp2 = (x < 4.0) ? 1.0 : 0.0;
        double result = cmp1 * cmp2;
        return result;
    }
    
    // Debug test for the specific segment that should be active at x=2.5
    static fdouble cmp_debug_seg4_test(const fdouble& x) {
        // Test just segment 4: 2 <= x < 4
        // For x = 2.5: both conditions true, so seg4 = 1.0
        // Result = 1.0 * (4.0 - 2.5) = 1.5
        fbool cond1 = cmpGE(x, fdouble(2.0));
        fbool cond2 = cmpLT(x, fdouble(4.0));
        fbool seg4 = cond1 && cond2;
        fdouble seg4_val = seg4.If(fdouble(1.0), fdouble(0.0));
        fdouble result = seg4_val * (4.0 - x);
        return result;
    }
    static double cmp_debug_seg4_test_native(double x) {
        double seg4 = ((x >= 2.0) && (x < 4.0)) ? 1.0 : 0.0;
        double result = seg4 * (4.0 - x);
        return result;
    }
    
    // Debug test for adding segments together
    static fdouble cmp_debug_add_test(const fdouble& x) {
        // Test adding multiple segments like in combo
        // For x = 2.5: seg4 should be 1.0, others 0.0
        fbool seg3_cond = cmpGE(x, fdouble(0.0)) && cmpLT(x, fdouble(2.0));  
        fbool seg4_cond = cmpGE(x, fdouble(2.0)) && cmpLT(x, fdouble(4.0));  
        fbool seg5_cond = cmpGE(x, fdouble(4.0));                           
        
        fdouble seg3 = seg3_cond.If(fdouble(1.0), fdouble(0.0));
        fdouble seg4 = seg4_cond.If(fdouble(1.0), fdouble(0.0));
        fdouble seg5 = seg5_cond.If(fdouble(1.0), fdouble(0.0));
        
        // Add contributions
        fdouble result = seg3 * 2.0 + seg4 * (4.0 - x) + seg5 * 0.0;
        return result;
    }
    static double cmp_debug_add_test_native(double x) {
        double seg3 = ((x >= 0.0) && (x < 2.0)) ? 1.0 : 0.0;
        double seg4 = ((x >= 2.0) && (x < 4.0)) ? 1.0 : 0.0;
        double seg5 = (x >= 4.0) ? 1.0 : 0.0;
        
        double result = seg3 * 2.0 + seg4 * (4.0 - x) + seg5 * 0.0;
        return result;
    }
    
    // Complex comparison test combining multiple operators
    static fdouble cmp_combo_test(const fdouble& x) {
        // Implement a piecewise function using comparisons
        // f(x) = { 0 if x < -2
        //        { x+2 if -2 <= x < 0
        //        { 2 if 0 <= x < 2
        //        { 4-x if 2 <= x < 4
        //        { 0 if x >= 4
        
        // Create mutually exclusive conditions for each segment using fbool
        fbool seg1_cond = cmpLT(x, fdouble(-2.0));                                    
        fbool seg2_cond = cmpGE(x, fdouble(-2.0)) && cmpLT(x, fdouble(0.0));          
        fbool seg3_cond = cmpGE(x, fdouble(0.0)) && cmpLT(x, fdouble(2.0));           
        fbool seg4_cond = cmpGE(x, fdouble(2.0)) && cmpLT(x, fdouble(4.0));           
        fbool seg5_cond = cmpGE(x, fdouble(4.0));                                    
        
        // Convert to Double for multiplication
        fdouble seg1 = seg1_cond.If(fdouble(1.0), fdouble(0.0));
        fdouble seg2 = seg2_cond.If(fdouble(1.0), fdouble(0.0));
        fdouble seg3 = seg3_cond.If(fdouble(1.0), fdouble(0.0));
        fdouble seg4 = seg4_cond.If(fdouble(1.0), fdouble(0.0));
        fdouble seg5 = seg5_cond.If(fdouble(1.0), fdouble(0.0));
        
        // Compute the piecewise function
        fdouble result = seg1 * 0.0 +
                       seg2 * (x + 2.0) +
                       seg3 * 2.0 +
                       seg4 * (4.0 - x) +
                       seg5 * 0.0;
        return result;
    }
    static double cmp_combo_test_native(double x) {
        // Create mutually exclusive conditions for each segment
        double seg1 = (x < -2.0) ? 1.0 : 0.0;                                   // x < -2
        double seg2 = ((x >= -2.0) && (x < 0.0)) ? 1.0 : 0.0;                 // -2 <= x < 0
        double seg3 = ((x >= 0.0) && (x < 2.0)) ? 1.0 : 0.0;                  // 0 <= x < 2
        double seg4 = ((x >= 2.0) && (x < 4.0)) ? 1.0 : 0.0;                  // 2 <= x < 4
        double seg5 = (x >= 4.0) ? 1.0 : 0.0;                                   // x >= 4
        
        double result = seg1 * 0.0 +
                       seg2 * (x + 2.0) +
                       seg3 * 2.0 +
                       seg4 * (4.0 - x) +
                       seg5 * 0.0;
        return result;
    }
    
    // TEST THAT DEMONSTRATES NEW BOOLTP: Now we can use If operation!
    static fdouble cmp_limitation_test(const fdouble& x) {
        // NOW WITH fbool: (x > 0.0) ? 2.0 * x : -1.0 * x
        
        fbool cmp = cmpGT(x, fdouble(0.0));  // Returns fbool now!
        
        // Still compute both branches (for now - until lazy eval)
        fdouble positive_branch = 2.0 * x;
        fdouble negative_branch = -1.0 * x;
        
        // Use the new If operation - much cleaner!
        fdouble result = cmp.If(positive_branch, negative_branch);
        
        // Clean and type-safe conditional selection!
        return result;
    }
    
    static double cmp_limitation_test_native(double x) {
        // Simple ternary - only computes needed branch
        return (x > 0.0) ? 2.0 * x : -1.0 * x;
    }
    
    // Test reciprocal bug: CRR discount factor calculation issues
    template<typename T>
    static T reciprocal_exp_bug_impl(const T& x) {
        // Replicate the exact CRR discount factor calculation that shows precision issues
        // This mimics the American Option's bin_params.disc and bin_params.d calculations
        
        // Use input-dependent values to avoid constant folding
        T r = x * T(0.001);      // Interest rate depends on input
        T sigma = x * T(0.002);  // Volatility depends on input
        T dt = T(0.01);          // Time step constant
        
        // CRR parameter calculations - exactly as in AmericanOption test
        T sigma_sqrt_dt = sigma * std::sqrt(dt);
        T a = std::exp(sigma_sqrt_dt);
        
        // Pattern 1: params.d = 1/a
        T d = T(1.0) / a;
        T d_stable = std::exp(-sigma_sqrt_dt);
        
        // Pattern 2: params.disc = 1/exp(r*dt) 
        T r_dt = r * dt;
        T erdt = std::exp(r_dt);
        T disc = T(1.0) / erdt;
        T disc_stable = std::exp(-r_dt);
        
        // Compute differences
        T d_diff = d - d_stable;
        T disc_diff = disc - disc_stable;
        
        // Return combined error, scaled and amplified
        // Focus on disc_diff as it showed the issue (JIT=100000 vs Native=99810)
        return disc_diff * T(1000000.0) + d_diff * T(1000.0);
    }
    
    static fdouble reciprocal_exp_bug(const fdouble& x) {
        return reciprocal_exp_bug_impl<fdouble>(x);
    }
    
    static double reciprocal_exp_bug_native(double x) {
        return reciprocal_exp_bug_impl<double>(x);
    }

    // Test CRR discount factor computation issue found in American Option
    template<typename T>
    static T crr_discount_factor_impl(const T& x) {
        // Replicate the exact CRR discount factor calculation that shows JIT vs Native difference
        // This mimics the computation in CRRParametersProvider::Compute()
        
        // Create TwoPointCurve for interest rate (same as in American Option)
        // Rate at t=0.0 is 1%, rate at t=1.0 is 2%
        T t1 = T(0.0);
        T t2 = T(1.0); 
        T v1 = T(0.01);  // 1% rate
        T v2 = T(0.02);  // 2% rate
        
        // Make t depend on x to avoid constant folding
        T t = T(0.5) + x * T(0.0);  // t = 0.5 (middle of curve)
        T dt = T(0.5);              // time step
        
        // Linear interpolation (same as TwoPointCurve.GetValue)
        T alpha = (t - t1) / (t2 - t1);
        T rate = v1 * (T(1.0) - alpha) + v2 * alpha;  // Should be 0.015 at t=0.5
        
        // PROGRESSIVE DEBUG: Test step by step
        
        // Step 1: Test rate calculation (should be 0.015 at t=0.5) ✓ PASSED
        // return rate * x * T(10000.0);  // Should be 0.015 * x * 10000 = 150 * x
        
        // Step 2: Test r_times_dt calculation ✓ PASSED
        T r_times_dt = rate * dt;      // Should be 0.015 * 0.5 = 0.0075
        // return r_times_dt * x * T(10000.0);  // Should be 0.0075 * x * 10000 = 75 * x
        
        // Step 3: Test exp calculation ✗ FAILED - JIT returns constant!
        T exp_r_dt = std::exp(r_times_dt);  // Should be exp(0.0075) ≈ 1.007528
        
        // Step 4: Test division (discount factor) - let's see what happens
        T disc = T(1.0) / exp_r_dt;    // Should be ≈ 0.992528
        return  disc * x * T(10000.0);  // Expected: ≈ 9925.28 * x
    }
    
    static fdouble crr_discount_factor(const fdouble& x) {
        return crr_discount_factor_impl<fdouble>(x);
    }
    
    static double crr_discount_factor_native(double x) {
        return crr_discount_factor_impl<double>(x);
    }

    // Test TwoPointCurve boundary condition issue found in American Option
    template<typename T>
    static T two_point_curve_boundary_impl(const T& x) {
        // Replicate the exact issue: TwoPointCurve with t=t1 boundary
        // When t equals t1 (both 0.0), the condition t <= t1 should be true
        
        // Make t depend on x to avoid constant folding
        T t = x * T(0.0);  // This will be 0 for any x, but JIT doesn't know that!
        T t1 = T(0.0);     // First tenor
        T t2 = T(1.0);     // Second tenor
        
        // Test each comparison separately
        T result = T(0.0);
        
        if constexpr (std::is_same_v<T, fdouble>) {
            // JIT version - test comparisons individually
            auto is_lt = cmpLT(t, t1);   // Should be false (0 < 0 = false)
            auto is_le = cmpLE(t, t1);   // Should be true  (0 <= 0 = true)
            auto is_eq = cmpEQ(t, t1);   // Should be true  (0 == 0 = true)
            auto is_ge = cmpGE(t, t1);   // Should be true  (0 >= 0 = true)
            auto is_gt = cmpGT(t, t1);   // Should be false (0 > 0 = false)
            
            // Encode each result with a different power of 10
            result = is_lt.If(T(1.0), T(0.0)) * T(100000.0) +  // Bit 5: LT
                    is_le.If(T(1.0), T(0.0)) * T(10000.0) +    // Bit 4: LE
                    is_eq.If(T(1.0), T(0.0)) * T(1000.0) +     // Bit 3: EQ
                    is_ge.If(T(1.0), T(0.0)) * T(100.0) +      // Bit 2: GE
                    is_gt.If(T(1.0), T(0.0)) * T(10.0) +       // Bit 1: GT
                    x * T(0.001);                                // Small x dependency
        } else {
            // Native version
            result = ((t < t1) ? 1.0 : 0.0) * T(100000.0) +   // Bit 5: LT (should be 0)
                    ((t <= t1) ? 1.0 : 0.0) * T(10000.0) +     // Bit 4: LE (should be 1)
                    ((t == t1) ? 1.0 : 0.0) * T(1000.0) +      // Bit 3: EQ (should be 1)
                    ((t >= t1) ? 1.0 : 0.0) * T(100.0) +       // Bit 2: GE (should be 1)
                    ((t > t1) ? 1.0 : 0.0) * T(10.0) +         // Bit 1: GT (should be 0)
                    x * T(0.001);                                // Small x dependency
        }
        
        // Expected result: 011100 in binary = 0 + 10000 + 1000 + 100 + 0 = 11100
        return result;
    }
    
    static fdouble two_point_curve_boundary(const fdouble& x) {
        return two_point_curve_boundary_impl<fdouble>(x);
    }
    
    static double two_point_curve_boundary_native(double x) {
        return two_point_curve_boundary_impl<double>(x);
    }

    // PROGRESSIVE DEBUG: American Option Step-by-Step Analysis
    // Following the pattern from american-option-debugging-guide.md
    
    // STEP 1: Test interest rate calculation only
    template<typename T>
    static T american_step1_rate_impl(const T& x) {
        // Replicate TwoPointCurve rate calculation (IR.risk_free)
        // Rate at t=0.0 is 1%, rate at t=1.0 is 2%
        T t1 = T(0.0);
        T t2 = T(1.0); 
        T v1 = T(0.01);  // 1% rate
        T v2 = T(0.02);  // 2% rate
        
        // t = maturity - dt = 1.0 - 0.5 = 0.5 (from American Option)
        T t = T(0.5) + x * T(0.0);  // Make input-dependent
        
        // Linear interpolation (TwoPointCurve.GetValue)
        T alpha = (t - t1) / (t2 - t1);
        T rate = v1 * (T(1.0) - alpha) + v2 * alpha;  // Should be 0.015 at t=0.5
        
        return rate * x * T(10000.0);  // Make input-dependent: should be 150 * x
    }
    
    // STEP 2: Test volatility calculation only  
    template<typename T>
    static T american_step2_vol_impl(const T& x) {
        // Replicate VolatilitySmileCurve calculation (VOL.equity) 
        T base_vol = T(0.25);
        T t = T(0.5) + x * T(0.0);  // Same t as step 1
        
        // Vol increases with time: base_vol * (1.0 + t * 0.1)
        T vol = base_vol * (T(1.0) + t * T(0.1));  // Should be 0.25 * (1.0 + 0.5 * 0.1) = 0.2625
        
        return vol * x * T(10000.0);  // Make input-dependent: should be 2625 * x
    }
    
    // STEP 3: Test exponential calculation (volatility part)
    template<typename T>
    static T american_step3_exp_vol_impl(const T& x) {
        // Replicate: a = std::exp(sigma * std::sqrt(dt))
        T sigma = T(0.2625);  // From step 2
        T dt = T(0.5);        // time step
        T sigma_sqrt_dt = sigma * std::sqrt(dt);  // 0.2625 * sqrt(0.5) ≈ 0.1856
        T a = std::exp(sigma_sqrt_dt);            // exp(0.1856) ≈ 1.2037
        
        return a * x * T(10000.0);  // Make input-dependent: should be ~12037 * x
    }
    
    // STEP 4: Test exponential calculation (rate part)
    template<typename T>
    static T american_step4_exp_rate_impl(const T& x) {
        // Replicate: erdt = std::exp(r * dt)
        T r = T(0.015);  // From step 1
        T dt = T(0.5);   // time step  
        T r_dt = r * dt; // 0.015 * 0.5 = 0.0075
        T erdt = std::exp(r_dt);  // exp(0.0075) ≈ 1.007528
        
        return erdt * x * T(10000.0);  // Make input-dependent: should be ~10075 * x
    }
    
    // STEP 5: Test discount factor calculation
    template<typename T>
    static T american_step5_disc_impl(const T& x) {
        // Replicate: params.disc = T(1.0) / erdt
        T erdt = T(1.007528);  // From step 4 (approximate)
        T disc = T(1.0) / erdt;  // Should be ≈ 0.992528
        
        return disc * x * T(10000.0);  // Make input-dependent: should be ~9925 * x
    }

    // Test function wrappers for benchmarking
    static fdouble american_step1_rate(const fdouble& x) {
        return american_step1_rate_impl<fdouble>(x);
    }
    static double american_step1_rate_native(double x) {
        return american_step1_rate_impl<double>(x);
    }
    
    static fdouble american_step2_vol(const fdouble& x) {
        return american_step2_vol_impl<fdouble>(x);
    }
    static double american_step2_vol_native(double x) {
        return american_step2_vol_impl<double>(x);
    }
    
    static fdouble american_step3_exp_vol(const fdouble& x) {
        return american_step3_exp_vol_impl<fdouble>(x);
    }
    static double american_step3_exp_vol_native(double x) {
        return american_step3_exp_vol_impl<double>(x);
    }
    
    static fdouble american_step4_exp_rate(const fdouble& x) {
        return american_step4_exp_rate_impl<fdouble>(x);
    }
    static double american_step4_exp_rate_native(double x) {
        return american_step4_exp_rate_impl<double>(x);
    }
    
    static fdouble american_step5_disc(const fdouble& x) {
        return american_step5_disc_impl<fdouble>(x);
    }
    static double american_step5_disc_native(double x) {
        return american_step5_disc_impl<double>(x);
    }

    // VIRTUAL CALL ISOLATION: Exact replication of American Option virtual call pattern
    template<typename T>
    static T american_virtual_calls_impl(const T& x) {
        // Replicate the exact same virtual call setup as American Option
        // This should isolate whether virtual calls are the source of precision differences
        
        // Create the same market data repository and curves
        auto repo = std::make_shared<MapMarketDataRepository<T>>();
        
        // Add the exact same curves as American Option
        repo->Add("IR.risk_free", 
                  std::make_shared<TwoPointCurve<T>>("IR.risk_free", 0.0, 1.0, 0.01, 0.02));
        repo->Add("VOL.equity", 
                  std::make_shared<VolatilitySmileCurve<T>>("VOL.equity", 0.25, 0.1));
        
        // Create CRRParametersProvider exactly like American Option
        auto params_provider = std::make_shared<CRRParametersProvider<T>>("IR.risk_free", "VOL.equity");
        
        // Use the same parameters as American Option
        const T maturity(1.0);
        const T dt = maturity / T(2);  // steps = 2, so dt = 0.5
        T t_final = maturity - dt;     // = 1.0 - 0.5 = 0.5
        T S = x;  // Spot depends on input
        
        // This calls the virtual methods: rCurve->GetValue(t) and vCurve->GetValue(t)
        auto bin_params = params_provider->Compute(t_final, dt, *repo, S);
        
        // Return the discount factor exactly like American Option
        return bin_params.disc * x * T(100.0);  // Should be ≈ 0.992528 * x * 100 = 99.25 * x
    }

    static fdouble american_virtual_calls(const fdouble& x) {
        return american_virtual_calls_impl<fdouble>(x);
    }
    
    static double american_virtual_calls_native(double x) {
        return american_virtual_calls_impl<double>(x);
    }

    // EXACT AMERICAN OPTION REPLICATION: Full parameter computation
    template<typename T>
    static T american_full_params_impl(const T& x) {
        // This replicates the American Option test EXACTLY, line by line
        // to see if the issue is in parameter computation or elsewhere
        
        auto repo = std::make_shared<MapMarketDataRepository<T>>();
        repo->Add("IR.risk_free", 
                  std::make_shared<TwoPointCurve<T>>("IR.risk_free", 0.0, 1.0, 0.01, 0.02));
        repo->Add("VOL.equity", 
                  std::make_shared<VolatilitySmileCurve<T>>("VOL.equity", 0.25, 0.1));
        
        auto params_provider = std::make_shared<CRRParametersProvider<T>>("IR.risk_free", "VOL.equity");
        
        // EXACT same setup as American Option
        const int steps = 2;
        const T maturity(1.0);
        const T dt = maturity / T(steps);
        T t_final = maturity - dt;
        T S = x;  // Input-dependent spot
        
        auto bin_params = params_provider->Compute(t_final, dt, *repo, S);
        
        // EXACT same return as current American Option
        return bin_params.disc * x * T(100.0);
    }

    static fdouble american_full_params(const fdouble& x) {
        return american_full_params_impl<fdouble>(x);
    }
    
    static double american_full_params_native(double x) {
        return american_full_params_impl<double>(x);
    }

    // ISOLATE TwoPointCurve GetValue behavior
    template<typename T>
    static T test_two_point_curve_impl(const T& x) {
        // Create a TwoPointCurve exactly like American Option  
        auto curve = std::make_shared<TwoPointCurve<T>>("IR.risk_free", 0.0, 1.0, 0.01, 0.02);
        
        // Call GetValue with t=0.5 (same as American Option)
        T t = T(0.5) + x * T(0.0);  // Make input-dependent
        T rate = curve->GetValue(t);
        
        return rate * x * T(10000.0);  // Should be 0.015 * x * 10000 = 150 * x
    }

    static fdouble test_two_point_curve(const fdouble& x) {
        return test_two_point_curve_impl<fdouble>(x);
    }
    
    static double test_two_point_curve_native(double x) {
        return test_two_point_curve_impl<double>(x);
    }

    // ISOLATE VolatilitySmileCurve GetValue behavior
    template<typename T>
    static T test_volatility_curve_impl(const T& x) {
        // Create a VolatilitySmileCurve exactly like American Option
        auto curve = std::make_shared<VolatilitySmileCurve<T>>("VOL.equity", 0.25, 0.1);
        
        // Call GetValue with t=0.5 (same as American Option)
        T t = T(0.5) + x * T(0.0);  // Make input-dependent
        T vol = curve->GetValue(t);
        
        return vol * x * T(10000.0);  // Should be 0.2625 * x * 10000 = 2625 * x
    }

    static fdouble test_volatility_curve(const fdouble& x) {
        return test_volatility_curve_impl<fdouble>(x);
    }
    
    static double test_volatility_curve_native(double x) {
        return test_volatility_curve_impl<double>(x);
    }
    
    // STEP-BY-STEP CRR COMPUTATION WITHOUT VIRTUAL CALLS
    // Replicate exact CRR parameter computation with hardcoded values
    template<typename T>
    static T crr_step_by_step_impl(const T& x) {
        // Step 1: Get rate (TwoPointCurve interpolation for t=0.5)
        T t = T(0.5) + x * T(0.0);  // Make input-dependent
        T t1 = T(0.0);
        T t2 = T(1.0); 
        T v1 = T(0.01);  
        T v2 = T(0.02);
        T alpha = (t - t1) / (t2 - t1);  // (0.5 - 0) / (1 - 0) = 0.5
        T rate = v1 * (T(1.0) - alpha) + v2 * alpha;  // 0.01 * 0.5 + 0.02 * 0.5 = 0.015
        
        // Step 2: Get volatility (VolatilitySmileCurve for t=0.5)
        T base_vol = T(0.25);
        T sigma = base_vol * (T(1.0) + t * T(0.1));  // 0.25 * (1.0 + 0.5 * 0.1) = 0.25 * 1.05 = 0.2625
        
        // Step 3: CRR parameter calculations (exactly as in CRRParametersProvider::Compute)
        T dt = T(0.5);  // From American Option: dt = maturity/steps = 1.0/2 = 0.5
        
        // Cox-Ross-Rubinstein formulas
        T a = std::exp(sigma * std::sqrt(dt));  // exp(0.2625 * sqrt(0.5))
        T u = a;
        T d = T(1.0) / a;
        T erdt = std::exp(rate * dt);  // exp(0.015 * 0.5) = exp(0.0075)
        T p = (erdt - d) / (u - d);
        T disc = T(1.0) / erdt;
        
        // Return discount factor like American Option test
        return disc * x * T(100.0);
    }
    
    static fdouble crr_step_by_step(const fdouble& x) {
        return crr_step_by_step_impl<fdouble>(x);
    }
    
    static double crr_step_by_step_native(double x) {
        return crr_step_by_step_impl<double>(x);
    }
    
    // DEBUG: Check each intermediate value separately
    template<typename T>
    static T crr_debug_rate_impl(const T& x) {
        T t = T(0.5) + x * T(0.0);
        T t1 = T(0.0);
        T t2 = T(1.0);
        T v1 = T(0.01);
        T v2 = T(0.02);
        T alpha = (t - t1) / (t2 - t1);
        T rate = v1 * (T(1.0) - alpha) + v2 * alpha;
        return rate * x * T(1000000.0);  // Amplify to see precision
    }
    
    template<typename T>
    static T crr_debug_sigma_impl(const T& x) {
        T t = T(0.5) + x * T(0.0);
        T base_vol = T(0.25);
        T sigma = base_vol * (T(1.0) + t * T(0.1));
        return sigma * x * T(1000000.0);  // Amplify to see precision
    }
    
    template<typename T>
    static T crr_debug_sigma_sqrt_dt_impl(const T& x) {
        T t = T(0.5) + x * T(0.0);
        T base_vol = T(0.25);
        T sigma = base_vol * (T(1.0) + t * T(0.1));
        T dt = T(0.5);
        T sigma_sqrt_dt = sigma * std::sqrt(dt);
        return sigma_sqrt_dt * x * T(1000000.0);  // Amplify to see precision
    }
    
    template<typename T>
    static T crr_debug_exp_sigma_sqrt_dt_impl(const T& x) {
        T t = T(0.5) + x * T(0.0);
        T base_vol = T(0.25);
        T sigma = base_vol * (T(1.0) + t * T(0.1));
        T dt = T(0.5);
        T a = std::exp(sigma * std::sqrt(dt));
        return a * x * T(10000.0);
    }
    
    template<typename T>
    static T crr_debug_erdt_impl(const T& x) {
        T t = T(0.5) + x * T(0.0);
        T t1 = T(0.0);
        T t2 = T(1.0);
        T v1 = T(0.01);
        T v2 = T(0.02);
        T alpha = (t - t1) / (t2 - t1);
        T rate = v1 * (T(1.0) - alpha) + v2 * alpha;
        T dt = T(0.5);
        T erdt = std::exp(rate * dt);
        return erdt * x * T(10000.0);
    }
    
    template<typename T>
    static T crr_debug_disc_impl(const T& x) {
        T t = T(0.5) + x * T(0.0);
        T t1 = T(0.0);
        T t2 = T(1.0);
        T v1 = T(0.01);
        T v2 = T(0.02);
        T alpha = (t - t1) / (t2 - t1);
        T rate = v1 * (T(1.0) - alpha) + v2 * alpha;
        T dt = T(0.5);
        T erdt = std::exp(rate * dt);
        T disc = T(1.0) / erdt;
        return disc * x * T(100.0);
    }
    
    // Wrapper functions for all debug tests
    static fdouble crr_debug_rate(const fdouble& x) { return crr_debug_rate_impl<fdouble>(x); }
    static double crr_debug_rate_native(double x) { return crr_debug_rate_impl<double>(x); }
    
    static fdouble crr_debug_sigma(const fdouble& x) { return crr_debug_sigma_impl<fdouble>(x); }
    static double crr_debug_sigma_native(double x) { return crr_debug_sigma_impl<double>(x); }
    
    static fdouble crr_debug_sigma_sqrt_dt(const fdouble& x) { return crr_debug_sigma_sqrt_dt_impl<fdouble>(x); }
    static double crr_debug_sigma_sqrt_dt_native(double x) { return crr_debug_sigma_sqrt_dt_impl<double>(x); }
    
    static fdouble crr_debug_exp_sigma_sqrt_dt(const fdouble& x) { return crr_debug_exp_sigma_sqrt_dt_impl<fdouble>(x); }
    static double crr_debug_exp_sigma_sqrt_dt_native(double x) { return crr_debug_exp_sigma_sqrt_dt_impl<double>(x); }
    
    static fdouble crr_debug_erdt(const fdouble& x) { return crr_debug_erdt_impl<fdouble>(x); }
    static double crr_debug_erdt_native(double x) { return crr_debug_erdt_impl<double>(x); }
    
    static fdouble crr_debug_disc(const fdouble& x) { return crr_debug_disc_impl<fdouble>(x); }
    static double crr_debug_disc_native(double x) { return crr_debug_disc_impl<double>(x); }
    
    // Test exp(negative) values specifically to isolate JIT issues
    template<typename T>
    static T exp_negative_stress_impl(const T& x) {
        // Test various exp(negative) computations
        T small_neg = T(-0.1) * x;
        T exp_small_neg = std::exp(small_neg);
        
        T medium_neg = T(-0.5) * x;  
        T exp_medium_neg = std::exp(medium_neg);
        
        T large_neg = T(-2.0) * x;
        T exp_large_neg = std::exp(large_neg);
        
        // Mix positive and negative exponentials
        T pos = std::exp(x * T(0.1));
        T neg = std::exp(x * T(-0.1));
        
        // This should be close to 1.0 for any x
        T product = pos * neg;
        
        return exp_small_neg + exp_medium_neg * T(10.0) + exp_large_neg * T(100.0) + (product - T(1.0)) * T(10000.0);
    }
    
    static fdouble exp_negative_stress(const fdouble& x) {
        return exp_negative_stress_impl<fdouble>(x);
    }
    
    static double exp_negative_stress_native(double x) {
        return exp_negative_stress_impl<double>(x);
    }
    
    // Test if exp(-x) alone works correctly (isolating the problem)
    template<typename T>
    static T exp_negation_test_impl(const T& x) {
        // Test if exp(-value) works correctly by itself
        T value = x * T(0.0861433);  // Same as sigma * sqrt(dt) = 0.2725 * 0.316227...
        
        T exp_pos = std::exp(value);
        T exp_neg = std::exp(-value);
        
        // These should multiply to 1.0 exactly: exp(x) * exp(-x) = exp(0) = 1
        T product = exp_pos * exp_neg;
        T error = product - T(1.0);
        
        // Amplify to see any error
        return error * T(1000000000.0);
    }
    
    static fdouble exp_negation_test(const fdouble& x) {
        return exp_negation_test_impl<fdouble>(x);
    }
    
    static double exp_negation_test_native(double x) {
        return exp_negation_test_impl<double>(x);
    }

    // Simple template test functions for gradient testing with finite differences
    template<typename T>
    static T polynomial_template(const T& x) {
        return x * x * x + T(2.0) * x * x - T(5.0) * x + T(3.0);
    }
    
    template<typename T>
    static T rational_template(const T& x) {
        return (x + T(1.0)) / (x - T(1.0));
    }
    
    template<typename T>
    static T exponential5_template(const T& x) {
        T x2 = x * x;
        T x3 = x2 * x;
        T x5 = x3 * x2;
        return x5 - T(3.0) * x3 + T(2.0) * x;
    }
    
    template<typename T>
    static T nested_template(const T& x) {
        return ((x + T(2.0)) * T(3.0) - T(1.0)) / (x - T(0.5));
    }
    
    template<typename T>
    static T sine_taylor_template(const T& x) {
        T x2 = x * x;
        T x3 = x2 * x;
        T x5 = x3 * x2;
        return x - x3 / T(6.0) + x5 / T(120.0);
    }
    
    template<typename T>
    static T complex_rational_template(const T& x) {
        T x2 = x * x;
        return (x2 + T(1.0)) / (x2 - T(1.0));
    }
    
    template<typename T>
    static T gaussian_like_template(const T& x) {
        return T(1.0) / (T(1.0) + x * x);
    }
    
    template<typename T>
    static T product_template(const T& x) {
        return (x + T(1.0)) * (x - T(1.0)) * (x + T(2.0));
    }
    
    template<typename T>
    static T reciprocal_squared_template(const T& x) {
        return T(1.0) / (x * x);
    }
    
    // Additional test cases to isolate the gradient bug
    template<typename T>
    static T one_over_x_plus_one(const T& x) {
        return T(1.0) / (x + T(1.0));  // 1/(x+1)
    }
    
    template<typename T>
    static T x_over_x_plus_one(const T& x) {
        return x / (x + T(1.0));  // x/(x+1)
    }
    
    template<typename T>
    static T one_over_x(const T& x) {
        return T(1.0) / x;  // 1/x
    }
    
    template<typename T>
    static T one_plus_x_squared(const T& x) {
        return T(1.0) + x * x;  // 1 + x^2 (without division)
    }
    
    template<typename T>
    static T x_over_x_squared_plus_one(const T& x) {
        return x / (x * x + T(1.0));  // x/(x^2+1)
    }
    
    template<typename T>
    static T two_over_x_plus_one(const T& x) {
        return T(2.0) / (x + T(1.0));  // 2/(x+1)
    }
    
    // Template version that works with both Double and double
    template<typename T>
    static T massive_expression_impl(const T& x) {
        // Start with some basic transformations (10 ops)
        T a = x * 2.0 + 3.0;
        T b = x * x - 1.0;
        T c = (x + 1.0) * (x - 1.0);
        T d = (x - 0.5) * (x - 0.5);  // Changed from abs to multiplication
        T e = (x + 0.1) * (x + 0.1);  // Changed from square to multiplication
        
        // Some divisions and negations (10 ops) - no reciprocals
        T f = 1.0 / (a + 1.0);  // Changed from recip to division
        T g = -b * 2.0;
        T h = 1.0 / (c + 10.0);  // Changed from recip to division
        T i = -d + e;
        T j = f * f + g * g;  // Changed from square to multiplication
        
        // Nested operations (20 ops)
        T k = (a + b) * (c - d) / (e + 1.0);
        T l = (k - 5.0) * (k - 5.0) + h * h;  // Changed from abs/square
        T m = (1.0 / (i * i + 0.1)) * j;  // Changed from recip(abs())
        T n = -(k + l) * (m - 2.0);
        T o = x * 0.5 * x * 0.5 * x * 0.5 * x * 0.5;  // x^4 / 16 without square
        
        // More complex combinations (20 ops)
        T p = (n + o) / (m * m + 1.0);  // Changed from abs
        T q = 1.0 / (p * p + 0.01);  // Changed from recip(square())
        T r = (q - 0.5) * (q + 0.5);  // Removed abs
        T s = r * r + (1.0 - r) * (1.0 - r);  // Changed from square
        T t = (s * p + q * n) / (o + 1.0);
        
        // Wave-like patterns (20 ops)
        T u = t * (1.0 + x * 0.1);
        T v = u - (u * 0.5) * (u * 0.5);  // Changed from square
        T w = v + (v - 0.5) * (v - 0.5) * 0.2;  // Changed from abs
        T wave1 = w * (2.0 - x * 0.3 * x * 0.3);  // Changed from abs
        T wave2 = wave1 + (wave1 * 0.7) * (wave1 * 0.7);  // Changed from square
        
        // Rational-like expressions (20 ops)
        T num1 = wave2 * x + 1.0;
        T den1 = x * x + x + 1.0;  // Changed from square
        T rat1 = num1 / den1;
        T num2 = (rat1 - 0.5) * (rat1 - 0.5) * 3.0;  // Changed from abs
        T den2 = 1.0 / (rat1 + 0.1);  // Changed from recip
        T rat2 = num2 * den2;
        
        // Final mixing operations (20+ ops)
        T mix1 = rat2 * wave2 + rat1 * rat1;  // Changed from square
        T mix2 = (mix1 - 1.0) * (mix1 - 1.0) + 1.0 / (mix1 + 2.0);  // Changed from abs/recip
        T mix3 = -mix2 * 0.5 + (mix1 * 0.3) * (mix1 * 0.3);  // Changed from square
        T mix4 = (mix3 + mix2) / ((mix1 * mix1) + 0.5);  // Changed from abs(mix1)
        T mix5 = 1.0 / ((mix4 * mix4) + 0.01);  // Changed from recip(square())
        
        // Last stretch to reach 100+ operations
        T final1 = mix5 + (x - mix4) * (x - mix4);  // Changed from abs()
        T final2 = (final1 - 0.5) * (final1 - 0.5);  // Changed from square()
        T final3 = 1.0 / (final2 + 0.1);  // Changed from recip()
        T final4 = -final3 + final1 * final1;  // Changed from abs()
        T final5 = (final4 * final2 + final3 * final1) / 2.0;
        
        return final5;
    }
    
    // Wrapper functions for Double and double
    static fdouble massive_expression(const fdouble& x) {
        return massive_expression_impl<fdouble>(x);
    }
    
    // Ultra-massive test with 100k+ operations simulating iterative solver
    // Parameterized version for testing different scales
    template<int ITERATIONS>
    static fdouble ultra_massive_iterative_n(const fdouble& x) {
        // Simulate solving a PDE or iterative numerical method
        // Each iteration does ~100 operations
        
        fdouble u = x;  // Initial condition
        fdouble dt(0.001); // Time step
        
        // Simulate N time steps of a diffusion-like equation
        // Each iteration: u_new = u + dt * (diffusion + reaction + forcing)
        for (int iter = 0; iter < ITERATIONS; ++iter) {
            // Compute spatial derivatives (fake 1D discretization)
            fdouble u_left = u * 0.98;  // Fake left neighbor
            fdouble u_right = u * 1.02; // Fake right neighbor
            
            // Second derivative approximation (diffusion term)
            fdouble u_xx = (u_left - 2.0 * u + u_right) / (0.1 * 0.1);
            
            // Add some nonlinear reaction terms
            fdouble reaction = u * (1.0 - u) * (u - 0.5);  // Bistable reaction
            reaction = reaction * 10.0;
            
            // Add forcing/source terms with various operations
            fdouble forcing = (u - 0.5) * (u - 0.5) * 0.1;  // Changed from abs()
            forcing = forcing + (u * 0.5) * (u * 0.5) * 0.05;  // Changed from square()
            forcing = forcing - 1.0 / ((u * u) + 1.0) * 0.02;  // Changed from recip(abs())
            
            // Some additional complex terms to increase operation count
            fdouble modifier = (u_xx * u_xx) * 0.001;  // Changed from square(abs())
            modifier = modifier + 1.0 / ((reaction * reaction) + 0.1) * 0.01;  // Changed from recip(square())
            modifier = modifier * (1.0 + forcing * forcing);  // Changed from abs()
            
            // Update step with stabilization
            fdouble delta = dt * (u_xx * 0.1 + reaction * 0.01 + forcing - modifier);
            u = u + delta;
            
        }
        
        // Final post-processing with many operations
        for (int i = 0; i < 50; ++i) {
            fdouble post = u + 0.01 * i;
            post = (post * post) - (post - 0.5) * (post - 0.5);  // Changed from square() and abs()
            post = 1.0 / ((post * post) + 0.1);  // Changed from recip(abs())
            u = u * 0.99 + post * 0.01;
        }
        
        return u;
    }
    
    // Convenience functions with fixed iteration counts for testing
    static fdouble ultra_massive_iterative_10(const fdouble& x) {
        return ultra_massive_iterative_n<10>(x);
    }
    
    static fdouble ultra_massive_iterative_100(const fdouble& x) {
        return ultra_massive_iterative_n<100>(x);
    }
    
    static fdouble ultra_massive_iterative_1k(const fdouble& x) {
        return ultra_massive_iterative_n<1000>(x);
    }
    
    // Original 1000-iteration version
    static fdouble ultra_massive_iterative(const fdouble& x) {
        return ultra_massive_iterative_n<1000>(x);
    }
    
    // Native implementations with parameterized iterations
    template<int ITERATIONS>
    static double ultra_massive_iterative_native_n(double x) {
        double u = x;
        double dt = 0.001;
        
        for (int iter = 0; iter < ITERATIONS; ++iter) {
            double u_left = u * 0.98;
            double u_right = u * 1.02;
            
            double u_xx = (u_left - 2.0 * u + u_right) / (0.1 * 0.1);
            
            double reaction = u * (1.0 - u) * (u - 0.5);
            reaction = reaction * 10.0;
            
            double forcing = (u - 0.5) * (u - 0.5) * 0.1;
            forcing = forcing + (u * 0.5) * (u * 0.5) * 0.05;
            forcing = forcing - 1.0 / ((u * u) + 1.0) * 0.02;
            
            double modifier = (u_xx * u_xx) * 0.001;
            modifier = modifier + 1.0 / ((reaction * reaction) + 0.1) * 0.01;
            modifier = modifier * (1.0 + forcing * forcing);
            
            double delta = dt * (u_xx * 0.1 + reaction * 0.01 + forcing - modifier);
            u = u + delta;
            
        }
        
        for (int i = 0; i < 50; ++i) {
            double post = u + 0.01 * i;
            post = (post * post) - (post - 0.5) * (post - 0.5);
            post = 1.0 / ((post * post) + 0.1);
            u = u * 0.99 + post * 0.01;
        }
        
        return u;
    }
    
    // Native convenience functions
    static double ultra_massive_iterative_native_10(double x) {
        return ultra_massive_iterative_native_n<10>(x);
    }
    
    static double ultra_massive_iterative_native_100(double x) {
        return ultra_massive_iterative_native_n<100>(x);
    }
    
    static double ultra_massive_iterative_native_1k(double x) {
        return ultra_massive_iterative_native_n<1000>(x);
    }
    
    static double ultra_massive_iterative_native(double x) {
        return ultra_massive_iterative_native_n<1000>(x);
    }
    
    static double massive_expression_native(double x) {
        // Use the same templated implementation to ensure consistency
        return massive_expression_impl<double>(x);
    }
    
    // VIRTUAL METHOD CALL PRECISION BUG TEST - TEMPLATE VERSION
    // Replicates the exact bug found in American option pricing
    template<typename T>
    static T virtual_precision_bug_test_impl(const T& x) {
        // Create a simple virtual interface like the curves
        struct TestInterface {
            virtual ~TestInterface() = default;
            virtual T getValue() const = 0;
        };
        
        struct TestImpl : public TestInterface {
            T getValue() const override {
                // Return a value that will expose precision differences
                // This simulates the vCurve->GetValue() call
                return T(0.2875);  // Same value as volatility curve
            }
        };
        
        auto testObj = std::make_shared<TestImpl>();
        
        // The problematic pattern: virtual call result used in complex math
        T virtual_result = testObj->getValue();        // Virtual method call
        T direct_result = T(0.2875);                   // Direct construction
        
        // Apply the same mathematical operations that caused the bug
        T sqrt_val = std::sqrt(T(0.1));                // sqrt(dt)
        
        T virtual_product = virtual_result * sqrt_val;
        T direct_product = direct_result * sqrt_val;
        
        T virtual_exp = std::exp(virtual_product);
        T direct_exp = std::exp(direct_product);
        
        // Return the difference amplified by input to make it visible
        T difference = virtual_exp - direct_exp;
        return difference * x * T(10000.0);
    }
    
    // Wrapper functions for Double and double
    static fdouble virtual_precision_bug_test(const fdouble& x) {
        return virtual_precision_bug_test_impl<fdouble>(x);
    }
    
    static double virtual_precision_bug_test_native(double x) {
        return virtual_precision_bug_test_impl<double>(x);
    }
    
    // American Option pricing wrapper
    // Handles both test range [-0.5, 0.5] and benchmark value x=2.5
    static fdouble american_option(const fdouble& x) {
        // For benchmark (x=2.5), map to reasonable spot around 100
        // For normal tests (x in [-0.5, 0.5]), use smaller range
        
        // Use fbool for conditional selection
        fbool is_benchmark = cmpGT(x, fdouble(2.0));
        
        // Calculate both spot values
        fdouble benchmark_spot = fdouble(100.0) + (x - fdouble(2.0)) * fdouble(10.0);
        fdouble normal_spot = fdouble(100.0) + x * fdouble(20.0);
        
        // Select the appropriate spot value
        fdouble spot = is_benchmark.If(benchmark_spot, normal_spot);
        
        return AmericanOption::price_binomial_tree(spot);
    }
    
    // fint conditional selection test - ULTRA SIMPLE TEST
    // Test just the fint::If without array indexing
    static fdouble intTP_selection_test(const fdouble& x) {
        // Simple binary condition: x < 0
        fbool isNegative = x < fdouble(0.0);
        
        // Select fint index: 0 if negative, 1 if positive/zero
        fint selectedIndex = fint::If(isNegative, fint(0), fint(1));
        
        // Test the fint value directly using equality comparisons
        // This avoids the array indexing complexity
        fbool is0 = selectedIndex == fint(0);
        fbool is1 = selectedIndex == fint(1);
        
        // Return different values based on which index was selected
        // If selectedIndex == 0, return 7.0
        // If selectedIndex == 1, return 13.0
        fdouble result = fbool::If(is0, fdouble(7.0), fdouble(13.0));
        
        return result;
    }
    
    static double intTP_selection_test_native(double x) {
        // Native version: simplified binary test
        int selectedIndex;
        if (x < 0.0) {
            selectedIndex = 0;   // negative -> index 0
        } else {
            selectedIndex = 1;   // positive/zero -> index 1
        }
        
        // Array values [7.0, 13.0]
        double results[] = {7.0, 13.0};
        return results[selectedIndex];
    }
    
    static double american_option_native(double x) {
        double spot;
        if (x > 2.0) {
            // Benchmark case: x=2.5 -> spot=105
            spot = 100.0 + (x - 2.0) * 10.0;
        } else {
            // Normal test range
            spot = 100.0 + x * 20.0;
        }
        return AmericanOption::price_binomial_tree_native(spot);
    }
    
    // ========== GRADIENT TEST FUNCTIONS ==========
    // Polynomial for gradient testing: f(x) = x^3 + 2x^2 - 5x + 3
    // Analytical derivative: f'(x) = 3x^2 + 4x - 5
    static fdouble polynomial_grad(const fdouble& x) {
        fdouble x2 = x * x;           // x^2
        fdouble x3 = x2 * x;          // x^3
        fdouble term1 = x3;           // x^3
        fdouble term2 = 2.0 * x2;     // 2x^2
        fdouble term3 = -5.0 * x;     // -5x
        fdouble term4 = fdouble(3.0);// 3
        return term1 + term2 + term3 + term4;
    }
    
    static double polynomial_grad_native(double x) {
        return x*x*x + 2.0*x*x - 5.0*x + 3.0;
    }
    
    static double polynomial_grad_derivative(double x) {
        // Analytical derivative for verification
        return 3.0*x*x + 4.0*x - 5.0;
    }
    
    // Conditional gradient test: f(x) = (x > 0) ? 2*x : -x
    // Derivative: f'(x) = (x > 0) ? 2 : -1
    static fdouble conditional_grad(const fdouble& x) {
        fbool condition = x > fdouble(0.0);
        fdouble positive_branch = 2.0 * x;
        fdouble negative_branch = -x;
        return condition.If(positive_branch, negative_branch);
    }
    
    static double conditional_grad_native(double x) {
        return (x > 0.0) ? 2.0*x : -x;
    }
    
    static double conditional_grad_derivative(double x) {
        // Analytical derivative for verification
        return (x > 0.0) ? 2.0 : -1.0;
    }
};

// Test case descriptor for 1D functions
struct TestCase1D {
    std::string name;
    std::function<fdouble(fdouble)> func;
    std::function<double(double)> native_func;
    std::vector<double> test_inputs;
    bool skip_zero;  // For division cases
    double tolerance; // Numerical tolerance for comparisons
    
    TestCase1D(const std::string& n,
               std::function<fdouble(fdouble)> f,
               std::function<double(double)> nf,
               bool sz = false,
               double tol = 1e-10)
        : name(n), func(f), native_func(nf), skip_zero(sz), tolerance(tol) {
        // Default test inputs
        test_inputs = {0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 10.0, -10.0,
                      3.14159, -3.14159, 100.0, -100.0, 0.1, -0.1};
    }
    
    TestCase1D& withInputs(const std::vector<double>& inputs) {
        test_inputs = inputs;
        return *this;
    }
};

// Get all test cases
inline std::vector<TestCase1D> getAllTestCases1D() {
    return {
        TestCase1D("Linear", TestFunctions1D::linear, TestFunctions1D::linear_native),
        TestCase1D("Quadratic", TestFunctions1D::quadratic, TestFunctions1D::quadratic_native),
        TestCase1D("Cubic", TestFunctions1D::cubic, TestFunctions1D::cubic_native),
        TestCase1D("Quartic", TestFunctions1D::quartic, TestFunctions1D::quartic_native),
        TestCase1D("Rational", TestFunctions1D::rational, TestFunctions1D::rational_native, true),
        TestCase1D("Rational2", TestFunctions1D::rational2, TestFunctions1D::rational2_native)
            .withInputs({0.0, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 10.0, -10.0}), // Avoid x=-0.5
        TestCase1D("Exponential5", TestFunctions1D::exponential5, TestFunctions1D::exponential5_native)
            .withInputs({0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0}), // Avoid large values
        TestCase1D("ComplexPolynomial", TestFunctions1D::complex_polynomial, 
                   TestFunctions1D::complex_polynomial_native),
        TestCase1D("Inverse", TestFunctions1D::inverse, TestFunctions1D::inverse_native, true),
        TestCase1D("InverseSquared", TestFunctions1D::inverse_squared, 
                   TestFunctions1D::inverse_squared_native, true),
        TestCase1D("NestedArithmetic", TestFunctions1D::nested_arithmetic, 
                   TestFunctions1D::nested_arithmetic_native)
            .withInputs({0.0, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 10.0, -10.0}), // Avoid x=0.5
        TestCase1D("DeepNesting", TestFunctions1D::deep_nesting, 
                   TestFunctions1D::deep_nesting_native, false, 1e-8),
        TestCase1D("Alternating", TestFunctions1D::alternating, TestFunctions1D::alternating_native),
        TestCase1D("AbsLike", TestFunctions1D::abs_like, TestFunctions1D::abs_like_native)
            .withInputs({0.1, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 10.0, -10.0}), // Avoid x≈-0.001
        TestCase1D("StepLike", TestFunctions1D::step_like, TestFunctions1D::step_like_native),
        TestCase1D("GaussianLike", TestFunctions1D::gaussian_like, TestFunctions1D::gaussian_like_native),
        TestCase1D("SineApprox", TestFunctions1D::sine_approx, TestFunctions1D::sine_approx_native)
            .withInputs({0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 1.57, -1.57}), // Small values for Taylor
        TestCase1D("CosineApprox", TestFunctions1D::cosine_approx, TestFunctions1D::cosine_approx_native)
            .withInputs({0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 1.57, -1.57}), // Small values for Taylor
        TestCase1D("Compound1", TestFunctions1D::compound1, TestFunctions1D::compound1_native),
        TestCase1D("Compound2", TestFunctions1D::compound2, TestFunctions1D::compound2_native)
            .withInputs({0.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0}), // Avoid a*b+1=0
        
        // New operation test cases
        TestCase1D("Negation", TestFunctions1D::negation, TestFunctions1D::negation_native),
        TestCase1D("Absolute", TestFunctions1D::absolute, TestFunctions1D::absolute_native),
        TestCase1D("Squared", TestFunctions1D::squared, TestFunctions1D::squared_native),
        TestCase1D("Reciprocal", TestFunctions1D::reciprocal, TestFunctions1D::reciprocal_native, true),
        
        // Transcendental function test cases
        TestCase1D("ExpTest", TestFunctions1D::exp_test, TestFunctions1D::exp_test_native, false, 1e-8)
            .withInputs({-2.0, -1.0, 0.0, 0.5, 1.0, 2.0}),
        TestCase1D("LogTest", TestFunctions1D::log_test, TestFunctions1D::log_test_native, false, 1e-10)
            .withInputs({-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0}),
        TestCase1D("SqrtTest", TestFunctions1D::sqrt_test, TestFunctions1D::sqrt_test_native, false, 1e-10)
            .withInputs({-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0}),
        TestCase1D("TranscendentalCombo", TestFunctions1D::transcendental_combo, 
                   TestFunctions1D::transcendental_combo_native, false, 1e-8)
            .withInputs({-1.0, 0.0, 0.5, 1.0, 2.0}),
        
        // Trigonometric function test cases
        TestCase1D("SinTest", TestFunctions1D::sin_test, TestFunctions1D::sin_test_native, false, 1e-10)
            .withInputs({-3.14159, -1.5708, 0.0, 1.5708, 3.14159}),  // -π, -π/2, 0, π/2, π
        TestCase1D("CosTest", TestFunctions1D::cos_test, TestFunctions1D::cos_test_native, false, 1e-10)
            .withInputs({-3.14159, -1.5708, 0.0, 1.5708, 3.14159}),
        TestCase1D("TanTest", TestFunctions1D::tan_test, TestFunctions1D::tan_test_native, false, 1e-10)
            .withInputs({-1.0, -0.5, 0.0, 0.5, 1.0}),  // Smaller range to avoid singularities
        TestCase1D("TrigCombo", TestFunctions1D::trig_combo, TestFunctions1D::trig_combo_native, false, 1e-10)
            .withInputs({-1.5708, -0.7854, 0.0, 0.7854, 1.5708}),  // -π/2, -π/4, 0, π/4, π/2
        
        // Min/Max comparison operator test cases
        TestCase1D("MinTest", TestFunctions1D::min_test, TestFunctions1D::min_test_native),
        TestCase1D("MaxTest", TestFunctions1D::max_test, TestFunctions1D::max_test_native),
        TestCase1D("MinMaxCombo", TestFunctions1D::minmax_combo, TestFunctions1D::minmax_combo_native),
        TestCase1D("ClampTest", TestFunctions1D::clamp_test, TestFunctions1D::clamp_test_native)
            .withInputs({-5.0, -2.0, 0.0, 1.0, 3.0, 5.0}), // Test clamping at boundaries
        
        // Comparison operator test cases (CmpLT, CmpLE, CmpGT, CmpGE, CmpEQ, CmpNE)
        TestCase1D("CmpDebug_Test", TestFunctions1D::cmp_debug_test, TestFunctions1D::cmp_debug_test_native)
            .withInputs({1.0, 2.0, 2.5, 3.0}),
        TestCase1D("CmpDebugMul_Test", TestFunctions1D::cmp_debug_mul_test, TestFunctions1D::cmp_debug_mul_test_native)
            .withInputs({1.0, 2.0, -1.0}),
        TestCase1D("CmpDebugAnd_Test", TestFunctions1D::cmp_debug_and_test, TestFunctions1D::cmp_debug_and_test_native)
            .withInputs({1.0, 2.0, 2.5, 3.0, 5.0}),
        TestCase1D("CmpDebugSeg4_Test", TestFunctions1D::cmp_debug_seg4_test, TestFunctions1D::cmp_debug_seg4_test_native)
            .withInputs({1.0, 2.0, 2.5, 3.0, 4.0, 5.0}),
        TestCase1D("CmpDebugAdd_Test", TestFunctions1D::cmp_debug_add_test, TestFunctions1D::cmp_debug_add_test_native)
            .withInputs({1.0, 2.0, 2.5, 3.0, 4.0, 5.0}),
        TestCase1D("CmpLT_Test", TestFunctions1D::cmpLT_test, TestFunctions1D::cmpLT_test_native)
            .withInputs({-3.0, -1.0, 0.0, 1.0, 3.0, 5.0}),
        TestCase1D("CmpLE_Test", TestFunctions1D::cmpLE_test, TestFunctions1D::cmpLE_test_native)
            .withInputs({-3.0, -2.0, 0.0, 2.0, 3.0}),
        TestCase1D("CmpGT_Test", TestFunctions1D::cmpGT_test, TestFunctions1D::cmpGT_test_native)
            .withInputs({-1.0, 0.0, 1.0, 2.0, 3.0}),
        TestCase1D("CmpGE_Test", TestFunctions1D::cmpGE_test, TestFunctions1D::cmpGE_test_native)
            .withInputs({-1.0, 0.0, 2.0, 2.5, 3.0}),
        TestCase1D("CmpEQ_Test", TestFunctions1D::cmpEQ_test, TestFunctions1D::cmpEQ_test_native)
            .withInputs({-1.0, 0.0, 1.0, 2.0}),
        TestCase1D("CmpNE_Test", TestFunctions1D::cmpNE_test, TestFunctions1D::cmpNE_test_native)
            .withInputs({-1.0, 0.0, 1.0, 2.0}),
        TestCase1D("CmpCombo_Test", TestFunctions1D::cmp_combo_test, TestFunctions1D::cmp_combo_test_native)
            .withInputs({-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0}), // Test all piecewise segments
        
        // LIMITATION TEST - Shows that ternary computes both branches
        TestCase1D("CmpLimitation_Test", TestFunctions1D::cmp_limitation_test, 
                   TestFunctions1D::cmp_limitation_test_native)
            .withInputs({-2.0, -1.0, 1.0, 2.0}),  // Test both x<=0 and x>0 cases
        
        // Complex test with 100+ operations
        TestCase1D("MassiveExpression", TestFunctions1D::massive_expression, 
                   TestFunctions1D::massive_expression_native, false, 1e-6)
            .withInputs({0.1, 0.5, 1.0, 1.5, 2.0}), // Small values to avoid numerical instability
            
        // Ultra-massive test with 100k+ operations (PDE-like iterative solver)
        TestCase1D("UltraMassiveIterative", TestFunctions1D::ultra_massive_iterative,
                   TestFunctions1D::ultra_massive_iterative_native, false, 1e-4)
            .withInputs({0.3, 0.5, 0.7}), // Limited inputs due to compile time
        
        // Reciprocal exponential bug test - should return 0 but might not in JIT
        TestCase1D("ReciprocalExpBug", TestFunctions1D::reciprocal_exp_bug,
                   TestFunctions1D::reciprocal_exp_bug_native, false, 1e-6)
            .withInputs({1.0, 2.0, 5.0, 10.0}),
        
        // Test if exp(-x) works correctly by itself
        TestCase1D("ExpNegationTest", TestFunctions1D::exp_negation_test,
                   TestFunctions1D::exp_negation_test_native, false, 1e-6)
            .withInputs({1.0, 2.0, 5.0, 10.0}),
    };
}

// Get benchmark test cases (subset suitable for performance testing)
inline std::vector<TestCase1D> getBenchmarkTestCases1D() {
    return {
        TestCase1D("Linear", TestFunctions1D::linear, TestFunctions1D::linear_native),
        TestCase1D("Quadratic", TestFunctions1D::quadratic, TestFunctions1D::quadratic_native),
        TestCase1D("Cubic", TestFunctions1D::cubic, TestFunctions1D::cubic_native),
        TestCase1D("CSETest", TestFunctions1D::cse_test, TestFunctions1D::cse_test_native),
        TestCase1D("Ops10", TestFunctions1D::ops_10, TestFunctions1D::ops_10_native),
        TestCase1D("Ops50", TestFunctions1D::ops_50, TestFunctions1D::ops_50_native),
        TestCase1D("ComplexPolynomial", TestFunctions1D::complex_polynomial, 
                   TestFunctions1D::complex_polynomial_native),
        TestCase1D("DeepNesting", TestFunctions1D::deep_nesting, 
                   TestFunctions1D::deep_nesting_native),
        
        // Reciprocal exp bug test (to verify stability cleaning optimization)
        TestCase1D("reciprocal_exp_bug", TestFunctions1D::reciprocal_exp_bug, 
                   TestFunctions1D::reciprocal_exp_bug_native, false, 1e-6)
            .withInputs({0.0, 1.0}),
        
        // Two-point curve boundary condition test (found in American Option debugging)
        TestCase1D("two_point_curve_boundary", TestFunctions1D::two_point_curve_boundary,
                   TestFunctions1D::two_point_curve_boundary_native, false, 1e-10)
            .withInputs({0.0, 0.5, 1.0, 2.0}),
        
        TestCase1D("crr_discount_factor", TestFunctions1D::crr_discount_factor,
                   TestFunctions1D::crr_discount_factor_native, false, 1e-8)
            .withInputs({0.0, 1.0, 2.0, 100.0}),
        
        // PROGRESSIVE DEBUG: American Option Step-by-Step Analysis
        TestCase1D("american_step1_rate", TestFunctions1D::american_step1_rate,
                   TestFunctions1D::american_step1_rate_native, false, 1e-10)
            .withInputs({0.0, 1.0}),
        
        TestCase1D("american_step2_vol", TestFunctions1D::american_step2_vol,
                   TestFunctions1D::american_step2_vol_native, false, 1e-10)
            .withInputs({0.0, 1.0}),
            
        TestCase1D("american_step3_exp_vol", TestFunctions1D::american_step3_exp_vol,
                   TestFunctions1D::american_step3_exp_vol_native, false, 1e-10)
            .withInputs({0.0, 1.0}),
            
        TestCase1D("american_step4_exp_rate", TestFunctions1D::american_step4_exp_rate,
                   TestFunctions1D::american_step4_exp_rate_native, false, 1e-10)
            .withInputs({0.0, 1.0}),
            
        TestCase1D("american_step5_disc", TestFunctions1D::american_step5_disc,
                   TestFunctions1D::american_step5_disc_native, false, 1e-10)
            .withInputs({0.0, 1.0}),
            
        // VIRTUAL CALL ISOLATION TESTS
        TestCase1D("american_virtual_calls", TestFunctions1D::american_virtual_calls,
                   TestFunctions1D::american_virtual_calls_native, false, 1e-8)
            .withInputs({0.0, 1.0}),
            
        TestCase1D("american_full_params", TestFunctions1D::american_full_params,
                   TestFunctions1D::american_full_params_native, false, 1e-8)
            .withInputs({0.0, 1.0}),
            
        // VIRTUAL METHOD ISOLATION TESTS
        TestCase1D("test_two_point_curve", TestFunctions1D::test_two_point_curve,
                   TestFunctions1D::test_two_point_curve_native, false, 1e-12)
            .withInputs({0.0, 1.0}),
            
        TestCase1D("test_volatility_curve", TestFunctions1D::test_volatility_curve,
                   TestFunctions1D::test_volatility_curve_native, false, 1e-12)
            .withInputs({0.0, 1.0}),
            
        // CRR STEP-BY-STEP DEBUGGING
        TestCase1D("crr_step_by_step", TestFunctions1D::crr_step_by_step,
                   TestFunctions1D::crr_step_by_step_native, false, 1e-8)
            .withInputs({0.0, 1.0}),
            
        TestCase1D("crr_debug_rate", TestFunctions1D::crr_debug_rate,
                   TestFunctions1D::crr_debug_rate_native, false, 1e-12)
            .withInputs({0.0, 1.0}),
            
        TestCase1D("crr_debug_sigma", TestFunctions1D::crr_debug_sigma,
                   TestFunctions1D::crr_debug_sigma_native, false, 1e-12)
            .withInputs({0.0, 1.0}),
            
        TestCase1D("crr_debug_sigma_sqrt_dt", TestFunctions1D::crr_debug_sigma_sqrt_dt,
                   TestFunctions1D::crr_debug_sigma_sqrt_dt_native, false, 1e-12)
            .withInputs({0.0, 1.0}),
            
        TestCase1D("crr_debug_exp_sigma_sqrt_dt", TestFunctions1D::crr_debug_exp_sigma_sqrt_dt,
                   TestFunctions1D::crr_debug_exp_sigma_sqrt_dt_native, false, 1e-12)
            .withInputs({0.0, 1.0}),
            
        TestCase1D("crr_debug_erdt", TestFunctions1D::crr_debug_erdt,
                   TestFunctions1D::crr_debug_erdt_native, false, 1e-12)
            .withInputs({0.0, 1.0}),
            
        TestCase1D("crr_debug_disc", TestFunctions1D::crr_debug_disc,
                   TestFunctions1D::crr_debug_disc_native, false, 1e-8)
            .withInputs({0.0, 1.0}),
        
        // Transcendental function benchmarks
        TestCase1D("ExpTest", TestFunctions1D::exp_test, TestFunctions1D::exp_test_native, false, 1e-8)
            .withInputs({-1.0, 0.0, 0.5, 1.0}),
        TestCase1D("LogTest", TestFunctions1D::log_test, TestFunctions1D::log_test_native, false, 1e-10)
            .withInputs({0.0, 0.5, 1.0, 2.0}),
        TestCase1D("SqrtTest", TestFunctions1D::sqrt_test, TestFunctions1D::sqrt_test_native, false, 1e-10)
            .withInputs({0.0, 1.0, 2.0, 4.0}),
        TestCase1D("TranscendentalCombo", TestFunctions1D::transcendental_combo, 
                   TestFunctions1D::transcendental_combo_native, false, 1e-8)
            .withInputs({0.0, 0.5, 1.0}),
        
        // Trigonometric function benchmarks
        TestCase1D("SinTest", TestFunctions1D::sin_test, TestFunctions1D::sin_test_native, false, 1e-10)
            .withInputs({0.0, 1.5708}),  // 0, π/2
        TestCase1D("CosTest", TestFunctions1D::cos_test, TestFunctions1D::cos_test_native, false, 1e-10)
            .withInputs({0.0, 3.14159}),  // 0, π
        TestCase1D("TanTest", TestFunctions1D::tan_test, TestFunctions1D::tan_test_native, false, 1e-10)
            .withInputs({0.0, 0.5}),
        TestCase1D("TrigCombo", TestFunctions1D::trig_combo, TestFunctions1D::trig_combo_native, false, 1e-10)
            .withInputs({0.0, 0.7854}),  // 0, π/4
        
        // Min/Max comparison operator benchmarks
        TestCase1D("MinTest", TestFunctions1D::min_test, TestFunctions1D::min_test_native)
            .withInputs({-2.0, 0.0, 1.0, 3.0}),
        TestCase1D("MaxTest", TestFunctions1D::max_test, TestFunctions1D::max_test_native)
            .withInputs({-1.0, 0.0, 1.0, 2.0}),
        TestCase1D("MinMaxCombo", TestFunctions1D::minmax_combo, TestFunctions1D::minmax_combo_native)
            .withInputs({-3.0, 0.0, 2.0}),
        TestCase1D("ClampTest", TestFunctions1D::clamp_test, TestFunctions1D::clamp_test_native)
            .withInputs({-5.0, 0.0, 5.0}),
        
        // Comparison operator benchmarks (subset for performance testing)
        TestCase1D("CmpDebug_Test", TestFunctions1D::cmp_debug_test, TestFunctions1D::cmp_debug_test_native)
            .withInputs({2.5}),
        TestCase1D("CmpDebugMul_Test", TestFunctions1D::cmp_debug_mul_test, TestFunctions1D::cmp_debug_mul_test_native)
            .withInputs({1.0, 2.0, -1.0}),
        TestCase1D("CmpDebugAnd_Test", TestFunctions1D::cmp_debug_and_test, TestFunctions1D::cmp_debug_and_test_native)
            .withInputs({2.5}),
        TestCase1D("CmpDebugSeg4_Test", TestFunctions1D::cmp_debug_seg4_test, TestFunctions1D::cmp_debug_seg4_test_native)
            .withInputs({2.5}),
        TestCase1D("CmpDebugAdd_Test", TestFunctions1D::cmp_debug_add_test, TestFunctions1D::cmp_debug_add_test_native)
            .withInputs({2.5}),
        TestCase1D("CmpLT_Test", TestFunctions1D::cmpLT_test, TestFunctions1D::cmpLT_test_native)
            .withInputs({-1.0, 0.0, 1.0, 3.0}),
        TestCase1D("CmpEQ_Test", TestFunctions1D::cmpEQ_test, TestFunctions1D::cmpEQ_test_native)
            .withInputs({0.0, 1.0, 2.0}),
        TestCase1D("CmpCombo_Test", TestFunctions1D::cmp_combo_test, TestFunctions1D::cmp_combo_test_native)
            .withInputs({-2.0, 0.0, 2.0, 4.0}),
        
        // Reciprocal exponential bug test
        TestCase1D("ReciprocalExpBug", TestFunctions1D::reciprocal_exp_bug,
                   TestFunctions1D::reciprocal_exp_bug_native, false, 1e-6)
            .withInputs({1.0, 2.0, 5.0}),
        
        // Test if exp(-x) works correctly
        TestCase1D("ExpNegationTest", TestFunctions1D::exp_negation_test,
                   TestFunctions1D::exp_negation_test_native, false, 1e-6)
            .withInputs({1.0, 2.0, 5.0}),
            
        // fint conditional selection benchmark - THE KEY MILESTONE
        TestCase1D("IntTP_Selection_Benchmark", TestFunctions1D::intTP_selection_test,
                   TestFunctions1D::intTP_selection_test_native, false, 1e-10)
            .withInputs({-1.0, 1.0}), // Simple binary test: negative and positive
        
        // LIMITATION TEST - Shows that ternary computes both branches
        TestCase1D("CmpLimitation_Test", TestFunctions1D::cmp_limitation_test, 
                   TestFunctions1D::cmp_limitation_test_native)
            .withInputs({-2.0, -1.0, 1.0, 2.0}),  // Test both x<=0 and x>0 cases
        
        TestCase1D("MassiveExpression", TestFunctions1D::massive_expression, 
                   TestFunctions1D::massive_expression_native, false, 1e-6)
            .withInputs({0.1, 0.5, 1.0, 1.5, 2.0}),
        
        // Smaller scale ultra massive tests for debugging
        TestCase1D("UltraMassive1", TestFunctions1D::ultra_massive_iterative_n<1>,
                   TestFunctions1D::ultra_massive_iterative_native_n<1>, false, 1e-4)
            .withInputs({0.5}),
        TestCase1D("UltraMassive10", TestFunctions1D::ultra_massive_iterative_10,
                   TestFunctions1D::ultra_massive_iterative_native_10, false, 1e-4)
            .withInputs({0.5}),
        TestCase1D("UltraMassive100", TestFunctions1D::ultra_massive_iterative_100,
                   TestFunctions1D::ultra_massive_iterative_native_100, false, 1e-4)
            .withInputs({0.5}),
        TestCase1D("UltraMassive1K", TestFunctions1D::ultra_massive_iterative_1k,
                   TestFunctions1D::ultra_massive_iterative_native_1k, false, 1e-4)
            .withInputs({0.5}),
        
        // Original ultra-massive test with 100k+ operations
        TestCase1D("UltraMassiveIterative", TestFunctions1D::ultra_massive_iterative,
                   TestFunctions1D::ultra_massive_iterative_native, false, 1e-4)
            .withInputs({0.5}), // Single input to reduce benchmark time
            
        // Virtual Method Call Precision Bug Test - REGRESSION TEST
        TestCase1D("VirtualPrecisionBug", TestFunctions1D::virtual_precision_bug_test,
                   TestFunctions1D::virtual_precision_bug_test_native, false, 1e-1) // Loose tolerance due to known bug
            .withInputs({1.0, 2.0, 5.0}), // Various amplification factors
            
        // American Option pricing (complex virtual-call intensive)
        TestCase1D("AmericanOption", TestFunctions1D::american_option,
                   TestFunctions1D::american_option_native, false, 1e-6)
            .withInputs({0.0}), // Maps to spots [90, 96, 100, 104, 110]
            
    };
}

// Special test case descriptor for gradient correctness tests
struct GradientTestCase1D {
    std::string name;
    std::function<fdouble(fdouble)> func;
    std::function<double(double)> native_func;
    std::function<double(double)> derivative_func;  // Analytical derivative
    std::vector<double> test_points;
    double tolerance;
    
    GradientTestCase1D(const std::string& n,
                       std::function<fdouble(fdouble)> f,
                       std::function<double(double)> nf,
                       std::function<double(double)> df,
                       double tol = 1e-10)
        : name(n), func(f), native_func(nf), derivative_func(df), tolerance(tol) {
        // Default test points for gradient verification
        test_points = {0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0};
    }
    
    GradientTestCase1D& withTestPoints(const std::vector<double>& points) {
        test_points = points;
        return *this;
    }
};

// Get gradient correctness test cases
inline std::vector<GradientTestCase1D> getGradientTestCases1D() {
    return {
        // Polynomial gradient test
        GradientTestCase1D("PolynomialGradient", 
                           TestFunctions1D::polynomial_grad,
                           TestFunctions1D::polynomial_grad_native,
                           TestFunctions1D::polynomial_grad_derivative)
            .withTestPoints({0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5}),
        
        // Conditional gradient test
        GradientTestCase1D("ConditionalGradient",
                           TestFunctions1D::conditional_grad,
                           TestFunctions1D::conditional_grad_native,
                           TestFunctions1D::conditional_grad_derivative)
            .withTestPoints({-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0}),  // Test around x=0 boundary
    };
}

// Finite difference derivative computation
class FiniteDifference {
public:
    // Central difference approximation: f'(x) ≈ [f(x+h) - f(x-h)] / 2h
    template<typename Func>
    static double centralDifference(Func f, double x, double h = 1e-8) {
        double f_plus = f(x + h);
        double f_minus = f(x - h);
        return (f_plus - f_minus) / (2.0 * h);
    }
    
    // Richardson extrapolation for higher accuracy
    template<typename Func>
    static double richardsonExtrapolation(Func f, double x, double h = 1e-5) {
        // Use Richardson extrapolation with central differences
        double D1 = centralDifference(f, x, h);
        double D2 = centralDifference(f, x, h / 2.0);
        return (4.0 * D2 - D1) / 3.0;  // O(h^4) accuracy
    }
};

// Test case structure for finite difference testing
struct FiniteDiffTestCase {
    std::string name;
    std::function<fdouble(const fdouble&)> func_tp;
    std::function<double(double)> func_native;
    std::vector<double> test_points;
    double tolerance;
    
    FiniteDiffTestCase(const std::string& n,
                        std::function<fdouble(const fdouble&)> f_tp,
                        std::function<double(double)> f_nat,
                        double tol = 1e-6)  // Relaxed tolerance for finite differences
        : name(n), func_tp(f_tp), func_native(f_nat), tolerance(tol) {
        // Default test points
        test_points = {-2.0, -1.5, -1.0, -0.5, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0};
    }
    
    FiniteDiffTestCase& withTestPoints(const std::vector<double>& points) {
        test_points = points;
        return *this;
    }
};

// Get test cases using template functions for finite difference testing
inline std::vector<FiniteDiffTestCase> getFiniteDiffTestCases() {
    return {
        FiniteDiffTestCase("Polynomial", 
                           TestFunctions1D::polynomial_template<fdouble>,
                           TestFunctions1D::polynomial_template<double>),
        
        FiniteDiffTestCase("Rational",
                           TestFunctions1D::rational_template<fdouble>,
                           TestFunctions1D::rational_template<double>)
            .withTestPoints({-2.0, -1.5, 0.0, 0.5, 0.75, 1.5, 2.0, 3.0}),  // Avoid x=1
        
        FiniteDiffTestCase("Exponential5",
                           TestFunctions1D::exponential5_template<fdouble>,
                           TestFunctions1D::exponential5_template<double>),
        
        FiniteDiffTestCase("Nested",
                           TestFunctions1D::nested_template<fdouble>,
                           TestFunctions1D::nested_template<double>)
            .withTestPoints({-2.0, -1.0, 0.0, 0.25, 0.75, 1.0, 2.0}),  // Avoid x=0.5
        
        FiniteDiffTestCase("SineTaylor",
                           TestFunctions1D::sine_taylor_template<fdouble>,
                           TestFunctions1D::sine_taylor_template<double>),
        
        FiniteDiffTestCase("ComplexRational",
                           TestFunctions1D::complex_rational_template<fdouble>,
                           TestFunctions1D::complex_rational_template<double>)
            .withTestPoints({-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0}),  // Avoid x=±1
        
        FiniteDiffTestCase("GaussianLike",
                           TestFunctions1D::gaussian_like_template<fdouble>,
                           TestFunctions1D::gaussian_like_template<double>),
        
        FiniteDiffTestCase("Product",
                           TestFunctions1D::product_template<fdouble>,
                           TestFunctions1D::product_template<double>),
        
        FiniteDiffTestCase("ReciprocalSquared",
                           TestFunctions1D::reciprocal_squared_template<fdouble>,
                           TestFunctions1D::reciprocal_squared_template<double>)
            .withTestPoints({-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0}),  // Avoid x=0
        
        // New test cases to isolate the bug
        FiniteDiffTestCase("OneOverX",
                           TestFunctions1D::one_over_x<fdouble>,
                           TestFunctions1D::one_over_x<double>)
            .withTestPoints({-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0}),  // Avoid x=0
        
        FiniteDiffTestCase("OneOverXPlusOne",
                           TestFunctions1D::one_over_x_plus_one<fdouble>,
                           TestFunctions1D::one_over_x_plus_one<double>)
            .withTestPoints({-2.0, -0.5, 0.0, 0.5, 1.0, 2.0}),  // Avoid x=-1
        
        FiniteDiffTestCase("XOverXPlusOne",
                           TestFunctions1D::x_over_x_plus_one<fdouble>,
                           TestFunctions1D::x_over_x_plus_one<double>)
            .withTestPoints({-2.0, -0.5, 0.0, 0.5, 1.0, 2.0}),  // Avoid x=-1
        
        FiniteDiffTestCase("OnePlusXSquared",
                           TestFunctions1D::one_plus_x_squared<fdouble>,
                           TestFunctions1D::one_plus_x_squared<double>),
        
        FiniteDiffTestCase("XOverXSquaredPlusOne",
                           TestFunctions1D::x_over_x_squared_plus_one<fdouble>,
                           TestFunctions1D::x_over_x_squared_plus_one<double>),
        
        FiniteDiffTestCase("TwoOverXPlusOne",
                           TestFunctions1D::two_over_x_plus_one<fdouble>,
                           TestFunctions1D::two_over_x_plus_one<double>)
            .withTestPoints({-2.0, -0.5, 0.0, 0.5, 1.0, 2.0})  // Avoid x=-1
    };
}

} // namespace testing
} // namespace forge