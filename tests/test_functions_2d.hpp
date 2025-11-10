#pragma once

#include "../../tools/types/fdouble.hpp"
#include <vector>
#include <functional>
#include <string>
#include <array>

namespace forge {
namespace testing {

// Repository of 2D test functions (ℝ² → ℝ)
struct TestFunctions2D {
    // Basic arithmetic operations
    static fdouble add(fdouble x, fdouble y) { return x + y; }
    static fdouble subtract(fdouble x, fdouble y) { return x - y; }
    static fdouble multiply(fdouble x, fdouble y) { return x * y; }
    static fdouble divide(fdouble x, fdouble y) { return x / y; }
    
    // Linear combinations
    static fdouble linear_combination(fdouble x, fdouble y) { return 2.0 * x + 3.0 * y + 1.0; }
    static fdouble weighted_sum(fdouble x, fdouble y) { return 0.7 * x + 0.3 * y; }
    
    // Polynomial functions
    static fdouble polynomial_2d(fdouble x, fdouble y) { return x * x + 2.0 * x * y + y * y; }
    static fdouble mixed_terms(fdouble x, fdouble y) { return x * x * y + x * y * y; }
    static fdouble cubic_2d(fdouble x, fdouble y) { return x * x * x + y * y * y - 3.0 * x * y; }
    
    // Rational functions
    static fdouble rational_2d(fdouble x, fdouble y) { return (x + y) / (x - y + 0.1); }
    static fdouble complex_rational(fdouble x, fdouble y) { return (x * x + y * y) / (x * y + 1.0); }
    
    // Distance and norm-like functions
    static fdouble manhattan_distance(fdouble x, fdouble y) { 
        // |x| + |y| approximation using smooth function
        fdouble abs_x = x * x / (x + 0.001);
        fdouble abs_y = y * y / (y + 0.001);
        return abs_x + abs_y;
    }
    static fdouble squared_norm(fdouble x, fdouble y) { return x * x + y * y; }
    
    // Trigonometric-like approximations
    static fdouble rotation_like(fdouble x, fdouble y) {
        // Approximates x*cos(θ) - y*sin(θ) for small θ
        return 0.9 * x - 0.1 * y;
    }
    
    // Min/max approximations (smooth)
    static fdouble smooth_max(fdouble x, fdouble y) {
        // Smooth approximation of max(x, y)
        return (x + y + ((x - y) * (x - y) + 0.01)) / 2.0;
    }
    
    // Complex expressions
    static fdouble nested_2d(fdouble x, fdouble y) {
        return ((x + 1.0) * (y - 1.0)) / ((x - 0.5) * (y + 0.5) + 0.1);
    }
    
    static fdouble compound_2d(fdouble x, fdouble y) {
        fdouble a = x + y;
        fdouble b = x - y;
        return (a * a + b * b) / (a * b + 1.0);
    }
    
    // New operation test function
    static fdouble modulo_2d(fdouble x, fdouble y) {
        return mod(x, y);
    }
    
    // Native versions for benchmarking
    static double add_native(double x, double y) { return x + y; }
    static double subtract_native(double x, double y) { return x - y; }
    static double multiply_native(double x, double y) { return x * y; }
    static double divide_native(double x, double y) { return x / y; }
    static double linear_combination_native(double x, double y) { return 2.0 * x + 3.0 * y + 1.0; }
    static double weighted_sum_native(double x, double y) { return 0.7 * x + 0.3 * y; }
    static double polynomial_2d_native(double x, double y) { return x * x + 2.0 * x * y + y * y; }
    static double mixed_terms_native(double x, double y) { return x * x * y + x * y * y; }
    static double cubic_2d_native(double x, double y) { return x * x * x + y * y * y - 3.0 * x * y; }
    static double rational_2d_native(double x, double y) { return (x + y) / (x - y + 0.1); }
    static double complex_rational_native(double x, double y) { return (x * x + y * y) / (x * y + 1.0); }
    static double manhattan_distance_native(double x, double y) { 
        double abs_x = x * x / (x + 0.001);
        double abs_y = y * y / (y + 0.001);
        return abs_x + abs_y;
    }
    static double squared_norm_native(double x, double y) { return x * x + y * y; }
    static double rotation_like_native(double x, double y) { return 0.9 * x - 0.1 * y; }
    static double smooth_max_native(double x, double y) {
        return (x + y + ((x - y) * (x - y) + 0.01)) / 2.0;
    }
    static double nested_2d_native(double x, double y) {
        return ((x + 1.0) * (y - 1.0)) / ((x - 0.5) * (y + 0.5) + 0.1);
    }
    static double compound_2d_native(double x, double y) {
        double a = x + y;
        double b = x - y;
        return (a * a + b * b) / (a * b + 1.0);
    }
    static double modulo_2d_native(double x, double y) {
        return std::fmod(x, y);
    }
};

// Test case descriptor for 2D→1D functions
struct TestCase2D {
    std::string name;
    std::function<fdouble(fdouble, fdouble)> func;
    std::function<double(double, double)> native_func;
    std::vector<std::array<double, 2>> test_inputs;
    double tolerance;
    
    TestCase2D(const std::string& n,
               std::function<fdouble(fdouble, fdouble)> f,
               std::function<double(double, double)> nf,
               double tol = 1e-10)
        : name(n), func(f), native_func(nf), tolerance(tol) {
        // Default test inputs - grid of values
        test_inputs = {
            {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0},
            {-1.0, 0.0}, {0.0, -1.0}, {-1.0, -1.0},
            {2.0, 3.0}, {-2.0, 3.0}, {2.0, -3.0},
            {0.5, 0.5}, {-0.5, 0.5}, {0.5, -0.5},
            {10.0, 1.0}, {1.0, 10.0}
        };
    }
    
    TestCase2D& withInputs(const std::vector<std::array<double, 2>>& inputs) {
        test_inputs = inputs;
        return *this;
    }
};

// Get all 2D test cases
inline std::vector<TestCase2D> getAllTestCases2D() {
    return {
        TestCase2D("Add", TestFunctions2D::add, TestFunctions2D::add_native),
        TestCase2D("Subtract", TestFunctions2D::subtract, TestFunctions2D::subtract_native),
        TestCase2D("Multiply", TestFunctions2D::multiply, TestFunctions2D::multiply_native),
        TestCase2D("Divide", TestFunctions2D::divide, TestFunctions2D::divide_native)
            .withInputs({{1.0, 2.0}, {2.0, 1.0}, {-1.0, 2.0}, {5.0, 0.5}, {10.0, 10.0}}),
        TestCase2D("LinearCombination", TestFunctions2D::linear_combination, 
                   TestFunctions2D::linear_combination_native),
        TestCase2D("WeightedSum", TestFunctions2D::weighted_sum, 
                   TestFunctions2D::weighted_sum_native),
        TestCase2D("Polynomial2D", TestFunctions2D::polynomial_2d, 
                   TestFunctions2D::polynomial_2d_native),
        TestCase2D("MixedTerms", TestFunctions2D::mixed_terms, 
                   TestFunctions2D::mixed_terms_native),
        TestCase2D("Cubic2D", TestFunctions2D::cubic_2d, 
                   TestFunctions2D::cubic_2d_native),
        TestCase2D("Rational2D", TestFunctions2D::rational_2d, 
                   TestFunctions2D::rational_2d_native),
        TestCase2D("ComplexRational", TestFunctions2D::complex_rational, 
                   TestFunctions2D::complex_rational_native)
            .withInputs({{1.0, 2.0}, {2.0, 1.0}, {-1.0, 2.0}, {0.5, 0.5}, {3.0, -2.0}}),
        TestCase2D("ManhattanDistance", TestFunctions2D::manhattan_distance, 
                   TestFunctions2D::manhattan_distance_native)
            .withInputs({{1.0, 1.0}, {-1.0, 1.0}, {2.0, 3.0}, {-2.0, -3.0}, {0.5, -0.5}}),
        TestCase2D("SquaredNorm", TestFunctions2D::squared_norm, 
                   TestFunctions2D::squared_norm_native),
        TestCase2D("RotationLike", TestFunctions2D::rotation_like, 
                   TestFunctions2D::rotation_like_native),
        TestCase2D("SmoothMax", TestFunctions2D::smooth_max, 
                   TestFunctions2D::smooth_max_native),
        TestCase2D("Nested2D", TestFunctions2D::nested_2d, 
                   TestFunctions2D::nested_2d_native),
        TestCase2D("Compound2D", TestFunctions2D::compound_2d, 
                   TestFunctions2D::compound_2d_native)
            .withInputs({{1.0, 2.0}, {2.0, 1.0}, {0.1, 0.2}, {3.0, 3.0}, {-1.0, 1.0}}),
        TestCase2D("Modulo2D", TestFunctions2D::modulo_2d,
                   TestFunctions2D::modulo_2d_native)
            .withInputs({{5.0, 3.0}, {7.0, 2.0}, {10.0, 4.0}, {-5.0, 3.0}, {5.5, 2.5}}),
    };
}

// Get benchmark test cases for 2D
inline std::vector<TestCase2D> getBenchmarkTestCases2D() {
    return {
        TestCase2D("Add", TestFunctions2D::add, TestFunctions2D::add_native),
        TestCase2D("LinearCombination", TestFunctions2D::linear_combination, 
                   TestFunctions2D::linear_combination_native),
        TestCase2D("Polynomial2D", TestFunctions2D::polynomial_2d, 
                   TestFunctions2D::polynomial_2d_native),
        TestCase2D("Nested2D", TestFunctions2D::nested_2d, 
                   TestFunctions2D::nested_2d_native),
    };
}

} // namespace testing
} // namespace forge