#pragma once

#include "../../tools/types/fdouble.hpp"
#include "../src/graph/handles.hpp"
#include <vector>
#include <functional>
#include <string>
#include <array>

namespace forge {
namespace testing {

// Structure to hold multiple outputs
struct MultiOutput {
    std::vector<fdouble> outputs;
    std::vector<double> expected_values;
};

// Repository of multi-input/multi-output test functions
struct TestFunctionsMultiIO {
    
    // 2D → 2D functions (ℝ² → ℝ²)
    static MultiOutput polar_to_cartesian_like(fdouble r, fdouble theta) {
        // Approximation: x ≈ r, y ≈ r * theta (for small theta)
        MultiOutput result;
        result.outputs.push_back(r);  // x
        result.outputs.push_back(r * theta);  // y
        return result;
    }
    
    static MultiOutput sum_and_product(fdouble x, fdouble y) {
        MultiOutput result;
        result.outputs.push_back(x + y);  // sum
        result.outputs.push_back(x * y);  // product
        return result;
    }
    
    static MultiOutput linear_transform_2d(fdouble x, fdouble y) {
        // Matrix multiplication: [2 1; 3 -1] * [x; y]
        MultiOutput result;
        result.outputs.push_back(2.0 * x + 1.0 * y);
        result.outputs.push_back(3.0 * x - 1.0 * y);
        return result;
    }
    
    // 3D → 1D functions (ℝ³ → ℝ)
    static fdouble volume_like(fdouble x, fdouble y, fdouble z) {
        return x * y * z;
    }
    
    static fdouble weighted_sum_3d(fdouble x, fdouble y, fdouble z) {
        return 0.5 * x + 0.3 * y + 0.2 * z;
    }
    
    static fdouble polynomial_3d(fdouble x, fdouble y, fdouble z) {
        return x * x + y * y + z * z + x * y * z;
    }
    
    // 1D → 3D functions (ℝ → ℝ³)
    static MultiOutput polynomial_derivatives(fdouble x) {
        // f(x) = x³ - 2x² + x
        // f'(x) = 3x² - 4x + 1
        // f''(x) = 6x - 4
        MultiOutput result;
        result.outputs.push_back(x * x * x - 2.0 * x * x + x);  // f
        result.outputs.push_back(3.0 * x * x - 4.0 * x + 1.0);  // f'
        result.outputs.push_back(6.0 * x - 4.0);                // f''
        return result;
    }
    
    static MultiOutput trig_like_expansion(fdouble x) {
        // Taylor series approximations
        MultiOutput result;
        result.outputs.push_back(x - x * x * x / 6.0);      // sin(x) approx
        result.outputs.push_back(1.0 - x * x / 2.0);        // cos(x) approx
        result.outputs.push_back(x + x * x * x / 3.0);      // tan(x) approx
        return result;
    }
    
    // 10D → 1D function (ℝ¹⁰ → ℝ)
    static fdouble sum_10d(const std::vector<fdouble>& inputs) {
        fdouble sum = inputs[0];
        for (size_t i = 1; i < inputs.size(); ++i) {
            sum = sum + inputs[i];
        }
        return sum;
    }
    
    static fdouble weighted_sum_10d(const std::vector<fdouble>& inputs) {
        fdouble sum = inputs[0] * 0.2;
        sum = sum + inputs[1] * 0.15;
        sum = sum + inputs[2] * 0.15;
        sum = sum + inputs[3] * 0.1;
        sum = sum + inputs[4] * 0.1;
        sum = sum + inputs[5] * 0.08;
        sum = sum + inputs[6] * 0.08;
        sum = sum + inputs[7] * 0.06;
        sum = sum + inputs[8] * 0.05;
        sum = sum + inputs[9] * 0.03;
        return sum;
    }
    
    // Native versions for verification
    static std::array<double, 2> polar_to_cartesian_like_native(double r, double theta) {
        return {r, r * theta};
    }
    
    static std::array<double, 2> sum_and_product_native(double x, double y) {
        return {x + y, x * y};
    }
    
    static std::array<double, 2> linear_transform_2d_native(double x, double y) {
        return {2.0 * x + 1.0 * y, 3.0 * x - 1.0 * y};
    }
    
    static double volume_like_native(double x, double y, double z) {
        return x * y * z;
    }
    
    static double weighted_sum_3d_native(double x, double y, double z) {
        return 0.5 * x + 0.3 * y + 0.2 * z;
    }
    
    static double polynomial_3d_native(double x, double y, double z) {
        return x * x + y * y + z * z + x * y * z;
    }
    
    static std::array<double, 3> polynomial_derivatives_native(double x) {
        return {
            x * x * x - 2.0 * x * x + x,
            3.0 * x * x - 4.0 * x + 1.0,
            6.0 * x - 4.0
        };
    }
    
    static std::array<double, 3> trig_like_expansion_native(double x) {
        return {
            x - x * x * x / 6.0,
            1.0 - x * x / 2.0,
            x + x * x * x / 3.0
        };
    }
    
    static double sum_10d_native(const std::array<double, 10>& inputs) {
        double sum = 0.0;
        for (double v : inputs) sum += v;
        return sum;
    }
    
    static double weighted_sum_10d_native(const std::array<double, 10>& inputs) {
        return inputs[0] * 0.2 + inputs[1] * 0.15 + inputs[2] * 0.15 + 
               inputs[3] * 0.1 + inputs[4] * 0.1 + inputs[5] * 0.08 + 
               inputs[6] * 0.08 + inputs[7] * 0.06 + inputs[8] * 0.05 + 
               inputs[9] * 0.03;
    }
};

// Test case for 2D → 2D functions
struct TestCase2Dto2D {
    std::string name;
    std::function<MultiOutput(fdouble, fdouble)> func;
    std::function<std::array<double, 2>(double, double)> native_func;
    std::vector<std::array<double, 2>> test_inputs;
    double tolerance;
    
    TestCase2Dto2D(const std::string& n,
                   std::function<MultiOutput(fdouble, fdouble)> f,
                   std::function<std::array<double, 2>(double, double)> nf,
                   double tol = 1e-10)
        : name(n), func(f), native_func(nf), tolerance(tol) {
        test_inputs = {
            {1.0, 2.0}, {-1.0, 2.0}, {2.0, -1.0}, {0.5, 0.5},
            {3.0, 4.0}, {0.0, 1.0}, {1.0, 0.0}
        };
    }
};

// Test case for 3D → 1D functions
struct TestCase3Dto1D {
    std::string name;
    std::function<fdouble(fdouble, fdouble, fdouble)> func;
    std::function<double(double, double, double)> native_func;
    std::vector<std::array<double, 3>> test_inputs;
    double tolerance;
    
    TestCase3Dto1D(const std::string& n,
                   std::function<fdouble(fdouble, fdouble, fdouble)> f,
                   std::function<double(double, double, double)> nf,
                   double tol = 1e-10)
        : name(n), func(f), native_func(nf), tolerance(tol) {
        test_inputs = {
            {1.0, 2.0, 3.0}, {-1.0, 2.0, -3.0}, {0.5, 0.5, 0.5},
            {2.0, 2.0, 2.0}, {0.0, 1.0, 2.0}
        };
    }
};

// Test case for 1D → 3D functions
struct TestCase1Dto3D {
    std::string name;
    std::function<MultiOutput(fdouble)> func;
    std::function<std::array<double, 3>(double)> native_func;
    std::vector<double> test_inputs;
    double tolerance;
    
    TestCase1Dto3D(const std::string& n,
                   std::function<MultiOutput(fdouble)> f,
                   std::function<std::array<double, 3>(double)> nf,
                   double tol = 1e-10)
        : name(n), func(f), native_func(nf), tolerance(tol) {
        test_inputs = {0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0};
    }
};

// Test case for 10D → 1D functions
struct TestCase10Dto1D {
    std::string name;
    std::function<fdouble(const std::vector<fdouble>&)> func;
    std::function<double(const std::array<double, 10>&)> native_func;
    std::vector<std::array<double, 10>> test_inputs;
    double tolerance;
    
    TestCase10Dto1D(const std::string& n,
                    std::function<fdouble(const std::vector<fdouble>&)> f,
                    std::function<double(const std::array<double, 10>&)> nf,
                    double tol = 1e-10)
        : name(n), func(f), native_func(nf), tolerance(tol) {
        test_inputs = {
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
        };
    }
};

// Get all multi-IO test cases
inline std::vector<TestCase2Dto2D> getAllTestCases2Dto2D() {
    return {
        TestCase2Dto2D("PolarToCartesianLike", TestFunctionsMultiIO::polar_to_cartesian_like,
                       TestFunctionsMultiIO::polar_to_cartesian_like_native),
        TestCase2Dto2D("SumAndProduct", TestFunctionsMultiIO::sum_and_product,
                       TestFunctionsMultiIO::sum_and_product_native),
        TestCase2Dto2D("LinearTransform2D", TestFunctionsMultiIO::linear_transform_2d,
                       TestFunctionsMultiIO::linear_transform_2d_native),
    };
}

inline std::vector<TestCase3Dto1D> getAllTestCases3Dto1D() {
    return {
        TestCase3Dto1D("VolumeLike", TestFunctionsMultiIO::volume_like,
                       TestFunctionsMultiIO::volume_like_native),
        TestCase3Dto1D("WeightedSum3D", TestFunctionsMultiIO::weighted_sum_3d,
                       TestFunctionsMultiIO::weighted_sum_3d_native),
        TestCase3Dto1D("Polynomial3D", TestFunctionsMultiIO::polynomial_3d,
                       TestFunctionsMultiIO::polynomial_3d_native),
    };
}

inline std::vector<TestCase1Dto3D> getAllTestCases1Dto3D() {
    return {
        TestCase1Dto3D("PolynomialDerivatives", TestFunctionsMultiIO::polynomial_derivatives,
                       TestFunctionsMultiIO::polynomial_derivatives_native),
        TestCase1Dto3D("TrigLikeExpansion", TestFunctionsMultiIO::trig_like_expansion,
                       TestFunctionsMultiIO::trig_like_expansion_native),
    };
}

inline std::vector<TestCase10Dto1D> getAllTestCases10Dto1D() {
    return {
        TestCase10Dto1D("Sum10D", TestFunctionsMultiIO::sum_10d,
                        TestFunctionsMultiIO::sum_10d_native),
        TestCase10Dto1D("WeightedSum10D", TestFunctionsMultiIO::weighted_sum_10d,
                        TestFunctionsMultiIO::weighted_sum_10d_native),
    };
}

} // namespace testing
} // namespace forge