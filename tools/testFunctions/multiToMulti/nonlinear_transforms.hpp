#pragma once

#include <vector>
#include <cmath>
#include <type_traits>

namespace forge {
namespace tools {
namespace test_functions {
namespace multi_to_multi {

template<typename T>
std::vector<T> polarToCartesian(const std::vector<T>& inputs) {
    T r = inputs[0];
    T theta = inputs[1];
    return {
        r * std::cos(theta),
        r * std::sin(theta)
    };
}


template<typename T>
std::vector<T> sphericalToCartesian(const std::vector<T>& inputs) {
    T r = inputs[0];
    T theta = inputs[1];
    T phi = inputs[2];
    return {
        r * std::sin(theta) * std::cos(phi),
        r * std::sin(theta) * std::sin(phi),
        r * std::cos(theta)
    };
}

template<typename T>
std::vector<T> nonlinearSystem2x2(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    return {
        x * x - y * y,
        T(2) * x * y
    };
}

template<typename T>
std::vector<T> trigonometricTransform2x3(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    return {
        std::sin(x) * std::cos(y),
        std::cos(x) * std::sin(y),
        std::sin(x + y)
    };
}

template<typename T>
std::vector<T> exponentialTransform2x2(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    return {
        std::exp(x + y),
        std::exp(x - y)
    };
}

template<typename T>
std::vector<T> logarithmicTransform2x2(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    // Use conditional operator for compatibility with both double and fdouble
    T x_safe = x;
    T y_safe = y;
    using namespace forge;
    if constexpr (std::is_same_v<T, double>) {
        x_safe = std::max(x, T(0.01));
        y_safe = std::max(y, T(0.01));
    } else {
        x_safe = max(x, T(0.01));
        y_safe = max(y, T(0.01));
    }
    return {
        std::log(x_safe),
        std::log(y_safe)
    };
}

template<typename T>
std::vector<T> polynomialTransform2x3(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    return {
        x * x + y * y,
        x * x - y * y,
        x * y
    };
}

template<typename T>
std::vector<T> rationalTransform2x2(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    T denom1 = T(1) + x * x;
    T denom2 = T(1) + y * y;
    return {
        x / denom1,
        y / denom2
    };
}

template<typename T>
std::vector<T> mixedTransform3x4(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    T z = inputs[2];
    return {
        x + y + z,
        x * y * z,
        std::exp(-x * x),
        std::sin(y) * std::cos(z)
    };
}

template<typename T>
std::vector<T> normalizationTransform3x3(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    T z = inputs[2];
    T norm = std::sqrt(x * x + y * y + z * z);
    T safe_norm = norm;
    using namespace forge;
    if constexpr (std::is_same_v<T, double>) {
        safe_norm = std::max(norm, T(1e-10));
    } else {
        safe_norm = max(norm, T(1e-10));
    }
    return {
        x / safe_norm,
        y / safe_norm,
        z / safe_norm
    };
}

template<typename T>
std::vector<T> sigmoidTransform2x2(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    return {
        T(1) / (T(1) + std::exp(-x)),
        T(1) / (T(1) + std::exp(-y))
    };
}


template<typename T>
std::vector<T> softmaxTransform3x3(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    T z = inputs[2];
    
    // Use appropriate max function based on type
    T max_xy;
    T max_val;
    using namespace forge;
    if constexpr (std::is_same_v<T, double>) {
        max_xy = std::max(x, y);
        max_val = std::max(max_xy, z);
    } else {
        max_xy = max(x, y);
        max_val = max(max_xy, z);
    }
    
    T exp_x = std::exp(x - max_val);
    T exp_y = std::exp(y - max_val);
    T exp_z = std::exp(z - max_val);
    T sum = exp_x + exp_y + exp_z;
    
    return {
        exp_x / sum,
        exp_y / sum,
        exp_z / sum
    };
}

inline std::vector<std::vector<double>> getPolarToCartesianInputs() {
    return {
        {1.0, 0.0},
        {1.0, 1.5708},
        {2.0, 3.14159},
        {0.5, 0.785398},
        {3.0, -0.785398}
    };
}


inline std::vector<std::vector<double>> getSphericalToCartesianInputs() {
    return {
        {1.0, 0.0, 0.0},
        {1.0, 1.5708, 0.0},
        {2.0, 0.785398, 0.785398},
        {1.0, 1.5708, 1.5708},
        {3.0, 1.0, 2.0}
    };
}

inline std::vector<std::vector<double>> getNonlinearSystem2x2Inputs() {
    return {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {2.0, 1.0},
        {-1.0, 2.0}
    };
}

inline std::vector<std::vector<double>> getTrigonometricTransform2x3Inputs() {
    return {
        {0.0, 0.0},
        {1.5708, 0.0},
        {0.0, 1.5708},
        {0.785398, 0.785398},
        {-0.785398, 0.785398}
    };
}

inline std::vector<std::vector<double>> getExponentialTransform2x2Inputs() {
    return {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {0.5, 0.5},
        {-0.5, 0.5}
    };
}

inline std::vector<std::vector<double>> getLogarithmicTransform2x2Inputs() {
    return {
        {1.0, 1.0},
        {2.718, 2.718},
        {0.5, 2.0},
        {10.0, 0.1},
        {0.02, 100.0}  // Changed from 0.01 to avoid non-differentiable boundary
    };
}

inline std::vector<std::vector<double>> getPolynomialTransform2x3Inputs() {
    return {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {2.0, -1.0}
    };
}

inline std::vector<std::vector<double>> getRationalTransform2x2Inputs() {
    return {
        {0.0, 0.0},
        {1.0, 1.0},
        {-1.0, 2.0},
        {3.0, -3.0},
        {0.5, 0.5}
    };
}

inline std::vector<std::vector<double>> getMixedTransform3x4Inputs() {
    return {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {1.0, 1.0, 1.0},
        {-0.5, 0.5, 1.0}
    };
}

inline std::vector<std::vector<double>> getNormalizationTransform3x3Inputs() {
    return {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 1.0, 1.0},
        {3.0, 4.0, 0.0}
    };
}

inline std::vector<std::vector<double>> getSigmoidTransform2x2Inputs() {
    return {
        {0.0, 0.0},
        {1.0, -1.0},
        {2.0, -2.0},
        {-3.0, 3.0},
        {5.0, -5.0}
    };
}


inline std::vector<std::vector<double>> getSoftmaxTransform3x3Inputs() {
    return {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 2.0, 3.0},
        {-1.0, 0.0, 1.0},
        {2.0, 2.0, 2.0}
    };
}

} // namespace multi_to_multi
} // namespace test_functions
} // namespace tools
} // namespace forge