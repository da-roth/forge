#pragma once

#include <vector>
#include <cmath>

namespace forge {
namespace tools {
namespace test_functions {
namespace multi_to_multi {

template<typename T>
std::vector<T> linearTransform2x3(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    return {
        x + T(2) * y,
        T(3) * x - y,
        x * y
    };
}

template<typename T>
std::vector<T> matrixMultiply3x2(const std::vector<T>& inputs) {
    T a = inputs[0];
    T b = inputs[1];
    T c = inputs[2];
    return {
        T(2) * a + T(3) * b - c,
        a - T(2) * b + T(4) * c
    };
}

template<typename T>
std::vector<T> identityTransform3x3(const std::vector<T>& inputs) {
    return inputs;
}

template<typename T>
std::vector<T> scalarMultiply2x2(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    T scalar = T(3.14159);
    return {
        scalar * x,
        scalar * y
    };
}

template<typename T>
std::vector<T> crossProduct3x3(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1]; 
    T z = inputs[2];
    T vx = T(1.0);
    T vy = T(2.0);
    T vz = T(3.0);
    return {
        y * vz - z * vy,
        z * vx - x * vz,
        x * vy - y * vx
    };
}

template<typename T>
std::vector<T> affineTransform2x2(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    return {
        T(2) * x + T(3) * y + T(5),
        T(-1) * x + T(4) * y - T(2)
    };
}

template<typename T>
std::vector<T> quadraticForm2x1(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    return {
        x * x + T(2) * x * y + T(3) * y * y
    };
}

template<typename T>
std::vector<T> expansionMap1x3(const std::vector<T>& inputs) {
    T x = inputs[0];
    return {
        x,
        x * x,
        x * x * x
    };
}

template<typename T>
std::vector<T> projectionMap4x2(const std::vector<T>& inputs) {
    T w = inputs[0];
    T x = inputs[1];
    T y = inputs[2];
    T z = inputs[3];
    return {
        w + x,
        y + z
    };
}

template<typename T>
std::vector<T> rotationTransform2x2(const std::vector<T>& inputs) {
    T x = inputs[0];
    T y = inputs[1];
    T theta = T(0.785398);
    T cos_theta = std::cos(theta);
    T sin_theta = std::sin(theta);
    return {
        x * cos_theta - y * sin_theta,
        x * sin_theta + y * cos_theta
    };
}

inline std::vector<std::vector<double>> getLinearTransform2x3Inputs() {
    return {
        {1.0, 2.0},
        {-1.0, 0.5},
        {0.0, 0.0},
        {3.14, 2.71},
        {-2.5, 1.5}
    };
}

inline std::vector<std::vector<double>> getMatrixMultiply3x2Inputs() {
    return {
        {1.0, 2.0, 3.0},
        {-1.0, 0.0, 1.0},
        {0.5, -0.5, 0.5},
        {2.0, 2.0, 2.0},
        {0.0, 0.0, 0.0}
    };
}

inline std::vector<std::vector<double>> getIdentityTransform3x3Inputs() {
    return {
        {1.0, 2.0, 3.0},
        {-1.0, -2.0, -3.0},
        {0.0, 0.0, 0.0},
        {3.14, 2.71, 1.41}
    };
}

inline std::vector<std::vector<double>> getScalarMultiply2x2Inputs() {
    return {
        {1.0, 1.0},
        {2.0, -2.0},
        {0.0, 0.0},
        {-1.5, 1.5}
    };
}

inline std::vector<std::vector<double>> getCrossProduct3x3Inputs() {
    return {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 2.0, 3.0},
        {-1.0, -2.0, -3.0}
    };
}

inline std::vector<std::vector<double>> getAffineTransform2x2Inputs() {
    return {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {-1.0, 2.0}
    };
}

inline std::vector<std::vector<double>> getQuadraticForm2x1Inputs() {
    return {
        {1.0, 1.0},
        {2.0, 0.0},
        {0.0, 2.0},
        {-1.0, 1.0},
        {0.5, 0.5}
    };
}

inline std::vector<std::vector<double>> getExpansionMap1x3Inputs() {
    return {
        {0.0},
        {1.0},
        {-1.0},
        {2.0},
        {0.5}
    };
}

inline std::vector<std::vector<double>> getProjectionMap4x2Inputs() {
    return {
        {1.0, 2.0, 3.0, 4.0},
        {0.0, 0.0, 0.0, 0.0},
        {-1.0, -2.0, -3.0, -4.0},
        {1.0, -1.0, 1.0, -1.0},
        {0.5, 0.5, 0.5, 0.5}
    };
}

inline std::vector<std::vector<double>> getRotationTransform2x2Inputs() {
    return {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {-1.0, 1.0},
        {2.0, -0.5}
    };
}

} // namespace multi_to_multi
} // namespace test_functions
} // namespace tools
} // namespace forge