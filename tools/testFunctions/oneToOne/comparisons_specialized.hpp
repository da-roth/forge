#pragma once

#include "../../types/fdouble.hpp"
#include "../../types/fbool.hpp"

namespace forge {
namespace tools {
namespace test_functions {
namespace one_to_one {

// ========== SPECIALIZED COMPARISON FUNCTIONS ==========
// These test pure comparison operations, returning bool/fbool directly
// No conversion to double values - just the comparison result

// --- Less Than ---
inline bool justLessThan_double(double x) {
    return x < 1.0;
}

inline forge::fbool justLessThan_doubleTP(forge::fdouble x) {
    return x < forge::fdouble(1.0);
}

// --- Less Than or Equal ---
inline bool justLessEqual_double(double x) {
    return x <= 1.0;
}

inline forge::fbool justLessEqual_doubleTP(forge::fdouble x) {
    return x <= forge::fdouble(1.0);
}

// --- Greater Than ---
inline bool justGreaterThan_double(double x) {
    return x > 1.0;
}

inline forge::fbool justGreaterThan_doubleTP(forge::fdouble x) {
    return x > forge::fdouble(1.0);
}

// --- Greater Than or Equal ---
inline bool justGreaterEqual_double(double x) {
    return x >= 1.0;
}

inline forge::fbool justGreaterEqual_doubleTP(forge::fdouble x) {
    return x >= forge::fdouble(1.0);
}

// --- Equal ---
inline bool justEqual_double(double x) {
    return x == 1.0;
}

inline forge::fbool justEqual_doubleTP(forge::fdouble x) {
    return x == forge::fdouble(1.0);
}

// --- Not Equal ---
inline bool justNotEqual_double(double x) {
    return x != 1.0;
}

inline forge::fbool justNotEqual_doubleTP(forge::fdouble x) {
    return x != forge::fdouble(1.0);
}

} // namespace one_to_one
} // namespace test_functions
} // namespace tools
} // namespace forge