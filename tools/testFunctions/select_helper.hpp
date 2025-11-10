#pragma once

#include <type_traits>
#include "../../tools/types/fbool.hpp"

namespace forge {
namespace tools {
namespace test_functions {

// Helper function for unified conditional selection
// Works with both bool (for double) and fbool (for fdouble)
template<typename BoolType, typename T>
inline T select(BoolType cond, T true_val, T false_val) {
    if constexpr (std::is_same_v<BoolType, bool>) {
        // Native bool: use ternary operator
        return cond ? true_val : false_val;
    } else {
        // fbool: use If method to preserve tape
        return cond.If(true_val, false_val);
    }
}

// Helper for nested selections (convenience function)
template<typename BoolType1, typename BoolType2, typename T>
inline T select2(BoolType1 cond1, T val1, 
                 BoolType2 cond2, T val2, 
                 T default_val) {
    return select(cond1, val1, 
                  select(cond2, val2, default_val));
}

// Helper for min using select (works with comparisons)
template<typename T>
inline T select_min(T a, T b) {
    auto cond = (a < b);
    return select(cond, a, b);
}

// Helper for max using select (works with comparisons)
template<typename T>
inline T select_max(T a, T b) {
    auto cond = (a > b);
    return select(cond, a, b);
}

// Helper for clamping using select
template<typename T>
inline T select_clamp(T x, T min_val, T max_val) {
    auto too_small = (x < min_val);
    auto too_large = (x > max_val);
    return select(too_small, min_val,
                  select(too_large, max_val, x));
}

// Helper for absolute value using select
template<typename T>
inline T select_abs(T x) {
    auto is_negative = (x < T(0));
    return select(is_negative, -x, x);
}

} // namespace test_functions
} // namespace tools
} // namespace forge