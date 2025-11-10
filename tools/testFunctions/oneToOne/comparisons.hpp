#pragma once

#include <cmath>
#include <vector>
#include <cstdio>
#include <type_traits>
#include "../select_helper.hpp"

namespace forge {
namespace tools {
namespace test_functions {
namespace one_to_one {

using namespace forge::tools::test_functions;

// Simple sign function using select
template<typename T>
T signFunc(T x) {
    auto is_negative = (x < T(0));
    auto is_positive = (x > T(0));
    return select(is_negative, T(-1), 
                  select(is_positive, T(1), T(0)));
}

// Piecewise linear function
template<typename T>
T piecewiseLinear(T x) {
    // f(x) = { -2x    if x < -1
    //        { x+1    if -1 <= x < 1
    //        { 2      if x >= 1
    auto cond1 = (x < T(-1));
    auto cond2 = (x < T(1));
    
    return select(cond1, 
                  T(-2) * x,                    // x < -1
                  select(cond2,
                         x + T(1),               // -1 <= x < 1
                         T(2)));                 // x >= 1
}

// Piecewise quadratic function
template<typename T>
T piecewiseQuadratic(T x) {
    // f(x) = { x^2        if x < 0
    //        { 2x         if 0 <= x < 2
    //        { x^2 - 4    if x >= 2
    auto cond1 = (x < T(0));
    auto cond2 = (x < T(2));
    
    return select(cond1,
                  x * x,                         // x < 0
                  select(cond2,
                         T(2) * x,               // 0 <= x < 2
                         x * x - T(4)));         // x >= 2
}

// Ramp function (ReLU-like)
template<typename T>
T rampFunction(T x) {
    auto is_positive = (x > T(0));
    return select(is_positive, x, T(0));
}

// Leaky ReLU
template<typename T>
T leakyReLU(T x, T alpha = T(0.1)) {
    auto is_positive = (x >= T(0));
    return select(is_positive, x, alpha * x);
}

// Soft clipping function
template<typename T>
T softClip(T x, T lower = T(-1), T upper = T(1)) {
    auto too_small = (x < lower);
    auto too_large = (x > upper);
    
    return select(too_small, 
                  lower + (x - lower) * T(0.1),    // Soft lower bound
                  select(too_large,
                         upper + (x - upper) * T(0.1),  // Soft upper bound
                         x));                            // Within bounds
}

// Min of three values
template<typename T>
T min3(T x) {
    T a = x;
    T b = x * x;
    T c = T(2) - x;
    
    // min(a, min(b, c))
    return select_min(a, select_min(b, c));
}

// Max of three values  
template<typename T>
T max3(T x) {
    T a = x;
    T b = x * x;
    T c = T(2) - x;
    
    // max(a, max(b, c))
    return select_max(a, select_max(b, c));
}

// Median of three values
template<typename T>
T median3(T x) {
    T a = x;
    T b = x * x;
    T c = T(2) - x;
    
    // median = max(min(a,b), min(max(a,b), c))
    T min_ab = select_min(a, b);
    T max_ab = select_max(a, b);
    T min_max_ab_c = select_min(max_ab, c);
    return select_max(min_ab, min_max_ab_c);
}

// Absolute difference
template<typename T>
T absDiff(T x) {
    T diff = x - T(1);
    return select_abs(diff);
}

// Triangle wave using comparisons
template<typename T>
T triangleWave(T x) {
    // Period of 4: rises from 0 to 2, then falls back to 0
    T period = T(4);
    T half_period = T(2);
    
    // Normalize to [0, 4)
    T normalized = x - std::floor(x / period) * period;
    
    auto in_first_half = (normalized < half_period);
    return select(in_first_half,
                  normalized,                    // Rising: 0 to 2
                  period - normalized);          // Falling: 2 to 0
}

// Step function approximation
template<typename T>
T stepFunction(T x, T threshold = T(0)) {
    auto above_threshold = (x >= threshold);
    return select(above_threshold, T(1), T(0));
}

// Double step (staircase)
template<typename T>
T doubleStep(T x) {
    auto cond1 = (x < T(-1));
    auto cond2 = (x < T(0));
    auto cond3 = (x < T(1));
    
    return select(cond1, T(0),
                  select(cond2, T(1),
                         select(cond3, T(2), T(3))));
}

// Saturating linear function
template<typename T>
T saturatingLinear(T x, T slope = T(2), T lower_sat = T(-2), T upper_sat = T(2)) {
    T linear_part = slope * x;
    auto too_low = (linear_part < lower_sat);
    auto too_high = (linear_part > upper_sat);
    
    return select(too_low, lower_sat,
                  select(too_high, upper_sat, linear_part));
}

// Dead zone function
template<typename T>
T deadZone(T x, T threshold = T(0.5)) {
    auto in_dead_zone = (select_abs(x) < threshold);
    return select(in_dead_zone, T(0), x);
}

// Hysteresis-like function (simplified)
template<typename T>
T hysteresisLike(T x) {
    // Different thresholds for positive and negative
    auto large_positive = (x > T(1));
    auto large_negative = (x < T(-2));
    
    return select(large_positive, T(1),
                  select(large_negative, T(-1), T(0)));
}

// Complex piecewise function
template<typename T>
T complexPiecewise(T x) {
    // f(x) = { -x^2      if x < -2
    //        { 2x + 4    if -2 <= x < 0
    //        { 4         if 0 <= x < 1
    //        { 5 - x     if 1 <= x < 3
    //        { x^2 - 7   if x >= 3
    
    auto c1 = (x < T(-2));
    auto c2 = (x < T(0));
    auto c3 = (x < T(1));
    auto c4 = (x < T(3));
    
    return select(c1, -x * x,                    // x < -2
                  select(c2, T(2) * x + T(4),    // -2 <= x < 0
                         select(c3, T(4),         // 0 <= x < 1
                                select(c4, T(5) - x,    // 1 <= x < 3
                                       x * x - T(7)))));  // x >= 3
}

// ========== DIAGNOSTIC FUNCTIONS ==========
// Simple tests to isolate comparison/select issues

// Test 1: Single comparison and select
template<typename T>
T diagnosticSimpleSelect(T x) {
    // Should return 4.0 for x >= 0, and 2.0 for x < 0
    auto cond = (x < T(0));
    return select(cond, T(2), T(4));
}

// Test 2: Check if comparison itself works
template<typename T>
T diagnosticComparisonOnly(T x) {
    // Returns 1.0 if x < 0.5, else 0.0
    auto cond = (x < T(0.5));
    return select(cond, T(1), T(0));
}

// Test 3: Two-level nested select (simpler than complex piecewise)
template<typename T>
T diagnosticNestedSelect(T x) {
    // { 1 if x < 0
    // { 2 if 0 <= x < 1  
    // { 3 if x >= 1
    auto c1 = (x < T(0));
    auto c2 = (x < T(1));
    
    return select(c1, T(1),
                  select(c2, T(2), T(3)));
}

// Test 4: Direct implementation of the failing segment
template<typename T>
T diagnosticFailingSegment(T x) {
    // Just test: return 4 if 0 <= x < 1, else 0
    auto c1 = (x < T(0));
    auto c2 = (x < T(1));
    
    // If x >= 0 AND x < 1, return 4
    return select(c1, T(0),           // x < 0: return 0
                  select(c2, T(4),     // x >= 0 && x < 1: return 4
                         T(0)));       // x >= 1: return 0
}

// Test 5: Check multiple conditions separately
template<typename T>
T diagnosticConditionValues(T x) {
    // Encode each condition as a digit to see what's happening
    // Returns: abc.def where each digit represents a condition
    auto c1 = (x < T(-2));
    auto c2 = (x < T(0));
    auto c3 = (x < T(1));
    auto c4 = (x < T(3));
    
    T result = T(0);
    result += select(c1, T(100), T(0));    // 100s place if x < -2
    result += select(c2, T(10), T(0));     // 10s place if x < 0
    result += select(c3, T(1), T(0));      // 1s place if x < 1
    result += select(c4, T(0.1), T(0));    // 0.1s place if x < 3
    
    return result;
}

// Test 6: Simplest possible nested case that should work like complexPiecewise at x=0.5
template<typename T>
T diagnosticMinimalNesting(T x) {
    // Exactly mimics the structure for x=0.5 case in complexPiecewise
    auto c2 = (x < T(0));   // Should be false for x=0.5
    auto c3 = (x < T(1));   // Should be true for x=0.5
    
    // This should return 4 for x=0.5
    return select(c2, T(999),    // if x < 0 (false for 0.5)
                  select(c3, T(4), T(777)));  // else if x < 1 (true for 0.5) -> return 4
}

// Test 7: Check If method directly without select wrapper
template<typename T>
T diagnosticDirectIf(T x) {
    if constexpr (std::is_same_v<T, double>) {
        return (x < 0.5) ? T(4) : T(2);
    } else {
        auto cond = (x < T(0.5));
        return cond.If(T(4), T(2));
    }
}

// Test 8: Three-level nesting (one level deeper than diagnosticNestedSelect)
template<typename T>
T diagnosticThreeLevelNesting(T x) {
    // Add one more level of nesting
    auto c1 = (x < T(-1));
    auto c2 = (x < T(0));
    auto c3 = (x < T(1));
    
    return select(c1, T(1),
                  select(c2, T(2),
                         select(c3, T(3), T(4))));
}

// Test 9: Four-level nesting (same depth as complexPiecewise)
template<typename T>
T diagnosticFourLevelNesting(T x) {
    // Same nesting depth as complexPiecewise but simpler values
    auto c1 = (x < T(-2));
    auto c2 = (x < T(0));
    auto c3 = (x < T(1));
    auto c4 = (x < T(3));
    
    return select(c1, T(10),
                  select(c2, T(20),
                         select(c3, T(30),
                                select(c4, T(40), T(50)))));
}

// Test 10: Test with exact same constants as complexPiecewise
template<typename T>
T diagnosticExactConstants(T x) {
    // For x=0.5, should return 4
    auto c2 = (x < T(0));   // false
    auto c3 = (x < T(1));   // true
    
    return select(c2, T(999),
                  select(c3, T(4), T(888)));  // Using exact T(4) constant
}

// Test 11: Test if the issue is with expression evaluation
template<typename T>
T diagnosticWithExpressions(T x) {
    // Use expressions instead of constants
    auto c1 = (x < T(0));
    auto c2 = (x < T(1));
    
    T expr1 = T(2) * x + T(4);  // Like in complexPiecewise
    T expr2 = T(5) - x;          // Like in complexPiecewise
    
    return select(c1, expr1,
                  select(c2, T(4), expr2));
}

// Test 12: Exact reproduction of complexPiecewise but with debug values
template<typename T>
T diagnosticComplexDebug(T x) {
    auto c1 = (x < T(-2));
    auto c2 = (x < T(0));
    auto c3 = (x < T(1));
    auto c4 = (x < T(3));
    
    // Use unique values to see which branch is taken
    return select(c1, T(111),        // x < -2
                  select(c2, T(222),  // -2 <= x < 0
                         select(c3, T(333),  // 0 <= x < 1 (should be this for x=0.5)
                                select(c4, T(444), T(555)))));  // 1 <= x < 3 or x >= 3
}

// Test 12: Exact reproduction of complexPiecewise but with debug values
template<typename T>
T diagnosticComplexDebug2(T x) {
    auto c1 = (x < T(-2));
    auto c2 = (x < T(0));
    auto c3 = (x < T(1));
    auto c4 = (x < T(3));
    
    // Use unique values to see which branch is taken
    return      select(c2, T(222),  // -2 <= x < 0
                         select(c3, T(333),  // 0 <= x < 1 (should be this for x=0.5)
                                select(c4, T(444), T(555))));  // 1 <= x < 3 or x >= 3
}

// Test 12: Exact reproduction of complexPiecewise but with debug values
template<typename T>
T diagnosticComplexDebug3(T x) {
    auto c1 = (x < T(-2));
    auto c2 = (x < T(0));
    auto c3 = (x < T(1));
    auto c4 = (x < T(3));
    
    // Use unique values to see which branch is taken
    return       select(c3, T(333),  // 0 <= x < 1 (should be this for x=0.5)
                                select(c4, T(444), T(555)));  // 1 <= x < 3 or x >= 3
}


// Test 13: Test if zero is coming from somewhere specific
template<typename T>
T diagnosticZeroSource(T x) {
    // Try to find where the zero is coming from
    auto c1 = (x < T(0));
    auto c2 = (x < T(1));
    
    // Use non-zero values everywhere
    return select(c1, T(100) + x * T(0),     // Force zero dependency
                  select(c2, T(4) + x * T(0), // Force x dependency but result is 4
                         T(200) + x * T(0)));  // Force zero dependency
}

// Test 14: Build up complexity gradually - start with complexPiecewise structure for negative x
template<typename T>
T diagnosticComplexNegativeOnly(T x) {
    // Only handle x < 0 part of complexPiecewise
    auto c1 = (x < T(-2));
    auto c2 = (x < T(0));
    
    return select(c1, -x * x,                    // x < -2
                  select(c2, T(2) * x + T(4),    // -2 <= x < 0
                         T(999)));                // x >= 0 (placeholder)
}

// Test 15: Build up complexity - add one more level
template<typename T>
T diagnosticComplexFirstThree(T x) {
    // Handle first three segments
    auto c1 = (x < T(-2));
    auto c2 = (x < T(0));
    auto c3 = (x < T(1));
    
    return select(c1, -x * x,                    // x < -2
                  select(c2, T(2) * x + T(4),    // -2 <= x < 0
                         select(c3, T(4),         // 0 <= x < 1
                                T(999))));        // x >= 1 (placeholder)
}

// Test 16: Exact copy of complexPiecewise to verify it's not a typo
template<typename T>
T diagnosticExactCopy(T x) {
    auto c1 = (x < T(-2));
    auto c2 = (x < T(0));
    auto c3 = (x < T(1));
    auto c4 = (x < T(3));
    
    return select(c1, -x * x,
                  select(c2, T(2) * x + T(4),
                         select(c3, T(4),
                                select(c4, T(5) - x,
                                       x * x - T(7)))));
}

// Test input sets for comparison functions
inline std::vector<double> getComparisonInputs() {
    return {-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3};
}

// Extended inputs for piecewise functions
inline std::vector<double> getPiecewiseInputs() {
    return {-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3};
}

// Simple test to isolate the conditional bug
template<typename T>
T simpleConditionalTest(T x) {
    // This should be similar to what American options do but simpler
    auto cond1 = (x > T(0));
    auto cond2 = (x > T(1));
    
    // Simple conditional like ramp function (this works)
    T result1 = select(cond1, x, T(0));
    
    // More complex conditional operations like in American options
    T value_a = T(2) * x + T(1);  // Some computation
    T value_b = x * x;           // Another computation 
    T result2 = select(cond2, value_a, value_b);
    
    // Combine results
    return result1 + result2;
}

// Test with vector-like operations (closer to American options)
template<typename T>
T vectorLikeConditionalTest(T x) {
    // Simulate what American options do with vector operations
    T values[3];
    values[0] = T(1) * x;
    values[1] = T(2) * x; 
    values[2] = T(3) * x;
    
    // Apply conditional selection on computed values
    auto cond = (x > T(0.5));
    T selected = select(cond, values[1], values[0]);
    
    return selected + values[2];
}

// Test with fixed-size array operations (avoiding std::vector issues)
template<typename T>
T stdVectorTest(T x) {
    // Using fixed-size array instead of std::vector to avoid vectorization issues
    T values[3];
    
    // Fill array with computations
    for (int i = 0; i < 3; ++i) {
        values[i] = T(i + 1) * x;
    }
    
    // Apply conditional selection on array elements
    auto cond = (x > T(0.5));
    T selected = select(cond, values[1], values[0]);
    
    return selected + values[2];
}

// Simplified test without arrays - unrolled version
template<typename T>
T vectorizedMaxIssue(T x) {
    // Completely unrolled version without arrays to avoid vectorization issues
    // Initial payoffs at terminal nodes
    T val0 = T(100) - x * T(0.8);    // For x=100: 100 - 80 = 20
    T val1 = T(100) - x * T(0.96);   // For x=100: 100 - 96 = 4
    T val2 = T(100) - x * T(1.0);    // For x=100: 100 - 100 = 0
    T val3 = T(100) - x * T(1.2);    // For x=100: 100 - 120 = -20
    
    // Apply max(0, value) to each
    auto is_pos0 = (val0 > T(0));
    val0 = select(is_pos0, val0, T(0));
    
    auto is_pos1 = (val1 > T(0));
    val1 = select(is_pos1, val1, T(0));
    
    auto is_pos2 = (val2 > T(0));
    val2 = select(is_pos2, val2, T(0));
    
    auto is_pos3 = (val3 > T(0));
    val3 = select(is_pos3, val3, T(0));
    
    // Backward iteration step i=2 (working with 3 nodes)
    // Node j=0: continuation from val0 and val1
    T cont20 = T(0.98) * (T(0.6) * val1 + T(0.4) * val0);
    T intr20 = T(100) - x * T(0.8);  // Same as initial val0
    auto is_pos_int20 = (intr20 > T(0));
    intr20 = select(is_pos_int20, intr20, T(0));
    auto should_ex20 = (intr20 > cont20);
    T new_val20 = select(should_ex20, intr20, cont20);
    
    // Node j=1: continuation from val1 and val2
    T cont21 = T(0.98) * (T(0.6) * val2 + T(0.4) * val1);
    T intr21 = T(100) - x * T(0.96);  // Same as initial val1
    auto is_pos_int21 = (intr21 > T(0));
    intr21 = select(is_pos_int21, intr21, T(0));
    auto should_ex21 = (intr21 > cont21);
    T new_val21 = select(should_ex21, intr21, cont21);
    
    // Node j=2: continuation from val2 and val3
    T cont22 = T(0.98) * (T(0.6) * val3 + T(0.4) * val2);
    T intr22 = T(100) - x * T(1.0);  // Same as initial val2
    auto is_pos_int22 = (intr22 > T(0));
    intr22 = select(is_pos_int22, intr22, T(0));
    auto should_ex22 = (intr22 > cont22);
    T new_val22 = select(should_ex22, intr22, cont22);
    
    // Backward iteration step i=1 (working with 2 nodes)
    // Node j=0: continuation from new_val20 and new_val21
    T cont10 = T(0.98) * (T(0.6) * new_val21 + T(0.4) * new_val20);
    T intr10 = T(100) - x * T(0.8);
    auto is_pos_int10 = (intr10 > T(0));
    intr10 = select(is_pos_int10, intr10, T(0));
    auto should_ex10 = (intr10 > cont10);
    T new_val10 = select(should_ex10, intr10, cont10);
    
    // Node j=1: continuation from new_val21 and new_val22
    T cont11 = T(0.98) * (T(0.6) * new_val22 + T(0.4) * new_val21);
    T intr11 = T(100) - x * T(0.96);
    auto is_pos_int11 = (intr11 > T(0));
    intr11 = select(is_pos_int11, intr11, T(0));
    auto should_ex11 = (intr11 > cont11);
    T new_val11 = select(should_ex11, intr11, cont11);
    
    // Backward iteration step i=0 (working with 1 node - the root)
    // Node j=0: continuation from new_val10 and new_val11
    T cont00 = T(0.98) * (T(0.6) * new_val11 + T(0.4) * new_val10);
    T intr00 = T(100) - x * T(0.8);
    auto is_pos_int00 = (intr00 > T(0));
    intr00 = select(is_pos_int00, intr00, T(0));
    auto should_ex00 = (intr00 > cont00);
    T final_val = select(should_ex00, intr00, cont00);
    
    return final_val;
}

// SelectDiagnostic Test 1: Array assignment WITHOUT select - pure arithmetic
template<typename T>
T selectDiagnosticArrayNoSelect(T x) {
    // Test if array assignment itself causes lane mixing
    // WITHOUT any select/conditional operations
    T values[4];
    
    // Initialize with simple arithmetic
    values[0] = x * T(1.0);
    values[1] = x * T(2.0);
    values[2] = x * T(3.0);
    values[3] = x * T(4.0);
    
    // Do some array-based computations without select
    for (int i = 2; i >= 0; --i) {
        // Simple arithmetic combination
        values[i] = values[i] * T(0.5) + values[i+1] * T(0.3);
    }
    
    // More assignments to test if lanes stay independent
    T temp = values[0];
    values[0] = values[1] + values[2];
    values[1] = temp * T(2.0);
    
    return values[0] + values[1];
}

// SelectDiagnostic Test 2: Select with divergent lanes - no arrays
template<typename T>
T selectDiagnosticLaneDivergence(T x) {
    // Test select operations where lanes should take different paths
    // WITHOUT using arrays
    
    // Create conditions that will be different for each lane
    // For x near 100: lane 0=100.0, lane 1=100.001, lane 2=100.002, lane 3=100.003
    
    // First select: threshold at 100.0005
    auto cond1 = (x > T(100.0005));
    T path1 = select(cond1, x * T(2.0), x * T(0.5));
    // Expected: lanes 0 gets x*0.5, lanes 1,2,3 get x*2.0
    
    // Second select: threshold at 100.0015  
    auto cond2 = (x > T(100.0015));
    T path2 = select(cond2, path1 + T(10.0), path1 - T(5.0));
    // Expected: lanes 0,1 get path1-5, lanes 2,3 get path1+10
    
    // Third select: threshold at 100.0025
    auto cond3 = (x > T(100.0025));
    T path3 = select(cond3, path2 * T(1.5), path2 * T(0.8));
    // Expected: lanes 0,1,2 get path2*0.8, lane 3 gets path2*1.5
    
    // Nested selects to really test lane independence
    auto cond4 = (path3 > T(100.0));
    auto cond5 = (path3 < T(200.0));
    T path4 = select(cond4, 
                    select(cond5, path3 + T(1.0), path3 - T(1.0)),
                    path3 * T(0.1));
    
    return path4;
}

// SelectDiagnostic Test 3: Arrays WITH select - the problematic combination
template<typename T>
T selectDiagnosticArrayWithSelect(T x) {
    // Combines arrays and select to reproduce the issue
    T values[4];
    
    // Initialize
    values[0] = x - T(80.0);   // ~20 for x=100
    values[1] = x - T(96.0);   // ~4 for x=100
    values[2] = x - T(100.0);  // ~0 for x=100
    values[3] = x - T(120.0);  // ~-20 for x=100
    
    // Log initial values for native version
    if constexpr (std::is_same_v<T, double>) {
        printf("[NATIVE] Initial: v[0]=%.3f, v[1]=%.3f, v[2]=%.3f, v[3]=%.3f\n", 
               values[0], values[1], values[2], values[3]);
    }
    
    // Apply select on array elements
    for (int i = 0; i < 4; ++i) {
        auto is_positive = (values[i] > T(0));
        values[i] = select(is_positive, values[i], T(0));
    }
    
    // Log after first select
    if constexpr (std::is_same_v<T, double>) {
        printf("[NATIVE] After first select: v[0]=%.3f, v[1]=%.3f, v[2]=%.3f, v[3]=%.3f\n", 
               values[0], values[1], values[2], values[3]);
    }
    
    // Now do backward iteration with selects (should show the issue)
    for (int i = 2; i >= 0; --i) {
        T combined = values[i] * T(0.6) + values[i+1] * T(0.4);
        auto should_replace = (combined > T(5.0));
        T old_val = values[i];
        values[i] = select(should_replace, combined, values[i] * T(1.1));
        
        if constexpr (std::is_same_v<T, double>) {
            printf("[NATIVE] Iteration i=%d: combined=%.3f, should_replace=%d, old_val=%.3f, new_val=%.3f\n", 
                   i, combined, should_replace ? 1 : 0, old_val, values[i]);
        }
    }
    
    // Log final result
    if constexpr (std::is_same_v<T, double>) {
        printf("[NATIVE] Final values: v[0]=%.3f, v[1]=%.3f, v[2]=%.3f, v[3]=%.3f\n", 
               values[0], values[1], values[2], values[3]);
        printf("[NATIVE] Returning: %.3f\n", values[0]);
    }
    
    return values[0];
}

// SelectDiagnostic Test 4: Simple chained select test
template<typename T>
T selectDiagnosticSimpleChained(T x) {
    // Very simple test: just chain two selects together
    // No arrays, no complex logic
    
    T val = x - T(95.0);  // For x=100: val=5
    
    // First select: threshold at 5.5
    auto cond1 = (val > T(5.5));
    T result1 = select(cond1, val * T(2.0), val * T(0.5));
    // For x near 100: some lanes will take *2, some *0.5
    
    // Second select using result of first
    auto cond2 = (result1 > T(4.0));
    T result2 = select(cond2, result1 + T(10.0), result1 - T(1.0));
    
    return result2;
}

// SelectDiagnostic Test 5: Select divergence WITHOUT arrays - just separate variables
template<typename T>
T selectDiagnosticDivergenceNoArray(T x) {
    // Test select operations with lane divergence but NO arrays at all
    // This should work correctly even with AVX2
    
    // Create values that will cause divergent select paths
    T val0 = x - T(80.0);   // ~20 for x=100
    T val1 = x - T(96.0);   // ~4 for x=100
    T val2 = x - T(100.0);  // ~0 for x=100
    T val3 = x - T(120.0);  // ~-20 for x=100
    
    // Apply select on each value
    auto is_pos0 = (val0 > T(0));
    val0 = select(is_pos0, val0, T(0));
    
    auto is_pos1 = (val1 > T(0));
    val1 = select(is_pos1, val1, T(0));
    
    auto is_pos2 = (val2 > T(0));
    val2 = select(is_pos2, val2, T(0));
    
    auto is_pos3 = (val3 > T(0));
    val3 = select(is_pos3, val3, T(0));
    
    // Now do the same backward iteration logic but with separate variables
    // Iteration i=2
    T combined2 = val2 * T(0.6) + val3 * T(0.4);
    auto should_replace2 = (combined2 > T(5.0));
    T new_val2 = select(should_replace2, combined2, val2 * T(1.1));
    
    // Iteration i=1
    T combined1 = val1 * T(0.6) + new_val2 * T(0.4);
    auto should_replace1 = (combined1 > T(5.0));
    T new_val1 = select(should_replace1, combined1, val1 * T(1.1));
    
    // Iteration i=0
    T combined0 = val0 * T(0.6) + new_val1 * T(0.4);
    auto should_replace0 = (combined0 > T(5.0));
    T new_val0 = select(should_replace0, combined0, val0 * T(1.1));
    
    return new_val0;
}

// Test the exact American option pattern: nested loops + vector reassignment
template<typename T>
T americanOptionPattern(T x) {
    // Replicate the exact problematic pattern from American options
    const int steps = 3;
    // Using fixed-size array instead of std::vector
    T values[4];  // steps + 1 = 4
    
    // Initialize vector (like terminal payoffs)
    for (int j = 0; j <= steps; ++j) {
        T S = x;
        // Apply some transformations (like up/down moves)
        for (int k = 0; k < j; ++k) {
            S = S * T(1.2);  // up factor
        }
        for (int k = 0; k < (steps - j); ++k) {
            S = S * T(0.8);  // down factor  
        }
        
        T payoff = T(100) - S;  // strike - spot
        auto is_positive = (payoff > T(0));
        values[j] = select(is_positive, payoff, T(0));
    }
    
    // Backward induction (the critical nested loop pattern!)
    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            // Get values from next time step
            T cont_up = values[j + 1];
            T cont_down = values[j];
            T continuation = T(0.98) * (T(0.6) * cont_up + T(0.4) * cont_down);
            
            // Calculate intrinsic value
            T S = x;
            for (int k = 0; k < j; ++k) {
                S = S * T(1.2);
            }
            for (int k = 0; k < (i - j); ++k) {
                S = S * T(0.8);
            }
            
            T intrinsic_val = T(100) - S;
            auto is_positive = (intrinsic_val > T(0));
            T intrinsic = select(is_positive, intrinsic_val, T(0));
            
            // Early exercise decision - THIS IS THE CRITICAL PART
            auto should_exercise = (intrinsic >= continuation);
            values[j] = select(should_exercise, intrinsic, continuation);
        }
    }
    
    return values[0];
}

// Test with EXACT transcendental functions like real American options
template<typename T>
T exactAmericanPattern(T spot) {
    // Use EXACTLY the same operations as the real American options
    const T strike = T(100.0);
    const T r = T(0.02);
    const T sigma = T(0.25);
    const T maturity = T(1.0);
    const int steps = 3;
    const T dt = maturity / T(steps);
    
    // Cox-Ross-Rubinstein parameters - EXACT same operations!
    T a = std::exp(sigma * std::sqrt(dt));
    T u = a;
    T d = T(1.0) / a;  // Division operation!
    T erdt = std::exp(r * dt);
    T p = (erdt - d) / (u - d);  // More division!
    T disc = T(1.0) / erdt;  // Another division!
    
    // Build terminal payoffs
    // Using fixed-size array instead of std::vector
    T values[4];  // steps + 1 = 4
    
    for (int j = 0; j <= steps; ++j) {
        T S = spot;
        // Apply up moves
        for (int k = 0; k < j; ++k) {
            S = S * u;  // Use computed u, not hardcoded
        }
        // Apply down moves  
        for (int k = 0; k < (steps - j); ++k) {
            S = S * d;  // Use computed d, not hardcoded
        }
        // Put payoff at terminal node
        T payoff = strike - S;
        auto is_positive = (payoff > T(0));
        values[j] = select(is_positive, payoff, T(0));
    }
    
    // Backward induction - EXACT same pattern
    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            // Calculate spot at this node
            T S = spot;
            for (int k = 0; k < j; ++k) {
                S = S * u;
            }
            for (int k = 0; k < (i - j); ++k) {
                S = S * d;
            }
            
            // Continuation value
            T cont_up = values[j + 1];
            T cont_down = values[j];
            T continuation = disc * (p * cont_up + (T(1.0) - p) * cont_down);
            
            // Intrinsic value
            T intrinsic_val = strike - S;
            auto is_positive = (intrinsic_val > T(0));
            T intrinsic = select(is_positive, intrinsic_val, T(0));
            
            // Early exercise decision
            auto should_exercise = (intrinsic >= continuation);
            values[j] = select(should_exercise, intrinsic, continuation);
        }
    }
    
    return values[0];
}

// Progressive tests to isolate the exact failing operation

// Test 1: Remove sqrt, keep exp
template<typename T>
T americanPatternNoSqrt(T spot) {
    const T strike = T(100.0);
    const T r = T(0.02);
    const T sigma = T(0.25);
    const T maturity = T(1.0);
    const int steps = 3;
    const T dt = maturity / T(steps);
    
    // Remove sqrt - use hardcoded value
    T a = std::exp(sigma * T(0.5773502691896));  // sqrt(1/3) ≈ 0.5773502691896
    T u = a;
    T d = T(1.0) / a;
    T erdt = std::exp(r * dt);
    T p = (erdt - d) / (u - d);
    T disc = T(1.0) / erdt;
    
    // Using fixed-size array instead of std::vector
    T values[4];  // steps + 1 = 4
    
    for (int j = 0; j <= steps; ++j) {
        T S = spot;
        for (int k = 0; k < j; ++k) {
            S = S * u;
        }
        for (int k = 0; k < (steps - j); ++k) {
            S = S * d;
        }
        T payoff = strike - S;
        auto is_positive = (payoff > T(0));
        values[j] = select(is_positive, payoff, T(0));
    }
    
    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            T S = spot;
            for (int k = 0; k < j; ++k) {
                S = S * u;
            }
            for (int k = 0; k < (i - j); ++k) {
                S = S * d;
            }
            
            T cont_up = values[j + 1];
            T cont_down = values[j];
            T continuation = disc * (p * cont_up + (T(1.0) - p) * cont_down);
            
            T intrinsic_val = strike - S;
            auto is_positive = (intrinsic_val > T(0));
            T intrinsic = select(is_positive, intrinsic_val, T(0));
            
            auto should_exercise = (intrinsic >= continuation);
            values[j] = select(should_exercise, intrinsic, continuation);
        }
    }
    
    return values[0];
}

// Test 2: Remove exp, keep sqrt  
template<typename T>
T americanPatternNoExp(T spot) {
    const T strike = T(100.0);
    const T r = T(0.02);
    const T sigma = T(0.25);
    const T maturity = T(1.0);
    const int steps = 3;
    const T dt = maturity / T(steps);
    
    // Remove exp - use hardcoded values
    T a = T(1.144122144);  // exp(0.25 * sqrt(1/3)) ≈ 1.144122144
    T u = a;
    T d = T(1.0) / a;
    T erdt = T(1.006711409);  // exp(0.02 * 1/3) ≈ 1.006711409
    T p = (erdt - d) / (u - d);
    T disc = T(1.0) / erdt;
    
    // Using fixed-size array instead of std::vector
    T values[4];  // steps + 1 = 4
    
    for (int j = 0; j <= steps; ++j) {
        T S = spot;
        for (int k = 0; k < j; ++k) {
            S = S * u;
        }
        for (int k = 0; k < (steps - j); ++k) {
            S = S * d;
        }
        T payoff = strike - S;
        auto is_positive = (payoff > T(0));
        values[j] = select(is_positive, payoff, T(0));
    }
    
    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            T S = spot;
            for (int k = 0; k < j; ++k) {
                S = S * u;
            }
            for (int k = 0; k < (i - j); ++k) {
                S = S * d;
            }
            
            T cont_up = values[j + 1];
            T cont_down = values[j];
            T continuation = disc * (p * cont_up + (T(1.0) - p) * cont_down);
            
            T intrinsic_val = strike - S;
            auto is_positive = (intrinsic_val > T(0));
            T intrinsic = select(is_positive, intrinsic_val, T(0));
            
            auto should_exercise = (intrinsic >= continuation);
            values[j] = select(should_exercise, intrinsic, continuation);
        }
    }
    
    return values[0];
}

// Test 3: Remove divisions, keep transcendentals
template<typename T>
T americanPatternNoDivision(T spot) {
    const T strike = T(100.0);
    const T r = T(0.02);
    const T sigma = T(0.25);
    const T maturity = T(1.0);
    const int steps = 3;
    const T dt = maturity / T(steps);
    
    T a = std::exp(sigma * std::sqrt(dt));
    T u = a;
    T d = T(0.87400575);  // Hardcode 1/a to avoid division
    T erdt = std::exp(r * dt);
    T p = T(0.6);  // Hardcode to avoid division
    T disc = T(0.993355);  // Hardcode 1/erdt to avoid division
    
    // Using fixed-size array instead of std::vector
    T values[4];  // steps + 1 = 4
    
    for (int j = 0; j <= steps; ++j) {
        T S = spot;
        for (int k = 0; k < j; ++k) {
            S = S * u;
        }
        for (int k = 0; k < (steps - j); ++k) {
            S = S * d;
        }
        T payoff = strike - S;
        auto is_positive = (payoff > T(0));
        values[j] = select(is_positive, payoff, T(0));
    }
    
    for (int i = steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            T S = spot;
            for (int k = 0; k < j; ++k) {
                S = S * u;
            }
            for (int k = 0; k < (i - j); ++k) {
                S = S * d;
            }
            
            T cont_up = values[j + 1];
            T cont_down = values[j];
            T continuation = disc * (p * cont_up + (T(1.0) - p) * cont_down);
            
            T intrinsic_val = strike - S;
            auto is_positive = (intrinsic_val > T(0));
            T intrinsic = select(is_positive, intrinsic_val, T(0));
            
            auto should_exercise = (intrinsic >= continuation);
            values[j] = select(should_exercise, intrinsic, continuation);
        }
    }
    
    return values[0];
}

// Test 4: Only transcendental functions in simple context (no loops/vectors)
template<typename T>
T simpleTranscendentalTest(T x) {
    T a = std::exp(T(0.25) * std::sqrt(T(0.333333)));
    T b = std::exp(T(0.02) * T(0.333333));
    T c = T(1.0) / a;
    T d = (b - c) / (a - c);
    return a + b + c + d + x;
}

// ===== Isolated exp() tests for debugging =====

// Test: Just exp
template<typename T>
T justExp(T x) {
    return std::exp(x);
}

// Test: Just addition
template<typename T>
T justAddition(T x) {
    return x + T(2.0);
}

// Test: Just subtraction
template<typename T>
T justSubtraction(T x) {
    return x - T(1.5);
}

// Test: Just multiplication
template<typename T>
T justMultiplication(T x) {
    return x * T(3.0);
}

// Test: Just division
template<typename T>
T justDivision(T x) {
    return x / T(2.0);
}

// Test: Just negation
template<typename T>
T justNegation(T x) {
    return -x;
}

// Test: Just absolute value
template<typename T>
T justAbsolute(T x) {
    if constexpr (std::is_same_v<T, double>) {
        return std::abs(x);
    } else {
        return select_abs(x);
    }
}

// Test: Just reciprocal
template<typename T>
T justReciprocal(T x) {
    return T(1.0) / x;
}

// Test: Just square root
template<typename T>
T justSquareRoot(T x) {
    return std::sqrt(x);
}

// Test: Just logarithm
template<typename T>
T justLogarithm(T x) {
    return std::log(x);
}

// Comparison functions moved to comparisons_specialized.hpp

// Test: Just power operation
template<typename T>
T justPower(T x) {
    return std::pow(x, T(2.0));
}

// Test: Just modulo operation
template<typename T>
T justModulo(T x) {
    return std::fmod(x, T(3.0));
}

// Test: Just if operation (conditional)
template<typename T>
T justIf(T x) {
    auto cond = (x > T(0.0));
    T true_val = x * T(2.0);
    T false_val = x + T(1.0);
    return select(cond, true_val, false_val);
}

// Test: Just 3-operand addition
template<typename T>
T justAddition3(T x) {
    return x + T(1.0) + T(2.0);
}

// Test: Just 3-operand subtraction
template<typename T>
T justSubtraction3(T x) {
    return x - T(1.0) - T(2.0);
}

// Test: Just 3-operand multiplication
template<typename T>
T justMultiplication3(T x) {
    return x * T(2.0) * T(3.0);
}

// Test: Just 3-operand division
template<typename T>
T justDivision3(T x) {
    return x / T(2.0) / T(3.0);
}

// Test: Just square
template<typename T>
T justSquare(T x) {
    return x * x;
}

// Test: Just sine
template<typename T>
T justSine(T x) {
    return std::sin(x);
}

// Test: Just cosine
template<typename T>
T justCosine(T x) {
    return std::cos(x);
}

// Test: Just tangent
template<typename T>
T justTangent(T x) {
    return std::tan(x);
}

// Test: Just minimum
template<typename T>
T justMinimum(T x) {
    // Use native min function (works with both double and fdouble)
    // For fdouble, this will use the min() function defined in double_tp.hpp
    // For double, this will use std::min or std::fmin
    using std::min;  // Enable ADL to find the right min function
    return min(x, T(2.0));
}

// Test: Just maximum
template<typename T>
T justMaximum(T x) {
    // Use native max function (works with both double and fdouble)
    // For fdouble, this will use the max() function defined in double_tp.hpp
    // For double, this will use std::max or std::fmax
    using std::max;  // Enable ADL to find the right max function
    return max(x, T(2.0));
}

// Test: exp then divide by constant
template<typename T> 
T expDivideConstant(T x) {
    T exp_val = std::exp(x);
    return T(1.0) / exp_val;
}

// Test: exp then divide by it
template<typename T>
T expSelfDivide(T x) {
    T exp_val = std::exp(x);
    T result = exp_val / exp_val;  // Should be 1.0
    return result;
}

// Test: Two exp calls
template<typename T>
T twoExpCalls(T x) {
    T exp1 = std::exp(x);
    T exp2 = std::exp(x * T(0.5));
    return exp1 + exp2;
}

// Test: Two exp calls with division
template<typename T>
T twoExpWithDiv(T x) {
    T exp1 = std::exp(x);
    T exp2 = std::exp(x * T(0.5));
    return exp1 / exp2;  // Should be exp(x/2)
}

// Test: exp in more complex expression
template<typename T>
T expComplexExpr(T x) {
    T a = std::exp(x * T(0.1));
    T b = T(2.0) * a;
    T c = b + T(3.0);
    T d = T(1.0) / c;
    return d;
}

// Test: Multiple divisions after exp
template<typename T>
T expMultipleDivisions(T x) {
    T exp_val = std::exp(x);
    T div1 = T(1.0) / exp_val;
    T div2 = T(2.0) / exp_val;
    T div3 = div1 / div2;  // Should be 0.5
    return div3;
}

// Test: exp with intermediate storage
template<typename T>
T expWithStorage(T x) {
    T temp1 = x * T(0.25);
    T temp2 = std::sqrt(T(0.333333));
    T temp3 = temp1 * temp2;
    T exp_result = std::exp(temp3);
    T inverse = T(1.0) / exp_result;
    return inverse;
}

// Test: Minimal failing pattern from American option
template<typename T>
T minimalAmericanPattern(T x) {
    // This is the exact pattern that fails in American options
    T sigma = T(0.25);
    T dt = T(0.333333);  // 1/3
    T sqrt_dt = std::sqrt(dt);
    T exp_arg = sigma * sqrt_dt;
    T a = std::exp(exp_arg);  
    T d = T(1.0) / a;  // This division after exp seems to be the issue
    return d * x;  // Use x to make it depend on input
}

// Test: Same pattern but without exp
template<typename T>
T minimalPatternNoExp(T x) {
    T sigma = T(0.25);
    T dt = T(0.333333);
    T sqrt_dt = std::sqrt(dt);
    T exp_arg = sigma * sqrt_dt;
    T a = T(1.144122144);  // Hardcoded exp(0.144337567) 
    T d = T(1.0) / a;
    return d * x;
}

// Test: exp then immediate use (no intermediate variable)
template<typename T>
T expImmediateUse(T x) {
    return T(1.0) / std::exp(x);
}

// Test: exp stored then used later
template<typename T>
T expStoredUse(T x) {
    T exp_val = std::exp(x);
    T other_calc = x * T(2.0) + T(3.0);  // Some other calculations
    return T(1.0) / exp_val;  // Use exp result later
}

// Test: Simplest failing case - just constant / exp
template<typename T>
T simplestExpFail(T x) {
    T exp_val = std::exp(x);
    T constant = T(1.0);  // Force constant into a variable
    return constant / exp_val;
}

// Test: Same but with constant loaded after exp
template<typename T>
T constantAfterExp(T x) {
    T exp_val = std::exp(x);
    // Force some operations to potentially move constant loading after exp
    T dummy = x * T(0.0);  // This should optimize away but might affect register allocation
    return (T(1.0) + dummy) / exp_val;
}

// Test: Verify register corruption hypothesis
template<typename T>
T registerCorruptionTest(T x) {
    T const1 = T(1.0);    // Load constant before exp
    T const2 = T(2.0);    // Load another constant
    T exp_val = std::exp(x);  // This might corrupt const1/const2 if they're in XMM0-XMM5
    return const1 / exp_val + const2;  // Use both constants after exp
}

// ========== NEW DIAGNOSTIC TESTS FOR AVX2 ISSUE ==========

// Test 2: Same as selectDiagnosticArrayWithSelect but only 20 operations (~100 total)
template<typename T>
T selectDiagnosticArrayWithSelect2(T x) {
    // Same initialization as original
    T values[4];
    values[0] = x - T(80.0);
    values[1] = x - T(96.0);
    values[2] = x - T(100.0);
    values[3] = x - T(120.0);
    
    // Apply select on array elements
    for (int i = 0; i < 4; ++i) {
        auto is_positive = (values[i] > T(0));
        values[i] = select(is_positive, values[i], T(0));
    }
    
    // Only 2 iterations instead of more (to get ~100 operations total)
    for (int iter = 0; iter < 2; ++iter) {
        for (int i = 2; i >= 0; --i) {
            T combined = values[i] * T(0.6) + values[i+1] * T(0.4);
            auto should_replace = (combined > T(5.0));
            values[i] = select(should_replace, combined, values[i] * T(1.1));
        }
    }
    
    return values[0];
}

// Test 3: Without multiplication - using only addition/subtraction
template<typename T>
T selectDiagnosticArrayWithSelect3(T x) {
    T values[4];
    values[0] = x - T(80.0);
    values[1] = x - T(96.0);
    values[2] = x - T(100.0);
    values[3] = x - T(120.0);
    
    // Apply select on array elements
    for (int i = 0; i < 4; ++i) {
        auto is_positive = (values[i] > T(0));
        values[i] = select(is_positive, values[i], T(0));
    }
    
    // Use addition instead of multiplication
    for (int i = 2; i >= 0; --i) {
        T combined = values[i] + values[i+1] - T(3.0);  // No multiplication
        auto should_replace = (combined > T(5.0));
        values[i] = select(should_replace, combined, values[i] + T(2.0));  // Add instead of mul
    }
    
    return values[0];
}

// Test 4: Without select/if - just arithmetic operations
template<typename T>
T selectDiagnosticArrayWithSelect4(T x) {
    T values[4];
    values[0] = x - T(80.0);
    values[1] = x - T(96.0);
    values[2] = x - T(100.0);
    values[3] = x - T(120.0);
    
    // No select/max/comparisons - just pure arithmetic
    // Skip the clamping entirely to avoid any comparison operations
    
    // Pure arithmetic combinations
    for (int i = 2; i >= 0; --i) {
        T combined = values[i] * T(0.6) + values[i+1] * T(0.4);
        // Always apply the multiplication without condition
        values[i] = combined * T(1.1);
    }
    
    return values[0];
}

// Test 5: Focus on the multiplication after select pattern
template<typename T>
T selectDiagnosticArrayWithSelect5(T x) {
    T values[4];
    values[0] = x - T(80.0);
    values[1] = x - T(96.0);
    values[2] = x - T(100.0);
    values[3] = x - T(120.0);
    
    // First select
    for (int i = 0; i < 4; ++i) {
        auto is_positive = (values[i] > T(0));
        values[i] = select(is_positive, values[i], T(0));
    }
    
    // Immediately do multiplication on selected values
    T result = T(0);
    for (int i = 0; i < 4; ++i) {
        result += values[i] * T(0.25);  // This multiplication might fail after select
    }
    
    return result;
}

// Test 6: Conditional multiplication pattern (the likely culprit)
template<typename T>
T selectDiagnosticArrayWithSelect6(T x) {
    T values[4];
    values[0] = x - T(80.0);
    values[1] = x - T(96.0);  
    values[2] = x - T(100.0);
    values[3] = x - T(120.0);
    
    T result = T(0);
    
    // Pattern: comparison -> select -> multiply selected result
    for (int i = 0; i < 4; ++i) {
        auto cond = (values[i] > T(0));
        T selected = select(cond, values[i], T(1.0));
        // This multiplication happens on a conditionally selected value
        result += selected * T(0.25);  
    }
    
    return result;
}

} // namespace one_to_one
} // namespace test_functions
} // namespace tools
} // namespace forge