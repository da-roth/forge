#pragma once

#include <cmath>
#include <vector>
#include "../select_helper.hpp"

namespace forge {
namespace tools {
namespace test_functions {
namespace one_to_one {

// Massive graph generator inspired by big computation patterns (~1.2M operations)
// Creates ~1M+ operations through smart nested iteration
// Based on operation statistics from actual big computation:
// - Multiply: 345,204 (29% - dominant)  
// - Add/Subtract: ~312K combined
// - Conditional logic: ~260K (Greater, Max, Min, Conditional)
// - Abs: 53,010 (4.5%)
// - Power: 25,843 (2.2%)
// - Exp: 7 (very rare)
template<typename T>
T massiveIterativeGraph(T x) {
    T result = x * T(0.1) + T(1.0);  // Initialize with bounded input
    T accumulator = T(1.0);
    T state1 = T(0.0);
    T state2 = T(1.0);
    
    // Outer loop: ~1000 iterations
    for (int i = 0; i < 1000; ++i) {
        T factor = T(static_cast<double>(i + 1)) * T(0.001); // Scale factor: 0.001 to 1.0
        
        // Inner loop: ~1000 iterations per outer iteration  
        for (int j = 0; j < 1000; ++j) {
            T subfactor = T(static_cast<double>(j + 1)) * T(0.0001); // 0.0001 to 0.1
            
            // Core arithmetic operations (matches multiply dominance)
            T temp1 = result * factor;           // Multiply
            T temp2 = temp1 * subfactor;         // Multiply  
            T temp3 = temp2 - accumulator;       // Subtract
            T temp4 = temp3 + state1;            // Add
            T temp5 = temp4 / (subfactor + T(0.01)); // Divide (avoid division by zero)
            
            // Conditional logic using proper select helper (matches high conditional/comparison counts)
            auto is_positive = (temp5 > T(0.0));                    // Greater comparison
            T max_result = select_max(temp5, state2);                // Max operation  
            T min_result = select_min(temp5, -state2);               // Min operation
            T conditional_result = select(is_positive, max_result, min_result);
            
            // Absolute value operations (matches ~53K abs ops)  
            T abs_result = select_abs(conditional_result);
            
            // Power operations every 40 iterations (matches ~25K power ops)
            T power_result;
            if ((i * 1000 + j) % 40 == 0) {
                T safe_base = abs_result * abs_result + T(1.0); // Ensure positive
                power_result = std::pow(safe_base, T(1.5));     // PowerPositiveBaseNonInteger
            } else {
                power_result = abs_result;
            }
            
            // Update states with bounded operations
            T bounded = power_result * T(0.001); // Keep values small
            state1 = (state1 * T(0.999)) + bounded;  // Slowly evolving state
            state2 = select_abs(state2 * T(0.998)) + (bounded * T(0.1));
            
            // Accumulate result with decay to prevent overflow
            accumulator = accumulator * T(0.9999) + bounded;
            result = result * T(0.9995) + (bounded * T(0.001));
        }
        
        // Occasional exp operations (matches low exp count: 7 total)
        if (i % 142 == 0) { // ~7 times in 1000 iterations
            T safe_exp_input = result * T(0.001); // Very small input for exp
            result = result + std::exp(safe_exp_input) * T(0.001);
        }
    }
    
    // Final bounded result
    return result * T(10.0) + T(x); // Scale back up and include input dependency
}

// Medium-scale version for practical testing: ~10K operations
// Same operation patterns, scaled down for routine testing
template<typename T>  
T mediumIterativeGraph(T x) {
    T result = x * T(0.1) + T(1.0);
    T accumulator = T(1.0);
    T state = T(0.0);
    
    // 100x100 = 10K iterations, each with ~1 effective operation
    for (int i = 0; i < 100; ++i) {
        T factor = T(static_cast<double>(i + 1)) * T(0.01);
        
        for (int j = 0; j < 100; ++j) {
            T subfactor = T(static_cast<double>(j + 1)) * T(0.001);
            
            // Core arithmetic operations
            T temp = result * factor * subfactor;
            temp = temp - accumulator + state;
            temp = temp / (subfactor + T(0.1));
            
            // Conditional logic using select helper
            auto is_positive = (temp > T(0.0));
            temp = select(is_positive, select_max(temp, state), temp);
            
            // Absolute value
            temp = select_abs(temp);
            
            // Power operations every 25 iterations
            if ((i * 100 + j) % 25 == 0) {
                T base = temp * temp + T(1.0);
                temp = std::pow(base, T(1.2));
            }
            
            // State updates
            state = state * T(0.99) + temp * T(0.001);
            accumulator = accumulator * T(0.999) + temp * T(0.0001);
            result = result * T(0.995) + temp * T(0.00001);
        }
    }
    
    return result * T(100.0) + x;
}

// Smaller version for quick testing: ~1K operations
template<typename T>
T smallIterativeGraph(T x) {
    T result = x * T(0.1) + T(1.0);
    T accumulator = T(1.0);
    
    // 50x20 = 1K iterations
    for (int i = 0; i < 50; ++i) {
        T factor = T(static_cast<double>(i + 1)) * T(0.02);
        
        for (int j = 0; j < 20; ++j) {
            T subfactor = T(static_cast<double>(j + 1)) * T(0.005);
            
            T temp = result * factor;
            temp = temp - accumulator * subfactor;
            temp = temp + T(static_cast<double>(j)) * T(0.01);
            
            auto is_positive = (temp > T(0.0));
            temp = select(is_positive, temp, -temp);
            temp = select_abs(temp);
            
            if ((i * 20 + j) % 10 == 0) {
                T base = temp + T(1.0);
                temp = std::pow(base, T(0.8));
            }
            
            accumulator = accumulator * T(0.98) + temp * T(0.01);
            result = result * T(0.99) + temp * T(0.001);
        }
    }
    
    return result * T(10.0) + x;
}

// Test input sets for big graph functions
inline std::vector<double> getBigGraphInputs() {
    return {-1, -0.5, 0, 0.5, 1};
}

inline std::vector<double> getSmallGraphInputs() {
    return {-2, -1, 0, 1, 2};
}

} // namespace one_to_one
} // namespace test_functions
} // namespace tools
} // namespace forge