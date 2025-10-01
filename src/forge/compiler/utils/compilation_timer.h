#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include "forge/core/computation_graph.h"

namespace forge::compiler::utils {

/**
 * Compilation timing utilities for measuring and reporting JIT compilation performance
 */
class CompilationTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    
    struct TimingData {
        double optimizationTimeMs = 0.0;
        double analysisTimeMs = 0.0;
        double codeGenerationTimeMs = 0.0;
        double totalTimeMs = 0.0;
        size_t originalNodeCount = 0;
        size_t optimizedNodeCount = 0;
        size_t deadNodeCount = 0;
    };
    
    /**
     * Get operation name for timing reports
     */
    static std::string getOpName(forge::core::OpCode op);
    
    /**
     * Print compilation timing summary
     */
    static void printTimingSummary(const TimingData& timing, bool verbose = false);
    
    // Note: Graph optimization statistics printing is handled by higher-level orchestrators
};

/**
 * RAII timer for measuring individual operation compilation time
 * Only collects timing data when profiling is enabled for zero overhead when disabled
 */
class OperationTimer {
public:
    /**
     * Start timing an operation
     * @param opName Name of the operation being timed
     * @param timeMap Map to accumulate total time per operation
     * @param countMap Map to count occurrences per operation  
     * @param enabled Whether profiling is enabled (false = zero overhead)
     */
    OperationTimer(const std::string& opName,
                   std::unordered_map<std::string, double>& timeMap,
                   std::unordered_map<std::string, int>& countMap,
                   bool enabled);
    
    /**
     * Stop timing and record results (automatic via destructor)
     */
    ~OperationTimer();

private:
    bool enabled_;
    std::string opName_;
    CompilationTimer::Clock::time_point start_;
    std::unordered_map<std::string, double>* timeMap_;
    std::unordered_map<std::string, int>* countMap_;
};

} // namespace forge::compiler::utils
