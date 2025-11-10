// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file compiler_config.hpp
 * @brief Configuration options for the Forge JIT compiler
 *
 * Defines CompilerConfig which controls optimization passes, debug output,
 * instruction set selection, and runtime tracing behavior.
 *
 * Thread Safety: Not thread-safe - each compilation should use its own instance.
 */

#pragma once

#include <cstdlib>  // For std::getenv
#include <string>

namespace forge {

/**
 * @brief Configuration settings for the ForgeEngine JIT compiler
 *
 * Controls all compilation behavior via public fields. Use factory methods
 * (Default, Debug, Fast, etc.) for common configurations.
 *
 * API Stability: Stable - new fields may be added but existing ones won't change.
 *
 * Example:
 * @code
 * ForgeEngine engine(CompilerConfig::Debug());  // Full debug output
 * @endcode
 */
struct CompilerConfig {
    // Optimization flags (matching GraphOptimizer::OptimizationConfig defaults)
    bool enableOptimizations = true;        // Master switch for all optimizations
    bool enableInactiveFolding = true;      // Fold constant subgraphs (isActive=false nodes)
    bool enableCSE = true;                  // Common subexpression elimination
    bool enableAlgebraicSimplification = true; // Apply algebraic identities (x*1=x, etc)
    bool enableStabilityCleaning = true;    // Fix numerical stability issues (1/exp(x) -> exp(-x))
    int maxOptimizationPasses = 5;          // Iterate until no changes or max passes
    
    // Debug output flags (all false by default in production)
    bool printOriginalGraph = false;        // Print the input graph before optimization
    bool printOptimizedGraph = false;       // Print the graph after optimization
    bool printAssembly = false;             // Print generated assembly code
    bool printRegisterAllocation = false;   // Print register allocation decisions
    bool printOptimizationStats = false;    // Print statistics about optimizations applied
    bool printStepByStepDebug = false;      // Print graph after each optimization step
    bool printGradientDebug = false;        // Print gradient computation debug info
    bool printNodeFlags = false;            // Print needsGradient and isActive flags for each node
    bool printRuntimeTrace = false;           // Default off: tracing can perturb YMM registers
    
    // Smart runtime trace filtering options  
    bool enableSmartTraceFilter = false;      // Enable intelligent corruption detection filtering
    bool traceCorruptedOnly = true;          // Only trace operations with detected corruption
    bool traceNearCorruption = true;         // Trace operations immediately before/after corruption
    int corruptionContextSize = 2;           // Number of operations to show before/after corruption
    
    // Corruption detection criteria
    bool detectNaNCorruption = true;         // Detect NaN values in vector lanes
    bool detectInfCorruption = true;         // Detect infinite values in vector lanes  
    bool detectZeroCorruption = true;        // Detect suspicious zero values in lanes 2-3 (AVX2)
    bool detectPatternCorruption = true;     // Detect suspicious patterns like 0.002, 0.003 etc.
    bool detectPartialCorruption = true;     // Detect when only some lanes work correctly
    double corruptionThreshold = 1e-10;     // Threshold for detecting suspicious small values
    
    // Performance tuning
    size_t maxRegisterCount = 16;           // Use XMM0-XMM15 (full set for maximum performance)
    
    // Safety and validation
    bool validateGraph = false;             // Validate graph structure before compilation
    bool boundsChecking = false;            // Add bounds checks in generated code
    
    // Debug recording for integration testing
    bool enableDebugRecording = false;      // Enable recording of intermediate values for debugging
                                            // This adds memory overhead (vector<double> + flag to Graph struct)
    
    // Instruction set selection (extensible for future additions)
    enum class InstructionSet {
        SSE2_SCALAR, // Current default: SSE2 scalar double-precision operations (1 double per operation)
        AVX2_PACKED  // AVX2 256-bit vectors (4 doubles per operation, YMM registers)
        // Contributors can add more instruction sets here without modifying existing code:
        // SSE2_PACKED, // SSE2 packed operations (2 doubles per operation)
        // AVX512_PACKED, // AVX-512 512-bit vectors (8 doubles per operation, ZMM registers)  
        // NEON,        // ARM NEON vectors
        // To add a new instruction set:
        // 1. Add the enum value here
        // 2. Create implementation class inheriting from IInstructionSet
        // 3. Add case in InstructionSetFactory::create()
    };
    InstructionSet instructionSet = InstructionSet::SSE2_SCALAR;

    /**
     * @brief Load configuration from FORGE_INSTRUCTION_SET environment variable
     *
     * Reads the environment to override instruction set selection at runtime.
     * Supported values: "SSE2" or "SSE2-Scalar", "AVX2" or "AVX2-Packed"
     */
    void loadFromEnvironment() {
        // Check for FORGE_INSTRUCTION_SET environment variable
        const char* env = std::getenv("FORGE_INSTRUCTION_SET");
        if (env) {
            std::string val(env);
            if (val == "SSE2-Scalar" || val == "SSE2") instructionSet = InstructionSet::SSE2_SCALAR;
            else if (val == "AVX2-Packed" || val == "AVX2") instructionSet = InstructionSet::AVX2_PACKED;
            // Add more as they're implemented:
            // else if (val == "SSE2-Packed") instructionSet = InstructionSet::SSE2_PACKED;
            // else if (val == "AVX512-Packed") instructionSet = InstructionSet::AVX512_PACKED;
        }
    }

    /** @brief Create default production configuration with optimizations enabled */
    static CompilerConfig Default() {
        return CompilerConfig{};
    }

    /** @brief Create debug configuration with full diagnostic output enabled */
    static CompilerConfig Debug() {
        CompilerConfig config;
        config.printOriginalGraph = true;
        config.printOptimizedGraph = true;
        config.printAssembly = true;
        config.printOptimizationStats = true;
        config.printGradientDebug = true;
        config.printNodeFlags = true;
        config.enableDebugRecording = true;  // Enable recording for debugging
        return config;
    }

    /** @brief Create configuration with all optimizations disabled for debugging */
    static CompilerConfig NoOptimization() {
        CompilerConfig config;
        config.enableOptimizations = false;
        config.enableInactiveFolding = false;
        config.enableCSE = false;
        config.enableAlgebraicSimplification = false;
        config.enableStabilityCleaning = false;
        config.maxOptimizationPasses = 0;
        return config;
    }

    /** @brief Create configuration with aggressive optimizations for performance */
    static CompilerConfig Fast() {
        CompilerConfig config;
        // Already using all 16 registers by default
        config.maxOptimizationPasses = 10;  // More aggressive optimization
        return config;
    }

    /** @brief Create configuration with validation and safety checks enabled */
    static CompilerConfig Validation() {
        CompilerConfig config;
        config.validateGraph = true;
        config.boundsChecking = true;
        config.printOptimizationStats = true;
        return config;
    }

    /** @brief Create configuration with intelligent runtime tracing (corruption detection) */
    static CompilerConfig SmartDebugTracing() {
        CompilerConfig config;
        config.printRuntimeTrace = true;
        config.enableSmartTraceFilter = true;
        config.traceCorruptedOnly = true;
        config.traceNearCorruption = true;
        config.corruptionContextSize = 3;  // Show 3 operations before/after corruption
        return config;
    }

    /** @brief Create configuration with comprehensive runtime tracing (full context) */
    static CompilerConfig SmartDebugWithContext() {
        CompilerConfig config;
        config.printRuntimeTrace = true;
        config.enableSmartTraceFilter = true;
        config.traceCorruptedOnly = false;  // Show everything
        config.traceNearCorruption = true;
        config.corruptionContextSize = 5;   // Larger context window
        return config;
    }
};

} // namespace forge