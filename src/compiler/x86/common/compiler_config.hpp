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
    bool enableOptimizations = false;       // Master switch for all optimizations (default: off, only stability cleaning enabled)
    bool enableInactiveFolding = false;     // Fold constant subgraphs (isActive=false nodes)
    bool enableCSE = false;                 // Common subexpression elimination
    bool enableAlgebraicSimplification = false; // Apply algebraic identities (x*1=x, etc)
    bool enableStabilityCleaning = true;    // Fix numerical stability issues (1/exp(x) -> exp(-x)) - DEFAULT: enabled
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

    // Dynamic instruction set selection (for runtime-loaded backends)
    std::string instructionSetName;      // Name of dynamically registered instruction set
    bool useNamedInstructionSet = false; // If true, use instructionSetName instead of enum

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

    /** @brief Create default production configuration with only stability cleaning enabled */
    static CompilerConfig Default() {
        CompilerConfig config;
        // Only enable stability cleaning by default (sanity cleaning)
        config.enableStabilityCleaning = true;
        // All other optimizations are disabled by default
        config.enableOptimizations = false;
        config.enableInactiveFolding = false;
        config.enableCSE = false;
        config.enableAlgebraicSimplification = false;
        return config;
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
        // Enable all optimizations for maximum performance
        config.enableOptimizations = true;
        config.enableInactiveFolding = true;
        config.enableCSE = true;
        config.enableAlgebraicSimplification = true;
        config.enableStabilityCleaning = true;
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

    /** @brief Create configuration with runtime tracing enabled */
    static CompilerConfig DebugTracing() {
        CompilerConfig config;
        config.printRuntimeTrace = true;
        return config;
    }
};

} // namespace forge