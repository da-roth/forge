// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file forge_engine.hpp
 * @brief Main JIT compiler interface for mathematical expression graphs
 *
 * Defines ForgeEngine (the compiler) and StitchedKernel (the compiled executable).
 * ForgeEngine compiles expression graphs into optimized machine code using AsmJit.
 *
 * Thread Safety: ForgeEngine instances are not thread-safe. StitchedKernel instances
 * are safe to execute concurrently from multiple threads.
 */

#pragma once

#include "../graph/graph.hpp"  // Reuse existing Graph structure
#include "node_value_buffers/node_value_buffer.hpp"
#include "register_allocator.hpp"  // For backward compatibility
#include "xmm_register_allocator.hpp"  // For SSE2
#include "ymm_register_allocator.hpp"  // For AVX2
#include "../graph/graph_optimizer.hpp"
#include "compiler_config.hpp"
#include "instruction_set.hpp"
#include "instruction_set_factory.hpp"
#include "runtime_trace.hpp"
#include <asmjit/x86.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace forge {

// Forward declaration
class StitchedKernel;

/**
 * @brief JIT compiler for mathematical expression graphs
 *
 * ForgeEngine compiles computational graphs (from GraphRecorder) into optimized
 * x86/x64 machine code with support for automatic differentiation. The compiler
 * applies graph optimizations (CSE, algebraic simplification, etc.) before
 * generating assembly code via AsmJit.
 *
 * Compilation Pipeline:
 * 1. Graph optimization (constant folding, CSE, simplification)
 * 2. Forward pass code generation
 * 3. Gradient pass code generation (if needed)
 * 4. JIT assembly and linking
 *
 * Thread Safety: Not thread-safe - create separate instances per thread
 *
 * API Stability: Stable - core methods won't change
 *
 * Example:
 * @code
 * ForgeEngine engine(CompilerConfig::Default());
 * auto kernel = engine.compile(graph);
 * kernel->execute(buffer);
 * @endcode
 */
class ForgeEngine {
public:
    /** @brief Construct engine with default configuration */
    ForgeEngine();

    /**
     * @brief Construct engine with custom configuration
     * @param config Compiler configuration controlling optimizations and output
     */
    explicit ForgeEngine(const CompilerConfig& config);

    ~ForgeEngine();

    /**
     * @brief Compile a computational graph into executable machine code
     *
     * Takes a recorded graph and compiles it through the full pipeline:
     * optimization, forward pass generation, gradient pass generation (if needed),
     * and JIT assembly.
     *
     * @param graph The computational graph from GraphRecorder::stop()
     * @return Compiled kernel ready for execution
     * @throws std::runtime_error if compilation fails
     *
     * Thread Safety: Not thread-safe
     */
    std::unique_ptr<StitchedKernel> compile(const forge::Graph& graph);

    /**
     * @brief Get current compiler configuration
     * @return Reference to active configuration
     */
    const CompilerConfig& getConfig() const { return config_; }

    /**
     * @brief Update compiler configuration
     * @param config New configuration to use
     */
    void setConfig(const CompilerConfig& config) { config_ = config; }

    /**
     * @brief Get the shared JIT runtime (for testing/debugging)
     * @return Reference to global JitRuntime instance
     */
    static asmjit::JitRuntime& getRuntime();

    /** @brief Constant pool information for JIT code generation */
    struct ConstantInfo {
        size_t poolOffset;  ///< Offset within the constant pool
        double value;       ///< The constant value
    };
    
private:
    // Compiler configuration
    CompilerConfig config_;
    
    // Instruction set implementation (based on config)
    std::unique_ptr<IInstructionSet> instructionSet_;
    
    // Shared JitRuntime for all compilers - long-lived per Design v3
    // This ensures executable memory remains valid after compiler destruction
    static asmjit::JitRuntime s_runtime;
    
    // Phase 2.3: Register tracking for optimization
    // Now creates appropriate allocator based on instruction set
    std::unique_ptr<IRegisterAllocator> createRegisterAllocator() const;
    
    // Phase 2.4: Block-based compilation structures
    struct FusionBlock {
        forge::NodeId startNode;
        forge::NodeId endNode;  // Exclusive
        std::vector<forge::NodeId> liveOut;  // Nodes needed after block
    };

    // Forward pass code generation has been migrated to forward_stitcher.cpp
    // See ForwardStitcher class for implementation

    // Phase 2.4: Block-based compilation helpers
    std::vector<FusionBlock> identifyFusionBlocks(const forge::Graph& graph);
};

/**
 * @brief Compiled executable kernel from ForgeEngine
 *
 * Represents a JIT-compiled mathematical function with automatic differentiation
 * support. Manages the lifetime of the executable code and provides both direct
 * and buffered execution interfaces.
 *
 * Thread Safety: Safe to execute concurrently from multiple threads (the compiled
 * code is reentrant). However, construction and destruction are not thread-safe.
 *
 * API Stability: Stable - execution interface won't change
 *
 * Example:
 * @code
 * auto kernel = engine.compile(graph);
 * INodeValueBuffer& buffer = ...;
 * kernel->execute(buffer);  // Compute forward + gradient passes
 * @endcode
 */
class StitchedKernel {
public:
    /** @brief Function signature for compiled kernels */
    using KernelFunc = void(*)(double* values, double* gradients, size_t count);

    StitchedKernel(KernelFunc func, asmjit::JitRuntime& runtime, size_t num_nodes, const IInstructionSet* instructionSet, const CompilerConfig& config, size_t max_node_id = 0, size_t working_nodes = 0)
        : func_(func), runtime_(&runtime), num_nodes_(num_nodes),
          vector_width_(instructionSet->getVectorWidth()),
          instruction_set_name_(instructionSet->getName()),
          config_(config), max_node_id_(max_node_id), working_nodes_(working_nodes > 0 ? working_nodes : num_nodes) {}

    // Constructor with node ID mapping
    StitchedKernel(KernelFunc func, asmjit::JitRuntime& runtime, size_t num_nodes, const IInstructionSet* instructionSet, const CompilerConfig& config,
                   const std::vector<forge::NodeId>& originalToOptimizedMapping, size_t max_node_id = 0, size_t working_nodes = 0,
                   const std::vector<forge::NodeId>& outputNodes = {})
        : func_(func), runtime_(&runtime), num_nodes_(num_nodes),
          vector_width_(instructionSet->getVectorWidth()),
          instruction_set_name_(instructionSet->getName()),
          config_(config),
          max_node_id_(max_node_id), working_nodes_(working_nodes > 0 ? working_nodes : num_nodes),
          originalToOptimizedMapping_(originalToOptimizedMapping), outputNodes_(outputNodes) {
        // std::cout << "[KERNEL CONSTRUCTOR] num_nodes=" << num_nodes_
        //           << ", max_node_id=" << max_node_id_
        //           << ", working_nodes=" << working_nodes_
        //           << ", getRequiredNodes()=" << getRequiredNodes()
        //           << ", getMaxNodeId()=" << getMaxNodeId() << std::endl;
    }
    
    ~StitchedKernel() {
        if (func_ && runtime_) {
            runtime_->release(func_);
        }
    }

    /**
     * @brief Execute kernel with raw pointers (zero-overhead)
     *
     * Direct execution bypassing the buffer interface for maximum performance.
     * Use when you need minimal overhead and have properly laid out memory.
     *
     * @param values Pointer to node values array (must be properly aligned)
     * @param gradients Pointer to gradient array (can be nullptr if no gradients)
     * @param count Number of nodes in the arrays
     *
     * Thread Safety: Reentrant - safe to call concurrently
     */
    inline void executeDirect(double* values, double* gradients, size_t count) {
        func_(values, gradients, count);
    }

    /**
     * @brief Execute kernel using a NodeValueBuffer
     *
     * Executes both forward and gradient passes (if enabled) using a buffer
     * that manages memory layout and alignment. Includes timing and output
     * logging in debug builds.
     *
     * @param buffer Value buffer containing inputs and receiving outputs
     *
     * Thread Safety: Reentrant - safe to call concurrently with different buffers
     */
    inline void execute(INodeValueBuffer& buffer) {
#ifndef FORGE_RELEASE_BUILD
        // Show kernel configuration on first call only (debug mode only)
        static bool firstCall = true;
        if (firstCall) {
            int vw = vector_width_;
            size_t bytesPerNode = vw * sizeof(double);
            size_t totalBufferBytes = buffer.getNumNodes() * bytesPerNode;
            std::cout << "[KERNEL] Configuration: " << instruction_set_name_
                      << " (width=" << vw << ", " << bytesPerNode << " bytes/node, "
                      << "buffer=" << totalBufferBytes << " bytes for " << buffer.getNumNodes() << " nodes)" << std::endl;
            firstCall = false;
        }
        /*std::cout << "[KERNEL] Executing kernel..." << std::endl;*/
#endif

#ifdef FORGE_RELEASE_BUILD
        // RELEASE: Use direct execution to avoid virtual function calls - no debug output
        executeDirect(buffer.getValuesPtr(), buffer.getGradientsPtr(), buffer.getNumNodes());
#else
        // DEBUG: Full validation and tracing support
        double* values = buffer.getValuesPtr();
        if (func_ && buffer.getNumNodes() >= getRequiredNodes()) {
            // std::cout << "[KERNEL] Conditions met, executing kernel function..." << std::endl;

            // Pass gradient pointer if available, otherwise nullptr
            // std::cout << "[KERNEL] Calling func_ (the compiled kernel)..." << std::endl;
            auto execStart = std::chrono::high_resolution_clock::now();
            func_(values, buffer.getGradientsPtr(), buffer.getNumNodes());
            auto execEnd = std::chrono::high_resolution_clock::now();
            auto execTimeUs = std::chrono::duration<double, std::micro>(execEnd - execStart).count();
            // std::cout << "[KERNEL] func_ returned successfully" << std::endl;

            // Show output value from first output node (for debugging)
            if (!outputNodes_.empty() && buffer.getNumNodes() > 0) {
                forge::NodeId outputNode = outputNodes_[0];
                double outputValue = values[outputNode * vector_width_];
                /*std::cout << "[KERNEL] Execution completed in " << std::fixed << std::setprecision(3) << execTimeUs
                          << " μs. Output node " << outputNode << " value: "
                          << std::fixed << std::setprecision(17) << outputValue << std::endl;*/
            } else if (buffer.getNumNodes() > 0) {
                // Fallback to last node if no output nodes specified
                size_t lastNode = buffer.getNumNodes() - 1;
                double outputValue = values[lastNode * vector_width_];
                /*std::cout << "[KERNEL] Execution completed in " << std::fixed << std::setprecision(3) << execTimeUs
                          << " μs. Last node value: "
                          << std::fixed << std::setprecision(17) << outputValue << std::endl;*/
            }

            // Print runtime trace after forward pass if tracing is enabled
            // std::cout << "[KERNEL] Checking if tracing is enabled..." << std::endl;
            bool tracingEnabled = forge::isTracingEnabled();
            // std::cout << "[KERNEL] isTracingEnabled() returned: " << (tracingEnabled ? "TRUE" : "FALSE") << std::endl;
            // std::cout << "[KERNEL] g_traceBuffer.enabled = " << (forge::g_traceBuffer.enabled ? "true" : "false") << std::endl;
            // std::cout << "[KERNEL] g_traceBuffer.records = " << (forge::g_traceBuffer.records ? "not null" : "NULL") << std::endl;

            if (tracingEnabled) {
                // std::cout << "[KERNEL] ==> ENTERING TRACE PRINT BLOCK" << std::endl;
                // std::cout << "[TRACE] Starting runtime trace records print..." << std::endl;
                // std::cout << "[TRACE] Instruction set: " << instruction_set_name_ << std::endl;
                // std::cout << "[TRACE] Trace buffer enabled: " << (forge::g_traceBuffer.enabled ? "yes" : "no") << std::endl;
                // std::cout << "[TRACE] Records captured: " << forge::g_traceBuffer.index.load() << std::endl;

                // std::cout << "[KERNEL] About to call printTraceRecords()..." << std::endl;
                forge::printTraceRecords();
                // std::cout << "[KERNEL] printTraceRecords() returned" << std::endl;

                // std::cout << "[TRACE] Finished runtime trace records print." << std::endl;
                // std::cout << "[KERNEL] ==> EXITING TRACE PRINT BLOCK" << std::endl;
            } else {
                // std::cout << "[KERNEL] ==> SKIPPING TRACE PRINT (tracing not enabled)" << std::endl;
                // std::cout << "[TRACE] Runtime tracing is NOT enabled (isTracingEnabled() returned false)" << std::endl;
            }
        } else {
            // std::cout << "[KERNEL] Conditions NOT met, skipping execution" << std::endl;
            // if (!func_) std::cout << "[KERNEL] Reason: func_ is NULL" << std::endl;
            // if (buffer.getNumNodes() < getRequiredNodes()) {
            //     std::cout << "[KERNEL] Reason: buffer too small (" << buffer.getNumNodes() << " < " << getRequiredNodes() << ")" << std::endl;
            //     std::cout << "[KERNEL] Note: num_nodes_=" << num_nodes_ << " but buffer sized for getRequiredNodes()=" << getRequiredNodes() << std::endl;
            // }
        }

        // std::cout << "[KERNEL] ==> execute() completed" << std::endl;
#endif
    }

    /**
     * @brief Get raw function pointer for direct calling
     * @return Function pointer to compiled kernel
     *
     * Use for benchmarking or when you need direct control over execution.
     * Most users should use execute() or executeDirect() instead.
     */
    KernelFunc getFunction() const { return func_; }

    /**
     * @brief Get SIMD vector width of this kernel
     * @return Number of doubles processed per operation (1 for scalar, 4 for AVX2)
     */
    int getVectorWidth() const { return vector_width_; }

    /**
     * @brief Get instruction set name used by this kernel
     * @return Name string (e.g., "SSE2-Scalar", "AVX2-Packed")
     */
    std::string getInstructionSetName() const { return instruction_set_name_; }

    /**
     * @brief Get maximum node ID accessed by this kernel
     * @return Highest node ID used during compilation
     */
    size_t getMaxNodeId() const { return max_node_id_ > 0 ? max_node_id_ : working_nodes_ - 1; }

    /**
     * @brief Get required buffer size in number of nodes
     * @return Minimum buffer size needed for execution
     */
    size_t getRequiredNodes() const {
        return getMaxNodeId() + 1;
    }

    /**
     * @brief Get node ID mapping from original to optimized graph
     * @return Vector mapping original node IDs to optimized positions
     */
    const std::vector<forge::NodeId>& getOriginalToOptimizedMapping() const {
        return originalToOptimizedMapping_;
    }
    
    // Disable copy
    StitchedKernel(const StitchedKernel&) = delete;
    StitchedKernel& operator=(const StitchedKernel&) = delete;
    
    // Enable move
    StitchedKernel(StitchedKernel&& other) noexcept
        : func_(other.func_), runtime_(other.runtime_), num_nodes_(other.num_nodes_),
          vector_width_(other.vector_width_),
          instruction_set_name_(std::move(other.instruction_set_name_)),
          config_(other.config_),
          max_node_id_(other.max_node_id_), working_nodes_(other.working_nodes_),
          originalToOptimizedMapping_(std::move(other.originalToOptimizedMapping_)),
          outputNodes_(std::move(other.outputNodes_)) {
        other.func_ = nullptr;
        other.runtime_ = nullptr;
        other.vector_width_ = 0;
        other.max_node_id_ = 0;
        other.working_nodes_ = 0;
    }

private:
    KernelFunc func_;
    asmjit::JitRuntime* runtime_;  // Points to shared static runtime
    size_t num_nodes_;              // Original graph size (for buffer compatibility)
    int vector_width_;              // SIMD vector width (1 for scalar, 4 for AVX2)
    std::string instruction_set_name_;  // Name of instruction set used
    CompilerConfig config_;         // Compiler configuration
    size_t max_node_id_;           // Maximum node ID accessed during compilation
    size_t working_nodes_;         // Working graph size (after optimizations)
    std::vector<forge::NodeId> originalToOptimizedMapping_;  // Node ID mapping
    std::vector<forge::NodeId> outputNodes_;  // Output node IDs (for debug display)
};

} // namespace forge