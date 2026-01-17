// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file compilation_policy.hpp
 * @brief Policy interface for controlling compilation decisions
 *
 * Provides a unified abstraction for customizing register allocation and
 * memory management strategies during JIT compilation. Custom policies can
 * implement pre-computed (analyze graph upfront) or dynamic (decide on-the-fly)
 * optimization patterns.
 *
 * Thread Safety: Policies are used during compilation which is single-threaded.
 * No thread-safety requirements.
 */

#pragma once

#include "../../graph/graph.hpp"
#include <asmjit/x86.h>
#include <cstdint>

namespace forge {

/**
 * @brief API version for the ICompilationPolicy interface
 *
 * Increment this when making breaking changes to the interface.
 */
constexpr uint32_t COMPILATION_POLICY_API_VERSION = 1;

/**
 * @brief Policy interface for controlling compilation decisions
 *
 * Allows customization of register allocation and memory management
 * strategies during JIT compilation. The default implementation preserves
 * current forge behavior.
 *
 * Two usage patterns are supported:
 *
 * 1. Pre-computed: Analyze the entire graph in onCompileBegin(), then
 *    return pre-computed decisions from requiresStore()/preferredRegister().
 *
 * 2. Dynamic: Make decisions on-the-fly based on runtime context,
 *    using onNodeBegin()/onNodeEnd() to track state.
 *
 * Example (pre-computed):
 * @code
 * class GlobalOptimizationPolicy : public ICompilationPolicy {
 *     std::vector<int> nodeToReg_;
 *     std::vector<bool> needsStore_;
 *
 * public:
 *     void onCompileBegin(const Graph& graph, asmjit::x86::Assembler& a) override {
 *         performLivenessAnalysis(graph);
 *         computeOptimalRegisterAssignment(graph);
 *     }
 *
 *     bool requiresStore(NodeId n, const Graph&) override {
 *         return needsStore_[n];
 *     }
 *
 *     int preferredRegister(NodeId n) override {
 *         return nodeToReg_[n];
 *     }
 * };
 * @endcode
 */
class ICompilationPolicy {
public:
    virtual ~ICompilationPolicy() = default;

    /**
     * @brief Get the API version this implementation was built against
     * @return API version number
     */
    virtual uint32_t apiVersion() const { return COMPILATION_POLICY_API_VERSION; }

    // === Lifecycle Hooks ===

    /**
     * @brief Called before code generation begins
     *
     * Use this to analyze the graph and precompute optimization decisions.
     *
     * @param graph The computation graph being compiled
     * @param a The assembler (for emitting setup code if needed)
     */
    virtual void onCompileBegin(const Graph& graph, asmjit::x86::Assembler& a) {
        (void)graph; (void)a;
    }

    /**
     * @brief Called after code generation completes
     *
     * Use this for cleanup or emitting teardown code.
     *
     * @param a The assembler
     */
    virtual void onCompileEnd(asmjit::x86::Assembler& a) {
        (void)a;
    }

    // === Per-Node Callbacks ===

    /**
     * @brief Called before emitting code for a node
     *
     * @param nodeId The node about to be processed
     * @param a The assembler
     */
    virtual void onNodeBegin(NodeId nodeId, asmjit::x86::Assembler& a) {
        (void)nodeId; (void)a;
    }

    /**
     * @brief Called after emitting code for a node
     *
     * @param nodeId The node that was processed
     * @param resultRegister The register holding the result (-1 if none)
     * @param a The assembler
     */
    virtual void onNodeEnd(NodeId nodeId, int resultRegister, asmjit::x86::Assembler& a) {
        (void)nodeId; (void)resultRegister; (void)a;
    }

    // === Allocation Decisions ===

    /**
     * @brief Should this node's result be written to memory immediately?
     *
     * Return false to defer the store (keep value in register longer).
     * The deferStore parameter in ForwardStitcher uses the inverse of this.
     *
     * @param nodeId The node being processed
     * @param graph The computation graph
     * @return true to store immediately, false to defer
     */
    virtual bool requiresStore(NodeId nodeId, const Graph& graph) {
        (void)nodeId; (void)graph;
        return true;  // Default: always store (current behavior)
    }

    /**
     * @brief Preferred register for this node's result
     *
     * Return a specific register index (0-15) to force placement,
     * or -1 to let the allocator decide.
     *
     * @param nodeId The node being processed
     * @return Register index (0-15) or -1 for no preference
     */
    virtual int preferredRegister(NodeId nodeId) {
        (void)nodeId;
        return -1;  // Default: let allocator decide
    }

    /**
     * @brief Check if node's value is already in a register
     *
     * For custom tracking of values across nodes. Return the register
     * index if you know where the value is, or -1 to use normal tracking.
     *
     * @param nodeId The node to look up
     * @return Register index (0-15) or -1 if unknown
     */
    virtual int findValueRegister(NodeId nodeId) {
        (void)nodeId;
        return -1;  // Default: use normal register allocator tracking
    }
};

/**
 * @brief Default compilation policy
 *
 * Preserves current forge behavior: always store results to memory,
 * let allocator decide register placement.
 */
class DefaultCompilationPolicy : public ICompilationPolicy {
    // All methods use base class defaults
};

} // namespace forge
