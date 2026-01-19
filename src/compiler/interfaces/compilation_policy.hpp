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
#include "../x86/common/register_allocator_base.hpp"
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
 * Allows full customization of register allocation and memory management
 * strategies during JIT compilation. The default implementation preserves
 * current forge behavior.
 *
 * Two usage patterns are supported:
 *
 * 1. Pre-computed: Analyze the entire graph in onCompileBegin(), compute
 *    register assignments upfront, return them from getRegisterAssignment().
 *
 * 2. Dynamic: Make decisions on-the-fly in getRegisterAssignment() based
 *    on lifespan analysis or other heuristics.
 *
 * Example (pre-computed full control):
 * @code
 * class ManualRegisterPolicy : public ICompilationPolicy {
 *     std::vector<int> assignment_;
 *
 * public:
 *     void onCompileBegin(const Graph& graph, asmjit::x86::Assembler& a) override {
 *         assignment_.resize(graph.nodes.size());
 *         // Pre-compute all register assignments
 *         assignment_[0] = 9;   // Input 0 in register 9
 *         assignment_[1] = 10;  // Input 1 in register 10
 *         assignment_[2] = 0;   // Add result in register 0
 *         // ...
 *     }
 *
 *     int getRegisterAssignment(NodeId nodeId, IRegisterAllocator&) override {
 *         return assignment_[nodeId];  // Full control
 *     }
 *
 *     bool requiresStore(NodeId, const Graph&) override {
 *         return false;  // Keep everything in registers
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

    // === Register Assignment ===

    /**
     * @brief Get the register assignment for a node's value
     *
     * This is the core method for register control. It is called:
     * - When allocating a register for a node's result
     * - When looking up where an operand's value is located
     *
     * Return a specific register index (0-15) for full control,
     * or -1 to use the default allocator's decision.
     *
     * @param nodeId The node whose register assignment is needed
     * @param defaultAllocator The default allocator (for fallback or queries)
     * @return Register index (0-15) for full control, or -1 for default behavior
     */
    virtual int getRegisterAssignment(NodeId nodeId, IRegisterAllocator& defaultAllocator) {
        (void)nodeId; (void)defaultAllocator;
        return -1;  // Default: let allocator decide
    }

    // === Memory Store Decisions ===

    /**
     * @brief Should this node's result be written to memory immediately?
     *
     * Return false to defer the store (keep value in register longer).
     *
     * @param nodeId The node being processed
     * @param graph The computation graph
     * @return true to store immediately, false to defer
     */
    virtual bool requiresStore(NodeId nodeId, const Graph& graph) {
        (void)nodeId; (void)graph;
        return true;  // Default: always store (current behavior)
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
