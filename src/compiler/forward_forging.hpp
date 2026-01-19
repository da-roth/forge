// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file forward_forging.hpp
 * @brief Forward forging - code generation for JIT compiler
 *
 * The ForwardForging generates x86/x64 assembly code for evaluating the
 * forward pass of mathematical expression graphs. It translates graph
 * operations into optimized machine code using register allocation and
 * SIMD instructions.
 *
 * Thread Safety: Static methods are not thread-safe (use from single thread)
 */

#pragma once

#include "../graph/graph.hpp"
#include "interfaces/register_allocator.hpp"  // IRegisterAllocator interface
#include "x86/common/compiler_config.hpp"
#include "forge_engine.hpp"
#include "interfaces/instruction_set.hpp"
#include "interfaces/compilation_policy.hpp"
#include <asmjit/x86.h>
#include <unordered_map>

namespace forge {

/**
 * @brief Code generator for forward pass evaluation
 *
 * Utility class containing static methods for generating x86/x64 assembly
 * code that evaluates mathematical expressions. Works with both SSE2 scalar
 * and AVX2 packed instruction sets.
 *
 * API Stability: Stable - interface won't change
 */
class ForwardForging {
public:
    /**
     * @brief Generate assembly for a single graph operation
     *
     * Translates one node in the computational graph into x86/x64 assembly
     * instructions, handling register allocation and instruction selection.
     *
     * @param a AsmJit assembler for code generation
     * @param node Graph node to compile
     * @param nodeId ID of the node being compiled
     * @param graph Full computational graph
     * @param constantMap Mapping of constant nodes to pool offsets
     * @param constPoolLabel Label for constant pool in generated code
     * @param regState Register allocator state
     * @param instructionSet Instruction set implementation (SSE2/AVX2)
     * @param policy Compilation policy for register decisions (nullptr for default)
     * @param deferStore If true, keep result in register without storing
     *
     * Thread Safety: Not thread-safe
     */
    static void generateForwardOperation(
        asmjit::x86::Assembler& a,
        const forge::Node& node,
        forge::NodeId nodeId,
        const forge::Graph& graph,
        const std::unordered_map<forge::NodeId, ForgeEngine::ConstantInfo>& constantMap,
        const asmjit::Label& constPoolLabel,
        IRegisterAllocator& regState,
        IInstructionSet* instructionSet,
        ICompilationPolicy* policy = nullptr,
        bool deferStore = false
    );

    /**
     * @brief Generate complete forward pass for entire graph
     *
     * Generates assembly code that evaluates all operations in the computational
     * graph in topological order. This is the main entry point for forward pass
     * code generation.
     *
     * @param a AsmJit assembler for code generation
     * @param graph Computational graph to compile
     * @param constantMap Mapping of constant nodes to pool offsets
     * @param constPoolLabel Label for constant pool in generated code
     * @param regState Register allocator state
     * @param instructionSet Instruction set implementation (SSE2/AVX2)
     * @param config Optional compiler configuration for debug output
     *
     * Thread Safety: Not thread-safe
     */
    static void forgeForwardPass(
        asmjit::x86::Assembler& a,
        const forge::Graph& graph,
        const std::unordered_map<forge::NodeId, ForgeEngine::ConstantInfo>& constantMap,
        const asmjit::Label& constPoolLabel,
        IRegisterAllocator& regState,
        IInstructionSet* instructionSet,
        const CompilerConfig* config = nullptr
    );

    /**
     * @brief Generate function prologue (currently placeholder)
     * @param a AsmJit assembler for code generation
     *
     * Thread Safety: Not thread-safe
     */
    static void generatePrologue(asmjit::x86::Assembler& a);

    /**
     * @brief Generate function epilogue (currently placeholder)
     * @param a AsmJit assembler for code generation
     *
     * Thread Safety: Not thread-safe
     */
    static void generateEpilogue(asmjit::x86::Assembler& a);

private:
    // Helper to ensure a value is in a register
    static int ensureInRegister(
        asmjit::x86::Assembler& a,
        forge::NodeId nodeId,
        IRegisterAllocator& regState,
        const forge::Graph& graph,
        const std::unordered_map<forge::NodeId, ForgeEngine::ConstantInfo>& constantMap,
        const asmjit::Label& constPoolLabel,
        std::unordered_set<forge::NodeId>& processedConstants,
        IInstructionSet* instructionSet,
        ICompilationPolicy* policy,
        std::initializer_list<int> avoid
    );
};

} // namespace forge
