// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file gradient_stitcher.hpp
 * @brief Gradient pass code generation for automatic differentiation
 *
 * The GradientStitcher generates x86/x64 assembly code for computing gradients
 * via reverse-mode automatic differentiation (backpropagation). It traverses
 * the computational graph in reverse topological order and accumulates partial
 * derivatives.
 *
 * Thread Safety: Static methods are not thread-safe (use from single thread)
 */

#pragma once

#include "../graph/graph.hpp"
#include "x86/common/register_allocator_base.hpp"  // Use IRegisterAllocator interface
#include "x86/common/compiler_config.hpp"
#include "forge_engine.hpp"
#include "interfaces/instruction_set.hpp"
#include <asmjit/x86.h>
#include <unordered_map>

namespace forge {

/**
 * @brief Code generator for gradient computation (backpropagation)
 *
 * Utility class containing static methods for generating x86/x64 assembly
 * code that computes gradients via reverse-mode automatic differentiation.
 * Implements the chain rule for all supported operations.
 *
 * API Stability: Stable - interface won't change
 */
class GradientStitcher {
public:
    /**
     * @brief Generate assembly for gradient of a single operation
     *
     * Generates code that computes and accumulates partial derivatives for
     * one node in the computational graph using the chain rule.
     *
     * @param a AsmJit assembler for code generation
     * @param node Graph node to compile gradient for
     * @param nodeId ID of the node being compiled
     * @param regState Register allocator state
     * @param graph Full computational graph
     * @param constantMap Mapping of constant nodes to pool offsets
     * @param constPoolLabel Label for constant pool in generated code
     * @param instructionSet Instruction set implementation (SSE2/AVX2)
     * @param config Optional compiler configuration for debug output
     *
     * Thread Safety: Not thread-safe
     */
    static void generateGradientOperation(
        asmjit::x86::Assembler& a,
        const forge::Node& node,
        forge::NodeId nodeId,
        IRegisterAllocator& regState,  // Changed to use interface
        const forge::Graph& graph,
        const std::unordered_map<forge::NodeId, ForgeEngine::ConstantInfo>& constantMap,
        const asmjit::Label& constPoolLabel,
        IInstructionSet* instructionSet,
        const CompilerConfig* config = nullptr
    );

    /**
     * @brief Generate complete gradient pass for entire graph
     *
     * Generates assembly code that computes all gradients for the computational
     * graph in reverse topological order (backpropagation). This is the main
     * entry point for gradient pass code generation.
     *
     * @param a AsmJit assembler for code generation
     * @param graph Computational graph to compile gradients for
     * @param constantMap Mapping of constant nodes to pool offsets
     * @param constPoolLabel Label for constant pool in generated code
     * @param regState Register allocator state
     * @param instructionSet Instruction set implementation (SSE2/AVX2)
     * @param config Optional compiler configuration for debug output
     *
     * Thread Safety: Not thread-safe
     */
    static void stitchGradientPass(
        asmjit::x86::Assembler& a,
        const forge::Graph& graph,
        const std::unordered_map<forge::NodeId, ForgeEngine::ConstantInfo>& constantMap,
        const asmjit::Label& constPoolLabel,
        IRegisterAllocator& regState,  // Changed to use interface
        IInstructionSet* instructionSet,
        const CompilerConfig* config = nullptr
    );
    
    // No private helper methods - all operations go through instruction set abstraction
};

} // namespace forge