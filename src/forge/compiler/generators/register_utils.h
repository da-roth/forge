#pragma once

#include "forge/core/computation_graph.h"
#include "constant_pool_manager.h"  // For ConstantInfo
#include "forge/x86/instruction_set.h"     // For IInstructionSet
#include "forge/x86/register_allocator.h"  // For IRegisterAllocator
#include <asmjit/x86.h>
#include <unordered_map>
#include <unordered_set>
#include <initializer_list>

namespace forge::compiler::generators {

/**
 * Utility functions for register management and memory access optimization.
 * These functions provide common register allocation patterns and memory
 * access helpers used throughout the JIT compiler.
 */
class RegisterUtils {
public:
    // Memory load/store result status
    enum class MemLoadResult {
        Success,
        NeedFallback
    };

    /**
     * Flush all dirty registers to memory
     * @param assembler The assembler to generate code with
     * @param regState The register allocator state
     * @param instructionSet The instruction set for optimized operations
     */
    static void flushDirtyRegisters(
        asmjit::x86::Assembler& assembler,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet);

    /**
     * Ensure a value is loaded into a register, with optimized constant handling
     * @param assembler The assembler to generate code with
     * @param nodeId The node ID to load
     * @param regState The register allocator state
     * @param graph The computation graph
     * @param constantMap Map of constant node IDs to constant info
     * @param constPoolLabel Label for the constant pool
     * @param processedConstants Set of constants already processed
     * @param avoid List of register indices to avoid allocating
     * @param instructionSet The instruction set for loading operations
     * @return Register index containing the value
     */
    static int ensureInRegister(
        asmjit::x86::Assembler& assembler,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        const forge::core::ComputationGraph& graph,
        const std::unordered_map<forge::core::NodeId, ConstantInfo>& constantMap,
        const asmjit::Label& constPoolLabel,
        std::unordered_set<forge::core::NodeId>& processedConstants,
        std::initializer_list<int> avoid,
        forge::x86::IInstructionSet* instructionSet);

    /**
     * Try to load a value using optimized memory addressing
     * @param assembler The assembler to generate code with
     * @param dstRegIdx Destination register index
     * @param nodeId The node ID to load
     * @param instructionSet The instruction set for loading
     * @return Success if loaded, NeedFallback if optimization failed
     */
    static MemLoadResult tryOptimizedLoad(
        asmjit::x86::Assembler& assembler,
        int dstRegIdx,
        forge::core::NodeId nodeId,
        forge::x86::IInstructionSet* instructionSet);

    /**
     * Try to store a value using optimized memory addressing
     * @param assembler The assembler to generate code with
     * @param srcRegIdx Source register index
     * @param nodeId The node ID to store to
     * @param instructionSet The instruction set for storing
     * @return Success if stored, NeedFallback if optimization failed
     */
    static MemLoadResult tryOptimizedStore(
        asmjit::x86::Assembler& assembler,
        int srcRegIdx,
        forge::core::NodeId nodeId,
        forge::x86::IInstructionSet* instructionSet);

    /**
     * Check if an offset fits in an 8-bit displacement
     */
    static bool fitsInDisp8(int64_t offset);

    /**
     * Check if an offset fits in a 32-bit displacement
     */
    static bool fitsInDisp32(int64_t offset);
};

} // namespace forge::compiler::generators