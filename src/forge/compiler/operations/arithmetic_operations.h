#pragma once

#include "forge/core/computation_graph.h"
#include "../generators/register_utils.h"
#include "forge/x86/instruction_set.h"
#include "forge/x86/register_allocator.h"
#include "../generators/constant_pool_manager.h"
#include <asmjit/x86.h>
#include <unordered_set>
#include <functional>

namespace forge::compiler::operations {

/**
 * Handles basic arithmetic operations: Add, Sub, Mul, Div, Neg
 * Extracted from AsmStitcher::generateOperation for better maintainability
 */
class ArithmeticOperations {
public:
    /**
     * Generate code for arithmetic operations
     */
    static void generateArithmetic(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        const forge::core::ComputationGraph& graph,
        const std::unordered_map<forge::core::NodeId, generators::ConstantInfo>& constantMap,
        const asmjit::Label& constPoolLabel,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::unordered_set<forge::core::NodeId>& processedConstants,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);

private:
    // Individual operation generators
    static void generateAdd(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateMul(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateSub(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateDiv(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateNeg(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
};

} // namespace forge::compiler::operations