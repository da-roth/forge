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
 * Handles mathematical functions: Abs, Square, Recip, Mod, Sqrt, Exp, Log, Pow, Sin, Cos, Tan
 * Extracted from AsmStitcher::generateOperation for better maintainability
 */
class MathFunctions {
public:
    /**
     * Generate code for mathematical functions
     */
    static void generateMathFunctions(
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
    // Individual function generators
    static void generateAbs(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateSquare(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateRecip(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateMod(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateSqrt(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateExp(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateLog(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generatePow(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateSin(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateCos(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
        
    static void generateTan(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,
        forge::x86::IInstructionSet* instructionSet,
        std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg);
};

} // namespace forge::compiler::operations