#include "math_functions.h"
#include "../generators/constant_pool_manager.h"  // For ConstantInfo
#include <stdexcept>

namespace forge::compiler::operations {

using namespace asmjit;
using namespace forge::core;

void MathFunctions::generateMathFunctions(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    const forge::core::ComputationGraph& graph,
    const std::unordered_map<forge::core::NodeId, generators::ConstantInfo>& constantMap,
    const asmjit::Label& constPoolLabel,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::unordered_set<forge::core::NodeId>& processedConstants,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    switch (node.op) {
        case OpCode::Abs:
            generateAbs(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Square:
            generateSquare(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Recip:
            generateRecip(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Mod:
            generateMod(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Sqrt:
            generateSqrt(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Exp:
            generateExp(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Log:
            generateLog(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Pow:
            generatePow(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Sin:
            generateSin(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Cos:
            generateCos(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Tan:
            generateTan(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        default:
            throw std::runtime_error("Unknown math function operation");
    }
}

void MathFunctions::generateAbs(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Phase 2.3: Register-aware absolute value with proper allocation
    int aRegIdx = regState.findNodeInRegister(node.a);
    
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    regState.lock(aRegIdx);  // Pin operand
    
    // Allocate register for mask, avoiding operand
    int maskRegIdx = regState.allocateAvoiding({aRegIdx});
    
    // Create mask to clear sign bit using instruction set abstraction
    instructionSet->emitCreateAllOnes(a, maskRegIdx);  // All 1s
    instructionSet->emitShiftRight(a, maskRegIdx, 1);   // Clear sign bit
    
    // Perform abs in-place on operand register
    instructionSet->emitAndPD(a, aRegIdx, maskRegIdx);
    
    // Update register state and store immediately
    regState.setRegister(aRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
    regState.unlock(aRegIdx);
}

void MathFunctions::generateSquare(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Phase 2.3: Register-aware squaring with proper allocation
    int aRegIdx = regState.findNodeInRegister(node.a);
    
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    // Square in-place
    instructionSet->emitSquare(a, aRegIdx);
    
    // Update register state and store immediately
    regState.setRegister(aRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
}

void MathFunctions::generateRecip(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Phase 2.3: Register-aware reciprocal with proper allocation
    int aRegIdx = regState.findNodeInRegister(node.a);
    
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    regState.lock(aRegIdx);  // Pin operand
    
    // Allocate register for 1.0, avoiding operand
    int oneRegIdx = regState.allocateAvoiding({aRegIdx});
    
    // Load 1.0 using instruction set abstraction
    instructionSet->emitLoadImmediate(a, oneRegIdx, 1.0);
    
    // Divide 1.0 by operand (result goes in oneReg)
    instructionSet->emitDiv(a, oneRegIdx, aRegIdx);
    
    regState.setRegister(oneRegIdx, nodeId, false);
    generators::RegisterUtils::tryOptimizedStore(a, oneRegIdx, nodeId, instructionSet);
    regState.unlock(aRegIdx);
}

void MathFunctions::generateMod(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Phase 2.3: Register-aware modulo with proper allocation
    int aRegIdx = regState.findNodeInRegister(node.a);
    int bRegIdx = regState.findNodeInRegister(node.b);
    
    // Ensure A is in a register
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    regState.lock(aRegIdx);  // Pin A
    
    // Ensure B is in a different register
    if (bRegIdx < 0 || bRegIdx == aRegIdx) {
        bRegIdx = ensureInReg(node.b, {aRegIdx});
    }
    regState.lock(bRegIdx);  // Pin B
    
    // Use instruction set to emit modulo operation
    instructionSet->emitMod(a, aRegIdx, bRegIdx, regState);
    
    // Update register state and store immediately
    regState.setRegister(aRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
    
    regState.unlock(bRegIdx);
    regState.unlock(aRegIdx);
}

void MathFunctions::generateSqrt(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Square root using native SSE2 instruction
    int aRegIdx = regState.findNodeInRegister(node.a);
    
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    // Use instruction set abstraction for sqrt
    instructionSet->emitSqrt(a, aRegIdx);
    
    // Update register state and store immediately
    regState.setRegister(aRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
}

void MathFunctions::generateExp(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    int aRegIdx = regState.findNodeInRegister(node.a);
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    
    int resultRegIdx = regState.allocateAvoiding({});
    instructionSet->emitExp(a, resultRegIdx, aRegIdx, regState);
    
    regState.setRegister(resultRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
}

void MathFunctions::generateLog(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    int aRegIdx = regState.findNodeInRegister(node.a);
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    
    int resultRegIdx = regState.allocateAvoiding({});
    instructionSet->emitLog(a, resultRegIdx, aRegIdx, regState);
    
    regState.setRegister(resultRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
}

void MathFunctions::generatePow(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Handle pow like other transcendental functions - let instruction set handle register management
    int baseRegIdx = regState.findNodeInRegister(node.a);
    if (baseRegIdx < 0) {
        baseRegIdx = ensureInReg(node.a, {});
    }
    
    int expRegIdx = regState.findNodeInRegister(node.b);
    if (expRegIdx < 0) {
        expRegIdx = ensureInReg(node.b, {baseRegIdx});
    }
    
    int resultRegIdx = regState.allocateAvoiding({});
    instructionSet->emitPow(a, resultRegIdx, baseRegIdx, expRegIdx, regState);
    
    // Store result immediately
    regState.setRegister(resultRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
}

void MathFunctions::generateSin(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    int aRegIdx = regState.findNodeInRegister(node.a);
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    
    int resultRegIdx = regState.allocateAvoiding({});
    instructionSet->emitSin(a, resultRegIdx, aRegIdx, regState);
    
    regState.setRegister(resultRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
}

void MathFunctions::generateCos(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    int aRegIdx = regState.findNodeInRegister(node.a);
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    
    int resultRegIdx = regState.allocateAvoiding({});
    instructionSet->emitCos(a, resultRegIdx, aRegIdx, regState);
    
    regState.setRegister(resultRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
}

void MathFunctions::generateTan(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    int aRegIdx = regState.findNodeInRegister(node.a);
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    
    int resultRegIdx = regState.allocateAvoiding({});
    instructionSet->emitTan(a, resultRegIdx, aRegIdx, regState);
    
    regState.setRegister(resultRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
}

} // namespace forge::compiler::operations