#include "boolean_operations.h"
#include "../generators/constant_pool_manager.h"  // For ConstantInfo
#include <stdexcept>

namespace forge::compiler::operations {

using namespace asmjit;
using namespace forge::core;

void BooleanOperations::generateBooleanOperations(
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
        case OpCode::BoolConstant:
            generateBoolConstant(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::BoolAnd:
            generateBoolAnd(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::BoolOr:
            generateBoolOr(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::BoolNot:
            generateBoolNot(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::BoolEq:
        case OpCode::BoolNe:
            generateBoolEqNe(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        default:
            throw std::runtime_error("Unknown boolean operation");
    }
}

void BooleanOperations::generateBoolConstant(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Boolean constant - just load 0.0 or 1.0
    // The value is stored in node.imm
    double value = node.imm;  // Should be 0.0 or 1.0
    
    // Load the constant efficiently
    int resultRegIdx = regState.allocateAvoiding({});
    
    if (value == 0.0) {
        instructionSet->emitZero(a, resultRegIdx);  // Efficient zero
    } else {
        // Load 1.0 using instruction set abstraction
        instructionSet->emitLoadImmediate(a, resultRegIdx, 1.0);
    }
    
    regState.setRegister(resultRegIdx, nodeId, false);
    generators::RegisterUtils::tryOptimizedStore(a, resultRegIdx, nodeId, instructionSet);
}

void BooleanOperations::generateBoolAnd(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Logical AND: can be implemented as multiplication
    // 1.0 * 1.0 = 1.0, 1.0 * 0.0 = 0.0, 0.0 * 0.0 = 0.0
    int aRegIdx = regState.findNodeInRegister(node.a);
    int bRegIdx = regState.findNodeInRegister(node.b);
    
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    
    if (bRegIdx < 0 || bRegIdx == aRegIdx) {
        bRegIdx = ensureInReg(node.b, {aRegIdx});
    }
    
    // Multiply for AND (1.0 * 1.0 = 1.0, any * 0.0 = 0.0)
    instructionSet->emitMul(a, aRegIdx, bRegIdx);
    
    regState.setRegister(aRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
}

void BooleanOperations::generateBoolOr(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Logical OR: a + b - a*b
    // This gives: 0+0-0*0=0, 0+1-0*1=1, 1+0-1*0=1, 1+1-1*1=1
    int aRegIdx = regState.findNodeInRegister(node.a);
    int bRegIdx = regState.findNodeInRegister(node.b);
    
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    regState.lock(aRegIdx);
    
    if (bRegIdx < 0) {
        bRegIdx = ensureInReg(node.b, {aRegIdx});
    }
    
    // Allocate temp register for a*b
    int tempRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx});
    
    // temp = a * b
    instructionSet->emitMove(a, tempRegIdx, aRegIdx);
    instructionSet->emitMul(a, tempRegIdx, bRegIdx);
    
    // result = a + b
    instructionSet->emitAdd(a, aRegIdx, bRegIdx);
    
    // result = result - temp
    instructionSet->emitSub(a, aRegIdx, tempRegIdx);
    
    regState.setRegister(aRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
    regState.unlock(aRegIdx);
}

void BooleanOperations::generateBoolNot(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Logical NOT: 1.0 - a
    // 1.0 - 0.0 = 1.0, 1.0 - 1.0 = 0.0
    int aRegIdx = regState.findNodeInRegister(node.a);
    
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    
    // Allocate register for 1.0
    int oneRegIdx = regState.allocateAvoiding({aRegIdx});
    
    // Load 1.0 using instruction set abstraction
    instructionSet->emitLoadImmediate(a, oneRegIdx, 1.0);
    
    // result = 1.0 - a
    instructionSet->emitSub(a, oneRegIdx, aRegIdx);
    
    regState.setRegister(oneRegIdx, nodeId, false);
    generators::RegisterUtils::tryOptimizedStore(a, oneRegIdx, nodeId, instructionSet);
}

void BooleanOperations::generateBoolEqNe(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Boolean equality/inequality
    // For BoolEq: returns 1.0 if both are equal, 0.0 otherwise
    // For BoolNe: returns 1.0 if different, 0.0 otherwise
    int aRegIdx = regState.findNodeInRegister(node.a);
    int bRegIdx = regState.findNodeInRegister(node.b);
    
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    regState.lock(aRegIdx);
    
    if (bRegIdx < 0) {
        bRegIdx = ensureInReg(node.b, {aRegIdx});
    }
    
    // Allocate result register
    int resultRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx});
    
    // Compare for equality or inequality
    if (node.op == OpCode::BoolEq) {
        instructionSet->emitCmpEQ(a, resultRegIdx, aRegIdx, bRegIdx, regState);
    } else {  // BoolNe
        instructionSet->emitCmpNE(a, resultRegIdx, aRegIdx, bRegIdx, regState);
    }
    
    // Convert to 0.0/1.0
    int oneRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx, resultRegIdx});
    instructionSet->emitLoadImmediate(a, oneRegIdx, 1.0);
    instructionSet->emitAndPD(a, resultRegIdx, oneRegIdx);
    
    regState.setRegister(resultRegIdx, nodeId, false);
    generators::RegisterUtils::tryOptimizedStore(a, resultRegIdx, nodeId, instructionSet);
    regState.unlock(aRegIdx);
}

} // namespace forge::compiler::operations