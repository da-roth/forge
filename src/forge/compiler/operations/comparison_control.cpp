#include "comparison_control.h"
#include "../generators/constant_pool_manager.h"  // For ConstantInfo
#include <stdexcept>

namespace forge::compiler::operations {

using namespace asmjit;
using namespace forge::core;

void ComparisonControl::generateComparisonControl(
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
        case OpCode::Min:
            generateMin(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Max:
            generateMax(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::CmpLT:
        case OpCode::CmpLE:
        case OpCode::CmpGT:
        case OpCode::CmpGE:
        case OpCode::CmpEQ:
        case OpCode::CmpNE:
            generateComparison(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::If:
            generateIf(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        default:
            throw std::runtime_error("Unknown comparison/control operation");
    }
}

void ComparisonControl::generateMin(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Minimum of two values using SSE2 minsd instruction
    int aRegIdx = regState.findNodeInRegister(node.a);
    int bRegIdx = regState.findNodeInRegister(node.b);
    
    // Ensure A is in a register
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    regState.lock(aRegIdx);
    
    // Ensure B is in a different register
    if (bRegIdx < 0 || bRegIdx == aRegIdx) {
        bRegIdx = ensureInReg(node.b, {aRegIdx});
    }
    regState.lock(bRegIdx);
    
    // Use instruction set abstraction for min operation
    instructionSet->emitMin(a, aRegIdx, bRegIdx);
    
    // Update register state and store immediately
    regState.setRegister(aRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
    
    regState.unlock(bRegIdx);
    regState.unlock(aRegIdx);
}

void ComparisonControl::generateMax(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Maximum of two values using SSE2 maxsd instruction
    int aRegIdx = regState.findNodeInRegister(node.a);
    int bRegIdx = regState.findNodeInRegister(node.b);
    
    // Ensure A is in a register
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    regState.lock(aRegIdx);
    
    // Ensure B is in a different register
    if (bRegIdx < 0 || bRegIdx == aRegIdx) {
        bRegIdx = ensureInReg(node.b, {aRegIdx});
    }
    regState.lock(bRegIdx);
    
    // Use instruction set abstraction for max operation
    instructionSet->emitMax(a, aRegIdx, bRegIdx);
    
    // Update register state and store immediately
    regState.setRegister(aRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
    
    regState.unlock(bRegIdx);
    regState.unlock(aRegIdx);
}

void ComparisonControl::generateComparison(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Comparison operators - return 1.0 for true, 0.0 for false
    // Get operands in registers
    int aRegIdx = regState.findNodeInRegister(node.a);
    int bRegIdx = regState.findNodeInRegister(node.b);
    
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    regState.lock(aRegIdx);
    
    if (bRegIdx < 0 || bRegIdx == aRegIdx) {
        bRegIdx = ensureInReg(node.b, {aRegIdx});
    }
    regState.lock(bRegIdx);
    
    // Allocate a result register
    int resultRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx});
    
    // Use instruction set abstraction for comparisons
    switch (node.op) {
        case OpCode::CmpLT:
            instructionSet->emitCmpLT(a, resultRegIdx, aRegIdx, bRegIdx, regState);
            break;
        case OpCode::CmpLE:
            instructionSet->emitCmpLE(a, resultRegIdx, aRegIdx, bRegIdx, regState);
            break;
        case OpCode::CmpGT:
            instructionSet->emitCmpGT(a, resultRegIdx, aRegIdx, bRegIdx, regState);
            break;
        case OpCode::CmpGE:
            instructionSet->emitCmpGE(a, resultRegIdx, aRegIdx, bRegIdx, regState);
            break;
        case OpCode::CmpEQ:
            instructionSet->emitCmpEQ(a, resultRegIdx, aRegIdx, bRegIdx, regState);
            break;
        case OpCode::CmpNE:
            instructionSet->emitCmpNE(a, resultRegIdx, aRegIdx, bRegIdx, regState);
            break;
        default:
            break;
    }
    
    // Convert all-ones/all-zeros to 1.0/0.0
    // cmpsd sets all bits to 1 for true (0xFFFFFFFFFFFFFFFF), 0 for false (0x0000000000000000)
    
    // Simple approach: AND the comparison mask with 1.0
    // Load 1.0 into a temp register
    int oneRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx, resultRegIdx});
    
    // Load 1.0 using instruction set abstraction
    instructionSet->emitLoadImmediate(a, oneRegIdx, 1.0);
    
    // AND: resultReg = resultReg & oneReg
    // If comparison was true (all 1s): 0xFFFFFFFFFFFFFFFF & 0x3FF0000000000000 = 0x3FF0000000000000 (1.0)
    // If comparison was false (all 0s): 0x0000000000000000 & 0x3FF0000000000000 = 0x0000000000000000 (0.0)
    instructionSet->emitAndPD(a, resultRegIdx, oneRegIdx);
    
    // Update register state and store immediately
    regState.setRegister(resultRegIdx, nodeId, false);
    generators::RegisterUtils::tryOptimizedStore(a, resultRegIdx, nodeId, instructionSet);
    
    regState.unlock(bRegIdx);
    regState.unlock(aRegIdx);
}

void ComparisonControl::generateIf(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Conditional selection: condition ? true_val : false_val
    // node.a = condition (Bool, represented as 0.0/1.0)
    // node.b = true value
    // node.c = false value
    
    // Get all three operands in registers
    int condRegIdx = regState.findNodeInRegister(node.a);
    int trueRegIdx = regState.findNodeInRegister(node.b);
    int falseRegIdx = regState.findNodeInRegister(node.c);
    
    if (condRegIdx < 0) {
        condRegIdx = ensureInReg(node.a, {});
    }
    regState.lock(condRegIdx);
    
    if (trueRegIdx < 0) {
        trueRegIdx = ensureInReg(node.b, {condRegIdx});
    }
    regState.lock(trueRegIdx);
    
    if (falseRegIdx < 0) {
        falseRegIdx = ensureInReg(node.c, {condRegIdx, trueRegIdx});
    }
    regState.lock(falseRegIdx);
    
    // Use the instruction set's emitIf function
    // The condition should already be a proper comparison mask (all 1s or all 0s)
    
    // Allocate result register
    int resultRegIdx = regState.allocateAvoiding({condRegIdx, trueRegIdx, falseRegIdx});
    
    // Call the instruction set's conditional operation
    instructionSet->emitIf(a, resultRegIdx, condRegIdx, trueRegIdx, falseRegIdx, regState);
    
    // Store result immediately
    regState.setRegister(resultRegIdx, nodeId, false);
    generators::RegisterUtils::tryOptimizedStore(a, resultRegIdx, nodeId, instructionSet);
    
    regState.unlock(condRegIdx);
    regState.unlock(trueRegIdx);
    regState.unlock(falseRegIdx);
}

} // namespace forge::compiler::operations