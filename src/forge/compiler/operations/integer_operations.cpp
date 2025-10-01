#include "integer_operations.h"
#include "../generators/constant_pool_manager.h"  // For ConstantInfo
#include <stdexcept>

namespace forge::compiler::operations {

using namespace asmjit;
using namespace forge::core;

void IntegerOperations::generateIntegerOperations(
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
        case OpCode::IntConstant:
            generateIntConstant(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::IntAdd:
        case OpCode::IntSub:
        case OpCode::IntMul:
        case OpCode::IntDiv:
        case OpCode::IntMod:
        case OpCode::IntNeg:
            generateIntArithmetic(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::IntCmpLT:
        case OpCode::IntCmpLE:
        case OpCode::IntCmpGT:
        case OpCode::IntCmpGE:
        case OpCode::IntCmpEQ:
        case OpCode::IntCmpNE:
            generateIntComparison(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::IntIf:
            generateIntIf(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        default:
            throw std::runtime_error("Unknown integer operation");
    }
}

void IntegerOperations::generateIntConstant(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Integer constant stored as double in node.imm
    double value = node.imm;  // Integer stored as double
    
    int resultRegIdx = regState.allocateAvoiding({});
    
    if (value == 0.0) {
        instructionSet->emitZero(a, resultRegIdx);  // Efficient zero
    } else {
        // Load the integer value (stored as double)
        instructionSet->emitLoadImmediate(a, resultRegIdx, value);
    }
    
    regState.setRegister(resultRegIdx, nodeId, false);
    generators::RegisterUtils::tryOptimizedStore(a, resultRegIdx, nodeId, instructionSet);
}

void IntegerOperations::generateIntArithmetic(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // TODO: Implement full integer arithmetic with proper truncation
    // This is a placeholder - the original implementation is quite complex (~200 lines)
    // involving proper integer truncation using emitRound with mode 3
    
    switch (node.op) {
        case OpCode::IntAdd:
            // Simplified version - would need full truncation logic
            {
                int aReg = ensureInReg(node.a, {});
                regState.lock(aReg);
                int bReg = ensureInReg(node.b, {aReg});
                instructionSet->emitAdd(a, aReg, bReg);
                regState.setRegister(aReg, nodeId, false);
                generators::RegisterUtils::tryOptimizedStore(a, aReg, nodeId, instructionSet);
                regState.unlock(aReg);
            }
            break;
        case OpCode::IntSub:
        case OpCode::IntMul:
        case OpCode::IntDiv:
        case OpCode::IntMod:
        case OpCode::IntNeg:
            throw std::runtime_error("Integer arithmetic operation not yet fully implemented");
            break;
        default:
            throw std::runtime_error("Unknown integer arithmetic operation");
    }
}

void IntegerOperations::generateIntComparison(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // TODO: Implement integer comparisons
    // Similar to regular comparisons but with integer truncation
    throw std::runtime_error("Integer comparison operations not yet implemented");
}

void IntegerOperations::generateIntIf(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // TODO: Implement integer conditional selection
    // Similar to regular If but for integer values
    throw std::runtime_error("Integer If operation not yet implemented");
}

} // namespace forge::compiler::operations