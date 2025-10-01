#include "arithmetic_operations.h"
#include "../generators/constant_pool_manager.h"  // For ConstantInfo
#include <stdexcept>

namespace forge::compiler::operations {

using namespace asmjit;
using namespace forge::core;

void ArithmeticOperations::generateArithmetic(
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
        case OpCode::Add:
            generateAdd(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Mul:
            generateMul(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Sub:
            generateSub(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Div:
            generateDiv(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        case OpCode::Neg:
            generateNeg(a, node, nodeId, regState, instructionSet, ensureInReg);
            break;
        default:
            throw std::runtime_error("Unknown arithmetic operation");
    }
}

void ArithmeticOperations::generateAdd(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Simplified approach: just ensure both operands are in registers and do the add
    int aReg = ensureInReg(node.a, {});
    regState.lock(aReg);
    int bReg = ensureInReg(node.b, {aReg});
    
    // Perform addition: aReg = aReg + bReg
    instructionSet->emitAdd(a, aReg, bReg);
    
    // Update register state and store immediately
    regState.setRegister(aReg, nodeId, false);
    generators::RegisterUtils::tryOptimizedStore(a, aReg, nodeId, instructionSet);
    
    regState.unlock(aReg);
}

void ArithmeticOperations::generateMul(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Phase 2.3: BinSel pattern for clean operand selection
    struct BinSel {
        int dstIdx, rhsIdx;
        NodeId dstId, rhsId;
    };
    
    // Reuse commutative selector logic
    auto selectCommutative = [&](NodeId aId, NodeId bId) -> BinSel {
        int aIdx = regState.findNodeInRegister(aId);
        int bIdx = regState.findNodeInRegister(bId);
        
        BinSel s{};
        if (aIdx >= 0) {
            s.dstIdx = aIdx; s.dstId = aId;
            s.rhsIdx = bIdx; s.rhsId = bId;
        } else if (bIdx >= 0) {
            s.dstIdx = bIdx; s.dstId = bId;
            s.rhsIdx = aIdx; s.rhsId = aId;
        } else {
            s.dstIdx = ensureInReg(aId, {});
            s.dstId = aId;
            s.rhsIdx = -1; s.rhsId = bId;
        }
        
        regState.lock(s.dstIdx);
        if (s.rhsIdx < 0 || s.rhsIdx == s.dstIdx) {
            s.rhsIdx = ensureInReg(s.rhsId, {s.dstIdx});
        }
        regState.lock(s.rhsIdx);
        return s;
    };
    
    auto s = selectCommutative(node.a, node.b);
    instructionSet->emitMul(a, s.dstIdx, s.rhsIdx);
    regState.setRegister(s.dstIdx, nodeId, false);
    // Store immediately
    generators::RegisterUtils::tryOptimizedStore(a, s.dstIdx, nodeId, instructionSet);
    regState.unlock(s.rhsIdx);
    regState.unlock(s.dstIdx);
}

void ArithmeticOperations::generateSub(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Phase 2.3: BinSel pattern for non-commutative op (A must be dst)
    struct BinSel {
        int dstIdx, rhsIdx;
        NodeId dstId, rhsId;
    };
    
    // Non-commutative selector - A must be destination
    auto selectNonCommutative = [&](NodeId aId, NodeId bId) -> BinSel {
        BinSel s{};
        int aIdx = regState.findNodeInRegister(aId);
        int bIdx = regState.findNodeInRegister(bId);
        
        // A must be destination
        if (aIdx < 0) {
            aIdx = ensureInReg(aId, {});
        }
        s.dstIdx = aIdx; s.dstId = aId;
        
        regState.lock(s.dstIdx);
        
        // B must be in a different register
        if (bIdx < 0 || bIdx == s.dstIdx) {
            bIdx = ensureInReg(bId, {s.dstIdx});
        }
        s.rhsIdx = bIdx; s.rhsId = bId;
        regState.lock(s.rhsIdx);
        return s;
    };
    
    auto s = selectNonCommutative(node.a, node.b);
    instructionSet->emitSub(a, s.dstIdx, s.rhsIdx);
    regState.setRegister(s.dstIdx, nodeId, false);
    // Store immediately
    generators::RegisterUtils::tryOptimizedStore(a, s.dstIdx, nodeId, instructionSet);
    regState.unlock(s.rhsIdx);
    regState.unlock(s.dstIdx);
}

void ArithmeticOperations::generateDiv(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Phase 2.3: BinSel pattern for non-commutative op (A must be dst)
    struct BinSel {
        int dstIdx, rhsIdx;
        NodeId dstId, rhsId;
    };
    
    // Reuse non-commutative selector logic
    auto selectNonCommutative = [&](NodeId aId, NodeId bId) -> BinSel {
        BinSel s{};
        int aIdx = regState.findNodeInRegister(aId);
        int bIdx = regState.findNodeInRegister(bId);
        
        if (aIdx < 0) {
            aIdx = ensureInReg(aId, {});
        }
        s.dstIdx = aIdx; s.dstId = aId;
        
        regState.lock(s.dstIdx);
        
        if (bIdx < 0 || bIdx == s.dstIdx) {
            bIdx = ensureInReg(bId, {s.dstIdx});
        }
        s.rhsIdx = bIdx; s.rhsId = bId;
        regState.lock(s.rhsIdx);
        return s;
    };
    
    auto s = selectNonCommutative(node.a, node.b);
    instructionSet->emitDiv(a, s.dstIdx, s.rhsIdx);
    regState.setRegister(s.dstIdx, nodeId, false);
    // Store immediately
    generators::RegisterUtils::tryOptimizedStore(a, s.dstIdx, nodeId, instructionSet);
    regState.unlock(s.rhsIdx);
    regState.unlock(s.dstIdx);
}

void ArithmeticOperations::generateNeg(
    asmjit::x86::Assembler& a,
    const forge::core::Node& node,
    forge::core::NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet,
    std::function<int(forge::core::NodeId, std::initializer_list<int>)> ensureInReg) {
    
    // Simplified negation: multiply by -1
    int aRegIdx = regState.findNodeInRegister(node.a);
    if (aRegIdx < 0) {
        aRegIdx = ensureInReg(node.a, {});
    }
    
    regState.lock(aRegIdx);
    
    // Allocate register for -1.0 constant
    int negOneRegIdx = regState.allocateAvoiding({aRegIdx});
    
    // Load -1.0 into the register
    instructionSet->emitLoadImmediate(a, negOneRegIdx, -1.0);
    
    // Multiply: aRegIdx = aRegIdx * (-1.0)
    instructionSet->emitMul(a, aRegIdx, negOneRegIdx);
    
    // Update register state and store immediately
    regState.setRegister(aRegIdx, nodeId, false);
    instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
    regState.unlock(aRegIdx);
}

} // namespace forge::compiler::operations