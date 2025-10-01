#include "register_utils.h"
#include <iostream>
#include <stdexcept>

namespace forge::compiler::generators {

using namespace asmjit;
using namespace forge::core;

void RegisterUtils::flushDirtyRegisters(
    asmjit::x86::Assembler& assembler,
    forge::x86::IRegisterAllocator& regState,
    forge::x86::IInstructionSet* instructionSet) {
    
    for (int i = 0; i < regState.getNumRegisters(); ++i) {
        if (regState.isDirty(i)) {
            int nodeId = regState.getNodeInRegister(i);
            if (nodeId >= 0) {
                // Store the register to memory using optimized addressing
                tryOptimizedStore(assembler, i, static_cast<NodeId>(nodeId), instructionSet);
                regState.markClean(i);
            }
        }
    }
}

int RegisterUtils::ensureInRegister(
    asmjit::x86::Assembler& assembler,
    NodeId nodeId,
    forge::x86::IRegisterAllocator& regState,
    const ComputationGraph& graph,
    const std::unordered_map<NodeId, ConstantInfo>& constantMap,
    const Label& constPoolLabel,
    std::unordered_set<NodeId>& processedConstants,
    std::initializer_list<int> avoid,
    forge::x86::IInstructionSet* instructionSet) {
    
    // First, check if the value is already in a register
    int existingReg = regState.findNodeInRegister(nodeId);
    if (existingReg >= 0) {
        // Value is already in a register, return it
        return existingReg;
    }
    
    // Value not in register, need to load it
    int newReg = regState.allocateAvoiding(avoid);
    
    // If this register was dirty, flush it first
    if (regState.isDirty(newReg)) {
        int oldNodeId = regState.getNodeInRegister(newReg);
        if (oldNodeId >= 0) {
            if (nodeId == 3 || oldNodeId == 3) {
                std::cout << "[DEBUG] Flushing register " << newReg << " which contains node " 
                          << oldNodeId << " before loading node " << nodeId << std::endl;
            }
            tryOptimizedStore(assembler, newReg, static_cast<NodeId>(oldNodeId), instructionSet);
        }
    }
    
    // Check if this is a constant node that needs special handling
    const Node& node = graph.nodes[nodeId];
    if (node.op == OpCode::Constant) {
        // Check if this constant has already been processed
        if (processedConstants.count(nodeId) > 0) {
            // Already processed and stored to memory, just load from there
            tryOptimizedLoad(assembler, newReg, nodeId, instructionSet);
        } else {
            // First time loading this constant - load from constant pool
            auto it = constantMap.find(nodeId);
            if (it != constantMap.end()) {
                if (it->second.value == 0.0) {
                    instructionSet->emitZero(assembler, newReg);
                } else {
                    instructionSet->emitLoadFromConstantPool(assembler, newReg, constPoolLabel, it->second.poolOffset);
                }
                // Store to memory so it's available for later use
                tryOptimizedStore(assembler, newReg, nodeId, instructionSet);
                // Mark as processed
                processedConstants.insert(nodeId);
            } else {
                // Constant not in pool - shouldn't happen
                throw std::runtime_error("Constant node not found in constant pool");
            }
        }
    } else {
        // Load the value from memory normally
        tryOptimizedLoad(assembler, newReg, nodeId, instructionSet);
    }
    
    regState.setRegister(newReg, nodeId, false); // Not dirty since we just loaded it
    
    return newReg;
}

RegisterUtils::MemLoadResult RegisterUtils::tryOptimizedLoad(
    asmjit::x86::Assembler& assembler,
    int dstRegIdx,
    NodeId nodeId,
    forge::x86::IInstructionSet* instructionSet) {
    
    // Calculate offset from base register
    int64_t offset = static_cast<int64_t>(nodeId) * sizeof(double);
    
    // Delegate to instruction set for optimized loading
    instructionSet->emitOptimizedLoad(assembler, dstRegIdx, nodeId);
    
    return MemLoadResult::Success;
}

RegisterUtils::MemLoadResult RegisterUtils::tryOptimizedStore(
    asmjit::x86::Assembler& assembler,
    int srcRegIdx,
    NodeId nodeId,
    forge::x86::IInstructionSet* instructionSet) {
    
    // Calculate offset from base register
    int64_t offset = static_cast<int64_t>(nodeId) * sizeof(double);
    
    // Delegate to instruction set for optimized storing
    instructionSet->emitOptimizedStore(assembler, srcRegIdx, nodeId);
    
    return MemLoadResult::Success;
}

bool RegisterUtils::fitsInDisp8(int64_t offset) {
    return offset >= -128 && offset <= 127;
}

bool RegisterUtils::fitsInDisp32(int64_t offset) {
    return offset >= INT32_MIN && offset <= INT32_MAX;
}

} // namespace forge::compiler::generators