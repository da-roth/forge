#include "forward_compiler.h"
#include "generators/register_utils.h"
#include "operations/arithmetic_operations.h"
#include "operations/math_functions.h"
#include "operations/comparison_control.h"
#include "operations/boolean_operations.h"
#include "operations/integer_operations.h"
#include <iostream>

namespace forge::compiler {

using namespace asmjit;
using namespace forge::core;

ForwardCompiler::ForwardCompiler(const forge::x86::CompilerConfig& config, forge::x86::IInstructionSet* instructionSet)
    : config_(config), instructionSet_(instructionSet) {
}

void ForwardCompiler::generateOperation(asmjit::x86::Assembler& a, 
                                      const forge::core::Node& node,
                                      forge::core::NodeId nodeId,
                                      const forge::core::ComputationGraph& graph,
                                      const std::unordered_map<forge::core::NodeId, generators::ConstantInfo>& constantMap,
                                      const asmjit::Label& constPoolLabel,
                                      forge::x86::IRegisterAllocator& regState) {
    
    // Phase 1.4: Minimal set of operations for Linear function
    // Using XMM0-XMM3 as working registers
    
    // Track which constants have been processed to avoid double-processing
    std::unordered_set<forge::core::NodeId> processedConstants;
    
    // Helper lambda to simplify ensureInRegister calls
    auto ensureInReg = [&](forge::core::NodeId nId, std::initializer_list<int> avoid = {}) {
        int reg = generators::RegisterUtils::ensureInRegister(a, nId, regState, graph, constantMap, constPoolLabel, processedConstants, avoid, instructionSet_);
        // Mark constant as processed if it is one (though this is now handled inside ensureInRegister)
        if (graph.nodes[nId].op == OpCode::Constant) {
            processedConstants.insert(nId);
        }
        return reg;
    };
    
    switch (node.op) {
        case OpCode::Input: {
            // Input nodes are already initialized in the workspace at values[nodeId]
            // No code generation needed - the value is already there
            // The test code sets input values directly at the input node's position
            break;
        }
        
        case OpCode::Constant: {
            // Check if this constant was already processed by ensureInRegister
            if (processedConstants.count(nodeId) > 0) {
                // Already processed, skip to avoid double-processing
                break;
            }

            // Check if this constant is already in a pinned register
            int existingReg = regState.findNodeInRegister(nodeId);
            if (existingReg >= 0) {
                // Constant is already preloaded in a pinned register, nothing to do
                break;
            }

            // Phase 2.2: Load from constant pool via RIP-relative addressing
            auto it = constantMap.find(nodeId);
            if (it != constantMap.end()) {
                // Phase 2.3 FIXED: Use new allocator
                int regIdx = regState.allocateAvoiding({});  // No constraints for constants
                // Special case for zero - use XOR instead of loading
                if (it->second.value == 0.0) {
                    instructionSet_->emitZero(a, regIdx);
                } else {
                    // Use instruction set to load from constant pool
                    instructionSet_->emitLoadFromConstantPool(a, regIdx, constPoolLabel, it->second.poolOffset);
                }

                // Mark register as containing this node and store immediately
                regState.setRegister(regIdx, nodeId, false);

                instructionSet_->emitOptimizedStore(a, regIdx, nodeId);

                // Mark as processed
                processedConstants.insert(nodeId);
            }
            break;
        }
        
        case OpCode::Add:
        case OpCode::Sub:
        case OpCode::Mul:
        case OpCode::Div:
        case OpCode::Neg:
            // Delegate to ArithmeticOperations class
            operations::ArithmeticOperations::generateArithmetic(a, node, nodeId, graph, constantMap, constPoolLabel, regState, instructionSet_, processedConstants, ensureInReg);
            break;
        
        case OpCode::Abs:
        case OpCode::Square:
        case OpCode::Recip:
        case OpCode::Mod:
        case OpCode::Sqrt:
        case OpCode::Exp:
        case OpCode::Log:
        case OpCode::Pow:
        case OpCode::Sin:
        case OpCode::Cos:
        case OpCode::Tan:
            // Delegate to MathFunctions class
            operations::MathFunctions::generateMathFunctions(a, node, nodeId, graph, constantMap, constPoolLabel, regState, instructionSet_, processedConstants, ensureInReg);
            break;
        
        // Group 3: Comparison & Control Operations
        case OpCode::Min:
        case OpCode::Max:
        case OpCode::CmpLT:
        case OpCode::CmpLE:
        case OpCode::CmpGT:
        case OpCode::CmpGE:
        case OpCode::CmpEQ:
        case OpCode::CmpNE:
        case OpCode::If:
            // Delegate to ComparisonControl class
            operations::ComparisonControl::generateComparisonControl(a, node, nodeId, graph, constantMap, constPoolLabel, regState, instructionSet_, processedConstants, ensureInReg);
            break;
            
        // Group 4: Boolean Operations
        case OpCode::BoolConstant:
        case OpCode::BoolAnd:
        case OpCode::BoolOr:
        case OpCode::BoolNot:
        case OpCode::BoolEq:
        case OpCode::BoolNe:
            // Delegate to BooleanOperations class
            operations::BooleanOperations::generateBooleanOperations(a, node, nodeId, graph, constantMap, constPoolLabel, regState, instructionSet_, processedConstants, ensureInReg);
            break;
            
        // Group 5: Integer Operations
        case OpCode::IntConstant:
        case OpCode::IntAdd:
        case OpCode::IntSub:
        case OpCode::IntMul:
        case OpCode::IntDiv:
        case OpCode::IntMod:
        case OpCode::IntNeg:
        case OpCode::IntCmpLT:
        case OpCode::IntCmpLE:
        case OpCode::IntCmpGT:
        case OpCode::IntCmpGE:
        case OpCode::IntCmpEQ:
        case OpCode::IntCmpNE:
        case OpCode::IntIf:
            // Delegate to IntegerOperations class
            operations::IntegerOperations::generateIntegerOperations(a, node, nodeId, graph, constantMap, constPoolLabel, regState, instructionSet_, processedConstants, ensureInReg);
            break;
        
        default:
            // For any unimplemented operations, store NaN as a debug aid
            // This helps identify missing operations
            a.xorpd(asmjit::x86::xmm0, asmjit::x86::xmm0);
            a.divsd(asmjit::x86::xmm0, asmjit::x86::xmm0);  // 0/0 = NaN
            a.movsd(asmjit::x86::ptr(asmjit::x86::rdi, nodeId * sizeof(double)), asmjit::x86::xmm0);
            break;
    }
}

} // namespace forge::compiler