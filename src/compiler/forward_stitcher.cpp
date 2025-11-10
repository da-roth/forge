// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file forward_stitcher.cpp
 * @brief Implementation of forward pass code generation for JIT compilation
 *
 * Generates x86/x64 assembly code for the forward evaluation pass of
 * mathematical expression graphs.
 */

#include "forward_stitcher.hpp"
#include <stdexcept>
#include <iostream>

namespace forge {

using namespace forge;
using namespace asmjit;

void ForwardStitcher::generatePrologue(x86::Assembler& a) {
    // This method will need the instruction set, which should be passed as a parameter
    // For now, this is a placeholder - the actual implementation will delegate to instruction set
    // This will be addressed when we refactor the call sites
}

void ForwardStitcher::generateEpilogue(x86::Assembler& a) {
    // This method will need the instruction set, which should be passed as a parameter
    // For now, this is a placeholder - the actual implementation will delegate to instruction set
    // This will be addressed when we refactor the call sites
}

void ForwardStitcher::generateForwardOperation(
    asmjit::x86::Assembler& a,
    const forge::Node& node,
    forge::NodeId nodeId,
    const forge::Graph& graph,
    const std::unordered_map<forge::NodeId, ForgeEngine::ConstantInfo>& constantMap,
    const asmjit::Label& constPoolLabel,
    IRegisterAllocator& regState,
    IInstructionSet* instructionSet,
    bool deferStore
) {
    // Phase 1.4: Minimal set of operations for Linear function
    // Using XMM0-XMM3 as working registers

    // Track which constants have been processed to avoid double-processing
    std::unordered_set<NodeId> processedConstants;

    // Helper lambda to simplify ensureInRegister calls
    auto ensureInReg = [&](NodeId nId, std::initializer_list<int> avoid = {}) {
        int reg = ensureInRegister(a, nId, regState, graph, constantMap, constPoolLabel, processedConstants, instructionSet, avoid);
        // Mark constant as processed if it is one (though this is now handled inside ensureInRegister)
        if (graph.nodes[nId].op == OpCode::Constant) {
            processedConstants.insert(nId);
        }
        return reg;
    };

    // Helper to try optimized store
    auto tryOptimizedStore = [&](int srcRegIdx, NodeId nId) {
        // Calculate offset from base register
        int64_t offset = static_cast<int64_t>(nId) * sizeof(double);

        // Delegate to instruction set for optimized storing
        instructionSet->emitOptimizedStore(a, srcRegIdx, nId);
    };

    switch (node.op) {
        case OpCode::Input: {
            // Input nodes are already initialized in the workspace at values[nodeId]
            // No code generation needed - the value is already there
            // The test code sets input values directly at the input node's position
            break;
        }

        case OpCode::Constant: {
            // Debug constant processing
            if (nodeId == 3) {
                std::cout << "[DEBUG] Processing Constant node 3, isActive=" << node.isActive
                          << ", isDead=" << node.isDead << ", value=" << node.imm << std::endl;
            }

            // Check if this constant was already processed by ensureInRegister
            if (processedConstants.count(nodeId) > 0) {
                // Already processed, skip to avoid double-processing
                if (nodeId == 3) {
                    std::cout << "[DEBUG] Constant node 3 already processed, skipping" << std::endl;
                }
                break;
            }

            // Check if this constant is already in a pinned register
            int existingReg = regState.findNodeInRegister(nodeId);
            if (existingReg >= 0) {
                // Constant is already preloaded in a pinned register, nothing to do
                if (nodeId == 3) {
                    std::cout << "[DEBUG] Constant node 3 already in register " << existingReg << std::endl;
                }
                break;
            }

            // Phase 2.2: Load from constant pool via RIP-relative addressing
            auto it = constantMap.find(nodeId);
            if (it != constantMap.end()) {
                // Phase 2.3 FIXED: Use new allocator
                int regIdx = regState.allocateAvoiding({});  // No constraints for constants
                // Special case for zero - use XOR instead of loading
                if (it->second.value == 0.0) {
                    instructionSet->emitZero(a, regIdx);
                } else {
                    // Use instruction set to load from constant pool
                    instructionSet->emitLoadFromConstantPool(a, regIdx, constPoolLabel, it->second.poolOffset);
                }

                // Mark register as containing this node
                // Note: isDirty should be false after storing, true if store is deferred
                regState.setRegister(regIdx, nodeId, deferStore);

                // Store to values[nodeId] unless deferred
                if (!deferStore) {
                    if (nodeId == 3) {
                        std::cout << "[DEBUG] Storing constant node 3 to memory from register " << regIdx << std::endl;
                    }
                    instructionSet->emitOptimizedStore(a, regIdx, nodeId);
                }

                // Mark as processed
                processedConstants.insert(nodeId);
            }
            break;
        }

        case OpCode::Add: {
            // Simplified approach: just ensure both operands are in registers and do the add
            int aReg = ensureInReg(node.a, {});
            regState.lock(aReg);
            int bReg = ensureInReg(node.b, {aReg});

            // Perform addition: aReg = aReg + bReg
            instructionSet->emitAdd(a, aReg, bReg);

            // Update register state
            regState.setRegister(aReg, nodeId, deferStore);

            // Store immediately or defer based on parameter
            if (!deferStore) {
                tryOptimizedStore(aReg, nodeId);
            }

            regState.unlock(aReg);
            break;
        }

        case OpCode::Mul: {
            // Phase 2.3: BinSel pattern for clean operand selection
            struct BinSel {
                int dstIdx, rhsIdx;
                forge::NodeId dstId, rhsId;
            };

            // Reuse commutative selector logic
            auto selectCommutative = [&](forge::NodeId aId, forge::NodeId bId) -> BinSel {
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
                    s.dstIdx = ensureInReg( aId, {});
                    s.dstId = aId;
                    s.rhsIdx = -1; s.rhsId = bId;
                }

                regState.lock(s.dstIdx);
                if (s.rhsIdx < 0 || s.rhsIdx == s.dstIdx) {
                    s.rhsIdx = ensureInReg( s.rhsId, {s.dstIdx});
                }
                regState.lock(s.rhsIdx);
                return s;
            };

            auto s = selectCommutative(node.a, node.b);
            instructionSet->emitMul(a, s.dstIdx, s.rhsIdx);
            regState.setRegister(s.dstIdx, nodeId, deferStore);
            // Store immediately or defer based on parameter
            if (!deferStore) {
                tryOptimizedStore(s.dstIdx, nodeId);
            }
            regState.unlock(s.rhsIdx);
            regState.unlock(s.dstIdx);
            break;
        }

        case OpCode::Sub: {
            // Phase 2.3: BinSel pattern for non-commutative op (A must be dst)
            struct BinSel {
                int dstIdx, rhsIdx;
                forge::NodeId dstId, rhsId;
            };

            // Non-commutative selector - A must be destination
            auto selectNonCommutative = [&](forge::NodeId aId, forge::NodeId bId) -> BinSel {
                BinSel s{};
                int aIdx = regState.findNodeInRegister(aId);
                int bIdx = regState.findNodeInRegister(bId);

                // A must be destination
                if (aIdx < 0) {
                    aIdx = ensureInReg( aId, {});
                }
                s.dstIdx = aIdx; s.dstId = aId;

                regState.lock(s.dstIdx);

                // B must be in a different register
                if (bIdx < 0 || bIdx == s.dstIdx) {
                    bIdx = ensureInReg( bId, {s.dstIdx});
                }
                s.rhsIdx = bIdx; s.rhsId = bId;
                regState.lock(s.rhsIdx);
                return s;
            };

            auto s = selectNonCommutative(node.a, node.b);
            instructionSet->emitSub(a, s.dstIdx, s.rhsIdx);
            regState.setRegister(s.dstIdx, nodeId, deferStore);
            // Store immediately or defer based on parameter
            if (!deferStore) {
                tryOptimizedStore(s.dstIdx, nodeId);
            }
            regState.unlock(s.rhsIdx);
            regState.unlock(s.dstIdx);
            break;
        }

        case OpCode::Div: {
            // Phase 2.3: BinSel pattern for non-commutative op (A must be dst)
            struct BinSel {
                int dstIdx, rhsIdx;
                forge::NodeId dstId, rhsId;
            };

            // Reuse non-commutative selector logic
            auto selectNonCommutative = [&](forge::NodeId aId, forge::NodeId bId) -> BinSel {
                BinSel s{};
                int aIdx = regState.findNodeInRegister(aId);
                int bIdx = regState.findNodeInRegister(bId);

                if (aIdx < 0) {
                    aIdx = ensureInReg( aId, {});
                }
                s.dstIdx = aIdx; s.dstId = aId;

                regState.lock(s.dstIdx);

                if (bIdx < 0 || bIdx == s.dstIdx) {
                    bIdx = ensureInReg( bId, {s.dstIdx});
                }
                s.rhsIdx = bIdx; s.rhsId = bId;
                regState.lock(s.rhsIdx);
                return s;
            };

            auto s = selectNonCommutative(node.a, node.b);
            instructionSet->emitDiv(a, s.dstIdx, s.rhsIdx);
            regState.setRegister(s.dstIdx, nodeId, deferStore);
            // Store immediately or defer based on parameter
            if (!deferStore) {
                tryOptimizedStore(s.dstIdx, nodeId);
            }
            regState.unlock(s.rhsIdx);
            regState.unlock(s.dstIdx);
            break;
        }

        case OpCode::Neg: {
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

            // Update register state and store
            regState.setRegister(aRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
            }
            regState.unlock(aRegIdx);
            break;
        }

        case OpCode::Abs: {
            // Phase 2.3: Register-aware absolute value with proper allocation
            int aRegIdx = regState.findNodeInRegister(node.a);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);  // Pin operand

            // Allocate register for mask, avoiding operand
            int maskRegIdx = regState.allocateAvoiding({aRegIdx});

            // Create mask to clear sign bit using instruction set abstraction
            instructionSet->emitCreateAllOnes(a, maskRegIdx);  // All 1s
            instructionSet->emitShiftRight(a, maskRegIdx, 1);   // Clear sign bit

            // Perform abs in-place on operand register
            instructionSet->emitAndPD(a, aRegIdx, maskRegIdx);

            // Update register state and store
            regState.setRegister(aRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
            }
            regState.unlock(aRegIdx);
            break;
        }

        case OpCode::Square: {
            // Phase 2.3: Register-aware squaring with proper allocation
            int aRegIdx = regState.findNodeInRegister(node.a);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            // Square in-place
            instructionSet->emitSquare(a, aRegIdx);

            // Update register state and store
            regState.setRegister(aRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
            }
            break;
        }

        case OpCode::Recip: {
            // Phase 2.3: Register-aware reciprocal with proper allocation
            int aRegIdx = regState.findNodeInRegister(node.a);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);  // Pin operand

            // Allocate register for 1.0, avoiding operand
            int oneRegIdx = regState.allocateAvoiding({aRegIdx});

            // Load 1.0 using instruction set abstraction
            instructionSet->emitLoadImmediate(a, oneRegIdx, 1.0);

            // Divide 1.0 by operand (result goes in oneReg)
            instructionSet->emitDiv(a, oneRegIdx, aRegIdx);

            regState.setRegister(oneRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(oneRegIdx, nodeId);
            }
            regState.unlock(aRegIdx);
            break;
        }

        case OpCode::Mod: {
            // Phase 2.3: Register-aware modulo with proper allocation
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            // Ensure A is in a register
            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);  // Pin A

            // Ensure B is in a different register
            if (bRegIdx < 0 || bRegIdx == aRegIdx) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
            }
            regState.lock(bRegIdx);  // Pin B

            // Use instruction set to emit modulo operation
            instructionSet->emitMod(a, aRegIdx, bRegIdx, regState);

            // Update register state and store
            regState.setRegister(aRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
            }

            regState.unlock(bRegIdx);
            regState.unlock(aRegIdx);
            break;
        }

        case OpCode::Sqrt: {
            // Square root using native SSE2 instruction
            int aRegIdx = regState.findNodeInRegister(node.a);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            // Use instruction set abstraction for sqrt
            instructionSet->emitSqrt(a, aRegIdx);

            // Update register state and store
            regState.setRegister(aRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
            }
            break;
        }

        case OpCode::Exp: {
            int aRegIdx = regState.findNodeInRegister(node.a);
            if (aRegIdx < 0) {
                aRegIdx = ensureInReg(node.a, {});
            }

            int resultRegIdx = regState.allocateAvoiding({});
            instructionSet->emitExp(a, resultRegIdx, aRegIdx, regState);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
            }
            break;
        }

        case OpCode::Log: {
            int aRegIdx = regState.findNodeInRegister(node.a);
            if (aRegIdx < 0) {
                aRegIdx = ensureInReg(node.a, {});
            }

            int resultRegIdx = regState.allocateAvoiding({});
            instructionSet->emitLog(a, resultRegIdx, aRegIdx, regState);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
            }
            break;
        }

        case OpCode::Pow: {
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

            // Store result
            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
            }
            break;
        }

        case OpCode::Sin: {
            int aRegIdx = regState.findNodeInRegister(node.a);
            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }

            int resultRegIdx = regState.allocateAvoiding({});
            instructionSet->emitSin(a, resultRegIdx, aRegIdx, regState);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
            }
            break;
        }

        case OpCode::Cos: {
            int aRegIdx = regState.findNodeInRegister(node.a);
            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }

            int resultRegIdx = regState.allocateAvoiding({});
            instructionSet->emitCos(a, resultRegIdx, aRegIdx, regState);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
            }
            break;
        }

        case OpCode::Tan: {
            int aRegIdx = regState.findNodeInRegister(node.a);
            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }

            int resultRegIdx = regState.allocateAvoiding({});
            instructionSet->emitTan(a, resultRegIdx, aRegIdx, regState);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, resultRegIdx, nodeId);
            }
            break;
        }

        case OpCode::Min: {
            // Minimum of two values using SSE2 minsd instruction
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            // Ensure A is in a register
            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);

            // Ensure B is in a different register
            if (bRegIdx < 0 || bRegIdx == aRegIdx) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
            }
            regState.lock(bRegIdx);

            // Use instruction set abstraction for min operation
            instructionSet->emitMin(a, aRegIdx, bRegIdx);

            // Update register state and store
            regState.setRegister(aRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
            }

            regState.unlock(bRegIdx);
            regState.unlock(aRegIdx);
            break;
        }

        case OpCode::Max: {
            // Maximum of two values using SSE2 maxsd instruction
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            // Ensure A is in a register
            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);

            // Ensure B is in a different register
            if (bRegIdx < 0 || bRegIdx == aRegIdx) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
            }
            regState.lock(bRegIdx);

            // Use instruction set abstraction for max operation
            instructionSet->emitMax(a, aRegIdx, bRegIdx);

            // Update register state and store
            regState.setRegister(aRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
            }

            regState.unlock(bRegIdx);
            regState.unlock(aRegIdx);
            break;
        }

        // Comparison operators - return 1.0 for true, 0.0 for false
        case OpCode::CmpLT:
        case OpCode::CmpLE:
        case OpCode::CmpGT:
        case OpCode::CmpGE:
        case OpCode::CmpEQ:
        case OpCode::CmpNE: {
            // Get operands in registers
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);

            if (bRegIdx < 0 || bRegIdx == aRegIdx) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
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

            // Update register state and store
            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }

            regState.unlock(bRegIdx);
            regState.unlock(aRegIdx);
            break;
        }

        // Output is a no-op - the result is already in values[nodeId]
        // The Graph marks certain nodes as outputs but they don't need special handling

        case OpCode::If: {
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

            // Store result
            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }


            regState.unlock(condRegIdx);
            regState.unlock(trueRegIdx);
            regState.unlock(falseRegIdx);
            break;
        }

        case OpCode::BoolConstant: {
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

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }
            break;
        }

        case OpCode::BoolAnd: {
            // Logical AND: can be implemented as multiplication
            // 1.0 * 1.0 = 1.0, 1.0 * 0.0 = 0.0, 0.0 * 0.0 = 0.0
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }

            if (bRegIdx < 0 || bRegIdx == aRegIdx) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
            }

            // Multiply for AND (1.0 * 1.0 = 1.0, any * 0.0 = 0.0)
            instructionSet->emitMul(a, aRegIdx, bRegIdx);

            regState.setRegister(aRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
            }
            break;
        }

        case OpCode::BoolOr: {
            // Logical OR: a + b - a*b
            // This gives: 0+0-0*0=0, 0+1-0*1=1, 1+0-1*0=1, 1+1-1*1=1
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);

            if (bRegIdx < 0) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
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

            regState.setRegister(aRegIdx, nodeId, deferStore);
            if (!deferStore) {
                instructionSet->emitOptimizedStore(a, aRegIdx, nodeId);
            }
            regState.unlock(aRegIdx);
            break;
        }

        case OpCode::BoolNot: {
            // Logical NOT: 1.0 - a
            // 1.0 - 0.0 = 1.0, 1.0 - 1.0 = 0.0
            int aRegIdx = regState.findNodeInRegister(node.a);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }

            // Allocate register for 1.0
            int oneRegIdx = regState.allocateAvoiding({aRegIdx});

            // Load 1.0 using instruction set abstraction
            instructionSet->emitLoadImmediate(a, oneRegIdx, 1.0);

            // result = 1.0 - a
            instructionSet->emitSub(a, oneRegIdx, aRegIdx);

            regState.setRegister(oneRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(oneRegIdx, nodeId);
            }
            break;
        }

        case OpCode::BoolEq:
        case OpCode::BoolNe: {
            // Boolean equality/inequality
            // For BoolEq: returns 1.0 if both are equal, 0.0 otherwise
            // For BoolNe: returns 1.0 if different, 0.0 otherwise
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);

            if (bRegIdx < 0) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
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

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }
            regState.unlock(aRegIdx);
            break;
        }

        // Integer operations
        case OpCode::IntConstant: {
            // Integer constant stored as double in node.imm
            double value = node.imm;  // Integer stored as double

            int resultRegIdx = regState.allocateAvoiding({});

            if (value == 0.0) {
                instructionSet->emitZero(a, resultRegIdx);  // Efficient zero
            } else {
                // Load the integer value (stored as double)
                instructionSet->emitLoadImmediate(a, resultRegIdx, value);
            }

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }
            break;
        }

        case OpCode::IntAdd: {
            // Integer addition: truncate inputs, add, truncate result
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);

            if (bRegIdx < 0) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
            }

            int resultRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx});

            // Truncate first operand to integer (don't modify original register)
            instructionSet->emitRound(a, resultRegIdx, aRegIdx, 3);  // Truncate mode (toward zero)

            // Truncate second operand and add
            int tempRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx, resultRegIdx});
            instructionSet->emitRound(a, tempRegIdx, bRegIdx, 3);  // Truncate mode

            // Add
            instructionSet->emitAdd(a, resultRegIdx, tempRegIdx);

            // Truncate result to ensure integer semantics
            instructionSet->emitRound(a, resultRegIdx, resultRegIdx, 3);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }
            regState.unlock(aRegIdx);
            break;
        }

        case OpCode::IntSub: {
            // Integer subtraction
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);

            if (bRegIdx < 0) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
            }

            int resultRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx});

            // Truncate operands in-place
            instructionSet->emitRound(a, aRegIdx, aRegIdx, 3);
            instructionSet->emitRound(a, bRegIdx, bRegIdx, 3);

            // Subtract
            instructionSet->emitMove(a, resultRegIdx, aRegIdx);
            instructionSet->emitSub(a, resultRegIdx, bRegIdx);

            // Truncate result
            instructionSet->emitRound(a, resultRegIdx, resultRegIdx, 3);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }
            regState.unlock(aRegIdx);
            break;
        }

        case OpCode::IntMul: {
            // Integer multiplication
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);

            if (bRegIdx < 0) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
            }

            int resultRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx});

            // Truncate operands in-place
            instructionSet->emitRound(a, aRegIdx, aRegIdx, 3);
            instructionSet->emitRound(a, bRegIdx, bRegIdx, 3);

            // Multiply
            instructionSet->emitMove(a, resultRegIdx, aRegIdx);
            instructionSet->emitMul(a, resultRegIdx, bRegIdx);

            // Truncate result
            instructionSet->emitRound(a, resultRegIdx, resultRegIdx, 3);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }
            regState.unlock(aRegIdx);
            break;
        }

        case OpCode::IntDiv: {
            // Integer division (truncating toward zero)
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);

            if (bRegIdx < 0) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
            }

            int resultRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx});

            // Truncate operands in-place
            instructionSet->emitRound(a, aRegIdx, aRegIdx, 3);
            instructionSet->emitRound(a, bRegIdx, bRegIdx, 3);

            // Divide
            instructionSet->emitMove(a, resultRegIdx, aRegIdx);
            instructionSet->emitDiv(a, resultRegIdx, bRegIdx);

            // Truncate result (integer division behavior)
            instructionSet->emitRound(a, resultRegIdx, resultRegIdx, 3);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }
            regState.unlock(aRegIdx);
            break;
        }

        case OpCode::IntMod: {
            // Integer modulo: a - b * trunc(a/b)
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);

            if (bRegIdx < 0) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
            }

            int resultRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx});
            int tempRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx, resultRegIdx});

            // Truncate operands in-place
            instructionSet->emitRound(a, aRegIdx, aRegIdx, 3);
            instructionSet->emitRound(a, bRegIdx, bRegIdx, 3);

            // Compute a / b
            instructionSet->emitMove(a, resultRegIdx, aRegIdx);
            instructionSet->emitDiv(a, resultRegIdx, bRegIdx);

            // Truncate the division result
            instructionSet->emitRound(a, resultRegIdx, resultRegIdx, 3);

            // Multiply by b: b * trunc(a/b)
            instructionSet->emitMove(a, tempRegIdx, bRegIdx);
            instructionSet->emitMul(a, tempRegIdx, resultRegIdx);

            // Subtract from a: a - b * trunc(a/b)
            instructionSet->emitMove(a, resultRegIdx, aRegIdx);
            instructionSet->emitSub(a, resultRegIdx, tempRegIdx);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }
            regState.unlock(aRegIdx);
            break;
        }

        case OpCode::IntNeg: {
            // Integer negation
            int aRegIdx = regState.findNodeInRegister(node.a);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }

            int resultRegIdx = regState.allocateAvoiding({aRegIdx});

            // Truncate operand in-place
            instructionSet->emitRound(a, aRegIdx, aRegIdx, 3);

            // Copy to result register
            instructionSet->emitMove(a, resultRegIdx, aRegIdx);

            // Create sign bit mask and negate
            int maskRegIdx = regState.allocateAvoiding({aRegIdx, resultRegIdx});
            instructionSet->emitCreateAllOnes(a, maskRegIdx);  // All 1s
            instructionSet->emitShiftLeft(a, maskRegIdx, 63);  // Sign bit only
            instructionSet->emitXorPD(a, resultRegIdx, maskRegIdx);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }
            break;
        }

        // No conversion operations - fint is purely integer-only

        // Integer comparisons (return Bool as 0.0/1.0)
        case OpCode::IntCmpLT:
        case OpCode::IntCmpLE:
        case OpCode::IntCmpGT:
        case OpCode::IntCmpGE:
        case OpCode::IntCmpEQ:
        case OpCode::IntCmpNE: {
            // Integer comparisons - truncate operands then compare
            int aRegIdx = regState.findNodeInRegister(node.a);
            int bRegIdx = regState.findNodeInRegister(node.b);

            if (aRegIdx < 0) {
                aRegIdx = ensureInReg( node.a, {});
            }
            regState.lock(aRegIdx);

            if (bRegIdx < 0) {
                bRegIdx = ensureInReg( node.b, {aRegIdx});
            }

            int resultRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx});
            int tempARegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx, resultRegIdx});
            int tempBRegIdx = regState.allocateAvoiding({aRegIdx, bRegIdx, resultRegIdx, tempARegIdx});

            // Use instruction set abstraction for integer comparison
            switch (node.op) {
                case OpCode::IntCmpLT:
                    instructionSet->emitIntCmpLT(a, resultRegIdx, aRegIdx, bRegIdx, regState);
                    break;
                case OpCode::IntCmpLE:
                    instructionSet->emitIntCmpLE(a, resultRegIdx, aRegIdx, bRegIdx, regState);
                    break;
                case OpCode::IntCmpGT:
                    instructionSet->emitIntCmpGT(a, resultRegIdx, aRegIdx, bRegIdx, regState);
                    break;
                case OpCode::IntCmpGE:
                    instructionSet->emitIntCmpGE(a, resultRegIdx, aRegIdx, bRegIdx, regState);
                    break;
                case OpCode::IntCmpEQ:
                    instructionSet->emitIntCmpEQ(a, resultRegIdx, aRegIdx, bRegIdx, regState);
                    break;
                case OpCode::IntCmpNE:
                    instructionSet->emitIntCmpNE(a, resultRegIdx, aRegIdx, bRegIdx, regState);
                    break;
                default:
                    break;
            }

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }
            regState.unlock(aRegIdx);
            break;
        }

        // Integer conditional selection
        case OpCode::IntIf: {
            // Conditional selection: condition ? int_true : int_false
            // Same as regular If, but truncate the selected result

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

            // Branch-free selection (same as regular If) using instruction set abstraction
            int resultRegIdx = regState.allocateAvoiding({condRegIdx, trueRegIdx, falseRegIdx});

            // Use instruction set abstraction for integer conditional
            instructionSet->emitIntIf(a, resultRegIdx, condRegIdx, trueRegIdx, falseRegIdx, regState);

            regState.setRegister(resultRegIdx, nodeId, deferStore);
            if (!deferStore) {
                tryOptimizedStore(resultRegIdx, nodeId);
            }

            regState.unlock(condRegIdx);
            regState.unlock(trueRegIdx);
            break;
        }

        default:
            // For any unimplemented operations, store NaN as a debug aid
            // This helps identify missing operations
            a.xorpd(x86::xmm0, x86::xmm0);
            a.divsd(x86::xmm0, x86::xmm0);  // 0/0 = NaN
            a.movsd(x86::ptr(x86::rdi, nodeId * sizeof(double)), x86::xmm0);
            break;
    }
}

void ForwardStitcher::stitchForwardPass(
    asmjit::x86::Assembler& a,
    const forge::Graph& graph,
    const std::unordered_map<forge::NodeId, ForgeEngine::ConstantInfo>& constantMap,
    const asmjit::Label& constPoolLabel,
    IRegisterAllocator& regState,
    IInstructionSet* instructionSet,
    const CompilerConfig* config
) {
    // This will be implemented when we refactor the compilation flow in ForgeEngine
    // For now, it's a placeholder that will call generateForwardOperation for each node
}

// Helper to ensure a value is in a register (register-aware load)
int ForwardStitcher::ensureInRegister(
    asmjit::x86::Assembler& a,
    forge::NodeId nodeId,
    IRegisterAllocator& regState,
    const forge::Graph& graph,
    const std::unordered_map<forge::NodeId, ForgeEngine::ConstantInfo>& constantMap,
    const asmjit::Label& constPoolLabel,
    std::unordered_set<forge::NodeId>& processedConstants,
    IInstructionSet* instructionSet,
    std::initializer_list<int> avoid
) {
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
            instructionSet->emitOptimizedStore(a, newReg, static_cast<NodeId>(oldNodeId));
        }
    }

    // Check if this is a constant node that needs special handling
    const Node& node = graph.nodes[nodeId];
    if (node.op == OpCode::Constant) {
        // Check if this constant has already been processed
        if (processedConstants.count(nodeId) > 0) {
            // Already processed and stored to memory, just load from there
            instructionSet->emitOptimizedLoad(a, newReg, nodeId);
        } else {
            // First time loading this constant - load from constant pool
            auto it = constantMap.find(nodeId);
            if (it != constantMap.end()) {
                if (it->second.value == 0.0) {
                    instructionSet->emitZero(a, newReg);
                } else {
                    instructionSet->emitLoadFromConstantPool(a, newReg, constPoolLabel, it->second.poolOffset);
                }
                // Store to memory so it's available for later use
                instructionSet->emitOptimizedStore(a, newReg, nodeId);
                // Mark as processed
                processedConstants.insert(nodeId);
            } else {
                // Constant not in pool - shouldn't happen
                throw std::runtime_error("Constant node not found in constant pool");
            }
        }
    } else {
        // Load the value from memory normally
        instructionSet->emitOptimizedLoad(a, newReg, nodeId);
    }

    regState.setRegister(newReg, nodeId, false); // Not dirty since we just loaded it

    return newReg;
}

} // namespace forge
