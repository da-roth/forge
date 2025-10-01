#pragma once

#include <asmjit/x86.h>
#include "../core/opcodes.h"
#include <memory>
#include <string>
#include <cstdint>

namespace forge {
namespace x86 {

// Forward declarations
class IRegisterAllocator;

// Abstract base class for instruction set implementations
// This allows contributors to add new instruction sets (like AVX2, AVX512, etc.)
// without modifying existing code
class IInstructionSet {
public:
    virtual ~IInstructionSet() = default;
    
    // Get instruction set name for debugging/logging
    virtual std::string getName() const = 0;
    
    // Capability queries
    virtual int getMaxRegisterCount() const = 0;
    virtual int getVectorWidth() const = 0; // Number of doubles that can be processed simultaneously
    virtual bool supportsOperation(forge::core::OpCode op) const = 0;
    
    // Core arithmetic operations
    virtual void emitAdd(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitSub(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitMul(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitDiv(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    
    // Three-operand arithmetic (dst = src1 op src2)
    virtual void emitAdd3(asmjit::x86::Assembler& a, int dstReg, int src1Reg, int src2Reg) = 0;
    virtual void emitSub3(asmjit::x86::Assembler& a, int dstReg, int src1Reg, int src2Reg) = 0;
    virtual void emitMul3(asmjit::x86::Assembler& a, int dstReg, int src1Reg, int src2Reg) = 0;
    virtual void emitDiv3(asmjit::x86::Assembler& a, int dstReg, int src1Reg, int src2Reg) = 0;
    
    // Unary operations
    virtual void emitNeg(asmjit::x86::Assembler& a, int dstReg) = 0;
    virtual void emitAbs(asmjit::x86::Assembler& a, int dstReg) = 0;
    virtual void emitSqrt(asmjit::x86::Assembler& a, int dstReg) = 0;
    virtual void emitRecip(asmjit::x86::Assembler& a, int dstReg) = 0;
    
    // Memory operations - using uint32_t instead of NodeId
    virtual void emitLoad(asmjit::x86::Assembler& a, int dstReg, uint32_t nodeId) = 0;
    virtual void emitStore(asmjit::x86::Assembler& a, int srcReg, uint32_t nodeId) = 0;
    virtual void emitLoadFromConstantPool(asmjit::x86::Assembler& a, int dstReg, 
                                          const asmjit::Label& poolLabel, size_t offset) = 0;
    
    // Register-to-register move
    virtual void emitMove(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    
    // Comparison operations (for conditional branches) - now require register state for safe temp allocation
    virtual void emitCmpLT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitCmpLE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitCmpGT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitCmpGE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitCmpEQ(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitCmpNE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    
    // Create a mask from a boolean value (0.0 or 1.0 -> all-zeros or all-ones)
    virtual void emitCreateMaskFromBool(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    
    // Min/Max operations
    virtual void emitMin(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitMax(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    
    // Special operations that might have optimized implementations
    virtual void emitSquare(asmjit::x86::Assembler& a, int dstReg) = 0;
    
    // Transcendental functions (using library calls)
    virtual void emitExp(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;
    virtual void emitLog(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;
    virtual void emitPow(asmjit::x86::Assembler& a, int dstReg, int baseReg, int expReg, IRegisterAllocator& regState) = 0;
    virtual void emitSin(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;
    virtual void emitCos(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;
    virtual void emitTan(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;
    
    // Modulo operation
    virtual void emitMod(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;
    
    // Conditional operations
    virtual void emitIf(asmjit::x86::Assembler& a, int dstReg, int condReg, int trueReg, int falseReg, IRegisterAllocator& regState) = 0;
    
    // Bitwise operations (for gradient masking)
    virtual void emitAndPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitXorPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitOrPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitAndNotPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;  // dst = ~dst & src
    
    // Bit manipulation for creating masks
    virtual void emitCreateAllOnes(asmjit::x86::Assembler& a, int dstReg) = 0;
    virtual void emitShiftLeft(asmjit::x86::Assembler& a, int dstReg, int bits) = 0;
    virtual void emitShiftRight(asmjit::x86::Assembler& a, int dstReg, int bits) = 0;
    
    // Load immediate constant
    virtual void emitLoadImmediate(asmjit::x86::Assembler& a, int dstReg, double value) = 0;
    virtual void emitLoadImmediateRaw(asmjit::x86::Assembler& a, int dstReg, uint64_t bits) = 0;
    
    // Rounding operations
    virtual void emitRound(asmjit::x86::Assembler& a, int dstReg, int srcReg, int mode) = 0;
    
    // Integer comparison operations (truncate inputs to int, then compare)
    virtual void emitIntCmpLT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitIntCmpLE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitIntCmpGT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitIntCmpGE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitIntCmpEQ(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitIntCmpNE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    
    // Integer conditional operation (truncates true/false values)
    virtual void emitIntIf(asmjit::x86::Assembler& a, int dstReg, int condReg, int trueReg, int falseReg, IRegisterAllocator& regState) = 0;
    
    // Blending/conditional move
    virtual void emitBlend(asmjit::x86::Assembler& a, int dstReg, int srcReg, int maskReg) = 0;
    
    // Zero register
    virtual void emitZero(asmjit::x86::Assembler& a, int dstReg) = 0;
    
    // Function prologue/epilogue with full context
    virtual void emitPrologue(asmjit::x86::Assembler& a) = 0;
    virtual void emitEpilogue(asmjit::x86::Assembler& a) = 0;
    
    // Register management
    virtual void emitSaveCalleeRegisters(asmjit::x86::Assembler& a) = 0;
    virtual void emitRestoreCalleeRegisters(asmjit::x86::Assembler& a) = 0;
    virtual int getStackSpaceNeeded() const = 0;
    
    // Helper to get register from index (implementation specific)
    virtual asmjit::x86::Xmm getRegister(int index) const = 0;
    
    // Register setup for function arguments
    virtual void emitMoveArgsToRegisters(asmjit::x86::Assembler& a) = 0;
    
    // Memory operations with optimized addressing modes
    virtual void emitOptimizedLoad(asmjit::x86::Assembler& a, int dstReg, uint32_t nodeId) = 0;
    virtual void emitOptimizedStore(asmjit::x86::Assembler& a, int srcReg, uint32_t nodeId) = 0;
    
    // Gradient-specific operations
    // Load gradient[nodeId] into register (RSI points to gradients array)
    virtual void emitLoadGradient(asmjit::x86::Assembler& a, int dstReg, uint32_t nodeId) = 0;
    
    // Store register into gradient[nodeId] (RSI points to gradients array)
    virtual void emitStoreGradient(asmjit::x86::Assembler& a, int srcReg, uint32_t nodeId) = 0;
    
    // Accumulate register into gradient[nodeId] (gradient[nodeId] += reg)
    virtual void emitAccumulateGradient(asmjit::x86::Assembler& a, int srcReg, uint32_t nodeId, int tempReg = 3) = 0;
    
    // Forward declaration for ComputationGraph-like data
    struct GraphView {
        struct Node {
            forge::core::OpCode op;
            uint32_t dst;
            uint32_t a;
            uint32_t b;
            uint32_t c;
            uint32_t flags;
            double imm;
            bool isActive;
            bool isDead;
            bool needsGradient;
        };
        const Node* nodes;
        const double* constPool;
        size_t nodeCount;
    };
    
    // Load value[nodeId] into register, handling constants from pool
    virtual void emitLoadValueForGradient(asmjit::x86::Assembler& a, int dstReg, uint32_t nodeId,
                                          const GraphView& graph,
                                          const void* constantMap,  // Will be cast to proper type in implementation
                                          const asmjit::Label& constPoolLabel) = 0;
};

} // namespace x86
} // namespace forge