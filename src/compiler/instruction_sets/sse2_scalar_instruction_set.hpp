#pragma once

#include "x86_instruction_set_base.hpp"
#include "../xmm_register_allocator.hpp"  // Use XMM-specific allocator
#include "../instruction_tracer.hpp"      // For runtime tracing
#include <cmath>
#include <climits>
#include <cstring>  // For memcpy
#include <unordered_map>

namespace forge {

// Simple struct to hold constant info - matches ForgeEngine::ConstantInfo
// Defined here to avoid circular dependency
struct ConstantInfo {
    size_t poolOffset;  // Offset within the constant pool
    double value;       // The constant value
};

// SSE2 Scalar instruction set implementation
// This uses SSE2 instructions but only processes ONE double at a time (scalar operations)
// The 'sd' suffix in instructions like 'addsd', 'mulsd' means 'Scalar Double'
// This is different from packed operations ('pd' suffix) which would process 2 doubles
class SSE2ScalarInstructionSet : public X86InstructionSetBase {
private:
    CompilerConfig config;
    InstructionTracer tracer;
    
public:
    // Constructor
    SSE2ScalarInstructionSet(const CompilerConfig& cfg = CompilerConfig::Default()) 
        : config(cfg), tracer(cfg) {}
    
    // Destructor - print trace records if tracing was enabled
    ~SSE2ScalarInstructionSet() {
        // Trace printing now happens after forward pass, not in destructor
    }
    
    std::string getName() const override { return "SSE2-Scalar"; }
    
    // SSE2 uses XMM0-XMM15 registers
    int getMaxRegisterCount() const override { return 16; }
    
    // SSE2 Scalar processes one double at a time (using only lower 64 bits of XMM registers)
    // Note: SSE2 Packed would process 2 doubles, AVX2 would process 4 doubles
    int getVectorWidth() const override { return 1; }
    
    bool supportsOperation(forge::OpCode op) const override {
        // SSE2 supports all current operations
        return true;
    }
    
    // Arithmetic operations
    void emitAdd(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        // Perform the operation
        a.addsd(getRegister(dstReg), getRegister(srcReg));
        
        // Trace output values after operation with register info
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::ADD, 1, -1, srcReg, dstReg);
    }
    
    void emitSub(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        // Perform the operation
        a.subsd(getRegister(dstReg), getRegister(srcReg));
        
        // Trace output values after operation with register info
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::SUB, 1, -1, srcReg, dstReg);
    }
    
    void emitMul(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        // Perform the operation
        a.mulsd(getRegister(dstReg), getRegister(srcReg));
        
        // Trace output values after operation with register info
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::MUL, 1, -1, srcReg, dstReg);
    }
    
    void emitDiv(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        // Perform the operation
        a.divsd(getRegister(dstReg), getRegister(srcReg));
        
        // Trace output values after operation with register info
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::DIV, 1, -1, srcReg, dstReg);
    }
    
    // Unary operations
    void emitNeg(asmjit::x86::Assembler& a, int dstReg, int tempReg) override {
        // Negate by subtracting from zero: result = 0 - value
        asmjit::x86::Vec tmpReg = getRegister(tempReg);
        a.xorpd(tmpReg, tmpReg); // Zero out temp register
        a.subsd(tmpReg, getRegister(dstReg));
        a.movsd(getRegister(dstReg), tmpReg);

        // Trace the negated value
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::NEG, 1, -1, dstReg, dstReg);
    }

    void emitAbs(asmjit::x86::Assembler& a, int dstReg, int tempReg) override {
        // Clear the sign bit using AND with 0x7FFFFFFFFFFFFFFF
        asmjit::x86::Vec reg = getRegister(dstReg);
        asmjit::x86::Vec tmpReg = getRegister(tempReg);

        // Create mask with all bits set except sign bit
        a.pcmpeqd(tmpReg, tmpReg); // All ones
        a.psrlq(tmpReg, 1);        // Shift right to clear sign bit
        a.andpd(reg, tmpReg);      // Apply mask

        // Trace the absolute value
        tracer.emitTraceXMM(a, reg, OperationType::ABS, 1, -1, dstReg, dstReg);
    }
    
    void emitSqrt(asmjit::x86::Assembler& a, int dstReg) override {
        // Trace input values before operation
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::SQRT, 1);

        // Perform the operation
        a.sqrtsd(getRegister(dstReg), getRegister(dstReg));

        // Trace output values after operation
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::SQRT, 1);
    }

    // Memory operations
    void emitLoad(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) override {
        // Load from workspace (RDI points to workspace)
        a.movsd(getRegister(dstReg), asmjit::x86::ptr(asmjit::x86::rdi, nodeId * sizeof(double)));
        
        // Trace the loaded value with node ID and destination register
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::LOAD, 1, nodeId, -1, dstReg);
    }
    
    void emitStore(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) override {
        // Trace the value before storing with node ID and source register
        tracer.emitTraceXMM(a, getRegister(srcReg), OperationType::STORE, 1, nodeId, srcReg, -1);
        
        // Store to workspace (RDI points to workspace)
        a.movsd(asmjit::x86::ptr(asmjit::x86::rdi, nodeId * sizeof(double)), getRegister(srcReg));
    }
    
    void emitLoadFromConstantPool(asmjit::x86::Assembler& a, int dstReg, 
                                  const asmjit::Label& poolLabel, size_t offset) override {
        // RIP-relative addressing for constant pool
        a.movsd(getRegister(dstReg), asmjit::x86::ptr(poolLabel, static_cast<int32_t>(offset)));
        
        // Trace the loaded constant value
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::LOAD_CONST, 1, -1, -1, dstReg);
    }
    
    void emitMove(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        if (dstReg != srcReg) {
            a.movsd(getRegister(dstReg), getRegister(srcReg));
            // Trace the move operation
            tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::MOVE, 1, -1, srcReg, dstReg);
        }
    }
    
    // Comparison operations
    void emitCmpLT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        a.movsd(getRegister(dstReg), getRegister(lhsReg));
        a.cmpsd(getRegister(dstReg), getRegister(rhsReg), 1); // _CMP_LT_OS
        // Trace the comparison result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::CMP_LT, 1, -1, rhsReg, dstReg);
    }
    
    void emitCmpLE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        a.movsd(getRegister(dstReg), getRegister(lhsReg));
        a.cmpsd(getRegister(dstReg), getRegister(rhsReg), 2); // _CMP_LE_OS
        // Trace the comparison result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::CMP_LE, 1, -1, rhsReg, dstReg);
    }
    
    void emitCmpGT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        a.movsd(getRegister(dstReg), getRegister(rhsReg));
        a.cmpsd(getRegister(dstReg), getRegister(lhsReg), 1); // Swap operands for GT
        // Trace the comparison result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::CMP_GT, 1, -1, lhsReg, dstReg);
    }
    
    void emitCmpGE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        a.movsd(getRegister(dstReg), getRegister(rhsReg));
        a.cmpsd(getRegister(dstReg), getRegister(lhsReg), 2); // Swap operands for GE
        // Trace the comparison result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::CMP_GE, 1, -1, lhsReg, dstReg);
    }
    
    void emitCmpEQ(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        a.movsd(getRegister(dstReg), getRegister(lhsReg));
        a.cmpsd(getRegister(dstReg), getRegister(rhsReg), 0); // _CMP_EQ_OQ
        // Trace the comparison result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::CMP_EQ, 1, -1, rhsReg, dstReg);
    }
    
    void emitCmpNE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        a.movsd(getRegister(dstReg), getRegister(lhsReg));
        a.cmpsd(getRegister(dstReg), getRegister(rhsReg), 4); // _CMP_NEQ_UQ
        // Trace the comparison result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::CMP_NE, 1, -1, rhsReg, dstReg);
    }
    
    // Min/Max operations
    void emitMin(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        a.minsd(getRegister(dstReg), getRegister(srcReg));
        // Trace the minimum value
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::MIN, 1, -1, srcReg, dstReg);
    }
    
    void emitMax(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        a.maxsd(getRegister(dstReg), getRegister(srcReg));
        // Trace the maximum value
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::MAX, 1, -1, srcReg, dstReg);
    }
    
    // Special optimized operations
    void emitSquare(asmjit::x86::Assembler& a, int dstReg) override {
        // x * x is often faster than pow(x, 2)
        asmjit::x86::Vec reg = getRegister(dstReg);
        a.mulsd(reg, reg);
    }
    
    // Transcendental functions using library calls - use base class helpers for non-XMM parts
    void emitExp(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        // Move argument to XMM0 for function call
        a.movsd(asmjit::x86::xmm0, getRegister(srcReg));
        
        // Use base class helpers for GP register handling
        beginFunctionCall(a);
        
        // Call function and invalidate volatile registers
        callFunctionAndInvalidate(a, reinterpret_cast<uint64_t>(static_cast<double(*)(double)>(std::exp)), regState);
        
        // Clean up using base class helpers
        endFunctionCall(a);
        
        // Move result from XMM0 to destination
        a.movsd(getRegister(dstReg), asmjit::x86::xmm0);
        
        // Trace the exponential operation result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::EXP, 1, -1, srcReg, dstReg);
    }
    
    void emitLog(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        a.movsd(asmjit::x86::xmm0, getRegister(srcReg));
        beginFunctionCall(a);
        callFunctionAndInvalidate(a, reinterpret_cast<uint64_t>(static_cast<double(*)(double)>(std::log)), regState);
        endFunctionCall(a);
        a.movsd(getRegister(dstReg), asmjit::x86::xmm0);
        
        // Trace the logarithm operation result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::LOG, 1, -1, srcReg, dstReg);
    }
    
    void emitSin(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        a.movsd(asmjit::x86::xmm0, getRegister(srcReg));
        beginFunctionCall(a);
        callFunctionAndInvalidate(a, reinterpret_cast<uint64_t>(static_cast<double(*)(double)>(std::sin)), regState);
        endFunctionCall(a);
        a.movsd(getRegister(dstReg), asmjit::x86::xmm0);
        
        // Trace the sine operation result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::SIN, 1, -1, srcReg, dstReg);
    }
    
    void emitCos(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        a.movsd(asmjit::x86::xmm0, getRegister(srcReg));
        beginFunctionCall(a);
        callFunctionAndInvalidate(a, reinterpret_cast<uint64_t>(static_cast<double(*)(double)>(std::cos)), regState);
        endFunctionCall(a);
        a.movsd(getRegister(dstReg), asmjit::x86::xmm0);
        
        // Trace the cosine operation result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::COS, 1, -1, srcReg, dstReg);
    }
    
    void emitTan(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        a.movsd(asmjit::x86::xmm0, getRegister(srcReg));
        beginFunctionCall(a);
        callFunctionAndInvalidate(a, reinterpret_cast<uint64_t>(static_cast<double(*)(double)>(std::tan)), regState);
        endFunctionCall(a);
        a.movsd(getRegister(dstReg), asmjit::x86::xmm0);
        
        // Trace the tangent operation result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::TAN, 1, -1, srcReg, dstReg);
    }
    
    // Simplified pow implementation - use base class pattern too
    void emitPow(asmjit::x86::Assembler& a, int dstReg, int baseReg, int expReg, IRegisterAllocator& regState) override {
        // Handle register conflicts when moving to XMM0 and XMM1 (same logic as before)
        if (expReg == 0) {
            if (baseReg == 1) {
                a.movsd(asmjit::x86::xmm2, asmjit::x86::xmm0);  // Save exp to XMM2
                a.movsd(asmjit::x86::xmm0, asmjit::x86::xmm1);  // Move base to XMM0
                a.movsd(asmjit::x86::xmm1, asmjit::x86::xmm2);  // Move exp to XMM1
            } else {
                a.movsd(asmjit::x86::xmm1, asmjit::x86::xmm0);  // Move exp to XMM1
                a.movsd(asmjit::x86::xmm0, getRegister(baseReg));  // Move base to XMM0
            }
        } else if (baseReg == 1) {
            a.movsd(asmjit::x86::xmm0, asmjit::x86::xmm1);  // Move base to XMM0
            a.movsd(asmjit::x86::xmm1, getRegister(expReg));  // Move exp to XMM1
        } else {
            a.movsd(asmjit::x86::xmm0, getRegister(baseReg));  // Move base to XMM0
            a.movsd(asmjit::x86::xmm1, getRegister(expReg));   // Move exp to XMM1
        }
        
        beginFunctionCall(a);
        callFunctionAndInvalidate(a, reinterpret_cast<uint64_t>(static_cast<double(*)(double, double)>(std::pow)), regState);
        endFunctionCall(a);
        a.movsd(getRegister(dstReg), asmjit::x86::xmm0);
        
        // Trace the power operation result
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::POW, 1, -1, baseReg, expReg);
    }
    
    // Modulo operation: a - b * trunc(a/b)
    void emitMod(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        // Need a temporary register
        int tmpReg = 15; // Use XMM15 as temp
        
        // Compute: a - b * trunc(a/b)
        a.movsd(getRegister(tmpReg), getRegister(dstReg));  // Copy a
        a.divsd(getRegister(tmpReg), getRegister(srcReg));  // a/b
        a.roundsd(getRegister(tmpReg), getRegister(tmpReg), 0x0B);  // trunc(a/b)
        a.mulsd(getRegister(tmpReg), getRegister(srcReg));  // b * trunc(a/b)
        a.subsd(getRegister(dstReg), getRegister(tmpReg));  // a - b * trunc(a/b)
    }
    
    // Conditional: result = cond ? trueVal : falseVal
    void emitIf(asmjit::x86::Assembler& a, int dstReg, int condReg, int trueReg, int falseReg, IRegisterAllocator& regState) override {
        // SSE2-compatible conditional selection using bitwise operations
        // This assumes condReg contains either 0.0 (false) or 1.0 (true)
        
        // First, we need to convert the condition to a proper mask
        // Compare condition with zero to create all 1s or all 0s mask
        int zeroRegIdx = regState.allocateAvoiding({condReg, trueReg, falseReg, dstReg});
        emitZero(a, zeroRegIdx);
        
        // Create mask: if cond != 0, mask = all 1s, else mask = all 0s
        int maskRegIdx = regState.allocateAvoiding({condReg, trueReg, falseReg, dstReg, zeroRegIdx});
        a.movsd(getRegister(maskRegIdx), getRegister(condReg));
        a.cmpsd(getRegister(maskRegIdx), getRegister(zeroRegIdx), 4); // NEQ_UQ
        
        // Now use the mask to blend trueReg and falseReg
        // temp = trueReg & mask
        int tempRegIdx = regState.allocateAvoiding({condReg, trueReg, falseReg, dstReg, zeroRegIdx, maskRegIdx});
        a.movsd(getRegister(tempRegIdx), getRegister(trueReg));
        a.andpd(getRegister(tempRegIdx), getRegister(maskRegIdx));
        
        // result = falseReg & ~mask
        a.movsd(getRegister(dstReg), getRegister(maskRegIdx));
        a.andnpd(getRegister(dstReg), getRegister(falseReg));
        
        // result = (trueReg & mask) | (falseReg & ~mask)
        a.orpd(getRegister(dstReg), getRegister(tempRegIdx));
    }
    
    // Bitwise AND for double precision (for gradient masking)
    void emitAndPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        a.andpd(getRegister(dstReg), getRegister(srcReg));
    }
    
    // Bitwise XOR for double precision
    void emitXorPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        a.xorpd(getRegister(dstReg), getRegister(srcReg));
    }
    
    // Bitwise OR for double precision
    void emitOrPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        a.orpd(getRegister(dstReg), getRegister(srcReg));
    }
    
    // Bitwise AND NOT for double precision (dst = ~dst & src)
    void emitAndNotPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        a.andnpd(getRegister(dstReg), getRegister(srcReg));
    }
    
    // Create all-ones mask
    void emitCreateAllOnes(asmjit::x86::Assembler& a, int dstReg) override {
        asmjit::x86::Vec reg = getRegister(dstReg);
        a.pcmpeqw(reg, reg);  // All 1s
    }
    
    // Shift left (for creating masks)
    void emitShiftLeft(asmjit::x86::Assembler& a, int dstReg, int bits) override {
        a.psllq(getRegister(dstReg), bits);
    }
    
    // Shift right (for creating masks)  
    void emitShiftRight(asmjit::x86::Assembler& a, int dstReg, int bits) override {
        a.psrlq(getRegister(dstReg), bits);
    }
    
    // Load immediate constant value into register
    void emitLoadImmediate(asmjit::x86::Assembler& a, int dstReg, double value) override {
        // Convert double to its bit representation
        uint64_t bits;
        memcpy(&bits, &value, sizeof(double));
        
        // Load via RAX register
        a.mov(asmjit::x86::rax, bits);
        a.movq(getRegister(dstReg), asmjit::x86::rax);
    }
    
    // Load raw bit pattern into register
    void emitLoadImmediateRaw(asmjit::x86::Assembler& a, int dstReg, uint64_t bits) override {
        a.mov(asmjit::x86::rax, bits);
        a.movq(getRegister(dstReg), asmjit::x86::rax);
    }
    
    // Rounding operation
    void emitRound(asmjit::x86::Assembler& a, int dstReg, int srcReg, int mode) override {
        a.roundsd(getRegister(dstReg), getRegister(srcReg), mode);
    }
    
    // Integer comparison operations
    void emitIntCmpLT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        // Allocate temp registers for truncated values
        int tempLhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg});
        int tempRhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx});
        
        // Truncate both operands to integers
        a.roundsd(getRegister(tempLhsIdx), getRegister(lhsReg), 3);
        a.roundsd(getRegister(tempRhsIdx), getRegister(rhsReg), 3);
        
        // Perform comparison
        a.movsd(getRegister(dstReg), getRegister(tempLhsIdx));
        a.cmpsd(getRegister(dstReg), getRegister(tempRhsIdx), 1); // LT
        
        // Convert result to 0.0/1.0
        int oneRegIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx, tempRhsIdx});
        a.mov(asmjit::x86::rax, 0x3FF0000000000000ULL);  // 1.0
        a.movq(getRegister(oneRegIdx), asmjit::x86::rax);
        a.andpd(getRegister(dstReg), getRegister(oneRegIdx));
    }
    
    void emitIntCmpLE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        int tempLhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg});
        int tempRhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx});
        
        a.roundsd(getRegister(tempLhsIdx), getRegister(lhsReg), 3);
        a.roundsd(getRegister(tempRhsIdx), getRegister(rhsReg), 3);
        a.movsd(getRegister(dstReg), getRegister(tempLhsIdx));
        a.cmpsd(getRegister(dstReg), getRegister(tempRhsIdx), 2); // LE
        
        int oneRegIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx, tempRhsIdx});
        a.mov(asmjit::x86::rax, 0x3FF0000000000000ULL);
        a.movq(getRegister(oneRegIdx), asmjit::x86::rax);
        a.andpd(getRegister(dstReg), getRegister(oneRegIdx));
    }
    
    void emitIntCmpGT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        int tempLhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg});
        int tempRhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx});
        
        a.roundsd(getRegister(tempLhsIdx), getRegister(lhsReg), 3);
        a.roundsd(getRegister(tempRhsIdx), getRegister(rhsReg), 3);
        a.movsd(getRegister(dstReg), getRegister(tempLhsIdx));
        a.cmpsd(getRegister(dstReg), getRegister(tempRhsIdx), 6); // NLE (not LE = GT)
        
        int oneRegIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx, tempRhsIdx});
        a.mov(asmjit::x86::rax, 0x3FF0000000000000ULL);
        a.movq(getRegister(oneRegIdx), asmjit::x86::rax);
        a.andpd(getRegister(dstReg), getRegister(oneRegIdx));
    }
    
    void emitIntCmpGE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        int tempLhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg});
        int tempRhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx});
        
        a.roundsd(getRegister(tempLhsIdx), getRegister(lhsReg), 3);
        a.roundsd(getRegister(tempRhsIdx), getRegister(rhsReg), 3);
        a.movsd(getRegister(dstReg), getRegister(tempLhsIdx));
        a.cmpsd(getRegister(dstReg), getRegister(tempRhsIdx), 5); // NLT (not LT = GE)
        
        int oneRegIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx, tempRhsIdx});
        a.mov(asmjit::x86::rax, 0x3FF0000000000000ULL);
        a.movq(getRegister(oneRegIdx), asmjit::x86::rax);
        a.andpd(getRegister(dstReg), getRegister(oneRegIdx));
    }
    
    void emitIntCmpEQ(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        int tempLhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg});
        int tempRhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx});
        
        a.roundsd(getRegister(tempLhsIdx), getRegister(lhsReg), 3);
        a.roundsd(getRegister(tempRhsIdx), getRegister(rhsReg), 3);
        a.movsd(getRegister(dstReg), getRegister(tempLhsIdx));
        a.cmpsd(getRegister(dstReg), getRegister(tempRhsIdx), 0); // EQ
        
        int oneRegIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx, tempRhsIdx});
        a.mov(asmjit::x86::rax, 0x3FF0000000000000ULL);
        a.movq(getRegister(oneRegIdx), asmjit::x86::rax);
        a.andpd(getRegister(dstReg), getRegister(oneRegIdx));
    }
    
    void emitIntCmpNE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override {
        int tempLhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg});
        int tempRhsIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx});
        
        a.roundsd(getRegister(tempLhsIdx), getRegister(lhsReg), 3);
        a.roundsd(getRegister(tempRhsIdx), getRegister(rhsReg), 3);
        a.movsd(getRegister(dstReg), getRegister(tempLhsIdx));
        a.cmpsd(getRegister(dstReg), getRegister(tempRhsIdx), 4); // NEQ
        
        int oneRegIdx = regState.allocateAvoiding({lhsReg, rhsReg, dstReg, tempLhsIdx, tempRhsIdx});
        a.mov(asmjit::x86::rax, 0x3FF0000000000000ULL);
        a.movq(getRegister(oneRegIdx), asmjit::x86::rax);
        a.andpd(getRegister(dstReg), getRegister(oneRegIdx));
    }
    
    // Integer conditional operation
    void emitIntIf(asmjit::x86::Assembler& a, int dstReg, int condReg, int trueReg, int falseReg, IRegisterAllocator& regState) override {
        // Allocate temp registers for truncated values
        int tempTrueIdx = regState.allocateAvoiding({condReg, trueReg, falseReg, dstReg});
        int tempFalseIdx = regState.allocateAvoiding({condReg, trueReg, falseReg, dstReg, tempTrueIdx});
        int maskIdx = regState.allocateAvoiding({condReg, trueReg, falseReg, dstReg, tempTrueIdx, tempFalseIdx});
        
        // Truncate integer operands
        a.roundsd(getRegister(tempTrueIdx), getRegister(trueReg), 3);
        a.roundsd(getRegister(tempFalseIdx), getRegister(falseReg), 3);
        
        // Create mask from condition
        a.xorpd(getRegister(maskIdx), getRegister(maskIdx));        // maskReg = 0.0
        a.movsd(getRegister(dstReg), getRegister(condReg));         // Copy condition to result
        a.cmpsd(getRegister(dstReg), getRegister(maskIdx), 4);      // NEQ: create mask from condition
        
        // Blend: result = (mask & true_val) | (~mask & false_val)
        a.movsd(getRegister(maskIdx), getRegister(tempTrueIdx));
        a.andpd(getRegister(maskIdx), getRegister(dstReg));         // maskReg = true_val & mask
        a.andnpd(getRegister(dstReg), getRegister(tempFalseIdx));   // dstReg = false_val & ~mask
        a.orpd(getRegister(dstReg), getRegister(maskIdx));          // dstReg = final result
        
        // Ensure result is truncated to integer
        a.roundsd(getRegister(dstReg), getRegister(dstReg), 3);
    }
    
    // Zero out a register
    void emitZero(asmjit::x86::Assembler& a, int dstReg) override {
        asmjit::x86::Vec reg = getRegister(dstReg);
        a.xorpd(reg, reg);
        // Trace the zeroed register
        tracer.emitTraceXMM(a, reg, OperationType::ZERO, 1, -1, -1, dstReg);
    }
    
    // Memory operations with optimized addressing
    void emitOptimizedLoad(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) override {
        int64_t offset = static_cast<int64_t>(nodeId) * sizeof(double);
        
        // Try different addressing modes
        if (offset >= -128 && offset <= 127) {
            a.movsd(getRegister(dstReg), asmjit::x86::ptr(asmjit::x86::rdi, static_cast<int32_t>(offset)));
        } else if (offset >= INT32_MIN && offset <= INT32_MAX) {
            a.movsd(getRegister(dstReg), asmjit::x86::ptr(asmjit::x86::rdi, static_cast<int32_t>(offset)));
        } else {
            // Large offset - need indirect addressing
            a.mov(asmjit::x86::rax, offset);
            a.add(asmjit::x86::rax, asmjit::x86::rdi);
            a.movsd(getRegister(dstReg), asmjit::x86::ptr(asmjit::x86::rax));
        }
        
        // Trace the loaded value with node ID and destination register
        tracer.emitTraceXMM(a, getRegister(dstReg), OperationType::LOAD, 1, nodeId, -1, dstReg);
    }
    
    void emitOptimizedStore(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) override {
        // Trace the value before storing with node ID and source register
        tracer.emitTraceXMM(a, getRegister(srcReg), OperationType::STORE, 1, nodeId, srcReg, -1);
        
        int64_t offset = static_cast<int64_t>(nodeId) * sizeof(double);
        
        if (offset >= -128 && offset <= 127) {
            a.movsd(asmjit::x86::ptr(asmjit::x86::rdi, static_cast<int32_t>(offset)), getRegister(srcReg));
        } else if (offset >= INT32_MIN && offset <= INT32_MAX) {
            a.movsd(asmjit::x86::ptr(asmjit::x86::rdi, static_cast<int32_t>(offset)), getRegister(srcReg));
        } else {
            a.mov(asmjit::x86::rax, offset);
            a.add(asmjit::x86::rax, asmjit::x86::rdi);
            a.movsd(asmjit::x86::ptr(asmjit::x86::rax), getRegister(srcReg));
        }
    }
    
    // Gradient-specific operations
    void emitLoadGradient(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) override {
        // RSI points to gradients array
        size_t offset = nodeId * sizeof(double);
        if (offset < 128) {
            a.movsd(getRegister(dstReg), asmjit::x86::ptr(asmjit::x86::rsi, static_cast<int32_t>(offset)));
        } else if (offset <= INT32_MAX) {
            a.movsd(getRegister(dstReg), asmjit::x86::ptr(asmjit::x86::rsi, static_cast<int32_t>(offset)));
        } else {
            a.mov(asmjit::x86::rax, offset);
            a.movsd(getRegister(dstReg), asmjit::x86::ptr(asmjit::x86::rsi, asmjit::x86::rax));
        }
    }
    
    void emitStoreGradient(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) override {
        // RSI points to gradients array
        size_t offset = nodeId * sizeof(double);
        if (offset < 128) {
            a.movsd(asmjit::x86::ptr(asmjit::x86::rsi, static_cast<int32_t>(offset)), getRegister(srcReg));
        } else if (offset <= INT32_MAX) {
            a.movsd(asmjit::x86::ptr(asmjit::x86::rsi, static_cast<int32_t>(offset)), getRegister(srcReg));
        } else {
            a.mov(asmjit::x86::rax, offset);
            a.movsd(asmjit::x86::ptr(asmjit::x86::rsi, asmjit::x86::rax), getRegister(srcReg));
        }
    }
    
    void emitAccumulateGradient(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId, int tempReg) override {
        // RSI points to gradients array
        size_t offset = nodeId * sizeof(double);
        asmjit::x86::Vec temp = getRegister(tempReg);
        
        if (offset < 128) {
            a.movsd(temp, asmjit::x86::ptr(asmjit::x86::rsi, static_cast<int32_t>(offset)));
            a.addsd(temp, getRegister(srcReg));
            a.movsd(asmjit::x86::ptr(asmjit::x86::rsi, static_cast<int32_t>(offset)), temp);
        } else if (offset <= INT32_MAX) {
            a.movsd(temp, asmjit::x86::ptr(asmjit::x86::rsi, static_cast<int32_t>(offset)));
            a.addsd(temp, getRegister(srcReg));
            a.movsd(asmjit::x86::ptr(asmjit::x86::rsi, static_cast<int32_t>(offset)), temp);
        } else {
            a.mov(asmjit::x86::rax, offset);
            a.movsd(temp, asmjit::x86::ptr(asmjit::x86::rsi, asmjit::x86::rax));
            a.addsd(temp, getRegister(srcReg));
            a.movsd(asmjit::x86::ptr(asmjit::x86::rsi, asmjit::x86::rax), temp);
        }
    }
    
    void emitLoadValueForGradient(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId,
                                  const forge::Graph& graph,
                                  const void* constantMapVoid,
                                  const asmjit::Label& constPoolLabel) override {
        // Cast to our local ConstantInfo type (which matches ForgeEngine::ConstantInfo)
        const auto& constantMap = *static_cast<const std::unordered_map<forge::NodeId, ConstantInfo>*>(constantMapVoid);
        
        // Check if this node is a constant
        if (nodeId < graph.nodes.size() && graph.nodes[nodeId].op == forge::OpCode::Constant) {
            // Load from constant pool
            auto it = constantMap.find(nodeId);
            if (it != constantMap.end()) {
                // Load from constant pool using RIP-relative addressing
                a.movsd(getRegister(dstReg), asmjit::x86::ptr(constPoolLabel, static_cast<int32_t>(it->second.poolOffset)));
            } else {
                // Fall back to loading from values array
                emitOptimizedLoad(a, dstReg, nodeId);
            }
        } else {
            // Load from values array (RDI points to values array)
            emitOptimizedLoad(a, dstReg, nodeId);
        }
    }
    
    // Implementation of virtual methods required by base class (XMM-specific)
    
    // Save XMM6-XMM15 (callee-saved on Win64)
    void emitSaveVectorRegisters(asmjit::x86::Assembler& a) const override {
#ifdef _WIN32
        // Win64: XMM registers start at offset 32 (before GP registers at 192+)
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 32), asmjit::x86::xmm6);   // offset 32
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 48), asmjit::x86::xmm7);   // offset 48
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 64), asmjit::x86::xmm8);   // offset 64
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 80), asmjit::x86::xmm9);   // offset 80
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 96), asmjit::x86::xmm10);  // offset 96
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 112), asmjit::x86::xmm11); // offset 112
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 128), asmjit::x86::xmm12); // offset 128
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 144), asmjit::x86::xmm13); // offset 144
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 160), asmjit::x86::xmm14); // offset 160
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 176), asmjit::x86::xmm15); // offset 176
#else
        // Linux System V: XMM registers start at offset 40 (after GP registers at 0-39)
        // Note: XMM registers are caller-saved on Linux, but we save them for consistency
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 40), asmjit::x86::xmm6);   // offset 40
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 56), asmjit::x86::xmm7);   // offset 56
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 72), asmjit::x86::xmm8);   // offset 72
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 88), asmjit::x86::xmm9);   // offset 88
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 104), asmjit::x86::xmm10);  // offset 104
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 120), asmjit::x86::xmm11); // offset 120
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 136), asmjit::x86::xmm12); // offset 136
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 152), asmjit::x86::xmm13); // offset 152
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 168), asmjit::x86::xmm14); // offset 168
        a.movups(asmjit::x86::ptr(asmjit::x86::rsp, 184), asmjit::x86::xmm15); // offset 184
#endif
    }
    
    // Restore XMM6-XMM15
    void emitRestoreVectorRegisters(asmjit::x86::Assembler& a) const override {
#ifdef _WIN32
        // Win64: XMM registers start at offset 32
        a.movups(asmjit::x86::xmm6, asmjit::x86::ptr(asmjit::x86::rsp, 32));
        a.movups(asmjit::x86::xmm7, asmjit::x86::ptr(asmjit::x86::rsp, 48));
        a.movups(asmjit::x86::xmm8, asmjit::x86::ptr(asmjit::x86::rsp, 64));
        a.movups(asmjit::x86::xmm9, asmjit::x86::ptr(asmjit::x86::rsp, 80));
        a.movups(asmjit::x86::xmm10, asmjit::x86::ptr(asmjit::x86::rsp, 96));
        a.movups(asmjit::x86::xmm11, asmjit::x86::ptr(asmjit::x86::rsp, 112));
        a.movups(asmjit::x86::xmm12, asmjit::x86::ptr(asmjit::x86::rsp, 128));
        a.movups(asmjit::x86::xmm13, asmjit::x86::ptr(asmjit::x86::rsp, 144));
        a.movups(asmjit::x86::xmm14, asmjit::x86::ptr(asmjit::x86::rsp, 160));
        a.movups(asmjit::x86::xmm15, asmjit::x86::ptr(asmjit::x86::rsp, 176));
#else
        // Linux System V: XMM registers start at offset 40
        a.movups(asmjit::x86::xmm6, asmjit::x86::ptr(asmjit::x86::rsp, 40));
        a.movups(asmjit::x86::xmm7, asmjit::x86::ptr(asmjit::x86::rsp, 56));
        a.movups(asmjit::x86::xmm8, asmjit::x86::ptr(asmjit::x86::rsp, 72));
        a.movups(asmjit::x86::xmm9, asmjit::x86::ptr(asmjit::x86::rsp, 88));
        a.movups(asmjit::x86::xmm10, asmjit::x86::ptr(asmjit::x86::rsp, 104));
        a.movups(asmjit::x86::xmm11, asmjit::x86::ptr(asmjit::x86::rsp, 120));
        a.movups(asmjit::x86::xmm12, asmjit::x86::ptr(asmjit::x86::rsp, 136));
        a.movups(asmjit::x86::xmm13, asmjit::x86::ptr(asmjit::x86::rsp, 152));
        a.movups(asmjit::x86::xmm14, asmjit::x86::ptr(asmjit::x86::rsp, 168));
        a.movups(asmjit::x86::xmm15, asmjit::x86::ptr(asmjit::x86::rsp, 184));
#endif
    }
    
    // Get stack space for vector registers (10 XMM registers * 16 bytes = 160 bytes)
    int getVectorStackSpace() const override {
        return 160;
    }
    
    // Get XMM register from index
    asmjit::x86::Vec getRegister(int index) const override {
        if (index >= 0 && index < 16) {
            return asmjit::x86::xmm(index);
        }
        return asmjit::x86::xmm0; // Fallback
    }
};

} // namespace forge