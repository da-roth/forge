#include "avx2_instruction_set.hpp"
#include "../ymm_register_allocator.hpp"  // Use YMM-specific allocator
#include "../runtime_trace.hpp"           // For runtime tracing
#include <immintrin.h>  // For AVX2 intrinsics
#include <algorithm>  // For std::max
#include <cmath>     // For std::exp and std::log
#include <limits>    // For std::numeric_limits
#include <math.h>    // For _set_FMA3_enable
#include <sleef.h>   // For vectorized math functions

// Perfect accuracy transcendental functions - call std::lib for each scalar value
extern "C" double call_std_exp(double x) {
    // CRITICAL: Disable FMA3 for consistency with other transcendental functions
    // This ensures all math operations use the same precision model
#ifdef _MSC_VER
    static bool fma3_disabled = false;
    if (!fma3_disabled) {
        _set_FMA3_enable(0);
        fma3_disabled = true;
    }
#endif
    return std::exp(x);
}


extern "C" double call_std_log(double x) {
    // CRITICAL: Disable FMA3 to prevent crashes when calling std::log from AVX/JIT context
    // Even with VS2022, std::log appears to have issues with FMA3 instructions in our JIT environment
    // This may be due to runtime library compatibility or CPU-specific behavior with asmjit
#ifdef _MSC_VER
    static bool fma3_disabled = false;
    if (!fma3_disabled) {
        _set_FMA3_enable(0);
        fma3_disabled = true;
    }
#endif
    return std::log(x);
}

// Note: Scalar sin, cos, tan functions removed - now using SLEEF vectorized versions

// Vectorized exp: processes 4 doubles at once using SLEEF
extern "C" void call_vexp4d(const double* input, double* out) {
    __m256d vinput = _mm256_loadu_pd(input);
    __m256d result = Sleef_expd4_u10avx2(vinput);
    _mm256_storeu_pd(out, result);
}

// Vectorized log: processes 4 doubles at once using SLEEF
extern "C" void call_vlog4d(const double* input, double* out) {
    __m256d vinput = _mm256_loadu_pd(input);
    __m256d result = Sleef_logd4_u10avx2(vinput);
    _mm256_storeu_pd(out, result);
}

// Vectorized sin: processes 4 doubles at once using SLEEF
extern "C" void call_vsin4d(const double* input, double* out) {
    __m256d vinput = _mm256_loadu_pd(input);
    __m256d result = Sleef_sind4_u10avx2(vinput);
    _mm256_storeu_pd(out, result);
}

// Vectorized cos: processes 4 doubles at once using SLEEF
extern "C" void call_vcos4d(const double* input, double* out) {
    __m256d vinput = _mm256_loadu_pd(input);
    __m256d result = Sleef_cosd4_u10avx2(vinput);
    _mm256_storeu_pd(out, result);
}

// Vectorized tan: processes 4 doubles at once using SLEEF
extern "C" void call_vtan4d(const double* input, double* out) {
    __m256d vinput = _mm256_loadu_pd(input);
    __m256d result = Sleef_tand4_u10avx2(vinput);
    _mm256_storeu_pd(out, result);
}

// Vectorized pow: processes 4 doubles at once using SLEEF
extern "C" void call_vpow4d(const double* base, const double* exp, double* out) {
    __m256d vbase = _mm256_loadu_pd(base);
    __m256d vexp = _mm256_loadu_pd(exp);
    __m256d result = Sleef_powd4_u10avx2(vbase, vexp);
    _mm256_storeu_pd(out, result);
}
namespace forge {

// Emit negation: dst = -dst
void AVX2InstructionSet::emitNeg(asmjit::x86::Assembler& a, int dstReg) {
    // Trace input values before operation
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::NEG, 4);
    
    // XOR with sign bit to negate
    auto signMask = asmjit::x86::ymm15;
    
    // Load sign bit mask (0x8000000000000000 for all 4 lanes)
    // CRITICAL: Don't use XMM15 as it corrupts YMM15 upper lanes!
    // Instead, broadcast directly from memory
    a.mov(asmjit::x86::rax, 0x8000000000000000ULL);
    a.push(asmjit::x86::rax);  // Push value to stack
    a.vbroadcastsd(signMask, asmjit::x86::qword_ptr(asmjit::x86::rsp));
    a.add(asmjit::x86::rsp, 8);  // Clean up stack
    
    // XOR to flip sign bit
    a.vxorpd(getYmmRegister(dstReg), getYmmRegister(dstReg), signMask);
    
    // Trace output values after operation
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::NEG, 4);
}

// Emit absolute value: dst = |dst|
void AVX2InstructionSet::emitAbs(asmjit::x86::Assembler& a, int dstReg) {
    // Runtime tracing configured in tracer
    
    // Trace input values before operation
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::ABS, 4);
    
    // AND with mask to clear sign bit
    auto absMask = asmjit::x86::ymm15;
    
    // Create mask to clear sign bit (0x7FFFFFFFFFFFFFFF for all 4 lanes)
    // CRITICAL: Don't use XMM15 as it corrupts YMM15 upper lanes!
    // Instead, broadcast directly from memory
    a.mov(asmjit::x86::rax, 0x7FFFFFFFFFFFFFFFULL);
    a.push(asmjit::x86::rax);  // Push value to stack
    a.vbroadcastsd(absMask, asmjit::x86::qword_ptr(asmjit::x86::rsp));
    a.add(asmjit::x86::rsp, 8);  // Clean up stack
    
    // AND to clear sign bit
    a.vandpd(getYmmRegister(dstReg), getYmmRegister(dstReg), absMask);
    
    // Trace output values after operation
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::ABS, 4);
}

// Emit reciprocal: dst = 1.0 / dst
void AVX2InstructionSet::emitRecip(asmjit::x86::Assembler& a, int dstReg) {
    // Load 1.0 and divide
    auto oneReg = asmjit::x86::ymm15;
    
    // Load 1.0 into all 4 lanes
    // CRITICAL: Don't use XMM15 as it corrupts YMM15 upper lanes!
    // Instead, broadcast directly from memory
    a.mov(asmjit::x86::rax, 0x3FF0000000000000ULL);  // IEEE 754 double 1.0
    a.push(asmjit::x86::rax);  // Push value to stack
    a.vbroadcastsd(oneReg, asmjit::x86::qword_ptr(asmjit::x86::rsp));
    a.add(asmjit::x86::rsp, 8);  // Clean up stack
    
    // Divide: 1.0 / dst
    a.vdivpd(getYmmRegister(dstReg), oneReg, getYmmRegister(dstReg));
    
    // Trace operation result
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::RECIP, 4, -1, dstReg, dstReg);
}


// Memory operations
void AVX2InstructionSet::emitLoad(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) {
    
    // AVX2: Load 4 doubles from workspace (RDI points to workspace values)
    // Memory layout: 4 doubles per node for SIMD vectorization
    size_t offset = nodeId * 4 * sizeof(double);  // 4 doubles per node
    
    // Load 256 bits (4 doubles) into YMM register
    if ((offset & 31) == 0) {
        // 32-byte aligned - use aligned load
        a.vmovapd(getYmmRegister(dstReg), asmjit::x86::ymmword_ptr(asmjit::x86::rdi, offset));
    } else {
        // Not aligned - use unaligned load
        a.vmovupd(getYmmRegister(dstReg), asmjit::x86::ymmword_ptr(asmjit::x86::rdi, offset));
    }
    
    // Trace the loaded values with node ID and destination register
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::LOAD, 4, nodeId, -1, dstReg);
    
    // DEBUG: If we just loaded into YMM14, trace some other registers to see corruption state
    if (dstReg == 14) {
        emitTraceSafeYMM(a, 0, "AFTER_LOAD_TO_YMM14_CHECK_YMM0");
        emitTraceSafeYMM(a, 1, "AFTER_LOAD_TO_YMM14_CHECK_YMM1");
        emitTraceSafeYMM(a, 13, "AFTER_LOAD_TO_YMM14_CHECK_YMM13");
    }
}

void AVX2InstructionSet::emitStore(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) {
    
    // DEBUG: If we're about to store from YMM14, trace state before store
    if (srcReg == 14) {
        emitTraceSafeYMM(a, 0, "BEFORE_STORE_FROM_YMM14_CHECK_YMM0");
        emitTraceSafeYMM(a, 1, "BEFORE_STORE_FROM_YMM14_CHECK_YMM1");
        emitTraceSafeYMM(a, 13, "BEFORE_STORE_FROM_YMM14_CHECK_YMM13");
    }
    
    // Trace the values before storing with node ID and source register
    tracer.emitTraceYMM(a, getYmmRegister(srcReg), OperationType::STORE, 4, nodeId, srcReg, -1);
    
    // AVX2: Store 4 doubles to workspace
    size_t offset = nodeId * 4 * sizeof(double);  // 4 doubles per node
    
    // Store 256 bits (4 doubles) from YMM register
    if ((offset & 31) == 0) {
        // 32-byte aligned - use aligned store
        a.vmovapd(asmjit::x86::ymmword_ptr(asmjit::x86::rdi, offset), getYmmRegister(srcReg));
    } else {
        // Not aligned - use unaligned store
        a.vmovupd(asmjit::x86::ymmword_ptr(asmjit::x86::rdi, offset), getYmmRegister(srcReg));
    }
}

void AVX2InstructionSet::emitLoadFromConstantPool(asmjit::x86::Assembler& a, int dstReg,
                                                  const asmjit::Label& poolLabel, size_t offset) {
    // Load from the actual constant pool using RIP-relative addressing
    // The constant pool is placed after the function code
    a.lea(asmjit::x86::rax, asmjit::x86::ptr(poolLabel));
    a.vbroadcastsd(getYmmRegister(dstReg), asmjit::x86::qword_ptr(asmjit::x86::rax, offset));
}

// Comparison operations
void AVX2InstructionSet::emitCmpLT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    // Perform double-precision less-than comparison: lhs < rhs
    // vcmppd with imm8=1 performs LT_OS (less than, ordered, signaling)
    // Result: all 1s (0xFFFFFFFFFFFFFFFF) for true, all 0s for false
    a.vcmppd(getYmmRegister(dstReg), getYmmRegister(lhsReg), getYmmRegister(rhsReg), 1);
    
    // CRITICAL FIX: Convert comparison masks to boolean values using SAFE register allocation
    // vcmppd produces: 0xFFFFFFFFFFFFFFFF (true) -> -nan when interpreted as double
    //                  0x0000000000000000 (false) -> 0.0 when interpreted as double
    // We need: 1.0 for true, 0.0 for false
    
    // SAFE: Allocate temporary registers from the allocator
    int oneReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg});
    int zeroReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg, oneReg});
    
    // Create 1.0 and 0.0 values using safe register allocation
    emitLoadImmediate(a, oneReg, 1.0);
    emitZero(a, zeroReg);
    
    // Use vblendvpd to select 1.0 or 0.0 based on comparison mask
    // If comparison mask bit is 1 (true), select 1.0; if 0 (false), select 0.0
    a.vblendvpd(getYmmRegister(dstReg), getYmmRegister(zeroReg), getYmmRegister(oneReg), getYmmRegister(dstReg));
    
    // SAFE: Release the temporary registers
    regState.unlock(oneReg);
    regState.unlock(zeroReg);
    
    // Trace the boolean conversion result
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::CMP_LT, 4, -1, rhsReg, dstReg);
}

void AVX2InstructionSet::emitCmpLE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    // Perform double-precision less-than-or-equal comparison: lhs <= rhs
    // vcmppd with imm8=2 performs LE_OS (less than or equal, ordered, signaling)
    // Result: all 1s (0xFFFFFFFFFFFFFFFF) for true, all 0s for false
    a.vcmppd(getYmmRegister(dstReg), getYmmRegister(lhsReg), getYmmRegister(rhsReg), 2);
    
    // CRITICAL FIX: Convert comparison masks to boolean values using SAFE register allocation
    int oneReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg});
    int zeroReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg, oneReg});
    
    emitLoadImmediate(a, oneReg, 1.0);
    emitZero(a, zeroReg);
    a.vblendvpd(getYmmRegister(dstReg), getYmmRegister(zeroReg), getYmmRegister(oneReg), getYmmRegister(dstReg));
    
    regState.unlock(oneReg);
    regState.unlock(zeroReg);
    
    // Trace the boolean conversion result
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::CMP_LE, 4, -1, rhsReg, dstReg);
}

void AVX2InstructionSet::emitCmpGT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    // FIXED: Proper floating-point greater-than comparison
    // vcmppd with immediate 14 = GT_OQ (Greater Than, Ordered, Quiet)
    a.vcmppd(getYmmRegister(dstReg), getYmmRegister(lhsReg), getYmmRegister(rhsReg), 14);
    
    // CRITICAL FIX: Convert comparison masks to boolean values using SAFE register allocation
    int oneReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg});
    int zeroReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg, oneReg});
    
    emitLoadImmediate(a, oneReg, 1.0);
    emitZero(a, zeroReg);
    a.vblendvpd(getYmmRegister(dstReg), getYmmRegister(zeroReg), getYmmRegister(oneReg), getYmmRegister(dstReg));
    
    regState.unlock(oneReg);
    regState.unlock(zeroReg);
    
    // Trace the boolean conversion result
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::CMP_GT, 4, -1, rhsReg, dstReg);
}

void AVX2InstructionSet::emitCmpGE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    // Perform double-precision greater-than-or-equal comparison: lhs >= rhs
    // vcmppd with imm8=13 performs GE_OQ (greater than or equal, ordered, quiet)
    // Result: all 1s (0xFFFFFFFFFFFFFFFF) for true, all 0s for false
    a.vcmppd(getYmmRegister(dstReg), getYmmRegister(lhsReg), getYmmRegister(rhsReg), 13);
    
    // CRITICAL FIX: Convert comparison masks to boolean values using SAFE register allocation
    int oneReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg});
    int zeroReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg, oneReg});
    
    emitLoadImmediate(a, oneReg, 1.0);
    emitZero(a, zeroReg);
    a.vblendvpd(getYmmRegister(dstReg), getYmmRegister(zeroReg), getYmmRegister(oneReg), getYmmRegister(dstReg));
    
    regState.unlock(oneReg);
    regState.unlock(zeroReg);
    
    // Trace the boolean conversion result
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::CMP_GE, 4, -1, rhsReg, dstReg);
}

void AVX2InstructionSet::emitCmpEQ(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    // Perform double-precision equal comparison: lhs == rhs
    // vcmppd with imm8=0 performs EQ_OQ (equal, ordered, quiet)
    // Result: all 1s (0xFFFFFFFFFFFFFFFF) for true, all 0s for false
    a.vcmppd(getYmmRegister(dstReg), getYmmRegister(lhsReg), getYmmRegister(rhsReg), 0);
    
    // CRITICAL FIX: Convert comparison masks to boolean values using SAFE register allocation
    int oneReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg});
    int zeroReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg, oneReg});
    
    emitLoadImmediate(a, oneReg, 1.0);
    emitZero(a, zeroReg);
    a.vblendvpd(getYmmRegister(dstReg), getYmmRegister(zeroReg), getYmmRegister(oneReg), getYmmRegister(dstReg));
    
    regState.unlock(oneReg);
    regState.unlock(zeroReg);
    
    // Trace the boolean conversion result
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::CMP_EQ, 4, -1, rhsReg, dstReg);
}

void AVX2InstructionSet::emitCmpNE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    // Perform double-precision not-equal comparison: lhs != rhs
    // vcmppd with imm8=4 performs NEQ_UQ (not equal, unordered, quiet)
    // Result: all 1s (0xFFFFFFFFFFFFFFFF) for true, all 0s for false
    a.vcmppd(getYmmRegister(dstReg), getYmmRegister(lhsReg), getYmmRegister(rhsReg), 4);
    
    // CRITICAL FIX: Convert comparison masks to boolean values using SAFE register allocation
    int oneReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg});
    int zeroReg = regState.allocateAvoiding({dstReg, lhsReg, rhsReg, oneReg});
    
    emitLoadImmediate(a, oneReg, 1.0);
    emitZero(a, zeroReg);
    a.vblendvpd(getYmmRegister(dstReg), getYmmRegister(zeroReg), getYmmRegister(oneReg), getYmmRegister(dstReg));
    
    regState.unlock(oneReg);
    regState.unlock(zeroReg);
    
    // Trace the boolean conversion result
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::CMP_NE, 4, -1, lhsReg, rhsReg);
}

// Create mask from boolean
void AVX2InstructionSet::emitCreateMaskFromBool(asmjit::x86::Assembler& a, int dstReg, int srcReg) {
    // Compare with zero to create mask
    auto ymm15 = asmjit::x86::ymm15;
    a.vxorpd(ymm15, ymm15, ymm15);  // Zero
    a.vcmppd(getYmmRegister(dstReg), getYmmRegister(srcReg), ymm15, 4);  // NEQ_OQ
    
    // Trace the mask creation
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::CREATE_MASK, 4, -1, srcReg, dstReg);
}

// Bit manipulation
void AVX2InstructionSet::emitCreateAllOnes(asmjit::x86::Assembler& a, int dstReg) {
    a.vpcmpeqq(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(dstReg));
    
    // Trace the all-ones creation
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::CREATE_ALL_ONES, 4, -1, -1, dstReg);
}

void AVX2InstructionSet::emitShiftLeft(asmjit::x86::Assembler& a, int dstReg, int bits) {
    a.vpsllq(getYmmRegister(dstReg), getYmmRegister(dstReg), bits);
    
    // Trace the left shift operation
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::SHIFT_LEFT, 4, bits, -1, dstReg);
}

void AVX2InstructionSet::emitShiftRight(asmjit::x86::Assembler& a, int dstReg, int bits) {
    a.vpsrlq(getYmmRegister(dstReg), getYmmRegister(dstReg), bits);
    
    // Trace the right shift operation
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::SHIFT_RIGHT, 4, bits, -1, dstReg);
}

// Load immediate
void AVX2InstructionSet::emitLoadImmediate(asmjit::x86::Assembler& a, int dstReg, double value) {
    // For now, use constant pool approach
    // In a real implementation, we'd need to allocate space in constant pool
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    emitLoadImmediateRaw(a, dstReg, bits);
}

void AVX2InstructionSet::emitLoadImmediateRaw(asmjit::x86::Assembler& a, int dstReg, uint64_t bits) {
    // Runtime tracing configured in tracer
    
    // Move to general purpose register then broadcast
    // CRITICAL: Don't use XMM15 as it gets corrupted by scalar IF operations
    // Instead, broadcast directly from memory
    a.mov(asmjit::x86::rax, bits);
    a.push(asmjit::x86::rax);  // Push value to stack
    a.vbroadcastsd(getYmmRegister(dstReg), asmjit::x86::qword_ptr(asmjit::x86::rsp));
    a.add(asmjit::x86::rsp, 8);  // Clean up stack
    
    // Trace the loaded immediate values
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::LOAD_CONST, 4);
    
    // DEBUG: If we just loaded a constant into YMM14, trace other registers
    if (dstReg == 14) {
        emitTraceSafeYMM(a, 0, "AFTER_LOAD_CONST_TO_YMM14_CHECK_YMM0");
        emitTraceSafeYMM(a, 1, "AFTER_LOAD_CONST_TO_YMM14_CHECK_YMM1");
        emitTraceSafeYMM(a, 13, "AFTER_LOAD_CONST_TO_YMM14_CHECK_YMM13");
    }
}

// Rounding operations
void AVX2InstructionSet::emitRound(asmjit::x86::Assembler& a, int dstReg, int srcReg, int mode) {
    a.vroundpd(getYmmRegister(dstReg), getYmmRegister(srcReg), mode);
    
    // Trace the rounding operation
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::ROUND, 4, mode, srcReg, dstReg);
}

// Transcendental functions are now implemented in the header using the base class helpers
// They extract scalar values, call standard math functions, and broadcast results

// Modulo operation
void AVX2InstructionSet::emitMod(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) {
    // Native AVX2 implementation of modulo: result = dividend - floor(dividend/divisor) * divisor
    // NOTE: This implementation is NOT fully IEEE 754 compliant - edge cases like NaN, infinity, 
    // and division by zero are not handled. For production use, consider adding edge case handling.
    
    // Assume dstReg contains dividend (a) and srcReg contains divisor (b)
    // We need a temporary register for intermediate calculations
    auto ymm_dividend = getYmmRegister(dstReg);
    auto ymm_divisor = getYmmRegister(srcReg);
    
    // FIX: Properly allocate temporary registers instead of using fixed YMM14/YMM15
    int tempReg1 = regState.allocateAvoiding({dstReg, srcReg});
    int tempReg2 = regState.allocateAvoiding({dstReg, srcReg, tempReg1});
    
    auto ymm_temp = getYmmRegister(tempReg1);
    auto ymm_quotient = getYmmRegister(tempReg2);
    
    // Step 1: Calculate a/b
    a.vdivpd(ymm_quotient, ymm_dividend, ymm_divisor);
    
    // Step 2: Floor the quotient (round towards negative infinity)
    // VROUNDPD with imm8=0x09 performs floor operation
    a.vroundpd(ymm_temp, ymm_quotient, 0x09);
    
    // Step 3: Multiply floor(a/b) * b
    a.vmulpd(ymm_temp, ymm_temp, ymm_divisor);
    
    // Step 4: Calculate a - floor(a/b) * b
    a.vsubpd(ymm_dividend, ymm_dividend, ymm_temp);
    
    // FIX: Unlock the allocated temporary registers
    regState.unlock(tempReg1);
    regState.unlock(tempReg2);
    
    // Result is now in dstReg (ymm_dividend)
    
    // Trace the modulo operation
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::MOD, 4, -1, dstReg, srcReg);
}

// Conditional operation using arithmetic blending - avoids problematic mask operations
void AVX2InstructionSet::emitIf(asmjit::x86::Assembler& a, int dstReg, int condReg, int trueReg, int falseReg, IRegisterAllocator& regState) {
    // ARITHMETIC BLENDING APPROACH: result = condition * trueValue + (1.0 - condition) * falseValue
    // This completely avoids mask operations that cause NaN corruption
    // Assumes condition is 0.0 (false) or 1.0 (true) from comparison operations
    
    // Runtime tracing of input registers BEFORE processing
    tracer.emitTraceYMM(a, getYmmRegister(condReg), OperationType::IF, 4, -10, condReg, -1);
    tracer.emitTraceYMM(a, getYmmRegister(trueReg), OperationType::IF, 4, -11, trueReg, -1);  
    tracer.emitTraceYMM(a, getYmmRegister(falseReg), OperationType::IF, 4, -12, falseReg, -1);
    
    // Allocate temporary registers safely
    int oneReg = regState.allocateAvoiding({dstReg, condReg, trueReg, falseReg});
    int invCondReg = regState.allocateAvoiding({dstReg, condReg, trueReg, falseReg, oneReg});
    int tempReg = regState.allocateAvoiding({dstReg, condReg, trueReg, falseReg, oneReg, invCondReg});
    
    // Step 1: Load 1.0 into oneReg
    emitLoadImmediate(a, oneReg, 1.0);
    
    // Step 2: Calculate (1.0 - condition) in invCondReg
    a.vsubpd(getYmmRegister(invCondReg), getYmmRegister(oneReg), getYmmRegister(condReg));
    
    // Step 3: Calculate condition * trueValue in tempReg
    a.vmulpd(getYmmRegister(tempReg), getYmmRegister(condReg), getYmmRegister(trueReg));
    
    // Step 4: Calculate (1.0 - condition) * falseValue in dstReg
    a.vmulpd(getYmmRegister(dstReg), getYmmRegister(invCondReg), getYmmRegister(falseReg));
    
    // Step 5: Add the two products: dstReg = condition * trueValue + (1.0 - condition) * falseValue
    a.vaddpd(getYmmRegister(dstReg), getYmmRegister(tempReg), getYmmRegister(dstReg));
    
    // Release temporary registers
    regState.unlock(oneReg);
    regState.unlock(invCondReg);
    regState.unlock(tempReg);
    
    // Trace the arithmetic blending result
    tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::IF, 4, -1, trueReg, falseReg);
}

// Integer comparison operations
void AVX2InstructionSet::emitIntCmpLT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    // TODO: Implement integer comparison with truncation
    emitCmpLT(a, dstReg, lhsReg, rhsReg, regState);
}

void AVX2InstructionSet::emitIntCmpLE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    emitCmpLE(a, dstReg, lhsReg, rhsReg, regState);
}

void AVX2InstructionSet::emitIntCmpGT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    emitCmpGT(a, dstReg, lhsReg, rhsReg, regState);
}

void AVX2InstructionSet::emitIntCmpGE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    emitCmpGE(a, dstReg, lhsReg, rhsReg, regState);
}

void AVX2InstructionSet::emitIntCmpEQ(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    emitCmpEQ(a, dstReg, lhsReg, rhsReg, regState);
}

void AVX2InstructionSet::emitIntCmpNE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) {
    emitCmpNE(a, dstReg, lhsReg, rhsReg, regState);
}

// Integer conditional
void AVX2InstructionSet::emitIntIf(asmjit::x86::Assembler& a, int dstReg, int condReg, int trueReg, int falseReg, IRegisterAllocator& regState) {
    emitIf(a, dstReg, condReg, trueReg, falseReg, regState);
}

// Function prologue/epilogue
void AVX2InstructionSet::emitPrologue(asmjit::x86::Assembler& a) {
    // Standard function prologue
    a.push(asmjit::x86::rbp);
    a.mov(asmjit::x86::rbp, asmjit::x86::rsp);
    
    // CRITICAL: Ensure 32-byte stack alignment for YMM operations
    // After push rbp, RSP is 8-byte misaligned from 16-byte boundary
    // We need to ensure RSP is 32-byte aligned after allocating space
    a.and_(asmjit::x86::rsp, -32);  // Align stack to 32-byte boundary
    
    // Allocate stack space (must maintain alignment)
    int stackSpace = getStackSpaceNeeded();
    // Round up to multiple of 32 to maintain alignment
    stackSpace = (stackSpace + 31) & ~31;
    a.sub(asmjit::x86::rsp, stackSpace);
    
    // Save callee-saved registers if needed
    emitSaveCalleeRegisters(a);
    
    // Save MXCSR for later restoration (avoid changing FTZ/DAZ to match scalar baseline numerics)
    a.sub(asmjit::x86::rsp, 8);
    a.stmxcsr(asmjit::x86::dword_ptr(asmjit::x86::rsp));
    
    // Move arguments to expected registers
    emitMoveArgsToRegisters(a);
}

void AVX2InstructionSet::emitEpilogue(asmjit::x86::Assembler& a) {
    // Clean up upper YMM bits before returning to potentially SSE code
    // This prevents AVX-SSE transition penalties
    a.vzeroupper();
    
    // Restore original MXCSR
    a.ldmxcsr(asmjit::x86::dword_ptr(asmjit::x86::rsp));  // Restore original MXCSR
    a.add(asmjit::x86::rsp, 8);  // Remove MXCSR space
    
    // Restore callee-saved registers
    emitRestoreCalleeRegisters(a);
    
    // Restore stack and frame pointer
    a.mov(asmjit::x86::rsp, asmjit::x86::rbp);
    a.pop(asmjit::x86::rbp);
    a.ret();
}

// Register management
void AVX2InstructionSet::emitSaveCalleeRegisters(asmjit::x86::Assembler& a) {
#ifdef _WIN32
    // On Windows x64, RDI and RSI are non-volatile (callee-saved)
    // We're using them, so we need to save them
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 32), asmjit::x86::rdi);  // After shadow space
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 40), asmjit::x86::rsi);

    // Windows x64: YMM6-YMM15 are non-volatile (callee-saved)
    // Save the upper 128 bits of YMM6-YMM15
    // The lower 128 bits (XMM6-XMM15) are already handled by the ABI
    int ymmSaveOffset = 64;  // Start after shadow space and GP register saves
    for (int i = 6; i < 16; i++) {
        // Save full YMM register (256 bits)
        a.vmovupd(asmjit::x86::ymmword_ptr(asmjit::x86::rsp, ymmSaveOffset), getYmmRegister(i));
        ymmSaveOffset += 32;
    }
#else
    // On Linux System V ABI, RDI and RSI are volatile (caller-saved)
    // No need to save them. Callee-saved registers are: RBX, RBP, R12-R15
    // We don't use these registers, so nothing to save
    (void)a; // Suppress unused parameter warning
#endif
}

void AVX2InstructionSet::emitRestoreCalleeRegisters(asmjit::x86::Assembler& a) {
#ifdef _WIN32
    // Windows x64: Restore YMM6-YMM15
    int ymmSaveOffset = 64;  // Start after shadow space and GP register saves
    for (int i = 6; i < 16; i++) {
        // Restore full YMM register (256 bits)
        a.vmovupd(getYmmRegister(i), asmjit::x86::ymmword_ptr(asmjit::x86::rsp, ymmSaveOffset));
        ymmSaveOffset += 32;
    }

    // Restore saved GP registers
    a.mov(asmjit::x86::rdi, asmjit::x86::ptr(asmjit::x86::rsp, 32));
    a.mov(asmjit::x86::rsi, asmjit::x86::ptr(asmjit::x86::rsp, 40));
#else
    // On Linux System V ABI, nothing to restore
    (void)a; // Suppress unused parameter warning
#endif
}

int AVX2InstructionSet::getStackSpaceNeeded() const {
#ifdef _WIN32
    // Windows x64 ABI requires:
    // - 32 bytes shadow space for register parameters
    // - Stack must be 32-byte aligned for YMM operations
    // - Space for saving YMM6-YMM15 if needed (10 * 32 = 320 bytes)
    // - Extra space for spills
    // Total: 32 (shadow) + 320 (YMM saves) + 32 (spills) = 384
    // Round up to 32-byte boundary = 384
    return 384;
#else
    // Linux System V ABI:
    // - No shadow space required
    // - Stack must be 32-byte aligned for YMM operations
    // - No YMM registers to save (all are caller-saved)
    // - Extra space for spills
    // Total: 32 (spills), round up to 32-byte boundary = 32
    return 32;
#endif
}

// Register setup
void AVX2InstructionSet::emitMoveArgsToRegisters(asmjit::x86::Assembler& a) {
#ifdef _WIN32
    // Win64 ABI: RCX = first arg (values), RDX = second arg (gradients), R8 = third arg (count)
    // We need: RDI = values, RSI = gradients (for our memory operations)
    a.mov(asmjit::x86::rdi, asmjit::x86::rcx);  // Move values pointer to RDI
    a.mov(asmjit::x86::rsi, asmjit::x86::rdx);  // Move gradients pointer to RSI
#else
    // Linux System V ABI: RDI = first arg (values), RSI = second arg (gradients)
    // Arguments are already in the correct registers - no move needed!
    (void)a; // Suppress unused parameter warning
#endif
}

// Optimized memory operations
void AVX2InstructionSet::emitOptimizedLoad(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) {
    // Use alignment-aware load
    emitLoad(a, dstReg, nodeId);
}

void AVX2InstructionSet::emitOptimizedStore(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) {
    // Use alignment-aware store
    emitStore(a, srcReg, nodeId);
}

// Gradient operations
void AVX2InstructionSet::emitLoadGradient(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) {
    // Load 4 gradient values from workspace (RSI points to gradients)
    size_t offset = nodeId * 4 * sizeof(double);  // 4 doubles per node
    a.vmovupd(getYmmRegister(dstReg), asmjit::x86::ymmword_ptr(asmjit::x86::rsi, offset));
}

void AVX2InstructionSet::emitStoreGradient(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) {
    // Store 4 gradient values to workspace
    size_t offset = nodeId * 4 * sizeof(double);  // 4 doubles per node
    a.vmovupd(asmjit::x86::ymmword_ptr(asmjit::x86::rsi, offset), getYmmRegister(srcReg));
}

void AVX2InstructionSet::emitAccumulateGradient(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId, int tempReg) {
    // Load existing gradient, add to it, store back (AVX2: 4 doubles)
    size_t offset = nodeId * 4 * sizeof(double);  // 4 doubles per node
    auto temp = getYmmRegister(tempReg);
    a.vmovupd(temp, asmjit::x86::ymmword_ptr(asmjit::x86::rsi, offset));
    a.vaddpd(temp, temp, getYmmRegister(srcReg));
    a.vmovupd(asmjit::x86::ymmword_ptr(asmjit::x86::rsi, offset), temp);
}

void AVX2InstructionSet::emitLoadValueForGradient(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId,
                                                  const forge::Graph& graph,
                                                  const void* constantMap,
                                                  const asmjit::Label& constPoolLabel) {
    // Load value for gradient computation
    // Check if it's a constant node
    const auto* constMap = static_cast<const std::unordered_map<forge::NodeId, AVX2ConstantInfo>*>(constantMap);
    auto it = constMap->find(nodeId);
    
    if (it != constMap->end()) {
        // Load from constant pool and broadcast
        emitLoadFromConstantPool(a, dstReg, constPoolLabel, it->second.poolOffset);
    } else {
        // Load from workspace
        emitLoad(a, dstReg, nodeId);
    }
}

// DEBUG: Helper function implementations for corruption tracking
void AVX2InstructionSet::emitTraceAllYMMRegisters_UNSAFE(asmjit::x86::Assembler& a, const char* context) {
    (void)a; // Suppress unused parameter warning
    (void)context; // Suppress unused parameter warning
    // DISABLED - this function corrupts the registers it's trying to trace
    // The tracer.emitTraceYMM uses YMM15 as temporary, corrupting it during tracing
}

void AVX2InstructionSet::emitTraceSafeYMM(asmjit::x86::Assembler& a, int regNum, const char* context) {
    if (!config.printRuntimeTrace) return;
    
    // Only trace if it's NOT YMM15 (since the tracer corrupts YMM15)
    if (regNum != 15) {
        std::cout << "[SAFE_YMM_TRACE] " << context << " YMM" << regNum << std::endl;
        tracer.emitTraceYMM(a, getYmmRegister(regNum), OperationType::UNKNOWN, 4, -100 - regNum, regNum, -1);
    } else {
        std::cout << "[SAFE_YMM_TRACE] " << context << " YMM15 - SKIPPED (would corrupt)" << std::endl;
    }
}

} // namespace forge