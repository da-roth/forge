#pragma once

#include <asmjit/asmjit.h>
#include <cstring>  // for memcpy

// Helper functions for implementing transcendental functions in AVX2
// These are building blocks used by exp, log, sin, cos, etc.

namespace forge {
namespace x86 {
namespace avx2_helpers {

// Constants for exp function (from SLEEF)
constexpr double R_LN2 = 1.442695040888963407359924681001892137426645954152985934135449406931;
constexpr double L2U = 0.69314718055966295651160180568695068359375;
constexpr double L2L = 0.28235290563031577122588448175013436025525412068e-12;
constexpr double LOG_DBL_MAX = 709.782712893384; // exp(x) overflows above this

// Round to nearest integer (vroundpd)
inline void emitRound(asmjit::x86::Assembler& a, asmjit::x86::Ymm dst, asmjit::x86::Ymm src) {
    // Mode 0 = round to nearest (even)
    a.vroundpd(dst, src, 0);
}

// Multiply-add: dst = a * b + c
inline void emitFMA(asmjit::x86::Assembler& a, 
                    asmjit::x86::Ymm dst, 
                    asmjit::x86::Ymm a_reg, 
                    asmjit::x86::Ymm b_reg, 
                    asmjit::x86::Ymm c_reg) {
    // Check if FMA is available would go here
    // For now, fallback to separate multiply and add
    a.vmulpd(dst, a_reg, b_reg);
    a.vaddpd(dst, dst, c_reg);
}

// Convert packed doubles to packed 32-bit integers
inline void emitConvertDoubleToInt(asmjit::x86::Assembler& a, asmjit::x86::Xmm dst, asmjit::x86::Ymm src) {
    // vcvtpd2dq converts 4 doubles in YMM to 4 32-bit ints in lower XMM
    a.vcvtpd2dq(dst, src);
}

// Convert packed 32-bit integers to packed doubles  
inline void emitConvertIntToDouble(asmjit::x86::Assembler& a, asmjit::x86::Ymm dst, asmjit::x86::Xmm src) {
    // vcvtdq2pd converts 4 32-bit ints to 4 doubles
    a.vcvtdq2pd(dst, src);
}

// Compare greater than: returns mask
inline void emitCmpGT(asmjit::x86::Assembler& a, asmjit::x86::Ymm dst, asmjit::x86::Ymm a_reg, asmjit::x86::Ymm b_reg) {
    // Compare mode 14 = GT (greater than, ordered, signaling)
    a.vcmppd(dst, a_reg, b_reg, 14);
}

// Blend based on mask: dst = mask ? true_val : false_val
inline void emitBlend(asmjit::x86::Assembler& a, 
                      asmjit::x86::Ymm dst, 
                      asmjit::x86::Ymm true_val, 
                      asmjit::x86::Ymm false_val, 
                      asmjit::x86::Ymm mask) {
    // vblendvpd uses high bit of each element in mask
    // First move false_val to dst if needed
    if (dst != false_val) {
        a.vmovapd(dst, false_val);
    }
    a.vblendvpd(dst, dst, true_val, mask);
}

// Load immediate constant into all lanes using stack
inline void emitBroadcastConstant(asmjit::x86::Assembler& a, asmjit::x86::Ymm dst, double value) {
    // Reduced logging verbosity
    
    // Allocate 8 bytes on stack for the constant
    a.sub(asmjit::x86::rsp, 8);
    
    // Move the constant to RAX and then to stack
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    a.mov(asmjit::x86::rax, bits);
    a.mov(asmjit::x86::qword_ptr(asmjit::x86::rsp), asmjit::x86::rax);
    
    // Broadcast from stack to all lanes of YMM register
    a.vbroadcastsd(dst, asmjit::x86::qword_ptr(asmjit::x86::rsp));
    
    // Clean up stack
    a.add(asmjit::x86::rsp, 8);
}

// Scale by power of 2: dst = src * 2^exp
// This implements ldexp functionality
inline void emitScaleByPowerOf2(asmjit::x86::Assembler& a,
                                asmjit::x86::Ymm dst,
                                asmjit::x86::Ymm src, 
                                asmjit::x86::Xmm exp_as_int,
                                asmjit::x86::Ymm tmp1,
                                asmjit::x86::Ymm tmp2) {
    
    // The trick is to construct 2^exp as a double by manipulating bits
    // Double format: [sign(1)][exponent(11)][mantissa(52)]
    // 2^n = has exponent field = 1023 + n, mantissa = 0
    
    // Step 1: Convert 32-bit integers to 64-bit
    // exp_as_int has 4x 32-bit integers, we need them as 64-bit
    a.vpmovsxdq(tmp1, exp_as_int);  // Sign extend to 64-bit
    
    // Step 2: Add 1023 (the bias for double exponent)
    // Need to create a vector of 1023 as 64-bit integers
    // First put 1023 in a general purpose register, then broadcast
    a.push(asmjit::x86::rax);
    a.mov(asmjit::x86::rax, 1023);
    a.vpbroadcastq(tmp2, asmjit::x86::rax);
    a.pop(asmjit::x86::rax);
    
    a.vpaddq(tmp1, tmp1, tmp2);  // exp + 1023
    
    // Step 3: Shift left by 52 to put in exponent field position
    a.vpsllq(tmp1, tmp1, 52);
    
    // Step 4: The bit pattern in tmp1 now represents 2^exp as doubles
    // Multiply source by this to get src * 2^exp
    a.vmulpd(dst, src, tmp1);
}

} // namespace avx2_helpers
} // namespace x86
} // namespace forge