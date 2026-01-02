#pragma once

#include "register_allocator_base.hpp"
#include <asmjit/x86.h>

namespace forge {

/**
 * YMM register allocator for AVX2 instruction set.
 * Manages YMM0-YMM15 registers for 256-bit packed AVX2 operations.
 */
class YmmRegisterAllocator : public RegisterAllocatorBase<asmjit::x86::Vec, 16> {
public:
    YmmRegisterAllocator() : RegisterAllocatorBase() {
        // Blacklist YMM14 and YMM15 due to corruption issues
        setBlacklisted(14, true);
        setBlacklisted(15, true);
    }

    asmjit::x86::Vec getRegister(int index) const override {
        return asmjit::x86::ymm(index);
    }

    int getFirstVolatileReg() const override { return 0; }
    int getLastVolatileReg() const override {
        // CRITICAL: On Windows x64, only the lower 128 bits (XMM6-15) are callee-saved.
        // The upper 128 bits of ALL YMM registers are VOLATILE according to the ABI.
        // Since we use full 256-bit YMM registers, we must treat ALL of them as volatile
        // after any external function call (like SLEEF). Otherwise, the upper 128 bits
        // of YMM6-15 may be corrupted by the callee.
        //
        // This is different from XMM where only 0-5 are volatile.
        // See: https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention
        return 15;  // All YMM registers are effectively volatile for 256-bit operations
    }
};

} // namespace forge
