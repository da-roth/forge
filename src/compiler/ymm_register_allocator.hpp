#pragma once

#include "register_allocator_base.hpp"
#include <asmjit/x86.h>

namespace forge {

/**
 * YMM register allocator for AVX2 instruction set.
 * Manages YMM0-YMM15 registers for 256-bit packed AVX2 operations.
 */
class YmmRegisterAllocator : public RegisterAllocatorBase<asmjit::x86::Ymm, 16> {
public:
    YmmRegisterAllocator() : RegisterAllocatorBase() {
        // Blacklist YMM14 and YMM15 due to corruption issues
        setBlacklisted(14, true);
        setBlacklisted(15, true);
    }

    asmjit::x86::Ymm getRegister(int index) const override {
        static const asmjit::x86::Ymm regs[] = {
            asmjit::x86::ymm0,  asmjit::x86::ymm1,  asmjit::x86::ymm2,  asmjit::x86::ymm3,
            asmjit::x86::ymm4,  asmjit::x86::ymm5,  asmjit::x86::ymm6,  asmjit::x86::ymm7,
            asmjit::x86::ymm8,  asmjit::x86::ymm9,  asmjit::x86::ymm10, asmjit::x86::ymm11,
            asmjit::x86::ymm12, asmjit::x86::ymm13, asmjit::x86::ymm14, asmjit::x86::ymm15
        };
        return regs[index];
    }

    int getFirstVolatileReg() const override { return 0; }
    int getLastVolatileReg() const override {
#ifdef _WIN32
        return 5;
#else
        return 15;
#endif
    }
};

} // namespace forge
