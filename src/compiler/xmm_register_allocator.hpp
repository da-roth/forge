#pragma once

#include "register_allocator_base.hpp"
#include <asmjit/x86.h>

namespace forge {

/**
 * XMM register allocator for SSE2 instruction set.
 * Manages XMM0-XMM15 registers for scalar and packed SSE2 operations.
 */
class XmmRegisterAllocator : public RegisterAllocatorBase<asmjit::x86::Xmm, 16> {
public:
    XmmRegisterAllocator() : RegisterAllocatorBase() {}

    asmjit::x86::Xmm getRegister(int index) const override {
        static const asmjit::x86::Xmm regs[] = {
            asmjit::x86::xmm0,  asmjit::x86::xmm1,  asmjit::x86::xmm2,  asmjit::x86::xmm3,
            asmjit::x86::xmm4,  asmjit::x86::xmm5,  asmjit::x86::xmm6,  asmjit::x86::xmm7,
            asmjit::x86::xmm8,  asmjit::x86::xmm9,  asmjit::x86::xmm10, asmjit::x86::xmm11,
            asmjit::x86::xmm12, asmjit::x86::xmm13, asmjit::x86::xmm14, asmjit::x86::xmm15
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
