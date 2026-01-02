#pragma once

#include "register_allocator_base.hpp"
#include <asmjit/x86.h>

namespace forge {

/**
 * XMM register allocator for SSE2 instruction set.
 * Manages XMM0-XMM15 registers for scalar and packed SSE2 operations.
 */
class XmmRegisterAllocator : public RegisterAllocatorBase<asmjit::x86::Vec, 16> {
public:
    XmmRegisterAllocator() : RegisterAllocatorBase() {}

    asmjit::x86::Vec getRegister(int index) const override {
        return asmjit::x86::xmm(index);
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
