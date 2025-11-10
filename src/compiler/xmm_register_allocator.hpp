#pragma once

#include "register_allocator_base.hpp"
#include <asmjit/x86.h>

namespace forge {

/**
 * XMM register allocator for SSE2 instruction set.
 * Manages XMM0-XMM15 registers for scalar and packed SSE2 operations.
 * 
 * Platform-specific details:
 * - Windows x64: XMM0-XMM5 are volatile, XMM6-XMM15 are non-volatile
 * - Linux x64: All XMM registers are volatile
 * - Alignment: 16-byte alignment required for movapd, movaps
 */
class XmmRegisterAllocator : public RegisterAllocatorBase<asmjit::x86::Xmm, 16> {
public:
    static constexpr size_t ALIGNMENT = 16;  // XMM requires 16-byte alignment
    static constexpr int NUM_XMM_REGS = 16;  // XMM0-XMM15
    
    XmmRegisterAllocator() : RegisterAllocatorBase() {}
    
    /**
     * Get the XMM register for a given index.
     * @param index Register index (0-15)
     * @return The corresponding XMM register
     */
    asmjit::x86::Xmm getRegister(int index) const override {
        static const asmjit::x86::Xmm registers[] = {
            asmjit::x86::xmm0,  asmjit::x86::xmm1,  asmjit::x86::xmm2,  asmjit::x86::xmm3,
            asmjit::x86::xmm4,  asmjit::x86::xmm5,  asmjit::x86::xmm6,  asmjit::x86::xmm7,
            asmjit::x86::xmm8,  asmjit::x86::xmm9,  asmjit::x86::xmm10, asmjit::x86::xmm11,
            asmjit::x86::xmm12, asmjit::x86::xmm13, asmjit::x86::xmm14, asmjit::x86::xmm15
        };
        
        if (index >= 0 && index < NUM_XMM_REGS) {
            return registers[index];
        }
        
        // Return XMM0 as fallback (should never happen with proper bounds checking)
        return asmjit::x86::xmm0;
    }
    
    /**
     * Save callee-saved XMM registers (XMM6-XMM15 on Windows).
     * Note: Only the lower 128 bits need to be saved for XMM registers.
     */
    void emitSaveCalleeRegisters(asmjit::x86::Assembler& a, int stackOffset = 0) {
#ifdef _WIN32
        // Windows x64: Save XMM6-XMM15
        for (int i = 6; i < 16; i++) {
            a.movdqu(asmjit::x86::xmmword_ptr(asmjit::x86::rsp, stackOffset + (i - 6) * 16), 
                     getRegister(i));
        }
#else
        // Linux x64: All XMM registers are volatile, nothing to save
        (void)a;
        (void)stackOffset;
#endif
    }
    
    /**
     * Restore callee-saved XMM registers.
     */
    void emitRestoreCalleeRegisters(asmjit::x86::Assembler& a, int stackOffset = 0) {
#ifdef _WIN32
        // Windows x64: Restore XMM6-XMM15
        for (int i = 6; i < 16; i++) {
            a.movdqu(getRegister(i),
                     asmjit::x86::xmmword_ptr(asmjit::x86::rsp, stackOffset + (i - 6) * 16));
        }
#else
        // Linux x64: Nothing to restore
        (void)a;
        (void)stackOffset;
#endif
    }
    
    /**
     * Get the stack space needed for saving callee-saved registers.
     */
    static constexpr int getCalleeStackSpace() {
#ifdef _WIN32
        return 10 * 16;  // XMM6-XMM15 = 10 registers * 16 bytes each
#else
        return 0;  // Linux doesn't require saving XMM registers
#endif
    }
    
    /**
     * Check if a memory address is properly aligned for XMM operations.
     */
    static bool isAligned(const void* ptr) {
        return (reinterpret_cast<uintptr_t>(ptr) & (ALIGNMENT - 1)) == 0;
    }
    
    /**
     * Platform-specific volatile register range.
     * Override base class defaults if needed for different platforms.
     */
    int getFirstVolatileReg() const override {
        return 0;  // XMM0 is always volatile
    }
    
    int getLastVolatileReg() const override {
#ifdef _WIN32
        return 5;  // Windows: XMM0-XMM5 are volatile
#else
        return 15; // Linux: All XMM registers are volatile
#endif
    }
    
    /**
     * Invalidate registers after a function call.
     * This correctly handles XMM registers based on platform ABI.
     */
    void invalidateAfterCall() {
        invalidateVolatileRegisters();
    }
    
    /**
     * Debug helper: Get register name as string.
     */
    static const char* getRegisterName(int index) {
        static const char* names[] = {
            "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
            "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
        };
        
        if (index >= 0 && index < NUM_XMM_REGS) {
            return names[index];
        }
        return "xmm?";
    }
};

} // namespace forge