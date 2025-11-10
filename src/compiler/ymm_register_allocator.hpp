#pragma once

#include "register_allocator_base.hpp"
#include <asmjit/x86.h>

namespace forge {

/**
 * YMM register allocator for AVX2 instruction set.
 * Manages YMM0-YMM15 registers for 256-bit packed AVX2 operations.
 * 
 * CRITICAL: This allocator is completely separate from XMM registers.
 * It only tracks YMM registers, preventing the XMM/YMM confusion that
 * caused release build failures.
 * 
 * Platform-specific details:
 * - Windows x64: YMM0-YMM5 are volatile, YMM6-YMM15 are non-volatile
 * - Linux x64: All YMM registers are volatile
 * - Alignment: 32-byte alignment required for vmovapd, vmovaps
 */
class YmmRegisterAllocator : public RegisterAllocatorBase<asmjit::x86::Ymm, 16> {
public:
    static constexpr size_t ALIGNMENT = 32;  // YMM requires 32-byte alignment
    static constexpr int NUM_YMM_REGS = 16;  // YMM0-YMM15
    static constexpr int VECTOR_WIDTH = 4;   // YMM processes 4 doubles
    
    YmmRegisterAllocator() : RegisterAllocatorBase() {
        // CRITICAL FIX: Blacklist YMM14 and YMM15 due to corruption issues
        // These registers get pre-corrupted with values like 0.002/0.003
        // and cause lane corruption in AVX2 operations
        setBlacklisted(14, true);  // Blacklist YMM14
        setBlacklisted(15, true);  // Blacklist YMM15
    }
    
    /**
     * Get the YMM register for a given index.
     * @param index Register index (0-15)
     * @return The corresponding YMM register
     */
    asmjit::x86::Ymm getRegister(int index) const override {
        static const asmjit::x86::Ymm registers[] = {
            asmjit::x86::ymm0,  asmjit::x86::ymm1,  asmjit::x86::ymm2,  asmjit::x86::ymm3,
            asmjit::x86::ymm4,  asmjit::x86::ymm5,  asmjit::x86::ymm6,  asmjit::x86::ymm7,
            asmjit::x86::ymm8,  asmjit::x86::ymm9,  asmjit::x86::ymm10, asmjit::x86::ymm11,
            asmjit::x86::ymm12, asmjit::x86::ymm13, asmjit::x86::ymm14, asmjit::x86::ymm15
        };
        
        if (index >= 0 && index < NUM_YMM_REGS) {
            return registers[index];
        }
        
        // Return YMM0 as fallback (should never happen with proper bounds checking)
        return asmjit::x86::ymm0;
    }
    
    /**
     * Save callee-saved YMM registers (YMM6-YMM15 on Windows).
     * Note: On Windows, we need to save the upper 128 bits of YMM6-YMM15.
     * The lower 128 bits (XMM part) are handled separately by the calling convention.
     */
    void emitSaveCalleeRegisters(asmjit::x86::Assembler& a, int stackOffset = 0) {
#ifdef _WIN32
        // Windows x64: Save upper 128 bits of YMM6-YMM15
        // Use vextractf128 to extract upper half
        for (int i = 6; i < 16; i++) {
            // Save full YMM register (256 bits)
            a.vmovupd(asmjit::x86::ymmword_ptr(asmjit::x86::rsp, stackOffset + (i - 6) * 32), 
                      getRegister(i));
        }
#else
        // Linux x64: All YMM registers are volatile, nothing to save
        (void)a;
        (void)stackOffset;
#endif
    }
    
    /**
     * Restore callee-saved YMM registers.
     */
    void emitRestoreCalleeRegisters(asmjit::x86::Assembler& a, int stackOffset = 0) {
#ifdef _WIN32
        // Windows x64: Restore YMM6-YMM15
        for (int i = 6; i < 16; i++) {
            a.vmovupd(getRegister(i),
                      asmjit::x86::ymmword_ptr(asmjit::x86::rsp, stackOffset + (i - 6) * 32));
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
        return 10 * 32;  // YMM6-YMM15 = 10 registers * 32 bytes each
#else
        return 0;  // Linux doesn't require saving YMM registers
#endif
    }
    
    /**
     * Check if a memory address is properly aligned for YMM operations.
     */
    static bool isAligned(const void* ptr) {
        return (reinterpret_cast<uintptr_t>(ptr) & (ALIGNMENT - 1)) == 0;
    }
    
    /**
     * Check if an offset is properly aligned for YMM operations.
     */
    static bool isOffsetAligned(size_t offset) {
        return (offset & (ALIGNMENT - 1)) == 0;
    }
    
    /**
     * Platform-specific volatile register range.
     * CRITICAL FIX: This correctly identifies YMM0-YMM5 as volatile,
     * not XMM registers. This fixes the register corruption bug.
     */
    int getFirstVolatileReg() const override {
        return 0;  // YMM0 is always volatile
    }
    
    int getLastVolatileReg() const override {
#ifdef _WIN32
        return 5;  // Windows: YMM0-YMM5 are volatile
#else
        return 15; // Linux: All YMM registers are volatile
#endif
    }
    
    /**
     * Invalidate registers after a function call.
     * CRITICAL FIX: This correctly invalidates YMM0-YMM5, not XMM0-XMM5.
     * This was the core bug causing release build failures.
     */
    void invalidateAfterCall() {
        invalidateVolatileRegisters();
        // After this call:
        // - YMM0-YMM5 are invalidated (volatile)
        // - YMM6-YMM15 remain valid (callee-saved)
        // This is correct for YMM registers!
    }
    
    /**
     * Setup MXCSR for consistent AVX2 operation.
     * Sets FTZ (Flush To Zero) and DAZ (Denormals Are Zero) for performance.
     */
    static void emitSetupMXCSR(asmjit::x86::Assembler& a) {
        // Save current MXCSR
        a.sub(asmjit::x86::rsp, 8);
        a.stmxcsr(asmjit::x86::dword_ptr(asmjit::x86::rsp));
        
        // Set FTZ (bit 15) and DAZ (bit 6)
        a.or_(asmjit::x86::dword_ptr(asmjit::x86::rsp), 0x8040);
        
        // Load modified MXCSR
        a.ldmxcsr(asmjit::x86::dword_ptr(asmjit::x86::rsp));
        a.add(asmjit::x86::rsp, 8);
    }
    
    /**
     * Ensure stack is aligned for YMM operations.
     * AVX2 requires 32-byte stack alignment.
     */
    static void emitAlignStack(asmjit::x86::Assembler& a) {
        a.and_(asmjit::x86::rsp, -32);  // Align to 32 bytes
    }
    
    /**
     * Debug helper: Get register name as string.
     */
    static const char* getRegisterName(int index) {
        static const char* names[] = {
            "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
            "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"
        };
        
        if (index >= 0 && index < NUM_YMM_REGS) {
            return names[index];
        }
        return "ymm?";
    }
    
    /**
     * Calculate aligned workspace size for YMM operations.
     * Ensures all node data is 32-byte aligned.
     */
    static size_t calculateAlignedWorkspaceSize(size_t nodeCount) {
        // Each node needs 4 doubles (32 bytes) for YMM
        size_t baseSize = nodeCount * VECTOR_WIDTH * sizeof(double);
        // Round up to next 32-byte boundary
        return (baseSize + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    }
};

} // namespace forge