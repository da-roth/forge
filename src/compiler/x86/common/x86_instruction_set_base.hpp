#pragma once

#include "../../interfaces/instruction_set.hpp"
#include <cstdint>
#include <cmath>

namespace forge {

// Base class for x86-64 instruction sets (SSE2, AVX2, etc.)
// Contains ONLY the logic that doesn't need XMM/YMM specifics:
// - Win64 ABI calling convention 
// - Stack management
// - GP register save/restore
// - Function call setup/cleanup
class X86InstructionSetBase : public IInstructionSet {
protected:
    // Abstract methods that derived classes must implement for vector register specifics
    
    // Save callee-saved vector registers (XMM6-XMM15 for SSE2, YMM6-YMM15 for AVX2)
    virtual void emitSaveVectorRegisters(asmjit::x86::Assembler& a) const = 0;
    
    // Restore callee-saved vector registers
    virtual void emitRestoreVectorRegisters(asmjit::x86::Assembler& a) const = 0;
    
    // Get stack space needed for vector registers (160 bytes for SSE2, 320 bytes for AVX2)
    virtual int getVectorStackSpace() const = 0;

    // Helper methods for common x86-64 operations (no SIMD register specifics)
    
    // Setup function call - save caller-saved GP registers, allocate shadow space
    void beginFunctionCall(asmjit::x86::Assembler& a) const;
    
    // Cleanup after function call - restore caller-saved GP registers, clean shadow space  
    void endFunctionCall(asmjit::x86::Assembler& a) const;
    
    // Call a function pointer and invalidate volatile registers
    void callFunctionAndInvalidate(asmjit::x86::Assembler& a, uint64_t functionPtr, IRegisterAllocator& regState) const;

public:
    // Common prologue/epilogue implementation (extracted from working SSE2 code)
    void emitPrologue(asmjit::x86::Assembler& a) override;
    void emitEpilogue(asmjit::x86::Assembler& a) override;
    void emitSaveCalleeRegisters(asmjit::x86::Assembler& a) override;
    void emitRestoreCalleeRegisters(asmjit::x86::Assembler& a) override;
    void emitMoveArgsToRegisters(asmjit::x86::Assembler& a) override;
    int getStackSpaceNeeded() const override;
};

} // namespace forge