#include "x86_instruction_set_base.hpp"

namespace forge {

// Helper methods for common x86-64 function call setup (no SIMD register specifics)

void X86InstructionSetBase::beginFunctionCall(asmjit::x86::Assembler& a) const {
#ifdef _WIN32
    // Win64 ABI: Save caller-saved GP registers and allocate shadow space
    a.push(asmjit::x86::rdi);
    a.push(asmjit::x86::rsi);
    a.sub(asmjit::x86::rsp, 32);  // Shadow space (32 bytes)
#else
    // Linux System V ABI: Save RDI/RSI (caller-saved, contain workspace/gradient pointers)
    // We need to preserve them across the call since they contain our working pointers
    a.push(asmjit::x86::rdi);
    a.push(asmjit::x86::rsi);
    // Stack is now 16-byte aligned after 2 pushes (each push is 8 bytes)
#endif
}

void X86InstructionSetBase::endFunctionCall(asmjit::x86::Assembler& a) const {
#ifdef _WIN32
    // Win64 ABI: Clean up shadow space and restore registers
    a.add(asmjit::x86::rsp, 32);
    a.pop(asmjit::x86::rsi);
    a.pop(asmjit::x86::rdi);
#else
    // Linux System V ABI: Restore RDI/RSI (in reverse order of push)
    a.pop(asmjit::x86::rsi);
    a.pop(asmjit::x86::rdi);
#endif
}

void X86InstructionSetBase::callFunctionAndInvalidate(asmjit::x86::Assembler& a, uint64_t functionPtr, IRegisterAllocator& regState) const {
    // Move function pointer to RAX
    a.mov(asmjit::x86::rax, functionPtr);
    
    // Call the function
    a.call(asmjit::x86::rax);
    
    // Invalidate volatile registers (crucial for register allocator)
    regState.invalidateVolatileRegisters();
}

// Common prologue/epilogue implementation extracted from working SSE2 code
void X86InstructionSetBase::emitPrologue(asmjit::x86::Assembler& a) {
    // Standard frame pointer setup
    a.push(asmjit::x86::rbp);
    a.mov(asmjit::x86::rbp, asmjit::x86::rsp);
    
    // Allocate stack space for callee-saved registers
    a.sub(asmjit::x86::rsp, getStackSpaceNeeded());
    
    // Save callee-saved registers
    emitSaveCalleeRegisters(a);
    
    // Move arguments to expected registers
    emitMoveArgsToRegisters(a);
}

void X86InstructionSetBase::emitEpilogue(asmjit::x86::Assembler& a) {
    // Restore callee-saved registers
    emitRestoreCalleeRegisters(a);
    
    // Restore stack
    a.add(asmjit::x86::rsp, getStackSpaceNeeded());
    a.pop(asmjit::x86::rbp);
    
    // Return
    a.ret();
}

// Common callee-saved register handling (GP registers only)
void X86InstructionSetBase::emitSaveCalleeRegisters(asmjit::x86::Assembler& a) {
#ifdef _WIN32
    // Win64 ABI: Save all callee-saved general purpose registers
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 192), asmjit::x86::rbx);   // RBX at offset 192
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 200), asmjit::x86::rdi);   // RDI at offset 200
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 208), asmjit::x86::rsi);   // RSI at offset 208
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 216), asmjit::x86::r12);   // R12 at offset 216
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 224), asmjit::x86::r13);   // R13 at offset 224
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 232), asmjit::x86::r14);   // R14 at offset 232
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 240), asmjit::x86::r15);   // R15 at offset 240
#else
    // Linux System V ABI: Save only the callee-saved registers we actually use
    // RBX, R12-R15 are callee-saved (we don't currently use them, but save for safety)
    // RDI, RSI are caller-saved, so we don't save them here
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 0), asmjit::x86::rbx);    // RBX at offset 0
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 8), asmjit::x86::r12);    // R12 at offset 8
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 16), asmjit::x86::r13);   // R13 at offset 16
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 24), asmjit::x86::r14);   // R14 at offset 24
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 32), asmjit::x86::r15);   // R15 at offset 32
#endif

    // Save vector registers - delegate to derived class (XMM vs YMM specific)
    emitSaveVectorRegisters(a);
}

void X86InstructionSetBase::emitRestoreCalleeRegisters(asmjit::x86::Assembler& a) {
    // Restore vector registers - delegate to derived class (XMM vs YMM specific)
    emitRestoreVectorRegisters(a);

#ifdef _WIN32
    // Win64 ABI: Restore all callee-saved general purpose registers (in reverse order)
    a.mov(asmjit::x86::r15, asmjit::x86::ptr(asmjit::x86::rsp, 240));   // R15
    a.mov(asmjit::x86::r14, asmjit::x86::ptr(asmjit::x86::rsp, 232));   // R14
    a.mov(asmjit::x86::r13, asmjit::x86::ptr(asmjit::x86::rsp, 224));   // R13
    a.mov(asmjit::x86::r12, asmjit::x86::ptr(asmjit::x86::rsp, 216));   // R12
    a.mov(asmjit::x86::rsi, asmjit::x86::ptr(asmjit::x86::rsp, 208));   // RSI
    a.mov(asmjit::x86::rdi, asmjit::x86::ptr(asmjit::x86::rsp, 200));   // RDI
    a.mov(asmjit::x86::rbx, asmjit::x86::ptr(asmjit::x86::rsp, 192));   // RBX
#else
    // Linux System V ABI: Restore callee-saved registers
    a.mov(asmjit::x86::r15, asmjit::x86::ptr(asmjit::x86::rsp, 32));   // R15
    a.mov(asmjit::x86::r14, asmjit::x86::ptr(asmjit::x86::rsp, 24));   // R14
    a.mov(asmjit::x86::r13, asmjit::x86::ptr(asmjit::x86::rsp, 16));   // R13
    a.mov(asmjit::x86::r12, asmjit::x86::ptr(asmjit::x86::rsp, 8));    // R12
    a.mov(asmjit::x86::rbx, asmjit::x86::ptr(asmjit::x86::rsp, 0));    // RBX
#endif
}

int X86InstructionSetBase::getStackSpaceNeeded() const {
#ifdef _WIN32
    // Win64 ABI: 32 (shadow space) + 56 (7 GP regs * 8 bytes) + vector register space
    // Round up to 16-byte alignment
    int vectorSpace = getVectorStackSpace();
    int totalSpace = 32 + 56 + vectorSpace;
    return (totalSpace + 15) & ~15;  // Align to 16 bytes
#else
    // Linux System V ABI: 40 (5 GP regs * 8 bytes) + vector register space
    // Round up to 16-byte alignment
    int vectorSpace = getVectorStackSpace();
    int totalSpace = 40 + vectorSpace;
    return (totalSpace + 15) & ~15;  // Align to 16 bytes
#endif
}

// Move function arguments to expected registers
void X86InstructionSetBase::emitMoveArgsToRegisters(asmjit::x86::Assembler& a) {
#ifdef _WIN32
    // Win64 ABI: RCX = first arg (values), RDX = second arg (gradients), R8 = third arg (count)
    // We want: RDI = values, RSI = gradients (for gradient operations)
    a.mov(asmjit::x86::rdi, asmjit::x86::rcx);  // Move values pointer to RDI
    a.mov(asmjit::x86::rsi, asmjit::x86::rdx);  // Move gradients pointer to RSI
#else
    // Linux System V ABI: RDI = first arg (values), RSI = second arg (gradients)
    // Arguments are already in the correct registers - no move needed!
    (void)a; // Suppress unused parameter warning
#endif
}

} // namespace forge