#include "x86_instruction_set_base.h"

namespace forge {
namespace x86 {

// Helper methods for common x86-64 function call setup (no SIMD register specifics)

void X86InstructionSetBase::beginFunctionCall(asmjit::x86::Assembler& a) const {
    // Save caller-saved GP registers (Win64 ABI)
    a.push(asmjit::x86::rdi);
    a.push(asmjit::x86::rsi);
    
    // Shadow space for Win64 ABI (32 bytes)
    a.sub(asmjit::x86::rsp, 32);
}

void X86InstructionSetBase::endFunctionCall(asmjit::x86::Assembler& a) const {
    // Clean up shadow space
    a.add(asmjit::x86::rsp, 32);
    
    // Restore caller-saved GP registers
    a.pop(asmjit::x86::rsi);
    a.pop(asmjit::x86::rdi);
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
    // Save all callee-saved general purpose registers (Win64 ABI)
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 192), asmjit::x86::rbx);   // RBX at offset 192
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 200), asmjit::x86::rdi);   // RDI at offset 200  
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 208), asmjit::x86::rsi);   // RSI at offset 208
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 216), asmjit::x86::r12);   // R12 at offset 216
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 224), asmjit::x86::r13);   // R13 at offset 224
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 232), asmjit::x86::r14);   // R14 at offset 232
    a.mov(asmjit::x86::ptr(asmjit::x86::rsp, 240), asmjit::x86::r15);   // R15 at offset 240
    
    // Save vector registers - delegate to derived class (XMM vs YMM specific)
    emitSaveVectorRegisters(a);
}

void X86InstructionSetBase::emitRestoreCalleeRegisters(asmjit::x86::Assembler& a) {
    // Restore vector registers - delegate to derived class (XMM vs YMM specific)
    emitRestoreVectorRegisters(a);
    
    // Restore all callee-saved general purpose registers (in reverse order)
    a.mov(asmjit::x86::r15, asmjit::x86::ptr(asmjit::x86::rsp, 240));   // R15 
    a.mov(asmjit::x86::r14, asmjit::x86::ptr(asmjit::x86::rsp, 232));   // R14
    a.mov(asmjit::x86::r13, asmjit::x86::ptr(asmjit::x86::rsp, 224));   // R13
    a.mov(asmjit::x86::r12, asmjit::x86::ptr(asmjit::x86::rsp, 216));   // R12
    a.mov(asmjit::x86::rsi, asmjit::x86::ptr(asmjit::x86::rsp, 208));   // RSI
    a.mov(asmjit::x86::rdi, asmjit::x86::ptr(asmjit::x86::rsp, 200));   // RDI
    a.mov(asmjit::x86::rbx, asmjit::x86::ptr(asmjit::x86::rsp, 192));   // RBX
}

int X86InstructionSetBase::getStackSpaceNeeded() const {
    // 32 (shadow space) + 56 (7 GP regs * 8 bytes) + vector register space
    // Round up to 16-byte alignment
    int vectorSpace = getVectorStackSpace();
    int totalSpace = 32 + 56 + vectorSpace;
    return (totalSpace + 15) & ~15;  // Align to 16 bytes
}

// Move function arguments to expected registers (Win64 ABI)
void X86InstructionSetBase::emitMoveArgsToRegisters(asmjit::x86::Assembler& a) {
    // Win64 ABI: RCX = first arg (values), RDX = second arg (gradients), R8 = third arg (count)
    // We want: RDI = values, RSI = gradients (for gradient operations)
    a.mov(asmjit::x86::rdi, asmjit::x86::rcx);  // Move values pointer to RDI
    a.mov(asmjit::x86::rsi, asmjit::x86::rdx);  // Move gradients pointer to RSI
}

} // namespace x86
} // namespace forge