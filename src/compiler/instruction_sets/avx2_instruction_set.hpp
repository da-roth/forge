#pragma once

#include "x86_instruction_set_base.hpp"
#include "../register_allocator_base.hpp"
#include "avx2_transcendental_helpers.hpp"  // For helper functions
#include "../instruction_tracer.hpp"        // For runtime tracing
#include <cmath>
#include <climits>
#include <cstring>  // For memcpy
#include <unordered_map>
#include <limits>   // For std::numeric_limits

// Forward declarations for scalar transcendental functions (kept for fallback/testing)
extern "C" double call_std_exp(double x);
extern "C" double call_std_log(double x);

// Vectorized math functions (process 4 doubles at once using SLEEF)
extern "C" void call_vexp4d(const double* input, double* out);
extern "C" void call_vlog4d(const double* input, double* out);
extern "C" void call_vsin4d(const double* input, double* out);
extern "C" void call_vcos4d(const double* input, double* out);
extern "C" void call_vtan4d(const double* input, double* out);
extern "C" void call_vpow4d(const double* base, const double* exp, double* out);

// Include intrinsics header for AVX2
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

// Vectorized math functions will be added later
#include "avx2_transcendental_helpers.hpp"

namespace forge {

// Simple struct to hold constant info - matches ForgeEngine::ConstantInfo
// Defined here to avoid circular dependency
struct AVX2ConstantInfo {
    size_t poolOffset;  // Offset within the constant pool
    double value;       // The constant value
};

// AVX2 Packed instruction set implementation
// This uses AVX2 instructions to process FOUR doubles at a time (packed operations)
// The 'pd' suffix in instructions like 'vaddpd', 'vmulpd' means 'Packed Double'
// Uses YMM registers (256-bit) instead of XMM registers (128-bit)
// Inherits all the working transcendental function patterns from X86InstructionSetBase
class AVX2InstructionSet : public X86InstructionSetBase {
private:
    CompilerConfig config;
    InstructionTracer tracer;
    
    // DEBUG: Helper functions for corruption tracking
    void emitTraceSafeYMM(asmjit::x86::Assembler& a, int regNum, const char* context);
    void emitTraceAllYMMRegisters_UNSAFE(asmjit::x86::Assembler& a, const char* context);
    
public:
    // Constructor
    AVX2InstructionSet(const CompilerConfig& cfg = CompilerConfig::Default()) 
        : config(cfg), tracer(cfg) {}
    
    // Destructor - print trace records if tracing was enabled
    ~AVX2InstructionSet() {
        // Trace printing now happens after forward pass, not in destructor
    }
    
    std::string getName() const override { return "AVX2-Packed"; }
    
    // AVX2 uses YMM0-YMM15 registers
    int getMaxRegisterCount() const override { return 16; }
    
    // AVX2 processes four doubles at a time (256 bits / 64 bits per double)
    int getVectorWidth() const override { return 4; }
    
    bool supportsOperation(forge::OpCode op) const override {
        // AVX2 supports all current operations
        return true;
    }
    
    // Arithmetic operations
    void emitAdd(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        // Perform the operation
        a.vaddpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(srcReg));
        
        // Trace output values after operation with register info
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::ADD, 4, -1, srcReg, dstReg);
    }
    
    void emitSub(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        // Perform the operation
        a.vsubpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(srcReg));
        
        // Trace output values after operation
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::SUB, 4);
    }
    
    void emitMul(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        // Perform the operation
        a.vmulpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(srcReg));
        
        // Trace output values after operation
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::MUL, 4);
    }
    
    void emitDiv(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        // Perform the operation: dst = dst / src
        a.vdivpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(srcReg));
        
        // Trace output values after operation
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::DIV, 4);
    }
    
    // Unary operations
    void emitNeg(asmjit::x86::Assembler& a, int dstReg, int tempReg) override;
    void emitAbs(asmjit::x86::Assembler& a, int dstReg, int tempReg) override;
    void emitSqrt(asmjit::x86::Assembler& a, int dstReg) override {
        // Trace input values before operation
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::SQRT, 4);

        // Perform the operation
        a.vsqrtpd(getYmmRegister(dstReg), getYmmRegister(dstReg));

        // Trace output values after operation
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::SQRT, 4);
    }
    
    // Memory operations
    void emitLoad(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) override;
    void emitStore(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) override;
    void emitLoadFromConstantPool(asmjit::x86::Assembler& a, int dstReg, 
                                  const asmjit::Label& poolLabel, size_t offset) override;
    
    // Register-to-register move
    void emitMove(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        if (dstReg != srcReg) {
            a.vmovapd(getYmmRegister(dstReg), getYmmRegister(srcReg));
            // Trace the move operation
            tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::MOVE, 4, -1, srcReg, dstReg);
        }
    }
    
    // Comparison operations
    void emitCmpLT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    void emitCmpLE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    void emitCmpGT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    void emitCmpGE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    void emitCmpEQ(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    void emitCmpNE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    
    // Min/Max operations
    void emitMin(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        // Perform the operation: dst = min(dst, src)
        a.vminpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(srcReg));
        
        // Trace operation with register info
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::MIN, 4, -1, dstReg, srcReg);
    }
    
    void emitMax(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        // Perform the operation: dst = max(dst, src)
        a.vmaxpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(srcReg));
        
        // Trace operation with register info
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::MAX, 4, -1, dstReg, srcReg);
    }
    
    // Special operations
    void emitSquare(asmjit::x86::Assembler& a, int dstReg) override {
        // Perform the operation: dst = dst * dst
        a.vmulpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(dstReg));
        
        // Trace operation with register info
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::SQUARE, 4, -1, dstReg, dstReg);
    }
    
    // Transcendental functions (will need vectorized implementations)
    // Vectorized transcendental functions using SLEEF
    void emitExp(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        // Use vectorized SLEEF implementation: ONE call for all 4 doubles!
        auto exp_addr = reinterpret_cast<uint64_t>(&call_vexp4d);
        emitVectorizedMathCall1Arg(a, dstReg, srcReg, regState, exp_addr);
    }

    void emitLog(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        // Use vectorized SLEEF implementation: ONE call for all 4 doubles!
        auto log_addr = reinterpret_cast<uint64_t>(&call_vlog4d);
        emitVectorizedMathCall1Arg(a, dstReg, srcReg, regState, log_addr);
    }
private:
    // Generic helper function for calling external scalar math functions on YMM registers
    // Takes each lane from srcReg, calls the function, stores results in dstReg
    void emitScalarMathFunctionCall(asmjit::x86::Assembler& a, int dstReg, int srcReg, 
                                   IRegisterAllocator& regState, uint64_t funcAddr) {
        using namespace asmjit::x86;
        
        // Get YMM registers
        auto getYmmRegister = [](int index) { return ymm(index); };
        
        // CRITICAL: Save ALL volatile YMM registers before function calls
        // forge's register allocator tracks register contents, so we must preserve them
        
        // 1. Save all volatile YMM registers (YMM0-YMM5 on Windows)
        int numVolatileRegs = regState.getLastVolatileReg() - regState.getFirstVolatileReg() + 1;
        a.sub(rsp, numVolatileRegs * 32);  // 32 bytes per YMM register
        
        for (int i = regState.getFirstVolatileReg(); i <= regState.getLastVolatileReg(); i++) {
            int offset = (i - regState.getFirstVolatileReg()) * 32;
            a.vmovupd(ymmword_ptr(rsp, offset), getYmmRegister(i));
        }
        
        // 2. Save general-purpose registers used by function calls
        a.push(rax);
        a.push(rcx);
        a.push(rdx);
        a.push(rsi);
        a.push(rdi);
        a.push(r8);
        a.push(r9);
        a.push(r10);
        a.push(r11);
        
        // 3. Allocate working space for 4 doubles
        a.sub(rsp, 32);
        
        // 4. Store input YMM to working space (use saved srcReg data)
        // We need to reload from the saved volatile registers if srcReg is volatile
        if (srcReg >= regState.getFirstVolatileReg() && srcReg <= regState.getLastVolatileReg()) {
            int srcOffset = (srcReg - regState.getFirstVolatileReg()) * 32;
            a.vmovupd(ymm15, ymmword_ptr(rsp, 32 + 72 + srcOffset)); // Skip work space + GP regs + find srcReg
            a.vmovupd(ymmword_ptr(rsp), ymm15);
        } else {
            a.vmovupd(ymmword_ptr(rsp), getYmmRegister(srcReg));
        }
        
        // 5. Process each lane by calling the external function
        for (int lane = 0; lane < 4; lane++) {
            int offset = lane * 8;
            
            // Load lane value to XMM0 (first argument register)
            a.vmovsd(xmm0, qword_ptr(rsp, offset));
            
            // Align stack to 16 bytes for function call (required by System V ABI)
            a.sub(rsp, 8);
            
            // Call the external function
            a.mov(rax, funcAddr);
            a.call(rax);
            
            // Restore stack alignment
            a.add(rsp, 8);
            
            // Store result back to working space (result is in XMM0)
            a.vmovsd(qword_ptr(rsp, offset), xmm0);
        }
        
        // 6. Load results to destination YMM register
        if (dstReg >= regState.getFirstVolatileReg() && dstReg <= regState.getLastVolatileReg()) {
            // Destination is volatile - we'll restore it later
            int dstOffset = (dstReg - regState.getFirstVolatileReg()) * 32;
            a.vmovupd(ymmword_ptr(rsp, 32 + 72 + dstOffset), ymm14); // Temp storage
            a.vmovupd(ymm14, ymmword_ptr(rsp)); // Load result
            a.vmovupd(ymmword_ptr(rsp, 32 + 72 + dstOffset), ymm14); // Store to volatile save area
        } else {
            a.vmovupd(getYmmRegister(dstReg), ymmword_ptr(rsp));
        }
        
        // 7. Clean up working space
        a.add(rsp, 32);
        
        // 8. Restore general-purpose registers
        a.pop(r11);
        a.pop(r10);
        a.pop(r9);
        a.pop(r8);
        a.pop(rdi);
        a.pop(rsi);
        a.pop(rdx);
        a.pop(rcx);
        a.pop(rax);
        
        // 9. Restore volatile YMM registers
        for (int i = regState.getFirstVolatileReg(); i <= regState.getLastVolatileReg(); i++) {
            int offset = (i - regState.getFirstVolatileReg()) * 32;
            a.vmovupd(getYmmRegister(i), ymmword_ptr(rsp, offset));
        }
        a.add(rsp, numVolatileRegs * 32);
        
        // 10. CRITICAL: Tell forge that volatile registers may be invalid
        // This is the key fix - inform the register allocator about the function call impact
        regState.invalidateVolatileRegisters();
    }

    // Vectorized one-argument math function call (processes all 4 lanes with one call)
    // Used for exp, log, sin, cos, tan, etc.
    void emitVectorizedMathCall1Arg(asmjit::x86::Assembler& a, int dstReg, int srcReg,
                                     IRegisterAllocator& regState, uint64_t funcAddr) {
        using namespace asmjit::x86;

        // Save RAX (used for function address)
        a.push(rax);

        // CRITICAL: Save RDI/RSI - kernel uses these as workspace pointers on Linux
        a.push(rdi);
        a.push(rsi);

        // Allocate space for 2 YMM values (input + result = 64 bytes) plus alignment.
        // On Windows, include shadow space (32 bytes) in a single allocation.
        //
        // Stack layout after allocation:
        // Windows: [rsp+0..31]=shadow, [rsp+40..71]=input, [rsp+72..103]=result (total: 104)
        // Linux:   [rsp+8..39]=input, [rsp+40..71]=result (total: 72)
#ifdef _WIN32
        constexpr int kInputOffset = 40;   // shadow(32) + alignment(8)
        constexpr int kResultOffset = 72;  // shadow(32) + alignment(8) + input(32)
        constexpr int kTotalStack = 104;   // shadow(32) + alignment(8) + input(32) + result(32)
#else
        constexpr int kInputOffset = 8;
        constexpr int kResultOffset = 40;
        constexpr int kTotalStack = 72;    // alignment(8) + input(32) + result(32)
#endif

        // Single stack allocation - all space including shadow space on Windows
        a.sub(rsp, kTotalStack);

        // Store input
        a.vmovupd(ymmword_ptr(rsp, kInputOffset), ymm(srcReg));

        // Set up function arguments - pointers are correct relative to final RSP
#ifdef _WIN32
        a.lea(rcx, ptr(rsp, kInputOffset));
        a.lea(rdx, ptr(rsp, kResultOffset));
#else
        a.lea(rdi, ptr(rsp, kInputOffset));
        a.lea(rsi, ptr(rsp, kResultOffset));
#endif

        // Call the vectorized function (ONE call for all 4 doubles!)
        a.mov(rax, funcAddr);
        a.call(rax);

        // Load result
        a.vmovupd(ymm(dstReg), ymmword_ptr(rsp, kResultOffset));

        // Cleanup stack space
        a.add(rsp, kTotalStack);

        // Restore RDI/RSI workspace pointers
        a.pop(rsi);
        a.pop(rdi);

        // Restore RAX
        a.pop(rax);

        // CRITICAL: Inform the register allocator that volatile registers may have been modified
        // Without this, the compiler will use stale register values!
        regState.invalidateVolatileRegisters();
    }

    // Vectorized two-argument math function call (processes all 4 lanes with one call)
    // This is much more efficient than calling scalar functions 4 times
    void emitVectorizedMathCall2Args(asmjit::x86::Assembler& a, int dstReg, int arg1Reg, int arg2Reg,
                                      IRegisterAllocator& regState, uint64_t funcAddr) {
        using namespace asmjit::x86;

        // Save RAX (used for function address)
        a.push(rax);

        // CRITICAL: Save RDI/RSI - kernel uses these as workspace pointers on Linux
        a.push(rdi);
        a.push(rsi);

        // Allocate space for 3 YMM values (arg1 + arg2 + result = 96 bytes) plus alignment.
        // On Windows, include shadow space (32 bytes) in a single allocation.
        //
        // Stack layout after allocation:
        // Windows: [rsp+0..31]=shadow, [rsp+40..71]=arg1, [rsp+72..103]=arg2, [rsp+104..135]=result (total: 136)
        // Linux:   [rsp+8..39]=arg1, [rsp+40..71]=arg2, [rsp+72..103]=result (total: 104)
#ifdef _WIN32
        constexpr int kArg1Offset = 40;    // shadow(32) + alignment(8)
        constexpr int kArg2Offset = 72;    // shadow(32) + alignment(8) + arg1(32)
        constexpr int kResultOffset = 104; // shadow(32) + alignment(8) + arg1(32) + arg2(32)
        constexpr int kTotalStack = 136;   // shadow(32) + alignment(8) + arg1(32) + arg2(32) + result(32)
#else
        constexpr int kArg1Offset = 8;
        constexpr int kArg2Offset = 40;
        constexpr int kResultOffset = 72;
        constexpr int kTotalStack = 104;   // alignment(8) + arg1(32) + arg2(32) + result(32)
#endif

        // Single stack allocation - all space including shadow space on Windows
        a.sub(rsp, kTotalStack);

        // Store arguments
        a.vmovupd(ymmword_ptr(rsp, kArg1Offset), ymm(arg1Reg));
        a.vmovupd(ymmword_ptr(rsp, kArg2Offset), ymm(arg2Reg));

        // Set up function arguments - pointers are correct relative to final RSP
#ifdef _WIN32
        a.lea(rcx, ptr(rsp, kArg1Offset));
        a.lea(rdx, ptr(rsp, kArg2Offset));
        a.lea(r8, ptr(rsp, kResultOffset));
#else
        a.lea(rdi, ptr(rsp, kArg1Offset));
        a.lea(rsi, ptr(rsp, kArg2Offset));
        a.lea(rdx, ptr(rsp, kResultOffset));
#endif

        // Call the vectorized function (ONE call for all 4 doubles!)
        a.mov(rax, funcAddr);
        a.call(rax);

        // Load result
        a.vmovupd(ymm(dstReg), ymmword_ptr(rsp, kResultOffset));

        // Cleanup stack space
        a.add(rsp, kTotalStack);

        // Restore RDI/RSI workspace pointers
        a.pop(rsi);
        a.pop(rdi);

        // Restore RAX
        a.pop(rax);

        // CRITICAL: Inform the register allocator that volatile registers may have been modified
        // Without this, the compiler will use stale register values!
        regState.invalidateVolatileRegisters();
    }

    // Two-argument scalar math function call (for pow, etc.)
    void emitScalarMathFunctionCall2Args(asmjit::x86::Assembler& a, int dstReg, int arg1Reg, int arg2Reg, 
                                        IRegisterAllocator& regState, uint64_t funcAddr) {
        using namespace asmjit::x86;
        
        // Callee-safe version with minimal register preservation
        auto ymm_src1 = ymm(arg1Reg);
        auto ymm_src2 = ymm(arg2Reg);
        auto ymm_dst = ymm(dstReg);
        
        // Save RAX (we use it for function address)
        a.push(rax);
        
        // Windows x64 ABI: XMM0-XMM5 are volatile (caller-saved)
        // We only need to save YMM0-2 since we use XMM0-1 and the callee might use others
        // But only if they contain live data (check with regState)
        int savedRegs = 0;
        constexpr int maxSaveReg = 3;  // Save YMM0-YMM2
        
        for (int i = 0; i < maxSaveReg; i++) {
            if (i != dstReg) {  // Don't save the destination
                savedRegs++;
            }
        }
        
        // Align stack after push rax
        int stackAdjust = 8;  // For RAX push
        int ymmSpace = savedRegs * 32;
        int totalSpace = stackAdjust + ymmSpace + 64 + 8;  // +64 for data, +8 for alignment
        totalSpace = (totalSpace + 15) & ~15;  // Round up to 16
        a.sub(rsp, totalSpace - 8);  // -8 because we already pushed RAX
        
        // Save YMM0-YMM2 (except destination)
        int saveOffset = 0;
        for (int i = 0; i < maxSaveReg; i++) {
            if (i != dstReg) {
                a.vmovupd(ymmword_ptr(rsp, saveOffset), ymm(i));
                saveOffset += 32;
            }
        }
        
        // Store inputs after saved registers
        a.vmovupd(ymmword_ptr(rsp, saveOffset), ymm_src1);
        a.vmovupd(ymmword_ptr(rsp, saveOffset + 32), ymm_src2);
        
        // Process each lane
        for (int i = 0; i < 4; i++) {
            // Load arguments
            a.vmovsd(xmm0, qword_ptr(rsp, saveOffset + i * 8));
            a.vmovsd(xmm1, qword_ptr(rsp, saveOffset + 32 + i * 8));

            // Use platform-aware function call (saves RDI/RSI on Linux, shadow space on Windows)
            beginFunctionCall(a);

            // Call
            a.mov(rax, funcAddr);
            a.call(rax);

            // Restore platform-specific state
            endFunctionCall(a);

            // Store result
            a.vmovsd(qword_ptr(rsp, saveOffset + i * 8), xmm0);
        }
        
        // Load results
        a.vmovupd(ymm_dst, ymmword_ptr(rsp, saveOffset));
        
        // Restore saved YMM registers
        saveOffset = 0;
        for (int i = 0; i < maxSaveReg; i++) {
            if (i != dstReg) {
                a.vmovupd(ymm(i), ymmword_ptr(rsp, saveOffset));
                saveOffset += 32;
            }
        }
        
        // Cleanup
        a.add(rsp, totalSpace - 8);
        a.pop(rax);
    }

    void emitSin(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        // Use vectorized SLEEF implementation: ONE call for all 4 doubles!
        auto sin_addr = reinterpret_cast<uint64_t>(&call_vsin4d);
        emitVectorizedMathCall1Arg(a, dstReg, srcReg, regState, sin_addr);
    }

    void emitCos(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        // Use vectorized SLEEF implementation: ONE call for all 4 doubles!
        auto cos_addr = reinterpret_cast<uint64_t>(&call_vcos4d);
        emitVectorizedMathCall1Arg(a, dstReg, srcReg, regState, cos_addr);
    }

    void emitTan(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        // Use vectorized SLEEF implementation: ONE call for all 4 doubles!
        auto tan_addr = reinterpret_cast<uint64_t>(&call_vtan4d);
        emitVectorizedMathCall1Arg(a, dstReg, srcReg, regState, tan_addr);
    }
    
    void emitPow(asmjit::x86::Assembler& a, int dstReg, int baseReg, int expReg, IRegisterAllocator& regState) override {
        // Use vectorized SLEEF implementation: ONE call for all 4 doubles!
        // This is 4x more efficient than the old scalar loop approach
        auto pow_addr = reinterpret_cast<uint64_t>(&call_vpow4d);
        emitVectorizedMathCall2Args(a, dstReg, baseReg, expReg, regState, pow_addr);

        // Trace the power operation result
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::POW, 4, -1, baseReg, expReg);
    }
    
    // Implementation of virtual methods required by base class (YMM-specific)
    
    void emitSaveVectorRegisters(asmjit::x86::Assembler& a) const override {
        // Save YMM6-YMM15 (callee-saved on Win64, 32 bytes each)
#ifdef _WIN32
        // Win64: YMM registers start at offset 32 (before GP registers at 192+)
        for (int i = 6; i < 16; i++) {
            a.vmovapd(asmjit::x86::ymmword_ptr(asmjit::x86::rsp, 32 + (i - 6) * 32), asmjit::x86::ymm(i));
        }
#else
        // Linux System V: YMM registers start at offset 40 (after GP registers at 0-39)
        // Note: YMM registers are caller-saved on Linux, but we save them for consistency
        for (int i = 6; i < 16; i++) {
            a.vmovapd(asmjit::x86::ymmword_ptr(asmjit::x86::rsp, 40 + (i - 6) * 32), asmjit::x86::ymm(i));
        }
#endif
    }

    void emitRestoreVectorRegisters(asmjit::x86::Assembler& a) const override {
        // Restore YMM6-YMM15
#ifdef _WIN32
        // Win64: YMM registers start at offset 32
        for (int i = 6; i < 16; i++) {
            a.vmovapd(asmjit::x86::ymm(i), asmjit::x86::ymmword_ptr(asmjit::x86::rsp, 32 + (i - 6) * 32));
        }
#else
        // Linux System V: YMM registers start at offset 40
        for (int i = 6; i < 16; i++) {
            a.vmovapd(asmjit::x86::ymm(i), asmjit::x86::ymmword_ptr(asmjit::x86::rsp, 40 + (i - 6) * 32));
        }
#endif
    }
    
    int getVectorStackSpace() const override {
        // 10 YMM registers * 32 bytes each = 320 bytes
        return 320;
    }
    
    // Modulo operation
    void emitMod(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override;
    
    // Conditional operations
    void emitIf(asmjit::x86::Assembler& a, int dstReg, int condReg, int trueReg, int falseReg, IRegisterAllocator& regState) override;
    
    // Bitwise operations
    void emitAndPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        a.vandpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(srcReg));
        
        // Trace operation with register info
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::AND, 4, -1, dstReg, srcReg);
    }
    
    void emitXorPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        a.vxorpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(srcReg));
        
        // Trace operation with register info
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::XOR, 4, -1, dstReg, srcReg);
    }
    
    void emitOrPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        a.vorpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(srcReg));
        
        // Trace operation with register info
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::OR, 4, -1, dstReg, srcReg);
    }
    
    void emitAndNotPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) override {
        a.vandnpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(srcReg));
        
        // Trace operation with register info
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::ANDNOT, 4, -1, dstReg, srcReg);
    }
    
    // Bit manipulation
    void emitCreateAllOnes(asmjit::x86::Assembler& a, int dstReg) override;
    void emitShiftLeft(asmjit::x86::Assembler& a, int dstReg, int bits) override;
    void emitShiftRight(asmjit::x86::Assembler& a, int dstReg, int bits) override;
    
    // Load immediate
    void emitLoadImmediate(asmjit::x86::Assembler& a, int dstReg, double value) override;
    void emitLoadImmediateRaw(asmjit::x86::Assembler& a, int dstReg, uint64_t bits) override;
    
    // Rounding operations
    void emitRound(asmjit::x86::Assembler& a, int dstReg, int srcReg, int mode) override;
    
    // Integer comparison operations
    void emitIntCmpLT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    void emitIntCmpLE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    void emitIntCmpGT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    void emitIntCmpGE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    void emitIntCmpEQ(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    void emitIntCmpNE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) override;
    
    // Integer conditional
    void emitIntIf(asmjit::x86::Assembler& a, int dstReg, int condReg, int trueReg, int falseReg, IRegisterAllocator& regState) override;

    // Zero register
    void emitZero(asmjit::x86::Assembler& a, int dstReg) override {
        a.vxorpd(getYmmRegister(dstReg), getYmmRegister(dstReg), getYmmRegister(dstReg));
        // Trace the zeroed register
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::ZERO, 4, -1, -1, dstReg);
    }
    
    // Function prologue/epilogue
    void emitPrologue(asmjit::x86::Assembler& a) override;
    void emitEpilogue(asmjit::x86::Assembler& a) override;
    
    // Register management
    void emitSaveCalleeRegisters(asmjit::x86::Assembler& a) override;
    void emitRestoreCalleeRegisters(asmjit::x86::Assembler& a) override;
    int getStackSpaceNeeded() const override;
    
    // Get XMM register from index (for compatibility)
    asmjit::x86::Xmm getRegister(int index) const override {
        return asmjit::x86::xmm(index);
    }
    
    // Get YMM register from index (for AVX2)
    asmjit::x86::Vec getYmmRegister(int index) const {
        // Use Vec<256> for YMM registers (256-bit)
        return asmjit::x86::ymm(index);
    }
    
    // Register setup
    void emitMoveArgsToRegisters(asmjit::x86::Assembler& a) override;
    
    // Optimized memory operations
    void emitOptimizedLoad(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) override;
    void emitOptimizedStore(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) override;
    
    // Gradient operations
    void emitLoadGradient(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) override;
    void emitStoreGradient(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) override;
    void emitAccumulateGradient(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId, int tempReg = 3) override;
    void emitLoadValueForGradient(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId,
                                  const forge::Graph& graph,
                                  const void* constantMap,
                                  const asmjit::Label& constPoolLabel) override;

public:
    
private:
};

} // namespace forge