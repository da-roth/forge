#pragma once

#include "x86_instruction_set_base.h"
#include "register_allocator.h"
#include "ymm_register_allocator.h"
#include "avx2_transcendental_helpers.h"  // For helper functions
#include "instruction_tracer.h"
#include <forge/core/opcodes.h>        // For runtime tracing
#include <cmath>
#include <climits>
#include <cstring>  // For memcpy
#include <unordered_map>
#include <limits>   // For std::numeric_limits

// Forward declarations for perfect transcendental functions
extern "C" double call_std_exp(double x);
extern "C" double call_std_log(double x);
extern "C" double call_std_sin(double x);
extern "C" double call_std_cos(double x);
extern "C" double call_std_tan(double x);
extern "C" double call_std_pow(double base, double exp);

// Include intrinsics header for AVX2
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

// Vectorized math functions will be added later

namespace forge {
namespace x86 {

// Simple struct to hold constant info - matches AsmStitcher::ConstantInfo
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
    
    bool supportsOperation(forge::core::OpCode op) const override {
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
    
    // Three-operand arithmetic (AVX2 naturally supports 3-operand forms)
    void emitAdd3(asmjit::x86::Assembler& a, int dstReg, int src1Reg, int src2Reg) override {
        // Perform the operation
        a.vaddpd(getYmmRegister(dstReg), getYmmRegister(src1Reg), getYmmRegister(src2Reg));
        
        // Trace output values after operation
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::ADD, 4);
    }
    
    void emitSub3(asmjit::x86::Assembler& a, int dstReg, int src1Reg, int src2Reg) override {
        // Perform the operation
        a.vsubpd(getYmmRegister(dstReg), getYmmRegister(src1Reg), getYmmRegister(src2Reg));
        
        // Trace output values after operation
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::SUB, 4);
    }
    
    void emitMul3(asmjit::x86::Assembler& a, int dstReg, int src1Reg, int src2Reg) override {
        // Trace input values before operation
        tracer.emitTraceYMM(a, getYmmRegister(src1Reg), OperationType::MUL, 4);
        tracer.emitTraceYMM(a, getYmmRegister(src2Reg), OperationType::MUL, 4);
        
        // Perform the operation
        a.vmulpd(getYmmRegister(dstReg), getYmmRegister(src1Reg), getYmmRegister(src2Reg));
        
        // Trace output values after operation
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::MUL, 4);
    }
    
    void emitDiv3(asmjit::x86::Assembler& a, int dstReg, int src1Reg, int src2Reg) override {
        // Trace input values before operation
        tracer.emitTraceYMM(a, getYmmRegister(src1Reg), OperationType::DIV, 4);
        tracer.emitTraceYMM(a, getYmmRegister(src2Reg), OperationType::DIV, 4);
        
        // Perform the operation: dst = src1 / src2
        a.vdivpd(getYmmRegister(dstReg), getYmmRegister(src1Reg), getYmmRegister(src2Reg));
        
        // Trace output values after operation
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::DIV, 4);
    }
    
    // Unary operations
    void emitNeg(asmjit::x86::Assembler& a, int dstReg) override;
    void emitAbs(asmjit::x86::Assembler& a, int dstReg) override;
    void emitSqrt(asmjit::x86::Assembler& a, int dstReg) override {
        // Runtime tracing configured in tracer
        
        // Trace input values before operation
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::SQRT, 4);
        
        // Perform the operation
        a.vsqrtpd(getYmmRegister(dstReg), getYmmRegister(dstReg));
        
        // Trace output values after operation
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::SQRT, 4);
    }
    void emitRecip(asmjit::x86::Assembler& a, int dstReg) override;
    
    // Memory operations
    void emitLoad(asmjit::x86::Assembler& a, int dstReg, uint32_t nodeId) override;
    void emitStore(asmjit::x86::Assembler& a, int srcReg, uint32_t nodeId) override;
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
    
    // Create mask from boolean
    void emitCreateMaskFromBool(asmjit::x86::Assembler& a, int dstReg, int srcReg) override;
    
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
    // Transcendental functions using scalar math but broadcasting to all YMM lanes
    void emitExp(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        auto exp_addr = reinterpret_cast<uint64_t>(&call_std_exp);
        emitScalarMathFunctionCall(a, dstReg, srcReg, regState, exp_addr);
    }

        // Transcendental functions (will need vectorized implementations)
    // Transcendental functions using scalar math but broadcasting to all YMM lanes
    void emitLog(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        // Option 1: Use std::log with FMA3 disabled (current approach)
        auto log_addr = reinterpret_cast<uint64_t>(&call_std_log);
        
        // Option 2: Use alternative log implementation (uncomment if std::log still crashes)
        // auto log_addr = reinterpret_cast<uint64_t>(&call_safe_log);
        
        emitScalarMathFunctionCall(a, dstReg, srcReg, regState, log_addr);
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
        // TapePresso's register allocator tracks register contents, so we must preserve them
        
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
        
        // 10. CRITICAL: Tell TapePresso that volatile registers may be invalid
        // This is the key fix - inform the register allocator about the function call impact
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
            
            // Shadow space
            a.sub(rsp, 32);
            
            // Call
            a.mov(rax, funcAddr);
            a.call(rax);
            
            // Remove shadow space
            a.add(rsp, 32);
            
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
        auto sin_addr = reinterpret_cast<uint64_t>(&call_std_sin);
        emitScalarMathFunctionCall(a, dstReg, srcReg, regState, sin_addr);
    }
    
    void emitCos(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        auto cos_addr = reinterpret_cast<uint64_t>(&call_std_cos);
        emitScalarMathFunctionCall(a, dstReg, srcReg, regState, cos_addr);
    }
    
    void emitTan(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) override {
        auto tan_addr = reinterpret_cast<uint64_t>(&call_std_tan);
        emitScalarMathFunctionCall(a, dstReg, srcReg, regState, tan_addr);
    }
    
    void emitPow(asmjit::x86::Assembler& a, int dstReg, int baseReg, int expReg, IRegisterAllocator& regState) override {
        using namespace asmjit::x86;
        
        // Let's debug with the absolute minimum that works
        auto ymm_base = ymm(baseReg);
        auto ymm_exp = ymm(expReg);
        auto ymm_dst = ymm(dstReg);
        
        // DON'T save any GP registers - see if that's the issue
        // DON'T save any YMM registers - just do the bare minimum
        
        // Allocate space for inputs/outputs only
        // Stack should be 8-byte misaligned on entry (after CALL)
        // We need to make it 16-byte aligned
        a.sub(rsp, 72);  // 64 bytes for data + 8 for alignment
        
        // Store inputs
        a.vmovupd(ymmword_ptr(rsp, 0), ymm_base);
        a.vmovupd(ymmword_ptr(rsp, 32), ymm_exp);
        
        // Process each lane
        for (int lane = 0; lane < 4; lane++) {
            // Load arguments
            a.vmovsd(xmm0, qword_ptr(rsp, lane * 8));
            a.vmovsd(xmm1, qword_ptr(rsp, 32 + lane * 8));
            
            // CRITICAL: Before call, save RAX inline
            a.push(rax);
            a.sub(rsp, 8);  // Align to 16 after push
            
            // Call pow function
            a.sub(rsp, 32);  // Shadow space
            auto pow_addr = reinterpret_cast<uint64_t>(&call_std_pow);
            a.mov(rax, pow_addr);
            a.call(rax);
            a.add(rsp, 32);
            
            // Restore alignment and RAX
            a.add(rsp, 8);
            a.pop(rax);
            
            // Store result
            a.vmovsd(qword_ptr(rsp, lane * 8), xmm0);
        }
        
        // Load results
        a.vmovupd(ymm_dst, ymmword_ptr(rsp, 0));
        
        // Clean up
        a.add(rsp, 72);
        
        // Trace the power operation result
        // TEMPORARILY DISABLED: POW tracing to debug lane corruption
        // tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::POW, 4, -1, baseReg, expReg);
    }
    
    // Implementation of virtual methods required by base class (YMM-specific)
    
    void emitSaveVectorRegisters(asmjit::x86::Assembler& a) const override {
        // Save YMM6-YMM15 (callee-saved on Win64, 32 bytes each)
        for (int i = 6; i < 16; i++) {
            a.vmovapd(asmjit::x86::ymmword_ptr(asmjit::x86::rsp, 32 + (i - 6) * 32), asmjit::x86::ymm(i));
        }
    }
    
    void emitRestoreVectorRegisters(asmjit::x86::Assembler& a) const override {
        // Restore YMM6-YMM15
        for (int i = 6; i < 16; i++) {
            a.vmovapd(asmjit::x86::ymm(i), asmjit::x86::ymmword_ptr(asmjit::x86::rsp, 32 + (i - 6) * 32));
        }
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
    
    // Blending
    void emitBlend(asmjit::x86::Assembler& a, int dstReg, int srcReg, int maskReg) override {
        // Flip selection to validate mask polarity: select from dst when mask=0, from src when mask=1
        // If semantics were inverted, swapping operands will fix select results.
        a.vblendvpd(getYmmRegister(dstReg), getYmmRegister(srcReg), getYmmRegister(dstReg), getYmmRegister(maskReg));
        
        // Trace operation with register info
        tracer.emitTraceYMM(a, getYmmRegister(dstReg), OperationType::BLEND, 4, -1, srcReg, maskReg);
    }
    
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
    void emitOptimizedLoad(asmjit::x86::Assembler& a, int dstReg, uint32_t nodeId) override;
    void emitOptimizedStore(asmjit::x86::Assembler& a, int srcReg, uint32_t nodeId) override;
    
    // Gradient operations
    void emitLoadGradient(asmjit::x86::Assembler& a, int dstReg, uint32_t nodeId) override;
    void emitStoreGradient(asmjit::x86::Assembler& a, int srcReg, uint32_t nodeId) override;
    void emitAccumulateGradient(asmjit::x86::Assembler& a, int srcReg, uint32_t nodeId, int tempReg = 3) override;
    void emitLoadValueForGradient(asmjit::x86::Assembler& a, int dstReg, uint32_t nodeId,
                                  const GraphView& graph,
                                  const void* constantMap,
                                  const asmjit::Label& constPoolLabel) override;

public:
    
private:
};

} // namespace x86
} // namespace forge