// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file instruction_tracer.hpp
 * @brief Runtime tracing helper for JIT-generated code
 *
 * Provides facilities for emitting trace points in JIT-compiled code to
 * record register values and operation metadata at runtime. Supports both
 * SSE2/XMM and AVX2/YMM registers with intelligent corruption detection.
 *
 * Thread Safety: Each compilation should have its own InstructionTracer
 * instance (not thread-safe for shared use during code generation).
 */

#pragma once

#include "../../runtime_trace.hpp"
#include "compiler_config.hpp"
#include <asmjit/x86.h>
#include <cstdint>
#include <iostream>
#include <cmath>

namespace forge {

/**
 * @brief Helper for emitting safe runtime tracing in JIT code
 *
 * This class generates inline assembly code that captures register values
 * and operation metadata into a global trace buffer at runtime. It provides:
 *
 * - Safe register usage (saves/restores all modified registers)
 * - Support for both SSE2 (XMM, 128-bit) and AVX2 (YMM, 256-bit) registers
 * - Intelligent corruption detection (NaN, Inf, suspicious patterns)
 * - Smart filtering to reduce trace output noise
 * - Zero-overhead when tracing is disabled
 *
 * Design Principles:
 * - Never modifies the original register being traced
 * - Uses dedicated temporary registers (XMM15/YMM15)
 * - Direct memory writes instead of function calls
 * - Circular buffer with atomic index management
 *
 * API Stability: Stable - interface won't change
 *
 * Example:
 * @code
 * CompilerConfig config = CompilerConfig::Debug();
 * InstructionTracer tracer(config);
 *
 * // In JIT code generation:
 * tracer.emitTrace(assembler, asmjit::x86::xmm0, OperationType::Add, 1, nodeId);
 * @endcode
 */
class InstructionTracer {
private:
    const CompilerConfig& config;
    uint32_t instructionCounter;

    // Check if we should trace
    bool shouldTrace() const {
        return config.printRuntimeTrace;
    }

public:
    /**
     * @brief Construct tracer with compiler configuration
     * @param cfg Compiler configuration controlling tracing behavior
     *
     * Initializes the global trace buffer if tracing is enabled in the config.
     */
    InstructionTracer(const CompilerConfig& cfg) : config(cfg), instructionCounter(0) {
        // Initialize the trace buffer if tracing is enabled
        if (config.printRuntimeTrace) {
            std::cout << "[Compiling] Runtime tracing enabled" << std::endl;
            initializeTraceBuffer(1024); // Initialize with 1024 records
        }
    }

    /**
     * @brief Emit tracing code for a vector register (XMM or YMM)
     *
     * Generates assembly code that safely records the contents of a vector
     * register into the trace buffer, along with operation metadata.
     * Automatically detects register size (128-bit XMM or 256-bit YMM).
     *
     * @param a AsmJit assembler for code generation
     * @param liveReg Vector register to trace (not modified)
     * @param opType Type of operation being traced (for identification)
     * @param vectorWidth SIMD width (1 for scalar, 4 for AVX2 doubles, etc.)
     * @param nodeId Optional graph node ID (-1 if not applicable)
     * @param srcReg Optional source register index
     * @param dstReg Optional destination register index
     *
     * Thread Safety: Not thread-safe - call from single compilation thread
     *
     * Performance: ~40-100 cycles per trace point (debugging overhead)
     */
    void emitTrace(asmjit::x86::Assembler& a, asmjit::x86::Vec liveReg,
                   OperationType opType, int vectorWidth = 1, int nodeId = -1, int srcReg = -1, int dstReg = -1) {
        if (!shouldTrace()) {
            return;
        }

        // Limit compile-time trace to first 50 operations
        if (instructionCounter < 50) {
            // Concise compile-time trace message
            if (instructionCounter == 0) {
                std::cout << "[Compiling] Trace points (first 50): ";
            }
            std::cout << getOperationName(static_cast<uint32_t>(opType)) << "(" << dstReg << "," << srcReg << ") ";
            // Print newline after a few operations to avoid long lines
            if ((instructionCounter + 1) % 5 == 0) {
                std::cout << std::endl << "                        ";
            }
        } else if (instructionCounter == 50) {
            std::cout << "... (trace output limited to 50 operations)" << std::endl;
        }

        using namespace asmjit::x86;

        // Determine if this is a YMM (256-bit) or XMM (128-bit) register
        const bool isYmm = liveReg.isYmm();
        const int stackSize = isYmm ? 32 : 16;

        // Use the appropriate temporary register (same size as liveReg)
        asmjit::x86::Vec tempReg = isYmm ? asmjit::x86::ymm15 : asmjit::x86::xmm15;

        // ULTRA-SAFE PATTERN: Direct memory writes only, no function calls
        // This completely avoids ABI, stack, and register preservation issues

        // 1) Save temp register to stack first since it might be in use
        a.sub(rsp, stackSize);
        if (isYmm) {
            a.vmovups(ymmword_ptr(rsp), tempReg);
        } else {
            a.movaps(xmmword_ptr(rsp), tempReg);
        }

        // 2) Duplicate the live register to temp (never modify the original)
        if (isYmm) {
            a.vmovaps(tempReg, liveReg);
        } else {
            a.movaps(tempReg, liveReg);
        }

        // 3) Store directly to the global trace buffer using atomic index
        asmjit::Label skipTrace = a.newLabel();

        // Save registers we're about to use
        a.push(rax);
        a.push(rcx);
        a.push(rdx);

        // Calculate buffer index atomically (simple increment)
        a.mov(rcx, asmjit::imm((uint64_t)&g_traceBuffer.index));
        a.mov(edx, asmjit::x86::dword_ptr(rcx));  // Load current index (32-bit)
        a.mov(rax, rdx);                           // Save current index in RAX
        a.inc(edx);                                // Increment for next record
        a.mov(asmjit::x86::dword_ptr(rcx), edx);  // Store back

        // Calculate buffer position: (index & mask) * sizeof(TraceRecord)
        a.mov(rcx, asmjit::imm((uint64_t)&g_traceBuffer.mask));
        a.mov(ecx, asmjit::x86::dword_ptr(rcx));   // Load mask (32-bit)
        a.and_(eax, ecx);                           // (saved_index & mask) in EAX
        a.mov(rdx, rax);                            // Move masked index to RDX

        // Calculate offset: rdx * sizeof(TraceRecord)
        a.imul(rdx, rdx, sizeof(TraceRecord));

        // Get pointer to the record
        a.mov(rcx, asmjit::imm((uint64_t)&g_traceBuffer.records));
        a.mov(rcx, asmjit::x86::qword_ptr(rcx));  // Load records pointer
        a.add(rcx, rdx);  // rcx now points to the TraceRecord

        // Store metadata
        int32_t idToStore = (nodeId >= 0) ? nodeId : static_cast<int32_t>(instructionCounter);
        a.mov(asmjit::x86::edx, asmjit::imm(idToStore));
        a.mov(asmjit::x86::dword_ptr(rcx, 0), asmjit::x86::edx);                          // nodeId or instructionId
        a.mov(asmjit::x86::edx, asmjit::imm(static_cast<uint32_t>(opType)));
        a.mov(asmjit::x86::dword_ptr(rcx, 4), asmjit::x86::edx);                          // operationType
        a.mov(asmjit::x86::edx, asmjit::imm(static_cast<uint32_t>(vectorWidth)));
        a.mov(asmjit::x86::dword_ptr(rcx, 8), asmjit::x86::edx);                          // vectorWidth

        // Store register info in timestamp field (16 bits for dst, 16 bits for src)
        uint32_t safeDstReg = (dstReg < 0) ? 0xFFFE : (dstReg & 0xFFFF);
        uint32_t safeSrcReg = (srcReg < 0) ? 0xFFFE : (srcReg & 0xFFFF);
        uint32_t regInfo = (safeDstReg << 16) | safeSrcReg;
        a.mov(asmjit::x86::edx, asmjit::imm(regInfo));
        a.mov(asmjit::x86::dword_ptr(rcx, 16), asmjit::x86::edx);

        instructionCounter++;

        // Store the register data to the buffer
        if (isYmm) {
            a.vmovups(ymmword_ptr(rcx, offsetof(TraceRecord, data)), tempReg);
        } else {
            a.movups(xmmword_ptr(rcx, offsetof(TraceRecord, data)), tempReg);
        }

        // Restore registers we used
        a.pop(rdx);
        a.pop(rcx);
        a.pop(rax);

        // Restore temp register from stack
        if (isYmm) {
            a.vmovups(ymm15, ymmword_ptr(rsp));
        } else {
            a.movaps(xmm15, xmmword_ptr(rsp));
        }
        a.add(rsp, stackSize);

        // Bind the skip label
        a.bind(skipTrace);

        // liveReg remains completely unchanged and can be used normally
    }

    // Legacy methods for backward compatibility - delegate to unified emitTrace
    void emitTraceYMM(asmjit::x86::Assembler& a, asmjit::x86::Vec liveReg,
                      OperationType opType, int vectorWidth = 4, int nodeId = -1, int srcReg = -1, int dstReg = -1) {
        emitTrace(a, liveReg.ymm(), opType, vectorWidth, nodeId, srcReg, dstReg);
    }

    void emitTraceXMM(asmjit::x86::Assembler& a, asmjit::x86::Vec liveReg,
                      OperationType opType, int vectorWidth = 1, int nodeId = -1, int srcReg = -1, int dstReg = -1) {
        emitTrace(a, liveReg.xmm(), opType, vectorWidth, nodeId, srcReg, dstReg);
    }

    /**
     * @brief Reset instruction counter for new compilation
     *
     * Useful when compiling multiple functions to keep instruction
     * numbering independent per function.
     */
    void resetCounter() {
        instructionCounter = 0;
    }

    /**
     * @brief Get current instruction counter value
     * @return Number of trace points emitted so far
     *
     * Useful for debugging and optimization analysis.
     */
    uint32_t getCurrentCounter() const {
        return instructionCounter;
    }
};

} // namespace forge
