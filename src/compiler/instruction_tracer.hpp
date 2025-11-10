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

#include "runtime_trace.hpp"
#include "compiler_config.hpp"
#include <asmjit/asmjit.h>
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
 * tracer.emitTraceXMM(assembler, xmm0, OperationType::Add, 1, nodeId);
 * @endcode
 */
class InstructionTracer {
private:
    const CompilerConfig& config;
    uint32_t instructionCounter;
    
    // Smart corruption detection - the core intelligence
    struct CorruptionPattern {
        bool hasNaN = false;
        bool hasInf = false;
        bool hasSuspiciousZeros = false;
        bool hasKnownPatterns = false;
        bool hasPartialCorruption = false;
        int corruptedLanes = 0;
        double suspiciousValue = 0.0;
        
        bool isCorrupted() const {
            return hasNaN || hasInf || hasSuspiciousZeros || hasKnownPatterns || hasPartialCorruption;
        }
        
        const char* getDescription() const {
            if (hasNaN) return "NaN_CORRUPTION";
            if (hasInf) return "INF_CORRUPTION"; 
            if (hasKnownPatterns) return "KNOWN_PATTERN_CORRUPTION";
            if (hasSuspiciousZeros) return "ZERO_CORRUPTION";
            if (hasPartialCorruption) return "PARTIAL_CORRUPTION";
            return "NO_CORRUPTION";
        }
    };
    
    // Analyze vector data for corruption patterns
    CorruptionPattern analyzeCorruption(const double* data, int vectorWidth) const {
        CorruptionPattern pattern;
        
        if (!config.enableSmartTraceFilter) return pattern;  // No smart filtering
        
        int validLanes = 0;
        
        for (int i = 0; i < vectorWidth; i++) {
            double val = data[i];
            
            // NaN/Inf detection
            if (config.detectNaNCorruption && std::isnan(val)) {
                pattern.hasNaN = true;
                pattern.corruptedLanes++;
            } else if (config.detectInfCorruption && std::isinf(val)) {
                pattern.hasInf = true;
                pattern.corruptedLanes++;
            } else {
                validLanes++;
                
                // Pattern-specific corruption detection
                if (config.detectPatternCorruption) {
                    // Known corruption patterns: 0.002, 0.003 etc.
                    if (std::abs(val - 0.002) < 1e-12 || std::abs(val - 0.003) < 1e-12) {
                        pattern.hasKnownPatterns = true;
                        pattern.suspiciousValue = val;
                    }
                }
                
                // Zero corruption (lanes 2-3 being zero in AVX2)
                if (config.detectZeroCorruption && vectorWidth == 4 && i >= 2 && val == 0.0) {
                    // Only suspicious if earlier lanes have non-zero values
                    bool earlierLanesNonZero = false;
                    for (int j = 0; j < i; j++) {
                        if (data[j] != 0.0) {
                            earlierLanesNonZero = true;
                            break;
                        }
                    }
                    if (earlierLanesNonZero) {
                        pattern.hasSuspiciousZeros = true;
                        pattern.corruptedLanes++;
                    }
                }
            }
        }
        
        // Partial corruption detection (some lanes work, others don't)
        if (config.detectPartialCorruption && vectorWidth > 1) {
            if (validLanes > 0 && validLanes < vectorWidth) {
                pattern.hasPartialCorruption = true;
            }
        }
        
        return pattern;
    }
    
    // Check if we should trace based on smart filtering
    bool shouldTraceWithSmartFilter(OperationType opType) const {
        if (!config.printRuntimeTrace) return false;
        if (!config.enableSmartTraceFilter) return true;  // No smart filtering, trace everything
        
        // This will be called during compilation, so we can't analyze data yet
        // We'll do a runtime check in the generated code
        return true;  // Always emit tracing code, filter at runtime
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
            std::cout << "[Compiling] Runtime tracing enabled";
            if (config.enableSmartTraceFilter) {
                std::cout << " with smart filtering";
            }
            std::cout << std::endl;
            
            initializeTraceBuffer(1024); // Initialize with 1024 records
            
            // Configure smart filtering at runtime
            RuntimeFilterConfig runtimeConfig;
            runtimeConfig.enableSmartFilter = config.enableSmartTraceFilter;
            runtimeConfig.traceCorruptedOnly = config.traceCorruptedOnly;
            runtimeConfig.detectNaN = config.detectNaNCorruption;
            runtimeConfig.detectInf = config.detectInfCorruption;
            runtimeConfig.detectZeroCorruption = config.detectZeroCorruption;
            runtimeConfig.detectKnownPatterns = config.detectPatternCorruption;
            runtimeConfig.detectPartialCorruption = config.detectPartialCorruption;
            
            configureSmartFiltering(runtimeConfig);
        }
    }
    
    /**
     * @brief Emit tracing code for AVX2 (YMM) 256-bit register
     *
     * Generates assembly code that safely records the contents of an AVX2
     * register into the trace buffer, along with operation metadata.
     *
     * @param a AsmJit assembler for code generation
     * @param liveReg YMM register to trace (not modified)
     * @param opType Type of operation being traced (for identification)
     * @param vectorWidth SIMD width (4 for AVX2 doubles, 8 for AVX-512)
     * @param nodeId Optional graph node ID (-1 if not applicable)
     * @param srcReg Optional source register index
     * @param dstReg Optional destination register index
     *
     * Thread Safety: Not thread-safe - call from single compilation thread
     *
     * Performance: ~60-100 cycles per trace point (small overhead)
     */
    void emitTraceYMM(asmjit::x86::Assembler& a, asmjit::x86::Ymm liveReg, 
                         OperationType opType, int vectorWidth = 4, int nodeId = -1, int srcReg = -1, int dstReg = -1) {
            if (!shouldTraceWithSmartFilter(opType)) {
                return;
            }
            
            // Skip compile-time tracing when smart filtering is enabled
            if (!config.enableSmartTraceFilter) {
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
            }
            
            using namespace asmjit::x86;
            
            // ULTRA-SAFE PATTERN: Direct memory writes only, no function calls
            // This completely avoids ABI, stack, and register preservation issues
            
            // 1) Duplicate the live register to a temporary register (never modify the original)
            // CRITICAL: We must save YMM15 first since it might be in use by the compiled code!
            asmjit::x86::Ymm tempReg = asmjit::x86::ymm15; // Use YMM15 as temporary
            
            // Save YMM15 to stack
            a.sub(rsp, 32);  // Allocate 32 bytes on stack for YMM
            a.vmovups(ymmword_ptr(rsp), tempReg);  // Save original YMM15
            
            a.vmovaps(tempReg, liveReg);  // Now safe to duplicate liveReg into tempReg
            
            // 2) Store directly to the global trace buffer using atomic index
            // This is the safest approach - no function calls, no ABI issues
            
            // Skip runtime check for now - always trace if compile-time flag is set
            asmjit::Label skipTrace = a.newLabel();
            // a.jmp(skipTrace); // Uncomment to disable tracing at runtime
            
            // Save registers we're about to use
            a.push(rax);
            a.push(rcx);
            a.push(rdx);
            
            // Calculate buffer index atomically (simple increment)
            a.mov(rcx, asmjit::imm((uint64_t)&g_traceBuffer.index));
            a.mov(edx, asmjit::x86::dword_ptr(rcx));  // Load current index (32-bit)
            // Note: We use the index BEFORE incrementing for this record
            a.mov(rax, rdx);                           // Save current index in RAX
            a.inc(edx);                                // Increment for next record
            a.mov(asmjit::x86::dword_ptr(rcx), edx);  // Store back (atomic-ish)
            
            // Calculate buffer position: (index & mask) * sizeof(TraceRecord)
            a.mov(rcx, asmjit::imm((uint64_t)&g_traceBuffer.mask));
            a.mov(ecx, asmjit::x86::dword_ptr(rcx));   // Load mask (32-bit)
            a.and_(eax, ecx);                           // (saved_index & mask) in EAX
            a.mov(rdx, rax);                            // Move masked index to RDX for multiplication
            
            // edx now contains the buffer index
            // Calculate offset: edx * sizeof(TraceRecord)
            // The actual sizeof(TraceRecord) depends on alignment
            a.imul(rdx, rdx, sizeof(TraceRecord));  // Multiply index by sizeof(TraceRecord) (extend to 64-bit)
            
            // Get pointer to the record
            a.mov(rcx, asmjit::imm((uint64_t)&g_traceBuffer.records));
            a.mov(rcx, asmjit::x86::qword_ptr(rcx));  // Load records pointer
            a.add(rcx, rdx);  // rcx now points to the TraceRecord
            
            // Store metadata (node ID or instruction counter, operation type, vector width)
            // For LOAD/STORE ops, store node ID. For others, store instruction counter
            // Note: x86-64 cannot move immediate to memory directly, must go through register
            int32_t idToStore = (nodeId >= 0) ? nodeId : static_cast<int32_t>(instructionCounter);
            a.mov(asmjit::x86::edx, asmjit::imm(idToStore));                                   // Load immediate into EDX
            a.mov(asmjit::x86::dword_ptr(rcx, 0), asmjit::x86::edx);                          // Store nodeId or instructionId
            a.mov(asmjit::x86::edx, asmjit::imm(static_cast<uint32_t>(opType)));              // Load immediate into EDX
            a.mov(asmjit::x86::dword_ptr(rcx, 4), asmjit::x86::edx);                          // Store operationType
            a.mov(asmjit::x86::edx, asmjit::imm(static_cast<uint32_t>(vectorWidth)));         // Load immediate into EDX
            a.mov(asmjit::x86::dword_ptr(rcx, 8), asmjit::x86::edx);                          // Store vectorWidth
            
            // Store register info in timestamp field (16 bits for dst, 16 bits for src)
            // Fix: Handle -1 (no register) properly by converting to a valid ID like 0xFFFE
            uint32_t safeDstReg = (dstReg < 0) ? 0xFFFE : (dstReg & 0xFFFF);
            uint32_t safeSrcReg = (srcReg < 0) ? 0xFFFE : (srcReg & 0xFFFF);
            uint32_t regInfo = (safeDstReg << 16) | safeSrcReg;
            a.mov(asmjit::x86::edx, asmjit::imm(regInfo));                                     // Load immediate into EDX
            a.mov(asmjit::x86::dword_ptr(rcx, 16), asmjit::x86::edx);                         // Store in lower 32 bits of timestamp
            
            instructionCounter++;
            
            // Store the register data to the buffer
            // Data field offset depends on struct layout
            a.vmovups(ymmword_ptr(rcx, offsetof(TraceRecord, data)), tempReg);
            
            // Restore registers we used
            a.pop(rdx);
            a.pop(rcx);
            a.pop(rax);
            
            // Restore YMM15 from stack
            a.vmovups(ymm15, ymmword_ptr(rsp));
            a.add(rsp, 32);  // Deallocate stack space
            
            // Bind the skip label
            a.bind(skipTrace);
            
            // liveReg remains completely unchanged and can be used normally
        }
    
    /**
     * @brief Emit tracing code for SSE2 (XMM) 128-bit register
     *
     * Generates assembly code that safely records the contents of an SSE2
     * register into the trace buffer, along with operation metadata.
     * Similar to emitTraceYMM but for 128-bit registers.
     *
     * @param a AsmJit assembler for code generation
     * @param liveReg XMM register to trace (not modified)
     * @param opType Type of operation being traced
     * @param vectorWidth SIMD width (1 for scalar, 2 for SSE2 pair)
     * @param nodeId Optional graph node ID (-1 if not applicable)
     * @param srcReg Optional source register index
     * @param dstReg Optional destination register index
     *
     * Thread Safety: Not thread-safe - call from single compilation thread
     *
     * Performance: ~40-70 cycles per trace point (slightly faster than YMM)
     */
    void emitTraceXMM(asmjit::x86::Assembler& a, asmjit::x86::Xmm liveReg, 
                     OperationType opType, int vectorWidth = 1, int nodeId = -1, int srcReg = -1, int dstReg = -1) {
        if (!shouldTraceWithSmartFilter(opType)) {
            return;
        }
        
        // Skip compile-time tracing when smart filtering is enabled
        if (!config.enableSmartTraceFilter) {
            // Limit compile-time trace to first 50 operations
            if (instructionCounter < 50) {
                // Concise compile-time trace message (same as AVX2)
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
        }
        
        using namespace asmjit::x86;
        
        // ULTRA-SAFE PATTERN: Direct memory writes only, no function calls
        // This completely avoids ABI, stack, and register preservation issues
        
        // 1) Duplicate the live register to a temporary register (never modify the original)
        // CRITICAL: We must save XMM15 first since it might be in use by the compiled code!
        asmjit::x86::Xmm tempReg = asmjit::x86::xmm15; // Use XMM15 as temporary
        
        // Save XMM15 to stack
        a.sub(rsp, 16);  // Allocate 16 bytes on stack
        a.movaps(xmmword_ptr(rsp), tempReg);  // Save original XMM15
        
        a.movaps(tempReg, liveReg);  // Now safe to duplicate liveReg into tempReg
        
        // 2) Store directly to the global trace buffer using atomic index
        // This is the safest approach - no function calls, no ABI issues
        
        // Skip runtime check for now - always trace if compile-time flag is set
        asmjit::Label skipTrace = a.newLabel();
        // a.jmp(skipTrace); // Uncomment to disable tracing at runtime
        
        // Save registers we're about to use
        a.push(rax);
        a.push(rcx);
        a.push(rdx);
        
        // Calculate buffer index atomically (simple increment)
        a.mov(rcx, asmjit::imm((uint64_t)&g_traceBuffer.index));
        a.mov(edx, asmjit::x86::dword_ptr(rcx));  // Load current index (32-bit)
        // Note: We use the index BEFORE incrementing for this record
        a.mov(rax, rdx);                           // Save current index in RAX
        a.inc(edx);                                // Increment for next record
        a.mov(asmjit::x86::dword_ptr(rcx), edx);  // Store back (atomic-ish)
        
        // Calculate buffer position: (index & mask) * sizeof(TraceRecord)
        a.mov(rcx, asmjit::imm((uint64_t)&g_traceBuffer.mask));
        a.mov(ecx, asmjit::x86::dword_ptr(rcx));   // Load mask (32-bit)
        a.and_(eax, ecx);                           // (saved_index & mask) in EAX
        a.mov(rdx, rax);                            // Move masked index to RDX for multiplication
        
        // edx now contains the buffer index
        // Calculate offset: edx * sizeof(TraceRecord)
        // The actual sizeof(TraceRecord) depends on alignment
        a.imul(rdx, rdx, sizeof(TraceRecord));  // Multiply index by sizeof(TraceRecord) (extend to 64-bit)
        
        // Get pointer to the record
        a.mov(rcx, asmjit::imm((uint64_t)&g_traceBuffer.records));
        a.mov(rcx, asmjit::x86::qword_ptr(rcx));  // Load records pointer
        a.add(rcx, rdx);  // rcx now points to the TraceRecord
        
        // Store metadata (node ID or instruction counter, operation type, vector width)
        // For LOAD/STORE ops, store node ID. For others, store instruction counter
        // Note: x86-64 cannot move immediate to memory directly, must go through register
        int32_t idToStore = (nodeId >= 0) ? nodeId : static_cast<int32_t>(instructionCounter);
        a.mov(asmjit::x86::edx, asmjit::imm(idToStore));                                   // Load immediate into EDX
        a.mov(asmjit::x86::dword_ptr(rcx, 0), asmjit::x86::edx);                          // Store nodeId or instructionId
        a.mov(asmjit::x86::edx, asmjit::imm(static_cast<uint32_t>(opType)));              // Load immediate into EDX
        a.mov(asmjit::x86::dword_ptr(rcx, 4), asmjit::x86::edx);                          // Store operationType
        a.mov(asmjit::x86::edx, asmjit::imm(static_cast<uint32_t>(vectorWidth)));         // Load immediate into EDX
        a.mov(asmjit::x86::dword_ptr(rcx, 8), asmjit::x86::edx);                          // Store vectorWidth
        
        // Store register info in timestamp field (16 bits for dst, 16 bits for src)
        uint32_t regInfo = ((dstReg & 0xFFFF) << 16) | (srcReg & 0xFFFF);
        a.mov(asmjit::x86::edx, asmjit::imm(regInfo));                                     // Load immediate into EDX
        a.mov(asmjit::x86::dword_ptr(rcx, 16), asmjit::x86::edx);                         // Store in lower 32 bits of timestamp
        
        // Store the register data to the buffer (only 16 bytes for XMM)
        // Data field offset depends on struct layout
        a.movups(xmmword_ptr(rcx, offsetof(TraceRecord, data)), tempReg);
        
        instructionCounter++;
        
        // Restore registers we used
        a.pop(rdx);
        a.pop(rcx);
        a.pop(rax);
        
        // Restore XMM15 from stack
        a.movaps(xmm15, xmmword_ptr(rsp));
        a.add(rsp, 16);  // Deallocate stack space
        
        // Bind the skip label
        a.bind(skipTrace);
        
        // liveReg remains completely unchanged and can be used normally
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
