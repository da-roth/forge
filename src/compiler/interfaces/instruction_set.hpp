// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file instruction_set.hpp
 * @brief Abstract interface for SIMD instruction set implementations
 *
 * Defines the IInstructionSet interface that all instruction set backends
 * (SSE2, AVX2, AVX-512, etc.) must implement. This abstraction allows adding
 * new SIMD backends without modifying existing code.
 *
 * Thread Safety: Implementations should be safe to use concurrently for
 * code generation (no mutable state).
 */

#pragma once

#include <asmjit/x86.h>
#include "../../graph/graph.hpp"
#include "register_allocator.hpp"  // IRegisterAllocator interface
#include <memory>
#include <string>

namespace forge {

/**
 * @brief API version for the IInstructionSet interface
 *
 * Increment this when making breaking changes to the interface.
 * Custom implementations built against a different version may be incompatible.
 */
constexpr uint32_t INSTRUCTION_SET_API_VERSION = 1;

// Forward declarations
class ForgeEngine;

/**
 * @brief Abstract interface for SIMD instruction set backends
 *
 * This interface defines all operations that an instruction set implementation
 * must provide for JIT code generation. It abstracts away the differences
 * between SSE2 (scalar), AVX2 (4-wide vectors), AVX-512 (8-wide vectors), etc.
 *
 * To add a new instruction set:
 * 1. Create a class inheriting from IInstructionSet
 * 2. Implement all pure virtual methods
 * 3. Add enum value to CompilerConfig::InstructionSet
 * 4. Add factory case in InstructionSetFactory::create()
 *
 * API Stability: Stable - new methods may be added but existing ones won't change
 *
 * Example Implementation (sketch):
 * @code
 * class AVX512InstructionSet : public IInstructionSet {
 *     std::string getName() const override { return "AVX512-Packed"; }
 *     int getVectorWidth() const override { return 8; }  // 8 doubles per ZMM register
 *     // ... implement all other methods ...
 * };
 * @endcode
 */
class IInstructionSet {
public:
    virtual ~IInstructionSet() = default;

    /**
     * @brief Get the API version this implementation was built against
     *
     * Used for version compatibility checking when loading custom implementations.
     * Override only if you need custom version reporting.
     *
     * @return API version number
     */
    virtual uint32_t apiVersion() const { return INSTRUCTION_SET_API_VERSION; }

    /** @brief Get instruction set name (e.g., "SSE2-Scalar", "AVX2-Packed") */
    virtual std::string getName() const = 0;

    /** @brief Get maximum number of registers available for this instruction set */
    virtual int getMaxRegisterCount() const = 0;

    /** @brief Get SIMD vector width (number of doubles per operation: 1 for scalar, 4 for AVX2, 8 for AVX-512) */
    virtual int getVectorWidth() const = 0;

    /**
     * @brief Check if this instruction set supports a given operation
     * @param op Operation code to check
     * @return true if operation is supported
     */
    virtual bool supportsOperation(forge::OpCode op) const = 0;

    ///@{ @name Two-operand arithmetic (dst = dst op src)
    virtual void emitAdd(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitSub(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitMul(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitDiv(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    ///@}

    ///@{ @name Unary operations (in-place modification)
    virtual void emitNeg(asmjit::x86::Assembler& a, int dstReg, int tempReg) = 0;
    virtual void emitAbs(asmjit::x86::Assembler& a, int dstReg, int tempReg) = 0;
    virtual void emitSqrt(asmjit::x86::Assembler& a, int dstReg) = 0;
    virtual void emitSquare(asmjit::x86::Assembler& a, int dstReg) = 0;  ///< Square: x*x
    ///@}

    ///@{ @name Memory operations (loads/stores from value buffer)
    virtual void emitLoad(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) = 0;
    virtual void emitStore(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) = 0;
    virtual void emitLoadFromConstantPool(asmjit::x86::Assembler& a, int dstReg,
                                          const asmjit::Label& poolLabel, size_t offset) = 0;
    ///@}

    /** @brief Move data between registers */
    virtual void emitMove(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;

    ///@{ @name Comparison operations (result is 1.0 for true, 0.0 for false)
    virtual void emitCmpLT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitCmpLE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitCmpGT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitCmpGE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitCmpEQ(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitCmpNE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    ///@}

    ///@{ @name Min/Max operations
    virtual void emitMin(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitMax(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    ///@}

    ///@{ @name Transcendental functions (may use external libraries like SLEEF)
    virtual void emitExp(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;
    virtual void emitLog(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;
    virtual void emitPow(asmjit::x86::Assembler& a, int dstReg, int baseReg, int expReg, IRegisterAllocator& regState) = 0;
    virtual void emitSin(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;
    virtual void emitCos(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;
    virtual void emitTan(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;
    ///@}

    /** @brief Modulo operation (fmod) */
    virtual void emitMod(asmjit::x86::Assembler& a, int dstReg, int srcReg, IRegisterAllocator& regState) = 0;

    /** @brief Conditional select: dst = cond ? trueVal : falseVal */
    virtual void emitIf(asmjit::x86::Assembler& a, int dstReg, int condReg, int trueReg, int falseReg, IRegisterAllocator& regState) = 0;

    ///@{ @name Bitwise operations (for gradient masking and conditional logic)
    virtual void emitAndPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitXorPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitOrPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;
    virtual void emitAndNotPD(asmjit::x86::Assembler& a, int dstReg, int srcReg) = 0;  ///< dst = ~dst & src
    ///@}

    ///@{ @name Bit manipulation for creating masks
    virtual void emitCreateAllOnes(asmjit::x86::Assembler& a, int dstReg) = 0;
    virtual void emitShiftLeft(asmjit::x86::Assembler& a, int dstReg, int bits) = 0;
    virtual void emitShiftRight(asmjit::x86::Assembler& a, int dstReg, int bits) = 0;
    ///@}

    ///@{ @name Load immediate constants
    virtual void emitLoadImmediate(asmjit::x86::Assembler& a, int dstReg, double value) = 0;
    virtual void emitLoadImmediateRaw(asmjit::x86::Assembler& a, int dstReg, uint64_t bits) = 0;
    ///@}

    /** @brief Rounding operation (mode: 0=nearest, 1=down, 2=up, 3=truncate) */
    virtual void emitRound(asmjit::x86::Assembler& a, int dstReg, int srcReg, int mode) = 0;

    ///@{ @name Integer comparisons (truncate to int first, then compare)
    virtual void emitIntCmpLT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitIntCmpLE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitIntCmpGT(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitIntCmpGE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitIntCmpEQ(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    virtual void emitIntCmpNE(asmjit::x86::Assembler& a, int dstReg, int lhsReg, int rhsReg, IRegisterAllocator& regState) = 0;
    ///@}

    /** @brief Integer conditional: dst = (int)cond ? (int)trueVal : (int)falseVal */
    virtual void emitIntIf(asmjit::x86::Assembler& a, int dstReg, int condReg, int trueReg, int falseReg, IRegisterAllocator& regState) = 0;

    /** @brief Set register to zero */
    virtual void emitZero(asmjit::x86::Assembler& a, int dstReg) = 0;

    ///@{ @name Function prologue/epilogue
    virtual void emitPrologue(asmjit::x86::Assembler& a) = 0;
    virtual void emitEpilogue(asmjit::x86::Assembler& a) = 0;
    ///@}

    ///@{ @name Register management for calling conventions
    virtual void emitSaveCalleeRegisters(asmjit::x86::Assembler& a) = 0;
    virtual void emitRestoreCalleeRegisters(asmjit::x86::Assembler& a) = 0;
    virtual int getStackSpaceNeeded() const = 0;
    virtual asmjit::x86::Vec getRegister(int index) const = 0;
    virtual void emitMoveArgsToRegisters(asmjit::x86::Assembler& a) = 0;
    ///@}

    ///@{ @name Optimized memory operations
    virtual void emitOptimizedLoad(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) = 0;
    virtual void emitOptimizedStore(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) = 0;
    ///@}

    ///@{ @name Gradient-specific operations (for automatic differentiation)
    virtual void emitLoadGradient(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId) = 0;
    virtual void emitStoreGradient(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId) = 0;
    virtual void emitAccumulateGradient(asmjit::x86::Assembler& a, int srcReg, forge::NodeId nodeId, int tempReg = 3) = 0;
    virtual void emitLoadValueForGradient(asmjit::x86::Assembler& a, int dstReg, forge::NodeId nodeId,
                                          const forge::Graph& graph,
                                          const void* constantMap,
                                          const asmjit::Label& constPoolLabel) = 0;
    ///@}
};

} // namespace forge