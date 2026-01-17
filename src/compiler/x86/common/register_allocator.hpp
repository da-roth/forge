// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file register_allocator.hpp
 * @brief Register allocator for XMM registers in the JIT compiler
 *
 * Implements LRU-based register allocation for x86-64 XMM (SSE2) registers.
 * Manages register assignment, tracking, and lifetime during code generation.
 * This is a concrete implementation of the IRegisterAllocator interface.
 *
 * Architecture:
 * - Allocates from 16 XMM registers (XMM0-XMM15)
 * - Prefers XMM0-XMM11 for working registers (hot path)
 * - XMM12-XMM15 used when necessary (cold path)
 * - XMM0-XMM5 are volatile on Win64 ABI (invalidated at call sites)
 * - XMM6-XMM15 are callee-saved (managed by caller)
 *
 * Allocation Strategy:
 * - First: Find empty register (LRU order for allocation availability)
 * - Second: Evict least-recently-used register
 * - Respects locking (pins registers during active instruction generation)
 * - Respects "avoid" list (e.g., for preserved argument registers)
 *
 * Thread Safety: Not thread-safe - each JIT compilation uses its own allocator
 *
 * API Stability: Stable (part of public JIT compiler interface)
 *
 * Note: This class is in transition. XmmRegisterAllocator and YmmRegisterAllocator
 * (in instruction_sets/) provide instruction-set-specific versions. This class
 * maintains backward compatibility during the refactoring.
 */

#pragma once

#include "../../../graph/graph.hpp"  // For NodeId
#include "register_allocator_base.hpp"  // For IRegisterAllocator interface
#include <initializer_list>

namespace forge {

/**
 * @brief Concrete register allocator for x86-64 XMM registers
 *
 * Manages allocation and tracking of SSE2/SSE3/AVX XMM registers (128-bit vector registers)
 * for the JIT compiler. Implements LRU eviction policy with support for register locking
 * and dirty tracking.
 *
 * The allocator divides registers into two tiers:
 * - Working registers (XMM0-XMM11): Preferred allocation targets
 * - Reserved registers (XMM12-XMM15): Fallback when working registers exhausted
 *
 * Volatile registers (XMM0-XMM5) are invalidated at function call sites to comply
 * with the x86-64 Win64 calling convention, which doesn't preserve them across calls.
 *
 * Example Usage:
 * @code
 * RegisterAllocator alloc;
 * int reg = alloc.allocateRegister();        // Allocate any XMM register
 * alloc.setRegister(reg, nodeId, false);     // Track node in register
 * alloc.lock(reg);                           // Pin for active use
 * // ... generate instructions using register ...
 * alloc.markDirty(reg);                      // Mark for writeback if needed
 * alloc.unlock(reg);                         // Unpin when done
 * @endcode
 */
class RegisterAllocator : public IRegisterAllocator {
public:
    /// Total number of XMM registers available (XMM0-XMM15)
    static constexpr int NUM_WORKING_REGS = 16;

    /// @brief Construct and initialize the allocator
    RegisterAllocator();

    /// @brief Destructor
    ~RegisterAllocator() = default;

    // ========== Core allocation interface ==========

    /**
     * @brief Allocate any available XMM register
     * @return Register index (0-15) for XMM0-XMM15
     * @throws std::runtime_error if no registers available
     */
    int allocateRegister() override;

    /**
     * @brief Allocate a register, avoiding specified ones
     * @param avoid List of register indices to skip
     * @return Register index that avoids the list if possible, fallback to general allocation
     *
     * Useful when specific registers must be preserved (e.g., arguments, return values).
     * If all non-avoided registers are locked, falls back to standard allocation.
     */
    int allocateAvoiding(std::initializer_list<int> avoid) override;

    // ========== Register state management ==========

    /// @brief Reset allocator to initial state (all registers empty and unlocked)
    void clear() override;

    /**
     * @brief Lock a register to prevent allocation
     * @param regIndex Register index (0-15)
     *
     * Used to pin a register during instruction generation. Locked registers
     * are not evicted or reallocated. Must be unlocked when instruction generation completes.
     */
    void lock(int regIndex) override;

    /**
     * @brief Unlock a register for reallocation
     * @param regIndex Register index (0-15)
     */
    void unlock(int regIndex) override;

    // ========== Register content tracking ==========

    /**
     * @brief Find which XMM register holds a specific graph node's value
     * @param nodeId Node identifier from computation graph
     * @return Register index (0-15) if found, -1 if not in any register
     *
     * Used to check if a value is already computed and available in a register,
     * avoiding redundant computation (dead code elimination).
     */
    int findNodeInRegister(forge::NodeId nodeId) const override;

    /**
     * @brief Store node value mapping in a register
     * @param regIndex Register index (0-15)
     * @param nodeId Node whose value is now in this register
     * @param isDirty Whether the register contains a modified value needing writeback
     *
     * Marks which computation node's value is currently held in a register.
     * Updates the LRU usage counter for eviction decisions.
     */
    void setRegister(int regIndex, forge::NodeId nodeId, bool isDirty = false) override;

    /**
     * @brief Get the node stored in a specific register
     * @param regIndex Register index (0-15)
     * @return Node ID stored in register, or -1 if empty
     */
    int getNodeInRegister(int regIndex) const override;

    // ========== Dirty register tracking ==========

    /**
     * @brief Mark register as containing modified data
     * @param regIndex Register index (0-15)
     *
     * Indicates that the register value differs from memory and must be written back
     * before the register is evicted. Used for optimization of memory operations.
     */
    void markDirty(int regIndex) override;

    /**
     * @brief Clear the dirty flag for a register
     * @param regIndex Register index (0-15)
     *
     * Call after writing back dirty data to memory.
     */
    void markClean(int regIndex) override;

    /**
     * @brief Check if register contains modified data
     * @param regIndex Register index (0-15)
     * @return true if register is marked dirty, false otherwise
     */
    bool isDirty(int regIndex) const override;

    // ========== Platform-specific invalidation ==========

    /**
     * @brief Invalidate volatile registers at function call sites
     *
     * On x86-64 Win64 ABI, XMM0-XMM5 are volatile (not preserved across calls).
     * This clears those registers to prevent using stale values after a call.
     * Called automatically by code generation when emitting function calls.
     */
    void invalidateVolatileRegisters() override;

    /// @brief First volatile register index on Win64 (XMM0)
    int getFirstVolatileReg() const override { return 0; }

    /// @brief Last volatile register index on Win64 (XMM5)
    int getLastVolatileReg() const override { return 5; }

    /// @brief Total number of managed registers
    int getNumRegisters() const override { return NUM_WORKING_REGS; }

    /// @brief Prevent copying (allocator is unique per compilation)
    RegisterAllocator(const RegisterAllocator&) = delete;

    /// @brief Prevent assignment (allocator is unique per compilation)
    RegisterAllocator& operator=(const RegisterAllocator&) = delete;

private:
    int xmmContents_[NUM_WORKING_REGS];  // nodeId or -1
    bool locked_[NUM_WORKING_REGS];      // Pinned registers during instruction generation
    bool dirty_[NUM_WORKING_REGS];       // Track if register needs to be stored
};

} // namespace forge