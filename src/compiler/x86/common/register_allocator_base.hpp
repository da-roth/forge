// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file register_allocator_base.hpp
 * @brief Template base implementation for register allocators
 *
 * Provides a shared template implementation that works across different
 * register types (SSE2/XMM, AVX2/YMM, AVX-512/ZMM, etc.).
 *
 * Thread Safety: Not thread-safe - each compilation uses its own allocator
 */

#pragma once

#include "../../interfaces/register_allocator.hpp"
#include <climits>
#include <stdexcept>

namespace forge {

/**
 * @brief Template implementation of register allocator
 *
 * Provides LRU-based allocation for any register type. Subclasses only need
 * to implement `getRegister()` to specify which actual AsmJit register
 * type (Xmm, Ymm, Zmm) they use.
 *
 * Implementation Details:
 * - O(n) allocation, where n is number of registers (typically 16)
 * - All operations are O(1) or O(n) with small n, suitable for compilation
 * - Thread safety must be ensured at higher level (ForgeEngine)
 *
 * @tparam RegType AsmJit register type (Vec for XMM/YMM/ZMM)
 * @tparam NUM_REGS Total number of registers of this type available (typically 16)
 *
 * Example:
 * @code
 * class MyAllocator : public RegisterAllocatorBase<asmjit::x86::Vec, 16> {
 *     asmjit::x86::Vec getRegister(int index) const override {
 *         return asmjit::x86::xmm(index);  // or ymm(index), zmm(index)
 *     }
 * };
 * @endcode
 */
template<typename RegType, int NUM_REGS>
class RegisterAllocatorBase : public IRegisterAllocator {
protected:
    int contents_[NUM_REGS];      // NodeId or -1 if empty
    bool locked_[NUM_REGS];        // Pinned registers during instruction generation
    bool dirty_[NUM_REGS];         // Track if register needs to be stored
    int usageCounter_[NUM_REGS];  // For LRU tracking
    int currentCounter_ = 0;       // Global usage counter
    bool blacklisted_[NUM_REGS];   // Registers to never allocate (corruption workaround)

public:
    RegisterAllocatorBase() {
        clear();
    }
    
    virtual ~RegisterAllocatorBase() = default;
    
    // ========== Core allocation interface ==========
    
    int allocateRegister() override {
        // First try to find an empty register (skip blacklisted)
        for (int i = 0; i < NUM_REGS; i++) {
            if (contents_[i] == -1 && !locked_[i] && !blacklisted_[i]) {
                usageCounter_[i] = ++currentCounter_;
                return i;
            }
        }
        
        // Find LRU unlocked register (skip blacklisted)
        int lruReg = -1;
        int lruCount = INT_MAX;
        for (int i = 0; i < NUM_REGS; i++) {
            if (!locked_[i] && !blacklisted_[i] && usageCounter_[i] < lruCount) {
                lruCount = usageCounter_[i];
                lruReg = i;
            }
        }
        
        if (lruReg >= 0) {
            contents_[lruReg] = -1;
            dirty_[lruReg] = false;
            usageCounter_[lruReg] = ++currentCounter_;
            return lruReg;
        }
        
        throw std::runtime_error("No allocatable registers available");
    }
    
    int allocateAvoiding(std::initializer_list<int> avoid) override {
        // First try to find an empty register not in avoid list (skip blacklisted)
        for (int i = 0; i < NUM_REGS; i++) {
            bool shouldAvoid = false;
            for (int avoidReg : avoid) {
                if (i == avoidReg) {
                    shouldAvoid = true;
                    break;
                }
            }
            
            if (!shouldAvoid && contents_[i] == -1 && !locked_[i] && !blacklisted_[i]) {
                usageCounter_[i] = ++currentCounter_;
                return i;
            }
        }
        
        // Find LRU unlocked register not in avoid list (skip blacklisted)
        int lruReg = -1;
        int lruCount = INT_MAX;
        for (int i = 0; i < NUM_REGS; i++) {
            bool shouldAvoid = false;
            for (int avoidReg : avoid) {
                if (i == avoidReg) {
                    shouldAvoid = true;
                    break;
                }
            }
            
            if (!shouldAvoid && !locked_[i] && !blacklisted_[i] && usageCounter_[i] < lruCount) {
                lruCount = usageCounter_[i];
                lruReg = i;
            }
        }
        
        if (lruReg >= 0) {
            contents_[lruReg] = -1;
            dirty_[lruReg] = false;
            usageCounter_[lruReg] = ++currentCounter_;
            return lruReg;
        }
        
        // If we can't avoid, just allocate normally
        return allocateRegister();
    }
    
    // ========== Register state management ==========
    
    void clear() override {
        for (int i = 0; i < NUM_REGS; i++) {
            contents_[i] = -1;
            locked_[i] = false;
            dirty_[i] = false;
            usageCounter_[i] = 0;
            blacklisted_[i] = false;  // Initialize blacklist
        }
        currentCounter_ = 0;
    }
    
    void lock(int regIndex) override {
        if (regIndex >= 0 && regIndex < NUM_REGS) {
            locked_[regIndex] = true;
        }
    }
    
    void unlock(int regIndex) override {
        if (regIndex >= 0 && regIndex < NUM_REGS) {
            locked_[regIndex] = false;
        }
    }
    
    // ========== Register content tracking ==========
    
    int findNodeInRegister(forge::NodeId nodeId) const override {
        for (int i = 0; i < NUM_REGS; i++) {
            if (contents_[i] == static_cast<int>(nodeId)) {
                return i;
            }
        }
        return -1;
    }
    
    void setRegister(int regIndex, forge::NodeId nodeId, bool isDirty) override {
        if (regIndex >= 0 && regIndex < NUM_REGS) {
            contents_[regIndex] = static_cast<int>(nodeId);
            dirty_[regIndex] = isDirty;
            usageCounter_[regIndex] = ++currentCounter_;
        }
    }
    
    int getNodeInRegister(int regIndex) const override {
        if (regIndex >= 0 && regIndex < NUM_REGS) {
            return contents_[regIndex];
        }
        return -1;
    }
    
    // ========== Dirty register tracking ==========
    
    void markDirty(int regIndex) override {
        if (regIndex >= 0 && regIndex < NUM_REGS) {
            dirty_[regIndex] = true;
        }
    }
    
    void markClean(int regIndex) override {
        if (regIndex >= 0 && regIndex < NUM_REGS) {
            dirty_[regIndex] = false;
        }
    }
    
    bool isDirty(int regIndex) const override {
        if (regIndex >= 0 && regIndex < NUM_REGS) {
            return dirty_[regIndex];
        }
        return false;
    }
    
    // ========== Platform-specific invalidation ==========
    
    void invalidateVolatileRegisters() override {
        // Invalidate volatile registers (platform-specific)
        // On Win64: registers 0-5 are volatile for both XMM and YMM
        int firstVolatile = getFirstVolatileReg();
        int lastVolatile = getLastVolatileReg();
        
        for (int i = firstVolatile; i <= lastVolatile && i < NUM_REGS; i++) {
            contents_[i] = -1;
            dirty_[i] = false;
            usageCounter_[i] = 0;
        }
    }
    
    // These need to be implemented by derived classes
    virtual RegType getRegister(int index) const = 0;
    
    // Default implementation for Win64 ABI
    int getFirstVolatileReg() const override { return 0; }
    int getLastVolatileReg() const override { return 5; }
    
    // Get the number of registers
    int getNumRegisters() const override { return NUM_REGS; }
    
    // Blacklist management (for corruption workarounds)
    void setBlacklisted(int regIndex, bool blacklisted = true) {
        if (regIndex >= 0 && regIndex < NUM_REGS) {
            blacklisted_[regIndex] = blacklisted;
            if (blacklisted) {
                // If we're blacklisting an allocated register, clear it
                contents_[regIndex] = -1;
                dirty_[regIndex] = false;
                usageCounter_[regIndex] = 0;
            }
        }
    }

protected:
    // Helper to get register count
    static constexpr int getRegisterCount() { return NUM_REGS; }
};

} // namespace forge