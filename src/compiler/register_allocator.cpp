// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file register_allocator.cpp
 * @brief Register allocator implementation for XMM registers
 *
 * Implements LRU-based register allocation for x86-64 SSE2 registers in the JIT compiler.
 * This is the concrete implementation of the IRegisterAllocator interface for XMM registers.
 *
 * Implementation Details:
 * - Maintains per-register tracking: contents (node IDs), locks, and dirty flags
 * - LRU eviction: allocateAvoiding() prioritizes XMM0-XMM11 (working registers) before
 *   falling back to XMM12-XMM15 (reserved registers)
 * - Respects avoid list: useful for preserving specific registers during multi-instruction sequences
 * - Win64 ABI compliant: XMM0-XMM5 invalidated at call sites
 *
 * Thread Safety: Not thread-safe - designed for single-threaded JIT compilation
 */

#include "register_allocator.hpp"

namespace forge {

RegisterAllocator::RegisterAllocator() {
    clear();
}

void RegisterAllocator::clear() {
    for (int i = 0; i < NUM_WORKING_REGS; ++i) {
        xmmContents_[i] = -1;
        locked_[i] = false;
        dirty_[i] = false;
    }
}

void RegisterAllocator::lock(int regIndex) {
    if (regIndex >= 0 && regIndex < NUM_WORKING_REGS) {
        locked_[regIndex] = true;
    }
}

void RegisterAllocator::unlock(int regIndex) {
    if (regIndex >= 0 && regIndex < NUM_WORKING_REGS) {
        locked_[regIndex] = false;
    }
}

int RegisterAllocator::findNodeInRegister(forge::NodeId nodeId) const {
    for (int i = 0; i < NUM_WORKING_REGS; ++i) {
        if (xmmContents_[i] == static_cast<int>(nodeId)) {
            return i;
        }
    }
    return -1;
}

void RegisterAllocator::setRegister(int regIndex, forge::NodeId nodeId, bool isDirty) {
    if (regIndex >= 0 && regIndex < NUM_WORKING_REGS) {
        xmmContents_[regIndex] = static_cast<int>(nodeId);
        dirty_[regIndex] = isDirty;
    }
}

int RegisterAllocator::getNodeInRegister(int regIndex) const {
    if (regIndex >= 0 && regIndex < NUM_WORKING_REGS) {
        return xmmContents_[regIndex];
    }
    return -1;
}

void RegisterAllocator::markDirty(int regIndex) {
    if (regIndex >= 0 && regIndex < NUM_WORKING_REGS) {
        dirty_[regIndex] = true;
    }
}

void RegisterAllocator::markClean(int regIndex) {
    if (regIndex >= 0 && regIndex < NUM_WORKING_REGS) {
        dirty_[regIndex] = false;
    }
}

bool RegisterAllocator::isDirty(int regIndex) const {
    return regIndex >= 0 && regIndex < NUM_WORKING_REGS && dirty_[regIndex];
}

int RegisterAllocator::allocateAvoiding(std::initializer_list<int> avoid) {
    auto disallow = [&](int i) {
        if (i < 0 || i >= NUM_WORKING_REGS) return false;
        if (locked_[i]) return true;
        for (int a : avoid) {
            if (a == i) return true;
        }
        return false;
    };
    
    // 1) Try to find a free register that's not disallowed
    for (int i = 0; i < NUM_WORKING_REGS; ++i) {
        if (xmmContents_[i] == -1 && !disallow(i)) {
            return i;
        }
    }
    
    // 2) Evict a victim that's not disallowed
    // Prefer to evict from XMM0-XMM11 first (working registers)
    for (int i = 0; i < 12; ++i) {  // Only evict from XMM0-XMM11
        if (!disallow(i)) {
            xmmContents_[i] = -1;
            dirty_[i] = false;
            return i;
        }
    }
    
    // If absolutely necessary, evict from XMM12-XMM15
    for (int i = 12; i < NUM_WORKING_REGS; ++i) {
        if (!disallow(i)) {
            xmmContents_[i] = -1;
            dirty_[i] = false;
            return i;
        }
    }
    return 0;
}

int RegisterAllocator::allocateRegister() {
    return allocateAvoiding({});
}

void RegisterAllocator::invalidateVolatileRegisters() {
    // Invalidate XMM0-XMM5 (volatile registers on Win64)
    for (int i = 0; i <= 5; ++i) {
        xmmContents_[i] = -1;
        dirty_[i] = false;
    }
}

} // namespace forge