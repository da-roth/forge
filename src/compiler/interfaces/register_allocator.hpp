// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file register_allocator.hpp
 * @brief Abstract interface for register allocators
 *
 * Defines the IRegisterAllocator interface that all register allocators
 * must implement to work with the JIT compiler.
 *
 * Thread Safety: Not thread-safe - each compilation uses its own allocator
 */

#pragma once

#include "../../graph/graph.hpp"
#include <initializer_list>

namespace forge {

/**
 * @brief Abstract interface for register allocators
 *
 * All register allocators must implement this interface to work with the
 * JIT compiler. The interface abstracts away differences between XMM, YMM,
 * and future register types while providing a common allocation strategy.
 *
 * Features:
 * - LRU (Least Recently Used) allocation strategy
 * - Register locking for values in use
 * - Dirty tracking for writeback optimization
 * - Node-to-register mapping
 * - Volatile register invalidation (calling convention support)
 * - Register blacklisting (corruption workaround)
 *
 * API Stability: Stable
 */
class IRegisterAllocator {
public:
    virtual ~IRegisterAllocator() = default;

    // Core allocation interface
    virtual int allocateRegister() = 0;
    virtual int allocateAvoiding(std::initializer_list<int> avoid) = 0;

    // Register state management
    virtual void clear() = 0;
    virtual void lock(int regIndex) = 0;
    virtual void unlock(int regIndex) = 0;

    // Register content tracking
    virtual int findNodeInRegister(forge::NodeId nodeId) const = 0;
    virtual void setRegister(int regIndex, forge::NodeId nodeId, bool isDirty = false) = 0;
    virtual int getNodeInRegister(int regIndex) const = 0;

    // Dirty register tracking
    virtual void markDirty(int regIndex) = 0;
    virtual void markClean(int regIndex) = 0;
    virtual bool isDirty(int regIndex) const = 0;

    // Platform-specific invalidation
    virtual void invalidateVolatileRegisters() = 0;
    virtual int getFirstVolatileReg() const = 0;
    virtual int getLastVolatileReg() const = 0;

    // Get the number of registers
    virtual int getNumRegisters() const = 0;
};

} // namespace forge
