#pragma once

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <initializer_list>

namespace forge {
namespace x86 {

/**
 * Abstract interface for register allocators.
 * This allows different instruction sets to use different register types (XMM, YMM, ZMM)
 * while sharing the same interface.
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
    virtual int findNodeInRegister(uint32_t nodeId) const = 0;
    virtual void setRegister(int regIndex, uint32_t nodeId, bool isDirty = false) = 0;
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

/**
 * Template base class for register allocators.
 * Provides shared implementation for XMM, YMM, and future register types.
 * 
 * @tparam RegType The register type (e.g., asmjit::x86::Xmm, asmjit::x86::Ymm)
 * @tparam NUM_REGS The number of registers available
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
    
    int findNodeInRegister(uint32_t nodeId) const override {
        for (int i = 0; i < NUM_REGS; i++) {
            if (contents_[i] == static_cast<int>(nodeId)) {
                return i;
            }
        }
        return -1;
    }
    
    void setRegister(int regIndex, uint32_t nodeId, bool isDirty) override {
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
    
    bool isBlacklisted(int regIndex) const {
        if (regIndex >= 0 && regIndex < NUM_REGS) {
            return blacklisted_[regIndex];
        }
        return false;
    }
    
protected:
    // Helper to get register count
    static constexpr int getRegisterCount() { return NUM_REGS; }
};

} // namespace x86
} // namespace forge