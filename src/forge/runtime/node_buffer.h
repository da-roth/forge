#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include "forge/core/computation_graph.h"
#include "forge/runtime/kernel_requirements.h"

namespace forge {
namespace runtime {

// Forward declaration
class CompiledKernel;

/**
 * Interface for node value storage that kernels read from and write to.
 * Different implementations handle different memory layouts (scalar vs SIMD).
 */
class INodeBuffer {
public:
    virtual ~INodeBuffer() = default;
    
    // Core value access
    virtual void setValue(uint64_t nodeId, double value) = 0;
    virtual double getValue(uint64_t nodeId) const = 0;
    
    // Vector lane access (for SIMD operations)
    virtual void setVectorValue(uint64_t nodeId, const std::vector<double>& values) = 0;
    virtual std::vector<double> getVectorValue(uint64_t nodeId) const = 0;
    
    // Gradient access (for automatic differentiation)
    virtual double getGradient(forge::core::NodeId node) const = 0;
    virtual std::vector<double> getVectorGradient(forge::core::NodeId node) const = 0;
    virtual std::vector<double> getGradients() const = 0;
    virtual void clearGradients() = 0;
    virtual bool hasGradients() const = 0;
    
    // Buffer info
    virtual int getVectorWidth() const = 0;
    virtual uint64_t getNumNodes() const = 0;
    
    // Raw access for kernel execution
    virtual double* getValuesPtr() = 0;
    virtual double* getGradientsPtr() = 0;
};

/**
 * Factory for creating appropriate NodeBuffer based on kernel requirements
 */
class NodeBufferFactory {
public:
    // New primary method using KernelRequirements
    static std::unique_ptr<INodeBuffer> create(
        const forge::core::ComputationGraph& tape,
        const KernelRequirements& requirements);
    
    // Legacy methods for backward compatibility (will delegate to new method)
    static std::unique_ptr<INodeBuffer> create(
        const forge::core::ComputationGraph& tape,
        const CompiledKernel& kernel);
    
    static std::unique_ptr<INodeBuffer> create(
        const forge::core::ComputationGraph& tape,
        const CompiledKernel& kernel,
        const std::vector<forge::core::NodeId>& originalToOptimizedMapping);
};

} // namespace runtime
} // namespace forge