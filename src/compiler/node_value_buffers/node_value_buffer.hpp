#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include "../../graph/graph.hpp"

namespace forge {

// Forward declaration
class StitchedKernel;

/**
 * Interface for node value storage that kernels read from and write to.
 * Different implementations handle different memory layouts (scalar vs SIMD).
 */
class INodeValueBuffer {
public:
    virtual ~INodeValueBuffer() = default;
    
    // Core value access
    virtual void setValue(uint64_t nodeId, double value) = 0;
    virtual double getValue(uint64_t nodeId) const = 0;
    
    // Vector lane access (for SIMD operations)
    virtual void setVectorValue(uint64_t nodeId, const std::vector<double>& values) = 0;
    virtual std::vector<double> getVectorValue(uint64_t nodeId) const = 0;
    
    // Gradient access (for automatic differentiation)
    virtual double getGradient(forge::NodeId node) const = 0;
    virtual std::vector<double> getVectorGradient(forge::NodeId node) const = 0;
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
 * Factory for creating appropriate NodeValueBuffer based on kernel requirements
 */
class NodeValueBufferFactory {
public:
    static std::unique_ptr<INodeValueBuffer> create(
        const forge::Graph& tape,
        const StitchedKernel& kernel);
    
    // New method that accepts node ID mapping
    static std::unique_ptr<INodeValueBuffer> create(
        const forge::Graph& tape,
        const StitchedKernel& kernel,
        const std::vector<forge::NodeId>& originalToOptimizedMapping);
};

} // namespace forge