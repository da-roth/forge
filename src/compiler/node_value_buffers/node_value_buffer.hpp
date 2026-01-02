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
 *
 * == API Design ==
 *
 * Primary API (Lanes): Raw pointer interface for performance-critical code.
 *   - setLanes(nodeId, ptr)           - Set all SIMD lanes from array
 *   - getLanes(nodeId, ptr)           - Get all SIMD lanes to array
 *   - getGradientLanes(indices, ptr)  - Get gradients (interleaved, adapts to vector width)
 *
 * Deprecated API: Convenience wrappers that internally call Lanes methods.
 *   - setValue(nodeId, value)         - Broadcasts to all lanes
 *   - getValue(nodeId)                - Returns lane 0
 *   - getGradient(nodeId)             - Returns lane 0 gradient
 *
 * The deprecated methods will be removed in a future version.
 * Migrate to the Lanes API for better performance.
 */
class INodeValueBuffer {
public:
    virtual ~INodeValueBuffer() = default;

    // ==========================================================================
    // PRIMARY API: Lanes (raw pointer, no allocation)
    // ==========================================================================

    /**
     * Set values for all SIMD lanes from a raw pointer.
     * @param nodeId The node to set values for
     * @param values Pointer to getVectorWidth() doubles (4 for AVX2, 1 for scalar)
     */
    virtual void setLanes(uint64_t nodeId, const double* values) = 0;

    /**
     * Get values for all SIMD lanes into a raw pointer.
     * @param nodeId The node to get values from
     * @param output Pointer with space for getVectorWidth() doubles
     */
    virtual void getLanes(uint64_t nodeId, double* output) const = 0;

    /**
     * Get gradients for multiple nodes, all lanes at once (interleaved layout).
     * This is the most efficient way to retrieve gradients in hot loops.
     * @param bufferIndices Pre-computed buffer indices (from getBufferIndex)
     * @param output Pointer with space for bufferIndices.size() * getVectorWidth() doubles.
     *               Layout: [node0_lane0, node0_lane1, ..., node1_lane0, node1_lane1, ...]
     *               For scalar (width=1): simply [grad0, grad1, grad2, ...]
     *               For AVX2 (width=4): [n0_L0, n0_L1, n0_L2, n0_L3, n1_L0, n1_L1, ...]
     */
    virtual void getGradientLanes(const std::vector<size_t>& bufferIndices, double* output) const = 0;

    // ==========================================================================
    // DEPRECATED API: Convenience wrappers (internally use Lanes)
    // ==========================================================================

    /**
     * @deprecated Use setLanes() for better performance.
     * Set a single value, broadcast to all SIMD lanes.
     */
    [[deprecated("Use setLanes() for better performance")]]
    virtual void setValue(uint64_t nodeId, double value) = 0;

    /**
     * @deprecated Use getLanes() for better performance.
     * Get a single value (lane 0).
     */
    [[deprecated("Use getLanes() for better performance")]]
    virtual double getValue(uint64_t nodeId) const = 0;

    /**
     * @deprecated Use getGradientLanes() for better performance.
     * Get gradient for a single node (lane 0).
     */
    [[deprecated("Use getGradientLanes() for better performance")]]
    virtual double getGradient(forge::NodeId node) const = 0;

    // ==========================================================================
    // Supporting methods
    // ==========================================================================

    /**
     * Get buffer index for a node ID.
     * Use this to pre-compute indices outside hot loops for use with getGradientLanes().
     * @return The base index into values_/gradients_ arrays, or SIZE_MAX if invalid
     */
    virtual size_t getBufferIndex(uint64_t nodeId) const = 0;

    /** Clear all gradients to zero */
    virtual void clearGradients() = 0;

    /** Check if gradients have been computed */
    virtual bool hasGradients() const = 0;

    /** Get the SIMD vector width (1 for scalar, 4 for AVX2) */
    virtual int getVectorWidth() const = 0;

    /** Get the number of nodes in this buffer */
    virtual uint64_t getNumNodes() const = 0;

    /** Get raw pointer to values buffer (for direct AVX2 intrinsic access) */
    virtual double* getValuesPtr() = 0;

    /** Get raw pointer to gradients buffer (for direct AVX2 intrinsic access) */
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
};

} // namespace forge
