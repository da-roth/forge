#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
#include <unordered_set>
#include <stdexcept>
#include "../../graph/graph.hpp"

#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc and _aligned_free on Windows
#else
#include <cstdlib>   // For aligned_alloc and free on Linux
#endif

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
     */
    virtual void getGradientLanes(const std::vector<size_t>& bufferIndices, double* output) const = 0;

    /**
     * Set values for multiple nodes at once using pre-computed buffer indices.
     * Batched equivalent of setLanes() - much faster due to single virtual call.
     * @param bufferIndices Pre-computed buffer indices (from getBufferIndex)
     * @param values Pointer to bufferIndices.size() * getVectorWidth() doubles
     */
    virtual void setValueLanes(const std::vector<size_t>& bufferIndices, const double* values) = 0;

    /**
     * Get values for multiple nodes at once using pre-computed buffer indices.
     * Batched equivalent of getLanes() - much faster due to single virtual call.
     * @param bufferIndices Pre-computed buffer indices (from getBufferIndex)
     * @param output Pointer with space for bufferIndices.size() * getVectorWidth() doubles
     */
    virtual void getValueLanes(const std::vector<size_t>& bufferIndices, double* output) const = 0;

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
 * Template base class for NodeValueBuffer implementations.
 * Provides common functionality with vector width as template parameter.
 *
 * @tparam VectorWidth Number of doubles per node (1 for scalar, 4 for AVX2)
 * @tparam Alignment Memory alignment in bytes (64 for scalar, 32 for AVX2)
 */
template<int VectorWidth, size_t Alignment>
class NodeValueBufferBase : public INodeValueBuffer {
public:
    static constexpr int VECTOR_WIDTH = VectorWidth;
    static constexpr size_t ALIGNMENT = Alignment;

    NodeValueBufferBase(const forge::Graph& tape,
                        const std::vector<forge::NodeId>& originalToOptimizedMapping,
                        size_t requiredNodes)
        : originalToOptimizedMapping_(originalToOptimizedMapping), num_nodes_(requiredNodes) {

        diff_inputs_ = tape.diff_inputs;
        // Build hash set for O(1) lookup in getGradient()
        diff_inputs_set_.insert(diff_inputs_.begin(), diff_inputs_.end());

        // Allocate values - VectorWidth doubles per node
        size_t totalDoubles = num_nodes_ * VectorWidth;

        // Safety check: ensure we allocate at least some memory
        if (totalDoubles == 0) {
            totalDoubles = VectorWidth;
        }

        // Calculate allocation size - must be multiple of alignment for aligned_alloc on Linux
        size_t allocSize = totalDoubles * sizeof(double);
        size_t alignedAllocSize = (allocSize + (Alignment - 1)) & ~(Alignment - 1);

        // Platform-specific aligned allocation
#ifdef _WIN32
        values_ = static_cast<double*>(_aligned_malloc(allocSize, Alignment));
#else
        values_ = static_cast<double*>(aligned_alloc(Alignment, alignedAllocSize));
#endif
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, allocSize);

        // Allocate gradients if needed
        if (!tape.diff_inputs.empty()) {
#ifdef _WIN32
            gradients_ = static_cast<double*>(_aligned_malloc(allocSize, Alignment));
#else
            gradients_ = static_cast<double*>(aligned_alloc(Alignment, alignedAllocSize));
#endif
            if (!gradients_) {
#ifdef _WIN32
                _aligned_free(values_);
#else
                free(values_);
#endif
                throw std::bad_alloc();
            }
            std::memset(gradients_, 0, allocSize);
        }
    }

    ~NodeValueBufferBase() override {
        if (values_) {
#ifdef _WIN32
            _aligned_free(values_);
#else
            free(values_);
#endif
        }
        if (gradients_) {
#ifdef _WIN32
            _aligned_free(gradients_);
#else
            free(gradients_);
#endif
        }
    }

    // ==========================================================================
    // PRIMARY API: Lanes (raw pointer, no allocation)
    // ==========================================================================

    void setLanes(uint64_t nodeId, const double* values) override {
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                size_t baseIdx = optimizedNodeId * VectorWidth;
                if constexpr (VectorWidth == 1) {
                    values_[baseIdx] = values[0];
                } else {
                    std::memcpy(&values_[baseIdx], values, VectorWidth * sizeof(double));
                }
            }
        }
    }

    void getLanes(uint64_t nodeId, double* output) const override {
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                size_t baseIdx = optimizedNodeId * VectorWidth;
                if constexpr (VectorWidth == 1) {
                    output[0] = values_[baseIdx];
                } else {
                    std::memcpy(output, &values_[baseIdx], VectorWidth * sizeof(double));
                }
            }
        }
    }

    void getGradientLanes(const std::vector<size_t>& bufferIndices, double* output) const override {
        if (!gradients_) return;
        for (size_t i = 0; i < bufferIndices.size(); ++i) {
            if constexpr (VectorWidth == 1) {
                output[i] = gradients_[bufferIndices[i]];
            } else {
                std::memcpy(&output[i * VectorWidth], &gradients_[bufferIndices[i]], VectorWidth * sizeof(double));
            }
        }
    }

    void setValueLanes(const std::vector<size_t>& bufferIndices, const double* values) override {
        for (size_t i = 0; i < bufferIndices.size(); ++i) {
            if constexpr (VectorWidth == 1) {
                values_[bufferIndices[i]] = values[i];
            } else {
                std::memcpy(&values_[bufferIndices[i]], &values[i * VectorWidth], VectorWidth * sizeof(double));
            }
        }
    }

    void getValueLanes(const std::vector<size_t>& bufferIndices, double* output) const override {
        for (size_t i = 0; i < bufferIndices.size(); ++i) {
            if constexpr (VectorWidth == 1) {
                output[i] = values_[bufferIndices[i]];
            } else {
                std::memcpy(&output[i * VectorWidth], &values_[bufferIndices[i]], VectorWidth * sizeof(double));
            }
        }
    }

    // ==========================================================================
    // DEPRECATED API: Convenience wrappers (internally use Lanes)
    // ==========================================================================

    void setValue(uint64_t nodeId, double value) override {
        if constexpr (VectorWidth == 1) {
            double data[1] = {value};
            setLanes(nodeId, data);
        } else {
            double data[VectorWidth];
            for (int i = 0; i < VectorWidth; ++i) data[i] = value;
            setLanes(nodeId, data);
        }
    }

    double getValue(uint64_t nodeId) const override {
        double data[VectorWidth];
        getLanes(nodeId, data);
        return data[0];
    }

    size_t getBufferIndex(uint64_t nodeId) const override {
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                return optimizedNodeId * VectorWidth;
            }
        }
        return SIZE_MAX;
    }

    // ==========================================================================
    // Gradient access
    // ==========================================================================

    double getGradient(forge::NodeId node) const override {
        if (!gradients_) {
            throw std::runtime_error("No gradients computed - no inputs marked with markInputAndDiff()");
        }

        // Map original node ID to optimized if mapping is available
        forge::NodeId mappedNode = node;
        if (node < originalToOptimizedMapping_.size()) {
            auto candidate = originalToOptimizedMapping_[node];
            if (candidate != static_cast<forge::NodeId>(UINT32_MAX)) {
                mappedNode = candidate;
            }
        }

        // O(1) lookup using hash set instead of O(n) linear search
        if (diff_inputs_set_.find(mappedNode) == diff_inputs_set_.end()) {
            throw std::runtime_error("Node was not marked for differentiation");
        }

        // Use getGradientLanes internally
        size_t bufferIdx = mappedNode * VectorWidth;
        std::vector<size_t> indices = {bufferIdx};
        double grad[VectorWidth];
        getGradientLanes(indices, grad);
        return grad[0];
    }

    void clearGradients() override {
        if (gradients_) {
            std::memset(gradients_, 0, num_nodes_ * VectorWidth * sizeof(double));
        }
    }

    bool hasGradients() const override {
        return gradients_ != nullptr;
    }

    // Buffer info
    int getVectorWidth() const override { return VectorWidth; }
    uint64_t getNumNodes() const override { return num_nodes_; }

    // Raw access
    double* getValuesPtr() override { return values_; }
    double* getGradientsPtr() override { return gradients_; }

    // Disable copy
    NodeValueBufferBase(const NodeValueBufferBase&) = delete;
    NodeValueBufferBase& operator=(const NodeValueBufferBase&) = delete;

    // Enable move
    NodeValueBufferBase(NodeValueBufferBase&& other) noexcept
        : values_(other.values_), gradients_(other.gradients_),
          num_nodes_(other.num_nodes_), diff_inputs_(std::move(other.diff_inputs_)),
          diff_inputs_set_(std::move(other.diff_inputs_set_)),
          originalToOptimizedMapping_(std::move(other.originalToOptimizedMapping_)) {
        other.values_ = nullptr;
        other.gradients_ = nullptr;
        other.num_nodes_ = 0;
    }

protected:
    double* values_ = nullptr;
    double* gradients_ = nullptr;
    uint64_t num_nodes_;
    std::vector<forge::NodeId> diff_inputs_;
    std::unordered_set<forge::NodeId> diff_inputs_set_;
    std::vector<forge::NodeId> originalToOptimizedMapping_;
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
