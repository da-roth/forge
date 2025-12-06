#pragma once

#include "node_value_buffer.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <iostream>
#include <unordered_set>
#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc and _aligned_free on Windows
#else
#include <cstdlib>   // For aligned_alloc and free on Linux
#endif

namespace forge {

/**
 * Scalar implementation of NodeValueBuffer
 * Memory layout: values[nodeId] contains one double per node
 */
class ScalarNodeValueBuffer : public INodeValueBuffer {
public:
    explicit ScalarNodeValueBuffer(const forge::Graph& tape)
        : num_nodes_(tape.nodes.size()) {
        diff_inputs_ = tape.diff_inputs;
        // Build hash set for O(1) lookup in getGradient()
        diff_inputs_set_.insert(diff_inputs_.begin(), diff_inputs_.end());
        // Initialize identity mapping
        originalToOptimizedMapping_.resize(tape.nodes.size());
        for (size_t i = 0; i < tape.nodes.size(); ++i) {
            originalToOptimizedMapping_[i] = i;
        }

        // Allocate values - one double per node
        size_t totalDoubles = num_nodes_;

        // Safety check: ensure we allocate at least some memory
        if (totalDoubles == 0) {
            totalDoubles = 1;  // At least one double
        }

        // Platform-specific aligned allocation
#ifdef _WIN32
        values_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 64));
#else
        values_ = static_cast<double*>(aligned_alloc(64, totalDoubles * sizeof(double)));
#endif
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, totalDoubles * sizeof(double));

        // Allocate gradients if needed
        if (!diff_inputs_.empty()) {
#ifdef _WIN32
            gradients_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 64));
#else
            gradients_ = static_cast<double*>(aligned_alloc(64, totalDoubles * sizeof(double)));
#endif
            std::memset(gradients_, 0, totalDoubles * sizeof(double));
        }
    }

    // Constructor with node ID mapping
    ScalarNodeValueBuffer(const forge::Graph& tape,
                          const std::vector<forge::NodeId>& originalToOptimizedMapping)
        : num_nodes_(tape.nodes.size()), originalToOptimizedMapping_(originalToOptimizedMapping) {
        diff_inputs_ = tape.diff_inputs;
        // Build hash set for O(1) lookup in getGradient()
        diff_inputs_set_.insert(diff_inputs_.begin(), diff_inputs_.end());

        // Allocate values - one double per node
        size_t totalDoubles = num_nodes_;

        // Safety check: ensure we allocate at least some memory
        if (totalDoubles == 0) {
            totalDoubles = 1;  // At least one double
        }

        // Platform-specific aligned allocation
#ifdef _WIN32
        values_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 64));
#else
        values_ = static_cast<double*>(aligned_alloc(64, totalDoubles * sizeof(double)));
#endif
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, totalDoubles * sizeof(double));

        // Allocate gradients if needed
        if (!diff_inputs_.empty()) {
#ifdef _WIN32
            gradients_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 64));
#else
            gradients_ = static_cast<double*>(aligned_alloc(64, totalDoubles * sizeof(double)));
#endif
            std::memset(gradients_, 0, totalDoubles * sizeof(double));
        }
    }

    ~ScalarNodeValueBuffer() {
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
        // For scalar, just use the first value
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId < num_nodes_) {
                values_[optimizedNodeId] = values[0];
            }
        }
    }

    void getLanes(uint64_t nodeId, double* output) const override {
        // For scalar, just return the single value
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                output[0] = values_[optimizedNodeId];
            }
        }
    }

    // For scalar, only outputs[0] is filled (single lane)
    void getGradientLanes(const std::vector<size_t>& bufferIndices, double* outputs[4]) const override {
        if (!gradients_ || !outputs[0]) return;
        for (size_t i = 0; i < bufferIndices.size(); ++i) {
            outputs[0][i] = gradients_[bufferIndices[i]];
        }
    }

    // ==========================================================================
    // DEPRECATED API: Convenience wrappers (internally use Lanes)
    // ==========================================================================

    void setValue(uint64_t nodeId, double value) override {
        double data[1] = {value};
        setLanes(nodeId, data);
    }

    double getValue(uint64_t nodeId) const override {
        double data[1] = {0.0};
        getLanes(nodeId, data);
        return data[0];
    }

    size_t getBufferIndex(uint64_t nodeId) const override {
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                return optimizedNodeId;  // vectorWidth is 1
            }
        }
        return SIZE_MAX;
    }

    // ==========================================================================
    // Gradient access (deprecated getGradient uses getGradientLanes internally)
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
        std::vector<size_t> indices = {mappedNode};
        double lane0[1];
        double* outputs[4] = {lane0, nullptr, nullptr, nullptr};
        getGradientLanes(indices, outputs);
        return lane0[0];
    }

    void clearGradients() override {
        if (gradients_) {
            std::memset(gradients_, 0, num_nodes_ * sizeof(double));
        }
    }

    bool hasGradients() const override {
        return gradients_ != nullptr;
    }

    // Buffer info
    int getVectorWidth() const override { return 1; }
    uint64_t getNumNodes() const override { return num_nodes_; }

    // Raw access
    double* getValuesPtr() override { return values_; }
    double* getGradientsPtr() override { return gradients_; }

    // Disable copy
    ScalarNodeValueBuffer(const ScalarNodeValueBuffer&) = delete;
    ScalarNodeValueBuffer& operator=(const ScalarNodeValueBuffer&) = delete;

    // Enable move
    ScalarNodeValueBuffer(ScalarNodeValueBuffer&& other) noexcept
        : values_(other.values_), gradients_(other.gradients_),
          num_nodes_(other.num_nodes_), diff_inputs_(std::move(other.diff_inputs_)) {
        other.values_ = nullptr;
        other.gradients_ = nullptr;
        other.num_nodes_ = 0;
    }

private:
    double* values_ = nullptr;
    double* gradients_ = nullptr;
    uint64_t num_nodes_;
    std::vector<forge::NodeId> diff_inputs_;
    std::unordered_set<forge::NodeId> diff_inputs_set_;  // O(1) lookup for getGradient()
    std::vector<forge::NodeId> originalToOptimizedMapping_;
};

} // namespace forge
