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
    
    // Core value access
        void setValue(uint64_t nodeId, double value) override {
            // Map original node ID to optimized node ID
            if (nodeId < originalToOptimizedMapping_.size()) {
                uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
                if (optimizedNodeId < num_nodes_) {
                    values_[optimizedNodeId] = value;
                    // Debug output commented out for cleaner test output
                    // std::cout << "[SCALAR BUFFER DEBUG] setValue: original=" << nodeId << ", optimized=" << optimizedNodeId << ", value=" << value << std::endl;
                } else {
                    // std::cout << "[SCALAR BUFFER DEBUG] setValue FAILED: original=" << nodeId << ", optimized=" << optimizedNodeId << " >= num_nodes=" << num_nodes_ << std::endl;
                }
            } else {
                // std::cout << "[SCALAR BUFFER DEBUG] setValue FAILED: original=" << nodeId << " >= mapping size=" << originalToOptimizedMapping_.size() << std::endl;
            }
        }
    
    double getValue(uint64_t nodeId) const override {
        // Map original node ID to optimized node ID
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                double result = values_[optimizedNodeId];
                // std::cout << "[SCALAR BUFFER DEBUG] getValue: original=" << nodeId << ", optimized=" << optimizedNodeId << ", result=" << result << std::endl;
                return result;
            } else {
                // std::cout << "[SCALAR BUFFER DEBUG] getValue FAILED: original=" << nodeId << ", optimized=" << optimizedNodeId << " >= num_nodes=" << num_nodes_ << std::endl;
            }
        } else {
            // std::cout << "[SCALAR BUFFER DEBUG] getValue FAILED: original=" << nodeId << " >= mapping size=" << originalToOptimizedMapping_.size() << std::endl;
        }
        return 0.0;
    }
    
    // Vector lane access (returns single value for scalar)
    void setVectorValue(uint64_t nodeId, const std::vector<double>& values) override {
        if (nodeId < num_nodes_ && !values.empty()) {
            values_[nodeId] = values[0];  // Use first value for scalar
        }
    }
    
    std::vector<double> getVectorValue(uint64_t nodeId) const override {
        std::vector<double> result;
        if (nodeId < num_nodes_) {
            result.push_back(values_[nodeId]);
        }
        return result;
    }
    
    // Gradient access
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

        return gradients_[mappedNode];
    }
    
    std::vector<double> getVectorGradient(forge::NodeId node) const override {
        return {getGradient(node)};
    }
    
    std::vector<double> getGradients() const override {
        if (!gradients_) {
            return {};
        }

        std::vector<double> result;
        for (auto node : diff_inputs_) {
            result.push_back(gradients_[node]);
        }
        return result;
    }

    // Fast batch gradient access - no validation, direct memory access
    std::vector<double> getGradientsBatch(const std::vector<forge::NodeId>& nodes) const override {
        std::vector<double> result;
        if (!gradients_) {
            return result;
        }
        result.resize(nodes.size());  // Resize once, no push_back
        for (size_t i = 0; i < nodes.size(); ++i) {
            // Map original node ID to optimized
            forge::NodeId node = nodes[i];
            forge::NodeId mappedNode = node;
            if (node < originalToOptimizedMapping_.size()) {
                auto candidate = originalToOptimizedMapping_[node];
                if (candidate != static_cast<forge::NodeId>(UINT32_MAX)) {
                    mappedNode = candidate;
                }
            }
            // Direct array access - no validation
            result[i] = gradients_[mappedNode];
        }
        return result;
    }

    // Ultra-fast: direct array read with pre-computed indices (no mapping, no allocation)
    void getGradientsDirect(const std::vector<size_t>& bufferIndices, double* output) const override {
        if (!gradients_) return;
        for (size_t i = 0; i < bufferIndices.size(); ++i) {
            output[i] = gradients_[bufferIndices[i]];
        }
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