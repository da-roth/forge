#pragma once

#include "node_value_buffer.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc and _aligned_free on Windows
#endif

namespace forge {

/**
 * AVX2 implementation of NodeValueBuffer
 * Memory layout: values[nodeId * 4] contains 4 doubles per node (YMM register width)
 */
class AVX2NodeValueBuffer : public INodeValueBuffer {
public:
    explicit AVX2NodeValueBuffer(const forge::Graph& tape)
        : num_nodes_(tape.nodes.size()), vector_width_(4) {
        diff_inputs_ = tape.diff_inputs;
        // Build hash set for O(1) lookup in getGradient()
        diff_inputs_set_.insert(diff_inputs_.begin(), diff_inputs_.end());
        // Initialize identity mapping
        originalToOptimizedMapping_.resize(tape.nodes.size());
        for (size_t i = 0; i < tape.nodes.size(); ++i) {
            originalToOptimizedMapping_[i] = i;
        }
        
        // Allocate values - 4 doubles per node for AVX2
        size_t totalDoubles = num_nodes_ * vector_width_;

        // Safety check: ensure we allocate at least some memory
        if (totalDoubles == 0) {
            totalDoubles = vector_width_;  // At least one YMM register worth
        }

        // Platform-specific aligned allocation
#ifdef _WIN32
        values_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));  // 32-byte alignment for AVX2
#else
        values_ = static_cast<double*>(aligned_alloc(32, totalDoubles * sizeof(double)));
#endif
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, totalDoubles * sizeof(double));

        // Allocate gradients if needed (also 4 doubles per node)
        if (!diff_inputs_.empty()) {
#ifdef _WIN32
            gradients_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));
#else
            gradients_ = static_cast<double*>(aligned_alloc(32, totalDoubles * sizeof(double)));
#endif
            if (gradients_) {
                std::memset(gradients_, 0, totalDoubles * sizeof(double));
            }
        }
    }

    // Constructor with node ID mapping and explicit buffer size
    AVX2NodeValueBuffer(const forge::Graph& tape,
                        const std::vector<forge::NodeId>& originalToOptimizedMapping,
                        size_t requiredNodes)
        : vector_width_(4), originalToOptimizedMapping_(originalToOptimizedMapping), num_nodes_(requiredNodes) {

        diff_inputs_ = tape.diff_inputs;
        // Build hash set for O(1) lookup in getGradient()
        diff_inputs_set_.insert(diff_inputs_.begin(), diff_inputs_.end());

        // Use the exact size provided by the kernel
        std::cout << "[AVX2 BUFFER] Using exact size from kernel: " << num_nodes_ << std::endl;

        // Allocate values - 4 doubles per node for AVX2
        size_t totalDoubles = num_nodes_ * vector_width_;

        // Safety check: ensure we allocate at least some memory
        if (totalDoubles == 0) {
            totalDoubles = vector_width_;  // At least one YMM register worth
        }

        // Platform-specific aligned allocation
#ifdef _WIN32
        values_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));  // 32-byte alignment for AVX2
#else
        values_ = static_cast<double*>(aligned_alloc(32, totalDoubles * sizeof(double)));
#endif
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, totalDoubles * sizeof(double));

        // Allocate gradients if needed (also 4 doubles per node)
        if (!tape.diff_inputs.empty()) {
#ifdef _WIN32
            gradients_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));
#else
            gradients_ = static_cast<double*>(aligned_alloc(32, totalDoubles * sizeof(double)));
#endif
            if (!gradients_) {
#ifdef _WIN32
                _aligned_free(values_);
#else
                free(values_);
#endif
                throw std::bad_alloc();
            }
            std::memset(gradients_, 0, totalDoubles * sizeof(double));
        } else {
            gradients_ = nullptr;
        }
    }
    
    // Constructor with node ID mapping (old version - calculates size from mapping)
    AVX2NodeValueBuffer(const forge::Graph& tape, 
                        const std::vector<forge::NodeId>& originalToOptimizedMapping)
        : vector_width_(4), originalToOptimizedMapping_(originalToOptimizedMapping) {
        
        // Check if this is an identity mapping (all values are -1 or sequential)
        bool isIdentityMapping = true;
        size_t maxOptimizedNodeId = 0;
        bool hasValidMapping = false;
        
        for (size_t i = 0; i < originalToOptimizedMapping.size(); ++i) {
            forge::NodeId optimizedId = originalToOptimizedMapping[i];
            if (optimizedId != static_cast<forge::NodeId>(-1)) {
                hasValidMapping = true;
                maxOptimizedNodeId = std::max(maxOptimizedNodeId, static_cast<size_t>(optimizedId));
                if (optimizedId != static_cast<forge::NodeId>(i)) {
                    isIdentityMapping = false;
                }
            }
        }
        
        // If no valid mappings found (all -1), use identity mapping
        if (!hasValidMapping || isIdentityMapping) {
            // Use the original tape size for identity mapping
            num_nodes_ = originalToOptimizedMapping.size();
            // Replace the mapping with identity
            originalToOptimizedMapping_.clear();
            originalToOptimizedMapping_.resize(num_nodes_);
            for (size_t i = 0; i < num_nodes_; ++i) {
                originalToOptimizedMapping_[i] = i;
            }
        } else {
            // Buffer needs to accommodate all optimized node IDs (0 to maxOptimizedNodeId inclusive)
            num_nodes_ = maxOptimizedNodeId + 1;
        }

        diff_inputs_ = tape.diff_inputs;
        // Build hash set for O(1) lookup in getGradient()
        diff_inputs_set_.insert(diff_inputs_.begin(), diff_inputs_.end());

        // Allocate values - 4 doubles per node for AVX2
        size_t totalDoubles = num_nodes_ * vector_width_;

        // Safety check: ensure we allocate at least some memory
        if (totalDoubles == 0) {
            totalDoubles = vector_width_;  // At least one YMM register worth
        }

        // Platform-specific aligned allocation
#ifdef _WIN32
        values_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));  // 32-byte alignment for AVX2
#else
        values_ = static_cast<double*>(aligned_alloc(32, totalDoubles * sizeof(double)));
#endif
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, totalDoubles * sizeof(double));

        // Allocate gradients if needed (also 4 doubles per node)
        if (!diff_inputs_.empty()) {
#ifdef _WIN32
            gradients_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));
#else
            gradients_ = static_cast<double*>(aligned_alloc(32, totalDoubles * sizeof(double)));
#endif
            if (gradients_) {
                std::memset(gradients_, 0, totalDoubles * sizeof(double));
            }
        }
    }

    ~AVX2NodeValueBuffer() {
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
            // FIX: Check for invalid mapping (-1 sentinel value)
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                // Broadcast to all 4 lanes
                size_t baseIdx = optimizedNodeId * vector_width_;
                // Safety check for buffer overflow
                size_t endIdx = baseIdx + vector_width_;
                if (endIdx <= num_nodes_ * vector_width_) {
                    for (int i = 0; i < vector_width_; i++) {
                        values_[baseIdx + i] = value;
                    }
                }
            }
        }
    }
    
    double getValue(uint64_t nodeId) const override {
        // Map original node ID to optimized node ID
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                size_t baseIdx = optimizedNodeId * vector_width_;
                double result = values_[baseIdx];
                
                return result;
            }
        }
        return 0.0;
    }
    
    // Vector lane access
    void setVectorValue(uint64_t nodeId, const std::vector<double>& values) override {
        // Map original node ID to optimized node ID
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                size_t baseIdx = optimizedNodeId * vector_width_;
                size_t numValues = std::min(static_cast<size_t>(vector_width_), values.size());
                
                // Set provided values
                for (size_t i = 0; i < numValues; i++) {
                    values_[baseIdx + i] = values[i];
                }
                
                // Replicate last value if fewer than 4 provided
                if (values.size() < static_cast<size_t>(vector_width_) && !values.empty()) {
                    double lastValue = values.back();
                    for (size_t i = numValues; i < static_cast<size_t>(vector_width_); i++) {
                        values_[baseIdx + i] = lastValue;
                    }
                }
            }
        }
    }
    
    std::vector<double> getVectorValue(uint64_t nodeId) const override {
        std::vector<double> result;
        
        // Map original node ID to optimized node ID
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                size_t baseIdx = optimizedNodeId * vector_width_;
                for (int i = 0; i < vector_width_; i++) {
                    result.push_back(values_[baseIdx + i]);
                }
                return result;
            } else {
            }
        } else {
        }
        
        return result;
    }
    
    // Gradient access
    double getGradient(forge::NodeId node) const override {
        if (!gradients_) {
            throw std::runtime_error("No gradients computed - no inputs marked with markInputAndDiff()");
        }
        // Map original node ID to optimized
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

        // Return first lane of gradient
        return gradients_[mappedNode * vector_width_];
    }
    
    std::vector<double> getVectorGradient(forge::NodeId node) const override {
        if (!gradients_) {
            throw std::runtime_error("No gradients computed - no inputs marked with markInputAndDiff()");
        }
        // Map original node ID to optimized
        forge::NodeId mappedNode = node;
        if (node < originalToOptimizedMapping_.size()) {
            auto candidate = originalToOptimizedMapping_[node];
            if (candidate != static_cast<forge::NodeId>(UINT32_MAX)) {
                mappedNode = candidate;
            }
        }
        
        auto it = std::find(diff_inputs_.begin(), diff_inputs_.end(), mappedNode);
        if (it == diff_inputs_.end()) {
            throw std::runtime_error("Node was not marked for differentiation");
        }
        
        std::vector<double> result;
        size_t baseIdx = mappedNode * vector_width_;
        for (int i = 0; i < vector_width_; i++) {
            result.push_back(gradients_[baseIdx + i]);
        }
        return result;
    }
    
    std::vector<double> getGradients() const override {
        if (!gradients_) {
            return {};
        }

        std::vector<double> result;
        for (auto node : diff_inputs_) {
            // Return first lane for each gradient
            result.push_back(gradients_[node * vector_width_]);
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
            result[i] = gradients_[mappedNode * vector_width_];
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
            std::memset(gradients_, 0, num_nodes_ * vector_width_ * sizeof(double));
        }
    }
    
    bool hasGradients() const override {
        return gradients_ != nullptr;
    }
    
    // Buffer info
    int getVectorWidth() const override { return vector_width_; }
    uint64_t getNumNodes() const override { return num_nodes_; }
    
    // Raw access
    double* getValuesPtr() override { return values_; }
    double* getGradientsPtr() override { return gradients_; }
    
    // Disable copy
    AVX2NodeValueBuffer(const AVX2NodeValueBuffer&) = delete;
    AVX2NodeValueBuffer& operator=(const AVX2NodeValueBuffer&) = delete;
    
    // Enable move
    AVX2NodeValueBuffer(AVX2NodeValueBuffer&& other) noexcept 
        : values_(other.values_), gradients_(other.gradients_),
          num_nodes_(other.num_nodes_), vector_width_(other.vector_width_),
          diff_inputs_(std::move(other.diff_inputs_)) {
        other.values_ = nullptr;
        other.gradients_ = nullptr;
        other.num_nodes_ = 0;
    }
    
private:
    double* values_ = nullptr;
    double* gradients_ = nullptr;
    uint64_t num_nodes_;
    const int vector_width_;
    std::vector<forge::NodeId> diff_inputs_;
    std::unordered_set<forge::NodeId> diff_inputs_set_;  // O(1) lookup for getGradient()
    std::vector<forge::NodeId> originalToOptimizedMapping_;
};

} // namespace forge