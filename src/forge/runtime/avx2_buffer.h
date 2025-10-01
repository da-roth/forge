#pragma once

#include "forge/runtime/node_buffer.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>

namespace forge {
namespace runtime {

/**
 * AVX2 implementation of NodeBuffer
 * Memory layout: values[nodeId * 4] contains 4 doubles per node (YMM register width)
 */
class AVX2NodeBuffer : public INodeBuffer {
public:
    explicit AVX2NodeBuffer(const forge::core::ComputationGraph& tape) 
        : num_nodes_(tape.nodes.size()), vector_width_(4) {
        diff_inputs_ = tape.diff_inputs;
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
        
        values_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));  // 32-byte alignment for AVX2
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, totalDoubles * sizeof(double));
        
        // Allocate gradients if needed (also 4 doubles per node)
        if (!diff_inputs_.empty()) {
            gradients_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));
            if (gradients_) {
                std::memset(gradients_, 0, totalDoubles * sizeof(double));
            }
        }
    }
    
    // Constructor with node ID mapping and explicit buffer size
    AVX2NodeBuffer(const forge::core::ComputationGraph& tape, 
                        const std::vector<forge::core::NodeId>& originalToOptimizedMapping,
                        size_t requiredNodes)
        : vector_width_(4), originalToOptimizedMapping_(originalToOptimizedMapping), num_nodes_(requiredNodes) {
        
        diff_inputs_ = tape.diff_inputs;
        
        // Use the exact size provided by the kernel
        // std::cout << "[AVX2 BUFFER] Using exact size from kernel: " << num_nodes_ << std::endl;
        
        // Allocate values - 4 doubles per node for AVX2
        size_t totalDoubles = num_nodes_ * vector_width_;
        
        // Safety check: ensure we allocate at least some memory
        if (totalDoubles == 0) {
            totalDoubles = vector_width_;  // At least one YMM register worth
        }
        
        values_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));  // 32-byte alignment for AVX2
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, totalDoubles * sizeof(double));
        
        // Allocate gradients if needed (also 4 doubles per node)
        if (!tape.diff_inputs.empty()) {
            gradients_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));
            if (!gradients_) {
                _aligned_free(values_);
                throw std::bad_alloc();
            }
            std::memset(gradients_, 0, totalDoubles * sizeof(double));
        } else {
            gradients_ = nullptr;
        }
    }
    
    // Constructor with node ID mapping (old version - calculates size from mapping)
    AVX2NodeBuffer(const forge::core::ComputationGraph& tape,
                        const std::vector<forge::core::NodeId>& originalToOptimizedMapping)
        : vector_width_(4), originalToOptimizedMapping_(originalToOptimizedMapping) {
        
        // Check if this is an identity mapping (all values are -1 or sequential)
        bool isIdentityMapping = true;
        size_t maxOptimizedNodeId = 0;
        bool hasValidMapping = false;
        
        for (size_t i = 0; i < originalToOptimizedMapping.size(); ++i) {
            forge::core::NodeId optimizedId = originalToOptimizedMapping[i];
            if (optimizedId != static_cast<forge::core::NodeId>(-1)) {
                hasValidMapping = true;
                maxOptimizedNodeId = std::max(maxOptimizedNodeId, static_cast<size_t>(optimizedId));
                if (optimizedId != static_cast<forge::core::NodeId>(i)) {
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
        
        // Allocate values - 4 doubles per node for AVX2
        size_t totalDoubles = num_nodes_ * vector_width_;
        
        // Safety check: ensure we allocate at least some memory
        if (totalDoubles == 0) {
            totalDoubles = vector_width_;  // At least one YMM register worth
        }
        
        values_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));  // 32-byte alignment for AVX2
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, totalDoubles * sizeof(double));
        
        // Allocate gradients if needed (also 4 doubles per node)
        if (!diff_inputs_.empty()) {
            gradients_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 32));
            if (gradients_) {
                std::memset(gradients_, 0, totalDoubles * sizeof(double));
            }
        }
    }
    
    ~AVX2NodeBuffer() {
        if (values_) {
            _aligned_free(values_);
        }
        if (gradients_) {
            _aligned_free(gradients_);
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
    double getGradient(forge::core::NodeId node) const override {
        if (!gradients_) {
            throw std::runtime_error("No gradients computed - no inputs marked with markInputAndDiff()");
        }
        // Map original node ID to optimized
        forge::core::NodeId mappedNode = node;
        if (node < originalToOptimizedMapping_.size()) {
            auto candidate = originalToOptimizedMapping_[node];
            if (candidate != static_cast<forge::core::NodeId>(UINT32_MAX)) {
                mappedNode = candidate;
            }
        }
        
        auto it = std::find(diff_inputs_.begin(), diff_inputs_.end(), mappedNode);
        if (it == diff_inputs_.end()) {
            throw std::runtime_error("Node was not marked for differentiation");
        }
        
        // Return first lane of gradient
        return gradients_[mappedNode * vector_width_];
    }
    
    std::vector<double> getVectorGradient(forge::core::NodeId node) const override {
        if (!gradients_) {
            throw std::runtime_error("No gradients computed - no inputs marked with markInputAndDiff()");
        }
        // Map original node ID to optimized
        forge::core::NodeId mappedNode = node;
        if (node < originalToOptimizedMapping_.size()) {
            auto candidate = originalToOptimizedMapping_[node];
            if (candidate != static_cast<forge::core::NodeId>(UINT32_MAX)) {
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
    AVX2NodeBuffer(const AVX2NodeBuffer&) = delete;
    AVX2NodeBuffer& operator=(const AVX2NodeBuffer&) = delete;
    
    // Enable move
    AVX2NodeBuffer(AVX2NodeBuffer&& other) noexcept 
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
    std::vector<forge::core::NodeId> diff_inputs_;
    std::vector<forge::core::NodeId> originalToOptimizedMapping_;
};

} // namespace runtime
} // namespace forge