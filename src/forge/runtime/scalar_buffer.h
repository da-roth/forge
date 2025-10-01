#pragma once

#include "forge/runtime/node_buffer.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <iostream>

namespace forge {
namespace runtime {

/**
 * Scalar implementation of NodeBuffer
 * Memory layout: values[nodeId] contains one double per node
 */
class ScalarNodeBuffer : public INodeBuffer {
public:
    explicit ScalarNodeBuffer(const forge::core::ComputationGraph& tape) 
        : num_nodes_(tape.nodes.size()) {
        diff_inputs_ = tape.diff_inputs;
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
        
        values_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 64));
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, totalDoubles * sizeof(double));
        
        // Allocate gradients if needed
        if (!diff_inputs_.empty()) {
            gradients_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 64));
            std::memset(gradients_, 0, totalDoubles * sizeof(double));
        }
    }
    
    // Constructor with node ID mapping
    ScalarNodeBuffer(const forge::core::ComputationGraph& tape, 
                          const std::vector<forge::core::NodeId>& originalToOptimizedMapping)
        : num_nodes_(tape.nodes.size()), originalToOptimizedMapping_(originalToOptimizedMapping) {
        diff_inputs_ = tape.diff_inputs;
        
        // Allocate values - one double per node
        size_t totalDoubles = num_nodes_;
        
        // Safety check: ensure we allocate at least some memory
        if (totalDoubles == 0) {
            totalDoubles = 1;  // At least one double
        }
        
        values_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 64));
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, totalDoubles * sizeof(double));
        
        // Allocate gradients if needed
        if (!diff_inputs_.empty()) {
            gradients_ = static_cast<double*>(_aligned_malloc(totalDoubles * sizeof(double), 64));
            std::memset(gradients_, 0, totalDoubles * sizeof(double));
        }
    }
    
    ~ScalarNodeBuffer() {
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
    double getGradient(forge::core::NodeId node) const override {
        if (!gradients_) {
            throw std::runtime_error("No gradients computed - no inputs marked with markInputAndDiff()");
        }
        
        // Map original node ID to optimized if mapping is available
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
        
        return gradients_[mappedNode];
    }
    
    std::vector<double> getVectorGradient(forge::core::NodeId node) const override {
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
    ScalarNodeBuffer(const ScalarNodeBuffer&) = delete;
    ScalarNodeBuffer& operator=(const ScalarNodeBuffer&) = delete;
    
    // Enable move
    ScalarNodeBuffer(ScalarNodeBuffer&& other) noexcept 
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
    std::vector<forge::core::NodeId> diff_inputs_;
    std::vector<forge::core::NodeId> originalToOptimizedMapping_;
};

} // namespace runtime
} // namespace forge