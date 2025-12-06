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
#include <chrono>  // For timing instrumentation
#include <immintrin.h>  // For AVX2 intrinsics
#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc and _aligned_free on Windows
#endif

// Timing instrumentation for AVX2 buffer operations
struct AVX2BufferTiming {
    static inline double setInputsLoopNs = 0.0;
    static inline double setInputsIntrinsicsNs = 0.0;
    static inline double getGradientsLoopNs = 0.0;
    static inline double getGradientsLoadNs = 0.0;
    static inline double getGradientsExtractNs = 0.0;
    static inline size_t setInputsCalls = 0;
    static inline size_t getGradientsCalls = 0;

    static void reset() {
        setInputsLoopNs = setInputsIntrinsicsNs = 0.0;
        getGradientsLoopNs = getGradientsLoadNs = getGradientsExtractNs = 0.0;
        setInputsCalls = getGradientsCalls = 0;
    }

    static void report() {
        std::cerr << "[AVX2 BUFFER TIMING] SetInputs calls: " << setInputsCalls
                  << " | Loop: " << (setInputsLoopNs / 1000.0) << " us"
                  << " | Intrinsics: " << (setInputsIntrinsicsNs / 1000.0) << " us" << std::endl;
        std::cerr << "[AVX2 BUFFER TIMING] GetGradients calls: " << getGradientsCalls
                  << " | Loop: " << (getGradientsLoopNs / 1000.0) << " us"
                  << " | Load: " << (getGradientsLoadNs / 1000.0) << " us"
                  << " | Extract: " << (getGradientsExtractNs / 1000.0) << " us" << std::endl;
    }
};

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

        // Calculate allocation size - must be multiple of alignment for aligned_alloc on Linux
        size_t allocSize = totalDoubles * sizeof(double);
        size_t alignedAllocSize = (allocSize + 31) & ~size_t(31);  // Round up to multiple of 32

        // Platform-specific aligned allocation
#ifdef _WIN32
        values_ = static_cast<double*>(_aligned_malloc(allocSize, 32));  // 32-byte alignment for AVX2
#else
        values_ = static_cast<double*>(aligned_alloc(32, alignedAllocSize));
#endif
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, allocSize);

        // Allocate gradients if needed (also 4 doubles per node)
        if (!diff_inputs_.empty()) {
#ifdef _WIN32
            gradients_ = static_cast<double*>(_aligned_malloc(allocSize, 32));
#else
            gradients_ = static_cast<double*>(aligned_alloc(32, alignedAllocSize));
#endif
            if (gradients_) {
                std::memset(gradients_, 0, allocSize);
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

        // Allocate values - 4 doubles per node for AVX2
        size_t totalDoubles = num_nodes_ * vector_width_;

        // Safety check: ensure we allocate at least some memory
        if (totalDoubles == 0) {
            totalDoubles = vector_width_;  // At least one YMM register worth
        }

        // Calculate allocation size - must be multiple of alignment for aligned_alloc on Linux
        size_t allocSize = totalDoubles * sizeof(double);
        size_t alignedAllocSize = (allocSize + 31) & ~size_t(31);  // Round up to multiple of 32

        // Platform-specific aligned allocation
#ifdef _WIN32
        values_ = static_cast<double*>(_aligned_malloc(allocSize, 32));  // 32-byte alignment for AVX2
#else
        values_ = static_cast<double*>(aligned_alloc(32, alignedAllocSize));
#endif
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, allocSize);

        // Allocate gradients if needed (also 4 doubles per node)
        if (!tape.diff_inputs.empty()) {
#ifdef _WIN32
            gradients_ = static_cast<double*>(_aligned_malloc(allocSize, 32));
#else
            gradients_ = static_cast<double*>(aligned_alloc(32, alignedAllocSize));
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

        // Calculate allocation size - must be multiple of alignment for aligned_alloc on Linux
        size_t allocSize = totalDoubles * sizeof(double);
        size_t alignedAllocSize = (allocSize + 31) & ~size_t(31);  // Round up to multiple of 32

        // Platform-specific aligned allocation
#ifdef _WIN32
        values_ = static_cast<double*>(_aligned_malloc(allocSize, 32));  // 32-byte alignment for AVX2
#else
        values_ = static_cast<double*>(aligned_alloc(32, alignedAllocSize));
#endif
        if (!values_) {
            throw std::bad_alloc();
        }
        std::memset(values_, 0, allocSize);

        // Allocate gradients if needed (also 4 doubles per node)
        if (!diff_inputs_.empty()) {
#ifdef _WIN32
            gradients_ = static_cast<double*>(_aligned_malloc(allocSize, 32));
#else
            gradients_ = static_cast<double*>(aligned_alloc(32, alignedAllocSize));
#endif
            if (gradients_) {
                std::memset(gradients_, 0, allocSize);
            }
        }
    }

    ~AVX2NodeValueBuffer() {
        // DISABLED: Timing report (adds noise to output)
        // if (AVX2BufferTiming::setInputsCalls > 0 || AVX2BufferTiming::getGradientsCalls > 0) {
        //     AVX2BufferTiming::report();
        //     AVX2BufferTiming::reset();
        // }

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
                size_t baseIdx = optimizedNodeId * vector_width_;
                // Direct memcpy - 32 bytes (4 doubles)
                std::memcpy(&values_[baseIdx], values, vector_width_ * sizeof(double));
            }
        }
    }

    void getLanes(uint64_t nodeId, double* output) const override {
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                size_t baseIdx = optimizedNodeId * vector_width_;
                // Direct memcpy - 32 bytes (4 doubles)
                std::memcpy(output, &values_[baseIdx], vector_width_ * sizeof(double));
            }
        }
    }

    void getGradientLanes(const std::vector<size_t>& bufferIndices, double* output) const override {
        if (!gradients_) return;

        for (size_t i = 0; i < bufferIndices.size(); ++i) {
            size_t baseIdx = bufferIndices[i];
            // Load 4 contiguous doubles and store interleaved
            __m256d grads = _mm256_load_pd(&gradients_[baseIdx]);
            _mm256_storeu_pd(&output[i * vector_width_], grads);
        }
    }

    void getGradientLanesSeparate(const std::vector<size_t>& bufferIndices, double* outputs[4]) const override {
        if (!gradients_) return;

        for (size_t i = 0; i < bufferIndices.size(); ++i) {
            size_t baseIdx = bufferIndices[i];
            // Load 4 contiguous doubles
            __m256d grads = _mm256_load_pd(&gradients_[baseIdx]);
            // Extract to 4 different output arrays (only write to non-null pointers)
            __m128d lo = _mm256_castpd256_pd128(grads);      // lanes 0, 1
            __m128d hi = _mm256_extractf128_pd(grads, 1);    // lanes 2, 3
            if (outputs[0]) outputs[0][i] = _mm_cvtsd_f64(lo);               // lane 0
            if (outputs[1]) outputs[1][i] = _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo));  // lane 1
            if (outputs[2]) outputs[2][i] = _mm_cvtsd_f64(hi);               // lane 2
            if (outputs[3]) outputs[3][i] = _mm_cvtsd_f64(_mm_unpackhi_pd(hi, hi));  // lane 3
        }
    }

    // ==========================================================================
    // DEPRECATED API: Convenience wrappers (internally use Lanes)
    // ==========================================================================

    void setValue(uint64_t nodeId, double value) override {
        // Broadcast to all 4 lanes
        double data[4] = {value, value, value, value};
        setLanes(nodeId, data);
    }

    double getValue(uint64_t nodeId) const override {
        double data[4];
        getLanes(nodeId, data);
        return data[0];  // Return lane 0
    }

    size_t getBufferIndex(uint64_t nodeId) const override {
        if (nodeId < originalToOptimizedMapping_.size()) {
            uint64_t optimizedNodeId = originalToOptimizedMapping_[nodeId];
            if (optimizedNodeId != static_cast<uint64_t>(-1) && optimizedNodeId < num_nodes_) {
                return optimizedNodeId * vector_width_;
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

        // Use getGradientLanes internally
        size_t bufferIdx = mappedNode * vector_width_;
        std::vector<size_t> indices = {bufferIdx};
        double lanes[4];
        getGradientLanes(indices, lanes);
        return lanes[0];
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
