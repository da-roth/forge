#pragma once

#include "../../../interfaces/node_value_buffer.hpp"
#include <immintrin.h>  // For AVX2 intrinsics

namespace forge {

/**
 * AVX2 implementation of NodeValueBuffer
 * Memory layout: values[nodeId * 4] contains 4 doubles per node (YMM register width)
 *
 * Uses NodeValueBufferBase<4, 32> for most functionality.
 * 32-byte alignment for AVX2 aligned loads/stores.
 *
 * Overrides getGradientLanes() with AVX2-optimized version using _mm256_load_pd.
 */
class AVX2NodeValueBuffer : public NodeValueBufferBase<4, 32> {
public:
    // Constructor delegates to base class
    AVX2NodeValueBuffer(const forge::Graph& tape,
                        const std::vector<forge::NodeId>& originalToOptimizedMapping,
                        size_t requiredNodes)
        : NodeValueBufferBase<4, 32>(tape, originalToOptimizedMapping, requiredNodes) {
    }

    // Override getGradientLanes with AVX2-optimized version
    void getGradientLanes(const std::vector<size_t>& bufferIndices, double* output) const override {
        if (!gradients_) return;

        for (size_t i = 0; i < bufferIndices.size(); ++i) {
            size_t baseIdx = bufferIndices[i];
            // Load 4 contiguous doubles and store using AVX2
            __m256d grads = _mm256_load_pd(&gradients_[baseIdx]);
            _mm256_storeu_pd(&output[i * VECTOR_WIDTH], grads);
        }
    }

    // All other functionality inherited from NodeValueBufferBase<4, 32>
};

} // namespace forge
