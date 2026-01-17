#pragma once

#include "../../../interfaces/node_value_buffer.hpp"

namespace forge {

/**
 * Scalar implementation of NodeValueBuffer
 * Memory layout: values[nodeId] contains one double per node
 *
 * Uses NodeValueBufferBase<1, 64> for all functionality.
 * 64-byte alignment for cache line efficiency.
 */
class ScalarNodeValueBuffer : public NodeValueBufferBase<1, 64> {
public:
    // Constructor delegates to base class
    ScalarNodeValueBuffer(const forge::Graph& tape,
                          const std::vector<forge::NodeId>& originalToOptimizedMapping)
        : NodeValueBufferBase<1, 64>(tape, originalToOptimizedMapping, tape.nodes.size()) {
    }

    // All functionality inherited from NodeValueBufferBase<1, 64>
    // No overrides needed - base class handles width=1 optimizations via constexpr if
};

} // namespace forge
