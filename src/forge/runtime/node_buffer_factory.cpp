#include "forge/runtime/node_buffer.h"
#include "forge/runtime/scalar_buffer.h"
#include "forge/runtime/avx2_buffer.h"
#include "forge/runtime/compiled_kernel.h"
#include <iostream>
#include <cstdio>

namespace forge {
namespace runtime {

// New primary implementation using KernelRequirements
std::unique_ptr<INodeBuffer> NodeBufferFactory::create(
    const forge::core::ComputationGraph& tape,
    const KernelRequirements& requirements) {
    
    // Create a temporary optimized tape with the correct size for the buffer
    forge::core::ComputationGraph optimizedTape;
    optimizedTape.nodes.resize(requirements.requiredNodes);
    optimizedTape.outputs = tape.outputs; // Keep original outputs for mapping
    
    // Convert uint32_t mapping back to NodeId for compatibility
    std::vector<forge::core::NodeId> nodeIdMapping;
    nodeIdMapping.reserve(requirements.nodeMapping.size());
    for (uint32_t id : requirements.nodeMapping) {
        nodeIdMapping.push_back(static_cast<forge::core::NodeId>(id));
    }
    
    // Propagate and map diff_inputs using the kernel requirements mapping
    optimizedTape.diff_inputs.clear();
    optimizedTape.diff_inputs.reserve(tape.diff_inputs.size());
    for (auto origId : tape.diff_inputs) {
        if (origId < nodeIdMapping.size()) {
            auto mapped = nodeIdMapping[origId];
            if (mapped != static_cast<forge::core::NodeId>(UINT32_MAX) &&
                mapped < requirements.requiredNodes) {
                optimizedTape.diff_inputs.push_back(mapped);
            }
        }
    }
    
    // Create appropriate buffer based on vector width
    if (requirements.vectorWidth == 1) {
        return std::make_unique<ScalarNodeBuffer>(optimizedTape, nodeIdMapping);
    } else if (requirements.vectorWidth == 4) {
        // Use the new constructor that takes exact kernel size for proper propagation
        return std::make_unique<AVX2NodeBuffer>(optimizedTape, nodeIdMapping, requirements.requiredNodes);
    } else {
        throw std::runtime_error("Unsupported vector width: " + std::to_string(requirements.vectorWidth));
    }
}

// Legacy method - delegates to new implementation
std::unique_ptr<INodeBuffer> NodeBufferFactory::create(
    const forge::core::ComputationGraph& tape,
    const CompiledKernel& kernel) {
    
    // Get requirements from kernel and delegate to new method
    return create(tape, kernel.getRequirements());
}

// Legacy method with explicit mapping - still delegates to new implementation
std::unique_ptr<INodeBuffer> NodeBufferFactory::create(
    const forge::core::ComputationGraph& tape,
    const CompiledKernel& kernel,
    const std::vector<forge::core::NodeId>& originalToOptimizedMapping) {
    
    // Get requirements from kernel but override the mapping
    KernelRequirements requirements = kernel.getRequirements();
    
    // Convert the provided NodeId mapping to uint32_t
    requirements.nodeMapping.clear();
    requirements.nodeMapping.reserve(originalToOptimizedMapping.size());
    for (auto nodeId : originalToOptimizedMapping) {
        requirements.nodeMapping.push_back(static_cast<uint32_t>(nodeId));
    }
    
    return create(tape, requirements);
}

} // namespace runtime
} // namespace forge