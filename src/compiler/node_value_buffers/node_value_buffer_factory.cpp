#include "node_value_buffer.hpp"
#include "scalar_node_value_buffer.hpp"
#include "avx2_node_value_buffer.hpp"
#include "../../compiler/forge_engine.hpp"

namespace forge {

std::unique_ptr<INodeValueBuffer> NodeValueBufferFactory::create(
    const forge::Graph& tape,
    const StitchedKernel& kernel) {
    
    int vectorWidth = kernel.getVectorWidth();
    
    // Use the mapping from the kernel
    const auto& mapping = kernel.getOriginalToOptimizedMapping();
    
    // Create a temporary optimized tape with the correct size for the buffer
    // The buffer needs to have slots for all optimized node IDs
    forge::Graph optimizedTape;
    size_t requiredNodes = kernel.getRequiredNodes();
    optimizedTape.nodes.resize(requiredNodes);
    optimizedTape.outputs = tape.outputs; // Keep original outputs for mapping
    
    // Propagate and map diff_inputs using the kernel mapping
    optimizedTape.diff_inputs.clear();
    optimizedTape.diff_inputs.reserve(tape.diff_inputs.size());
    for (auto origId : tape.diff_inputs) {
        if (origId < mapping.size()) {
            auto mapped = mapping[origId];
            if (mapped != static_cast<forge::NodeId>(UINT32_MAX) &&
                mapped < kernel.getRequiredNodes()) {
                optimizedTape.diff_inputs.push_back(mapped);
            }
        }
    }
    
    if (vectorWidth == 1) {
        return std::make_unique<ScalarNodeValueBuffer>(optimizedTape, mapping);
    } else if (vectorWidth == 4) {
        // Use the new constructor that takes exact kernel size for proper propagation
        return std::make_unique<AVX2NodeValueBuffer>(optimizedTape, mapping, requiredNodes);
    } else {
        throw std::runtime_error("Unsupported vector width: " + std::to_string(vectorWidth));
    }
}

std::unique_ptr<INodeValueBuffer> NodeValueBufferFactory::create(
    const forge::Graph& tape,
    const StitchedKernel& kernel,
    const std::vector<forge::NodeId>& originalToOptimizedMapping) {
    
    int vectorWidth = kernel.getVectorWidth();

    // Create a temporary optimized tape with the correct size for the buffer
    // The buffer needs to have slots for all optimized node IDs
    forge::Graph optimizedTape;
    size_t requiredNodes = kernel.getRequiredNodes();
    optimizedTape.nodes.resize(requiredNodes);
    optimizedTape.outputs = tape.outputs; // Keep original outputs for mapping
    
    // Propagate and map diff_inputs using the provided mapping
    optimizedTape.diff_inputs.clear();
    optimizedTape.diff_inputs.reserve(tape.diff_inputs.size());
    for (auto origId : originalToOptimizedMapping) {
        // originalToOptimizedMapping is sized to original tape; but we need to use tape.diff_inputs indices
    }
    // Correct loop over tape.diff_inputs with mapping
    optimizedTape.diff_inputs.clear();
    optimizedTape.diff_inputs.reserve(tape.diff_inputs.size());
    for (auto origId : tape.diff_inputs) {
        if (origId < originalToOptimizedMapping.size()) {
            auto mapped = originalToOptimizedMapping[origId];
            if (mapped != static_cast<forge::NodeId>(UINT32_MAX) &&
                mapped < kernel.getRequiredNodes()) {
                optimizedTape.diff_inputs.push_back(mapped);
            }
        }
    }
    
    if (vectorWidth == 1) {
        return std::make_unique<ScalarNodeValueBuffer>(optimizedTape, originalToOptimizedMapping);
    } else if (vectorWidth == 4) {
        // Use the new constructor that takes exact kernel size for proper propagation
        return std::make_unique<AVX2NodeValueBuffer>(optimizedTape, originalToOptimizedMapping, requiredNodes);
    } else {
        throw std::runtime_error("Unsupported vector width: " + std::to_string(vectorWidth));
    }
}

} // namespace forge