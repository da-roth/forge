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

} // namespace forge