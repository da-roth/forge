#include "../../interfaces/node_value_buffer.hpp"
#include "../double/scalar/scalar_node_value_buffer.hpp"
#include "../../forge_engine.hpp"

namespace forge {

// Registry for buffer creators (allows AVX2 to register without header dependency)
namespace {
    // Function signature for creating buffers
    using BufferCreatorFunc = std::unique_ptr<INodeValueBuffer>(*)(
        const forge::Graph& optimizedTape,
        const std::vector<forge::NodeId>& mapping,
        size_t requiredNodes);

    // Registry of buffer creators by vector width
    BufferCreatorFunc g_avx2BufferCreator = nullptr;
}

// Called by AVX2 static registration to register the AVX2 buffer creator
void NodeValueBufferFactory::registerAVX2BufferCreator(
    std::unique_ptr<INodeValueBuffer>(*creator)(
        const forge::Graph&,
        const std::vector<forge::NodeId>&,
        size_t)) {
    g_avx2BufferCreator = creator;
}

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
        if (g_avx2BufferCreator) {
            return g_avx2BufferCreator(optimizedTape, mapping, requiredNodes);
        }
        throw std::runtime_error("AVX2 buffer creator not registered. "
                                 "Bundle AVX2 or load AVX2 backend at runtime first.");
    } else {
        throw std::runtime_error("Unsupported vector width: " + std::to_string(vectorWidth));
    }
}

} // namespace forge