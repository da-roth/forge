#include "../../interfaces/node_value_buffer.hpp"
#include "../double/scalar/scalar_node_value_buffer.hpp"
#include "../../forge_engine.hpp"
#include <unordered_map>

namespace forge {

// Registry for buffer creators by vector width
namespace {
    std::unordered_map<int, NodeValueBufferFactory::BufferCreatorFunc>& getBufferCreatorRegistry() {
        static std::unordered_map<int, NodeValueBufferFactory::BufferCreatorFunc> registry;
        return registry;
    }
}

void NodeValueBufferFactory::registerBufferCreator(int vectorWidth, BufferCreatorFunc creator) {
    getBufferCreatorRegistry()[vectorWidth] = creator;
}

bool NodeValueBufferFactory::hasBufferCreator(int vectorWidth) {
    auto& registry = getBufferCreatorRegistry();
    return registry.find(vectorWidth) != registry.end();
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

    // Scalar (vectorWidth == 1) is always available
    if (vectorWidth == 1) {
        return std::make_unique<ScalarNodeValueBuffer>(optimizedTape, mapping);
    }

    // Look up registered buffer creator for this vector width
    auto& registry = getBufferCreatorRegistry();
    auto it = registry.find(vectorWidth);
    if (it != registry.end() && it->second != nullptr) {
        return it->second(optimizedTape, mapping, requiredNodes);
    }

    throw std::runtime_error(
        "No buffer creator registered for vector width " + std::to_string(vectorWidth) +
        ". Ensure the appropriate backend is bundled or loaded at runtime.");
}

} // namespace forge