/*
 * Forge C API Implementation
 *
 * Wraps the C++ Forge library in a stable C ABI interface.
 *
 * SPDX-License-Identifier: Zlib
 */

#define FORGE_CAPI_BUILDING

#include "forge_c_api.h"

#include <graph/graph.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/x86/common/compiler_config.hpp>
#include <compiler/interfaces/node_value_buffer.hpp>

#include <memory>
#include <string>
#include <cstring>

// Thread-local error message storage
static thread_local std::string g_last_error;

// Version information
#ifndef FORGE_VERSION_MAJOR
#define FORGE_VERSION_MAJOR 0
#endif
#ifndef FORGE_VERSION_MINOR
#define FORGE_VERSION_MINOR 1
#endif
#ifndef FORGE_VERSION_PATCH
#define FORGE_VERSION_PATCH 0
#endif

// ==========================================================================
// Internal wrapper structures
// ==========================================================================

struct ForgeGraph_ {
    forge::Graph graph;
};

struct ForgeConfig_ {
    forge::CompilerConfig config;
};

struct ForgeKernel_ {
    std::unique_ptr<forge::StitchedKernel> kernel;
};

struct ForgeBuffer_ {
    std::unique_ptr<forge::INodeValueBuffer> buffer;
    std::vector<size_t> bufferIndexCache;  // Cache for gradient retrieval
};

// ==========================================================================
// Helper functions
// ==========================================================================

static void set_error(const char* msg) {
    g_last_error = msg ? msg : "Unknown error";
}

static void set_error(const std::string& msg) {
    g_last_error = msg;
}

static forge::OpCode to_forge_opcode(ForgeOpCode op) {
    // Direct cast since enum values match
    return static_cast<forge::OpCode>(op);
}

// ==========================================================================
// Error handling
// ==========================================================================

extern "C" {

FORGE_API const char* forge_error_string(ForgeError error) {
    switch (error) {
        case FORGE_SUCCESS: return "Success";
        case FORGE_ERROR_NULL_HANDLE: return "Null handle";
        case FORGE_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case FORGE_ERROR_COMPILATION_FAILED: return "Compilation failed";
        case FORGE_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case FORGE_ERROR_INDEX_OUT_OF_RANGE: return "Index out of range";
        case FORGE_ERROR_NOT_COMPILED: return "Not compiled";
        default: return "Unknown error";
    }
}

FORGE_API const char* forge_get_last_error(void) {
    return g_last_error.c_str();
}

// ==========================================================================
// Graph API
// ==========================================================================

FORGE_API ForgeGraphHandle forge_graph_create(void) {
    try {
        return new ForgeGraph_();
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("Unknown exception");
        return nullptr;
    }
}

FORGE_API void forge_graph_destroy(ForgeGraphHandle graph) {
    delete graph;
}

FORGE_API ForgeError forge_graph_clear(ForgeGraphHandle graph) {
    if (!graph) {
        set_error("Null graph handle");
        return FORGE_ERROR_NULL_HANDLE;
    }
    graph->graph.clear();
    return FORGE_SUCCESS;
}

FORGE_API size_t forge_graph_node_count(ForgeGraphHandle graph) {
    if (!graph) return 0;
    return graph->graph.size();
}

FORGE_API uint32_t forge_graph_add_input(ForgeGraphHandle graph) {
    if (!graph) {
        set_error("Null graph handle");
        return UINT32_MAX;
    }
    try {
        return graph->graph.addInput();
    } catch (const std::exception& e) {
        set_error(e.what());
        return UINT32_MAX;
    }
}

FORGE_API uint32_t forge_graph_add_constant(ForgeGraphHandle graph, double value) {
    if (!graph) {
        set_error("Null graph handle");
        return UINT32_MAX;
    }
    try {
        return graph->graph.addConstant(value);
    } catch (const std::exception& e) {
        set_error(e.what());
        return UINT32_MAX;
    }
}

FORGE_API uint32_t forge_graph_add_unary(ForgeGraphHandle graph, ForgeOpCode op, uint32_t a) {
    if (!graph) {
        set_error("Null graph handle");
        return UINT32_MAX;
    }
    try {
        forge::Node node;
        node.op = to_forge_opcode(op);
        node.a = a;
        node.b = 0;
        node.c = 0;
        node.imm = 0.0;
        node.isActive = true;
        node.isDead = false;
        node.needsGradient = false;
        return graph->graph.addNode(node);
    } catch (const std::exception& e) {
        set_error(e.what());
        return UINT32_MAX;
    }
}

FORGE_API uint32_t forge_graph_add_binary(ForgeGraphHandle graph, ForgeOpCode op, uint32_t a, uint32_t b) {
    if (!graph) {
        set_error("Null graph handle");
        return UINT32_MAX;
    }
    try {
        forge::Node node;
        node.op = to_forge_opcode(op);
        node.a = a;
        node.b = b;
        node.c = 0;
        node.imm = 0.0;
        node.isActive = true;
        node.isDead = false;
        node.needsGradient = false;
        return graph->graph.addNode(node);
    } catch (const std::exception& e) {
        set_error(e.what());
        return UINT32_MAX;
    }
}

FORGE_API uint32_t forge_graph_add_ternary(ForgeGraphHandle graph, ForgeOpCode op, uint32_t a, uint32_t b, uint32_t c) {
    if (!graph) {
        set_error("Null graph handle");
        return UINT32_MAX;
    }
    try {
        forge::Node node;
        node.op = to_forge_opcode(op);
        node.a = a;
        node.b = b;
        node.c = c;
        node.imm = 0.0;
        node.isActive = true;
        node.isDead = false;
        node.needsGradient = false;
        return graph->graph.addNode(node);
    } catch (const std::exception& e) {
        set_error(e.what());
        return UINT32_MAX;
    }
}

FORGE_API uint32_t forge_graph_add_node(
    ForgeGraphHandle graph,
    ForgeOpCode op,
    uint32_t a,
    uint32_t b,
    uint32_t c,
    double imm,
    int is_active,
    int needs_gradient)
{
    if (!graph) {
        set_error("Null graph handle");
        return UINT32_MAX;
    }
    try {
        forge::Node node;
        node.op = to_forge_opcode(op);
        node.a = a;
        node.b = b;
        node.c = c;
        node.imm = imm;
        node.isActive = is_active != 0;
        node.isDead = false;
        node.needsGradient = needs_gradient != 0;
        return graph->graph.addNode(node);
    } catch (const std::exception& e) {
        set_error(e.what());
        return UINT32_MAX;
    }
}

FORGE_API ForgeError forge_graph_mark_output(ForgeGraphHandle graph, uint32_t node_id) {
    if (!graph) {
        set_error("Null graph handle");
        return FORGE_ERROR_NULL_HANDLE;
    }
    if (node_id >= graph->graph.size()) {
        set_error("Node ID out of range");
        return FORGE_ERROR_INDEX_OUT_OF_RANGE;
    }
    graph->graph.markOutput(node_id);
    return FORGE_SUCCESS;
}

FORGE_API ForgeError forge_graph_mark_diff_input(ForgeGraphHandle graph, uint32_t node_id) {
    if (!graph) {
        set_error("Null graph handle");
        return FORGE_ERROR_NULL_HANDLE;
    }
    if (node_id >= graph->graph.size()) {
        set_error("Node ID out of range");
        return FORGE_ERROR_INDEX_OUT_OF_RANGE;
    }
    graph->graph.diff_inputs.push_back(node_id);
    return FORGE_SUCCESS;
}

FORGE_API size_t forge_graph_output_count(ForgeGraphHandle graph) {
    if (!graph) return 0;
    return graph->graph.outputs.size();
}

FORGE_API size_t forge_graph_diff_input_count(ForgeGraphHandle graph) {
    if (!graph) return 0;
    return graph->graph.diff_inputs.size();
}

FORGE_API ForgeError forge_graph_propagate_gradients(ForgeGraphHandle graph) {
    if (!graph) {
        set_error("Null graph handle");
        return FORGE_ERROR_NULL_HANDLE;
    }

    auto& nodes = graph->graph.nodes;

    // First, mark all diff_input nodes as needing gradients
    for (auto inputId : graph->graph.diff_inputs) {
        if (inputId < nodes.size()) {
            nodes[inputId].needsGradient = true;
        }
    }

    // Forward propagation: if any operand needs gradient, result needs gradient
    // This matches the C++ ForgeBackend implementation exactly
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto& node = nodes[i];
        if (node.isDead) continue;

        bool operandNeedsGrad = false;
        if (node.a < nodes.size())
            operandNeedsGrad |= nodes[node.a].needsGradient;
        if (node.b < nodes.size())
            operandNeedsGrad |= nodes[node.b].needsGradient;
        if (node.c < nodes.size())
            operandNeedsGrad |= nodes[node.c].needsGradient;

        // Only set needsGradient if the node is active
        // (constants are inactive and should not have needsGradient=true)
        if (operandNeedsGrad && node.isActive)
            node.needsGradient = true;
    }

    return FORGE_SUCCESS;
}

// ==========================================================================
// Compiler Configuration API
// ==========================================================================

FORGE_API ForgeConfigHandle forge_config_create_default(void) {
    try {
        auto* handle = new ForgeConfig_();
        handle->config = forge::CompilerConfig::Default();
        return handle;
    } catch (...) {
        set_error("Failed to create config");
        return nullptr;
    }
}

FORGE_API ForgeConfigHandle forge_config_create_debug(void) {
    try {
        auto* handle = new ForgeConfig_();
        handle->config = forge::CompilerConfig::Debug();
        return handle;
    } catch (...) {
        set_error("Failed to create config");
        return nullptr;
    }
}

FORGE_API ForgeConfigHandle forge_config_create_fast(void) {
    try {
        auto* handle = new ForgeConfig_();
        handle->config = forge::CompilerConfig::Fast();
        return handle;
    } catch (...) {
        set_error("Failed to create config");
        return nullptr;
    }
}

FORGE_API ForgeConfigHandle forge_config_create_no_opt(void) {
    try {
        auto* handle = new ForgeConfig_();
        handle->config = forge::CompilerConfig::NoOptimization();
        return handle;
    } catch (...) {
        set_error("Failed to create config");
        return nullptr;
    }
}

FORGE_API void forge_config_destroy(ForgeConfigHandle config) {
    delete config;
}

FORGE_API ForgeError forge_config_set_instruction_set(ForgeConfigHandle config, ForgeInstructionSet instruction_set) {
    if (!config) {
        set_error("Null config handle");
        return FORGE_ERROR_NULL_HANDLE;
    }
    switch (instruction_set) {
        case FORGE_INSTRUCTION_SET_SSE2_SCALAR:
            config->config.instructionSet = forge::CompilerConfig::InstructionSet::SSE2_SCALAR;
            break;
        case FORGE_INSTRUCTION_SET_AVX2_PACKED:
            config->config.instructionSet = forge::CompilerConfig::InstructionSet::AVX2_PACKED;
            break;
        default:
            set_error("Invalid instruction set");
            return FORGE_ERROR_INVALID_ARGUMENT;
    }
    return FORGE_SUCCESS;
}

FORGE_API ForgeError forge_config_set_optimizations(ForgeConfigHandle config, int enable) {
    if (!config) return FORGE_ERROR_NULL_HANDLE;
    config->config.enableOptimizations = enable != 0;
    return FORGE_SUCCESS;
}

FORGE_API ForgeError forge_config_set_cse(ForgeConfigHandle config, int enable) {
    if (!config) return FORGE_ERROR_NULL_HANDLE;
    config->config.enableCSE = enable != 0;
    return FORGE_SUCCESS;
}

FORGE_API ForgeError forge_config_set_algebraic_simplification(ForgeConfigHandle config, int enable) {
    if (!config) return FORGE_ERROR_NULL_HANDLE;
    config->config.enableAlgebraicSimplification = enable != 0;
    return FORGE_SUCCESS;
}

FORGE_API ForgeError forge_config_set_stability_cleaning(ForgeConfigHandle config, int enable) {
    if (!config) return FORGE_ERROR_NULL_HANDLE;
    config->config.enableStabilityCleaning = enable != 0;
    return FORGE_SUCCESS;
}

// ==========================================================================
// Compilation API
// ==========================================================================

FORGE_API ForgeKernelHandle forge_compile(ForgeGraphHandle graph, ForgeConfigHandle config) {
    if (!graph) {
        set_error("Null graph handle");
        return nullptr;
    }

    try {
        forge::CompilerConfig cfg = config ? config->config : forge::CompilerConfig::Default();
        forge::ForgeEngine engine(cfg);

        auto kernel = engine.compile(graph->graph);
        if (!kernel) {
            set_error("Compilation returned null kernel");
            return nullptr;
        }

        auto* handle = new ForgeKernel_();
        handle->kernel = std::move(kernel);
        return handle;
    } catch (const std::exception& e) {
        set_error(std::string("Compilation failed: ") + e.what());
        return nullptr;
    } catch (...) {
        set_error("Compilation failed with unknown exception");
        return nullptr;
    }
}

FORGE_API void forge_kernel_destroy(ForgeKernelHandle kernel) {
    delete kernel;
}

FORGE_API int forge_kernel_get_vector_width(ForgeKernelHandle kernel) {
    if (!kernel || !kernel->kernel) return 0;
    return kernel->kernel->getVectorWidth();
}

FORGE_API size_t forge_kernel_get_required_nodes(ForgeKernelHandle kernel) {
    if (!kernel || !kernel->kernel) return 0;
    return kernel->kernel->getRequiredNodes();
}

// ==========================================================================
// Buffer API
// ==========================================================================

FORGE_API ForgeBufferHandle forge_buffer_create(ForgeGraphHandle graph, ForgeKernelHandle kernel) {
    if (!graph || !kernel || !kernel->kernel) {
        set_error("Null handle");
        return nullptr;
    }

    try {
        auto buffer = forge::NodeValueBufferFactory::create(graph->graph, *kernel->kernel);
        if (!buffer) {
            set_error("Buffer creation failed");
            return nullptr;
        }

        auto* handle = new ForgeBuffer_();
        handle->buffer = std::move(buffer);
        return handle;
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

FORGE_API void forge_buffer_destroy(ForgeBufferHandle buffer) {
    delete buffer;
}

FORGE_API ForgeError forge_buffer_set_lanes(ForgeBufferHandle buffer, uint32_t node_id, const double* values) {
    if (!buffer || !buffer->buffer) return FORGE_ERROR_NULL_HANDLE;
    if (!values) return FORGE_ERROR_INVALID_ARGUMENT;

    try {
        buffer->buffer->setLanes(node_id, values);
        return FORGE_SUCCESS;
    } catch (...) {
        return FORGE_ERROR_INDEX_OUT_OF_RANGE;
    }
}

FORGE_API ForgeError forge_buffer_set_value(ForgeBufferHandle buffer, uint32_t node_id, double value) {
    if (!buffer || !buffer->buffer) return FORGE_ERROR_NULL_HANDLE;

    try {
        // Broadcast to all lanes
        int width = buffer->buffer->getVectorWidth();
        double values[8] = {value, value, value, value, value, value, value, value};
        buffer->buffer->setLanes(node_id, values);
        return FORGE_SUCCESS;
    } catch (...) {
        return FORGE_ERROR_INDEX_OUT_OF_RANGE;
    }
}

FORGE_API ForgeError forge_buffer_get_lanes(ForgeBufferHandle buffer, uint32_t node_id, double* output) {
    if (!buffer || !buffer->buffer) return FORGE_ERROR_NULL_HANDLE;
    if (!output) return FORGE_ERROR_INVALID_ARGUMENT;

    try {
        buffer->buffer->getLanes(node_id, output);
        return FORGE_SUCCESS;
    } catch (...) {
        return FORGE_ERROR_INDEX_OUT_OF_RANGE;
    }
}

FORGE_API ForgeError forge_buffer_get_value(ForgeBufferHandle buffer, uint32_t node_id, double* output) {
    if (!buffer || !buffer->buffer) return FORGE_ERROR_NULL_HANDLE;
    if (!output) return FORGE_ERROR_INVALID_ARGUMENT;

    try {
        int width = buffer->buffer->getVectorWidth();
        double values[8];
        buffer->buffer->getLanes(node_id, values);
        *output = values[0];
        return FORGE_SUCCESS;
    } catch (...) {
        return FORGE_ERROR_INDEX_OUT_OF_RANGE;
    }
}

FORGE_API ForgeError forge_buffer_get_gradient(ForgeBufferHandle buffer, uint32_t node_id, double* output) {
    if (!buffer || !buffer->buffer) return FORGE_ERROR_NULL_HANDLE;
    if (!output) return FORGE_ERROR_INVALID_ARGUMENT;

    try {
        size_t idx = buffer->buffer->getBufferIndex(node_id);
        double* grads = buffer->buffer->getGradientsPtr();
        if (grads) {
            *output = grads[idx];
        } else {
            *output = 0.0;
        }
        return FORGE_SUCCESS;
    } catch (...) {
        return FORGE_ERROR_INDEX_OUT_OF_RANGE;
    }
}

FORGE_API ForgeError forge_buffer_get_gradient_lanes(
    ForgeBufferHandle buffer,
    const uint32_t* node_ids,
    size_t count,
    double* output)
{
    if (!buffer || !buffer->buffer) return FORGE_ERROR_NULL_HANDLE;
    if (!node_ids || !output) return FORGE_ERROR_INVALID_ARGUMENT;

    try {
        // Build buffer indices
        std::vector<size_t> indices(count);
        for (size_t i = 0; i < count; ++i) {
            indices[i] = buffer->buffer->getBufferIndex(node_ids[i]);
        }
        buffer->buffer->getGradientLanes(indices, output);
        return FORGE_SUCCESS;
    } catch (...) {
        return FORGE_ERROR_INDEX_OUT_OF_RANGE;
    }
}

FORGE_API ForgeError forge_buffer_clear_gradients(ForgeBufferHandle buffer) {
    if (!buffer || !buffer->buffer) return FORGE_ERROR_NULL_HANDLE;
    buffer->buffer->clearGradients();
    return FORGE_SUCCESS;
}

FORGE_API int forge_buffer_get_vector_width(ForgeBufferHandle buffer) {
    if (!buffer || !buffer->buffer) return 0;
    return buffer->buffer->getVectorWidth();
}

FORGE_API size_t forge_buffer_get_num_nodes(ForgeBufferHandle buffer) {
    if (!buffer || !buffer->buffer) return 0;
    return static_cast<size_t>(buffer->buffer->getNumNodes());
}

FORGE_API size_t forge_buffer_get_index(ForgeBufferHandle buffer, uint32_t node_id) {
    if (!buffer || !buffer->buffer) return SIZE_MAX;
    try {
        return buffer->buffer->getBufferIndex(node_id);
    } catch (...) {
        return SIZE_MAX;
    }
}

// ==========================================================================
// Execution API
// ==========================================================================

FORGE_API ForgeError forge_execute(ForgeKernelHandle kernel, ForgeBufferHandle buffer) {
    if (!kernel || !kernel->kernel) {
        set_error("Null kernel handle");
        return FORGE_ERROR_NULL_HANDLE;
    }
    if (!buffer || !buffer->buffer) {
        set_error("Null buffer handle");
        return FORGE_ERROR_NULL_HANDLE;
    }

    try {
        kernel->kernel->execute(*buffer->buffer);
        return FORGE_SUCCESS;
    } catch (const std::exception& e) {
        set_error(e.what());
        return FORGE_ERROR_UNKNOWN;
    }
}

// ==========================================================================
// Version API
// ==========================================================================

FORGE_API const char* forge_version(void) {
    static char version[32];
    snprintf(version, sizeof(version), "%d.%d.%d",
             FORGE_VERSION_MAJOR, FORGE_VERSION_MINOR, FORGE_VERSION_PATCH);
    return version;
}

FORGE_API void forge_version_numbers(int* major, int* minor, int* patch) {
    if (major) *major = FORGE_VERSION_MAJOR;
    if (minor) *minor = FORGE_VERSION_MINOR;
    if (patch) *patch = FORGE_VERSION_PATCH;
}

} // extern "C"
