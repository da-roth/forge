/*
 * Forge C API - Stable ABI for cross-compiler compatibility
 *
 * This header provides a C interface to the Forge JIT compiler library.
 * It enables binary compatibility across different C++ compilers and
 * standard library implementations on the same platform.
 *
 * Usage:
 *   1. Create a graph: forge_graph_create()
 *   2. Add nodes: forge_graph_add_input(), forge_graph_add_*()
 *   3. Mark outputs: forge_graph_mark_output()
 *   4. Create config: forge_config_create_*()
 *   5. Compile: forge_compile()
 *   6. Create buffer: forge_buffer_create()
 *   7. Set inputs, execute, get outputs
 *   8. Clean up handles
 *
 * Thread Safety:
 *   - Graph building: NOT thread-safe (use one graph per thread)
 *   - Kernel execution: Thread-safe (same kernel can run on multiple threads)
 *   - Buffer operations: NOT thread-safe (use one buffer per thread)
 *
 * Memory Management:
 *   All forge_*_create() functions return handles that must be freed
 *   with the corresponding forge_*_destroy() function.
 *
 * SPDX-License-Identifier: Zlib
 */

#ifndef FORGE_C_API_H
#define FORGE_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * Export/Import macros for shared library
 * ========================================================================== */

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef FORGE_CAPI_BUILDING
        #define FORGE_API __declspec(dllexport)
    #else
        #define FORGE_API __declspec(dllimport)
    #endif
#else
    #if __GNUC__ >= 4
        #define FORGE_API __attribute__((visibility("default")))
    #else
        #define FORGE_API
    #endif
#endif

/* ==========================================================================
 * Opaque handle types
 * ========================================================================== */

typedef struct ForgeGraph_* ForgeGraphHandle;
typedef struct ForgeConfig_* ForgeConfigHandle;
typedef struct ForgeKernel_* ForgeKernelHandle;
typedef struct ForgeBuffer_* ForgeBufferHandle;

/* ==========================================================================
 * Error handling
 * ========================================================================== */

typedef enum ForgeError {
    FORGE_SUCCESS = 0,
    FORGE_ERROR_NULL_HANDLE = -1,
    FORGE_ERROR_INVALID_ARGUMENT = -2,
    FORGE_ERROR_COMPILATION_FAILED = -3,
    FORGE_ERROR_OUT_OF_MEMORY = -4,
    FORGE_ERROR_INDEX_OUT_OF_RANGE = -5,
    FORGE_ERROR_NOT_COMPILED = -6,
    FORGE_ERROR_UNKNOWN = -99
} ForgeError;

/**
 * Get a human-readable error message for an error code.
 */
FORGE_API const char* forge_error_string(ForgeError error);

/**
 * Get the last error message (thread-local, more detailed than error code).
 */
FORGE_API const char* forge_get_last_error(void);

/* ==========================================================================
 * OpCode enumeration (mirrors forge::OpCode)
 * ========================================================================== */

typedef enum ForgeOpCode {
    FORGE_OP_INPUT = 0,
    FORGE_OP_CONSTANT,
    FORGE_OP_ADD,
    FORGE_OP_SUB,
    FORGE_OP_MUL,
    FORGE_OP_DIV,
    FORGE_OP_NEG,
    FORGE_OP_ABS,
    FORGE_OP_SQUARE,
    FORGE_OP_RECIP,
    FORGE_OP_MOD,
    FORGE_OP_EXP,
    FORGE_OP_LOG,
    FORGE_OP_SQRT,
    FORGE_OP_POW,
    FORGE_OP_SIN,
    FORGE_OP_COS,
    FORGE_OP_TAN,
    FORGE_OP_MIN,
    FORGE_OP_MAX,
    FORGE_OP_IF,
    FORGE_OP_CMP_LT,
    FORGE_OP_CMP_LE,
    FORGE_OP_CMP_GT,
    FORGE_OP_CMP_GE,
    FORGE_OP_CMP_EQ,
    FORGE_OP_CMP_NE,
    FORGE_OP_BOOL_CONSTANT,
    FORGE_OP_BOOL_AND,
    FORGE_OP_BOOL_OR,
    FORGE_OP_BOOL_NOT,
    FORGE_OP_BOOL_EQ,
    FORGE_OP_BOOL_NE,
    FORGE_OP_INT_CONSTANT,
    FORGE_OP_INT_ADD,
    FORGE_OP_INT_SUB,
    FORGE_OP_INT_MUL,
    FORGE_OP_INT_DIV,
    FORGE_OP_INT_MOD,
    FORGE_OP_INT_NEG,
    FORGE_OP_INT_CMP_LT,
    FORGE_OP_INT_CMP_LE,
    FORGE_OP_INT_CMP_GT,
    FORGE_OP_INT_CMP_GE,
    FORGE_OP_INT_CMP_EQ,
    FORGE_OP_INT_CMP_NE,
    FORGE_OP_INT_IF,
    FORGE_OP_ARRAY_INDEX
} ForgeOpCode;

/* ==========================================================================
 * Instruction set enumeration
 * ========================================================================== */

typedef enum ForgeInstructionSet {
    FORGE_INSTRUCTION_SET_SSE2_SCALAR = 0,
    FORGE_INSTRUCTION_SET_AVX2_PACKED = 1
} ForgeInstructionSet;

/* ==========================================================================
 * Graph API
 * ========================================================================== */

/**
 * Create a new empty graph.
 * @return Handle to the graph, or NULL on failure
 */
FORGE_API ForgeGraphHandle forge_graph_create(void);

/**
 * Destroy a graph and free its resources.
 */
FORGE_API void forge_graph_destroy(ForgeGraphHandle graph);

/**
 * Clear all nodes from a graph (reuse without reallocating).
 */
FORGE_API ForgeError forge_graph_clear(ForgeGraphHandle graph);

/**
 * Get the number of nodes in the graph.
 */
FORGE_API size_t forge_graph_node_count(ForgeGraphHandle graph);

/**
 * Add an input node.
 * @return Node ID, or UINT32_MAX on error
 */
FORGE_API uint32_t forge_graph_add_input(ForgeGraphHandle graph);

/**
 * Add a constant node.
 * @param value The constant value
 * @return Node ID, or UINT32_MAX on error
 */
FORGE_API uint32_t forge_graph_add_constant(ForgeGraphHandle graph, double value);

/**
 * Add a unary operation node (e.g., NEG, ABS, EXP, LOG, SQRT, SIN, COS, TAN).
 * @param op The operation code
 * @param a Operand node ID
 * @return Node ID, or UINT32_MAX on error
 */
FORGE_API uint32_t forge_graph_add_unary(ForgeGraphHandle graph, ForgeOpCode op, uint32_t a);

/**
 * Add a binary operation node (e.g., ADD, SUB, MUL, DIV, POW, MIN, MAX).
 * @param op The operation code
 * @param a First operand node ID
 * @param b Second operand node ID
 * @return Node ID, or UINT32_MAX on error
 */
FORGE_API uint32_t forge_graph_add_binary(ForgeGraphHandle graph, ForgeOpCode op, uint32_t a, uint32_t b);

/**
 * Add a ternary operation node (e.g., IF).
 * @param op The operation code
 * @param a First operand node ID (condition for IF)
 * @param b Second operand node ID (true value for IF)
 * @param c Third operand node ID (false value for IF)
 * @return Node ID, or UINT32_MAX on error
 */
FORGE_API uint32_t forge_graph_add_ternary(ForgeGraphHandle graph, ForgeOpCode op, uint32_t a, uint32_t b, uint32_t c);

/**
 * Add a generic node with full control over all fields.
 * @param op Operation code
 * @param a First operand node ID
 * @param b Second operand node ID
 * @param c Third operand node ID
 * @param imm Immediate value (for constants)
 * @param is_active Whether node depends on inputs
 * @param needs_gradient Whether node requires gradient computation
 * @return Node ID, or UINT32_MAX on error
 */
FORGE_API uint32_t forge_graph_add_node(
    ForgeGraphHandle graph,
    ForgeOpCode op,
    uint32_t a,
    uint32_t b,
    uint32_t c,
    double imm,
    int is_active,
    int needs_gradient);

/**
 * Mark a node as an output of the graph.
 */
FORGE_API ForgeError forge_graph_mark_output(ForgeGraphHandle graph, uint32_t node_id);

/**
 * Mark a node as requiring differentiation (for AAD).
 */
FORGE_API ForgeError forge_graph_mark_diff_input(ForgeGraphHandle graph, uint32_t node_id);

/**
 * Get the number of outputs in the graph.
 */
FORGE_API size_t forge_graph_output_count(ForgeGraphHandle graph);

/**
 * Get the number of diff inputs in the graph.
 */
FORGE_API size_t forge_graph_diff_input_count(ForgeGraphHandle graph);

/**
 * Propagate needsGradient flags through the graph.
 * Must be called after marking diff inputs and before compilation.
 * This marks all nodes that depend on diff inputs as needing gradients.
 */
FORGE_API ForgeError forge_graph_propagate_gradients(ForgeGraphHandle graph);

/* ==========================================================================
 * Compiler Configuration API
 * ========================================================================== */

/**
 * Create a default configuration (stability cleaning only).
 */
FORGE_API ForgeConfigHandle forge_config_create_default(void);

/**
 * Create a debug configuration (full diagnostic output).
 */
FORGE_API ForgeConfigHandle forge_config_create_debug(void);

/**
 * Create a fast configuration (all optimizations enabled).
 */
FORGE_API ForgeConfigHandle forge_config_create_fast(void);

/**
 * Create a configuration with no optimizations.
 */
FORGE_API ForgeConfigHandle forge_config_create_no_opt(void);

/**
 * Destroy a configuration.
 */
FORGE_API void forge_config_destroy(ForgeConfigHandle config);

/**
 * Set the instruction set for compilation.
 */
FORGE_API ForgeError forge_config_set_instruction_set(ForgeConfigHandle config, ForgeInstructionSet instruction_set);

/**
 * Enable/disable specific optimizations.
 */
FORGE_API ForgeError forge_config_set_optimizations(ForgeConfigHandle config, int enable);
FORGE_API ForgeError forge_config_set_cse(ForgeConfigHandle config, int enable);
FORGE_API ForgeError forge_config_set_algebraic_simplification(ForgeConfigHandle config, int enable);
FORGE_API ForgeError forge_config_set_stability_cleaning(ForgeConfigHandle config, int enable);

/* ==========================================================================
 * Compilation API
 * ========================================================================== */

/**
 * Compile a graph into an executable kernel.
 * @param graph The graph to compile
 * @param config Compiler configuration (can be NULL for default)
 * @return Handle to compiled kernel, or NULL on failure
 */
FORGE_API ForgeKernelHandle forge_compile(ForgeGraphHandle graph, ForgeConfigHandle config);

/**
 * Destroy a compiled kernel and free its resources.
 */
FORGE_API void forge_kernel_destroy(ForgeKernelHandle kernel);

/**
 * Get the vector width of a compiled kernel (1 for scalar, 4 for AVX2).
 */
FORGE_API int forge_kernel_get_vector_width(ForgeKernelHandle kernel);

/**
 * Get the required buffer size (number of nodes) for a kernel.
 */
FORGE_API size_t forge_kernel_get_required_nodes(ForgeKernelHandle kernel);

/* ==========================================================================
 * Buffer API
 * ========================================================================== */

/**
 * Create a buffer for kernel execution.
 * @param graph The graph (for sizing)
 * @param kernel The compiled kernel (determines vector width)
 * @return Handle to buffer, or NULL on failure
 */
FORGE_API ForgeBufferHandle forge_buffer_create(ForgeGraphHandle graph, ForgeKernelHandle kernel);

/**
 * Destroy a buffer.
 */
FORGE_API void forge_buffer_destroy(ForgeBufferHandle buffer);

/**
 * Set input values for a node (all SIMD lanes).
 * @param node_id The node to set
 * @param values Pointer to vector_width doubles
 */
FORGE_API ForgeError forge_buffer_set_lanes(ForgeBufferHandle buffer, uint32_t node_id, const double* values);

/**
 * Set a single input value (broadcast to all SIMD lanes).
 */
FORGE_API ForgeError forge_buffer_set_value(ForgeBufferHandle buffer, uint32_t node_id, double value);

/**
 * Get output values for a node (all SIMD lanes).
 * @param node_id The node to get
 * @param output Pointer to receive vector_width doubles
 */
FORGE_API ForgeError forge_buffer_get_lanes(ForgeBufferHandle buffer, uint32_t node_id, double* output);

/**
 * Get a single output value (lane 0).
 */
FORGE_API ForgeError forge_buffer_get_value(ForgeBufferHandle buffer, uint32_t node_id, double* output);

/**
 * Get gradient value for a node (lane 0).
 */
FORGE_API ForgeError forge_buffer_get_gradient(ForgeBufferHandle buffer, uint32_t node_id, double* output);

/**
 * Get gradients for multiple nodes (all lanes, interleaved).
 * @param node_ids Array of node IDs
 * @param count Number of nodes
 * @param output Pointer to receive count * vector_width doubles
 */
FORGE_API ForgeError forge_buffer_get_gradient_lanes(
    ForgeBufferHandle buffer,
    const uint32_t* node_ids,
    size_t count,
    double* output);

/**
 * Clear all gradients to zero.
 */
FORGE_API ForgeError forge_buffer_clear_gradients(ForgeBufferHandle buffer);

/**
 * Get the vector width of a buffer.
 */
FORGE_API int forge_buffer_get_vector_width(ForgeBufferHandle buffer);

/**
 * Get the number of nodes in a buffer.
 */
FORGE_API size_t forge_buffer_get_num_nodes(ForgeBufferHandle buffer);

/**
 * Get the buffer index for a node ID.
 * Returns SIZE_MAX on error.
 */
FORGE_API size_t forge_buffer_get_index(ForgeBufferHandle buffer, uint32_t node_id);

/* ==========================================================================
 * Execution API
 * ========================================================================== */

/**
 * Execute a kernel with a buffer (forward and backward passes).
 */
FORGE_API ForgeError forge_execute(ForgeKernelHandle kernel, ForgeBufferHandle buffer);

/* ==========================================================================
 * Version API
 * ========================================================================== */

/**
 * Get the Forge library version string.
 */
FORGE_API const char* forge_version(void);

/**
 * Get the Forge library version as integers.
 */
FORGE_API void forge_version_numbers(int* major, int* minor, int* patch);

#ifdef __cplusplus
}
#endif

#endif /* FORGE_C_API_H */
