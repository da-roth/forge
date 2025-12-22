/*
 * Forge C API Test
 *
 * Basic test that exercises the C API:
 * - Create a graph: f(x) = x^2 + 2*x + 1
 * - Compile and execute
 * - Verify output
 *
 * SPDX-License-Identifier: Zlib
 */

#include "forge_c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAILED: %s\n", msg); \
        fprintf(stderr, "  Last error: %s\n", forge_get_last_error()); \
        return 1; \
    } \
} while(0)

#define TEST_ASSERT_EQ(a, b, eps, msg) do { \
    double _a = (a), _b = (b); \
    if (fabs(_a - _b) > (eps)) { \
        fprintf(stderr, "FAILED: %s (expected %f, got %f)\n", msg, _b, _a); \
        return 1; \
    } \
} while(0)

int test_version(void) {
    printf("Testing version API...\n");

    const char* version = forge_version();
    TEST_ASSERT(version != NULL, "forge_version() returned NULL");
    printf("  Version: %s\n", version);

    int major, minor, patch;
    forge_version_numbers(&major, &minor, &patch);
    printf("  Version numbers: %d.%d.%d\n", major, minor, patch);

    printf("  PASSED\n");
    return 0;
}

int test_simple_computation(void) {
    printf("Testing simple computation: f(x) = x^2 + 2*x + 1...\n");

    ForgeError err;

    /* Create graph */
    ForgeGraphHandle graph = forge_graph_create();
    TEST_ASSERT(graph != NULL, "forge_graph_create() failed");

    /* Build: f(x) = x^2 + 2*x + 1 */
    /* x */
    uint32_t x = forge_graph_add_input(graph);
    TEST_ASSERT(x != UINT32_MAX, "add_input failed");

    /* 2.0 */
    uint32_t two = forge_graph_add_constant(graph, 2.0);
    TEST_ASSERT(two != UINT32_MAX, "add_constant(2.0) failed");

    /* 1.0 */
    uint32_t one = forge_graph_add_constant(graph, 1.0);
    TEST_ASSERT(one != UINT32_MAX, "add_constant(1.0) failed");

    /* x^2 */
    uint32_t x_sq = forge_graph_add_unary(graph, FORGE_OP_SQUARE, x);
    TEST_ASSERT(x_sq != UINT32_MAX, "add_unary(SQUARE) failed");

    /* 2*x */
    uint32_t two_x = forge_graph_add_binary(graph, FORGE_OP_MUL, two, x);
    TEST_ASSERT(two_x != UINT32_MAX, "add_binary(MUL) failed");

    /* x^2 + 2*x */
    uint32_t sum1 = forge_graph_add_binary(graph, FORGE_OP_ADD, x_sq, two_x);
    TEST_ASSERT(sum1 != UINT32_MAX, "add_binary(ADD) failed");

    /* x^2 + 2*x + 1 */
    uint32_t result = forge_graph_add_binary(graph, FORGE_OP_ADD, sum1, one);
    TEST_ASSERT(result != UINT32_MAX, "add_binary(ADD) failed");

    /* Mark output */
    err = forge_graph_mark_output(graph, result);
    TEST_ASSERT(err == FORGE_SUCCESS, "mark_output failed");

    printf("  Graph has %zu nodes\n", forge_graph_node_count(graph));

    /* Create config */
    ForgeConfigHandle config = forge_config_create_default();
    TEST_ASSERT(config != NULL, "forge_config_create_default() failed");

    /* Compile */
    ForgeKernelHandle kernel = forge_compile(graph, config);
    TEST_ASSERT(kernel != NULL, "forge_compile() failed");

    printf("  Kernel vector width: %d\n", forge_kernel_get_vector_width(kernel));
    printf("  Kernel required nodes: %zu\n", forge_kernel_get_required_nodes(kernel));

    /* Create buffer */
    ForgeBufferHandle buffer = forge_buffer_create(graph, kernel);
    TEST_ASSERT(buffer != NULL, "forge_buffer_create() failed");

    /* Test with x = 3.0: f(3) = 9 + 6 + 1 = 16 */
    err = forge_buffer_set_value(buffer, x, 3.0);
    TEST_ASSERT(err == FORGE_SUCCESS, "set_value failed");

    /* Execute */
    err = forge_execute(kernel, buffer);
    TEST_ASSERT(err == FORGE_SUCCESS, "forge_execute() failed");

    /* Get result */
    double output;
    err = forge_buffer_get_value(buffer, result, &output);
    TEST_ASSERT(err == FORGE_SUCCESS, "get_value failed");

    printf("  f(3.0) = %f (expected 16.0)\n", output);
    TEST_ASSERT_EQ(output, 16.0, 1e-10, "Output value mismatch");

    /* Test with x = -1.0: f(-1) = 1 - 2 + 1 = 0 */
    err = forge_buffer_set_value(buffer, x, -1.0);
    TEST_ASSERT(err == FORGE_SUCCESS, "set_value failed");

    err = forge_execute(kernel, buffer);
    TEST_ASSERT(err == FORGE_SUCCESS, "forge_execute() failed");

    err = forge_buffer_get_value(buffer, result, &output);
    TEST_ASSERT(err == FORGE_SUCCESS, "get_value failed");

    printf("  f(-1.0) = %f (expected 0.0)\n", output);
    TEST_ASSERT_EQ(output, 0.0, 1e-10, "Output value mismatch");

    /* Clean up */
    forge_buffer_destroy(buffer);
    forge_kernel_destroy(kernel);
    forge_config_destroy(config);
    forge_graph_destroy(graph);

    printf("  PASSED\n");
    return 0;
}

int test_transcendental(void) {
    printf("Testing transcendental functions: f(x) = exp(x) + sin(x)...\n");

    ForgeError err;

    ForgeGraphHandle graph = forge_graph_create();
    TEST_ASSERT(graph != NULL, "forge_graph_create() failed");

    uint32_t x = forge_graph_add_input(graph);
    uint32_t exp_x = forge_graph_add_unary(graph, FORGE_OP_EXP, x);
    uint32_t sin_x = forge_graph_add_unary(graph, FORGE_OP_SIN, x);
    uint32_t result = forge_graph_add_binary(graph, FORGE_OP_ADD, exp_x, sin_x);

    err = forge_graph_mark_output(graph, result);
    TEST_ASSERT(err == FORGE_SUCCESS, "mark_output failed");

    ForgeKernelHandle kernel = forge_compile(graph, NULL);
    TEST_ASSERT(kernel != NULL, "forge_compile() failed");

    ForgeBufferHandle buffer = forge_buffer_create(graph, kernel);
    TEST_ASSERT(buffer != NULL, "forge_buffer_create() failed");

    /* Test with x = 0: f(0) = exp(0) + sin(0) = 1 + 0 = 1 */
    err = forge_buffer_set_value(buffer, x, 0.0);
    TEST_ASSERT(err == FORGE_SUCCESS, "set_value failed");

    err = forge_execute(kernel, buffer);
    TEST_ASSERT(err == FORGE_SUCCESS, "forge_execute() failed");

    double output;
    err = forge_buffer_get_value(buffer, result, &output);
    TEST_ASSERT(err == FORGE_SUCCESS, "get_value failed");

    printf("  f(0.0) = %f (expected 1.0)\n", output);
    TEST_ASSERT_EQ(output, 1.0, 1e-10, "Output value mismatch");

    /* Test with x = 1: f(1) = exp(1) + sin(1) */
    double expected = exp(1.0) + sin(1.0);
    err = forge_buffer_set_value(buffer, x, 1.0);
    err = forge_execute(kernel, buffer);
    err = forge_buffer_get_value(buffer, result, &output);

    printf("  f(1.0) = %f (expected %f)\n", output, expected);
    TEST_ASSERT_EQ(output, expected, 1e-10, "Output value mismatch");

    forge_buffer_destroy(buffer);
    forge_kernel_destroy(kernel);
    forge_graph_destroy(graph);

    printf("  PASSED\n");
    return 0;
}

int test_error_handling(void) {
    printf("Testing error handling...\n");

    /* Test null handle errors */
    ForgeError err = forge_graph_clear(NULL);
    TEST_ASSERT(err == FORGE_ERROR_NULL_HANDLE, "Expected NULL_HANDLE error");

    err = forge_execute(NULL, NULL);
    TEST_ASSERT(err == FORGE_ERROR_NULL_HANDLE, "Expected NULL_HANDLE error");

    /* Test error string */
    const char* msg = forge_error_string(FORGE_ERROR_NULL_HANDLE);
    TEST_ASSERT(msg != NULL, "forge_error_string returned NULL");
    printf("  Error string for NULL_HANDLE: %s\n", msg);

    printf("  PASSED\n");
    return 0;
}

int main(void) {
    int failed = 0;

    printf("=== Forge C API Tests ===\n\n");

    failed += test_version();
    failed += test_simple_computation();
    failed += test_transcendental();
    failed += test_error_handling();

    printf("\n=== Results ===\n");
    if (failed == 0) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("%d test(s) FAILED\n", failed);
        return 1;
    }
}
