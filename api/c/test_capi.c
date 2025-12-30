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

int test_gradient_simple(void) {
    printf("Testing gradient computation: f(x) = x^2, df/dx = 2x...\n");

    ForgeError err;

    ForgeGraphHandle graph = forge_graph_create();
    TEST_ASSERT(graph != NULL, "forge_graph_create() failed");

    /* Build: f(x) = x^2 */
    uint32_t x = forge_graph_add_input(graph);
    TEST_ASSERT(x != UINT32_MAX, "add_input failed");

    uint32_t x_sq = forge_graph_add_unary(graph, FORGE_OP_SQUARE, x);
    TEST_ASSERT(x_sq != UINT32_MAX, "add_unary(SQUARE) failed");

    /* Mark output and diff input */
    err = forge_graph_mark_output(graph, x_sq);
    TEST_ASSERT(err == FORGE_SUCCESS, "mark_output failed");

    err = forge_graph_mark_diff_input(graph, x);
    TEST_ASSERT(err == FORGE_SUCCESS, "mark_diff_input failed");

    /* Propagate gradients */
    err = forge_graph_propagate_gradients(graph);
    TEST_ASSERT(err == FORGE_SUCCESS, "propagate_gradients failed");

    /* Compile */
    ForgeConfigHandle config = forge_config_create_default();
    TEST_ASSERT(config != NULL, "forge_config_create_default() failed");

    ForgeKernelHandle kernel = forge_compile(graph, config);
    TEST_ASSERT(kernel != NULL, "forge_compile() failed");

    ForgeBufferHandle buffer = forge_buffer_create(graph, kernel);
    TEST_ASSERT(buffer != NULL, "forge_buffer_create() failed");

    /* Test with x = 3.0: f(3) = 9, df/dx = 6 */
    err = forge_buffer_set_value(buffer, x, 3.0);
    TEST_ASSERT(err == FORGE_SUCCESS, "set_value failed");

    err = forge_buffer_clear_gradients(buffer);
    TEST_ASSERT(err == FORGE_SUCCESS, "clear_gradients failed");

    err = forge_execute(kernel, buffer);
    TEST_ASSERT(err == FORGE_SUCCESS, "forge_execute() failed");

    double output;
    err = forge_buffer_get_value(buffer, x_sq, &output);
    TEST_ASSERT(err == FORGE_SUCCESS, "get_value failed");
    printf("  f(3.0) = %f (expected 9.0)\n", output);
    TEST_ASSERT_EQ(output, 9.0, 1e-10, "Output value mismatch");

    double grad;
    err = forge_buffer_get_gradient(buffer, x, &grad);
    TEST_ASSERT(err == FORGE_SUCCESS, "get_gradient failed");
    printf("  df/dx at x=3.0: %f (expected 6.0)\n", grad);
    TEST_ASSERT_EQ(grad, 6.0, 1e-10, "Gradient mismatch");

    /* Test with x = -2.0: f(-2) = 4, df/dx = -4 */
    err = forge_buffer_set_value(buffer, x, -2.0);
    err = forge_buffer_clear_gradients(buffer);
    err = forge_execute(kernel, buffer);

    err = forge_buffer_get_value(buffer, x_sq, &output);
    printf("  f(-2.0) = %f (expected 4.0)\n", output);
    TEST_ASSERT_EQ(output, 4.0, 1e-10, "Output value mismatch");

    err = forge_buffer_get_gradient(buffer, x, &grad);
    printf("  df/dx at x=-2.0: %f (expected -4.0)\n", grad);
    TEST_ASSERT_EQ(grad, -4.0, 1e-10, "Gradient mismatch");

    forge_buffer_destroy(buffer);
    forge_kernel_destroy(kernel);
    forge_config_destroy(config);
    forge_graph_destroy(graph);

    printf("  PASSED\n");
    return 0;
}

int test_gradient_multivariate(void) {
    printf("Testing multivariate gradient: f(x,y) = x*y + x^2...\n");

    ForgeError err;

    ForgeGraphHandle graph = forge_graph_create();
    TEST_ASSERT(graph != NULL, "forge_graph_create() failed");

    /* Build: f(x,y) = x*y + x^2 */
    uint32_t x = forge_graph_add_input(graph);
    uint32_t y = forge_graph_add_input(graph);
    TEST_ASSERT(x != UINT32_MAX && y != UINT32_MAX, "add_input failed");

    uint32_t x_sq = forge_graph_add_unary(graph, FORGE_OP_SQUARE, x);
    uint32_t xy = forge_graph_add_binary(graph, FORGE_OP_MUL, x, y);
    uint32_t result = forge_graph_add_binary(graph, FORGE_OP_ADD, xy, x_sq);

    /* Mark output and diff inputs */
    err = forge_graph_mark_output(graph, result);
    TEST_ASSERT(err == FORGE_SUCCESS, "mark_output failed");

    err = forge_graph_mark_diff_input(graph, x);
    TEST_ASSERT(err == FORGE_SUCCESS, "mark_diff_input x failed");

    err = forge_graph_mark_diff_input(graph, y);
    TEST_ASSERT(err == FORGE_SUCCESS, "mark_diff_input y failed");

    /* Propagate gradients */
    err = forge_graph_propagate_gradients(graph);
    TEST_ASSERT(err == FORGE_SUCCESS, "propagate_gradients failed");

    /* Compile */
    ForgeKernelHandle kernel = forge_compile(graph, NULL);
    TEST_ASSERT(kernel != NULL, "forge_compile() failed");

    ForgeBufferHandle buffer = forge_buffer_create(graph, kernel);
    TEST_ASSERT(buffer != NULL, "forge_buffer_create() failed");

    /* Test with x=2, y=3: f = 2*3 + 4 = 10, df/dx = y + 2x = 7, df/dy = x = 2 */
    err = forge_buffer_set_value(buffer, x, 2.0);
    err = forge_buffer_set_value(buffer, y, 3.0);
    err = forge_buffer_clear_gradients(buffer);
    err = forge_execute(kernel, buffer);

    double output;
    err = forge_buffer_get_value(buffer, result, &output);
    printf("  f(2,3) = %f (expected 10.0)\n", output);
    TEST_ASSERT_EQ(output, 10.0, 1e-10, "Output value mismatch");

    double grad_x, grad_y;
    err = forge_buffer_get_gradient(buffer, x, &grad_x);
    err = forge_buffer_get_gradient(buffer, y, &grad_y);
    printf("  df/dx at (2,3): %f (expected 7.0)\n", grad_x);
    printf("  df/dy at (2,3): %f (expected 2.0)\n", grad_y);
    TEST_ASSERT_EQ(grad_x, 7.0, 1e-10, "Gradient df/dx mismatch");
    TEST_ASSERT_EQ(grad_y, 2.0, 1e-10, "Gradient df/dy mismatch");

    forge_buffer_destroy(buffer);
    forge_kernel_destroy(kernel);
    forge_graph_destroy(graph);

    printf("  PASSED\n");
    return 0;
}

int test_avx2_gradient(void) {
    printf("Testing AVX2 SIMD gradient computation...\n");

    ForgeError err;

    ForgeGraphHandle graph = forge_graph_create();
    TEST_ASSERT(graph != NULL, "forge_graph_create() failed");

    /* Build: f(x) = x^2 */
    uint32_t x = forge_graph_add_input(graph);
    uint32_t x_sq = forge_graph_add_unary(graph, FORGE_OP_SQUARE, x);

    err = forge_graph_mark_output(graph, x_sq);
    err = forge_graph_mark_diff_input(graph, x);
    err = forge_graph_propagate_gradients(graph);

    /* Compile with AVX2 */
    ForgeConfigHandle config = forge_config_create_default();
    forge_config_set_instruction_set(config, FORGE_INSTRUCTION_SET_AVX2_PACKED);

    ForgeKernelHandle kernel = forge_compile(graph, config);
    TEST_ASSERT(kernel != NULL, "forge_compile() with AVX2 failed");

    int width = forge_kernel_get_vector_width(kernel);
    printf("  Kernel vector width: %d\n", width);
    TEST_ASSERT(width == 4, "Expected vector width of 4 for AVX2");

    ForgeBufferHandle buffer = forge_buffer_create(graph, kernel);
    TEST_ASSERT(buffer != NULL, "forge_buffer_create() failed");

    /* Set 4 input values: x = [1, 2, 3, 4] */
    double inputs[4] = {1.0, 2.0, 3.0, 4.0};
    err = forge_buffer_set_lanes(buffer, x, inputs);
    TEST_ASSERT(err == FORGE_SUCCESS, "set_lanes failed");

    err = forge_buffer_clear_gradients(buffer);
    err = forge_execute(kernel, buffer);
    TEST_ASSERT(err == FORGE_SUCCESS, "forge_execute() failed");

    /* Get outputs: f(x) = x^2 = [1, 4, 9, 16] */
    double outputs[4];
    err = forge_buffer_get_lanes(buffer, x_sq, outputs);
    TEST_ASSERT(err == FORGE_SUCCESS, "get_lanes failed");

    printf("  f([1,2,3,4]) = [%f, %f, %f, %f] (expected [1,4,9,16])\n",
           outputs[0], outputs[1], outputs[2], outputs[3]);
    TEST_ASSERT_EQ(outputs[0], 1.0, 1e-10, "Output[0] mismatch");
    TEST_ASSERT_EQ(outputs[1], 4.0, 1e-10, "Output[1] mismatch");
    TEST_ASSERT_EQ(outputs[2], 9.0, 1e-10, "Output[2] mismatch");
    TEST_ASSERT_EQ(outputs[3], 16.0, 1e-10, "Output[3] mismatch");

    /* Get gradients: df/dx = 2x = [2, 4, 6, 8] */
    double grads[4];
    err = forge_buffer_get_gradient_lanes(buffer, &x, 1, grads);
    TEST_ASSERT(err == FORGE_SUCCESS, "get_gradient_lanes failed");

    printf("  df/dx at [1,2,3,4] = [%f, %f, %f, %f] (expected [2,4,6,8])\n",
           grads[0], grads[1], grads[2], grads[3]);
    TEST_ASSERT_EQ(grads[0], 2.0, 1e-10, "Gradient[0] mismatch");
    TEST_ASSERT_EQ(grads[1], 4.0, 1e-10, "Gradient[1] mismatch");
    TEST_ASSERT_EQ(grads[2], 6.0, 1e-10, "Gradient[2] mismatch");
    TEST_ASSERT_EQ(grads[3], 8.0, 1e-10, "Gradient[3] mismatch");

    forge_buffer_destroy(buffer);
    forge_kernel_destroy(kernel);
    forge_config_destroy(config);
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
    failed += test_gradient_simple();
    failed += test_gradient_multivariate();
    failed += test_avx2_gradient();
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
