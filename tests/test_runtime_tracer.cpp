/**
 * @file test_runtime_tracer.cpp
 * @brief Educational tests demonstrating runtime tracing in Forge
 *
 * This test file shows how to enable and use the InstructionTracer for debugging
 * JIT-compiled code. The tracer records register values at runtime, which is
 * invaluable for diagnosing issues in generated machine code.
 *
 * == WHEN TO USE RUNTIME TRACING ==
 *
 * Runtime tracing is a debugging tool for when:
 *   - Computed values are incorrect and you need to find where corruption happens
 *   - You suspect a specific operation is producing wrong results
 *   - You want to understand the data flow through the JIT-compiled kernel
 *   - You're developing new operations and need to verify intermediate values
 *
 * == HOW IT WORKS ==
 *
 * When tracing is enabled via CompilerConfig:
 *   1. The compiler injects extra assembly code at each operation
 *   2. This code safely copies register values to a trace buffer
 *   3. After execution, you can inspect the trace to see all intermediate values
 *   4. Smart filtering can detect corruption (NaN, Inf, suspicious patterns)
 *
 * == PERFORMANCE IMPACT ==
 *
 * Tracing adds ~60-100 cycles per operation, so it should only be used for
 * debugging, not in production code. When tracing is disabled (the default),
 * there is zero overhead.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../src/graph/graph.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/x86/common/compiler_config.hpp"
#include "../src/compiler/runtime_trace.hpp"
#include "../src/compiler/interfaces/node_value_buffer.hpp"
#include "test_graphs.hpp"

using namespace forge;
using namespace forge_tests;

// ============================================================================
// EXAMPLE 1: Basic Tracing with SSE2 (Scalar)
// ============================================================================
//
// This test demonstrates how to enable runtime tracing for SSE2 scalar mode.
// We create a simple graph: z = (x + y) * 2, and trace all operations.
//
// KEY DIFFERENCE FROM NORMAL USAGE:
//   Normal:  ForgeEngine engine(CompilerConfig::Default());
//   Tracing: ForgeEngine engine(configWithTracing);  // printRuntimeTrace = true
//

TEST(DebugHelperTraceTest, SSE2_BasicTracingExample) {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  EXAMPLE: Enabling Runtime Tracing (SSE2 Scalar Mode)\n";
    std::cout << "============================================================\n";
    std::cout << "\n";
    std::cout << "This example shows how to enable tracing to debug JIT code.\n";
    std::cout << "We'll compute z = (x + y) * 2 with x=3, y=4 and trace each step.\n";
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // STEP 1: Create a CompilerConfig with tracing enabled
    // -------------------------------------------------------------------------
    //
    // The key setting is: printRuntimeTrace = true
    //
    // You can either:
    //   a) Start with Default() and enable tracing manually
    //   b) Use DebugTracing() preset
    //
    std::cout << "STEP 1: Configure the compiler for tracing\n";
    std::cout << "-------\n";
    std::cout << "  // Option A: Enable tracing on a default config\n";
    std::cout << "  CompilerConfig config = CompilerConfig::Default();\n";
    std::cout << "  config.printRuntimeTrace = true;\n";
    std::cout << "\n";
    std::cout << "  // Option B: Use the DebugTracing preset\n";
    std::cout << "  CompilerConfig config = CompilerConfig::DebugTracing();\n";
    std::cout << "\n";

    CompilerConfig config = CompilerConfig::Default();
    config.printRuntimeTrace = true;  // <-- This is the key setting!
    config.instructionSet = CompilerConfig::InstructionSet::SSE2_SCALAR;

    // -------------------------------------------------------------------------
    // STEP 2: Build the computation graph (same as normal usage)
    // -------------------------------------------------------------------------
    std::cout << "STEP 2: Build the computation graph (unchanged from normal)\n";
    std::cout << "-------\n";
    std::cout << "  Graph: z = (x + y) * 2\n";
    std::cout << "\n";

    Graph graph;
    NodeId x = graph.addInput();                              // Node 0: input x
    NodeId y = graph.addInput();                              // Node 1: input y
    NodeId sum = addBinaryOp(graph, OpCode::Add, x, y);       // Node 2: x + y
    NodeId two = graph.addConstant(2.0);                      // Node 3: constant 2
    NodeId result = addBinaryOp(graph, OpCode::Mul, sum, two); // Node 4: (x + y) * 2
    graph.markOutput(result);

    // -------------------------------------------------------------------------
    // STEP 3: Compile with the tracing-enabled config
    // -------------------------------------------------------------------------
    std::cout << "STEP 3: Compile the graph (tracing code is injected here)\n";
    std::cout << "-------\n";

    ForgeEngine engine(config);
    auto kernel = engine.compile(graph);
    ASSERT_NE(kernel, nullptr) << "Kernel compilation failed";

    std::cout << "\n";

    // -------------------------------------------------------------------------
    // STEP 4: Execute and observe the trace output
    // -------------------------------------------------------------------------
    std::cout << "STEP 4: Execute the kernel (trace records are captured)\n";
    std::cout << "-------\n";
    std::cout << "  Inputs: x=3.0, y=4.0\n";
    std::cout << "  Expected: (3 + 4) * 2 = 14\n";
    std::cout << "\n";

    auto buffer = NodeValueBufferFactory::create(graph, *kernel);
    buffer->setValue(x, 3.0);
    buffer->setValue(y, 4.0);

    kernel->execute(*buffer);

    double output = buffer->getValue(result);
    std::cout << "  Result: " << output << "\n";
    std::cout << "\n";

    EXPECT_DOUBLE_EQ(output, 14.0);

    // -------------------------------------------------------------------------
    // STEP 5: Print the trace records
    // -------------------------------------------------------------------------
    std::cout << "STEP 5: Examine the trace buffer\n";
    std::cout << "-------\n";
    std::cout << "  The trace buffer contains all register values captured during execution.\n";
    std::cout << "  Call printTraceRecords() to see the full trace:\n";
    std::cout << "\n";

    printTraceRecords();

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  END OF EXAMPLE\n";
    std::cout << "============================================================\n";
    std::cout << "\n";

    // Cleanup trace buffer
    cleanupTraceBuffer();
}

// ============================================================================
// EXAMPLE 2: Basic Tracing with AVX2 (Packed)
// ============================================================================
//
// This test demonstrates tracing in AVX2 mode, where 4 values are processed
// simultaneously in YMM (256-bit) registers.
//

#ifdef FORGE_BUNDLE_AVX2
TEST(DebugHelperTraceTest, AVX2_BasicTracingExample) {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  EXAMPLE: Enabling Runtime Tracing (AVX2 Packed Mode)\n";
    std::cout << "============================================================\n";
    std::cout << "\n";
    std::cout << "AVX2 mode processes 4 doubles simultaneously in YMM registers.\n";
    std::cout << "The tracer captures all 4 lanes for each operation.\n";
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Configure for AVX2 with tracing
    // -------------------------------------------------------------------------
    std::cout << "Configuration:\n";
    std::cout << "  config.printRuntimeTrace = true;\n";
    std::cout << "  config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;\n";
    std::cout << "\n";

    CompilerConfig config = CompilerConfig::Default();
    config.printRuntimeTrace = true;
    config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;

    // Build a simple graph: z = sin(x + y)
    Graph graph;
    NodeId x = graph.addInput();
    NodeId y = graph.addInput();
    NodeId sum = addBinaryOp(graph, OpCode::Add, x, y);
    NodeId result = addUnaryOp(graph, OpCode::Sin, sum);
    graph.markOutput(result);

    std::cout << "Graph: z = sin(x + y)\n";
    std::cout << "\n";

    // Compile
    ForgeEngine engine(config);
    auto kernel = engine.compile(graph);
    ASSERT_NE(kernel, nullptr) << "Kernel compilation failed";

    std::cout << "\n";

    // Execute with 4 input sets (AVX2 processes all 4 simultaneously)
    std::cout << "Executing with 4 input sets (processed in parallel by AVX2):\n";
    std::cout << "  Lane 0: x=0.0, y=0.0 -> sin(0.0) = 0.0\n";
    std::cout << "  Lane 1: x=1.0, y=0.57 -> sin(1.57) ~ 1.0\n";
    std::cout << "  Lane 2: x=3.14, y=0.0 -> sin(3.14) ~ 0.0\n";
    std::cout << "  Lane 3: x=0.0, y=4.71 -> sin(4.71) ~ -1.0\n";
    std::cout << "\n";

    auto buffer = NodeValueBufferFactory::create(graph, *kernel);

    // Set input values for all 4 lanes using setLanes
    double xVals[4] = {0.0, 1.0, 3.14159265, 0.0};
    double yVals[4] = {0.0, 0.57079632, 0.0, 4.71238898};

    buffer->setLanes(x, xVals);
    buffer->setLanes(y, yVals);

    kernel->execute(*buffer);

    // Get all 4 output lanes using getLanes
    double outputs[4];
    buffer->getLanes(result, outputs);

    std::cout << "Results:\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "  Lane " << i << ": " << std::fixed << std::setprecision(6) << outputs[i] << "\n";
    }
    std::cout << "\n";

    // Verify results (approximately)
    EXPECT_NEAR(outputs[0], 0.0, 1e-6);   // sin(0) = 0
    EXPECT_NEAR(outputs[1], 1.0, 1e-6);   // sin(pi/2) = 1
    EXPECT_NEAR(outputs[2], 0.0, 1e-6);   // sin(pi) = 0
    EXPECT_NEAR(outputs[3], -1.0, 1e-6);  // sin(3pi/2) = -1

    // Print trace
    std::cout << "Trace records (showing all 4 lanes for each YMM register):\n";
    std::cout << "\n";
    printTraceRecords();

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  END OF EXAMPLE\n";
    std::cout << "============================================================\n";
    std::cout << "\n";

    cleanupTraceBuffer();
}
#endif // FORGE_BUNDLE_AVX2

// ============================================================================
// EXAMPLE 3: Comparison - With and Without Tracing
// ============================================================================
//
// This test shows the difference in compilation output between normal
// mode and tracing mode.
//

TEST(DebugHelperTraceTest, ComparisonWithAndWithoutTracing) {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  COMPARISON: Normal vs Tracing Mode\n";
    std::cout << "============================================================\n";
    std::cout << "\n";

    Graph graph;
    NodeId x = graph.addInput();
    NodeId result = addUnaryOp(graph, OpCode::Square, x);
    graph.markOutput(result);

    // -------------------------------------------------------------------------
    // Normal mode (no tracing)
    // -------------------------------------------------------------------------
    std::cout << "NORMAL MODE (CompilerConfig::Default()):\n";
    std::cout << "  - No tracing code injected\n";
    std::cout << "  - Zero overhead\n";
    std::cout << "  - Use for production\n";
    std::cout << "\n";

    {
        CompilerConfig config = CompilerConfig::Default();
        ForgeEngine engine(config);
        auto kernel = engine.compile(graph);
        ASSERT_NE(kernel, nullptr);

        auto buffer = NodeValueBufferFactory::create(graph, *kernel);
        buffer->setValue(x, 5.0);
        kernel->execute(*buffer);

        std::cout << "  Result: 5^2 = " << buffer->getValue(result) << "\n";
    }

    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Tracing mode
    // -------------------------------------------------------------------------
    std::cout << "TRACING MODE (printRuntimeTrace = true):\n";
    std::cout << "  - Tracing code injected at each operation\n";
    std::cout << "  - ~60-100 cycles overhead per operation\n";
    std::cout << "  - Use only for debugging\n";
    std::cout << "\n";

    {
        CompilerConfig config = CompilerConfig::Default();
        config.printRuntimeTrace = true;
        ForgeEngine engine(config);
        auto kernel = engine.compile(graph);
        ASSERT_NE(kernel, nullptr);

        std::cout << "\n";

        auto buffer = NodeValueBufferFactory::create(graph, *kernel);
        buffer->setValue(x, 5.0);
        kernel->execute(*buffer);

        std::cout << "  Result: 5^2 = " << buffer->getValue(result) << "\n";
        std::cout << "\n";

        printTraceRecords();
        cleanupTraceBuffer();
    }

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  END OF COMPARISON\n";
    std::cout << "============================================================\n";
    std::cout << "\n";
}
