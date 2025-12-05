#include <gtest/gtest.h>
#include <iostream>
#include "../src/graph/graph.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/compiler_config.hpp"
#include "../src/compiler/node_value_buffers/scalar_node_value_buffer.hpp"
#include "../src/compiler/node_value_buffers/avx2_node_value_buffer.hpp"

using namespace forge;

TEST(ForgeEngineTest, CompileAndExecuteSimpleGraph) {
    // Build a simple graph: output = input + 1.0
    Graph graph;
    NodeId inputId = graph.addInput();
    NodeId constId = graph.addConstant(1.0);

    Node addNode;
    addNode.op = OpCode::Add;
    addNode.a = inputId;
    addNode.b = constId;
    addNode.isActive = true;
    NodeId addId = graph.addNode(addNode);

    graph.markOutput(addId);

    // Compile with default config
    ForgeEngine engine(CompilerConfig::Default());
    auto kernel = engine.compile(graph);

    ASSERT_NE(kernel, nullptr);
    EXPECT_GT(kernel->getRequiredNodes(), 0);

    // Create buffer and set input
    ScalarNodeValueBuffer buffer(graph);
    buffer.setValue(inputId, 5.0);

    // Execute and check result: 5.0 + 1.0 = 6.0
    kernel->execute(buffer);
    std::cout << "  Input: 5.0, Result: " << buffer.getValue(addId) << ", Expected: 6.0" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 6.0);

    // Re-execute with different inputs to verify kernel can be reused
    buffer.setValue(inputId, 10.0);
    kernel->execute(buffer);
    std::cout << "  Input: 10.0, Result: " << buffer.getValue(addId) << ", Expected: 11.0" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 11.0);

    buffer.setValue(inputId, -3.0);
    kernel->execute(buffer);
    std::cout << "  Input: -3.0, Result: " << buffer.getValue(addId) << ", Expected: -2.0" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), -2.0);

    buffer.setValue(inputId, 0.0);
    kernel->execute(buffer);
    std::cout << "  Input: 0.0, Result: " << buffer.getValue(addId) << ", Expected: 1.0" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 1.0);
}

TEST(ForgeEngineTest, CompileAndExecuteWithGradient) {
    // Build a simple graph: output = input + 1.0
    // Derivative: d(output)/d(input) = 1.0 (constant, regardless of input value)
    Graph graph;
    NodeId inputId = graph.addInput();
    NodeId constId = graph.addConstant(1.0);

    // Mark input for differentiation
    graph.diff_inputs.push_back(inputId);
    graph.nodes[inputId].needsGradient = true;

    Node addNode;
    addNode.op = OpCode::Add;
    addNode.a = inputId;
    addNode.b = constId;
    addNode.isActive = true;
    addNode.needsGradient = true;
    NodeId addId = graph.addNode(addNode);

    graph.markOutput(addId);

    // Compile with default config
    ForgeEngine engine(CompilerConfig::Default());
    auto kernel = engine.compile(graph);

    ASSERT_NE(kernel, nullptr);

    // Create buffer and set input
    ScalarNodeValueBuffer buffer(graph);
    buffer.setValue(inputId, 5.0);

    // Execute and check result: 5.0 + 1.0 = 6.0
    kernel->execute(buffer);
    std::cout << "  Input: 5.0, Result: " << buffer.getValue(addId) << " (expected: 6.0), "
              << "Gradient: " << buffer.getGradient(inputId) << " (expected: 1.0)" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 6.0);
    EXPECT_DOUBLE_EQ(buffer.getGradient(inputId), 1.0);

    // Re-execute with different input
    buffer.setValue(inputId, 10.0);
    buffer.clearGradients();
    kernel->execute(buffer);
    std::cout << "  Input: 10.0, Result: " << buffer.getValue(addId) << " (expected: 11.0), "
              << "Gradient: " << buffer.getGradient(inputId) << " (expected: 1.0)" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 11.0);
    EXPECT_DOUBLE_EQ(buffer.getGradient(inputId), 1.0);

    // Re-execute with negative input
    buffer.setValue(inputId, -7.5);
    buffer.clearGradients();
    kernel->execute(buffer);
    std::cout << "  Input: -7.5, Result: " << buffer.getValue(addId) << " (expected: -6.5), "
              << "Gradient: " << buffer.getGradient(inputId) << " (expected: 1.0)" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), -6.5);
    EXPECT_DOUBLE_EQ(buffer.getGradient(inputId), 1.0);
}

// ============================================================================
// AVX2 versions of the same tests
// ============================================================================

TEST(ForgeEngineTestAVX2, CompileAndExecuteSimpleGraph) {
    // Build a simple graph: output = input + 1.0
    Graph graph;
    NodeId inputId = graph.addInput();
    NodeId constId = graph.addConstant(1.0);

    Node addNode;
    addNode.op = OpCode::Add;
    addNode.a = inputId;
    addNode.b = constId;
    addNode.isActive = true;
    NodeId addId = graph.addNode(addNode);

    graph.markOutput(addId);

    // Compile with AVX2 config
    CompilerConfig config = CompilerConfig::Default();
    config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;
    ForgeEngine engine(config);
    auto kernel = engine.compile(graph);

    ASSERT_NE(kernel, nullptr);
    EXPECT_GT(kernel->getRequiredNodes(), 0);
    EXPECT_EQ(kernel->getVectorWidth(), 4);

    // Create AVX2 buffer and set input
    AVX2NodeValueBuffer buffer(graph);
    buffer.setValue(inputId, 5.0);

    // Execute and check result: 5.0 + 1.0 = 6.0
    kernel->execute(buffer);
    std::cout << "  [AVX2] Input: 5.0, Result: " << buffer.getValue(addId) << ", Expected: 6.0" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 6.0);

    // Re-execute with different inputs to verify kernel can be reused
    buffer.setValue(inputId, 10.0);
    kernel->execute(buffer);
    std::cout << "  [AVX2] Input: 10.0, Result: " << buffer.getValue(addId) << ", Expected: 11.0" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 11.0);

    buffer.setValue(inputId, -3.0);
    kernel->execute(buffer);
    std::cout << "  [AVX2] Input: -3.0, Result: " << buffer.getValue(addId) << ", Expected: -2.0" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), -2.0);

    buffer.setValue(inputId, 0.0);
    kernel->execute(buffer);
    std::cout << "  [AVX2] Input: 0.0, Result: " << buffer.getValue(addId) << ", Expected: 1.0" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 1.0);
}

TEST(ForgeEngineTestAVX2, CompileAndExecuteWithGradient) {
    // Build a simple graph: output = input + 1.0
    // Derivative: d(output)/d(input) = 1.0 (constant, regardless of input value)
    Graph graph;
    NodeId inputId = graph.addInput();
    NodeId constId = graph.addConstant(1.0);

    // Mark input for differentiation
    graph.diff_inputs.push_back(inputId);
    graph.nodes[inputId].needsGradient = true;

    Node addNode;
    addNode.op = OpCode::Add;
    addNode.a = inputId;
    addNode.b = constId;
    addNode.isActive = true;
    addNode.needsGradient = true;
    NodeId addId = graph.addNode(addNode);

    graph.markOutput(addId);

    // Compile with AVX2 config
    CompilerConfig config = CompilerConfig::Default();
    config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;
    ForgeEngine engine(config);
    auto kernel = engine.compile(graph);

    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->getVectorWidth(), 4);

    // Create AVX2 buffer and set input
    AVX2NodeValueBuffer buffer(graph);
    buffer.setValue(inputId, 5.0);

    // Execute and check result: 5.0 + 1.0 = 6.0
    kernel->execute(buffer);
    std::cout << "  [AVX2] Input: 5.0, Result: " << buffer.getValue(addId) << " (expected: 6.0), "
              << "Gradient: " << buffer.getGradient(inputId) << " (expected: 1.0)" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 6.0);
    EXPECT_DOUBLE_EQ(buffer.getGradient(inputId), 1.0);

    // Re-execute with different input
    buffer.setValue(inputId, 10.0);
    buffer.clearGradients();
    kernel->execute(buffer);
    std::cout << "  [AVX2] Input: 10.0, Result: " << buffer.getValue(addId) << " (expected: 11.0), "
              << "Gradient: " << buffer.getGradient(inputId) << " (expected: 1.0)" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 11.0);
    EXPECT_DOUBLE_EQ(buffer.getGradient(inputId), 1.0);

    // Re-execute with negative input
    buffer.setValue(inputId, -7.5);
    buffer.clearGradients();
    kernel->execute(buffer);
    std::cout << "  [AVX2] Input: -7.5, Result: " << buffer.getValue(addId) << " (expected: -6.5), "
              << "Gradient: " << buffer.getGradient(inputId) << " (expected: 1.0)" << std::endl;
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), -6.5);
    EXPECT_DOUBLE_EQ(buffer.getGradient(inputId), 1.0);
}
