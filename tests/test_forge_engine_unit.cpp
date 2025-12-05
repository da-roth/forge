#include <gtest/gtest.h>
#include "../src/graph/graph.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/compiler_config.hpp"
#include "../src/compiler/node_value_buffers/scalar_node_value_buffer.hpp"

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
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 6.0);

    // Re-execute with different inputs to verify kernel can be reused
    buffer.setValue(inputId, 10.0);
    kernel->execute(buffer);
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 11.0);

    buffer.setValue(inputId, -3.0);
    kernel->execute(buffer);
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), -2.0);

    buffer.setValue(inputId, 0.0);
    kernel->execute(buffer);
    EXPECT_DOUBLE_EQ(buffer.getValue(addId), 1.0);
}
