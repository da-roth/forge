#include <gtest/gtest.h>
#include "../tools/graphSerialization/graph_serialization.hpp"
#include "../src/graph/graph_recorder.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/node_value_buffers/node_value_buffer.hpp"
#include <native/fdouble.hpp>
#include <cmath>
#include <vector>

using namespace forge;

// Test fixture for graph serialization tests
class GraphSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper to evaluate a kernel for given inputs
    std::vector<double> evaluateKernel(
        StitchedKernel& kernel,
        INodeValueBuffer& workspace,
        const std::vector<NodeId>& inputNodeIds,
        const std::vector<std::vector<double>>& testInputs,
        NodeId outputNodeId)
    {
        std::vector<double> results;
        for (const auto& inputs : testInputs) {
            // Set input values
            for (size_t i = 0; i < inputs.size(); ++i) {
                workspace.setValue(inputNodeIds[i], inputs[i]);
            }
            // Execute kernel
            kernel.execute(workspace);
            // Get output
            results.push_back(workspace.getValue(outputNodeId));
        }
        return results;
    }
};

// Test basic serialization and deserialization
TEST_F(GraphSerializationTest, BasicSerializationRoundTrip) {
    // Build a simple graph: result = x * 2.0 + 3.14
    GraphRecorder recorder;
    recorder.start();

    fdouble x;
    x.markInput();
    auto result = x * 2.0 + 3.14;
    result.markOutput();

    Graph originalGraph = recorder.graph();
    recorder.stop();

    // Serialize to JSON
    std::string json = serializeGraphToJson(originalGraph, true);

    // Verify JSON is non-empty and looks reasonable
    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("\"version\""), std::string::npos);
    EXPECT_NE(json.find("\"nodes\""), std::string::npos);
    EXPECT_NE(json.find("\"constPool\""), std::string::npos);

    // Deserialize from JSON
    Graph deserializedGraph = deserializeGraphFromJson(json);

    // Verify basic structure matches
    EXPECT_EQ(originalGraph.nodes.size(), deserializedGraph.nodes.size());
    EXPECT_EQ(originalGraph.constPool.size(), deserializedGraph.constPool.size());
    EXPECT_EQ(originalGraph.outputs.size(), deserializedGraph.outputs.size());
    EXPECT_EQ(originalGraph.diff_inputs.size(), deserializedGraph.diff_inputs.size());

    // Verify node details match
    for (size_t i = 0; i < originalGraph.nodes.size(); ++i) {
        const auto& orig = originalGraph.nodes[i];
        const auto& deser = deserializedGraph.nodes[i];

        EXPECT_EQ(orig.op, deser.op);
        EXPECT_EQ(orig.dst, deser.dst);
        EXPECT_EQ(orig.a, deser.a);
        EXPECT_EQ(orig.b, deser.b);
        EXPECT_EQ(orig.c, deser.c);
        EXPECT_EQ(orig.flags, deser.flags);
        EXPECT_DOUBLE_EQ(orig.imm, deser.imm);
        EXPECT_EQ(orig.isActive, deser.isActive);
        EXPECT_EQ(orig.isDead, deser.isDead);
        EXPECT_EQ(orig.needsGradient, deser.needsGradient);
    }

    // Verify constant pool matches
    for (size_t i = 0; i < originalGraph.constPool.size(); ++i) {
        EXPECT_DOUBLE_EQ(originalGraph.constPool[i], deserializedGraph.constPool[i]);
    }
}

// Test round-trip compilation equivalence
TEST_F(GraphSerializationTest, CompilationEquivalence) {
    // Build a more complex graph: result = sin(x * 2.0) + exp(y)
    GraphRecorder recorder;
    recorder.start();

    fdouble x, y;
    auto xHandle = x.markInput();
    auto yHandle = y.markInput();
    auto result = sin(x * 2.0) + exp(y);
    auto resultHandle = result.markOutput();

    Graph originalGraph = recorder.graph();
    NodeId xNodeId = xHandle.node;
    NodeId yNodeId = yHandle.node;
    NodeId resultNodeId = resultHandle.node;
    recorder.stop();

    // Path 1: Compile original graph
    ForgeEngine engine1;
    auto kernel1 = engine1.compile(originalGraph);
    auto workspace1 = NodeValueBufferFactory::create(originalGraph, *kernel1);

    // Path 2: Serialize → Deserialize → Compile
    std::string json = serializeGraphToJson(originalGraph, true);
    Graph deserializedGraph = deserializeGraphFromJson(json);

    ForgeEngine engine2;
    auto kernel2 = engine2.compile(deserializedGraph);
    auto workspace2 = NodeValueBufferFactory::create(deserializedGraph, *kernel2);

    // Test inputs
    std::vector<std::vector<double>> testInputs = {
        {0.0, 0.0},
        {1.0, 0.5},
        {-1.0, 1.0},
        {3.14159, 2.71828},
        {-2.5, -0.5}
    };

    // Evaluate both kernels
    auto results1 = evaluateKernel(*kernel1, *workspace1, {xNodeId, yNodeId}, testInputs, resultNodeId);
    auto results2 = evaluateKernel(*kernel2, *workspace2, {xNodeId, yNodeId}, testInputs, resultNodeId);

    // Verify results match
    ASSERT_EQ(results1.size(), results2.size());
    for (size_t i = 0; i < results1.size(); ++i) {
        EXPECT_DOUBLE_EQ(results1[i], results2[i])
            << "Mismatch at test input " << i
            << " (x=" << testInputs[i][0] << ", y=" << testInputs[i][1] << ")";
    }
}

// Test file I/O
TEST_F(GraphSerializationTest, FileIOTest) {
    // Build a simple graph
    GraphRecorder recorder;
    recorder.start();

    fdouble x;
    x.markInput();
    auto result = x * x + 1.0;
    result.markOutput();

    Graph originalGraph = recorder.graph();
    recorder.stop();

    // Save to file
    std::string filename = "test_graph_serialization_temp.json";
    EXPECT_TRUE(saveGraphToFile(originalGraph, filename, true));

    // Load from file
    Graph loadedGraph = loadGraphFromFile(filename);

    // Verify loaded graph matches original
    EXPECT_EQ(originalGraph.nodes.size(), loadedGraph.nodes.size());
    EXPECT_EQ(originalGraph.constPool.size(), loadedGraph.constPool.size());

    // Clean up temp file
    std::remove(filename.c_str());
}

// Test with various operations
TEST_F(GraphSerializationTest, ComplexOperationsRoundTrip) {
    // Build a graph with many different operations
    GraphRecorder recorder;
    recorder.start();

    fdouble x, y;
    x.markInput();
    y.markInput();

    // Mix of operations
    auto a = x + y;
    auto b = x - y;
    auto c = x * y;
    auto d = x / y;
    auto e = pow(x, fdouble(2.0));
    auto f = sqrt(abs(x));
    auto g = log(abs(x) + 1.0);
    auto h = sin(x) * cos(y);

    auto result = a + b + c + d + e + f + g + h;
    result.markOutput();

    Graph originalGraph = recorder.graph();
    recorder.stop();

    // Serialize and deserialize
    std::string json = serializeGraphToJson(originalGraph, false);  // Test compact format
    Graph deserializedGraph = deserializeGraphFromJson(json);

    // Verify structure
    EXPECT_EQ(originalGraph.nodes.size(), deserializedGraph.nodes.size());

    // Verify all operation types preserved
    for (size_t i = 0; i < originalGraph.nodes.size(); ++i) {
        EXPECT_EQ(originalGraph.nodes[i].op, deserializedGraph.nodes[i].op)
            << "OpCode mismatch at node " << i;
    }
}

// Test empty graph
TEST_F(GraphSerializationTest, EmptyGraphRoundTrip) {
    Graph emptyGraph;

    std::string json = serializeGraphToJson(emptyGraph);
    Graph deserializedGraph = deserializeGraphFromJson(json);

    EXPECT_TRUE(deserializedGraph.empty());
    EXPECT_EQ(deserializedGraph.nodes.size(), 0);
    EXPECT_EQ(deserializedGraph.constPool.size(), 0);
    EXPECT_EQ(deserializedGraph.outputs.size(), 0);
}

// Test graph with constants mixed with inputs
TEST_F(GraphSerializationTest, ConstantMixedGraphRoundTrip) {
    GraphRecorder recorder;
    recorder.start();

    fdouble x;
    x.markInput();
    // Mix input with constants to create a graph
    auto result = x * 3.14159 + 2.71828;
    result.markOutput();

    Graph originalGraph = recorder.graph();
    recorder.stop();

    // Serialize and deserialize
    std::string json = serializeGraphToJson(originalGraph);
    Graph deserializedGraph = deserializeGraphFromJson(json);

    // Verify constant pool preserved
    EXPECT_EQ(originalGraph.constPool.size(), deserializedGraph.constPool.size());
    for (size_t i = 0; i < originalGraph.constPool.size(); ++i) {
        EXPECT_DOUBLE_EQ(originalGraph.constPool[i], deserializedGraph.constPool[i]);
    }
}

// Test special double values (inf, nan)
TEST_F(GraphSerializationTest, SpecialDoubleValuesRoundTrip) {
    Graph graph;

    // Add some special values to constant pool
    graph.constPool.push_back(0.0);
    graph.constPool.push_back(-0.0);
    graph.constPool.push_back(std::numeric_limits<double>::infinity());  // inf
    graph.constPool.push_back(-std::numeric_limits<double>::infinity()); // -inf
    graph.constPool.push_back(std::numeric_limits<double>::max());
    graph.constPool.push_back(std::numeric_limits<double>::min());

    std::string json = serializeGraphToJson(graph);
    Graph deserializedGraph = deserializeGraphFromJson(json);

    ASSERT_EQ(graph.constPool.size(), deserializedGraph.constPool.size());

    EXPECT_DOUBLE_EQ(deserializedGraph.constPool[0], 0.0);
    EXPECT_DOUBLE_EQ(deserializedGraph.constPool[1], -0.0);
    EXPECT_TRUE(std::isinf(deserializedGraph.constPool[2]));
    EXPECT_TRUE(std::isinf(deserializedGraph.constPool[3]));
    EXPECT_DOUBLE_EQ(deserializedGraph.constPool[4], std::numeric_limits<double>::max());
    EXPECT_DOUBLE_EQ(deserializedGraph.constPool[5], std::numeric_limits<double>::min());
}
