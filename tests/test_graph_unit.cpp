#include <gtest/gtest.h>
#include "../src/graph/graph.hpp"
#include "../src/graph/graph_recorder.hpp"

using namespace forge;

// Test Graph basic operations
class GraphTest : public ::testing::Test {
protected:
    Graph graph;
    
    void SetUp() override {
        graph.clear();
    }
};

TEST_F(GraphTest, EmptyGraph) {
    EXPECT_TRUE(graph.empty());
    EXPECT_EQ(graph.nodes.size(), 0);
    EXPECT_EQ(graph.constPool.size(), 0);
    EXPECT_EQ(graph.outputs.size(), 0);
    EXPECT_EQ(graph.diff_inputs.size(), 0);
}

TEST_F(GraphTest, AddInput) {
    NodeId inputId = graph.addInput();
    EXPECT_EQ(inputId, 0);
    EXPECT_FALSE(graph.empty());
    EXPECT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.nodes[0].op, OpCode::Input);
    EXPECT_EQ(graph.nodes[0].dst, 0);
    EXPECT_TRUE(graph.nodes[0].isActive);
}

TEST_F(GraphTest, AddConstant) {
    NodeId constId = graph.addConstant(3.14);
    EXPECT_EQ(constId, 0);
    EXPECT_EQ(graph.nodes.size(), 1);
    EXPECT_EQ(graph.constPool.size(), 1);
    EXPECT_DOUBLE_EQ(graph.constPool[0], 3.14);
    EXPECT_EQ(graph.nodes[0].op, OpCode::Constant);
    EXPECT_FALSE(graph.nodes[0].isActive);  // Constants are not active
}

TEST_F(GraphTest, AddMultipleConstants) {
    NodeId c1 = graph.addConstant(1.0);
    NodeId c2 = graph.addConstant(2.0);
    NodeId c3 = graph.addConstant(3.0);
    
    EXPECT_EQ(c1, 0);
    EXPECT_EQ(c2, 1);
    EXPECT_EQ(c3, 2);
    EXPECT_EQ(graph.constPool.size(), 3);
    EXPECT_DOUBLE_EQ(graph.constPool[0], 1.0);
    EXPECT_DOUBLE_EQ(graph.constPool[1], 2.0);
    EXPECT_DOUBLE_EQ(graph.constPool[2], 3.0);
}

TEST_F(GraphTest, AddNode) {
    Node node;
    node.op = OpCode::Add;
    node.a = 0;
    node.b = 1;
    
    NodeId nodeId = graph.addNode(node);
    EXPECT_EQ(nodeId, 0);
    EXPECT_EQ(graph.nodes[0].op, OpCode::Add);
    EXPECT_EQ(graph.nodes[0].dst, 0);
    EXPECT_EQ(graph.nodes[0].a, 0);
    EXPECT_EQ(graph.nodes[0].b, 1);
}

TEST_F(GraphTest, MarkOutput) {
    NodeId inputId = graph.addInput();
    graph.markOutput(inputId);
    
    EXPECT_EQ(graph.outputs.size(), 1);
    EXPECT_EQ(graph.outputs[0], inputId);
    
    NodeId constId = graph.addConstant(5.0);
    graph.markOutput(constId);
    
    EXPECT_EQ(graph.outputs.size(), 2);
    EXPECT_EQ(graph.outputs[1], constId);
}

TEST_F(GraphTest, Clear) {
    graph.addInput();
    graph.addConstant(1.0);
    graph.markOutput(0);
    
    graph.clear();
    
    EXPECT_TRUE(graph.empty());
    EXPECT_EQ(graph.nodes.size(), 0);
    EXPECT_EQ(graph.constPool.size(), 0);
    EXPECT_EQ(graph.outputs.size(), 0);
    EXPECT_EQ(graph.diff_inputs.size(), 0);
}

// Test GraphRecorder
class GraphRecorderTest : public ::testing::Test {
protected:
    GraphRecorder recorder;
    
    void TearDown() override {
        // Ensure recorder is stopped
        if (recorder.isRecording()) {
            try {
                recorder.stop();
            } catch (...) {
                // Ignore errors during cleanup
            }
        }
    }
};

TEST_F(GraphRecorderTest, InitialState) {
    EXPECT_FALSE(recorder.isRecording());
    EXPECT_FALSE(GraphRecorder::isAnyRecording());
    EXPECT_EQ(GraphRecorder::active(), nullptr);
}

TEST_F(GraphRecorderTest, StartStop) {
    recorder.start();
    EXPECT_TRUE(recorder.isRecording());
    EXPECT_TRUE(GraphRecorder::isAnyRecording());
    EXPECT_EQ(GraphRecorder::active(), &recorder);
    
    // Add an output to allow stopping
    recorder.graph().addInput();
    recorder.graph().markOutput(0);
    
    recorder.stop();
    EXPECT_FALSE(recorder.isRecording());
    EXPECT_FALSE(GraphRecorder::isAnyRecording());
    EXPECT_EQ(GraphRecorder::active(), nullptr);
}

TEST_F(GraphRecorderTest, StartClearsGraph) {
    // Add some nodes
    recorder.graph().addInput();
    recorder.graph().addConstant(1.0);
    EXPECT_FALSE(recorder.graph().empty());
    
    recorder.start();
    EXPECT_TRUE(recorder.graph().empty());  // Should be cleared on start
}

TEST_F(GraphRecorderTest, StopWithoutOutputThrows) {
    recorder.start();
    EXPECT_THROW(recorder.stop(), std::runtime_error);
}

TEST_F(GraphRecorderTest, DoubleStartThrows) {
    recorder.start();
    EXPECT_THROW(recorder.start(), std::runtime_error);
    
    // Clean up
    recorder.graph().addInput();
    recorder.graph().markOutput(0);
    recorder.stop();
}

TEST_F(GraphRecorderTest, StopWithoutStartThrows) {
    EXPECT_THROW(recorder.stop(), std::runtime_error);
}

TEST_F(GraphRecorderTest, MultipleRecorders) {
    GraphRecorder recorder1;
    GraphRecorder recorder2;
    
    recorder1.start();
    EXPECT_EQ(GraphRecorder::active(), &recorder1);
    
    // Cannot start second recorder while first is active
    EXPECT_THROW(recorder2.start(), std::runtime_error);
    
    // Clean up
    recorder1.graph().addInput();
    recorder1.graph().markOutput(0);
    recorder1.stop();
}

TEST_F(GraphRecorderTest, GraphAccess) {
    recorder.start();
    
    NodeId inputId = recorder.graph().addInput();
    NodeId constId = recorder.graph().addConstant(2.5);
    recorder.graph().markOutput(inputId);
    
    EXPECT_EQ(recorder.graph().nodes.size(), 2);
    EXPECT_DOUBLE_EQ(recorder.graph().constPool[0], 2.5);
    
    recorder.stop();
}

