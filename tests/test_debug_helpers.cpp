#include <gtest/gtest.h>
#include "../src/graph/graph.hpp"
#include "../src/graph/graph_optimizer.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/node_value_buffers/node_value_buffer.hpp"

#include <cmath>
#include <iostream>

using namespace forge;

// ============================================================================
// Local helper functions for building graphs
// ============================================================================
namespace {

NodeId addBinaryOp(Graph& g, OpCode op, NodeId a, NodeId b) {
    Node node{};
    node.op = op;
    node.a = a;
    node.b = b;
    node.isActive = g.nodes[a].isActive || g.nodes[b].isActive;
    return g.addNode(node);
}

NodeId addUnaryOp(Graph& g, OpCode op, NodeId a) {
    Node node{};
    node.op = op;
    node.a = a;
    node.isActive = g.nodes[a].isActive;
    return g.addNode(node);
}

} // anonymous namespace

// ============================================================================
// Debug Helper Coverage Tests
// These tests exercise the debug output code paths in graph_optimizer.cpp
// that are not covered by the regular GraphOptimizationTest tests.
// Specifically: printGraphDebug, printStepByStepDebug, printOriginalGraph,
// printOptimizedGraph flags, getOpCodeName, and optimizeWithMapping debug output.
// ============================================================================

TEST(DebugHelperOptimizationAnalysisTest, FullDebugOutputCoverage) {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  DEBUG HELPER: Exercising All Debug Output Code Paths\n";
    std::cout << "============================================================\n";
    std::cout << "\n";

    // Build a graph that exercises multiple optimization passes and opcodes
    Graph graph;
    NodeId x = graph.addInput();
    graph.addInput();

    // Add various opcodes to exercise getOpCodeName
    NodeId c2 = graph.addConstant(2.0);
    NodeId c3 = graph.addConstant(3.0);
    NodeId constMul = addBinaryOp(graph, OpCode::Mul, c2, c3);  // Constant folding
    NodeId neg = addUnaryOp(graph, OpCode::Neg, x);
    NodeId expOp = addUnaryOp(graph, OpCode::Exp, x);
    NodeId logOp = addUnaryOp(graph, OpCode::Log, x);
    NodeId sinOp = addUnaryOp(graph, OpCode::Sin, x);
    NodeId cosOp = addUnaryOp(graph, OpCode::Cos, x);
    NodeId tanOp = addUnaryOp(graph, OpCode::Tan, x);
    NodeId sqrtOp = addUnaryOp(graph, OpCode::Sqrt, x);
    NodeId absOp = addUnaryOp(graph, OpCode::Abs, x);
    NodeId sub = addBinaryOp(graph, OpCode::Sub, x, c2);
    NodeId div = addBinaryOp(graph, OpCode::Div, x, c2);
    NodeId powOp = addBinaryOp(graph, OpCode::Pow, x, c2);

    // Combine into output
    NodeId sum1 = addBinaryOp(graph, OpCode::Add, neg, expOp);
    NodeId sum2 = addBinaryOp(graph, OpCode::Add, logOp, sinOp);
    NodeId sum3 = addBinaryOp(graph, OpCode::Add, cosOp, tanOp);
    NodeId sum4 = addBinaryOp(graph, OpCode::Add, sqrtOp, absOp);
    NodeId sum5 = addBinaryOp(graph, OpCode::Add, sub, div);
    NodeId sum6 = addBinaryOp(graph, OpCode::Add, powOp, constMul);
    NodeId sum7 = addBinaryOp(graph, OpCode::Add, sum1, sum2);
    NodeId sum8 = addBinaryOp(graph, OpCode::Add, sum3, sum4);
    NodeId sum9 = addBinaryOp(graph, OpCode::Add, sum5, sum6);
    NodeId sum10 = addBinaryOp(graph, OpCode::Add, sum7, sum8);
    graph.markOutput(addBinaryOp(graph, OpCode::Add, sum9, sum10));

    // Test 1: optimize() with all debug flags enabled
    {
        GraphOptimizer optimizer;
        GraphOptimizer::OptimizationConfig config;
        config.enableInactiveFolding = true;
        config.enableCSE = true;
        config.enableAlgebraicSimplification = true;
        config.enableStabilityCleaning = true;
        config.enableConstantCleanup = true;
        config.printStepByStepDebug = true;   // Covers step-by-step debug output
        config.printOriginalGraph = true;      // Covers original graph printing
        config.printOptimizedGraph = true;     // Covers optimized graph printing
        optimizer.setConfig(config);

        std::cout << "--- optimize() with debug flags ---\n";
        Graph optimized = optimizer.optimize(graph);

        const auto& stats = optimizer.getLastStats();
        EXPECT_TRUE(stats.changesApplied);

        // Compile and execute
        ForgeEngine engine;
        auto kernel = engine.compile(optimized);
        auto buffer = NodeValueBufferFactory::create(optimized, *kernel);
        buffer->setValue(0, 1.0);
        buffer->setValue(1, 0.0);
        kernel->execute(*buffer);
        double result = buffer->getValue(optimized.outputs[0]);
        EXPECT_FALSE(std::isnan(result));
    }

    // Test 2: optimizeWithMapping() with all debug flags enabled
    {
        GraphOptimizer optimizer;
        GraphOptimizer::OptimizationConfig config;
        config.enableInactiveFolding = true;
        config.enableCSE = true;
        config.enableAlgebraicSimplification = true;
        config.enableStabilityCleaning = true;
        config.enableConstantCleanup = true;
        config.printStepByStepDebug = true;
        config.printOriginalGraph = true;
        config.printOptimizedGraph = true;
        optimizer.setConfig(config);

        std::cout << "\n--- optimizeWithMapping() with debug flags ---\n";
        auto result = optimizer.optimizeWithMapping(graph);

        EXPECT_EQ(result.originalToOptimizedMapping.size(), graph.nodes.size());

        // Compile and execute
        ForgeEngine engine;
        auto kernel = engine.compile(result.optimizedTape);
        auto buffer = NodeValueBufferFactory::create(result.optimizedTape, *kernel);
        buffer->setValue(0, 1.0);
        buffer->setValue(1, 0.0);
        kernel->execute(*buffer);
        double value = buffer->getValue(result.optimizedTape.outputs[0]);
        EXPECT_FALSE(std::isnan(value));
    }

    // Test 3: ForgeEngine with all debug flags enabled
    // This covers the optimization stats, timing info, and graph printing in forge_engine.cpp
    {
        std::cout << "\n--- ForgeEngine compile() with all debug flags ---\n";

        CompilerConfig compilerConfig = CompilerConfig::Default();
        compilerConfig.printOptimizationStats = true;   // Covers optimization stats output
        compilerConfig.printOriginalGraph = true;       // Covers original graph printing
        compilerConfig.printOptimizedGraph = true;      // Covers optimized graph printing
        compilerConfig.printNodeFlags = true;           // Covers node flags printing
        compilerConfig.printGradientDebug = true;       // Covers gradient debug output

        ForgeEngine engine(compilerConfig);
        auto kernel = engine.compile(graph);
        ASSERT_NE(kernel, nullptr);

        auto buffer = NodeValueBufferFactory::create(graph, *kernel);
        buffer->setValue(0, 1.0);
        buffer->setValue(1, 0.0);
        kernel->execute(*buffer);
        double result = buffer->getValue(graph.outputs[0]);
        EXPECT_FALSE(std::isnan(result));
    }

    // Test 4: ForgeEngine with gradient computation and debug output
    {
        std::cout << "\n--- ForgeEngine with gradient debug output ---\n";

        // Build a simple graph with gradients enabled
        Graph gradGraph;
        NodeId gx = gradGraph.addInput();
        gradGraph.nodes[gx].needsGradient = true;
        gradGraph.nodes[gx].isActive = true;
        gradGraph.diff_inputs.push_back(gx);

        NodeId gc = gradGraph.addConstant(2.0);
        NodeId gmul = addBinaryOp(gradGraph, OpCode::Mul, gx, gc);
        gradGraph.nodes[gmul].needsGradient = true;
        gradGraph.nodes[gmul].isActive = true;
        gradGraph.markOutput(gmul);

        CompilerConfig compilerConfig = CompilerConfig::Default();
        compilerConfig.printGradientDebug = true;

        ForgeEngine engine(compilerConfig);
        auto kernel = engine.compile(gradGraph);
        ASSERT_NE(kernel, nullptr);

        auto buffer = NodeValueBufferFactory::create(gradGraph, *kernel);
        buffer->setValue(gx, 3.0);
        kernel->execute(*buffer);
        double result = buffer->getValue(gradGraph.outputs[0]);
        EXPECT_DOUBLE_EQ(result, 6.0);  // 3.0 * 2.0
    }

    std::cout << "\nDebug output coverage test completed.\n";
}
