#include <gtest/gtest.h>
#include "../src/graph/graph.hpp"
#include "../src/graph/graph_optimizer.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/node_value_buffers/node_value_buffer.hpp"

#include <functional>
#include <tuple>
#include <vector>
#include <string>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace forge;

// ============================================================================
// Local helper functions for building graphs (only uses /src APIs)
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

// Helper to execute a kernel and get output
double executeKernel(const Graph& graph, double x, double y) {
    ForgeEngine engine;
    auto kernel = engine.compile(graph);

    std::vector<NodeId> inputNodes;
    for (NodeId i = 0; i < graph.nodes.size(); ++i) {
        if (graph.nodes[i].op == OpCode::Input) {
            inputNodes.push_back(i);
        }
    }

    auto buffer = NodeValueBufferFactory::create(graph, *kernel);
    if (inputNodes.size() >= 1) buffer->setValue(inputNodes[0], x);
    if (inputNodes.size() >= 2) buffer->setValue(inputNodes[1], y);

    kernel->execute(*buffer);
    return buffer->getValue(graph.outputs[0]);
}

} // anonymous namespace

// ============================================================================
// Graph builder functions
// ============================================================================
namespace Graphs {

inline void AddXY(Graph& g) {
    NodeId x = g.addInput();
    NodeId y = g.addInput();
    g.markOutput(addBinaryOp(g, OpCode::Add, x, y));
}

inline void MulXY(Graph& g) {
    NodeId x = g.addInput();
    NodeId y = g.addInput();
    g.markOutput(addBinaryOp(g, OpCode::Mul, x, y));
}

inline void AddXWithConstantMul(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c2 = g.addConstant(2.0);
    NodeId c3 = g.addConstant(3.0);
    NodeId mul = addBinaryOp(g, OpCode::Mul, c2, c3);
    g.markOutput(addBinaryOp(g, OpCode::Add, x, mul));
}

inline void ConstantAddMulX(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c2 = g.addConstant(2.0);
    NodeId c3 = g.addConstant(3.0);
    NodeId add = addBinaryOp(g, OpCode::Add, c2, c3);
    g.markOutput(addBinaryOp(g, OpCode::Mul, add, x));
}

inline void DuplicateAddXY(Graph& g) {
    NodeId x = g.addInput();
    NodeId y = g.addInput();
    NodeId sum1 = addBinaryOp(g, OpCode::Add, x, y);
    NodeId sum2 = addBinaryOp(g, OpCode::Add, x, y);
    g.markOutput(addBinaryOp(g, OpCode::Add, sum1, sum2));
}

inline void DuplicateExpX(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId exp1 = addUnaryOp(g, OpCode::Exp, x);
    NodeId exp2 = addUnaryOp(g, OpCode::Exp, x);
    g.markOutput(addBinaryOp(g, OpCode::Add, exp1, exp2));
}

inline void AddXZero(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    g.markOutput(addBinaryOp(g, OpCode::Add, x, g.addConstant(0.0)));
}

inline void MulXOne(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    g.markOutput(addBinaryOp(g, OpCode::Mul, x, g.addConstant(1.0)));
}

inline void MulXZero(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    g.markOutput(addBinaryOp(g, OpCode::Mul, x, g.addConstant(0.0)));
}

inline void SubXX(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    g.markOutput(addBinaryOp(g, OpCode::Sub, x, x));
}

inline void DivXX(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    g.markOutput(addBinaryOp(g, OpCode::Div, x, x));
}

inline void OneOverExpX(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId expX = addUnaryOp(g, OpCode::Exp, x);
    g.markOutput(addBinaryOp(g, OpCode::Div, g.addConstant(1.0), expX));
}

inline void DifferenceOfSquaresFactored(Graph& g) {
    NodeId x = g.addInput();
    NodeId y = g.addInput();
    NodeId sum = addBinaryOp(g, OpCode::Add, x, y);
    NodeId diff = addBinaryOp(g, OpCode::Sub, x, y);
    g.markOutput(addBinaryOp(g, OpCode::Mul, sum, diff));
}

inline void NestedConstantFolding(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c2 = g.addConstant(2.0);
    NodeId c3 = g.addConstant(3.0);
    NodeId c4 = g.addConstant(4.0);
    NodeId mul = addBinaryOp(g, OpCode::Mul, c2, c3);
    NodeId div = addBinaryOp(g, OpCode::Div, c4, c2);
    NodeId add1 = addBinaryOp(g, OpCode::Add, mul, div);
    g.markOutput(addBinaryOp(g, OpCode::Add, x, add1));
}

inline void MultipleAlgebraic(Graph& g) {
    NodeId x = g.addInput();
    NodeId y = g.addInput();
    NodeId zero = g.addConstant(0.0);
    NodeId one = g.addConstant(1.0);
    NodeId xPlusZero = addBinaryOp(g, OpCode::Add, x, zero);
    NodeId yTimesOne = addBinaryOp(g, OpCode::Mul, y, one);
    g.markOutput(addBinaryOp(g, OpCode::Add, xPlusZero, yTimesOne));
}

inline void SinXPlusCosY(Graph& g) {
    NodeId x = g.addInput();
    NodeId y = g.addInput();
    NodeId sinX = addUnaryOp(g, OpCode::Sin, x);
    NodeId cosY = addUnaryOp(g, OpCode::Cos, y);
    g.markOutput(addBinaryOp(g, OpCode::Add, sinX, cosY));
}

inline void LogExpX(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId expX = addUnaryOp(g, OpCode::Exp, x);
    g.markOutput(addUnaryOp(g, OpCode::Log, expX));
}

// log(exp(x + 2)) - x is not directly the input to exp
inline void LogExpXPlus2(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId two = g.addConstant(2.0);
    NodeId xPlus2 = addBinaryOp(g, OpCode::Add, x, two);  // x + 2
    NodeId expXPlus2 = addUnaryOp(g, OpCode::Exp, xPlus2);  // exp(x + 2)
    g.markOutput(addUnaryOp(g, OpCode::Log, expXPlus2));  // log(exp(x + 2)) = x + 2
}

// log(exp(x)) * 2 - result is not directly the output
inline void LogExpXTimes2(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId two = g.addConstant(2.0);
    NodeId expX = addUnaryOp(g, OpCode::Exp, x);  // exp(x)
    NodeId logExp = addUnaryOp(g, OpCode::Log, expX);  // log(exp(x)) = x
    g.markOutput(addBinaryOp(g, OpCode::Mul, logExp, two));  // x * 2
}

// log(exp(x + 2)) * 3 - more complex case
inline void LogExpXPlus2Times3(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId two = g.addConstant(2.0);
    NodeId three = g.addConstant(3.0);
    NodeId xPlus2 = addBinaryOp(g, OpCode::Add, x, two);  // x + 2
    NodeId expXPlus2 = addUnaryOp(g, OpCode::Exp, xPlus2);  // exp(x + 2)
    NodeId logExp = addUnaryOp(g, OpCode::Log, expXPlus2);  // log(exp(x + 2)) = x + 2
    g.markOutput(addBinaryOp(g, OpCode::Mul, logExp, three));  // (x + 2) * 3
}

inline void SqrtXSquared(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId sqrtX = addUnaryOp(g, OpCode::Sqrt, x);
    g.markOutput(addBinaryOp(g, OpCode::Mul, sqrtX, sqrtX));
}

// ============================================================================
// Algebraic simplification coverage graphs
// ============================================================================

// 0 + x → x (additive identity with zero on left)
inline void ZeroPlusX(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    g.markOutput(addBinaryOp(g, OpCode::Add, g.addConstant(0.0), x));
}

// x - 0 → x (subtractive identity)
inline void SubXZero(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    g.markOutput(addBinaryOp(g, OpCode::Sub, x, g.addConstant(0.0)));
}

// x / 1 → x (division by one)
inline void DivXOne(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    g.markOutput(addBinaryOp(g, OpCode::Div, x, g.addConstant(1.0)));
}

// -(-x) → x (double negation)
inline void DoubleNegation(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId negX = addUnaryOp(g, OpCode::Neg, x);
    g.markOutput(addUnaryOp(g, OpCode::Neg, negX));
}

// Square(0) → 0
inline void SquareZero(Graph& g) {
    g.addInput();
    g.addInput();
    NodeId zero = g.addConstant(0.0);
    Node squareNode{};
    squareNode.op = OpCode::Square;
    squareNode.a = zero;
    squareNode.isActive = false;  // Constant input means inactive
    g.markOutput(g.addNode(squareNode));
}

// Square(1) → 1
inline void SquareOne(Graph& g) {
    g.addInput();
    g.addInput();
    NodeId one = g.addConstant(1.0);
    Node squareNode{};
    squareNode.op = OpCode::Square;
    squareNode.a = one;
    squareNode.isActive = false;
    g.markOutput(g.addNode(squareNode));
}

// Sqrt(0) → 0
inline void SqrtZero(Graph& g) {
    g.addInput();
    g.addInput();
    NodeId zero = g.addConstant(0.0);
    g.markOutput(addUnaryOp(g, OpCode::Sqrt, zero));
}

// Sqrt(1) → 1
inline void SqrtOne(Graph& g) {
    g.addInput();
    g.addInput();
    NodeId one = g.addConstant(1.0);
    g.markOutput(addUnaryOp(g, OpCode::Sqrt, one));
}

// Exp(0) → 1
inline void ExpZero(Graph& g) {
    g.addInput();
    g.addInput();
    NodeId zero = g.addConstant(0.0);
    g.markOutput(addUnaryOp(g, OpCode::Exp, zero));
}

// Log(1) → 0
inline void LogOne(Graph& g) {
    g.addInput();
    g.addInput();
    NodeId one = g.addConstant(1.0);
    g.markOutput(addUnaryOp(g, OpCode::Log, one));
}

} // namespace Graphs

// ============================================================================
// Graph test case with name and test data
// ============================================================================
struct GraphTestCase {
    std::string name;
    std::function<void(Graph&)> build;
    std::vector<std::tuple<double, double, double>> testData;
};

std::vector<GraphTestCase> GetAllGraphs() {
    return {
        {"AddXY", Graphs::AddXY, {{1.0, 2.0, 3.0}, {-1.0, 1.0, 0.0}, {0.5, 0.5, 1.0}}},
        {"MulXY", Graphs::MulXY, {{2.0, 3.0, 6.0}, {-2.0, 3.0, -6.0}, {0.0, 5.0, 0.0}}},
        {"AddXWithConstantMul", Graphs::AddXWithConstantMul, {{1.0, 0.0, 7.0}, {5.0, 0.0, 11.0}, {0.0, 0.0, 6.0}}},
        {"ConstantAddMulX", Graphs::ConstantAddMulX, {{2.0, 0.0, 10.0}, {3.0, 0.0, 15.0}, {0.0, 0.0, 0.0}}},
        {"DuplicateAddXY", Graphs::DuplicateAddXY, {{1.0, 2.0, 6.0}, {3.0, 4.0, 14.0}, {0.0, 0.0, 0.0}}},
        {"DuplicateExpX", Graphs::DuplicateExpX, {std::make_tuple(0.0, 0.0, 2.0), std::make_tuple(1.0, 0.0, 2.0 * std::exp(1.0))}},
        {"AddXZero", Graphs::AddXZero, {{5.0, 0.0, 5.0}, {-3.0, 0.0, -3.0}, {0.0, 0.0, 0.0}}},
        {"MulXOne", Graphs::MulXOne, {{5.0, 0.0, 5.0}, {-3.0, 0.0, -3.0}, {0.0, 0.0, 0.0}}},
        {"MulXZero", Graphs::MulXZero, {{5.0, 0.0, 0.0}, {-3.0, 0.0, 0.0}, {100.0, 0.0, 0.0}}},
        {"SubXX", Graphs::SubXX, {{5.0, 0.0, 0.0}, {-3.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}},
        {"DivXX", Graphs::DivXX, {{5.0, 0.0, 1.0}, {-3.0, 0.0, 1.0}, {100.0, 0.0, 1.0}}},
        {"OneOverExpX", Graphs::OneOverExpX, {std::make_tuple(0.0, 0.0, 1.0), std::make_tuple(1.0, 0.0, std::exp(-1.0)), std::make_tuple(-1.0, 0.0, std::exp(1.0))}},
        {"DifferenceOfSquaresFactored", Graphs::DifferenceOfSquaresFactored, {{3.0, 2.0, 5.0}, {5.0, 3.0, 16.0}, {2.0, 2.0, 0.0}}},
        {"NestedConstantFolding", Graphs::NestedConstantFolding, {{1.0, 0.0, 9.0}, {2.0, 0.0, 10.0}, {0.0, 0.0, 8.0}}},
        {"MultipleAlgebraic", Graphs::MultipleAlgebraic, {{1.0, 2.0, 3.0}, {5.0, 3.0, 8.0}, {0.0, 0.0, 0.0}}},
        {"SinXPlusCosY", Graphs::SinXPlusCosY, {std::make_tuple(0.0, 0.0, 1.0), std::make_tuple(M_PI / 2, 0.0, 2.0), std::make_tuple(0.0, M_PI, -1.0)}},
        {"LogExpX", Graphs::LogExpX, {{1.0, 0.0, 1.0}, {2.0, 0.0, 2.0}, {0.5, 0.0, 0.5}}},
        {"LogExpXPlus2", Graphs::LogExpXPlus2, {{1.0, 0.0, 3.0}, {0.0, 0.0, 2.0}, {-2.0, 0.0, 0.0}}},  // x+2
        {"LogExpXTimes2", Graphs::LogExpXTimes2, {{1.0, 0.0, 2.0}, {2.0, 0.0, 4.0}, {0.5, 0.0, 1.0}}},  // x*2
        {"LogExpXPlus2Times3", Graphs::LogExpXPlus2Times3, {{1.0, 0.0, 9.0}, {0.0, 0.0, 6.0}, {-2.0, 0.0, 0.0}}},  // (x+2)*3
        {"SqrtXSquared", Graphs::SqrtXSquared, {{4.0, 0.0, 4.0}, {9.0, 0.0, 9.0}, {1.0, 0.0, 1.0}}},
        // Algebraic simplification coverage
        {"ZeroPlusX", Graphs::ZeroPlusX, {{5.0, 0.0, 5.0}, {-3.0, 0.0, -3.0}, {0.0, 0.0, 0.0}}},
        {"SubXZero", Graphs::SubXZero, {{5.0, 0.0, 5.0}, {-3.0, 0.0, -3.0}, {0.0, 0.0, 0.0}}},
        {"DivXOne", Graphs::DivXOne, {{5.0, 0.0, 5.0}, {-3.0, 0.0, -3.0}, {0.0, 0.0, 0.0}}},
        {"DoubleNegation", Graphs::DoubleNegation, {{5.0, 0.0, 5.0}, {-3.0, 0.0, -3.0}, {0.0, 0.0, 0.0}}},
        {"SquareZero", Graphs::SquareZero, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {5.0, 0.0, 0.0}}},
        {"SquareOne", Graphs::SquareOne, {{0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {5.0, 0.0, 1.0}}},
        {"SqrtZero", Graphs::SqrtZero, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {5.0, 0.0, 0.0}}},
        {"SqrtOne", Graphs::SqrtOne, {{0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {5.0, 0.0, 1.0}}},
        {"ExpZero", Graphs::ExpZero, {{0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {5.0, 0.0, 1.0}}},
        {"LogOne", Graphs::LogOne, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {5.0, 0.0, 0.0}}},
    };
}

// ============================================================================
// Optimization configs
// ============================================================================
GraphOptimizer::OptimizationConfig MakeConfig(bool inactive, bool cse, bool algebraic, bool stability, bool cleanup) {
    GraphOptimizer::OptimizationConfig config{};
    config.enableInactiveFolding = inactive;
    config.enableCSE = cse;
    config.enableAlgebraicSimplification = algebraic;
    config.enableStabilityCleaning = stability;
    config.enableConstantCleanup = cleanup;
    return config;
}

// ============================================================================
// Test fixture - single class, all tests grouped under GraphOptimizationTest
// ============================================================================
class GraphOptimizationTest : public ::testing::Test {
protected:
    void runAllGraphsWithConfig(const GraphOptimizer::OptimizationConfig& config, const std::string& configName) {
        const double tolerance = 1e-9;
        auto graphs = GetAllGraphs();
        int passed = 0;
        int failed = 0;

        for (const auto& testCase : graphs) {
            Graph originalGraph;
            testCase.build(originalGraph);

            GraphOptimizer optimizer;
            optimizer.setConfig(config);
            Graph optimizedGraph = optimizer.optimize(originalGraph);

            bool graphPassed = true;
            std::string failureMsg;

            for (const auto& [inputX, inputY, expected] : testCase.testData) {
                double optimizedResult = executeKernel(optimizedGraph, inputX, inputY);
                double nonOptimizedResult = executeKernel(originalGraph, inputX, inputY);

                if (std::abs(optimizedResult - expected) > tolerance) {
                    graphPassed = false;
                    failureMsg = "expected " + std::to_string(expected) + ", got " + std::to_string(optimizedResult);
                    EXPECT_NEAR(optimizedResult, expected, tolerance)
                        << "Config: " << configName
                        << ", Graph: " << testCase.name
                        << ", inputs: (" << inputX << ", " << inputY << ")"
                        << ", expected: " << expected
                        << ", got: " << optimizedResult;
                }

                if (std::abs(optimizedResult - nonOptimizedResult) > tolerance) {
                    graphPassed = false;
                    if (failureMsg.empty()) {
                        failureMsg = "optimized vs non-optimized mismatch";
                    }
                    EXPECT_NEAR(optimizedResult, nonOptimizedResult, tolerance)
                        << "Config: " << configName
                        << ", Graph: " << testCase.name
                        << " - optimized vs non-optimized mismatch"
                        << ", inputs: (" << inputX << ", " << inputY << ")"
                        << ", optimized: " << optimizedResult
                        << ", non-optimized: " << nonOptimizedResult;
                }
            }

            if (graphPassed) {
                std::cout << "  [PASS] " << testCase.name << "\n";
                passed++;
            } else {
                std::cout << "  [FAIL] " << testCase.name << " - " << failureMsg << "\n";
                failed++;
            }
        }

        std::cout << "  ----\n";
        std::cout << "  " << passed << " passed, " << failed << " failed\n";
    }
};

// Conf_FFFFF - no optimizations
TEST_F(GraphOptimizationTest, Conf_FFFFF) {
    runAllGraphsWithConfig(MakeConfig(false, false, false, false, false), "Conf_FFFFF");
}

// Conf_FFFFT - cleanup only
TEST_F(GraphOptimizationTest, Conf_FFFFT) {
    runAllGraphsWithConfig(MakeConfig(false, false, false, false, true), "Conf_FFFFT");
}

// Conf_FFFTF - stability only
TEST_F(GraphOptimizationTest, Conf_FFFTF) {
    runAllGraphsWithConfig(MakeConfig(false, false, false, true, false), "Conf_FFFTF");
}

// Conf_FFFTT - stability + cleanup
TEST_F(GraphOptimizationTest, Conf_FFFTT) {
    runAllGraphsWithConfig(MakeConfig(false, false, false, true, true), "Conf_FFFTT");
}

// Conf_FFTFF - algebraic only
TEST_F(GraphOptimizationTest, Conf_FFTFF) {
    runAllGraphsWithConfig(MakeConfig(false, false, true, false, false), "Conf_FFTFF");
}

// Conf_FTFFF - CSE only
TEST_F(GraphOptimizationTest, Conf_FTFFF) {
    runAllGraphsWithConfig(MakeConfig(false, true, false, false, false), "Conf_FTFFF");
}

// Conf_TFFFF - inactive folding only
TEST_F(GraphOptimizationTest, Conf_TFFFF) {
    runAllGraphsWithConfig(MakeConfig(true, false, false, false, false), "Conf_TFFFF");
}

// Conf_TTTTT - all enabled
TEST_F(GraphOptimizationTest, Conf_TTTTT) {
    runAllGraphsWithConfig(MakeConfig(true, true, true, true, true), "Conf_TTTTT");
}

// ============================================================================
// Debug test to understand LogExpX failure
// ============================================================================
TEST_F(GraphOptimizationTest, DebugLogExpX) {
    Graph originalGraph;
    Graphs::LogExpX(originalGraph);

    std::cout << "\n=== Original Graph ===" << std::endl;
    std::cout << "Nodes: " << originalGraph.nodes.size() << std::endl;
    for (size_t i = 0; i < originalGraph.nodes.size(); ++i) {
        const auto& n = originalGraph.nodes[i];
        std::cout << "  [" << i << "] op=" << static_cast<int>(n.op)
                  << " a=" << n.a << " b=" << n.b << " dst=" << n.dst << std::endl;
    }
    std::cout << "Outputs: ";
    for (auto o : originalGraph.outputs) std::cout << o << " ";
    std::cout << std::endl;

    // Apply stability cleaning only
    GraphOptimizer optimizer;
    auto config = MakeConfig(false, false, false, true, false);  // stability only
    optimizer.setConfig(config);
    Graph optimizedGraph = optimizer.optimize(originalGraph);

    std::cout << "\n=== Optimized Graph (stability cleaning) ===" << std::endl;
    std::cout << "Nodes: " << optimizedGraph.nodes.size() << std::endl;
    for (size_t i = 0; i < optimizedGraph.nodes.size(); ++i) {
        const auto& n = optimizedGraph.nodes[i];
        std::cout << "  [" << i << "] op=" << static_cast<int>(n.op)
                  << " a=" << n.a << " b=" << n.b << " dst=" << n.dst << std::endl;
    }
    std::cout << "Outputs: ";
    for (auto o : optimizedGraph.outputs) std::cout << o << " ";
    std::cout << std::endl;

    // Try to execute
    double result = executeKernel(optimizedGraph, 1.0, 0.0);
    std::cout << "\nResult for x=1: " << result << " (expected 1)" << std::endl;
}

// ============================================================================
// Edge case tests
// ============================================================================
TEST_F(GraphOptimizationTest, EmptyGraphOptimization) {
    Graph graph;
    GraphOptimizer optimizer;
    Graph optimized = optimizer.optimize(graph);
    EXPECT_TRUE(optimized.empty());
}

TEST_F(GraphOptimizationTest, SingleInputGraph) {
    Graph graph;
    NodeId x = graph.addInput();
    graph.markOutput(x);

    GraphOptimizer optimizer;
    Graph optimized = optimizer.optimize(graph);
    EXPECT_EQ(optimized.nodes.size(), 1);
    EXPECT_EQ(optimized.outputs.size(), 1);
}

TEST_F(GraphOptimizationTest, SingleConstantGraph) {
    Graph graph;
    NodeId c = graph.addConstant(42.0);
    graph.markOutput(c);

    GraphOptimizer optimizer;
    Graph optimized = optimizer.optimize(graph);
    EXPECT_EQ(optimized.outputs.size(), 1);
}

TEST_F(GraphOptimizationTest, MultiplePassesConverge) {
    Graph graph;
    NodeId x = graph.addInput();
    NodeId zero = graph.addConstant(0.0);

    NodeId add1 = addBinaryOp(graph, OpCode::Add, x, zero);
    NodeId add2 = addBinaryOp(graph, OpCode::Add, add1, zero);
    NodeId add3 = addBinaryOp(graph, OpCode::Add, add2, zero);
    graph.markOutput(add3);

    GraphOptimizer optimizer;
    GraphOptimizer::OptimizationConfig config;
    config.enableAlgebraicSimplification = true;
    config.maxOptimizationPasses = 10;
    optimizer.setConfig(config);

    Graph optimized = optimizer.optimize(graph);

    const auto& stats = optimizer.getLastStats();
    EXPECT_LE(stats.passesPerformed, config.maxOptimizationPasses);
}

// ============================================================================
// GraphStabilityCleaningTest - Stress tests demonstrating numerical instability
// These tests use extreme values that cause overflow/NaN without stability cleaning
// ============================================================================
namespace StabilityGraphs {

// 1/exp(x) with large x causes exp(x) to overflow, but exp(-x) is stable
inline void OneOverExpX(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId expX = addUnaryOp(g, OpCode::Exp, x);
    g.markOutput(addBinaryOp(g, OpCode::Div, g.addConstant(1.0), expX));
}

// exp(x)/exp(y) with large x,y causes overflow, but exp(x-y) is stable
inline void ExpXDivExpY(Graph& g) {
    NodeId x = g.addInput();
    NodeId y = g.addInput();
    NodeId expX = addUnaryOp(g, OpCode::Exp, x);
    NodeId expY = addUnaryOp(g, OpCode::Exp, y);
    g.markOutput(addBinaryOp(g, OpCode::Div, expX, expY));
}

// log(exp(x)) with large x causes exp(x) to overflow, but x is stable
inline void LogExpX(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId expX = addUnaryOp(g, OpCode::Exp, x);
    g.markOutput(addUnaryOp(g, OpCode::Log, expX));
}

// sqrt(x*x) with large x causes x*x to overflow, but abs(x) is stable
inline void SqrtXSquared(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId xSquared = addBinaryOp(g, OpCode::Mul, x, x);
    g.markOutput(addUnaryOp(g, OpCode::Sqrt, xSquared));
}

} // namespace StabilityGraphs

class GraphStabilityCleaningTest : public ::testing::Test {
protected:
    // Helper to run with and without stability cleaning
    struct StabilityTestResult {
        double withStability;
        double withoutStability;
        bool withStabilityValid;
        bool withoutStabilityValid;
    };

    StabilityTestResult runWithAndWithoutStability(std::function<void(Graph&)> buildGraph, double x, double y) {
        Graph graph;
        buildGraph(graph);

        // Without stability cleaning
        GraphOptimizer optimizerNoStability;
        auto configNoStability = MakeConfig(false, false, false, false, false);
        optimizerNoStability.setConfig(configNoStability);
        Graph graphNoStability = optimizerNoStability.optimize(graph);

        // With stability cleaning
        GraphOptimizer optimizerWithStability;
        auto configWithStability = MakeConfig(false, false, false, true, false);
        optimizerWithStability.setConfig(configWithStability);
        Graph graphWithStability = optimizerWithStability.optimize(graph);

        StabilityTestResult result;
        result.withoutStability = executeKernel(graphNoStability, x, y);
        result.withStability = executeKernel(graphWithStability, x, y);
        result.withoutStabilityValid = std::isfinite(result.withoutStability);
        result.withStabilityValid = std::isfinite(result.withStability);

        return result;
    }
};

// ============================================================================
// Pattern 1: 1/exp(x) -> exp(-x)
// ============================================================================

// Large positive x: exp(750) overflows, 1/inf = 0 (wrong), but exp(-750) is tiny but finite
TEST_F(GraphStabilityCleaningTest, OneOverExpX_LargePositive) {
    const double x = 750.0;
    const double expectedResult = std::exp(-x);

    auto result = runWithAndWithoutStability(StabilityGraphs::OneOverExpX, x, 0.0);

    std::cout << "  1/exp(" << x << "):\n";
    std::cout << "    Without stability: " << result.withoutStability
              << (result.withoutStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    With stability:    " << result.withStability
              << (result.withStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    Expected:          " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid) << "With stability cleaning, result should be finite";
    // Note: Without stability, exp(750)=inf, 1/inf=0 which is finite but wrong
}

// Large negative x: exp(-750) underflows to 0, 1/0 = inf, but exp(750) = inf
TEST_F(GraphStabilityCleaningTest, OneOverExpX_LargeNegative) {
    const double x = -750.0;
    const double expectedResult = std::exp(-x);  // exp(750) = inf

    auto result = runWithAndWithoutStability(StabilityGraphs::OneOverExpX, x, 0.0);

    std::cout << "  1/exp(" << x << "):\n";
    std::cout << "    Without stability: " << result.withoutStability
              << (result.withoutStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    With stability:    " << result.withStability
              << (result.withStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    Expected:          " << expectedResult << " (inf)\n";

    // Both should produce inf for this case (exp(-(-750)) = exp(750) = inf)
    EXPECT_FALSE(result.withStabilityValid) << "exp(750) should be inf";
    EXPECT_FALSE(result.withoutStabilityValid) << "1/exp(-750) = 1/0 = inf";
}

// Moderate positive x: both approaches work
TEST_F(GraphStabilityCleaningTest, OneOverExpX_ModeratePositive) {
    const double x = 5.0;
    const double expectedResult = std::exp(-x);

    auto result = runWithAndWithoutStability(StabilityGraphs::OneOverExpX, x, 0.0);

    std::cout << "  1/exp(" << x << "):\n";
    std::cout << "    Without stability: " << result.withoutStability << "\n";
    std::cout << "    With stability:    " << result.withStability << "\n";
    std::cout << "    Expected:          " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid);
    EXPECT_TRUE(result.withoutStabilityValid);
    EXPECT_NEAR(result.withStability, expectedResult, 1e-12);
    EXPECT_NEAR(result.withoutStability, expectedResult, 1e-12);
}

// Moderate negative x: both approaches work
TEST_F(GraphStabilityCleaningTest, OneOverExpX_ModerateNegative) {
    const double x = -2.0;
    const double expectedResult = std::exp(-x);  // exp(2) ~ 7.389

    auto result = runWithAndWithoutStability(StabilityGraphs::OneOverExpX, x, 0.0);

    std::cout << "  1/exp(" << x << "):\n";
    std::cout << "    Without stability: " << result.withoutStability << "\n";
    std::cout << "    With stability:    " << result.withStability << "\n";
    std::cout << "    Expected:          " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid);
    EXPECT_TRUE(result.withoutStabilityValid);
    EXPECT_NEAR(result.withStability, expectedResult, 1e-12);
    EXPECT_NEAR(result.withoutStability, expectedResult, 1e-12);
}

// ============================================================================
// Pattern 2: exp(x)/exp(y) -> exp(x-y)
// ============================================================================

// Both large, but difference is small
TEST_F(GraphStabilityCleaningTest, ExpXDivExpY_BothLargePositive) {
    const double x = 800.0;
    const double y = 795.0;  // x - y = 5, so exp(5) ~ 148.4
    const double expectedResult = std::exp(x - y);

    auto result = runWithAndWithoutStability(StabilityGraphs::ExpXDivExpY, x, y);

    std::cout << "  exp(" << x << ")/exp(" << y << "):\n";
    std::cout << "    Without stability: " << result.withoutStability
              << (result.withoutStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    With stability:    " << result.withStability
              << (result.withStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    Expected (exp(" << x - y << ")): " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid) << "With stability cleaning, result should be finite";
    if (result.withStabilityValid) {
        EXPECT_NEAR(result.withStability, expectedResult, expectedResult * 1e-9);
    }
    EXPECT_FALSE(result.withoutStabilityValid)
        << "Without stability cleaning, exp(800)/exp(795) should be NaN (inf/inf)";
}

// Both large negative
TEST_F(GraphStabilityCleaningTest, ExpXDivExpY_BothLargeNegative) {
    const double x = -795.0;
    const double y = -800.0;  // x - y = 5, so exp(5) ~ 148.4
    const double expectedResult = std::exp(x - y);

    auto result = runWithAndWithoutStability(StabilityGraphs::ExpXDivExpY, x, y);

    std::cout << "  exp(" << x << ")/exp(" << y << "):\n";
    std::cout << "    Without stability: " << result.withoutStability
              << (result.withoutStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    With stability:    " << result.withStability
              << (result.withStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    Expected (exp(" << x - y << ")): " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid) << "With stability cleaning, result should be finite";
    if (result.withStabilityValid) {
        EXPECT_NEAR(result.withStability, expectedResult, expectedResult * 1e-9);
    }
    // Without stability, both exp(-795) and exp(-800) underflow to 0, so 0/0 = NaN
    EXPECT_FALSE(result.withoutStabilityValid)
        << "Without stability cleaning, exp(-795)/exp(-800) should be NaN (0/0)";
}

// Moderate values
TEST_F(GraphStabilityCleaningTest, ExpXDivExpY_Moderate) {
    const double x = 3.0;
    const double y = 1.0;
    const double expectedResult = std::exp(x - y);  // exp(2) ~ 7.389

    auto result = runWithAndWithoutStability(StabilityGraphs::ExpXDivExpY, x, y);

    std::cout << "  exp(" << x << ")/exp(" << y << "):\n";
    std::cout << "    Without stability: " << result.withoutStability << "\n";
    std::cout << "    With stability:    " << result.withStability << "\n";
    std::cout << "    Expected:          " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid);
    EXPECT_TRUE(result.withoutStabilityValid);
    EXPECT_NEAR(result.withStability, expectedResult, 1e-12);
    EXPECT_NEAR(result.withoutStability, expectedResult, 1e-12);
}

// ============================================================================
// Pattern 3: log(exp(x)) -> x
// ============================================================================

// Large positive x: exp overflows
TEST_F(GraphStabilityCleaningTest, LogExpX_LargePositive) {
    const double x = 750.0;
    const double expectedResult = x;

    auto result = runWithAndWithoutStability(StabilityGraphs::LogExpX, x, 0.0);

    std::cout << "  log(exp(" << x << ")):\n";
    std::cout << "    Without stability: " << result.withoutStability
              << (result.withoutStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    With stability:    " << result.withStability
              << (result.withStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    Expected:          " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid) << "With stability cleaning, result should be finite";
    if (result.withStabilityValid) {
        EXPECT_DOUBLE_EQ(result.withStability, expectedResult);
    }
    EXPECT_FALSE(result.withoutStabilityValid)
        << "Without stability cleaning, log(exp(750)) should be inf";
}

// Large negative x: exp underflows to 0, log(0) = -inf
TEST_F(GraphStabilityCleaningTest, LogExpX_LargeNegative) {
    const double x = -750.0;
    const double expectedResult = x;

    auto result = runWithAndWithoutStability(StabilityGraphs::LogExpX, x, 0.0);

    std::cout << "  log(exp(" << x << ")):\n";
    std::cout << "    Without stability: " << result.withoutStability
              << (result.withoutStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    With stability:    " << result.withStability
              << (result.withStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    Expected:          " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid) << "With stability cleaning, result should be finite";
    if (result.withStabilityValid) {
        EXPECT_DOUBLE_EQ(result.withStability, expectedResult);
    }
    EXPECT_FALSE(result.withoutStabilityValid)
        << "Without stability cleaning, log(exp(-750)) = log(0) = -inf";
}

// Moderate value
TEST_F(GraphStabilityCleaningTest, LogExpX_Moderate) {
    const double x = 3.5;
    const double expectedResult = x;

    auto result = runWithAndWithoutStability(StabilityGraphs::LogExpX, x, 0.0);

    std::cout << "  log(exp(" << x << ")):\n";
    std::cout << "    Without stability: " << result.withoutStability << "\n";
    std::cout << "    With stability:    " << result.withStability << "\n";
    std::cout << "    Expected:          " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid);
    EXPECT_TRUE(result.withoutStabilityValid);
    EXPECT_NEAR(result.withStability, expectedResult, 1e-12);
    EXPECT_NEAR(result.withoutStability, expectedResult, 1e-12);
}

// ============================================================================
// Pattern 4: sqrt(x*x) -> abs(x)
// ============================================================================

// Large positive x: x*x overflows
TEST_F(GraphStabilityCleaningTest, SqrtXSquared_LargePositive) {
    const double x = 1e200;
    const double expectedResult = std::abs(x);

    auto result = runWithAndWithoutStability(StabilityGraphs::SqrtXSquared, x, 0.0);

    std::cout << "  sqrt(" << x << "^2):\n";
    std::cout << "    Without stability: " << result.withoutStability
              << (result.withoutStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    With stability:    " << result.withStability
              << (result.withStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    Expected:          " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid) << "With stability cleaning, result should be finite";
    if (result.withStabilityValid) {
        EXPECT_DOUBLE_EQ(result.withStability, expectedResult);
    }
    EXPECT_FALSE(result.withoutStabilityValid)
        << "Without stability cleaning, sqrt((1e200)^2) should be inf";
}

// Large negative x: x*x overflows
TEST_F(GraphStabilityCleaningTest, SqrtXSquared_LargeNegative) {
    const double x = -1e200;
    const double expectedResult = std::abs(x);

    auto result = runWithAndWithoutStability(StabilityGraphs::SqrtXSquared, x, 0.0);

    std::cout << "  sqrt((" << x << ")^2):\n";
    std::cout << "    Without stability: " << result.withoutStability
              << (result.withoutStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    With stability:    " << result.withStability
              << (result.withStabilityValid ? " (finite)" : " (inf/NaN)") << "\n";
    std::cout << "    Expected:          " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid) << "With stability cleaning, result should be finite";
    if (result.withStabilityValid) {
        EXPECT_DOUBLE_EQ(result.withStability, expectedResult);
    }
    EXPECT_FALSE(result.withoutStabilityValid);
}

// Moderate positive
TEST_F(GraphStabilityCleaningTest, SqrtXSquared_ModeratePositive) {
    const double x = 5.0;
    const double expectedResult = std::abs(x);

    auto result = runWithAndWithoutStability(StabilityGraphs::SqrtXSquared, x, 0.0);

    std::cout << "  sqrt(" << x << "^2):\n";
    std::cout << "    Without stability: " << result.withoutStability << "\n";
    std::cout << "    With stability:    " << result.withStability << "\n";
    std::cout << "    Expected:          " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid);
    EXPECT_TRUE(result.withoutStabilityValid);
    EXPECT_NEAR(result.withStability, expectedResult, 1e-12);
    EXPECT_NEAR(result.withoutStability, expectedResult, 1e-12);
}

// Moderate negative
TEST_F(GraphStabilityCleaningTest, SqrtXSquared_ModerateNegative) {
    const double x = -7.0;
    const double expectedResult = std::abs(x);

    auto result = runWithAndWithoutStability(StabilityGraphs::SqrtXSquared, x, 0.0);

    std::cout << "  sqrt((" << x << ")^2):\n";
    std::cout << "    Without stability: " << result.withoutStability << "\n";
    std::cout << "    With stability:    " << result.withStability << "\n";
    std::cout << "    Expected:          " << expectedResult << "\n";

    EXPECT_TRUE(result.withStabilityValid);
    EXPECT_TRUE(result.withoutStabilityValid);
    EXPECT_NEAR(result.withStability, expectedResult, 1e-12);
    EXPECT_NEAR(result.withoutStability, expectedResult, 1e-12);
}

// ============================================================================
// GraphInactiveFoldingTest - Tests for inactive (constant) subgraph folding
// These tests verify that constant subgraphs are correctly evaluated and folded
// ============================================================================
namespace InactiveFoldingGraphs {

// Helper to add inactive (constant-only) binary op
inline NodeId addInactiveBinaryOp(Graph& g, OpCode op, NodeId a, NodeId b) {
    Node node{};
    node.op = op;
    node.a = a;
    node.b = b;
    node.isActive = false;  // Force inactive for constant folding
    return g.addNode(node);
}

// Helper to add inactive (constant-only) unary op
inline NodeId addInactiveUnaryOp(Graph& g, OpCode op, NodeId a) {
    Node node{};
    node.op = op;
    node.a = a;
    node.isActive = false;  // Force inactive for constant folding
    return g.addNode(node);
}

// Helper to add inactive ternary op (If)
inline NodeId addInactiveTernaryOp(Graph& g, OpCode op, NodeId a, NodeId b, NodeId c) {
    Node node{};
    node.op = op;
    node.a = a;
    node.b = b;
    node.c = c;
    node.isActive = false;
    return g.addNode(node);
}

// x + (constant Sub: 5 - 3) => x + 2
inline void ConstantSub(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c5 = g.addConstant(5.0);
    NodeId c3 = g.addConstant(3.0);
    NodeId sub = addInactiveBinaryOp(g, OpCode::Sub, c5, c3);  // 5 - 3 = 2
    g.markOutput(addBinaryOp(g, OpCode::Add, x, sub));  // x + 2
}

// x + (constant Neg: -3) => x + (-3)
inline void ConstantNeg(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c3 = g.addConstant(3.0);
    NodeId neg = addInactiveUnaryOp(g, OpCode::Neg, c3);  // -3
    g.markOutput(addBinaryOp(g, OpCode::Add, x, neg));  // x + (-3)
}

// x + (constant Exp: exp(1)) => x + e
inline void ConstantExp(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c1 = g.addConstant(1.0);
    NodeId expC = addInactiveUnaryOp(g, OpCode::Exp, c1);  // e^1 = e
    g.markOutput(addBinaryOp(g, OpCode::Add, x, expC));  // x + e
}

// x + (constant Log: log(e)) => x + 1
inline void ConstantLog(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId ce = g.addConstant(std::exp(1.0));
    NodeId logC = addInactiveUnaryOp(g, OpCode::Log, ce);  // log(e) = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, logC));  // x + 1
}

// x + (constant Sqrt: sqrt(16)) => x + 4
inline void ConstantSqrt(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c16 = g.addConstant(16.0);
    NodeId sqrtC = addInactiveUnaryOp(g, OpCode::Sqrt, c16);  // sqrt(16) = 4
    g.markOutput(addBinaryOp(g, OpCode::Add, x, sqrtC));  // x + 4
}

// x + (constant Square: 3^2) => x + 9
inline void ConstantSquare(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c3 = g.addConstant(3.0);
    NodeId sqrC = addInactiveUnaryOp(g, OpCode::Square, c3);  // 3^2 = 9
    g.markOutput(addBinaryOp(g, OpCode::Add, x, sqrC));  // x + 9
}

// x + (constant Recip: 1/4) => x + 0.25
inline void ConstantRecip(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c4 = g.addConstant(4.0);
    NodeId recipC = addInactiveUnaryOp(g, OpCode::Recip, c4);  // 1/4 = 0.25
    g.markOutput(addBinaryOp(g, OpCode::Add, x, recipC));  // x + 0.25
}

// x + (constant Abs: abs(-5)) => x + 5
inline void ConstantAbs(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId cm5 = g.addConstant(-5.0);
    NodeId absC = addInactiveUnaryOp(g, OpCode::Abs, cm5);  // abs(-5) = 5
    g.markOutput(addBinaryOp(g, OpCode::Add, x, absC));  // x + 5
}

// x + (constant Sin: sin(0)) => x + 0
inline void ConstantSin(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c0 = g.addConstant(0.0);
    NodeId sinC = addInactiveUnaryOp(g, OpCode::Sin, c0);  // sin(0) = 0
    g.markOutput(addBinaryOp(g, OpCode::Add, x, sinC));  // x + 0
}

// x + (constant Cos: cos(0)) => x + 1
inline void ConstantCos(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c0 = g.addConstant(0.0);
    NodeId cosC = addInactiveUnaryOp(g, OpCode::Cos, c0);  // cos(0) = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, cosC));  // x + 1
}

// x + (constant Tan: tan(0)) => x + 0
inline void ConstantTan(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c0 = g.addConstant(0.0);
    NodeId tanC = addInactiveUnaryOp(g, OpCode::Tan, c0);  // tan(0) = 0
    g.markOutput(addBinaryOp(g, OpCode::Add, x, tanC));  // x + 0
}

// x + (constant Pow: 2^3) => x + 8
inline void ConstantPow(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c2 = g.addConstant(2.0);
    NodeId c3 = g.addConstant(3.0);
    NodeId powC = addInactiveBinaryOp(g, OpCode::Pow, c2, c3);  // 2^3 = 8
    g.markOutput(addBinaryOp(g, OpCode::Add, x, powC));  // x + 8
}

// x + (constant Min: min(3, 7)) => x + 3
inline void ConstantMin(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c3 = g.addConstant(3.0);
    NodeId c7 = g.addConstant(7.0);
    NodeId minC = addInactiveBinaryOp(g, OpCode::Min, c3, c7);  // min(3, 7) = 3
    g.markOutput(addBinaryOp(g, OpCode::Add, x, minC));  // x + 3
}

// x + (constant Max: max(3, 7)) => x + 7
inline void ConstantMax(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c3 = g.addConstant(3.0);
    NodeId c7 = g.addConstant(7.0);
    NodeId maxC = addInactiveBinaryOp(g, OpCode::Max, c3, c7);  // max(3, 7) = 7
    g.markOutput(addBinaryOp(g, OpCode::Add, x, maxC));  // x + 7
}

// x + (constant CmpLT: 2 < 5) => x + 1
inline void ConstantCmpLT(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c2 = g.addConstant(2.0);
    NodeId c5 = g.addConstant(5.0);
    NodeId cmpC = addInactiveBinaryOp(g, OpCode::CmpLT, c2, c5);  // 2 < 5 = true = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, cmpC));  // x + 1
}

// x + (constant CmpLE: 5 <= 5) => x + 1
inline void ConstantCmpLE(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c5a = g.addConstant(5.0);
    NodeId c5b = g.addConstant(5.0);
    NodeId cmpC = addInactiveBinaryOp(g, OpCode::CmpLE, c5a, c5b);  // 5 <= 5 = true = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, cmpC));  // x + 1
}

// x + (constant CmpGT: 7 > 3) => x + 1
inline void ConstantCmpGT(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c7 = g.addConstant(7.0);
    NodeId c3 = g.addConstant(3.0);
    NodeId cmpC = addInactiveBinaryOp(g, OpCode::CmpGT, c7, c3);  // 7 > 3 = true = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, cmpC));  // x + 1
}

// x + (constant CmpGE: 3 >= 3) => x + 1
inline void ConstantCmpGE(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c3a = g.addConstant(3.0);
    NodeId c3b = g.addConstant(3.0);
    NodeId cmpC = addInactiveBinaryOp(g, OpCode::CmpGE, c3a, c3b);  // 3 >= 3 = true = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, cmpC));  // x + 1
}

// x + (constant CmpEQ: 4 == 4) => x + 1
inline void ConstantCmpEQ(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c4a = g.addConstant(4.0);
    NodeId c4b = g.addConstant(4.0);
    NodeId cmpC = addInactiveBinaryOp(g, OpCode::CmpEQ, c4a, c4b);  // 4 == 4 = true = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, cmpC));  // x + 1
}

// x + (constant CmpNE: 2 != 5) => x + 1
inline void ConstantCmpNE(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c2 = g.addConstant(2.0);
    NodeId c5 = g.addConstant(5.0);
    NodeId cmpC = addInactiveBinaryOp(g, OpCode::CmpNE, c2, c5);  // 2 != 5 = true = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, cmpC));  // x + 1
}

// x + (constant If: if(1, 10, 20)) => x + 10
inline void ConstantIfTrue(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId cond = g.addConstant(1.0);   // true
    NodeId ctrue = g.addConstant(10.0);
    NodeId cfalse = g.addConstant(20.0);
    NodeId ifC = addInactiveTernaryOp(g, OpCode::If, cond, ctrue, cfalse);  // if(true, 10, 20) = 10
    g.markOutput(addBinaryOp(g, OpCode::Add, x, ifC));  // x + 10
}

// x + (constant If: if(0, 10, 20)) => x + 20
inline void ConstantIfFalse(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId cond = g.addConstant(0.0);   // false
    NodeId ctrue = g.addConstant(10.0);
    NodeId cfalse = g.addConstant(20.0);
    NodeId ifC = addInactiveTernaryOp(g, OpCode::If, cond, ctrue, cfalse);  // if(false, 10, 20) = 20
    g.markOutput(addBinaryOp(g, OpCode::Add, x, ifC));  // x + 20
}

// x + (constant BoolAnd: 1 && 1) => x + 1
inline void ConstantBoolAnd(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c1a = g.addConstant(1.0);
    NodeId c1b = g.addConstant(1.0);
    NodeId andC = addInactiveBinaryOp(g, OpCode::BoolAnd, c1a, c1b);  // 1 && 1 = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, andC));  // x + 1
}

// x + (constant BoolOr: 0 || 1) => x + 1
inline void ConstantBoolOr(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c0 = g.addConstant(0.0);
    NodeId c1 = g.addConstant(1.0);
    NodeId orC = addInactiveBinaryOp(g, OpCode::BoolOr, c0, c1);  // 0 || 1 = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, orC));  // x + 1
}

// x + (constant BoolNot: !0) => x + 1
inline void ConstantBoolNot(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c0 = g.addConstant(0.0);
    NodeId notC = addInactiveUnaryOp(g, OpCode::BoolNot, c0);  // !0 = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, notC));  // x + 1
}

// x + (constant BoolEq: (1 != 0) == (1 != 0)) => x + 1
inline void ConstantBoolEq(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c1a = g.addConstant(1.0);
    NodeId c1b = g.addConstant(1.0);
    NodeId eqC = addInactiveBinaryOp(g, OpCode::BoolEq, c1a, c1b);  // true == true = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, eqC));  // x + 1
}

// x + (constant BoolNe: (1 != 0) != (0 != 0)) => x + 1
inline void ConstantBoolNe(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c1 = g.addConstant(1.0);
    NodeId c0 = g.addConstant(0.0);
    NodeId neC = addInactiveBinaryOp(g, OpCode::BoolNe, c1, c0);  // true != false = 1
    g.markOutput(addBinaryOp(g, OpCode::Add, x, neC));  // x + 1
}

// Nested constant subgraph: x + ((2 + 3) * 4) => x + 20
// This tests the markProcessed recursive function that marks child nodes
inline void NestedConstantSubgraph(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c2 = g.addConstant(2.0);
    NodeId c3 = g.addConstant(3.0);
    NodeId c4 = g.addConstant(4.0);
    // Build: (2 + 3) * 4 = 20
    NodeId add23 = addInactiveBinaryOp(g, OpCode::Add, c2, c3);  // 2 + 3 = 5
    NodeId mul = addInactiveBinaryOp(g, OpCode::Mul, add23, c4);  // 5 * 4 = 20
    g.markOutput(addBinaryOp(g, OpCode::Add, x, mul));  // x + 20
}

// Deeply nested: x + (((1 + 2) + 3) + 4) => x + 10
// Tests multiple levels of recursive markProcessed
inline void DeeplyNestedConstantSubgraph(Graph& g) {
    NodeId x = g.addInput();
    g.addInput();
    NodeId c1 = g.addConstant(1.0);
    NodeId c2 = g.addConstant(2.0);
    NodeId c3 = g.addConstant(3.0);
    NodeId c4 = g.addConstant(4.0);
    // Build: ((1 + 2) + 3) + 4 = 10
    NodeId add12 = addInactiveBinaryOp(g, OpCode::Add, c1, c2);   // 1 + 2 = 3
    NodeId add123 = addInactiveBinaryOp(g, OpCode::Add, add12, c3);  // 3 + 3 = 6
    NodeId add1234 = addInactiveBinaryOp(g, OpCode::Add, add123, c4);  // 6 + 4 = 10
    g.markOutput(addBinaryOp(g, OpCode::Add, x, add1234));  // x + 10
}

} // namespace InactiveFoldingGraphs

class GraphInactiveFoldingTest : public ::testing::Test {
protected:
    // Run test with inactive folding enabled
    void runInactiveFoldingTest(std::function<void(Graph&)> buildGraph,
                                 double inputX, double inputY, double expected,
                                 const std::string& testName) {
        Graph graph;
        buildGraph(graph);

        // Enable only inactive folding
        GraphOptimizer optimizer;
        auto config = MakeConfig(true, false, false, false, false);  // Only inactive folding
        optimizer.setConfig(config);
        Graph optimizedGraph = optimizer.optimize(graph);

        double result = executeKernel(optimizedGraph, inputX, inputY);

        std::cout << "  " << testName << ": result=" << result << ", expected=" << expected << "\n";

        EXPECT_NEAR(result, expected, 1e-9) << "Test: " << testName;
    }
};

// Unary operations
TEST_F(GraphInactiveFoldingTest, ConstantSub) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantSub, 10.0, 0.0, 12.0, "ConstantSub");  // 10 + (5-3) = 12
}

TEST_F(GraphInactiveFoldingTest, ConstantNeg) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantNeg, 10.0, 0.0, 7.0, "ConstantNeg");  // 10 + (-3) = 7
}

TEST_F(GraphInactiveFoldingTest, ConstantExp) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantExp, 0.0, 0.0, std::exp(1.0), "ConstantExp");  // 0 + e
}

TEST_F(GraphInactiveFoldingTest, ConstantLog) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantLog, 0.0, 0.0, 1.0, "ConstantLog");  // 0 + log(e) = 1
}

TEST_F(GraphInactiveFoldingTest, ConstantSqrt) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantSqrt, 0.0, 0.0, 4.0, "ConstantSqrt");  // 0 + sqrt(16) = 4
}

TEST_F(GraphInactiveFoldingTest, ConstantSquare) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantSquare, 0.0, 0.0, 9.0, "ConstantSquare");  // 0 + 3^2 = 9
}

TEST_F(GraphInactiveFoldingTest, ConstantRecip) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantRecip, 0.0, 0.0, 0.25, "ConstantRecip");  // 0 + 1/4 = 0.25
}

TEST_F(GraphInactiveFoldingTest, ConstantAbs) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantAbs, 0.0, 0.0, 5.0, "ConstantAbs");  // 0 + abs(-5) = 5
}

TEST_F(GraphInactiveFoldingTest, ConstantSin) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantSin, 5.0, 0.0, 5.0, "ConstantSin");  // 5 + sin(0) = 5
}

TEST_F(GraphInactiveFoldingTest, ConstantCos) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantCos, 5.0, 0.0, 6.0, "ConstantCos");  // 5 + cos(0) = 6
}

TEST_F(GraphInactiveFoldingTest, ConstantTan) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantTan, 5.0, 0.0, 5.0, "ConstantTan");  // 5 + tan(0) = 5
}

// Binary operations
TEST_F(GraphInactiveFoldingTest, ConstantPow) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantPow, 0.0, 0.0, 8.0, "ConstantPow");  // 0 + 2^3 = 8
}

TEST_F(GraphInactiveFoldingTest, ConstantMin) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantMin, 0.0, 0.0, 3.0, "ConstantMin");  // 0 + min(3,7) = 3
}

TEST_F(GraphInactiveFoldingTest, ConstantMax) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantMax, 0.0, 0.0, 7.0, "ConstantMax");  // 0 + max(3,7) = 7
}

// Comparison operations
TEST_F(GraphInactiveFoldingTest, ConstantCmpLT) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantCmpLT, 0.0, 0.0, 1.0, "ConstantCmpLT");  // 0 + (2<5) = 1
}

TEST_F(GraphInactiveFoldingTest, ConstantCmpLE) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantCmpLE, 0.0, 0.0, 1.0, "ConstantCmpLE");  // 0 + (5<=5) = 1
}

TEST_F(GraphInactiveFoldingTest, ConstantCmpGT) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantCmpGT, 0.0, 0.0, 1.0, "ConstantCmpGT");  // 0 + (7>3) = 1
}

TEST_F(GraphInactiveFoldingTest, ConstantCmpGE) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantCmpGE, 0.0, 0.0, 1.0, "ConstantCmpGE");  // 0 + (3>=3) = 1
}

TEST_F(GraphInactiveFoldingTest, ConstantCmpEQ) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantCmpEQ, 0.0, 0.0, 1.0, "ConstantCmpEQ");  // 0 + (4==4) = 1
}

TEST_F(GraphInactiveFoldingTest, ConstantCmpNE) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantCmpNE, 0.0, 0.0, 1.0, "ConstantCmpNE");  // 0 + (2!=5) = 1
}

// Conditional operation
TEST_F(GraphInactiveFoldingTest, ConstantIfTrue) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantIfTrue, 0.0, 0.0, 10.0, "ConstantIfTrue");  // 0 + if(1,10,20) = 10
}

TEST_F(GraphInactiveFoldingTest, ConstantIfFalse) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantIfFalse, 0.0, 0.0, 20.0, "ConstantIfFalse");  // 0 + if(0,10,20) = 20
}

// Boolean operations
TEST_F(GraphInactiveFoldingTest, ConstantBoolAnd) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantBoolAnd, 0.0, 0.0, 1.0, "ConstantBoolAnd");  // 0 + (1&&1) = 1
}

TEST_F(GraphInactiveFoldingTest, ConstantBoolOr) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantBoolOr, 0.0, 0.0, 1.0, "ConstantBoolOr");  // 0 + (0||1) = 1
}

TEST_F(GraphInactiveFoldingTest, ConstantBoolNot) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantBoolNot, 0.0, 0.0, 1.0, "ConstantBoolNot");  // 0 + !0 = 1
}

TEST_F(GraphInactiveFoldingTest, ConstantBoolEq) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantBoolEq, 0.0, 0.0, 1.0, "ConstantBoolEq");  // 0 + (t==t) = 1
}

TEST_F(GraphInactiveFoldingTest, ConstantBoolNe) {
    runInactiveFoldingTest(InactiveFoldingGraphs::ConstantBoolNe, 0.0, 0.0, 1.0, "ConstantBoolNe");  // 0 + (t!=f) = 1
}

// Nested constant subgraph tests - exercise markProcessed recursive marking
TEST_F(GraphInactiveFoldingTest, NestedConstantSubgraph) {
    runInactiveFoldingTest(InactiveFoldingGraphs::NestedConstantSubgraph, 5.0, 0.0, 25.0, "NestedConstantSubgraph");  // 5 + ((2+3)*4) = 5 + 20 = 25
}

TEST_F(GraphInactiveFoldingTest, DeeplyNestedConstantSubgraph) {
    runInactiveFoldingTest(InactiveFoldingGraphs::DeeplyNestedConstantSubgraph, 5.0, 0.0, 15.0, "DeeplyNestedConstantSubgraph");  // 5 + (((1+2)+3)+4) = 5 + 10 = 15
}

