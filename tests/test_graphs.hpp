#pragma once

#include <string>
#include <vector>
#include <cmath>
#include "../src/graph/graph.hpp"

namespace forge_tests {

struct TestCase {
    std::vector<double> inputs;  // Values for each input node
    double expectedOutput;
    double expectedGradient;     // Gradient w.r.t. first input (x)
};

struct TestGraph {
    std::string name;
    forge::Graph graph;
    std::vector<forge::NodeId> inputIds;  // Multiple input nodes
    forge::NodeId outputId;
    bool hasGradient;
    size_t numInputs = 1;
    size_t numOutputs = 1;
    std::vector<TestCase> testCases;
};

// Helper to create a unary op node
inline forge::NodeId addUnaryOp(forge::Graph& graph, forge::OpCode op, forge::NodeId a, bool needsGrad = false) {
    forge::Node node;
    node.op = op;
    node.a = a;
    node.isActive = true;
    node.needsGradient = needsGrad;
    return graph.addNode(node);
}

// Helper to create a binary op node
inline forge::NodeId addBinaryOp(forge::Graph& graph, forge::OpCode op, forge::NodeId a, forge::NodeId b, bool needsGrad = false) {
    forge::Node node;
    node.op = op;
    node.a = a;
    node.b = b;
    node.isActive = true;
    node.needsGradient = needsGrad;
    return graph.addNode(node);
}

// Helper to create a ternary op node (If)
inline forge::NodeId addTernaryOp(forge::Graph& graph, forge::OpCode op, forge::NodeId cond, forge::NodeId a, forge::NodeId b, bool needsGrad = false) {
    forge::Node node;
    node.op = op;
    node.a = cond;
    node.b = a;
    node.c = b;
    node.isActive = true;
    node.needsGradient = needsGrad;
    return graph.addNode(node);
}

// Factory function to create test graphs without gradients
inline std::vector<TestGraph> createTestGraphs() {
    std::vector<TestGraph> graphs;

    // ========================================================================
    // Basic arithmetic: z = x + y, then apply operation to z
    // ========================================================================

    // Sub: (x + y) - c
    {
        TestGraph tg;
        tg.name = "Sub: (x+y) - 3";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        forge::NodeId c = tg.graph.addConstant(3.0);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Sub, z, c);
        tg.graph.markOutput(tg.outputId);

        // (x + y) - 3
        tg.testCases = {
            {{2.0, 3.0}, 2.0, 0.0},   // (2+3)-3 = 2
            {{5.0, 1.0}, 3.0, 0.0},   // (5+1)-3 = 3
            {{0.0, 0.0}, -3.0, 0.0},  // (0+0)-3 = -3
            {{-1.0, -2.0}, -6.0, 0.0} // (-1+-2)-3 = -6
        };
        graphs.push_back(std::move(tg));
    }

    // Mul: (x + y) * c
    {
        TestGraph tg;
        tg.name = "Mul: (x+y) * 2";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        forge::NodeId c = tg.graph.addConstant(2.0);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Mul, z, c);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{2.0, 3.0}, 10.0, 0.0},  // (2+3)*2 = 10
            {{5.0, 1.0}, 12.0, 0.0},  // (5+1)*2 = 12
            {{0.0, 0.0}, 0.0, 0.0},   // (0+0)*2 = 0
            {{-1.0, 4.0}, 6.0, 0.0}   // (-1+4)*2 = 6
        };
        graphs.push_back(std::move(tg));
    }

    // Div: (x + y) / c
    {
        TestGraph tg;
        tg.name = "Div: (x+y) / 2";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        forge::NodeId c = tg.graph.addConstant(2.0);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Div, z, c);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{4.0, 2.0}, 3.0, 0.0},   // (4+2)/2 = 3
            {{10.0, 0.0}, 5.0, 0.0},  // (10+0)/2 = 5
            {{1.0, 1.0}, 1.0, 0.0},   // (1+1)/2 = 1
            {{-2.0, 6.0}, 2.0, 0.0}   // (-2+6)/2 = 2
        };
        graphs.push_back(std::move(tg));
    }

    // ========================================================================
    // Unary operations on z = x + y
    // ========================================================================

    // Neg: -(x + y)
    {
        TestGraph tg;
        tg.name = "Neg: -(x+y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Neg, z);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{2.0, 3.0}, -5.0, 0.0},
            {{-1.0, -2.0}, 3.0, 0.0},
            {{0.0, 0.0}, 0.0, 0.0},
            {{5.0, -5.0}, 0.0, 0.0}
        };
        graphs.push_back(std::move(tg));
    }

    // Abs: |x + y|
    {
        TestGraph tg;
        tg.name = "Abs: |x+y|";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Abs, z);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{2.0, 3.0}, 5.0, 0.0},
            {{-3.0, -2.0}, 5.0, 0.0},
            {{-10.0, 3.0}, 7.0, 0.0},
            {{0.0, 0.0}, 0.0, 0.0}
        };
        graphs.push_back(std::move(tg));
    }

    // Square: (x + y)^2
    {
        TestGraph tg;
        tg.name = "Square: (x+y)^2";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Square, z);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{2.0, 1.0}, 9.0, 0.0},   // (2+1)^2 = 9
            {{3.0, -1.0}, 4.0, 0.0},  // (3-1)^2 = 4
            {{0.0, 0.0}, 0.0, 0.0},
            {{-2.0, -1.0}, 9.0, 0.0}  // (-3)^2 = 9
        };
        graphs.push_back(std::move(tg));
    }

    // Recip: 1 / (x + y)
    {
        TestGraph tg;
        tg.name = "Recip: 1/(x+y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Recip, z);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{1.0, 1.0}, 0.5, 0.0},    // 1/2
            {{4.0, 1.0}, 0.2, 0.0},    // 1/5
            {{0.5, 0.5}, 1.0, 0.0},    // 1/1
            {{2.0, 2.0}, 0.25, 0.0}    // 1/4
        };
        graphs.push_back(std::move(tg));
    }

    // Sqrt: sqrt(x + y)
    {
        TestGraph tg;
        tg.name = "Sqrt: sqrt(x+y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Sqrt, z);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{3.0, 1.0}, 2.0, 0.0},    // sqrt(4) = 2
            {{5.0, 4.0}, 3.0, 0.0},    // sqrt(9) = 3
            {{0.0, 16.0}, 4.0, 0.0},   // sqrt(16) = 4
            {{1.0, 0.0}, 1.0, 0.0}     // sqrt(1) = 1
        };
        graphs.push_back(std::move(tg));
    }

    // Exp: exp(x + y)
    {
        TestGraph tg;
        tg.name = "Exp: exp(x+y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Exp, z);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{0.0, 0.0}, 1.0, 0.0},              // exp(0) = 1
            {{1.0, 0.0}, std::exp(1.0), 0.0},    // exp(1)
            {{0.5, 0.5}, std::exp(1.0), 0.0},    // exp(1)
            {{-1.0, 0.0}, std::exp(-1.0), 0.0}   // exp(-1)
        };
        graphs.push_back(std::move(tg));
    }

    // Log: log(x + y)
    {
        TestGraph tg;
        tg.name = "Log: log(x+y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Log, z);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{1.0, 0.0}, 0.0, 0.0},              // log(1) = 0
            {{std::exp(1.0), 0.0}, 1.0, 0.0},    // log(e) = 1
            {{std::exp(2.0), 0.0}, 2.0, 0.0},    // log(e^2) = 2
            {{0.5, 0.5}, 0.0, 0.0}               // log(1) = 0
        };
        graphs.push_back(std::move(tg));
    }

    // Sin: sin(x + y)
    {
        TestGraph tg;
        tg.name = "Sin: sin(x+y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Sin, z);
        tg.graph.markOutput(tg.outputId);

        const double pi = 3.14159265358979323846;
        tg.testCases = {
            {{0.0, 0.0}, 0.0, 0.0},              // sin(0) = 0
            {{pi/2, 0.0}, 1.0, 0.0},             // sin(pi/2) = 1
            {{pi, 0.0}, std::sin(pi), 0.0},      // sin(pi) ~ 0
            {{-pi/2, 0.0}, -1.0, 0.0}            // sin(-pi/2) = -1
        };
        graphs.push_back(std::move(tg));
    }

    // Cos: cos(x + y)
    {
        TestGraph tg;
        tg.name = "Cos: cos(x+y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Cos, z);
        tg.graph.markOutput(tg.outputId);

        const double pi = 3.14159265358979323846;
        tg.testCases = {
            {{0.0, 0.0}, 1.0, 0.0},              // cos(0) = 1
            {{pi/2, 0.0}, std::cos(pi/2), 0.0},  // cos(pi/2) ~ 0
            {{pi, 0.0}, -1.0, 0.0},              // cos(pi) = -1
            {{2*pi, 0.0}, 1.0, 0.0}              // cos(2pi) = 1
        };
        graphs.push_back(std::move(tg));
    }

    // Tan: tan(x + y)
    {
        TestGraph tg;
        tg.name = "Tan: tan(x+y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Tan, z);
        tg.graph.markOutput(tg.outputId);

        const double pi = 3.14159265358979323846;
        tg.testCases = {
            {{0.0, 0.0}, 0.0, 0.0},              // tan(0) = 0
            {{pi/4, 0.0}, std::tan(pi/4), 0.0},  // tan(pi/4) ~ 1
            {{pi, 0.0}, std::tan(pi), 0.0},      // tan(pi) ~ 0
            {{-pi/4, 0.0}, std::tan(-pi/4), 0.0} // tan(-pi/4) ~ -1
        };
        graphs.push_back(std::move(tg));
    }

    // ========================================================================
    // Binary operations with third input u: op(z, u) where z = x + y
    // ========================================================================

    // Pow: (x + y) ^ u
    {
        TestGraph tg;
        tg.name = "Pow: (x+y)^u";
        tg.hasGradient = false;
        tg.numInputs = 3;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        forge::NodeId u = tg.graph.addInput();
        tg.inputIds = {x, y, u};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Pow, z, u);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{2.0, 1.0, 2.0}, 9.0, 0.0},    // (2+1)^2 = 9
            {{1.0, 1.0, 3.0}, 8.0, 0.0},    // (1+1)^3 = 8
            {{3.0, 1.0, 0.5}, 2.0, 0.0},    // (3+1)^0.5 = 2
            {{2.0, 2.0, 0.0}, 1.0, 0.0}     // (2+2)^0 = 1
        };
        graphs.push_back(std::move(tg));
    }

    // Min: min(x + y, u)
    {
        TestGraph tg;
        tg.name = "Min: min(x+y, u)";
        tg.hasGradient = false;
        tg.numInputs = 3;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        forge::NodeId u = tg.graph.addInput();
        tg.inputIds = {x, y, u};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Min, z, u);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{2.0, 3.0, 4.0}, 4.0, 0.0},    // min(5, 4) = 4
            {{2.0, 3.0, 6.0}, 5.0, 0.0},    // min(5, 6) = 5
            {{-1.0, -2.0, 0.0}, -3.0, 0.0}, // min(-3, 0) = -3
            {{1.0, 1.0, 2.0}, 2.0, 0.0}     // min(2, 2) = 2
        };
        graphs.push_back(std::move(tg));
    }

    // Max: max(x + y, u)
    {
        TestGraph tg;
        tg.name = "Max: max(x+y, u)";
        tg.hasGradient = false;
        tg.numInputs = 3;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        forge::NodeId u = tg.graph.addInput();
        tg.inputIds = {x, y, u};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Max, z, u);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{2.0, 3.0, 4.0}, 5.0, 0.0},    // max(5, 4) = 5
            {{2.0, 3.0, 6.0}, 6.0, 0.0},    // max(5, 6) = 6
            {{-1.0, -2.0, 0.0}, 0.0, 0.0},  // max(-3, 0) = 0
            {{1.0, 1.0, 2.0}, 2.0, 0.0}     // max(2, 2) = 2
        };
        graphs.push_back(std::move(tg));
    }

    // ========================================================================
    // Comparison + If: (x + y) cmp u ? trueVal : falseVal
    // ========================================================================

    // If with CmpLT: (x+y) < u ? 1 : 0
    {
        TestGraph tg;
        tg.name = "If: (x+y)<u ? 1 : 0";
        tg.hasGradient = false;
        tg.numInputs = 3;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        forge::NodeId u = tg.graph.addInput();
        tg.inputIds = {x, y, u};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        forge::NodeId cmp = addBinaryOp(tg.graph, forge::OpCode::CmpLT, z, u);
        forge::NodeId trueVal = tg.graph.addConstant(1.0);
        forge::NodeId falseVal = tg.graph.addConstant(0.0);
        tg.outputId = addTernaryOp(tg.graph, forge::OpCode::If, cmp, trueVal, falseVal);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{1.0, 1.0, 5.0}, 1.0, 0.0},    // 2 < 5 -> 1
            {{3.0, 3.0, 5.0}, 1.0, 0.0},    // 6 < 5 -> 0... wait 6 > 5
            {{3.0, 2.0, 5.0}, 0.0, 0.0},    // 5 < 5 -> 0
            {{1.0, 1.0, 1.0}, 0.0, 0.0}     // 2 < 1 -> 0
        };
        // Fix: 3+3=6, 6<5 is false -> 0
        tg.testCases[1] = {{3.0, 3.0, 5.0}, 0.0, 0.0};
        graphs.push_back(std::move(tg));
    }

    // If with CmpLE: (x+y) <= u ? 1 : 0
    {
        TestGraph tg;
        tg.name = "If: (x+y)<=u ? 1 : 0";
        tg.hasGradient = false;
        tg.numInputs = 3;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        forge::NodeId u = tg.graph.addInput();
        tg.inputIds = {x, y, u};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        forge::NodeId cmp = addBinaryOp(tg.graph, forge::OpCode::CmpLE, z, u);
        forge::NodeId trueVal = tg.graph.addConstant(1.0);
        forge::NodeId falseVal = tg.graph.addConstant(0.0);
        tg.outputId = addTernaryOp(tg.graph, forge::OpCode::If, cmp, trueVal, falseVal);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{1.0, 1.0, 5.0}, 1.0, 0.0},    // 2 <= 5 -> 1
            {{2.0, 3.0, 5.0}, 1.0, 0.0},    // 5 <= 5 -> 1
            {{3.0, 3.0, 5.0}, 0.0, 0.0},    // 6 <= 5 -> 0
            {{0.0, 0.0, 0.0}, 1.0, 0.0}     // 0 <= 0 -> 1
        };
        graphs.push_back(std::move(tg));
    }

    // If with CmpGT: (x+y) > u ? 1 : 0
    {
        TestGraph tg;
        tg.name = "If: (x+y)>u ? 1 : 0";
        tg.hasGradient = false;
        tg.numInputs = 3;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        forge::NodeId u = tg.graph.addInput();
        tg.inputIds = {x, y, u};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        forge::NodeId cmp = addBinaryOp(tg.graph, forge::OpCode::CmpGT, z, u);
        forge::NodeId trueVal = tg.graph.addConstant(1.0);
        forge::NodeId falseVal = tg.graph.addConstant(0.0);
        tg.outputId = addTernaryOp(tg.graph, forge::OpCode::If, cmp, trueVal, falseVal);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{3.0, 3.0, 5.0}, 1.0, 0.0},    // 6 > 5 -> 1
            {{2.0, 3.0, 5.0}, 0.0, 0.0},    // 5 > 5 -> 0
            {{1.0, 1.0, 5.0}, 0.0, 0.0},    // 2 > 5 -> 0
            {{5.0, 5.0, 0.0}, 1.0, 0.0}     // 10 > 0 -> 1
        };
        graphs.push_back(std::move(tg));
    }

    // If with CmpGE: (x+y) >= u ? 1 : 0
    {
        TestGraph tg;
        tg.name = "If: (x+y)>=u ? 1 : 0";
        tg.hasGradient = false;
        tg.numInputs = 3;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        forge::NodeId u = tg.graph.addInput();
        tg.inputIds = {x, y, u};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        forge::NodeId cmp = addBinaryOp(tg.graph, forge::OpCode::CmpGE, z, u);
        forge::NodeId trueVal = tg.graph.addConstant(1.0);
        forge::NodeId falseVal = tg.graph.addConstant(0.0);
        tg.outputId = addTernaryOp(tg.graph, forge::OpCode::If, cmp, trueVal, falseVal);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{3.0, 3.0, 5.0}, 1.0, 0.0},    // 6 >= 5 -> 1
            {{2.0, 3.0, 5.0}, 1.0, 0.0},    // 5 >= 5 -> 1
            {{1.0, 1.0, 5.0}, 0.0, 0.0},    // 2 >= 5 -> 0
            {{0.0, 0.0, 0.0}, 1.0, 0.0}     // 0 >= 0 -> 1
        };
        graphs.push_back(std::move(tg));
    }

    // If with CmpEQ: (x+y) == u ? 1 : 0
    {
        TestGraph tg;
        tg.name = "If: (x+y)==u ? 1 : 0";
        tg.hasGradient = false;
        tg.numInputs = 3;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        forge::NodeId u = tg.graph.addInput();
        tg.inputIds = {x, y, u};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        forge::NodeId cmp = addBinaryOp(tg.graph, forge::OpCode::CmpEQ, z, u);
        forge::NodeId trueVal = tg.graph.addConstant(1.0);
        forge::NodeId falseVal = tg.graph.addConstant(0.0);
        tg.outputId = addTernaryOp(tg.graph, forge::OpCode::If, cmp, trueVal, falseVal);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{2.0, 3.0, 5.0}, 1.0, 0.0},    // 5 == 5 -> 1
            {{3.0, 3.0, 5.0}, 0.0, 0.0},    // 6 == 5 -> 0
            {{0.0, 0.0, 0.0}, 1.0, 0.0},    // 0 == 0 -> 1
            {{1.0, 2.0, 4.0}, 0.0, 0.0}     // 3 == 4 -> 0
        };
        graphs.push_back(std::move(tg));
    }

    // If with CmpNE: (x+y) != u ? 1 : 0
    {
        TestGraph tg;
        tg.name = "If: (x+y)!=u ? 1 : 0";
        tg.hasGradient = false;
        tg.numInputs = 3;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        forge::NodeId u = tg.graph.addInput();
        tg.inputIds = {x, y, u};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        forge::NodeId cmp = addBinaryOp(tg.graph, forge::OpCode::CmpNE, z, u);
        forge::NodeId trueVal = tg.graph.addConstant(1.0);
        forge::NodeId falseVal = tg.graph.addConstant(0.0);
        tg.outputId = addTernaryOp(tg.graph, forge::OpCode::If, cmp, trueVal, falseVal);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{2.0, 3.0, 5.0}, 0.0, 0.0},    // 5 != 5 -> 0
            {{3.0, 3.0, 5.0}, 1.0, 0.0},    // 6 != 5 -> 1
            {{0.0, 0.0, 0.0}, 0.0, 0.0},    // 0 != 0 -> 0
            {{1.0, 2.0, 4.0}, 1.0, 0.0}     // 3 != 4 -> 1
        };
        graphs.push_back(std::move(tg));
    }

    return graphs;
}

// Factory function to create test graphs with gradients
// Only include operations where gradient makes sense
inline std::vector<TestGraph> createTestGraphsWithGradient() {
    std::vector<TestGraph> graphs;

    // For gradient tests, we mark x for differentiation
    // z = x + y, so dz/dx = 1

    // Add: z = x + y, output = z + c, gradient w.r.t x = 1
    {
        TestGraph tg;
        tg.name = "Add: (x+y)+1 (grad=1)";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        forge::NodeId c = tg.graph.addConstant(1.0);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Add, z, c, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[(x+y)+1] = 1
        tg.testCases = {
            {{2.0, 3.0}, 6.0, 1.0},
            {{0.0, 0.0}, 1.0, 1.0},
            {{-1.0, 5.0}, 5.0, 1.0},
            {{10.0, -3.0}, 8.0, 1.0}
        };
        graphs.push_back(std::move(tg));
    }

    // Mul: z = x + y, output = z * c, gradient w.r.t x = c = 2
    {
        TestGraph tg;
        tg.name = "Mul: (x+y)*2 (grad=2)";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        forge::NodeId c = tg.graph.addConstant(2.0);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Mul, z, c, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[(x+y)*2] = 2
        tg.testCases = {
            {{2.0, 3.0}, 10.0, 2.0},
            {{0.0, 0.0}, 0.0, 2.0},
            {{-1.0, 5.0}, 8.0, 2.0},
            {{10.0, -3.0}, 14.0, 2.0}
        };
        graphs.push_back(std::move(tg));
    }

    // Square: z = x + y, output = z^2, gradient w.r.t x = 2*z = 2*(x+y)
    {
        TestGraph tg;
        tg.name = "Square: (x+y)^2 (grad=2*(x+y))";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Square, z, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[(x+y)^2] = 2*(x+y)
        tg.testCases = {
            {{2.0, 1.0}, 9.0, 6.0},    // z=3, grad=2*3=6
            {{1.0, 1.0}, 4.0, 4.0},    // z=2, grad=2*2=4
            {{3.0, 2.0}, 25.0, 10.0},  // z=5, grad=2*5=10
            {{0.0, 0.0}, 0.0, 0.0}     // z=0, grad=2*0=0
        };
        graphs.push_back(std::move(tg));
    }

    // Exp: z = x + y, output = exp(z), gradient w.r.t x = exp(z)
    {
        TestGraph tg;
        tg.name = "Exp: exp(x+y) (grad=exp(x+y))";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Exp, z, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[exp(x+y)] = exp(x+y)
        tg.testCases = {
            {{0.0, 0.0}, 1.0, 1.0},                        // exp(0)=1, grad=1
            {{1.0, 0.0}, std::exp(1.0), std::exp(1.0)},    // exp(1), grad=exp(1)
            {{0.5, 0.5}, std::exp(1.0), std::exp(1.0)},    // exp(1), grad=exp(1)
            {{-1.0, 0.0}, std::exp(-1.0), std::exp(-1.0)}  // exp(-1), grad=exp(-1)
        };
        graphs.push_back(std::move(tg));
    }

    // Sin: z = x + y, output = sin(z), gradient w.r.t x = cos(z)
    {
        TestGraph tg;
        tg.name = "Sin: sin(x+y) (grad=cos(x+y))";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Sin, z, true);
        tg.graph.markOutput(tg.outputId);

        const double pi = 3.14159265358979323846;
        // d/dx[sin(x+y)] = cos(x+y)
        tg.testCases = {
            {{0.0, 0.0}, 0.0, 1.0},                        // sin(0)=0, cos(0)=1
            {{pi/2, 0.0}, 1.0, std::cos(pi/2)},            // sin(pi/2)=1, cos(pi/2)~0
            {{pi, 0.0}, std::sin(pi), std::cos(pi)},       // sin(pi)~0, cos(pi)=-1
            {{0.0, pi/2}, 1.0, std::cos(pi/2)}             // sin(pi/2)=1
        };
        graphs.push_back(std::move(tg));
    }

    // Cos: z = x + y, output = cos(z), gradient w.r.t x = -sin(z)
    {
        TestGraph tg;
        tg.name = "Cos: cos(x+y) (grad=-sin(x+y))";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Cos, z, true);
        tg.graph.markOutput(tg.outputId);

        const double pi = 3.14159265358979323846;
        // d/dx[cos(x+y)] = -sin(x+y)
        tg.testCases = {
            {{0.0, 0.0}, 1.0, 0.0},                         // cos(0)=1, -sin(0)=0
            {{pi/2, 0.0}, std::cos(pi/2), -1.0},            // cos(pi/2)~0, -sin(pi/2)=-1
            {{pi, 0.0}, -1.0, -std::sin(pi)},               // cos(pi)=-1, -sin(pi)~0
            {{0.0, pi}, -1.0, -std::sin(pi)}                // cos(pi)=-1
        };
        graphs.push_back(std::move(tg));
    }

    // Log: z = x + y, output = log(z), gradient w.r.t x = 1/z
    {
        TestGraph tg;
        tg.name = "Log: log(x+y) (grad=1/(x+y))";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Log, z, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[log(x+y)] = 1/(x+y)
        tg.testCases = {
            {{1.0, 0.0}, 0.0, 1.0},                    // log(1)=0, grad=1/1=1
            {{1.0, 1.0}, std::log(2.0), 0.5},          // log(2), grad=1/2
            {{2.0, 2.0}, std::log(4.0), 0.25},         // log(4), grad=1/4
            {{4.0, 1.0}, std::log(5.0), 0.2}           // log(5), grad=1/5
        };
        graphs.push_back(std::move(tg));
    }

    // Sqrt: z = x + y, output = sqrt(z), gradient w.r.t x = 1/(2*sqrt(z))
    {
        TestGraph tg;
        tg.name = "Sqrt: sqrt(x+y) (grad=1/(2*sqrt))";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Sqrt, z, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[sqrt(x+y)] = 1/(2*sqrt(x+y))
        tg.testCases = {
            {{3.0, 1.0}, 2.0, 0.25},     // sqrt(4)=2, grad=1/(2*2)=0.25
            {{8.0, 1.0}, 3.0, 1.0/6.0},  // sqrt(9)=3, grad=1/(2*3)=1/6
            {{0.0, 1.0}, 1.0, 0.5},      // sqrt(1)=1, grad=1/(2*1)=0.5
            {{15.0, 1.0}, 4.0, 0.125}    // sqrt(16)=4, grad=1/(2*4)=0.125
        };
        graphs.push_back(std::move(tg));
    }

    // Sub: z = x + y, output = z - c, gradient w.r.t x = 1
    {
        TestGraph tg;
        tg.name = "Sub: (x+y)-3 (grad=1)";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        forge::NodeId c = tg.graph.addConstant(3.0);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Sub, z, c, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[(x+y)-3] = 1
        tg.testCases = {
            {{2.0, 3.0}, 2.0, 1.0},
            {{5.0, 1.0}, 3.0, 1.0},
            {{0.0, 0.0}, -3.0, 1.0},
            {{-1.0, -2.0}, -6.0, 1.0}
        };
        graphs.push_back(std::move(tg));
    }

    // Div: z = x + y, output = z / c, gradient w.r.t x = 1/c = 0.5
    {
        TestGraph tg;
        tg.name = "Div: (x+y)/2 (grad=0.5)";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        forge::NodeId c = tg.graph.addConstant(2.0);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Div, z, c, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[(x+y)/2] = 1/2 = 0.5
        tg.testCases = {
            {{4.0, 2.0}, 3.0, 0.5},
            {{10.0, 0.0}, 5.0, 0.5},
            {{1.0, 1.0}, 1.0, 0.5},
            {{-2.0, 6.0}, 2.0, 0.5}
        };
        graphs.push_back(std::move(tg));
    }

    // Neg: z = x + y, output = -z, gradient w.r.t x = -1
    {
        TestGraph tg;
        tg.name = "Neg: -(x+y) (grad=-1)";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Neg, z, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[-(x+y)] = -1
        tg.testCases = {
            {{2.0, 3.0}, -5.0, -1.0},
            {{-1.0, -2.0}, 3.0, -1.0},
            {{0.0, 0.0}, 0.0, -1.0},
            {{5.0, -5.0}, 0.0, -1.0}
        };
        graphs.push_back(std::move(tg));
    }

    // Recip: z = x + y, output = 1/z, gradient w.r.t x = -1/z^2
    {
        TestGraph tg;
        tg.name = "Recip: 1/(x+y) (grad=-1/(x+y)^2)";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Recip, z, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[1/(x+y)] = -1/(x+y)^2
        tg.testCases = {
            {{1.0, 1.0}, 0.5, -0.25},     // 1/2=0.5, grad=-1/4=-0.25
            {{4.0, 1.0}, 0.2, -0.04},     // 1/5=0.2, grad=-1/25=-0.04
            {{0.5, 0.5}, 1.0, -1.0},      // 1/1=1, grad=-1/1=-1
            {{2.0, 2.0}, 0.25, -0.0625}   // 1/4=0.25, grad=-1/16=-0.0625
        };
        graphs.push_back(std::move(tg));
    }

    // Tan: z = x + y, output = tan(z), gradient w.r.t x = 1/cos^2(z) = sec^2(z)
    {
        TestGraph tg;
        tg.name = "Tan: tan(x+y) (grad=sec^2(x+y))";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Tan, z, true);
        tg.graph.markOutput(tg.outputId);

        const double pi = 3.14159265358979323846;
        // d/dx[tan(x+y)] = sec^2(x+y) = 1/cos^2(x+y)
        auto sec2 = [](double x) { double c = std::cos(x); return 1.0/(c*c); };
        tg.testCases = {
            {{0.0, 0.0}, 0.0, 1.0},                          // tan(0)=0, sec^2(0)=1
            {{pi/4, 0.0}, std::tan(pi/4), sec2(pi/4)},       // tan(pi/4)~1, sec^2(pi/4)=2
            {{pi/6, 0.0}, std::tan(pi/6), sec2(pi/6)},       // tan(pi/6)
            {{-pi/4, 0.0}, std::tan(-pi/4), sec2(-pi/4)}     // tan(-pi/4)~-1
        };
        graphs.push_back(std::move(tg));
    }

    // Pow: z = x + y, output = z^u, gradient w.r.t x = u * z^(u-1)
    {
        TestGraph tg;
        tg.name = "Pow: (x+y)^u (grad=u*(x+y)^(u-1))";
        tg.hasGradient = true;
        tg.numInputs = 3;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        forge::NodeId u = tg.graph.addInput();
        tg.inputIds = {x, y, u};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Pow, z, u, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[(x+y)^u] = u * (x+y)^(u-1)
        tg.testCases = {
            {{2.0, 1.0, 2.0}, 9.0, 6.0},     // (3)^2=9, grad=2*3^1=6
            {{1.0, 1.0, 3.0}, 8.0, 12.0},    // (2)^3=8, grad=3*2^2=12
            {{3.0, 1.0, 0.5}, 2.0, 0.25},    // (4)^0.5=2, grad=0.5*4^(-0.5)=0.5*0.5=0.25
            {{2.0, 2.0, 1.0}, 4.0, 1.0}      // (4)^1=4, grad=1*4^0=1
        };
        graphs.push_back(std::move(tg));
    }

    return graphs;
}

} // namespace forge_tests
