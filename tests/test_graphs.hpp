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

    // Mod: (x + y) % c
    {
        TestGraph tg;
        tg.name = "Mod: (x+y) % 3";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId z = addBinaryOp(tg.graph, forge::OpCode::Add, x, y);
        forge::NodeId c = tg.graph.addConstant(3.0);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Mod, z, c);
        tg.graph.markOutput(tg.outputId);

        // (x + y) % 3
        tg.testCases = {
            {{5.0, 2.0}, std::fmod(7.0, 3.0), 0.0},   // 7 % 3 = 1
            {{4.0, 2.0}, std::fmod(6.0, 3.0), 0.0},   // 6 % 3 = 0
            {{1.0, 1.0}, std::fmod(2.0, 3.0), 0.0},   // 2 % 3 = 2
            {{10.0, 1.0}, std::fmod(11.0, 3.0), 0.0}  // 11 % 3 = 2
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

    // ========================================================================
    // Boolean operations
    // ========================================================================

    // BoolConstant: true (1.0)
    {
        TestGraph tg;
        tg.name = "BoolConstant: true";
        tg.hasGradient = false;
        tg.numInputs = 1;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        tg.inputIds = {x};

        // Create a BoolConstant node with value 1.0 (true)
        forge::Node boolNode;
        boolNode.op = forge::OpCode::BoolConstant;
        boolNode.imm = 1.0;
        boolNode.isActive = true;
        tg.outputId = tg.graph.addNode(boolNode);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{0.0}, 1.0, 0.0},
            {{1.0}, 1.0, 0.0},
            {{5.0}, 1.0, 0.0},
            {{-1.0}, 1.0, 0.0}
        };
        graphs.push_back(std::move(tg));
    }

    // BoolConstant: false (0.0)
    {
        TestGraph tg;
        tg.name = "BoolConstant: false";
        tg.hasGradient = false;
        tg.numInputs = 1;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        tg.inputIds = {x};

        // Create a BoolConstant node with value 0.0 (false)
        forge::Node boolNode;
        boolNode.op = forge::OpCode::BoolConstant;
        boolNode.imm = 0.0;
        boolNode.isActive = true;
        tg.outputId = tg.graph.addNode(boolNode);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{0.0}, 0.0, 0.0},
            {{1.0}, 0.0, 0.0},
            {{5.0}, 0.0, 0.0},
            {{-1.0}, 0.0, 0.0}
        };
        graphs.push_back(std::move(tg));
    }

    // BoolAnd: (x > 0) && (y > 0)
    {
        TestGraph tg;
        tg.name = "BoolAnd: (x>0) && (y>0)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId zero = tg.graph.addConstant(0.0);
        forge::NodeId cmpX = addBinaryOp(tg.graph, forge::OpCode::CmpGT, x, zero);
        forge::NodeId cmpY = addBinaryOp(tg.graph, forge::OpCode::CmpGT, y, zero);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::BoolAnd, cmpX, cmpY);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{1.0, 1.0}, 1.0, 0.0},    // true && true = true
            {{1.0, -1.0}, 0.0, 0.0},   // true && false = false
            {{-1.0, 1.0}, 0.0, 0.0},   // false && true = false
            {{-1.0, -1.0}, 0.0, 0.0}   // false && false = false
        };
        graphs.push_back(std::move(tg));
    }

    // BoolOr: (x > 0) || (y > 0)
    {
        TestGraph tg;
        tg.name = "BoolOr: (x>0) || (y>0)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId zero = tg.graph.addConstant(0.0);
        forge::NodeId cmpX = addBinaryOp(tg.graph, forge::OpCode::CmpGT, x, zero);
        forge::NodeId cmpY = addBinaryOp(tg.graph, forge::OpCode::CmpGT, y, zero);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::BoolOr, cmpX, cmpY);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{1.0, 1.0}, 1.0, 0.0},    // true || true = true
            {{1.0, -1.0}, 1.0, 0.0},   // true || false = true
            {{-1.0, 1.0}, 1.0, 0.0},   // false || true = true
            {{-1.0, -1.0}, 0.0, 0.0}   // false || false = false
        };
        graphs.push_back(std::move(tg));
    }

    // BoolNot: !(x > 0)
    {
        TestGraph tg;
        tg.name = "BoolNot: !(x>0)";
        tg.hasGradient = false;
        tg.numInputs = 1;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        tg.inputIds = {x};

        forge::NodeId zero = tg.graph.addConstant(0.0);
        forge::NodeId cmpX = addBinaryOp(tg.graph, forge::OpCode::CmpGT, x, zero);
        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::BoolNot, cmpX);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{1.0}, 0.0, 0.0},    // !true = false
            {{-1.0}, 1.0, 0.0},   // !false = true
            {{0.0}, 1.0, 0.0},    // !(0>0) = !false = true
            {{5.0}, 0.0, 0.0}     // !true = false
        };
        graphs.push_back(std::move(tg));
    }

    // BoolEq: (x > 0) == (y > 0)
    {
        TestGraph tg;
        tg.name = "BoolEq: (x>0) == (y>0)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId zero = tg.graph.addConstant(0.0);
        forge::NodeId cmpX = addBinaryOp(tg.graph, forge::OpCode::CmpGT, x, zero);
        forge::NodeId cmpY = addBinaryOp(tg.graph, forge::OpCode::CmpGT, y, zero);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::BoolEq, cmpX, cmpY);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{1.0, 1.0}, 1.0, 0.0},    // true == true = true
            {{1.0, -1.0}, 0.0, 0.0},   // true == false = false
            {{-1.0, 1.0}, 0.0, 0.0},   // false == true = false
            {{-1.0, -1.0}, 1.0, 0.0}   // false == false = true
        };
        graphs.push_back(std::move(tg));
    }

    // BoolNe: (x > 0) != (y > 0)
    {
        TestGraph tg;
        tg.name = "BoolNe: (x>0) != (y>0)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        forge::NodeId zero = tg.graph.addConstant(0.0);
        forge::NodeId cmpX = addBinaryOp(tg.graph, forge::OpCode::CmpGT, x, zero);
        forge::NodeId cmpY = addBinaryOp(tg.graph, forge::OpCode::CmpGT, y, zero);
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::BoolNe, cmpX, cmpY);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{1.0, 1.0}, 0.0, 0.0},    // true != true = false
            {{1.0, -1.0}, 1.0, 0.0},   // true != false = true
            {{-1.0, 1.0}, 1.0, 0.0},   // false != true = true
            {{-1.0, -1.0}, 0.0, 0.0}   // false != false = false
        };
        graphs.push_back(std::move(tg));
    }

    // ========================================================================
    // Integer operations (values are truncated to integers before operation)
    // ========================================================================

    // IntAdd: int(x) + int(y)
    {
        TestGraph tg;
        tg.name = "IntAdd: int(x) + int(y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::IntAdd, x, y);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{3.7, 2.3}, 5.0, 0.0},    // int(3.7) + int(2.3) = 3 + 2 = 5
            {{-3.7, 2.3}, -1.0, 0.0},  // int(-3.7) + int(2.3) = -3 + 2 = -1
            {{5.0, -3.0}, 2.0, 0.0},   // 5 + (-3) = 2
            {{0.9, 0.9}, 0.0, 0.0}     // int(0.9) + int(0.9) = 0 + 0 = 0
        };
        graphs.push_back(std::move(tg));
    }

    // IntSub: int(x) - int(y)
    {
        TestGraph tg;
        tg.name = "IntSub: int(x) - int(y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::IntSub, x, y);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{5.7, 2.3}, 3.0, 0.0},    // int(5.7) - int(2.3) = 5 - 2 = 3
            {{3.0, 5.0}, -2.0, 0.0},   // 3 - 5 = -2
            {{-3.7, -2.3}, -1.0, 0.0}  // int(-3.7) - int(-2.3) = -3 - (-2) = -1
        };
        graphs.push_back(std::move(tg));
    }

    // IntMul: int(x) * int(y)
    {
        TestGraph tg;
        tg.name = "IntMul: int(x) * int(y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::IntMul, x, y);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{3.7, 2.3}, 6.0, 0.0},    // int(3.7) * int(2.3) = 3 * 2 = 6
            {{-3.0, 4.0}, -12.0, 0.0}, // -3 * 4 = -12
            {{5.9, 0.5}, 0.0, 0.0}     // 5 * 0 = 0
        };
        graphs.push_back(std::move(tg));
    }

    // IntDiv: int(x) / int(y) (truncating division)
    {
        TestGraph tg;
        tg.name = "IntDiv: int(x) / int(y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::IntDiv, x, y);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{7.9, 2.1}, 3.0, 0.0},    // int(7.9) / int(2.1) = 7 / 2 = 3
            {{10.0, 3.0}, 3.0, 0.0},   // 10 / 3 = 3
            {{-7.0, 2.0}, -3.0, 0.0}   // -7 / 2 = -3 (truncates toward zero)
        };
        graphs.push_back(std::move(tg));
    }

    // IntCmpLT: int(x) < int(y)
    {
        TestGraph tg;
        tg.name = "IntCmpLT: int(x) < int(y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::IntCmpLT, x, y);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{2.9, 3.1}, 1.0, 0.0},    // int(2.9) < int(3.1) => 2 < 3 = true
            {{3.9, 3.1}, 0.0, 0.0},    // int(3.9) < int(3.1) => 3 < 3 = false
            {{4.0, 3.0}, 0.0, 0.0},    // 4 < 3 = false
            {{-2.0, -1.0}, 1.0, 0.0}   // -2 < -1 = true
        };
        graphs.push_back(std::move(tg));
    }

    // IntCmpEQ: int(x) == int(y)
    {
        TestGraph tg;
        tg.name = "IntCmpEQ: int(x) == int(y)";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::IntCmpEQ, x, y);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{3.1, 3.9}, 1.0, 0.0},    // int(3.1) == int(3.9) => 3 == 3 = true
            {{3.0, 4.0}, 0.0, 0.0},    // 3 == 4 = false
            {{-2.5, -2.1}, 1.0, 0.0}   // int(-2.5) == int(-2.1) => -2 == -2 = true
        };
        graphs.push_back(std::move(tg));
    }

    // IntIf: cond ? int(x) : int(y)
    {
        TestGraph tg;
        tg.name = "IntIf: (x>0) ? int(y) : int(z)";
        tg.hasGradient = false;
        tg.numInputs = 3;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        forge::NodeId z = tg.graph.addInput();
        tg.inputIds = {x, y, z};

        forge::NodeId zero = tg.graph.addConstant(0.0);
        forge::NodeId cond = addBinaryOp(tg.graph, forge::OpCode::CmpGT, x, zero);
        tg.outputId = addTernaryOp(tg.graph, forge::OpCode::IntIf, cond, y, z);
        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {{1.0, 5.7, 2.3}, 5.0, 0.0},   // true ? int(5.7) : int(2.3) = 5
            {{-1.0, 5.7, 2.3}, 2.0, 0.0},  // false ? int(5.7) : int(2.3) = 2
            {{1.0, -3.9, 7.1}, -3.0, 0.0}  // true ? int(-3.9) : int(7.1) = -3
        };
        graphs.push_back(std::move(tg));
    }

    // ========================================================================
    // Complex graph to test register pressure (forces LRU eviction)
    // This creates many intermediate values that must stay live simultaneously
    // ========================================================================

    // RegisterPressure: Complex expression with many live intermediates
    // result = ((a+b)*(c+d) + (e+f)*(g+h)) * ((i+j)*(k+l) + (m+n)*(o+p))
    // This creates 14 intermediate add results + 6 mul results = ~20 live values
    // which exceeds the 16 available registers, forcing LRU eviction
    {
        TestGraph tg;
        tg.name = "RegisterPressure: large expression tree";
        tg.hasGradient = false;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        // Create many constants that will be used in parallel computations
        forge::NodeId c1 = tg.graph.addConstant(1.0);
        forge::NodeId c2 = tg.graph.addConstant(2.0);
        forge::NodeId c3 = tg.graph.addConstant(3.0);
        forge::NodeId c4 = tg.graph.addConstant(4.0);
        forge::NodeId c5 = tg.graph.addConstant(5.0);
        forge::NodeId c6 = tg.graph.addConstant(6.0);
        forge::NodeId c7 = tg.graph.addConstant(7.0);
        forge::NodeId c8 = tg.graph.addConstant(8.0);

        // Level 1: 8 adds using x, y, and constants
        forge::NodeId a1 = addBinaryOp(tg.graph, forge::OpCode::Add, x, c1);   // x+1
        forge::NodeId a2 = addBinaryOp(tg.graph, forge::OpCode::Add, y, c2);   // y+2
        forge::NodeId a3 = addBinaryOp(tg.graph, forge::OpCode::Add, x, c3);   // x+3
        forge::NodeId a4 = addBinaryOp(tg.graph, forge::OpCode::Add, y, c4);   // y+4
        forge::NodeId a5 = addBinaryOp(tg.graph, forge::OpCode::Add, x, c5);   // x+5
        forge::NodeId a6 = addBinaryOp(tg.graph, forge::OpCode::Add, y, c6);   // y+6
        forge::NodeId a7 = addBinaryOp(tg.graph, forge::OpCode::Add, x, c7);   // x+7
        forge::NodeId a8 = addBinaryOp(tg.graph, forge::OpCode::Add, y, c8);   // y+8

        // Level 2: 4 muls
        forge::NodeId m1 = addBinaryOp(tg.graph, forge::OpCode::Mul, a1, a2);  // (x+1)*(y+2)
        forge::NodeId m2 = addBinaryOp(tg.graph, forge::OpCode::Mul, a3, a4);  // (x+3)*(y+4)
        forge::NodeId m3 = addBinaryOp(tg.graph, forge::OpCode::Mul, a5, a6);  // (x+5)*(y+6)
        forge::NodeId m4 = addBinaryOp(tg.graph, forge::OpCode::Mul, a7, a8);  // (x+7)*(y+8)

        // Level 3: 2 adds
        forge::NodeId s1 = addBinaryOp(tg.graph, forge::OpCode::Add, m1, m2);  // (x+1)*(y+2) + (x+3)*(y+4)
        forge::NodeId s2 = addBinaryOp(tg.graph, forge::OpCode::Add, m3, m4);  // (x+5)*(y+6) + (x+7)*(y+8)

        // Level 4: final mul
        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Mul, s1, s2);
        tg.graph.markOutput(tg.outputId);

        // Test case: x=1, y=2
        // a1=2, a2=4, a3=4, a4=6, a5=6, a6=8, a7=8, a8=10
        // m1=8, m2=24, m3=48, m4=80
        // s1=32, s2=128
        // result=32*128=4096
        tg.testCases = {
            {{1.0, 2.0}, 4096.0, 0.0},
            {{0.0, 0.0}, 1.0 * 2.0 * 3.0 * 4.0 + 5.0 * 6.0 * 7.0 * 8.0, 0.0},  // Simplified
        };
        // Recalculate for x=0, y=0:
        // a1=1, a2=2, a3=3, a4=4, a5=5, a6=6, a7=7, a8=8
        // m1=2, m2=12, m3=30, m4=56
        // s1=14, s2=86
        // result=14*86=1204
        tg.testCases[1] = {{0.0, 0.0}, 1204.0, 0.0};
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

    // ========================================================================
    // Tests with BOTH inputs marked for differentiation
    // These cover the node.b gradient paths in backward_forging.cpp
    // ========================================================================

    // Add with both inputs: output = x + y, grad w.r.t x = 1, grad w.r.t y = 1
    // This covers Add node.b gradient path (line 92-93 in backward_forging.cpp)
    {
        TestGraph tg;
        tg.name = "Add: x+y (both grads=1)";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        // Mark BOTH inputs for differentiation
        tg.graph.diff_inputs.push_back(x);
        tg.graph.diff_inputs.push_back(y);
        tg.graph.nodes[x].needsGradient = true;
        tg.graph.nodes[y].needsGradient = true;

        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Add, x, y, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[x+y] = 1, d/dy[x+y] = 1, we check gradient w.r.t first input (x)
        tg.testCases = {
            {{2.0, 3.0}, 5.0, 1.0},
            {{0.0, 0.0}, 0.0, 1.0},
            {{-1.0, 5.0}, 4.0, 1.0},
            {{10.0, -3.0}, 7.0, 1.0}
        };
        graphs.push_back(std::move(tg));
    }

    // Sub with both inputs: output = x - y, grad w.r.t x = 1, grad w.r.t y = -1
    // This covers Sub node.b gradient path (line 111-112 in backward_forging.cpp)
    {
        TestGraph tg;
        tg.name = "Sub: x-y (grad_x=1, grad_y=-1)";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        // Mark BOTH inputs for differentiation
        tg.graph.diff_inputs.push_back(x);
        tg.graph.diff_inputs.push_back(y);
        tg.graph.nodes[x].needsGradient = true;
        tg.graph.nodes[y].needsGradient = true;

        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Sub, x, y, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[x-y] = 1
        tg.testCases = {
            {{5.0, 3.0}, 2.0, 1.0},
            {{0.0, 0.0}, 0.0, 1.0},
            {{-1.0, 5.0}, -6.0, 1.0},
            {{10.0, -3.0}, 13.0, 1.0}
        };
        graphs.push_back(std::move(tg));
    }

    // Mul with both inputs: output = x * y, grad w.r.t x = y, grad w.r.t y = x
    // This covers Mul node.b gradient path (line 127-130 in backward_forging.cpp)
    {
        TestGraph tg;
        tg.name = "Mul: x*y (grad_x=y)";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        // Mark BOTH inputs for differentiation
        tg.graph.diff_inputs.push_back(x);
        tg.graph.diff_inputs.push_back(y);
        tg.graph.nodes[x].needsGradient = true;
        tg.graph.nodes[y].needsGradient = true;

        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Mul, x, y, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[x*y] = y
        tg.testCases = {
            {{2.0, 3.0}, 6.0, 3.0},      // grad_x = y = 3
            {{4.0, 5.0}, 20.0, 5.0},     // grad_x = y = 5
            {{-1.0, 2.0}, -2.0, 2.0},    // grad_x = y = 2
            {{3.0, -4.0}, -12.0, -4.0}   // grad_x = y = -4
        };
        graphs.push_back(std::move(tg));
    }

    // Div with both inputs: output = x / y, grad w.r.t x = 1/y, grad w.r.t y = -x/y^2
    // This covers Div node.b gradient path (line 165-172 in backward_forging.cpp)
    {
        TestGraph tg;
        tg.name = "Div: x/y (grad_x=1/y)";
        tg.hasGradient = true;
        tg.numInputs = 2;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        forge::NodeId y = tg.graph.addInput();
        tg.inputIds = {x, y};

        // Mark BOTH inputs for differentiation
        tg.graph.diff_inputs.push_back(x);
        tg.graph.diff_inputs.push_back(y);
        tg.graph.nodes[x].needsGradient = true;
        tg.graph.nodes[y].needsGradient = true;

        tg.outputId = addBinaryOp(tg.graph, forge::OpCode::Div, x, y, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[x/y] = 1/y
        tg.testCases = {
            {{6.0, 2.0}, 3.0, 0.5},      // grad_x = 1/2 = 0.5
            {{10.0, 5.0}, 2.0, 0.2},     // grad_x = 1/5 = 0.2
            {{4.0, 4.0}, 1.0, 0.25},     // grad_x = 1/4 = 0.25
            {{9.0, 3.0}, 3.0, 1.0/3.0}   // grad_x = 1/3
        };
        graphs.push_back(std::move(tg));
    }

    // Abs: output = |x|, grad w.r.t x = sign(x)
    // This covers Abs gradient path (line 198+ in backward_forging.cpp)
    {
        TestGraph tg;
        tg.name = "Abs: |x| (grad=sign(x))";
        tg.hasGradient = true;
        tg.numInputs = 1;
        tg.numOutputs = 1;

        forge::NodeId x = tg.graph.addInput();
        tg.inputIds = {x};

        tg.graph.diff_inputs.push_back(x);
        tg.graph.nodes[x].needsGradient = true;

        tg.outputId = addUnaryOp(tg.graph, forge::OpCode::Abs, x, true);
        tg.graph.markOutput(tg.outputId);

        // d/dx[|x|] = sign(x) = 1 if x > 0, -1 if x < 0
        tg.testCases = {
            {{5.0}, 5.0, 1.0},       // |5| = 5, sign(5) = 1
            {{3.0}, 3.0, 1.0},       // |3| = 3, sign(3) = 1
            {{-4.0}, 4.0, -1.0},     // |-4| = 4, sign(-4) = -1
            {{-7.0}, 7.0, -1.0}      // |-7| = 7, sign(-7) = -1
        };
        graphs.push_back(std::move(tg));
    }

    return graphs;
}

} // namespace forge_tests
