#pragma once

#include <string>
#include <vector>
#include <functional>
#include "../src/graph/graph.hpp"

namespace forge_tests {

struct TestCase {
    double input;
    double expectedOutput;
    double expectedGradient;
};

struct TestGraph {
    std::string name;
    forge::Graph graph;
    forge::NodeId inputId;
    forge::NodeId outputId;
    bool hasGradient;
    size_t numInputs = 1;   // Number of graph inputs (marked as input)
    size_t numOutputs = 1;  // Number of graph outputs (marked as output)
    std::vector<TestCase> testCases;  // Each test case is one "input set" for re-evaluation
};

// Factory function to create test graphs without gradients
inline std::vector<TestGraph> createTestGraphs() {
    std::vector<TestGraph> graphs;

    // Graph 1: output = input + 1.0
    {
        TestGraph tg;
        tg.name = "x + 1";
        tg.hasGradient = false;

        tg.inputId = tg.graph.addInput();
        forge::NodeId constId = tg.graph.addConstant(1.0);

        forge::Node addNode;
        addNode.op = forge::OpCode::Add;
        addNode.a = tg.inputId;
        addNode.b = constId;
        addNode.isActive = true;
        tg.outputId = tg.graph.addNode(addNode);

        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {5.0, 6.0, 0.0},
            {0.0, 1.0, 0.0},
            {-3.0, -2.0, 0.0},
            {100.5, 101.5, 0.0}
        };

        graphs.push_back(std::move(tg));
    }

    // Graph 2: output = input * 2.0
    {
        TestGraph tg;
        tg.name = "x * 2";
        tg.hasGradient = false;

        tg.inputId = tg.graph.addInput();
        forge::NodeId constId = tg.graph.addConstant(2.0);

        forge::Node mulNode;
        mulNode.op = forge::OpCode::Mul;
        mulNode.a = tg.inputId;
        mulNode.b = constId;
        mulNode.isActive = true;
        tg.outputId = tg.graph.addNode(mulNode);

        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {5.0, 10.0, 0.0},
            {0.0, 0.0, 0.0},
            {-3.0, -6.0, 0.0},
            {7.5, 15.0, 0.0}
        };

        graphs.push_back(std::move(tg));
    }

    // Graph 3: output = (input + 1) * 2
    {
        TestGraph tg;
        tg.name = "(x + 1) * 2";
        tg.hasGradient = false;

        tg.inputId = tg.graph.addInput();
        forge::NodeId const1 = tg.graph.addConstant(1.0);
        forge::NodeId const2 = tg.graph.addConstant(2.0);

        forge::Node addNode;
        addNode.op = forge::OpCode::Add;
        addNode.a = tg.inputId;
        addNode.b = const1;
        addNode.isActive = true;
        forge::NodeId addId = tg.graph.addNode(addNode);

        forge::Node mulNode;
        mulNode.op = forge::OpCode::Mul;
        mulNode.a = addId;
        mulNode.b = const2;
        mulNode.isActive = true;
        tg.outputId = tg.graph.addNode(mulNode);

        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {5.0, 12.0, 0.0},   // (5+1)*2 = 12
            {0.0, 2.0, 0.0},    // (0+1)*2 = 2
            {-3.0, -4.0, 0.0},  // (-3+1)*2 = -4
            {4.5, 11.0, 0.0}    // (4.5+1)*2 = 11
        };

        graphs.push_back(std::move(tg));
    }

    return graphs;
}

// Factory function to create test graphs with gradients
inline std::vector<TestGraph> createTestGraphsWithGradient() {
    std::vector<TestGraph> graphs;

    // Graph 1: output = input + 1.0, gradient = 1.0
    {
        TestGraph tg;
        tg.name = "x + 1 (grad=1)";
        tg.hasGradient = true;

        tg.inputId = tg.graph.addInput();
        tg.graph.diff_inputs.push_back(tg.inputId);
        tg.graph.nodes[tg.inputId].needsGradient = true;

        forge::NodeId constId = tg.graph.addConstant(1.0);

        forge::Node addNode;
        addNode.op = forge::OpCode::Add;
        addNode.a = tg.inputId;
        addNode.b = constId;
        addNode.isActive = true;
        addNode.needsGradient = true;
        tg.outputId = tg.graph.addNode(addNode);

        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {5.0, 6.0, 1.0},
            {0.0, 1.0, 1.0},
            {-3.0, -2.0, 1.0},
            {100.5, 101.5, 1.0}
        };

        graphs.push_back(std::move(tg));
    }

    // Graph 2: output = input * 2.0, gradient = 2.0
    {
        TestGraph tg;
        tg.name = "x * 2 (grad=2)";
        tg.hasGradient = true;

        tg.inputId = tg.graph.addInput();
        tg.graph.diff_inputs.push_back(tg.inputId);
        tg.graph.nodes[tg.inputId].needsGradient = true;

        forge::NodeId constId = tg.graph.addConstant(2.0);

        forge::Node mulNode;
        mulNode.op = forge::OpCode::Mul;
        mulNode.a = tg.inputId;
        mulNode.b = constId;
        mulNode.isActive = true;
        mulNode.needsGradient = true;
        tg.outputId = tg.graph.addNode(mulNode);

        tg.graph.markOutput(tg.outputId);

        tg.testCases = {
            {5.0, 10.0, 2.0},
            {0.0, 0.0, 2.0},
            {-3.0, -6.0, 2.0},
            {7.5, 15.0, 2.0}
        };

        graphs.push_back(std::move(tg));
    }

    // Graph 3: output = (input + 1) * 2, gradient = 2.0
    {
        TestGraph tg;
        tg.name = "(x + 1) * 2 (grad=2)";
        tg.hasGradient = true;

        tg.inputId = tg.graph.addInput();
        tg.graph.diff_inputs.push_back(tg.inputId);
        tg.graph.nodes[tg.inputId].needsGradient = true;

        forge::NodeId const1 = tg.graph.addConstant(1.0);
        forge::NodeId const2 = tg.graph.addConstant(2.0);

        forge::Node addNode;
        addNode.op = forge::OpCode::Add;
        addNode.a = tg.inputId;
        addNode.b = const1;
        addNode.isActive = true;
        addNode.needsGradient = true;
        forge::NodeId addId = tg.graph.addNode(addNode);

        forge::Node mulNode;
        mulNode.op = forge::OpCode::Mul;
        mulNode.a = addId;
        mulNode.b = const2;
        mulNode.isActive = true;
        mulNode.needsGradient = true;
        tg.outputId = tg.graph.addNode(mulNode);

        tg.graph.markOutput(tg.outputId);

        // d/dx[(x+1)*2] = 2
        tg.testCases = {
            {5.0, 12.0, 2.0},
            {0.0, 2.0, 2.0},
            {-3.0, -4.0, 2.0},
            {4.5, 11.0, 2.0}
        };

        graphs.push_back(std::move(tg));
    }

    return graphs;
}

} // namespace forge_tests
