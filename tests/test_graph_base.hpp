#pragma once

#include <gtest/gtest.h>
#include "../src/graph/graph.hpp"
#include "../src/graph/graph_recorder.hpp"
#include "../../tools/types/fdouble.hpp"
#include "../src/graph/handles.hpp"
#include <vector>
#include <functional>
#include <random>

namespace forge::testing {

class GraphTestBase : public ::testing::Test {
protected:
    GraphRecorder recorder;
    std::vector<double> input_data;
    std::vector<double> output_data;
    std::vector<double> expected_data;
    
    InputHandle input_handle{0};
    ResultHandle output_handle{0};
    
    void SetUp() override {
        // Clean state is now handled by the recorder itself
    }
    
    void TearDown() override {
        // Clean state is now handled by the recorder destructor
    }
    
    template<typename GraphBuilder>
    void BuildGraph(GraphBuilder builder, size_t data_size = 100) {
        input_data.resize(data_size);
        output_data.resize(data_size);
        expected_data.resize(data_size);
        
        recorder.start();
        
        fdouble x;
        input_handle = x.markInput();
        
        fdouble result = builder(x);
        output_handle = result.markOutput();
        
        recorder.stop(BuildOptions{});
    }
    
    void GenerateTestData(double min_val = -10.0, double max_val = 10.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min_val, max_val);
        
        for (auto& val : input_data) {
            val = dis(gen);
        }
    }
    
    void GenerateLinearTestData(double start, double step) {
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = start + i * step;
        }
    }
    
    void ComputeExpected(std::function<double(double)> fn) {
        for (size_t i = 0; i < input_data.size(); ++i) {
            expected_data[i] = fn(input_data[i]);
        }
    }
    
    void ValidateGraph() {
        const auto& graph = recorder.graph();
        EXPECT_FALSE(graph.empty()) << "Graph should not be empty after recording";
        EXPECT_GE(graph.outputs.size(), 1u) << "Graph should have at least one output";
    }
    
    void PrintGraphInfo() {
        const auto& graph = recorder.graph();
        std::cout << "Graph info:\n";
        std::cout << "  Nodes: " << graph.nodes.size() << "\n";
        std::cout << "  Constants: " << graph.constPool.size() << "\n";
        std::cout << "  Outputs: " << graph.outputs.size() << "\n";
    }
};

struct GraphTestCase {
    std::string name;
    std::function<fdouble(fdouble)> graph_builder;
    std::function<double(double)> expected;
};

class GraphOperationTest : public GraphTestBase,
                          public ::testing::WithParamInterface<GraphTestCase> {
};

} // namespace forge::testing