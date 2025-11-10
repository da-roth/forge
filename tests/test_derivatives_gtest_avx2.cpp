#include <gtest/gtest.h>
#include "test_functions_1d.hpp"
#include "../src/graph/graph_recorder.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/node_value_buffers/node_value_buffer.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace forge;
using namespace forge;
using namespace forge::testing;

// Test fixture for AVX gradient tests
class AVXGradientCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Any setup if needed
    }
    
    void TearDown() override {
        // Any cleanup if needed
    }
};

// AVX gradient correctness test using finite differences with batched evaluation
TEST_F(AVXGradientCorrectnessTest, AVXGradientComputation) {
    auto test_cases = getFiniteDiffTestCases();
    
    for (const auto& test_case : test_cases) {
        std::cout << "\n=== Testing " << test_case.name << " (AVX Batched) ===" << std::endl;
        
        bool all_passed = true;
        
        // Process test points in batches of 4
        std::vector<double> valid_test_points;
        for (double x_value : test_case.test_points) {
            // Skip problematic points
            if (std::abs(x_value) < 1e-10 && test_case.name == "ReciprocalSquared") {
                continue;  // Skip x≈0 for 1/x^2
            }
            valid_test_points.push_back(x_value);
        }
        
        // Process in batches of 4
        for (size_t batch_start = 0; batch_start < valid_test_points.size(); batch_start += 4) {
            // Create batch of up to 4 values
            std::vector<double> batch_values;
            for (size_t i = 0; i < 4 && (batch_start + i) < valid_test_points.size(); ++i) {
                batch_values.push_back(valid_test_points[batch_start + i]);
            }
            
            // Pad with last value if needed for AVX (need exactly 4)
            while (batch_values.size() < 4) {
                batch_values.push_back(batch_values.back());
            }
            
            std::cout << "  Batch: [" << batch_values[0] << ", " << batch_values[1] 
                      << ", " << batch_values[2] << ", " << batch_values[3] << "]" << std::endl;
            
            // Start recording for automatic differentiation
            GraphRecorder recorder;
            recorder.start();
            
            // Create input with gradient tracking
            fdouble x(0.0);
            x.markInputAndDiff();  // Mark for differentiation
            
            // Apply the function
            fdouble y = test_case.func_tp(x);
            y.markOutput();
            
            // Stop recording and get the tape
            recorder.stop();
            Graph graph = recorder.graph();
            
            // Compile the graph with AVX configuration
            auto config = CompilerConfig::Default();
            config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;
            ForgeEngine compiler(config);
            
            auto kernel = compiler.compile(graph);
            
            // Check if kernel was created successfully
            ASSERT_NE(kernel, nullptr) << "Failed to compile kernel";
            
            // Create NodeValueBuffer with kernel (vector-aware)
            auto buffer = NodeValueBufferFactory::create(graph, *kernel);
            
            // Set vector input values (4 different test points)
            NodeId input_node = graph.diff_inputs[0];
            NodeId output_node = graph.outputs[0];
            buffer->setVectorValue(input_node, batch_values);
            buffer->clearGradients();
            
            // Execute forward and backward pass
            kernel->execute(*buffer);
            
            // Get vector results from automatic differentiation
            std::vector<double> f_autodiff = buffer->getVectorValue(output_node);
            std::vector<double> df_autodiff = buffer->getVectorGradient(input_node);
            
            // Validate each lane in the batch
            size_t actual_test_points = std::min(size_t(4), valid_test_points.size() - batch_start);
            for (size_t lane = 0; lane < actual_test_points; ++lane) {
                double x_value = valid_test_points[batch_start + lane];
                
                // Get expected function value from native evaluation
                double f_native = test_case.func_native(x_value);
                
                // Compute expected derivative using finite differences
                double df_finite_diff = FiniteDifference::richardsonExtrapolation(
                    test_case.func_native, x_value);
                
                // Check correctness
                bool f_correct = std::abs(f_autodiff[lane] - f_native) < 1e-10;
                bool df_correct = std::abs(df_autodiff[lane] - df_finite_diff) < test_case.tolerance;
                
                // Output results
                std::cout << std::fixed << std::setprecision(8);
                std::cout << "    Lane " << lane << " x=" << std::setw(7) << x_value;
                std::cout << " | f(x): AD=" << std::setw(12) << f_autodiff[lane];
                std::cout << " Native=" << std::setw(12) << f_native;
                std::cout << " | f'(x): AD=" << std::setw(12) << df_autodiff[lane];
                std::cout << " FD=" << std::setw(12) << df_finite_diff;
                
                if (f_correct && df_correct) {
                    std::cout << " [PASS]" << std::endl;
                } else {
                    std::cout << " [FAIL]";
                    if (!f_correct) {
                        std::cout << " (f err=" << std::abs(f_autodiff[lane] - f_native) << ")";
                    }
                    if (!df_correct) {
                        std::cout << " (f' err=" << std::abs(df_autodiff[lane] - df_finite_diff) << ")";
                    }
                    std::cout << std::endl;
                    all_passed = false;
                }
                
                // Assert for test failure
                EXPECT_NEAR(f_autodiff[lane], f_native, 1e-10) 
                    << "Function value mismatch at x=" << x_value << " (lane " << lane << ")";
                EXPECT_NEAR(df_autodiff[lane], df_finite_diff, test_case.tolerance) 
                    << "Derivative mismatch at x=" << x_value << " (lane " << lane << ")"
                    << " (AD=" << df_autodiff[lane] << ", FD=" << df_finite_diff << ")";
            }
        }
        
        if (all_passed) {
            std::cout << test_case.name << " AVX gradient test: ALL PASSED" << std::endl;
        }
    }
}