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

// Test fixture for gradient tests
class GradientCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Any setup if needed
    }
    
    void TearDown() override {
        // Any cleanup if needed
    }
};

// Main gradient correctness test using finite differences
TEST_F(GradientCorrectnessTest, GradientComputation) {
    auto test_cases = getFiniteDiffTestCases();
    
    for (const auto& test_case : test_cases) {
        std::cout << "\n=== Testing " << test_case.name << " ===" << std::endl;
        
        bool all_passed = true;
        
        for (double x_value : test_case.test_points) {
            // Skip problematic points
            if (std::abs(x_value) < 1e-10 && test_case.name == "ReciprocalSquared") {
                continue;  // Skip x≈0 for 1/x^2
            }
            
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
            
            // Compile the graph
            ForgeEngine compiler;
            
            //// Enable debug output for failing test cases
            //if (test_case.name == "OneOverXPlusOne") {
            //    if (x_value == 0.0) {
            //        // Try with NO optimizations to isolate the issue
            //        auto config = CompilerConfig::NoOptimization();
            //        config.printOriginalGraph = true;
            //        config.printOptimizedGraph = true;
            //        config.printOptimizationStats = true;
            //        config.printNodeFlags = true;
            //        config.printGradientDebug = true;
            //        compiler.setConfig(config);
            //    }
            //}
            
            auto kernel = compiler.compile(graph);
            
            // Create NodeValueBuffer with gradient support
            auto buffer = NodeValueBufferFactory::create(graph, *kernel);
            
            // Set input value
            NodeId input_node = graph.diff_inputs[0];
            NodeId output_node = graph.outputs[0];
            buffer->setValue(input_node, x_value);
            buffer->clearGradients();
            
            // Execute forward and backward pass
            kernel->execute(*buffer);
            
            // Get results from automatic differentiation
            double f_autodiff = buffer->getValue(output_node);
            double df_autodiff = buffer->getGradient(input_node);
            
            // Get expected function value from native evaluation
            double f_native = test_case.func_native(x_value);
            
            // Compute expected derivative using finite differences
            double df_finite_diff = FiniteDifference::richardsonExtrapolation(
                test_case.func_native, x_value);
            
            // Check correctness
            bool f_correct = std::abs(f_autodiff - f_native) < 1e-10;
            bool df_correct = std::abs(df_autodiff - df_finite_diff) < test_case.tolerance;
            
            // Output results
            std::cout << std::fixed << std::setprecision(8);
            std::cout << "  x=" << std::setw(7) << x_value;
            std::cout << " | f(x): AD=" << std::setw(12) << f_autodiff;
            std::cout << " Native=" << std::setw(12) << f_native;
            std::cout << " | f'(x): AD=" << std::setw(12) << df_autodiff;
            std::cout << " FD=" << std::setw(12) << df_finite_diff;
            
            if (f_correct && df_correct) {
                std::cout << " [PASS]" << std::endl;
            } else {
                std::cout << " [FAIL]";
                if (!f_correct) {
                    std::cout << " (f err=" << std::abs(f_autodiff - f_native) << ")";
                }
                if (!df_correct) {
                    std::cout << " (f' err=" << std::abs(df_autodiff - df_finite_diff) << ")";
                }
                std::cout << std::endl;
                all_passed = false;
            }
            
            // Assert for test failure
            EXPECT_NEAR(f_autodiff, f_native, 1e-10) 
                << "Function value mismatch at x=" << x_value;
            EXPECT_NEAR(df_autodiff, df_finite_diff, test_case.tolerance) 
                << "Derivative mismatch at x=" << x_value 
                << " (AD=" << df_autodiff << ", FD=" << df_finite_diff << ")";
        }
        
        if (all_passed) {
            std::cout << test_case.name << " gradient test: ALL PASSED" << std::endl;
        }
    }
}

