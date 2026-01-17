/**
 * Basic Gradient Computation Example
 *
 * This example demonstrates how to:
 * 1. Record a mathematical expression using FORGE
 * 2. Compile it to optimized machine code
 * 3. Evaluate the function and its gradient
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <native/fdouble.hpp>
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/interfaces/node_value_buffer.hpp>
#include <iostream>
#include <iomanip>

using namespace forge;

int main() {
    std::cout << "FORGE Basic Gradient Example\n";
    std::cout << "=============================\n\n";

    // Example 1: Simple quadratic function
    // f(x) = x^2 + 3x + 2
    {
        std::cout << "Example 1: f(x) = x^2 + 3x + 2\n";
        std::cout << "Expected: f'(x) = 2x + 3\n\n";

        // Start recording
        GraphRecorder recorder;
        recorder.start();

        // Define the function
        fdouble x(0.0);
        x.markInputAndDiff();  // Mark as input and request gradient

        fdouble fx = square(x) + 3.0 * x + 2.0;
        fx.markOutput();

        // Stop recording and get the graph
        recorder.stop();
        Graph graph = recorder.graph();

        // Compile the graph
        ForgeEngine compiler;
        auto kernel = compiler.compile(graph);

        // Create buffer for execution
        auto buffer = NodeValueBufferFactory::create(graph, *kernel);

        // Test at multiple points
        double test_points[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "   x    |  f(x)  | f'(x)  | Expected f'(x)\n";
        std::cout << "--------|--------|--------|---------------\n";

        for (double x_val : test_points) {
            // Set input value
            buffer->setValue(graph.diff_inputs[0], x_val);
            buffer->clearGradients();

            // Execute (automatically computes both function and gradient)
            kernel->execute(*buffer);

            // Get results
            double f_val = buffer->getValue(graph.outputs[0]);
            double df_val = buffer->getGradient(graph.diff_inputs[0]);

            // Expected values
            double expected_f = x_val * x_val + 3 * x_val + 2;
            double expected_df = 2 * x_val + 3;

            std::cout << std::setw(6) << x_val << "  | "
                      << std::setw(6) << f_val << " | "
                      << std::setw(6) << df_val << " | "
                      << std::setw(6) << expected_df << "\n";

            // Verify correctness
            if (std::abs(f_val - expected_f) > 1e-10) {
                std::cerr << "ERROR: f(x) mismatch!\n";
            }
            if (std::abs(df_val - expected_df) > 1e-10) {
                std::cerr << "ERROR: f'(x) mismatch!\n";
            }
        }
        std::cout << "\n";
    }

    // Example 2: Trigonometric function
    // f(x) = sin(x) + cos(x)
    {
        std::cout << "Example 2: f(x) = sin(x) + cos(x)\n";
        std::cout << "Expected: f'(x) = cos(x) - sin(x)\n\n";

        // Start recording
        GraphRecorder recorder;
        recorder.start();

        // Define the function
        fdouble x(0.0);
        x.markInputAndDiff();

        fdouble fx = sin(x) + cos(x);
        fx.markOutput();

        // Stop recording and get the graph
        recorder.stop();
        Graph graph = recorder.graph();

        // Compile with optimizations
        CompilerConfig config;
        config.enableOptimizations = true;
        ForgeEngine compiler(config);
        auto kernel = compiler.compile(graph);

        // Create buffer
        auto buffer = NodeValueBufferFactory::create(graph, *kernel);

        // Test at key angles
        struct TestPoint {
            double x;
            const char* label;
        };

        TestPoint test_points[] = {
            {0.0, "0"},
            {M_PI/4, "π/4"},
            {M_PI/2, "π/2"},
            {M_PI, "π"},
            {3*M_PI/2, "3π/2"}
        };

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "   x    |  f(x)  | f'(x)  | Expected f'(x)\n";
        std::cout << "--------|--------|--------|---------------\n";

        for (const auto& pt : test_points) {
            // Set input value
            buffer->setValue(graph.diff_inputs[0], pt.x);
            buffer->clearGradients();

            // Execute
            kernel->execute(*buffer);

            // Get results
            double f_val = buffer->getValue(graph.outputs[0]);
            double df_val = buffer->getGradient(graph.diff_inputs[0]);

            // Expected values
            double expected_f = std::sin(pt.x) + std::cos(pt.x);
            double expected_df = std::cos(pt.x) - std::sin(pt.x);

            std::cout << std::setw(6) << pt.label << "  | "
                      << std::setw(6) << f_val << " | "
                      << std::setw(6) << df_val << " | "
                      << std::setw(6) << expected_df << "\n";

            // Verify correctness (with tolerance for floating point)
            if (std::abs(f_val - expected_f) > 1e-6) {
                std::cerr << "ERROR: f(x) mismatch at " << pt.label << "!\n";
            }
            if (std::abs(df_val - expected_df) > 1e-6) {
                std::cerr << "ERROR: f'(x) mismatch at " << pt.label << "!\n";
            }
        }
        std::cout << "\n";
    }

    // Example 3: Composite function with multiple operations
    // f(x) = exp(x) * sin(x)
    {
        std::cout << "Example 3: f(x) = exp(x) * sin(x)\n";
        std::cout << "Expected: f'(x) = exp(x) * (sin(x) + cos(x))\n\n";

        // Start recording
        GraphRecorder recorder;
        recorder.start();

        // Define the function
        fdouble x(0.0);
        x.markInputAndDiff();

        fdouble fx = exp(x) * sin(x);
        fx.markOutput();

        // Stop recording
        recorder.stop();
        Graph graph = recorder.graph();

        // Compile
        ForgeEngine compiler;
        auto kernel = compiler.compile(graph);
        auto buffer = NodeValueBufferFactory::create(graph, *kernel);

        // Test points
        double test_points[] = {0.0, 0.5, 1.0, 1.5, 2.0};

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "   x    |  f(x)  | f'(x)  | Expected f'(x)\n";
        std::cout << "--------|--------|--------|---------------\n";

        for (double x_val : test_points) {
            // Set input value
            buffer->setValue(graph.diff_inputs[0], x_val);
            buffer->clearGradients();

            // Execute
            kernel->execute(*buffer);

            // Get results
            double f_val = buffer->getValue(graph.outputs[0]);
            double df_val = buffer->getGradient(graph.diff_inputs[0]);

            // Expected values
            double expected_f = std::exp(x_val) * std::sin(x_val);
            double expected_df = std::exp(x_val) * (std::sin(x_val) + std::cos(x_val));

            std::cout << std::setw(6) << x_val << "  | "
                      << std::setw(6) << f_val << " | "
                      << std::setw(6) << df_val << " | "
                      << std::setw(6) << expected_df << "\n";

            // Verify correctness
            if (std::abs(f_val - expected_f) > 1e-6) {
                std::cerr << "ERROR: f(x) mismatch!\n";
            }
            if (std::abs(df_val - expected_df) > 1e-6) {
                std::cerr << "ERROR: f'(x) mismatch!\n";
            }
        }
        std::cout << "\n";
    }

    std::cout << "All examples completed successfully!\n";
    return 0;
}
