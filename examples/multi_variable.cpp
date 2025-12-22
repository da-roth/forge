/**
 * Multi-Variable Gradient Example
 *
 * This example demonstrates gradient computation with multiple input variables.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <native/fdouble.hpp>
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>
#include <iostream>
#include <iomanip>

using namespace forge;

int main() {
    std::cout << "FORGE Multi-Variable Gradient Example\n";
    std::cout << "======================================\n\n";

    // Example 1: Two-variable function
    // f(x,y) = x^2 + y^2 + x*y
    {
        std::cout << "Example 1: f(x,y) = x^2 + y^2 + x*y\n";
        std::cout << "Partial derivatives:\n";
        std::cout << "  \u2202f/\u2202x = 2x + y\n";
        std::cout << "  \u2202f/\u2202y = 2y + x\n\n";

        // Start recording
        GraphRecorder recorder;
        recorder.start();

        // Define inputs
        fdouble x(0.0);
        fdouble y(0.0);
        x.markInputAndDiff();
        y.markInputAndDiff();

        // Define the function: f(x,y) = x^2 + y^2 + x*y
        fdouble f = square(x) + square(y) + x * y;
        f.markOutput();

        // Stop recording
        recorder.stop();
        Graph graph = recorder.graph();

        // Compile
        ForgeEngine compiler;
        auto kernel = compiler.compile(graph);
        auto buffer = NodeValueBufferFactory::create(graph, *kernel);

        // Test points
        struct TestPoint {
            double x, y;
        };

        TestPoint test_points[] = {
            {1.0, 1.0},
            {2.0, 3.0},
            {-1.0, 2.0},
            {0.0, 0.0},
            {3.0, -2.0}
        };

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  x   |  y   |  f(x,y) | \u2202f/\u2202x | \u2202f/\u2202y | Expected \u2202f/\u2202x | Expected \u2202f/\u2202y\n";
        std::cout << "------|------|---------|-------|-------|----------------|----------------\n";

        for (const auto& pt : test_points) {
            // Set input values
            buffer->setValue(graph.diff_inputs[0], pt.x);  // x
            buffer->setValue(graph.diff_inputs[1], pt.y);  // y
            buffer->clearGradients();

            // Execute (automatically computes both function and gradients)
            kernel->execute(*buffer);

            // Get results
            double f_val = buffer->getValue(graph.outputs[0]);
            double dfdx = buffer->getGradient(graph.diff_inputs[0]);
            double dfdy = buffer->getGradient(graph.diff_inputs[1]);

            // Expected values
            double expected_f = pt.x * pt.x + pt.y * pt.y + pt.x * pt.y;
            double expected_dfdx = 2 * pt.x + pt.y;
            double expected_dfdy = 2 * pt.y + pt.x;

            std::cout << std::setw(5) << pt.x << " | "
                      << std::setw(4) << pt.y << " | "
                      << std::setw(7) << f_val << " | "
                      << std::setw(5) << dfdx << " | "
                      << std::setw(5) << dfdy << " | "
                      << std::setw(14) << expected_dfdx << " | "
                      << std::setw(14) << expected_dfdy << "\n";

            // Verify
            if (std::abs(f_val - expected_f) > 1e-10 ||
                std::abs(dfdx - expected_dfdx) > 1e-10 ||
                std::abs(dfdy - expected_dfdy) > 1e-10) {
                std::cerr << "ERROR: Mismatch detected!\n";
            }
        }
        std::cout << "\n";
    }

    // Example 2: Three-variable function with transcendental functions
    // f(x,y,z) = exp(x) + sin(y) + z^2
    {
        std::cout << "Example 2: f(x,y,z) = exp(x) + sin(y) + z^2\n";
        std::cout << "Partial derivatives:\n";
        std::cout << "  \u2202f/\u2202x = exp(x)\n";
        std::cout << "  \u2202f/\u2202y = cos(y)\n";
        std::cout << "  \u2202f/\u2202z = 2z\n\n";

        // Start recording
        GraphRecorder recorder;
        recorder.start();

        // Define inputs
        fdouble x(0.0);
        fdouble y(0.0);
        fdouble z(0.0);
        x.markInputAndDiff();
        y.markInputAndDiff();
        z.markInputAndDiff();

        // Define the function
        fdouble f = exp(x) + sin(y) + square(z);
        f.markOutput();

        // Stop recording
        recorder.stop();
        Graph graph = recorder.graph();

        // Compile with optimizations
        CompilerConfig config;
        config.enableOptimizations = true;
        ForgeEngine compiler(config);
        auto kernel = compiler.compile(graph);
        auto buffer = NodeValueBufferFactory::create(graph, *kernel);

        // Test specific points
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Testing at (x=1, y=\u03c0/2, z=2):\n";

        double x_val = 1.0;
        double y_val = M_PI / 2;
        double z_val = 2.0;

        buffer->setValue(graph.diff_inputs[0], x_val);
        buffer->setValue(graph.diff_inputs[1], y_val);
        buffer->setValue(graph.diff_inputs[2], z_val);
        buffer->clearGradients();

        kernel->execute(*buffer);

        double f_val = buffer->getValue(graph.outputs[0]);
        double dfdx = buffer->getGradient(graph.diff_inputs[0]);
        double dfdy = buffer->getGradient(graph.diff_inputs[1]);
        double dfdz = buffer->getGradient(graph.diff_inputs[2]);

        // Expected values
        double expected_f = std::exp(x_val) + std::sin(y_val) + z_val * z_val;
        double expected_dfdx = std::exp(x_val);
        double expected_dfdy = std::cos(y_val);
        double expected_dfdz = 2 * z_val;

        std::cout << "  f(1, \u03c0/2, 2) = " << f_val << " (expected: " << expected_f << ")\n";
        std::cout << "  \u2202f/\u2202x = " << dfdx << " (expected: " << expected_dfdx << ")\n";
        std::cout << "  \u2202f/\u2202y = " << dfdy << " (expected: " << expected_dfdy << ")\n";
        std::cout << "  \u2202f/\u2202z = " << dfdz << " (expected: " << expected_dfdz << ")\n\n";

        // Verify
        if (std::abs(f_val - expected_f) > 1e-6 ||
            std::abs(dfdx - expected_dfdx) > 1e-6 ||
            std::abs(dfdy - expected_dfdy) > 1e-6 ||
            std::abs(dfdz - expected_dfdz) > 1e-6) {
            std::cerr << "ERROR: Mismatch detected!\n";
        } else {
            std::cout << "All gradients correct!\n\n";
        }
    }

    // Example 3: Chain rule with nested functions
    // f(x,y) = sin(x*y) + exp(x-y)
    {
        std::cout << "Example 3: f(x,y) = sin(x*y) + exp(x-y)\n";
        std::cout << "Partial derivatives (via chain rule):\n";
        std::cout << "  \u2202f/\u2202x = y*cos(x*y) + exp(x-y)\n";
        std::cout << "  \u2202f/\u2202y = x*cos(x*y) - exp(x-y)\n\n";

        GraphRecorder recorder;
        recorder.start();

        fdouble x(0.0);
        fdouble y(0.0);
        x.markInputAndDiff();
        y.markInputAndDiff();

        fdouble f = sin(x * y) + exp(x - y);
        f.markOutput();

        recorder.stop();
        Graph graph = recorder.graph();

        ForgeEngine compiler;
        auto kernel = compiler.compile(graph);
        auto buffer = NodeValueBufferFactory::create(graph, *kernel);

        // Test at (2, 1)
        double x_val = 2.0;
        double y_val = 1.0;

        buffer->setValue(graph.diff_inputs[0], x_val);
        buffer->setValue(graph.diff_inputs[1], y_val);
        buffer->clearGradients();

        kernel->execute(*buffer);

        double f_val = buffer->getValue(graph.outputs[0]);
        double dfdx = buffer->getGradient(graph.diff_inputs[0]);
        double dfdy = buffer->getGradient(graph.diff_inputs[1]);

        // Expected
        double expected_f = std::sin(x_val * y_val) + std::exp(x_val - y_val);
        double expected_dfdx = y_val * std::cos(x_val * y_val) + std::exp(x_val - y_val);
        double expected_dfdy = x_val * std::cos(x_val * y_val) - std::exp(x_val - y_val);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "At point (x=2, y=1):\n";
        std::cout << "  f(2,1) = " << f_val << " (expected: " << expected_f << ")\n";
        std::cout << "  \u2202f/\u2202x = " << dfdx << " (expected: " << expected_dfdx << ")\n";
        std::cout << "  \u2202f/\u2202y = " << dfdy << " (expected: " << expected_dfdy << ")\n\n";

        if (std::abs(f_val - expected_f) < 1e-6 &&
            std::abs(dfdx - expected_dfdx) < 1e-6 &&
            std::abs(dfdy - expected_dfdy) < 1e-6) {
            std::cout << "Chain rule gradient computation successful!\n";
        }
    }

    std::cout << "\nAll multi-variable examples completed successfully!\n";
    return 0;
}
