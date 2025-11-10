/**
 * Performance Demonstration Example
 *
 * This example demonstrates the performance of FORGE's JIT compilation
 * by comparing optimized vs unoptimized compilation and measuring throughput.
 */

#include <types/fdouble.hpp>
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>

using namespace forge;
using namespace std::chrono;

// Helper function to measure execution time
template<typename Func>
double measureTime(Func f, int iterations = 1000) {
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        f();
    }
    auto end = high_resolution_clock::now();
    return duration<double, std::micro>(end - start).count() / iterations;
}

int main() {
    std::cout << "FORGE Performance Demonstration\n";
    std::cout << "================================\n\n";

    // Build a moderately complex function for benchmarking
    // f(x,y) = exp(sin(x) + cos(y)) * sqrt(x^2 + y^2) + log(abs(x*y) + 1)
    std::cout << "Test function: f(x,y) = exp(sin(x) + cos(y)) * sqrt(x^2 + y^2) + log(|x*y| + 1)\n\n";

    // Record the graph once
    GraphRecorder recorder;
    recorder.start();

    fdouble x(0.0);
    fdouble y(0.0);
    x.markInputAndDiff();
    y.markInputAndDiff();

    // Build complex expression
    fdouble sinx = sin(x);
    fdouble cosy = cos(y);
    fdouble exp_term = exp(sinx + cosy);
    fdouble x2 = square(x);
    fdouble y2 = square(y);
    fdouble sqrt_term = sqrt(x2 + y2);
    fdouble product = x * y;
    fdouble abs_product = abs(product);
    fdouble log_term = log(abs_product + 1.0);
    fdouble result = exp_term * sqrt_term + log_term;

    result.markOutput();

    recorder.stop();
    Graph graph = recorder.graph();

    std::cout << "Graph statistics:\n";
    std::cout << "  Total nodes: " << graph.nodes.size() << "\n";
    std::cout << "  Input nodes: " << graph.diff_inputs.size() << "\n";
    std::cout << "  Output nodes: " << graph.outputs.size() << "\n\n";

    // Compile with no optimizations
    std::cout << "Compiling without optimizations...\n";
    auto startCompile = high_resolution_clock::now();

    CompilerConfig noOptConfig;
    noOptConfig.enableOptimizations = false;
    ForgeEngine noOptCompiler(noOptConfig);
    auto noOptKernel = noOptCompiler.compile(graph);

    auto endCompile = high_resolution_clock::now();
    double noOptCompileTime = duration<double, std::milli>(endCompile - startCompile).count();

    std::cout << "  Compile time: " << std::fixed << std::setprecision(2)
              << noOptCompileTime << " ms\n";
    std::cout << "  Vector width: " << noOptKernel->getVectorWidth() << "\n\n";

    // Compile with optimizations
    std::cout << "Compiling with optimizations...\n";
    startCompile = high_resolution_clock::now();

    CompilerConfig optConfig;
    optConfig.enableOptimizations = true;
    optConfig.enableCSE = true;
    optConfig.enableAlgebraicSimplification = true;
    optConfig.enableInactiveFolding = true;
    ForgeEngine optCompiler(optConfig);
    auto optKernel = optCompiler.compile(graph);

    endCompile = high_resolution_clock::now();
    double optCompileTime = duration<double, std::milli>(endCompile - startCompile).count();

    std::cout << "  Compile time: " << std::fixed << std::setprecision(2)
              << optCompileTime << " ms\n\n";

    // Create buffers
    auto noOptBuffer = NodeValueBufferFactory::create(graph, *noOptKernel);
    auto optBuffer = NodeValueBufferFactory::create(graph, *optKernel);

    // Benchmark execution performance
    std::cout << "Benchmarking execution performance...\n";
    std::cout << "======================================\n";

    // Test points
    std::vector<std::pair<double, double>> testPoints = {
        {1.0, 2.0},
        {-0.5, 1.5},
        {3.14159, 2.71828},
        {0.1, 0.1},
        {10.0, -5.0}
    };

    for (const auto& [x_val, y_val] : testPoints) {
        std::cout << "\nTest point: x=" << x_val << ", y=" << y_val << "\n";

        // Prepare unoptimized buffer
        noOptBuffer->setValue(graph.diff_inputs[0], x_val);
        noOptBuffer->setValue(graph.diff_inputs[1], y_val);
        noOptBuffer->clearGradients();

        // Prepare optimized buffer
        optBuffer->setValue(graph.diff_inputs[0], x_val);
        optBuffer->setValue(graph.diff_inputs[1], y_val);
        optBuffer->clearGradients();

        // Measure unoptimized execution (1000 iterations for accurate timing)
        double noOptTime = measureTime([&]() {
            noOptKernel->execute(*noOptBuffer);
        }, 1000);

        // Measure optimized execution (1000 iterations for accurate timing)
        double optTime = measureTime([&]() {
            optKernel->execute(*optBuffer);
        }, 1000);

        // Get results (should be the same)
        double noOptResult = noOptBuffer->getValue(graph.outputs[0]);
        double optResult = optBuffer->getValue(graph.outputs[0]);
        double noOptGradX = noOptBuffer->getGradient(graph.diff_inputs[0]);
        double optGradX = optBuffer->getGradient(graph.diff_inputs[0]);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Function value: " << optResult;
        if (std::abs(noOptResult - optResult) > 1e-10) {
            std::cout << " (ERROR: mismatch!)";
        }
        std::cout << "\n";

        std::cout << "  Gradient \u2202f/\u2202x: " << optGradX;
        if (std::abs(noOptGradX - optGradX) > 1e-10) {
            std::cout << " (ERROR: mismatch!)";
        }
        std::cout << "\n";

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Execution time:\n";
        std::cout << "    No optimization: " << noOptTime << " \u03bcs\n";
        std::cout << "    With optimization: " << optTime << " \u03bcs\n";
        std::cout << "    Speedup: " << (noOptTime / optTime) << "x\n";
    }

    // Large-scale throughput test
    std::cout << "\n======================================\n";
    std::cout << "Throughput test (100,000 evaluations):\n";

    const int numEvals = 100000;

    // Prepare test data
    std::vector<double> x_values(numEvals);
    std::vector<double> y_values(numEvals);
    for (int i = 0; i < numEvals; ++i) {
        x_values[i] = static_cast<double>(i % 100) / 10.0 - 5.0;
        y_values[i] = static_cast<double>((i / 100) % 100) / 10.0 - 5.0;
    }

    // Throughput with optimized kernel
    auto startThroughput = high_resolution_clock::now();
    for (int i = 0; i < numEvals; ++i) {
        optBuffer->setValue(graph.diff_inputs[0], x_values[i]);
        optBuffer->setValue(graph.diff_inputs[1], y_values[i]);
        optBuffer->clearGradients();
        optKernel->execute(*optBuffer);
    }
    auto endThroughput = high_resolution_clock::now();

    double totalTime = duration<double>(endThroughput - startThroughput).count();
    double throughput = numEvals / totalTime;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total time: " << totalTime << " seconds\n";
    std::cout << "  Throughput: " << std::scientific << std::setprecision(2)
              << throughput << " evaluations/second\n";
    std::cout << "  Average time per evaluation: " << std::fixed << std::setprecision(3)
              << (totalTime * 1e6 / numEvals) << " \u03bcs\n";

    // Memory efficiency
    std::cout << "\n======================================\n";
    std::cout << "Memory efficiency:\n";

    size_t bufferSize = optBuffer->getNumNodes() * optKernel->getVectorWidth() * sizeof(double);
    size_t gradientSize = optBuffer->hasGradients() ? bufferSize : 0;

    std::cout << "  Buffer nodes: " << optBuffer->getNumNodes() << "\n";
    std::cout << "  Vector width: " << optKernel->getVectorWidth() << "\n";
    std::cout << "  Value buffer size: " << (bufferSize / 1024.0) << " KB\n";
    std::cout << "  Gradient buffer size: " << (gradientSize / 1024.0) << " KB\n";
    std::cout << "  Total memory: " << ((bufferSize + gradientSize) / 1024.0) << " KB\n";

    std::cout << "\nPerformance demonstration completed!\n";
    return 0;
}
