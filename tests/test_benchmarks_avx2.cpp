#include <gtest/gtest.h>
#include "test_functions_1d.hpp"
#include "../src/graph/graph_recorder.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/node_value_buffers/node_value_buffer.hpp"
#include <memory>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace forge;
using namespace forge::testing;
using namespace forge;

class AVX2Benchmark1D : public ::testing::Test {
protected:
    std::unique_ptr<StitchedKernel> kernel;
    std::unique_ptr<INodeValueBuffer> buffer;
    std::chrono::nanoseconds compile_time;
    NodeId input_node{0};
    NodeId output_node{0};
    size_t num_nodes{0};
    Graph graph;  // Store graph for buffer creation
    
    void SetUp() override {
        // Configure for AVX2 from the start
        auto config = CompilerConfig::Default();
        config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;
        
        // Measure compilation time
        auto compile_start = std::chrono::high_resolution_clock::now();
        
        // Record the tape using the AVX2 polynomial function
        GraphRecorder recorder;
        recorder.start();
        fdouble x(0.0);
        x.markInput();
        fdouble y = TestFunctions1D::avx2_polynomial(x);
        y.markOutput();
        
        // Get the graph
        graph = recorder.graph();
        num_nodes = graph.nodes.size();
        
        // Find input and output nodes
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            if (graph.nodes[i].op == OpCode::Input) {
                input_node = i;
                break;
            }
        }
        if (!graph.outputs.empty()) {
            output_node = graph.outputs[0];
        }
        
        // Compile with AVX2 configuration
        ForgeEngine compiler(config);
        kernel = compiler.compile(graph);
        
        auto compile_end = std::chrono::high_resolution_clock::now();
        compile_time = compile_end - compile_start;
        
        ASSERT_NE(kernel, nullptr) << "Failed to compile AVX2 kernel";
        
        // Create AVX2-aware workspace
        buffer = NodeValueBufferFactory::create(graph, *kernel);
        
        // Verify it's using AVX2 (vector width = 4)
        ASSERT_EQ(buffer->getVectorWidth(), 4) << "Workspace not configured for AVX2";
    }
};

TEST_F(AVX2Benchmark1D, VectorizedVsScalarPerformance) {
    const int warmup_iterations = 1000;
    const int benchmark_iterations = 100000;
    
    // Test with 4 different input values for vectorized execution
    std::vector<double> test_inputs = {1.0, 2.0, 3.0, 4.0};
    
    std::cout << "\n=== AVX2 Vectorized Benchmark: avx2_polynomial ===" << std::endl;
    std::cout << "Testing f(x) = 3x^3 - 2x^2 + 5x - 7" << std::endl;
    std::cout << "Input values: [" << test_inputs[0] << ", " << test_inputs[1] 
              << ", " << test_inputs[2] << ", " << test_inputs[3] << "]" << std::endl;
    
    // ====== WARM-UP PHASE ======
    // Warm-up AVX2 kernel
    for (int i = 0; i < warmup_iterations; i++) {
        buffer->setVectorValue(input_node, test_inputs);
        kernel->execute(*buffer);
        volatile auto dummy = buffer->getVectorValue(output_node);
        (void)dummy;
    }
    
    // Warm-up native scalar
    for (int i = 0; i < warmup_iterations; i++) {
        for (double x : test_inputs) {
            volatile double dummy = TestFunctions1D::avx2_polynomial_native(x);
            (void)dummy;
        }
    }
    
    // ====== VERIFICATION PHASE ======
    // Verify correctness before benchmarking
    buffer->setVectorValue(input_node, test_inputs);
    kernel->execute(*buffer);
    std::vector<double> avx2_results = buffer->getVectorValue(output_node);
    
    std::cout << "\nVerification:" << std::endl;
    bool all_correct = true;
    for (size_t i = 0; i < test_inputs.size(); i++) {
        double native_result = TestFunctions1D::avx2_polynomial_native(test_inputs[i]);
        double diff = std::abs(avx2_results[i] - native_result);
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  x=" << std::setw(4) << test_inputs[i] 
                  << " | AVX2=" << std::setw(10) << avx2_results[i]
                  << " | Native=" << std::setw(10) << native_result
                  << " | Diff=" << std::scientific << diff;
        
        if (diff < 1e-10) {
            std::cout << " [PASS]" << std::endl;
        } else {
            std::cout << " [FAIL]" << std::endl;
            all_correct = false;
        }
    }
    
    ASSERT_TRUE(all_correct) << "AVX2 results don't match native computation";
    
    // ====== BENCHMARK PHASE ======
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << "  Iterations: " << benchmark_iterations << std::endl;
    
    // Benchmark AVX2 vectorized execution (processes 4 values at once)
    auto avx2_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_iterations; i++) {
        kernel->execute(*buffer);  // Processes all 4 values in parallel
    }
    auto avx2_end = std::chrono::high_resolution_clock::now();
    auto avx2_time = std::chrono::duration_cast<std::chrono::nanoseconds>(avx2_end - avx2_start);
    
    // Benchmark native scalar execution (4 separate calls)
    auto native_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_iterations; i++) {
        volatile double r1 = TestFunctions1D::avx2_polynomial_native(test_inputs[0]);
        volatile double r2 = TestFunctions1D::avx2_polynomial_native(test_inputs[1]);
        volatile double r3 = TestFunctions1D::avx2_polynomial_native(test_inputs[2]);
        volatile double r4 = TestFunctions1D::avx2_polynomial_native(test_inputs[3]);
        (void)r1; (void)r2; (void)r3; (void)r4;
    }
    auto native_end = std::chrono::high_resolution_clock::now();
    auto native_time = std::chrono::duration_cast<std::chrono::nanoseconds>(native_end - native_start);
    
    // Calculate metrics
    double avx2_ns_per_batch = (double)avx2_time.count() / benchmark_iterations;
    double native_ns_per_batch = (double)native_time.count() / benchmark_iterations;
    double avx2_ns_per_value = avx2_ns_per_batch / 4.0;  // Per single value
    double native_ns_per_value = native_ns_per_batch / 4.0;  // Per single value
    double speedup = (double)native_time.count() / avx2_time.count();
    
    std::cout << "  AVX2 total:        " << avx2_time.count() / 1e6 << " ms" << std::endl;
    std::cout << "  Native total:      " << native_time.count() / 1e6 << " ms" << std::endl;
    std::cout << std::setprecision(1);
    std::cout << "  AVX2 per batch:    " << avx2_ns_per_batch << " ns (4 values)" << std::endl;
    std::cout << "  Native per batch:  " << native_ns_per_batch << " ns (4 values)" << std::endl;
    std::cout << "  AVX2 per value:    " << avx2_ns_per_value << " ns" << std::endl;
    std::cout << "  Native per value:  " << native_ns_per_value << " ns" << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "  Speedup:           " << speedup << "x ";
    
    if (speedup > 1.0) {
        std::cout << "(AVX2 faster)" << std::endl;
    } else if (speedup < 1.0) {
        std::cout << "(Native faster)" << std::endl;
    } else {
        std::cout << "(Equal)" << std::endl;
    }
    
    // Theoretical speedup analysis
    std::cout << "\nAnalysis:" << std::endl;
    std::cout << "  Theoretical max speedup: 4.0x (perfect SIMD)" << std::endl;
    std::cout << "  Achieved efficiency:     " << (speedup / 4.0 * 100.0) << "%" << std::endl;
    std::cout << "  Compile time:            " << compile_time.count() / 1e6 << " ms" << std::endl;
    
    // Record properties for test infrastructure
    RecordProperty("compile_time_ms", compile_time.count() / 1e6);
    RecordProperty("avx2_time_ms", avx2_time.count() / 1e6);
    RecordProperty("native_time_ms", native_time.count() / 1e6);
    RecordProperty("avx2_ns_per_batch", avx2_ns_per_batch);
    RecordProperty("native_ns_per_batch", native_ns_per_batch);
    RecordProperty("speedup", speedup);
    RecordProperty("efficiency_percent", speedup / 4.0 * 100.0);
    RecordProperty("iterations", benchmark_iterations);
    
    // Performance assertion
    EXPECT_GT(speedup, 0.0) << "Speedup calculation failed";
    
    // Add a soft expectation for reasonable performance
    // AVX2 should be at least somewhat faster for this simple polynomial
    if (speedup < 0.5) {
        std::cout << "\nWARNING: AVX2 is significantly slower than expected!" << std::endl;
        std::cout << "This might indicate an issue with the AVX2 implementation." << std::endl;
    }
}

// Test with different batch sizes to understand vectorization overhead
TEST_F(AVX2Benchmark1D, DifferentInputSets) {
    const int iterations = 10000;
    
    // Different test input sets to see how AVX2 performs with various values
    std::vector<std::vector<double>> input_sets = {
        {0.0, 0.0, 0.0, 0.0},      // All zeros
        {1.0, 1.0, 1.0, 1.0},      // All ones (broadcast case)
        {-2.0, -1.0, 1.0, 2.0},    // Symmetric around zero
        {0.1, 0.5, 2.5, 10.0},     // Different magnitudes
        {-10.0, -5.0, 5.0, 10.0}   // Larger range
    };
    
    std::cout << "\n=== AVX2 Different Input Sets Test ===" << std::endl;
    
    for (const auto& inputs : input_sets) {
        std::cout << "\nTesting inputs: [" << inputs[0] << ", " << inputs[1] 
                  << ", " << inputs[2] << ", " << inputs[3] << "]" << std::endl;
        
        // Set inputs and execute
        buffer->setVectorValue(input_node, inputs);
        
        // Time AVX2 execution
        auto avx2_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            kernel->execute(*buffer);
        }
        auto avx2_end = std::chrono::high_resolution_clock::now();
        auto avx2_time = std::chrono::duration_cast<std::chrono::microseconds>(avx2_end - avx2_start);
        
        // Get results for verification
        std::vector<double> avx2_results = buffer->getVectorValue(output_node);
        
        // Time native execution
        auto native_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            for (double x : inputs) {
                volatile double r = TestFunctions1D::avx2_polynomial_native(x);
                (void)r;
            }
        }
        auto native_end = std::chrono::high_resolution_clock::now();
        auto native_time = std::chrono::duration_cast<std::chrono::microseconds>(native_end - native_start);
        
        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < inputs.size(); i++) {
            double expected = TestFunctions1D::avx2_polynomial_native(inputs[i]);
            double diff = std::abs(avx2_results[i] - expected);
            if (diff > 1e-10) {
                correct = false;
                std::cout << "  ERROR at x=" << inputs[i] 
                          << ": AVX2=" << avx2_results[i] 
                          << ", Expected=" << expected << std::endl;
            }
        }
        
        double speedup = (double)native_time.count() / avx2_time.count();
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  AVX2 time:  " << avx2_time.count() << " µs" << std::endl;
        std::cout << "  Native time: " << native_time.count() << " µs" << std::endl;
        std::cout << "  Speedup:     " << speedup << "x";
        if (correct) {
            std::cout << " [CORRECT]" << std::endl;
        } else {
            std::cout << " [INCORRECT]" << std::endl;
        }
        
        EXPECT_TRUE(correct) << "Results don't match for input set";
    }
}