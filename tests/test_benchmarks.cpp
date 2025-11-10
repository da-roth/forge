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

using namespace forge;
using namespace forge::testing;
using namespace forge;

class JitBenchmark1D_New2 : public ::testing::TestWithParam<TestCase1D> {
protected:
    std::unique_ptr<StitchedKernel> kernel;
    std::unique_ptr<INodeValueBuffer> buffer;
    std::chrono::nanoseconds compile_time;
    NodeId input_node{0};
    NodeId output_node{0};
    size_t num_nodes{0};
    Graph graph;  // Store graph for buffer creation
    
    void SetUp() override {
        auto& test_case = GetParam();
        
        // Measure compilation time separately
        auto compile_start = std::chrono::high_resolution_clock::now();
        
        GraphRecorder recorder;
        recorder.start();
        fdouble x(0.0);
        x.markInput();
        fdouble y = test_case.func(x);
        y.markOutput();
        
        // Get the graph (store as member for buffer creation)
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
        
        // Compile with new compiler (JitRuntime is now static/shared - no need to keep compiler alive)
        ForgeEngine compiler;
        kernel = compiler.compile(graph);
        
        auto compile_end = std::chrono::high_resolution_clock::now();
        compile_time = compile_end - compile_start;
        
        ASSERT_NE(kernel, nullptr) << "Failed to compile: " << test_case.name;
        
        // Create NodeValueBuffer based on kernel requirements
        buffer = NodeValueBufferFactory::create(graph, *kernel);
        
        // Phase 2.2: Constants are now loaded from ConstPool by the kernel itself
        // No need to pre-fill constants in buffer anymore!
    }
};

TEST_P(JitBenchmark1D_New2, Performance) {
    auto& test_case = GetParam();
    const int warmup_iterations = 1000;
    const int benchmark_iterations = 10000;
    
    // Warm-up phase for JIT
    for (int i = 0; i < warmup_iterations; i++) {
        buffer->setValue(input_node, i * 0.001);
        kernel->execute(*buffer);
        volatile double dummy = buffer->getValue(output_node);
        (void)dummy;
    }
    
    // Warm-up phase for native
    for (int i = 0; i < warmup_iterations; i++) {
        volatile double dummy = test_case.native_func(i * 0.001);
        (void)dummy;
    }
    
    // Set input value once before benchmarking
    buffer->setValue(input_node, 2.5);
    
    // Benchmark ONLY JIT execute() calls
    auto jit_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_iterations; i++) {
        kernel->execute(*buffer);
    }
    auto jit_end = std::chrono::high_resolution_clock::now();
    auto jit_time = std::chrono::duration_cast<std::chrono::nanoseconds>(jit_end - jit_start);
    
    // Get output once to ensure computation happened
    volatile double jit_result = buffer->getValue(output_node);
    
    // Benchmark native function with same constant input
    volatile double native_input = 2.5;
    volatile double native_result = test_case.native_func(native_input);
    auto native_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_iterations; i++) {
        volatile double result = test_case.native_func(native_input);
        (void)result;
    }
    auto native_end = std::chrono::high_resolution_clock::now();
    auto native_time = std::chrono::duration_cast<std::chrono::nanoseconds>(native_end - native_start);
    
    // Calculate metrics
    double speedup = (double)native_time.count() / jit_time.count();
    double jit_ns_per_call = (double)jit_time.count() / benchmark_iterations;
    double native_ns_per_call = (double)native_time.count() / benchmark_iterations;
    
    // Build detailed message for VS Test Explorer
    std::stringstream summary;
    summary << std::fixed << std::setprecision(2);
    summary << "Compile: " << compile_time.count() / 1e6 << "ms, ";
    summary << "JIT: " << jit_ns_per_call << "ns/call, ";
    summary << "Native: " << native_ns_per_call << "ns/call, ";
    summary << "Speedup: " << speedup << "x";
    if (speedup > 1.0) {
        summary << " (JIT faster)";
    } else {
        summary << " (Native faster)";
    }
    
    // Report results (visible in test output)
    std::cout << "\n=== Benchmark (NEW2): " << test_case.name << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Compile time:    " << compile_time.count() / 1e6 << " ms" << std::endl;
    std::cout << "  Iterations:      " << benchmark_iterations << std::endl;
    std::cout << "  JIT total:       " << jit_time.count() / 1e6 << " ms" << std::endl;
    std::cout << "  Native total:    " << native_time.count() / 1e6 << " ms" << std::endl;
    std::cout << std::setprecision(1);
    std::cout << "  JIT per call:    " << jit_ns_per_call << " ns" << std::endl;
    std::cout << "  Native per call: " << native_ns_per_call << " ns" << std::endl;
    // Verification with all test inputs
    std::cout << std::setprecision(6);
    
    // Get test inputs - use default if none specified
    std::vector<double> test_inputs = test_case.test_inputs;
    if (test_inputs.empty()) {
        // Default inputs if none specified
        test_inputs = {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.5, 3.0};
    }
    
    // Run verification for each input
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        double input_val = test_inputs[i];
        
        // Skip zero if requested
        if (test_case.skip_zero && std::abs(input_val) < 1e-10) {
            continue;
        }
        
        // JIT execution
        buffer->setValue(input_node, input_val);
        kernel->execute(*buffer);
        double jit_val = buffer->getValue(output_node);
        
        // Native execution
        double native_val = test_case.native_func(input_val);
        
        // Print verification line
        std::cout << "  Verification " << (i + 1) << ": "
                  << "JIT=" << jit_val 
                  << " Native=" << native_val 
                  << " (x=" << input_val << ")" << std::endl;
    }
    
    std::cout << std::setprecision(2);
    std::cout << "  Speedup:         " << speedup << "x ";
    
    if (speedup > 1.0) {
        std::cout << "(JIT faster)" << std::endl;
    } else if (speedup < 1.0) {
        std::cout << "(Native faster)" << std::endl;
    } else {
        std::cout << "(Equal)" << std::endl;
    }
    
    // Record properties for test infrastructure
    RecordProperty("compile_time_ms", compile_time.count() / 1e6);
    RecordProperty("jit_time_ms", jit_time.count() / 1e6);
    RecordProperty("native_time_ms", native_time.count() / 1e6);
    RecordProperty("jit_ns_per_call", jit_ns_per_call);
    RecordProperty("native_ns_per_call", native_ns_per_call);
    RecordProperty("speedup", speedup);
    RecordProperty("iterations", benchmark_iterations);
    
    // For VS Test Explorer: Use test name property to show summary
    RecordProperty("TestSummary", summary.str());
    
    // Add performance assertion with detailed message
    EXPECT_GT(speedup, 0.0) << "\n" << summary.str() << "\n";
    
    // Also use SCOPED_TRACE for additional visibility
    SCOPED_TRACE(summary.str());
}

// Instantiate benchmark tests with suitable subset
INSTANTIATE_TEST_SUITE_P(
    Benchmark1D_New2_File_test_phase3_benchmark_1d_new2,
    JitBenchmark1D_New2,
    ::testing::ValuesIn(getBenchmarkTestCases1D()),
    [](const ::testing::TestParamInfo<TestCase1D>& info) {
        return info.param.name;
    }
);