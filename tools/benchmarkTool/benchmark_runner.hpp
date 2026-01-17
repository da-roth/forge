#pragma once

#include <functional>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <limits>
#include <native/fdouble.hpp>
#include "../../src/graph/graph_recorder.hpp"
#include "../../src/graph/graph_optimizer.hpp"
#include "../../src/compiler/forge_engine.hpp"
#include "../../src/compiler/interfaces/node_value_buffer.hpp"
#include "../../src/compiler/compiler_config.hpp"

namespace forge {
namespace tools {

// Result structure for comprehensive benchmarking
struct ComprehensiveBenchmarkResult {
    std::string functionName;
    double graphOptimizationTimeMs;
    double kernelCreationTimeMs;
    double kernelEvalTimeNs;
    double nativeEvalTimeNs;
    double speedup;
    std::vector<double> testInputs;  // Test inputs used for this function
    std::vector<bool> verificationPerInput;  // Verification status for each input
    size_t graphNodes;
    
    // Optimization statistics
    size_t originalNodeCount = 0;
    size_t optimizedNodeCount = 0;
    size_t inactiveNodesFolded = 0;
    size_t duplicatesEliminated = 0;
    size_t algebraicSimplifications = 0;
    size_t stabilityFixes = 0;
    size_t deadNodesMarked = 0;
    double optimizationRatio = 0.0;
    
    // Optimization timing (in milliseconds)
    double inactiveFoldingTimeMs = 0.0;
    double cseTimeMs = 0.0;
    double algebraicTimeMs = 0.0;
    double stabilityTimeMs = 0.0;
    double totalOptimizationTimeMs = 0.0;
    int passesPerformed = 0;
    
    // Node type breakdown
    size_t inputNodes = 0;
    size_t constantNodes = 0;
    size_t arithmeticNodes = 0;  // Add, Sub, Mul, Div, etc.
    size_t transcendentalNodes = 0;  // Exp, Log, Sin, Cos, etc.
    size_t comparisonNodes = 0;  // CmpLT, CmpGT, etc.
    size_t controlFlowNodes = 0;  // If, conditional
    
    // AVX2 benchmark results (if tested)
    bool avx2Tested = false;
    double avx2CompileTimeMs = 0.0;
    double avx2TimePerEvaluation = 0.0;
    double avx2RelativeSpeedup = 0.0;  // vs SSE2
    size_t avx2VectorWidth = 0;
};

// Configuration for the benchmark runner (internal use)
struct BenchmarkConfig {
    int warmupIterations = 5;
    int benchmarkIterations = 10;
    bool verifyResults = true;
    double tolerance = 1e-10;
    bool testAvx2 = true;  // Test AVX2 in addition to SSE2
};

// Main benchmark runner class
class BenchmarkRunner {
private:
    std::vector<ComprehensiveBenchmarkResult> results_;
    BenchmarkConfig config_;  // Fixed internal config
    size_t maxInputCount_ = 0;  // Track maximum number of inputs across all functions
    
public:
    BenchmarkRunner() : config_() {}
    
    // Template version for automatic type deduction
    template<typename FuncDouble, typename FuncTP>
    void AddFunction(const std::string& name, 
                     FuncDouble nativeFunc, 
                     FuncTP tapeFunc, 
                     const std::vector<double>& testInputs = {0.5, 1.0, 1.5, 2.0}) {
        // Convert to std::function and call the implementation
        AddFunctionImpl(name,
                       std::function<double(double)>(nativeFunc),
                       std::function<forge::fdouble(forge::fdouble)>(tapeFunc),
                       testInputs);
    }

private:
    void AddFunctionImpl(const std::string& name,
                        std::function<double(double)> nativeFunc,
                        std::function<forge::fdouble(forge::fdouble)> tapeFunc, 
                        const std::vector<double>& testInputs) {
        using namespace std::chrono;
        
        ComprehensiveBenchmarkResult result;
        result.functionName = name;
        result.testInputs = testInputs;
        
        // Track maximum number of inputs
        maxInputCount_ = std::max(maxInputCount_, testInputs.size());
        
        // Step 1: Record the tape and measure graph creation time
        auto graphStart = high_resolution_clock::now();
        
        forge::GraphRecorder recorder;
        recorder.start();

        forge::fdouble x(0.0);
        x.markInput();

        forge::fdouble y = tapeFunc(x);
        y.markOutput();
        
        recorder.stop();
        forge::Graph graph = recorder.graph();
        
        auto graphEnd = high_resolution_clock::now();
        result.graphOptimizationTimeMs = duration<double, std::milli>(graphEnd - graphStart).count();
        result.graphNodes = graph.nodes.size();
        
        // Analyze graph node types
        for (const auto& node : graph.nodes) {
            switch (node.op) {
                case forge::OpCode::Input:
                    result.inputNodes++;
                    break;
                case forge::OpCode::Constant:
                case forge::OpCode::IntConstant:
                case forge::OpCode::BoolConstant:
                    result.constantNodes++;
                    break;
                case forge::OpCode::Add:
                case forge::OpCode::Sub:
                case forge::OpCode::Mul:
                case forge::OpCode::Div:
                case forge::OpCode::Neg:
                case forge::OpCode::Abs:
                case forge::OpCode::Square:
                case forge::OpCode::Recip:
                case forge::OpCode::Mod:
                case forge::OpCode::Min:
                case forge::OpCode::Max:
                    result.arithmeticNodes++;
                    break;
                case forge::OpCode::Exp:
                case forge::OpCode::Log:
                case forge::OpCode::Sqrt:
                case forge::OpCode::Pow:
                case forge::OpCode::Sin:
                case forge::OpCode::Cos:
                case forge::OpCode::Tan:
                    result.transcendentalNodes++;
                    break;
                case forge::OpCode::CmpLT:
                case forge::OpCode::CmpLE:
                case forge::OpCode::CmpGT:
                case forge::OpCode::CmpGE:
                case forge::OpCode::CmpEQ:
                case forge::OpCode::CmpNE:
                    result.comparisonNodes++;
                    break;
                case forge::OpCode::If:
                    result.controlFlowNodes++;
                    break;
                default:
                    // Other nodes not categorized
                    break;
            }
        }
        
        // Step 1.5: Optimize the graph and capture optimization statistics
        forge::GraphOptimizer optimizer;
        forge::GraphOptimizer::OptimizationConfig optConfig;
        optConfig.enableInactiveFolding = true;
        optConfig.enableCSE = true;
        optConfig.enableAlgebraicSimplification = true;
        optConfig.enableStabilityCleaning = true;
        optConfig.maxOptimizationPasses = 5;
        optimizer.setConfig(optConfig);
        
        auto optStart = high_resolution_clock::now();
        forge::Graph optimizedGraph = optimizer.optimize(graph);
        auto optEnd = high_resolution_clock::now();
        
        // Capture optimization statistics
        const auto& optStats = optimizer.getLastStats();
        result.originalNodeCount = optStats.originalNodeCount;
        result.optimizedNodeCount = optStats.optimizedNodeCount;
        result.inactiveNodesFolded = optStats.inactiveNodesFolded;
        result.duplicatesEliminated = optStats.duplicatesEliminated;
        result.algebraicSimplifications = optStats.algebraicSimplifications;
        result.stabilityFixes = optStats.stabilityFixes;
        
        // Capture timing information
        result.inactiveFoldingTimeMs = optStats.inactiveFoldingTimeMs;
        result.cseTimeMs = optStats.cseTimeMs;
        result.algebraicTimeMs = optStats.algebraicTimeMs;
        result.stabilityTimeMs = optStats.stabilityTimeMs;
        result.totalOptimizationTimeMs = optStats.totalOptimizationTimeMs;
        result.passesPerformed = optStats.passesPerformed;
        
        // Count dead nodes
        size_t deadCount = 0;
        for (const auto& node : optimizedGraph.nodes) {
            if (node.isDead) deadCount++;
        }
        result.deadNodesMarked = deadCount;
        result.optimizationRatio = (optStats.originalNodeCount > 0) ? 
            (100.0 * deadCount / optStats.originalNodeCount) : 0.0;
        
        // Step 2: Compile the kernel and measure kernel creation time
        auto kernelStart = high_resolution_clock::now();
        
        // Use a config that disables optimization (we already optimized)
        forge::CompilerConfig compilerConfig;
        compilerConfig.enableOptimizations = false;  // Already optimized above
        forge::ForgeEngine compiler(compilerConfig);
        auto kernel = compiler.compile(optimizedGraph);
        
        auto kernelEnd = high_resolution_clock::now();
        result.kernelCreationTimeMs = duration<double, std::milli>(kernelEnd - kernelStart).count();
        
        // Step 3: Setup workspace (use optimized graph for workspace)
        auto buffer = forge::NodeValueBufferFactory::create(optimizedGraph, *kernel);
        forge::NodeId inputNode = 0;
        forge::NodeId outputNode = optimizedGraph.outputs[0];
        
        // Use middle value from test inputs for benchmarking
        double testValue = testInputs[testInputs.size() / 2];
        
        // Step 4: Warmup for kernel
        double inputData[4] = {testValue, testValue, testValue, testValue};
        double outputData[4];
        for (int i = 0; i < config_.warmupIterations; ++i) {
            buffer->setLanes(inputNode, inputData);
            kernel->execute(*buffer);
            buffer->getLanes(outputNode, outputData);
            volatile double dummy = outputData[0];
            (void)dummy;
        }

        // Step 5: Benchmark kernel execution
        auto kernelBenchStart = high_resolution_clock::now();
        for (int i = 0; i < config_.benchmarkIterations; ++i) {
            buffer->setLanes(inputNode, inputData);
            kernel->execute(*buffer);
            buffer->getLanes(outputNode, outputData);
            volatile double dummy = outputData[0];
            (void)dummy;
        }
        auto kernelBenchEnd = high_resolution_clock::now();
        result.kernelEvalTimeNs = duration<double, std::nano>(kernelBenchEnd - kernelBenchStart).count() 
                                  / config_.benchmarkIterations;
        
        // Step 6: Warmup for native
        for (int i = 0; i < config_.warmupIterations; ++i) {
            volatile double dummy = nativeFunc(testValue);
            (void)dummy;
        }
        
        // Step 7: Benchmark native execution
        auto nativeBenchStart = high_resolution_clock::now();
        for (int i = 0; i < config_.benchmarkIterations; ++i) {
            volatile double dummy = nativeFunc(testValue);
            (void)dummy;
        }
        auto nativeBenchEnd = high_resolution_clock::now();
        result.nativeEvalTimeNs = duration<double, std::nano>(nativeBenchEnd - nativeBenchStart).count() 
                                  / config_.benchmarkIterations;
        
        // Calculate speedup
        result.speedup = result.nativeEvalTimeNs / result.kernelEvalTimeNs;
        
        // Step 8: Test AVX2 if requested
        if (config_.testAvx2) {
            try {
                // Create AVX2 compiler configuration
                forge::CompilerConfig avx2Config;
                avx2Config.enableOptimizations = false;  // Already optimized
                avx2Config.instructionSet = forge::CompilerConfig::InstructionSet::AVX2_PACKED;
                
                // Compile with AVX2
                auto avx2CompileStart = high_resolution_clock::now();
                forge::ForgeEngine avx2Compiler(avx2Config);
                auto avx2Kernel = avx2Compiler.compile(optimizedGraph);
                auto avx2CompileEnd = high_resolution_clock::now();
                result.avx2CompileTimeMs = duration<double, std::milli>(avx2CompileEnd - avx2CompileStart).count();
                
                // Create AVX2 workspace
                auto avx2Buffer = forge::NodeValueBufferFactory::create(optimizedGraph, *avx2Kernel);
                result.avx2VectorWidth = avx2Buffer->getVectorWidth();
                
                if (result.avx2VectorWidth == 4) {
                    // Test with 4 inputs simultaneously
                    double batch[4];
                    for (size_t i = 0; i < 4; ++i) {
                        batch[i] = (i < testInputs.size()) ? testInputs[i] : testInputs.back();
                    }

                    // Warmup
                    double avx2OutputData[4];
                    for (int i = 0; i < config_.warmupIterations; ++i) {
                        avx2Buffer->setLanes(inputNode, batch);
                        avx2Kernel->execute(*avx2Buffer);
                        avx2Buffer->getLanes(outputNode, avx2OutputData);
                        volatile double d = avx2OutputData[0];
                        (void)d;
                    }

                    // Benchmark AVX2 execution
                    auto avx2BenchStart = high_resolution_clock::now();
                    for (int i = 0; i < config_.benchmarkIterations; ++i) {
                        avx2Buffer->setLanes(inputNode, batch);
                        avx2Kernel->execute(*avx2Buffer);
                        avx2Buffer->getLanes(outputNode, avx2OutputData);
                        volatile double d = avx2OutputData[0];
                        (void)d;
                    }
                    auto avx2BenchEnd = high_resolution_clock::now();
                    
                    // AVX2 processes 4 values at once
                    // Store the total time for 4 values (for fair comparison)
                    double avx2TotalTimeFor4 = duration<double, std::nano>(avx2BenchEnd - avx2BenchStart).count() 
                                               / config_.benchmarkIterations;
                    result.avx2TimePerEvaluation = avx2TotalTimeFor4 / 4.0;  // Per-value time
                    result.avx2RelativeSpeedup = result.kernelEvalTimeNs / result.avx2TimePerEvaluation;
                    result.avx2Tested = true;
                }
            } catch (const std::exception& e) {
                // AVX2 not available or compilation failed - silently continue
                result.avx2Tested = false;
            }
        }
        
        // Verify results if requested
        result.verificationPerInput.clear();
        if (config_.verifyResults) {
            for (double input : testInputs) {
                double verifyInputData[4] = {input, input, input, input};
                buffer->setLanes(inputNode, verifyInputData);
                kernel->execute(*buffer);
                double verifyOutputData[4];
                buffer->getLanes(outputNode, verifyOutputData);
                double kernelResult = verifyOutputData[0];
                double nativeResult = nativeFunc(input);
                
                // Handle inf/nan cases - if both are inf/nan, consider it verified
                bool verified = false;
                if ((std::isinf(kernelResult) && std::isinf(nativeResult)) &&
                    ((kernelResult > 0) == (nativeResult > 0))) {
                    // Both are infinity with same sign
                    verified = true;
                } else if (std::isnan(kernelResult) && std::isnan(nativeResult)) {
                    // Both are NaN
                    verified = true;
                } else {
                    // Normal comparison
                    verified = std::abs(kernelResult - nativeResult) <= config_.tolerance;
                }
                result.verificationPerInput.push_back(verified);
            }
        }
        
        results_.push_back(result);
    }
    
public:
    bool RunBenchmarks() {
        PrintResults();
        
        // Return true if all functions verified successfully
        bool allPassed = true;
        for (const auto& result : results_) {
            for (bool v : result.verificationPerInput) {
                if (!v) {
                    allPassed = false;
                    break;
                }
            }
        }
        return allPassed;
    }
    
    const std::vector<ComprehensiveBenchmarkResult>& GetResults() const { return results_; }
    
private:
    void PrintResults() {
        if (results_.empty()) {
            std::cout << "No benchmark results to display." << std::endl;
            return;
        }
        
        // Print header with separator lines
        std::cout << "\n";
        std::cout << "===========================================================================================================" << std::endl;
        std::cout << "Comprehensive Benchmark Results" << std::endl;
        std::cout << "===========================================================================================================" << std::endl;
        std::cout << "Configuration: " << config_.benchmarkIterations << " iterations, " 
                  << config_.warmupIterations << " warmup runs" << std::endl;
        std::cout << "\n";
        
        // SECTION 1: Graph Recording Information
        std::cout << "SECTION 1: GRAPH RECORDING & STRUCTURE" << std::endl;
        std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
        PrintGraphInfo();
        
        // SECTION 2: Optimization Statistics
        std::cout << "\nSECTION 2: OPTIMIZATION PASSES & STATISTICS" << std::endl;
        std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
        PrintOptimizationStats();
        
        // SECTION 3: Performance Metrics (formerly Section 1)
        std::cout << "\nSECTION 3: PERFORMANCE METRICS" << std::endl;
        std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
        
        // Determine max number of inputs for verification columns
        size_t maxInputs = maxInputCount_;
        
        // Determine the maximum function name length for better formatting
        size_t maxFunctionNameLength = 8; // minimum width for "Function"
        for (const auto& result : results_) {
            maxFunctionNameLength = std::max(maxFunctionNameLength, result.functionName.length());
        }
        
        // Print table header
        std::cout << "| " << std::left << std::setw(maxFunctionNameLength) << "Function" 
                  << " | Graph Opt(ms) | Creation: Kernel(ms) | Eval: Native(ns) | Eval: Kernel(ns) | Speedup | Nodes |";
        if (config_.verifyResults) {
            for (size_t i = 0; i < maxInputs; ++i) {
                std::cout << " V" << (i+1) << " |";
            }
        }
        std::cout << std::endl;
        
        // Print separator line with dynamic width
        std::cout << "|" << std::string(maxFunctionNameLength + 2, '-') 
                  << "|---------------|----------------------|------------------|------------------|---------|-------|";
        if (config_.verifyResults) {
            for (size_t i = 0; i < maxInputs; ++i) {
                std::cout << "----|";
            }
        }
        std::cout << std::endl;
        
        // Print results
        std::cout << std::fixed;
        for (const auto& result : results_) {
            std::cout << "| " << std::left << std::setw(maxFunctionNameLength) << result.functionName << " | "
                      << std::right << std::setw(13) << std::setprecision(3) << result.graphOptimizationTimeMs << " | "
                      << std::setw(20) << std::setprecision(3) << result.kernelCreationTimeMs << " | "
                      << std::setw(16) << std::setprecision(2) << result.nativeEvalTimeNs << " | "
                      << std::setw(16) << std::setprecision(2) << result.kernelEvalTimeNs << " | "
                      << std::setw(5) << std::setprecision(2) << result.speedup << "x | "
                      << std::setw(5) << result.graphNodes << " |";
            
            if (config_.verifyResults) {
                for (size_t i = 0; i < maxInputs; ++i) {
                    if (i < result.verificationPerInput.size()) {
                        std::cout << " " << (result.verificationPerInput[i] ? "Y" : "N") << "  |";
                    } else {
                        std::cout << " -  |";
                    }
                }
            }
            std::cout << std::endl;
        }
        
        // Add legend for verification columns
        if (config_.verifyResults && maxInputs > 0) {
            std::cout << "\nVerification columns (V1, V2, ...): Y=passed, N=failed, -=no data" << std::endl;
            std::cout << "Each function may have different test inputs:" << std::endl;
            for (const auto& result : results_) {
                std::cout << "  " << result.functionName << ": ";
                for (size_t i = 0; i < result.testInputs.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << result.testInputs[i];
                }
                std::cout << std::endl;
            }
        }
        
        // Print detailed verification section if enabled (now Section 4)
        if (config_.verifyResults && !results_.empty()) {
            std::cout << "\nSECTION 4: VERIFICATION DETAILS" << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
            PrintVerificationDetails();
        }
        
        // Print AVX2 performance comparison if tested (new Section 5)
        if (config_.testAvx2) {
            PrintAvx2Comparison();
        }
        
        // Calculate and print summary statistics (now Section 6)
        PrintSummary();
    }
    
    void PrintGraphInfo() {
        for (const auto& result : results_) {
            std::cout << "\n" << result.functionName << " - Graph Recording Details:" << std::endl;
            
            // Graph structure breakdown
            std::cout << "\nGraph Structure Breakdown:" << std::endl;
            std::cout << "| Node Type          | Count     | % of Total | Description                            |" << std::endl;
            std::cout << "|--------------------|-----------|------------|----------------------------------------|" << std::endl;
            std::cout << "| Total Nodes        | " << std::setw(9) << result.graphNodes 
                      << " |     100.0% | Complete computation graph            |" << std::endl;
            
            if (result.inputNodes > 0) {
                std::cout << "| Input Nodes        | " << std::setw(9) << result.inputNodes 
                         << " | " << std::setw(9) << std::fixed << std::setprecision(1) 
                         << (100.0 * result.inputNodes / result.graphNodes) << "% | Function parameters                   |" << std::endl;
            }
            
            if (result.constantNodes > 0) {
                std::cout << "| Constant Nodes     | " << std::setw(9) << result.constantNodes 
                         << " | " << std::setw(9) << std::fixed << std::setprecision(1)
                         << (100.0 * result.constantNodes / result.graphNodes) << "% | Compile-time constants                |" << std::endl;
            }
            
            if (result.arithmeticNodes > 0) {
                std::cout << "| Arithmetic Ops     | " << std::setw(9) << result.arithmeticNodes 
                         << " | " << std::setw(9) << std::fixed << std::setprecision(1)
                         << (100.0 * result.arithmeticNodes / result.graphNodes) << "% | +, -, *, /, abs, min, max             |" << std::endl;
            }
            
            if (result.transcendentalNodes > 0) {
                std::cout << "| Transcendental Ops | " << std::setw(9) << result.transcendentalNodes 
                         << " | " << std::setw(9) << std::fixed << std::setprecision(1)
                         << (100.0 * result.transcendentalNodes / result.graphNodes) << "% | exp, log, sin, cos, pow, sqrt         |" << std::endl;
            }
            
            if (result.comparisonNodes > 0) {
                std::cout << "| Comparison Ops     | " << std::setw(9) << result.comparisonNodes 
                         << " | " << std::setw(9) << std::fixed << std::setprecision(1)
                         << (100.0 * result.comparisonNodes / result.graphNodes) << "% | <, >, ==, !=, <=, >=                  |" << std::endl;
            }
            
            if (result.controlFlowNodes > 0) {
                std::cout << "| Control Flow       | " << std::setw(9) << result.controlFlowNodes 
                         << " | " << std::setw(9) << std::fixed << std::setprecision(1)
                         << (100.0 * result.controlFlowNodes / result.graphNodes) << "% | if-then-else conditionals             |" << std::endl;
            }
            
            // Graph recording timing
            std::cout << "\nGraph Recording Performance:" << std::endl;
            std::cout << "| Metric                  | Value         | Description                              |" << std::endl;
            std::cout << "|-------------------------|---------------|------------------------------------------|" << std::endl;
            std::cout << "| Recording Time          | " << std::setw(10) << std::fixed << std::setprecision(3) 
                      << result.graphOptimizationTimeMs << " ms | Time to record computation graph        |" << std::endl;
            std::cout << "| Recording Speed         | " << std::setw(10) << std::fixed << std::setprecision(0)
                      << (result.graphNodes * 1000.0 / std::max(0.001, result.graphOptimizationTimeMs)) 
                      << " n/s | Nodes recorded per second                |" << std::endl;
            std::cout << "| Avg Node Complexity     | " << std::setw(10) << std::fixed << std::setprecision(2)
                      << (result.graphOptimizationTimeMs * 1000000.0 / std::max(size_t(1), result.graphNodes)) 
                      << " ns | Average time per node                    |" << std::endl;
        }
    }
    
    void PrintOptimizationStats() {
        for (const auto& result : results_) {
            std::cout << "\n" << result.functionName << " - Graph Optimization Details:" << std::endl;
            std::cout << "  Optimization Passes Performed: " << result.passesPerformed 
                      << " (max " << 5 << " allowed)" << std::endl;
            
            // First show timing table
            std::cout << "\nOptimization Pass Timing (across all " << result.passesPerformed << " iterations):" << std::endl;
            std::cout << "| Optimization Pass          | Time (ms) | Nodes Changed | Effectiveness |" << std::endl;
            std::cout << "|----------------------------|-----------|---------------|---------------|" << std::endl;
            
            if (result.inactiveFoldingTimeMs > 0.0 || result.inactiveNodesFolded > 0) {
                std::cout << "| Inactive Folding           | " << std::setw(9) << std::fixed << std::setprecision(2) 
                         << result.inactiveFoldingTimeMs << " | " << std::setw(13) << result.inactiveNodesFolded 
                         << " | " << std::setw(11) << std::setprecision(1) 
                         << (result.inactiveNodesFolded > 0 ? result.inactiveNodesFolded * 1000.0 / std::max(0.01, result.inactiveFoldingTimeMs) : 0.0)
                         << " nodes/sec |" << std::endl;
            }
            
            if (result.cseTimeMs > 0.0 || result.duplicatesEliminated > 0) {
                std::cout << "| Common Subexpr. Elim.      | " << std::setw(9) << std::fixed << std::setprecision(2) 
                         << result.cseTimeMs << " | " << std::setw(13) << result.duplicatesEliminated 
                         << " | " << std::setw(11) << std::setprecision(1)
                         << (result.duplicatesEliminated > 0 ? result.duplicatesEliminated * 1000.0 / std::max(0.01, result.cseTimeMs) : 0.0)
                         << " nodes/sec |" << std::endl;
            }
            
            if (result.algebraicTimeMs > 0.0 || result.algebraicSimplifications > 0) {
                std::cout << "| Algebraic Simplification   | " << std::setw(9) << std::fixed << std::setprecision(2) 
                         << result.algebraicTimeMs << " | " << std::setw(13) << result.algebraicSimplifications 
                         << " | " << std::setw(11) << std::setprecision(1)
                         << (result.algebraicSimplifications > 0 ? result.algebraicSimplifications * 1000.0 / std::max(0.01, result.algebraicTimeMs) : 0.0)
                         << " nodes/sec |" << std::endl;
            }
            
            if (result.stabilityTimeMs > 0.0 || result.stabilityFixes > 0) {
                std::cout << "| Stability Cleaning         | " << std::setw(9) << std::fixed << std::setprecision(2) 
                         << result.stabilityTimeMs << " | " << std::setw(13) << result.stabilityFixes 
                         << " | " << std::setw(11) << std::setprecision(1)
                         << (result.stabilityFixes > 0 ? result.stabilityFixes * 1000.0 / std::max(0.01, result.stabilityTimeMs) : 0.0)
                         << " nodes/sec |" << std::endl;
            }
            
            std::cout << "|----------------------------|-----------|---------------|---------------|" << std::endl;
            std::cout << "| TOTAL OPTIMIZATION         | " << std::setw(9) << std::fixed << std::setprecision(2) 
                     << result.totalOptimizationTimeMs << " | " << std::setw(13) 
                     << (result.inactiveNodesFolded + result.duplicatesEliminated + result.algebraicSimplifications + result.stabilityFixes)
                     << " | " << std::setw(11) << std::setprecision(1)
                     << (result.originalNodeCount * 1000.0 / std::max(0.01, result.totalOptimizationTimeMs))
                     << " nodes/sec |" << std::endl;
            
            std::cout << "\nNote: Nodes are marked as 'dead' but remain in the graph structure to preserve workspace compatibility." << std::endl;
            std::cout << "      Dead nodes are skipped during JIT execution, providing the performance benefit without memory reallocation." << std::endl;
            
            // Then show impact summary table
            std::cout << "\nOptimization Impact Summary:" << std::endl;
            std::cout << "| Metric                     | Count/Value | % of Original | Description                      |" << std::endl;
            std::cout << "|----------------------------|-------------|---------------|----------------------------------|" << std::endl;
            std::cout << "| Original Node Count        | " << std::setw(11) << result.originalNodeCount 
                      << " |         100.0% | Initial computation graph        |" << std::endl;
            
            if (result.inactiveNodesFolded > 0) {
                std::cout << "| Inactive Nodes Folded      | " << std::setw(11) << result.inactiveNodesFolded 
                         << " | " << std::setw(13) << std::fixed << std::setprecision(1) 
                         << (100.0 * result.inactiveNodesFolded / result.originalNodeCount) << "% "
                         << "| Constant subgraph elimination   |" << std::endl;
            }
            
            if (result.duplicatesEliminated > 0) {
                std::cout << "| Duplicates Eliminated      | " << std::setw(11) << result.duplicatesEliminated 
                         << " | " << std::setw(13) << std::fixed << std::setprecision(1)
                         << (100.0 * result.duplicatesEliminated / result.originalNodeCount) << "% "
                         << "| Common subexpression elimination|" << std::endl;
            }
            
            if (result.algebraicSimplifications > 0) {
                std::cout << "| Algebraic Simplifications  | " << std::setw(11) << result.algebraicSimplifications 
                         << " | " << std::setw(13) << std::fixed << std::setprecision(1)
                         << (100.0 * result.algebraicSimplifications / result.originalNodeCount) << "% "
                         << "| x*1=x, x+0=x, etc.              |" << std::endl;
            }
            
            if (result.stabilityFixes > 0) {
                std::cout << "| Stability Fixes            | " << std::setw(11) << result.stabilityFixes 
                         << " | " << std::setw(13) << std::fixed << std::setprecision(1)
                         << (100.0 * result.stabilityFixes / result.originalNodeCount) << "% "
                         << "| Numerical stability improvements|" << std::endl;
            }
            
            std::cout << "| Dead Nodes Marked          | " << std::setw(11) << result.deadNodesMarked 
                      << " | " << std::setw(13) << std::fixed << std::setprecision(1)
                      << result.optimizationRatio << "% "
                      << "| Nodes marked dead (skipped)     |" << std::endl;
            
            size_t effectiveNodes = result.originalNodeCount - result.deadNodesMarked;
            std::cout << "| Active Nodes Remaining     | " << std::setw(11) << effectiveNodes 
                      << " | " << std::setw(13) << std::fixed << std::setprecision(1)
                      << (100.0 * effectiveNodes / result.originalNodeCount) << "% "
                      << "| Nodes actively computed         |" << std::endl;
            
            // Show optimization effectiveness
            std::cout << "\n  Optimization Impact: ";
            if (result.optimizationRatio > 50) {
                std::cout << "EXCELLENT - Marked " << std::setprecision(1) << result.optimizationRatio << "% of nodes as dead (will be skipped during execution)";
            } else if (result.optimizationRatio > 20) {
                std::cout << "GOOD - Marked " << std::setprecision(1) << result.optimizationRatio << "% of nodes as dead";
            } else if (result.optimizationRatio > 5) {
                std::cout << "MODERATE - Marked " << std::setprecision(1) << result.optimizationRatio << "% of nodes as dead";
            } else if (result.optimizationRatio > 0.1) {
                std::cout << "MINIMAL - Marked " << std::setprecision(1) << result.optimizationRatio << "% of nodes as dead";
            } else {
                std::cout << "NEGLIGIBLE - Marked " << std::setprecision(2) << result.optimizationRatio << "% of nodes as dead";
            }
            std::cout << std::endl;
        }
    }
    
    void PrintVerificationDetails() {
        for (const auto& result : results_) {
            std::cout << "\n" << result.functionName << " - Test Inputs & Verification:" << std::endl;
            std::cout << "| Input Value | Verification | Native Time(ns) | JIT Time(ns) | Speedup |" << std::endl;
            std::cout << "|-------------|--------------|-----------------|--------------|---------|" << std::endl;
            
            for (size_t i = 0; i < result.testInputs.size(); ++i) {
                std::cout << "| " << std::setw(11) << std::fixed << std::setprecision(2) << result.testInputs[i] << " | ";
                
                if (i < result.verificationPerInput.size()) {
                    std::cout << std::setw(12) << (result.verificationPerInput[i] ? "PASS" : "FAIL") << " | ";
                } else {
                    std::cout << std::setw(12) << "N/A" << " | ";
                }
                
                // Show per-input performance (using average for demonstration)
                std::cout << std::setw(15) << std::setprecision(2) << result.nativeEvalTimeNs << " | "
                         << std::setw(12) << std::setprecision(2) << result.kernelEvalTimeNs << " | "
                         << std::setw(6) << std::setprecision(2) << result.speedup << "x |" << std::endl;
            }
        }
    }
    
    void PrintAvx2Comparison() {
        bool anyAvx2Tested = false;
        for (const auto& result : results_) {
            if (result.avx2Tested) {
                anyAvx2Tested = true;
                break;
            }
        }
        
        if (!anyAvx2Tested) return;
        
        std::cout << "\nSECTION 5: AVX2 VECTORIZATION COMPARISON" << std::endl;
        std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Comparing performance for processing 4 values:" << std::endl;
        std::cout << "\n";
        
        // Determine max function name length
        size_t maxFunctionNameLength = 8;
        for (const auto& result : results_) {
            if (result.avx2Tested) {
                maxFunctionNameLength = std::max(maxFunctionNameLength, result.functionName.length());
            }
        }
        
        // Print detailed comparison table
        std::cout << "| " << std::left << std::setw(maxFunctionNameLength) << "Function" 
                  << " | Native 4x (ns) | SSE2 4x (ns) | AVX2 4x (ns) | AVX2 vs Native | AVX2 vs SSE2 |" << std::endl;
        std::cout << "|" << std::string(maxFunctionNameLength + 2, '-') 
                  << "|----------------|--------------|--------------|----------------|--------------|" << std::endl;
        
        for (const auto& result : results_) {
            if (result.avx2Tested) {
                double native4x = result.nativeEvalTimeNs * 4.0;
                double sse2_4x = result.kernelEvalTimeNs * 4.0;
                double avx2_4x = result.avx2TimePerEvaluation * 4.0;
                double avx2VsNative = native4x / avx2_4x;
                double avx2VsSSE2 = sse2_4x / avx2_4x;
                
                std::cout << "| " << std::left << std::setw(maxFunctionNameLength) << result.functionName
                          << " | " << std::right << std::setw(14) << std::fixed << std::setprecision(2) 
                          << native4x
                          << " | " << std::setw(12) << sse2_4x
                          << " | " << std::setw(12) << avx2_4x
                          << " | " << std::setw(13) << std::setprecision(2) << avx2VsNative << "x"
                          << " | " << std::setw(11) << std::setprecision(2) << avx2VsSSE2 << "x |" << std::endl;
            }
        }
        
        std::cout << "\nPer-value timing comparison:" << std::endl;
        std::cout << "| " << std::left << std::setw(maxFunctionNameLength) << "Function" 
                  << " | Native (ns) | SSE2 (ns) | AVX2 (ns/val) | Speedup |" << std::endl;
        std::cout << "|" << std::string(maxFunctionNameLength + 2, '-') 
                  << "|-------------|-----------|---------------|---------|" << std::endl;
        
        for (const auto& result : results_) {
            if (result.avx2Tested) {
                std::cout << "| " << std::left << std::setw(maxFunctionNameLength) << result.functionName
                          << " | " << std::right << std::setw(11) << std::fixed << std::setprecision(2) 
                          << result.nativeEvalTimeNs
                          << " | " << std::setw(9) << result.kernelEvalTimeNs
                          << " | " << std::setw(13) << result.avx2TimePerEvaluation
                          << " | " << std::setw(6) << std::setprecision(2) << result.avx2RelativeSpeedup << "x |" << std::endl;
            }
        }
        
        std::cout << "\nNote: AVX2 processes 4 values in parallel using SIMD instructions." << std::endl;
        std::cout << "      Best for batch processing where multiple evaluations are needed." << std::endl;
    }
    
    void PrintSummary() {
        if (results_.empty()) return;
        
        std::cout << "\n";
        std::cout << "SECTION 6: SUMMARY STATISTICS" << std::endl;
        std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
        
        // Calculate statistics
        double avgSpeedup = 0.0;
        double avgGraphTime = 0.0;
        double avgKernelTime = 0.0;
        double totalGraphNodes = 0;
        double maxSpeedup = 0.0;
        double minSpeedup = std::numeric_limits<double>::max();
        std::string bestFunc, worstFunc;
        
        for (const auto& result : results_) {
            avgSpeedup += result.speedup;
            avgGraphTime += result.graphOptimizationTimeMs;
            avgKernelTime += result.kernelCreationTimeMs;
            totalGraphNodes += result.graphNodes;
            
            if (result.speedup > maxSpeedup) {
                maxSpeedup = result.speedup;
                bestFunc = result.functionName;
            }
            if (result.speedup < minSpeedup) {
                minSpeedup = result.speedup;
                worstFunc = result.functionName;
            }
        }
        
        avgSpeedup /= results_.size();
        avgGraphTime /= results_.size();
        avgKernelTime /= results_.size();
        double avgNodes = totalGraphNodes / results_.size();
        
        // Print detailed statistics
        std::cout << "\nPerformance Analysis:" << std::endl;
        std::cout << "  • Average speedup: " << std::setprecision(2) << avgSpeedup << "x" << std::endl;
        std::cout << "  • Best speedup: " << maxSpeedup << "x (" << bestFunc << ")" << std::endl;
        std::cout << "  • Worst speedup: " << minSpeedup << "x (" << worstFunc << ")" << std::endl;
        std::cout << "\nCompilation Statistics:" << std::endl;
        std::cout << "  • Avg graph optimization: " << std::setprecision(3) << avgGraphTime << " ms" << std::endl;
        std::cout << "  • Avg kernel generation: " << avgKernelTime << " ms" << std::endl;
        std::cout << "  • Avg graph size: " << std::setprecision(0) << avgNodes << " nodes" << std::endl;
        
        std::cout << "\nVERDICT: ";
        if (avgSpeedup > 1.0) {
            std::cout << "[PERFORMANCE GAIN] JIT compilation provides " << std::setprecision(1) << avgSpeedup << "x average speedup" << std::endl;
        } else {
            std::cout << "[PERFORMANCE LOSS] Native execution is " << std::setprecision(1) << (1.0/avgSpeedup) << "x faster on average" << std::endl;
        }
        
        // Check if all verifications passed
        bool allPassed = true;
        for (const auto& result : results_) {
            for (bool v : result.verificationPerInput) {
                if (!v) {
                    allPassed = false;
                    break;
                }
            }
        }
        
        if (config_.verifyResults) {
            if (allPassed) {
                std::cout << "  * All verification tests: PASSED ✓" << std::endl;
            } else {
                std::cout << "  * Verification tests: FAILED ✗" << std::endl;
            }
        }
        
        std::cout << "===========================================================================================================" << std::endl;
    }
};

// Helper function to create benchmark runner
inline auto makeBenchmarkRunner() {
    return BenchmarkRunner();
}

} // namespace tools
} // namespace forge