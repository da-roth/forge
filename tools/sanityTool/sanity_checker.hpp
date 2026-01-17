#pragma once

#include <functional>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <chrono>
#include <limits>
#include <type_traits>
#include <native/fdouble.hpp>
#include <native/fbool.hpp>
#include "../../src/graph/graph_recorder.hpp"
#include "../../src/compiler/forge_engine.hpp"
#include "../../src/compiler/interfaces/node_value_buffer.hpp"
#include "../../src/compiler/compiler_config.hpp"
#include "../../src/compiler/x86/double/avx2/avx2_instruction_set.hpp"

namespace forge {
namespace tools {

// Helper trait to detect return type and convert to double if needed
template<typename T>
struct ConvertToDouble {
    static double convert(T value) {
        if constexpr (std::is_same_v<T, bool>) {
            return value ? 1.0 : 0.0;
        } else {
            return static_cast<double>(value);
        }
    }
};

// Helper trait to handle marking output for different types
template<typename T>
struct OutputMarker {
    static void markOutput(T& value) {
        value.markOutput();
    }
    
    static forge::NodeId getNodeId(T& value) {
        return value.node();
    }
};

// Specialization for fbool - it doesn't have markOutput()
template<>
struct OutputMarker<forge::fbool> {
    static void markOutput(forge::fbool& value) {
        // fbool doesn't have markOutput(), so we need to access the recorder directly
        auto* recorder = forge::GraphRecorder::active();
        if (!recorder) {
            throw std::runtime_error("No active tape recorder");
        }
        
        // Get the node ID from fbool
        auto nodeId = value.nodeId();
        if (nodeId == static_cast<forge::NodeId>(-1)) {
            throw std::runtime_error("fbool has no associated node");
        }
        
        // Mark it as output in the tape
        recorder->graph().markOutput(nodeId);
    }
    
    static forge::NodeId getNodeId(forge::fbool& value) {
        return value.nodeId();
    }
};

// Result structure for a single test
struct SanityTestResult {
    double input;
    double nativeResult;
    double tapeResult;
    double absoluteError;
    double relativeError;
    bool passed;
    double nativeTimeUs;  // microseconds
    double tapeTimeUs;    // microseconds
    
};

// Configuration for the sanity checker
struct SanityConfig {
    double absoluteTolerance = 1e-10;
    double relativeTolerance = 1e-10;
    bool verbose = true;
    bool showTimings = true;
    bool stopOnFirstFailure = false;
    bool showOnlyFailures = false;           // Only show failed test entries in tables
    int warmupIterations = 0;     // Warmup iterations for timing
    int timingIterations = 1;    // Iterations for timing measurement
    CompilerConfig::InstructionSet instructionSet = CompilerConfig::InstructionSet::SSE2_SCALAR;  // Backend selection
};

template<typename FuncDouble, typename FuncTP>
class SanityChecker {
private:
    FuncDouble funcDouble_;
    FuncTP funcTP_;
    std::vector<double> testInputs_;
    SanityConfig config_;
    std::vector<SanityTestResult> results_;
    std::string functionName_;
    
public:
    // Constructor with separate functions for double and fdouble
    SanityChecker(const std::string& name, FuncDouble funcDouble, FuncTP funcTP,
                  const std::vector<double>& inputs, const SanityConfig& config = SanityConfig())
        : functionName_(name), funcDouble_(funcDouble), funcTP_(funcTP), 
          testInputs_(inputs), config_(config) {}
    
    // Run all tests and return overall pass/fail
    bool RunTests() {
        using namespace std::chrono;
        
        results_.clear();
        bool allPassed = true;
        
        std::cout << "\n=== Sanity Check: " << functionName_ << " ===" << std::endl;
        std::cout << "Testing " << testInputs_.size() << " input values" << std::endl;
        
        // Show which kernels will be tested
        std::string kernelName = (config_.instructionSet == CompilerConfig::InstructionSet::AVX2_PACKED) 
                                 ? "AVX2_PACKED" : "SSE2_SCALAR";
        std::cout << "Kernels: " << kernelName << std::endl;
        
        if (config_.verbose) {
            std::cout << std::fixed << std::setprecision(12);
            std::cout << "\n" << std::setw(15) << "Input" 
                      << std::setw(20) << "Native Result" 
                      << std::setw(20) << kernelName + " Result"
                      << std::setw(15) << kernelName + " Error"
                      << std::setw(15) << "Rel Error";
            
            
            if (config_.showTimings) {
                std::cout << std::setw(12) << "Native(µs)"
                          << std::setw(12) << kernelName + "(µs)";
            }
            std::cout << std::setw(10) << "Status" << std::endl;
            
            int lineWidth = 140;
            std::cout << std::string(lineWidth, '-') << std::endl;
        }
        
        for (double input : testInputs_) {
            SanityTestResult result;
            result.input = input;
            
            // ===== Native double evaluation =====
            // Warmup
            for (int i = 0; i < config_.warmupIterations; ++i) {
                volatile double dummy = funcDouble_(input);
                (void)dummy;
            }
            
            // Timed evaluation
            auto nativeStart = high_resolution_clock::now();
            for (int i = 0; i < config_.timingIterations; ++i) {
                auto rawResult = funcDouble_(input);
                result.nativeResult = ConvertToDouble<decltype(rawResult)>::convert(rawResult);
            }
            auto nativeEnd = high_resolution_clock::now();
            result.nativeTimeUs = duration<double, std::micro>(nativeEnd - nativeStart).count() 
                                  / config_.timingIterations;
            
            // ===== Graph-based evaluation =====
            // Record the tape
            forge::GraphRecorder tapeRecorder;
            tapeRecorder.start();
            
            // Create input with proper marking
            forge::fdouble x_tp(0.0);
            x_tp.markInput();  // Mark as input (no diff needed for forward eval)

            // Apply the function
            auto y_tp = funcTP_(x_tp);
            OutputMarker<decltype(y_tp)>::markOutput(y_tp);
            
            // Stop recording and get the tape
            tapeRecorder.stop();
            forge::Graph tapeGraph = tapeRecorder.graph();
            
            // Compile the tape with the configured instruction set
            forge::CompilerConfig compilerConfig;
            compilerConfig.instructionSet = config_.instructionSet;
            forge::ForgeEngine compiler(compilerConfig);
            auto kernel = compiler.compile(tapeGraph);
            
            // Create NodeValueBuffer - automatically uses correct memory layout
            auto buffer = forge::NodeValueBufferFactory::create(tapeGraph, *kernel);
            std::cout << "[Sanity] Created NodeValueBuffer, vector width: " << buffer->getVectorWidth() << std::endl;
            
            
            // Find input and output nodes
            forge::NodeId inputNode = 0;  // First node should be input
            forge::NodeId outputNode = tapeGraph.outputs[0];
            std::cout << "[Sanity] Input node: " << inputNode << ", Output node: " << outputNode << std::endl;
            
            // For AVX2, print legend for runtime value logging
            if (config_.instructionSet == CompilerConfig::InstructionSet::AVX2_PACKED && config_.verbose) {
                std::cout << "\n=== AVX2 Runtime Value Log ===\n"
                          << "Format: [OP] = lane0, lane1, lane2, lane3\n"
                          << "  LD n#->y#  = Load from node to YMM register\n"
                          << "  ST y#->n#  = Store from YMM register to node\n"
                          << "  MIN/MAX.pre/post = Before/after min/max operation\n"
                          << "  TAN.in/out = Tangent input/output\n"
                          << "==============================\n";
            }
            
            // Warmup with special logging for first execution
            for (int i = 0; i < config_.warmupIterations; ++i) {
                // For AVX2, set different values in each lane for better debugging
                if (config_.instructionSet == CompilerConfig::InstructionSet::AVX2_PACKED) {
                    double laneValues[4] = {
                        input,
                        input + 0.001,
                        input + 0.002,
                        input + 0.003
                    };
                    buffer->setLanes(inputNode, laneValues);
                } else {
                    double laneValue[1] = {input};
                    buffer->setLanes(inputNode, laneValue);
                }
                kernel->execute(*buffer);
                double outputData[4];
                buffer->getLanes(outputNode, outputData);
                volatile double dummy = outputData[0];
                (void)dummy;

                // After first execution, print buffer state for debugging
                if (i == 0 && config_.instructionSet == CompilerConfig::InstructionSet::AVX2_PACKED &&
                    config_.verbose && results_.size() < 1) {
                    std::cout << "\n=== Buffer State After First Execution ===\n";
                    std::cout << "Graph has " << tapeGraph.nodes.size() << " nodes\n";
                    for (uint64_t nodeId = 0; nodeId < buffer->getNumNodes() && nodeId < 10; nodeId++) {
                        double values[4];
                        buffer->getLanes(nodeId, values);
                        std::cout << "Node " << nodeId << " (";
                        // Print node type if available
                        if (nodeId < tapeGraph.nodes.size()) {
                            auto opCode = tapeGraph.nodes[nodeId].op;
                            switch(opCode) {
                                case forge::OpCode::Input: std::cout << "Input"; break;
                                case forge::OpCode::Constant: std::cout << "Constant"; break;
                                case forge::OpCode::Add: std::cout << "Add"; break;
                                case forge::OpCode::Sub: std::cout << "Sub"; break;
                                case forge::OpCode::Mul: std::cout << "Mul"; break;
                                case forge::OpCode::Div: std::cout << "Div"; break;
                                case forge::OpCode::Neg: std::cout << "Neg"; break;
                                case forge::OpCode::Max: std::cout << "Max"; break;
                                case forge::OpCode::Min: std::cout << "Min"; break;
                                case forge::OpCode::Tan: std::cout << "Tan"; break;
                                default: std::cout << "Op" << static_cast<int>(opCode); break;
                            }
                            // Print constant value if it's a constant node
                            if (opCode == forge::OpCode::Constant &&
                                nodeId < tapeGraph.nodes.size()) {
                                std::cout << " = " << tapeGraph.nodes[nodeId].imm;
                            }
                        }
                        std::cout << "): ";
                        int width = buffer->getVectorWidth();
                        for (int v = 0; v < width; v++) {
                            std::cout << std::fixed << std::setprecision(3) << values[v] << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << "==========================================\n\n";
                }
            }
            
            // Timed evaluation
            auto tapeStart = high_resolution_clock::now();
            for (int i = 0; i < config_.timingIterations; ++i) {
                // For AVX2, set different values in each lane for better debugging
                if (config_.instructionSet == CompilerConfig::InstructionSet::AVX2_PACKED) {
                    double laneValues[4] = {
                        input,
                        input + 0.001,
                        input + 0.002,
                        input + 0.003
                    };
                    buffer->setLanes(inputNode, laneValues);
                } else {
                    double laneValue[1] = {input};
                    buffer->setLanes(inputNode, laneValue);
                }
                kernel->execute(*buffer);
                double outputData[4];
                buffer->getLanes(outputNode, outputData);
                result.tapeResult = outputData[0];
            }
            auto tapeEnd = high_resolution_clock::now();
            result.tapeTimeUs = duration<double, std::micro>(tapeEnd - tapeStart).count()
                               / config_.timingIterations;


            // ===== Compare results =====
            // For AVX2, we need to check all lanes
            if (config_.instructionSet == CompilerConfig::InstructionSet::AVX2_PACKED) {
                double outputLanes[4];
                buffer->getLanes(outputNode, outputLanes);

                // Compare each lane with native result for the corresponding input
                double laneInputs[4] = {input, input + 0.001, input + 0.002, input + 0.003};
                bool allLanesPassed = true;
                double maxError = 0.0;
                for (int i = 0; i < 4; ++i) {
                    auto nativeRaw = funcDouble_(laneInputs[i]);
                    double nativeForLane = ConvertToDouble<decltype(nativeRaw)>::convert(nativeRaw);
                    double laneError = std::abs(outputLanes[i] - nativeForLane);
                    maxError = std::max(maxError, laneError);
                    if (laneError > config_.absoluteTolerance) {
                        allLanesPassed = false;
                    }
                }
            }
            
            // Special handling for infinity and NaN
            if (std::isinf(result.nativeResult) && std::isinf(result.tapeResult)) {
                // Both are infinity - check they have the same sign
                if ((result.nativeResult > 0) == (result.tapeResult > 0)) {
                    result.absoluteError = 0.0;
                    result.relativeError = 0.0;
                    result.passed = true;
                } else {
                    // Different signs of infinity
                    result.absoluteError = std::numeric_limits<double>::infinity();
                    result.relativeError = std::numeric_limits<double>::infinity();
                    result.passed = false;
                }
            } else if (std::isnan(result.nativeResult) && std::isnan(result.tapeResult)) {
                // Both are NaN - consider this a pass
                result.absoluteError = 0.0;
                result.relativeError = 0.0;
                result.passed = true;
            } else {
                // Normal comparison
                result.absoluteError = std::abs(result.tapeResult - result.nativeResult);
                
                // Relative error calculation (handle near-zero values)
                if (std::abs(result.nativeResult) > 1e-15) {
                    result.relativeError = result.absoluteError / std::abs(result.nativeResult);
                } else {
                    result.relativeError = result.absoluteError;  // Use absolute error when native is near zero
                }
                
                // Check if test passed
                result.passed = (result.absoluteError <= config_.absoluteTolerance) ||
                               (result.relativeError <= config_.relativeTolerance);
            }
            
            
            // Check overall test result
            if (!result.passed) {
                allPassed = false;
            }
            
            results_.push_back(result);
            
            // Print result based on showOnlyFailures setting
            if (config_.verbose && (!config_.showOnlyFailures || !result.passed)) {
                // Add special marker for debugging AVX2 identity function
                if (config_.instructionSet == CompilerConfig::InstructionSet::AVX2_PACKED &&
                    functionName_.find("Tangent") != std::string::npos) {
                    std::cout << " [DEBUG] ";
                }
                std::cout << std::setw(15) << result.input
                          << std::setw(20) << result.nativeResult
                          << std::setw(20) << result.tapeResult
                          << std::setw(15) << std::scientific << result.absoluteError
                          << std::setw(15) << result.relativeError << std::fixed;
                
                
                if (config_.showTimings) {
                    std::cout << std::setw(12) << std::setprecision(3) << result.nativeTimeUs
                              << std::setw(12) << result.tapeTimeUs;
                }
                
                // Status column
                std::string status = result.passed ? "PASS" : "FAIL";
                std::cout << std::setw(10) << status << std::endl;
            }
            
            // Stop if configured to stop on first failure
            if (!result.passed && config_.stopOnFirstFailure) {
                std::cout << "\nStopping on first failure." << std::endl;
                break;
            }
        }
        
        // Show which kernels were actually available/tested
        std::cout << "\nKernel Status: " << kernelName << " [OK]" << std::endl;
        
        
        // Print summary
        PrintSummary(allPassed);
        
        return allPassed;
    }
    
    // Get detailed results
    const std::vector<SanityTestResult>& GetResults() const { return results_; }
    
    // Get config (for modification)
    SanityConfig& Config() { return config_; }
    
private:
    void PrintSummary(bool allPassed) {
        int passedCount = 0;
        int failedCount = 0;
        double maxAbsError = 0.0;
        double maxRelError = 0.0;
        double avgSpeedup = 0.0;
        
        for (const auto& result : results_) {
            if (result.passed) {
                passedCount++;
            } else {
                failedCount++;
            }
            maxAbsError = std::max(maxAbsError, result.absoluteError);
            maxRelError = std::max(maxRelError, result.relativeError);
            if (config_.showTimings && result.tapeTimeUs > 0) {
                avgSpeedup += result.nativeTimeUs / result.tapeTimeUs;
            }
        }
        
        if (config_.showTimings && !results_.empty()) {
            avgSpeedup /= results_.size();
        }
        
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Total tests: " << results_.size() << std::endl;
        std::cout << "Passed: " << passedCount << std::endl;
        std::cout << "Failed: " << failedCount << std::endl;
        std::cout << "Max absolute error: " << std::scientific << maxAbsError << std::endl;
        std::cout << "Max relative error: " << maxRelError << std::endl;
        
        if (config_.showTimings && !results_.empty()) {
            std::cout << "Average speedup: " << std::fixed << std::setprecision(2) 
                      << avgSpeedup << "x" << std::endl;
        }
        
        if (allPassed) {
            std::cout << "\n[PASS] All tests PASSED" << std::endl;
        } else {
            std::cout << "\n[FAIL] Some tests FAILED" << std::endl;
        }
    }
};

// Helper function to create sanity checker with type deduction
template<typename FuncDouble, typename FuncTP>
auto makeSanityChecker(const std::string& name, FuncDouble funcDouble, FuncTP funcTP,
                       const std::vector<double>& inputs,
                       const SanityConfig& config = SanityConfig()) {
    return SanityChecker<FuncDouble, FuncTP>(name, funcDouble, funcTP, inputs, config);
}

} // namespace tools
} // namespace forge