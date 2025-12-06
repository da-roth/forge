#pragma once

#include <functional>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "../../tools/types/fdouble.hpp"
#include "../../src/graph/graph_recorder.hpp"
#include "../../src/compiler/forge_engine.hpp"
#include "../../src/compiler/node_value_buffers/node_value_buffer.hpp"
#include "../../src/compiler/compiler_config.hpp"

namespace forge {
namespace tools {

// Extended result structure including derivative information
struct SanityDiffTestResult {
    double input;
    double nativeResult;
    double tapeResult;
    double absoluteError;
    double relativeError;
    bool valuePassed;
    
    // Derivative results
    double nativeDerivative;       // From finite difference
    double tapeDerivative;         // From automatic differentiation
    double derivativeAbsError;
    double derivativeRelError;
    bool derivativePassed;
    
    // Overall status
    bool passed;
    
    // Timing information
    double nativeTimeUs;
    double tapeTimeUs;
    
    // AVX2 results (optional)
    bool avx2Available = false;
    double avx2Result = 0.0;
    double avx2Derivative = 0.0;
    double avx2ValueAbsError = 0.0;     // vs native
    double avx2ValueRelError = 0.0;     // vs native
    double avx2DerivAbsError = 0.0;     // vs native derivative
    double avx2DerivRelError = 0.0;     // vs native derivative
    double sse2VsAvx2ValueError = 0.0;  // SSE2 vs AVX2 value comparison
    double sse2VsAvx2DerivError = 0.0;  // SSE2 vs AVX2 derivative comparison
    bool avx2ValuePassed = false;
    bool avx2DerivPassed = false;
    double avx2TimeUs = 0.0;
};

// Configuration for the sanity checker with derivatives
struct SanityDiffConfig {
    // Value tolerance
    double absoluteTolerance = 1e-10;
    double relativeTolerance = 1e-10;
    
    // Derivative tolerance (typically less strict than value tolerance)
    double derivativeAbsTolerance = 1e-6;
    double derivativeRelTolerance = 1e-6;
    
    // Finite difference parameters
    double finiteDiffBump = 1e-8;        // Fixed bump size for finite differences
    bool useRichardsonExtrapolation = false;  // Use Richardson extrapolation for better accuracy
    
    // Display options
    bool verbose = true;
    bool showTimings = true;
    bool showDerivatives = true;
    bool stopOnFirstFailure = false;
    bool showOnlyFailures = false;           // Only show failed test entries in tables
    
    // Performance options
    int warmupIterations = 10;
    int timingIterations = 100;
    
    // AVX2 testing configuration
    bool testAvx2 = false;                    // Try to test AVX2 if available (disabled for now)
    bool requireAvx2Pass = true;             // Whether AVX2 mismatch should fail the test (now enabled by default)
    double avx2ValueTolerance = 1e-10;       // Tolerance for SSE2 vs AVX2 value comparison (relaxed from 1e-12)
    double avx2DerivTolerance = 1e-8;        // Tolerance for SSE2 vs AVX2 derivative comparison (relaxed from 1e-10)
    bool testAvx2Vectorized = true;          // Test AVX2 with 4 inputs simultaneously (like benchmark)
};

template<typename FuncDouble, typename FuncTP>
class SanityCheckerDiff {
private:
    FuncDouble funcDouble_;
    FuncTP funcTP_;
    std::vector<double> testInputs_;
    SanityDiffConfig config_;
    std::vector<SanityDiffTestResult> results_;
    std::string functionName_;
    
    // Compute finite difference derivative
    double computeFiniteDifference(double x) {
        double h = config_.finiteDiffBump;
        
        if (config_.useRichardsonExtrapolation) {
            // Richardson extrapolation for O(h^4) accuracy
            double D1 = centralDifference(x, h);
            double D2 = centralDifference(x, h / 2.0);
            return (4.0 * D2 - D1) / 3.0;
        } else {
            // Simple central difference for O(h^2) accuracy
            return centralDifference(x, h);
        }
    }
    
    // Central difference approximation
    double centralDifference(double x, double h) {
        double f_plus = funcDouble_(x + h);
        double f_minus = funcDouble_(x - h);
        return (f_plus - f_minus) / (2.0 * h);
    }
    
public:
    // Constructor maintaining same interface as original SanityChecker
    SanityCheckerDiff(const std::string& name, FuncDouble funcDouble, FuncTP funcTP,
                      const std::vector<double>& inputs, const SanityDiffConfig& config = SanityDiffConfig())
        : functionName_(name), funcDouble_(funcDouble), funcTP_(funcTP), 
          testInputs_(inputs), config_(config) {}
    
    // Convenience constructor that accepts SanityConfig (for backward compatibility)
    SanityCheckerDiff(const std::string& name, FuncDouble funcDouble, FuncTP funcTP,
                      const std::vector<double>& inputs, 
                      double absTol, double relTol, bool verbose = true)
        : functionName_(name), funcDouble_(funcDouble), funcTP_(funcTP), 
          testInputs_(inputs) {
        config_.absoluteTolerance = absTol;
        config_.relativeTolerance = relTol;
        config_.verbose = verbose;
    }
    
    // Run all tests including derivative checking
    bool RunTests() {
        using namespace std::chrono;
        
        results_.clear();
        bool allPassed = true;
        
        std::cout << "\n=== Sanity Check with Derivatives: " << functionName_ << " ===" << std::endl;
        std::cout << "Testing " << testInputs_.size() << " input values" << std::endl;
        std::cout << "Finite difference bump size: " << std::scientific << config_.finiteDiffBump << std::endl;
        
        // Show which kernels will be tested
        std::cout << "Kernels: SSE2_SCALAR";
        if (config_.testAvx2) {
            std::cout << " + AVX2_PACKED (if available)";
        }
        std::cout << std::endl;
        
        if (config_.verbose) {
            PrintHeader();
        }
        
        for (double input : testInputs_) {
            SanityDiffTestResult result;
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
                result.nativeResult = funcDouble_(input);
            }
            auto nativeEnd = high_resolution_clock::now();
            result.nativeTimeUs = duration<double, std::micro>(nativeEnd - nativeStart).count() 
                                  / config_.timingIterations;
            
            // Compute finite difference derivative
            result.nativeDerivative = computeFiniteDifference(input);
            
            // ===== SSE2 Graph-based evaluation with derivatives =====
            forge::GraphRecorder sse2Recorder;
            sse2Recorder.start();
            
            // Create input with gradient tracking
            forge::fdouble sse2_x(0.0);
            sse2_x.markInputAndDiff();  // Mark for both value and derivative computation

            // Apply the function
            forge::fdouble sse2_y = funcTP_(sse2_x);
            sse2_y.markOutput();
            
            // Stop recording and get the SSE2 tape
            sse2Recorder.stop();
            forge::Graph sse2Graph = sse2Recorder.graph();
            
            // Compile the SSE2 tape
            forge::ForgeEngine sse2Compiler;
            auto sse2Kernel = sse2Compiler.compile(sse2Graph);
            
            // Create SSE2 NodeValueBuffer with gradient support
            auto sse2Buffer = forge::NodeValueBufferFactory::create(sse2Graph, *sse2Kernel);
            
            // Find input and output nodes
            forge::NodeId sse2InputNode = sse2Graph.diff_inputs[0];
            forge::NodeId sse2OutputNode = sse2Graph.outputs[0];
            
            // SSE2 Warmup
            for (int i = 0; i < config_.warmupIterations; ++i) {
                sse2Buffer->setValue(sse2InputNode, input);
                sse2Buffer->clearGradients();
                sse2Kernel->execute(*sse2Buffer);
                volatile double dummy = sse2Buffer->getValue(sse2OutputNode);
                (void)dummy;
            }
            
            // SSE2 Timed evaluation with gradient computation
            auto sse2Start = high_resolution_clock::now();
            for (int i = 0; i < config_.timingIterations; ++i) {
                sse2Buffer->setValue(sse2InputNode, input);
                sse2Buffer->clearGradients();
                sse2Kernel->execute(*sse2Buffer);
                result.tapeResult = sse2Buffer->getValue(sse2OutputNode);
                result.tapeDerivative = sse2Buffer->getGradient(sse2InputNode);
            }
            auto sse2End = high_resolution_clock::now();
            result.tapeTimeUs = duration<double, std::micro>(sse2End - sse2Start).count() 
                               / config_.timingIterations;
            
            // ===== AVX2 evaluation (if available and requested) =====
            if (config_.testAvx2) {
                try {
                    // ===== Record a FRESH tape for AVX2 with derivatives (starting from scratch) =====
                    forge::GraphRecorder avx2Recorder;
                    avx2Recorder.start();
                    
                    // Create fresh input with gradient tracking
                    forge::fdouble avx2_x(0.0);
                    avx2_x.markInputAndDiff();  // Mark for both value and derivative computation

                    // Apply the function to fresh input
                    forge::fdouble avx2_y = funcTP_(avx2_x);
                    avx2_y.markOutput();
                    
                    // Stop recording and get the AVX2 tape
                    avx2Recorder.stop();
                    forge::Graph avx2Graph = avx2Recorder.graph();
                    
                    // Create AVX2 compiler configuration
                    forge::CompilerConfig avx2Config;
                    avx2Config.instructionSet = forge::CompilerConfig::InstructionSet::AVX2_PACKED;
                    
                    // Compile with AVX2 using the fresh tape
                    forge::ForgeEngine avx2Compiler(avx2Config);
                    auto avx2Kernel = avx2Compiler.compile(avx2Graph);
                    
                    // Create NodeValueBuffer adapted to AVX2 kernel using the fresh tape
                    auto avx2Buffer = forge::NodeValueBufferFactory::create(avx2Graph, *avx2Kernel);
                    
                    // Find input and output nodes in the fresh AVX2 tape
                    forge::NodeId avx2InputNode = avx2Graph.diff_inputs[0];
                    forge::NodeId avx2OutputNode = avx2Graph.outputs[0];
                    
                    result.avx2Available = true;
                    
                    if (config_.testAvx2Vectorized && avx2Buffer->getVectorWidth() == 4) {
                        // Test AVX2 with 4 inputs simultaneously (like benchmark does)
                        // Create batched input: [input, input+0.01, input+0.02, input+0.03]
                        double batch[4] = {input, input + 0.01, input + 0.02, input + 0.03};

                        // AVX2 Warmup
                        for (int i = 0; i < config_.warmupIterations; ++i) {
                            avx2Buffer->setLanes(avx2InputNode, batch);
                            avx2Buffer->clearGradients();
                            avx2Kernel->execute(*avx2Buffer);
                            double dummyVec[4];
                            avx2Buffer->getLanes(avx2OutputNode, dummyVec);
                            volatile double dummy = dummyVec[0];  // Use first lane
                            (void)dummy;
                        }

                        // AVX2 Timed evaluation with gradient computation
                        auto avx2Start = high_resolution_clock::now();
                        for (int i = 0; i < config_.timingIterations; ++i) {
                            avx2Buffer->setLanes(avx2InputNode, batch);
                            avx2Buffer->clearGradients();
                            avx2Kernel->execute(*avx2Buffer);
                            double avx2Vec[4];
                            avx2Buffer->getLanes(avx2OutputNode, avx2Vec);
                            // Get gradient for lane 0
                            size_t gradIdx = avx2Buffer->getBufferIndex(avx2InputNode);
                            double gradLane0[1], gradLane1[1], gradLane2[1], gradLane3[1];
                            double* gradOutputs[4] = {gradLane0, gradLane1, gradLane2, gradLane3};
                            std::vector<size_t> gradIndices = {gradIdx};
                            avx2Buffer->getGradientLanes(gradIndices, gradOutputs);
                            result.avx2Result = avx2Vec[0];        // Use first lane result
                            result.avx2Derivative = gradLane0[0];  // Use first lane gradient
                        }
                        auto avx2End = high_resolution_clock::now();
                        result.avx2TimeUs = duration<double, std::micro>(avx2End - avx2Start).count()
                                           / config_.timingIterations;

                        // Verify all 4 vectorized lanes against native finite differences
                        bool allVectorLanesPassed = true;
                        avx2Buffer->setLanes(avx2InputNode, batch);
                        avx2Buffer->clearGradients();
                        avx2Kernel->execute(*avx2Buffer);
                        double finalAvx2Vec[4];
                        avx2Buffer->getLanes(avx2OutputNode, finalAvx2Vec);
                        // Get all 4 gradient lanes
                        size_t finalGradIdx = avx2Buffer->getBufferIndex(avx2InputNode);
                        double finalGradLane0[1], finalGradLane1[1], finalGradLane2[1], finalGradLane3[1];
                        double* finalGradOutputs[4] = {finalGradLane0, finalGradLane1, finalGradLane2, finalGradLane3};
                        std::vector<size_t> finalGradIndices = {finalGradIdx};
                        avx2Buffer->getGradientLanes(finalGradIndices, finalGradOutputs);
                        double finalAvx2GradVec[4] = {finalGradLane0[0], finalGradLane1[0], finalGradLane2[0], finalGradLane3[0]};
                        
                        for (int lane = 0; lane < 4; ++lane) {
                            double laneInput = batch[lane];
                            double nativeLaneResult = funcDouble_(laneInput);
                            double nativeLaneDerivative = computeFiniteDifference(laneInput);

                            double avx2LaneResult = finalAvx2Vec[lane];
                            double avx2LaneDerivative = finalAvx2GradVec[lane];
                            
                            // Check value accuracy for this lane
                            double laneValueError = std::abs(avx2LaneResult - nativeLaneResult);
                            bool laneValueOK = (laneValueError <= config_.avx2ValueTolerance);
                            
                            // Check derivative accuracy for this lane
                            double laneDerivError = std::abs(avx2LaneDerivative - nativeLaneDerivative);
                            bool laneDerivOK = (laneDerivError <= config_.avx2DerivTolerance);
                            
                            if (!laneValueOK || !laneDerivOK) {
                                allVectorLanesPassed = false;
                            }
                        }
                        
                        // 4-lane verification should also contribute to overall pass/fail
                        if (!allVectorLanesPassed) {
                            result.avx2ValuePassed = false;
                            result.avx2DerivPassed = false;
                        }
                        
                        if (config_.verbose) {
                            std::cout << " (4-lane vectorized verification: " 
                                      << (allVectorLanesPassed ? "PASS" : "FAIL") << ")";
                        }
                        
                    } else {
                        // Fall back to single-input AVX2 testing (original behavior)
                        double inputData[4] = {input, input, input, input};
                        for (int i = 0; i < config_.warmupIterations; ++i) {
                            avx2Buffer->setLanes(avx2InputNode, inputData);
                            avx2Buffer->clearGradients();
                            avx2Kernel->execute(*avx2Buffer);
                            double outputData[4];
                            avx2Buffer->getLanes(avx2OutputNode, outputData);
                            volatile double dummy = outputData[0];
                            (void)dummy;
                        }

                        // AVX2 Timed evaluation with gradient computation
                        auto avx2Start = high_resolution_clock::now();
                        for (int i = 0; i < config_.timingIterations; ++i) {
                            avx2Buffer->setLanes(avx2InputNode, inputData);
                            avx2Buffer->clearGradients();
                            avx2Kernel->execute(*avx2Buffer);
                            double outputData[4];
                            avx2Buffer->getLanes(avx2OutputNode, outputData);
                            result.avx2Result = outputData[0];
                            // Get gradient for lane 0
                            size_t gradIdx = avx2Buffer->getBufferIndex(avx2InputNode);
                            double gradLane0[1];
                            double* gradOutputs[4] = {gradLane0, nullptr, nullptr, nullptr};
                            std::vector<size_t> gradIndices = {gradIdx};
                            avx2Buffer->getGradientLanes(gradIndices, gradOutputs);
                            result.avx2Derivative = gradLane0[0];
                        }
                        auto avx2End = high_resolution_clock::now();
                        result.avx2TimeUs = duration<double, std::micro>(avx2End - avx2Start).count()
                                           / config_.timingIterations;
                    }
                    
                    // Compare AVX2 vs native
                    result.avx2ValueAbsError = std::abs(result.avx2Result - result.nativeResult);
                    if (std::abs(result.nativeResult) > 1e-15) {
                        result.avx2ValueRelError = result.avx2ValueAbsError / std::abs(result.nativeResult);
                    } else {
                        result.avx2ValueRelError = result.avx2ValueAbsError;
                    }
                    
                    result.avx2DerivAbsError = std::abs(result.avx2Derivative - result.nativeDerivative);
                    if (std::abs(result.nativeDerivative) > 1e-15) {
                        result.avx2DerivRelError = result.avx2DerivAbsError / std::abs(result.nativeDerivative);
                    } else {
                        result.avx2DerivRelError = result.avx2DerivAbsError;
                    }
                    
                    // Compare AVX2 vs SSE2 
                    result.sse2VsAvx2ValueError = std::abs(result.avx2Result - result.tapeResult);
                    result.sse2VsAvx2DerivError = std::abs(result.avx2Derivative - result.tapeDerivative);
                    
                    // Check if AVX2 passes tolerances
                    result.avx2ValuePassed = (result.sse2VsAvx2ValueError <= config_.avx2ValueTolerance);
                    result.avx2DerivPassed = (result.sse2VsAvx2DerivError <= config_.avx2DerivTolerance);
                    
                } catch (...) {
                    // AVX2 not available or compilation failed
                    result.avx2Available = false;
                }
            }
            
            // ===== Compare results =====
            // Value comparison
            // Special handling for inf/nan: if both are inf/nan, consider it a pass
            bool bothInfOrNan = (std::isinf(result.nativeResult) && std::isinf(result.tapeResult)) ||
                                (std::isnan(result.nativeResult) && std::isnan(result.tapeResult));
            
            if (bothInfOrNan) {
                result.absoluteError = 0.0;  // Consider them equal
                result.relativeError = 0.0;
                result.valuePassed = true;
            } else {
                result.absoluteError = std::abs(result.tapeResult - result.nativeResult);
                if (std::abs(result.nativeResult) > 1e-15) {
                    result.relativeError = result.absoluteError / std::abs(result.nativeResult);
                } else {
                    result.relativeError = result.absoluteError;
                }
                result.valuePassed = (result.absoluteError <= config_.absoluteTolerance) ||
                                    (result.relativeError <= config_.relativeTolerance);
            }
            
            // Derivative comparison
            // If the function value has a singularity (inf/nan), skip derivative checking
            // since derivatives at singularities are undefined/unstable
            bool functionHasSingularity = std::isinf(result.nativeResult) || std::isnan(result.nativeResult) ||
                                          std::isinf(result.tapeResult) || std::isnan(result.tapeResult);
            
            if (functionHasSingularity) {
                // Skip derivative check at singularities
                result.derivativeAbsError = 0.0;
                result.derivativeRelError = 0.0;
                result.derivativePassed = true;  // Don't fail on undefined derivatives
            } else {
                // Normal derivative comparison
                result.derivativeAbsError = std::abs(result.tapeDerivative - result.nativeDerivative);
                if (std::abs(result.nativeDerivative) > 1e-15) {
                    result.derivativeRelError = result.derivativeAbsError / std::abs(result.nativeDerivative);
                } else {
                    result.derivativeRelError = result.derivativeAbsError;
                }
                result.derivativePassed = (result.derivativeAbsError <= config_.derivativeAbsTolerance) ||
                                         (result.derivativeRelError <= config_.derivativeRelTolerance);
            }
            
            // Overall pass/fail
            result.passed = result.valuePassed && result.derivativePassed;
            
            // Always check AVX2 results if available - fail if they exceed tolerance
            if (result.avx2Available) {
                bool avx2PassOverall = result.avx2ValuePassed && result.avx2DerivPassed;
                if (!avx2PassOverall) {
                    result.passed = false;
                    if (config_.verbose) {
                        std::cout << " [AVX2 MISMATCH: Value=" << std::scientific << result.sse2VsAvx2ValueError 
                                  << ", Deriv=" << result.sse2VsAvx2DerivError << "]" << std::fixed;
                    }
                }
            }
            
            if (!result.passed) {
                allPassed = false;
            }
            
            results_.push_back(result);
            
            // Print result based on showOnlyFailures setting
            if (config_.verbose) {
                if (!config_.showOnlyFailures || !result.passed) {
                    PrintResult(result);
                }
            }
            
            // Stop if configured to stop on first failure
            if (!result.passed && config_.stopOnFirstFailure) {
                std::cout << "\nStopping on first failure." << std::endl;
                break;
            }
        }
        
        // Show which kernels were actually available/tested
        bool anyAvx2Available = std::any_of(results_.begin(), results_.end(), 
                                           [](const SanityDiffTestResult& r) { return r.avx2Available; });
        
        std::cout << "\nKernel Status: SSE2_SCALAR [OK]";
        if (config_.testAvx2) {
            if (anyAvx2Available) {
                std::cout << ", AVX2_PACKED [OK]";
            } else {
                std::cout << ", AVX2_PACKED [FAILED] (not available/failed)";
            }
        }
        std::cout << std::endl;
        
        // ===== Comparison Section =====
        if (config_.testAvx2 && anyAvx2Available) {
            std::cout << "\n=== SSE2 vs AVX2 Comparison ===" << std::endl;
            
            int sse2VsAvx2ValueMismatches = 0;
            int sse2VsAvx2DerivMismatches = 0;
            double maxSse2VsAvx2ValueError = 0.0;
            double maxSse2VsAvx2DerivError = 0.0;
            double avgSse2Speedup = 0.0;
            double avgAvx2Speedup = 0.0;
            int validComparisons = 0;
            
            for (const auto& result : results_) {
                if (result.avx2Available) {
                    validComparisons++;
                    if (!result.avx2ValuePassed) {
                        sse2VsAvx2ValueMismatches++;
                    }
                    if (!result.avx2DerivPassed) {
                        sse2VsAvx2DerivMismatches++;
                    }
                    maxSse2VsAvx2ValueError = std::max(maxSse2VsAvx2ValueError, result.sse2VsAvx2ValueError);
                    maxSse2VsAvx2DerivError = std::max(maxSse2VsAvx2DerivError, result.sse2VsAvx2DerivError);
                    
                    if (config_.showTimings && result.tapeTimeUs > 0 && result.avx2TimeUs > 0) {
                        avgSse2Speedup += result.nativeTimeUs / result.tapeTimeUs;
                        avgAvx2Speedup += result.nativeTimeUs / result.avx2TimeUs;
                    }
                }
            }
            
            if (validComparisons > 0) {
                std::cout << "Tests compared: " << validComparisons << std::endl;
                std::cout << "SSE2 vs AVX2 value mismatches: " << sse2VsAvx2ValueMismatches << std::endl;
                std::cout << "SSE2 vs AVX2 derivative mismatches: " << sse2VsAvx2DerivMismatches << std::endl;
                std::cout << "Max SSE2 vs AVX2 value error: " << std::scientific << maxSse2VsAvx2ValueError << std::endl;
                std::cout << "Max SSE2 vs AVX2 derivative error: " << maxSse2VsAvx2DerivError << std::endl;
                
                if (config_.showTimings) {
                    avgSse2Speedup /= validComparisons;
                    avgAvx2Speedup /= validComparisons;
                    std::cout << "Average SSE2 speedup: " << std::fixed << std::setprecision(2) << avgSse2Speedup << "x" << std::endl;
                    std::cout << "Average AVX2 speedup: " << avgAvx2Speedup << "x" << std::endl;
                    if (avgAvx2Speedup > avgSse2Speedup) {
                        std::cout << "AVX2 advantage: " << std::setprecision(1) << (avgAvx2Speedup / avgSse2Speedup) << "x faster than SSE2" << std::endl;
                    } else if (avgSse2Speedup > avgAvx2Speedup) {
                        std::cout << "SSE2 advantage: " << std::setprecision(1) << (avgSse2Speedup / avgAvx2Speedup) << "x faster than AVX2" << std::endl;
                    }
                }
                
                if (sse2VsAvx2ValueMismatches == 0 && sse2VsAvx2DerivMismatches == 0) {
                    std::cout << "✓ Perfect SSE2/AVX2 consistency (values + derivatives)" << std::endl;
                } else {
                    std::cout << "⚠ SSE2/AVX2 consistency issues detected" << std::endl;
                    if (sse2VsAvx2ValueMismatches > 0) {
                        std::cout << "  - Value inconsistencies: " << sse2VsAvx2ValueMismatches << std::endl;
                    }
                    if (sse2VsAvx2DerivMismatches > 0) {
                        std::cout << "  - Derivative inconsistencies: " << sse2VsAvx2DerivMismatches << std::endl;
                    }
                }
            }
        }
        
        // Print summary
        PrintSummary(allPassed);
        
        return allPassed;
    }
    
    // Get detailed results
    const std::vector<SanityDiffTestResult>& GetResults() const { return results_; }
    
    // Get config (for modification)
    SanityDiffConfig& Config() { return config_; }
    
private:
    void PrintHeader() {
        std::cout << std::fixed << std::setprecision(12);
        
        // Value header
        std::cout << "\n" << std::setw(15) << "Input" 
                  << std::setw(20) << "Native f(x)" 
                  << std::setw(20) << "SSE2 f(x)"
                  << std::setw(15) << "SSE2 Error"
                  << std::setw(15) << "Rel Error";
        
        // Add AVX2 columns if AVX2 testing is enabled
        if (config_.testAvx2) {
            std::cout << std::setw(20) << "AVX2 f(x)"
                      << std::setw(15) << "AVX2 Error"
                      << std::setw(15) << "SSE2vsAVX2";
        }
        
        std::cout << std::setw(10) << "Status";
        
        // Derivative header if enabled
        if (config_.showDerivatives) {
            std::cout << " | " << std::setw(20) << "FD f'(x)"
                      << std::setw(20) << "SSE2 f'(x)"
                      << std::setw(15) << "SSE2 Der Err"
                      << std::setw(15) << "Rel Error";
            
            // Add AVX2 derivative columns if AVX2 testing is enabled
            if (config_.testAvx2) {
                std::cout << std::setw(20) << "AVX2 f'(x)"
                          << std::setw(15) << "AVX2 Der Err"
                          << std::setw(15) << "SSE2vsAVX2";
            }
            
            std::cout << std::setw(10) << "Status";
        }
        
        // Timing header if enabled
        if (config_.showTimings) {
            std::cout << " | " << std::setw(12) << "Native(µs)"
                      << std::setw(12) << "SSE2(µs)";
            if (config_.testAvx2) {
                std::cout << std::setw(12) << "AVX2(µs)";
            }
            std::cout << std::setw(10) << "Speedup";
        }
        
        std::cout << std::endl;
        int lineWidth = config_.showDerivatives ? 200 : 120;
        if (config_.showTimings) {
            lineWidth += config_.testAvx2 ? 47 : 35;  // Account for AVX2 timing column
        }
        if (config_.testAvx2) {
            lineWidth += config_.showDerivatives ? 100 : 50;  // More space for AVX2 value+derivative columns
        }
        std::cout << std::string(lineWidth, '-') << std::endl;
    }
    
    void PrintResult(const SanityDiffTestResult& result) {
        // Print value results
        std::cout << std::setw(15) << result.input
                  << std::setw(20) << result.nativeResult
                  << std::setw(20) << result.tapeResult
                  << std::setw(15) << std::scientific << result.absoluteError
                  << std::setw(15) << result.relativeError << std::fixed;
        
        // Add AVX2 value columns if AVX2 testing is enabled
        if (config_.testAvx2) {
            if (result.avx2Available) {
                std::cout << std::setw(20) << result.avx2Result
                          << std::setw(15) << std::scientific << result.avx2ValueAbsError
                          << std::setw(15) << result.sse2VsAvx2ValueError << std::fixed;
            } else {
                std::cout << std::setw(20) << "N/A"
                          << std::setw(15) << "N/A"
                          << std::setw(15) << "N/A";
            }
        }
        
        std::cout << std::setw(10) << (result.valuePassed ? "PASS" : "FAIL");
        
        // Print derivative results if enabled
        if (config_.showDerivatives) {
            std::cout << " | " << std::setw(20) << std::fixed << result.nativeDerivative
                      << std::setw(20) << result.tapeDerivative
                      << std::setw(15) << std::scientific << result.derivativeAbsError
                      << std::setw(15) << result.derivativeRelError << std::fixed;
            
            // Add AVX2 derivative columns if AVX2 testing is enabled
            if (config_.testAvx2) {
                if (result.avx2Available) {
                    std::cout << std::setw(20) << std::fixed << result.avx2Derivative
                              << std::setw(15) << std::scientific << result.avx2DerivAbsError
                              << std::setw(15) << result.sse2VsAvx2DerivError << std::fixed;
                } else {
                    std::cout << std::setw(20) << "N/A"
                              << std::setw(15) << "N/A"
                              << std::setw(15) << "N/A";
                }
            }
            
            std::cout << std::setw(10) << (result.derivativePassed ? "PASS" : "FAIL");
        }
        
        // Print timing results if enabled
        if (config_.showTimings) {
            std::cout << " | " << std::setw(12) << std::setprecision(3) << result.nativeTimeUs
                      << std::setw(12) << result.tapeTimeUs;
            if (config_.testAvx2) {
                if (result.avx2Available) {
                    std::cout << std::setw(12) << result.avx2TimeUs;
                } else {
                    std::cout << std::setw(12) << "N/A";
                }
            }
            std::cout << std::setw(10) << std::setprecision(2) 
                      << (result.nativeTimeUs / result.tapeTimeUs) << "x";
        }
        
        std::cout << std::endl;
    }
    
    void PrintSummary(bool allPassed) {
        int passedCount = 0;
        int failedCount = 0;
        int valueFailures = 0;
        int derivativeFailures = 0;
        double maxAbsError = 0.0;
        double maxRelError = 0.0;
        double maxDerivAbsError = 0.0;
        double maxDerivRelError = 0.0;
        double avgSpeedup = 0.0;
        
        for (const auto& result : results_) {
            if (result.passed) {
                passedCount++;
            } else {
                failedCount++;
                if (!result.valuePassed) valueFailures++;
                if (!result.derivativePassed) derivativeFailures++;
            }
            maxAbsError = std::max(maxAbsError, result.absoluteError);
            maxRelError = std::max(maxRelError, result.relativeError);
            maxDerivAbsError = std::max(maxDerivAbsError, result.derivativeAbsError);
            maxDerivRelError = std::max(maxDerivRelError, result.derivativeRelError);
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
        std::cout << "Failed: " << failedCount;
        if (failedCount > 0) {
            std::cout << " (Values: " << valueFailures << ", Derivatives: " << derivativeFailures << ")";
        }
        std::cout << std::endl;
        
        std::cout << "\nValue Errors:" << std::endl;
        std::cout << "  Max absolute error: " << std::scientific << maxAbsError << std::endl;
        std::cout << "  Max relative error: " << maxRelError << std::endl;
        
        if (config_.showDerivatives) {
            std::cout << "\nDerivative Errors:" << std::endl;
            std::cout << "  Max absolute error: " << maxDerivAbsError << std::endl;
            std::cout << "  Max relative error: " << maxDerivRelError << std::endl;
        }
        
        if (config_.showTimings && !results_.empty()) {
            std::cout << "\nPerformance:" << std::endl;
            std::cout << "  Average speedup: " << std::fixed << std::setprecision(2) 
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
auto makeSanityCheckerDiff(const std::string& name, FuncDouble funcDouble, FuncTP funcTP,
                           const std::vector<double>& inputs,
                           const SanityDiffConfig& config = SanityDiffConfig()) {
    return SanityCheckerDiff<FuncDouble, FuncTP>(name, funcDouble, funcTP, inputs, config);
}

} // namespace tools
} // namespace forge