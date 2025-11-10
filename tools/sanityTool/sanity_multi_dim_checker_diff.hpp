#pragma once

#include <functional>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <cmath>
#include <chrono>
#include <memory>
#include "../../tools/types/fdouble.hpp"
#include "../../src/graph/graph_recorder.hpp"
#include "../../src/compiler/forge_engine.hpp"
#include "../../src/compiler/node_value_buffers/node_value_buffer.hpp"

namespace forge {
namespace tools {

// Result structure for multi-dimensional tests with derivatives
struct MultiDimDiffTestResult {
    std::vector<double> inputs;
    std::vector<double> nativeResults;
    std::vector<double> tapeResults;
    std::vector<double> absoluteErrors;
    std::vector<double> relativeErrors;
    std::vector<bool> valuesPassed;
    
    // Jacobian results
    std::vector<std::vector<double>> fdJacobian;  // Finite differences
    std::vector<std::vector<double>> adJacobian;  // Automatic differentiation
    std::vector<std::vector<double>> jacobianAbsErrors;
    std::vector<std::vector<double>> jacobianRelErrors;
    std::vector<std::vector<bool>> jacobianPassed;
    
    bool overallPassed;
    double maxValueError;
    double maxJacobianError;
    
    // Timing
    double nativeTimeUs;
    double tapeTimeUs;
};

// Configuration for multi-dim sanity checker with derivatives
struct MultiDimDiffConfig {
    // Value tolerance
    double absoluteTolerance = 1e-10;
    double relativeTolerance = 1e-10;
    
    // Derivative tolerance (typically less strict than value tolerance)
    double derivativeAbsTolerance = 1e-6;
    double derivativeRelTolerance = 1e-6;
    
    // Finite difference parameters
    double finiteDiffBump = 1e-8;
    bool useRichardsonExtrapolation = false;
    
    // Display options
    bool verbose = true;
    bool showTimings = true;
    bool showJacobian = true;
    bool stopOnFirstFailure = false;
    
    // Performance options
    int warmupIterations = 10;
    int timingIterations = 100;
};

template<typename FuncDouble, typename FuncTP>
class SanityMultiDimCheckerDiff {
private:
    FuncDouble funcDouble_;
    FuncTP funcTP_;
    std::vector<std::vector<double>> testCases_;
    MultiDimDiffConfig config_;
    std::vector<MultiDimDiffTestResult> results_;
    std::string functionName_;
    
    size_t numInputs_ = 0;
    size_t numOutputs_ = 0;
    bool dimensionsDetected_ = false;
    
    // Detect dimensions from first test case
    void detectDimensions() {
        if (dimensionsDetected_ || testCases_.empty()) return;
        
        numInputs_ = testCases_[0].size();
        auto result = funcDouble_(testCases_[0]);
        numOutputs_ = result.size();
        dimensionsDetected_ = true;
        
        std::cout << "Detected dimensions: R^" << numInputs_ 
                  << " -> R^" << numOutputs_ << std::endl;
    }
    
    // Compute finite difference Jacobian
    std::vector<std::vector<double>> computeFiniteDifferenceJacobian(
        const std::vector<double>& input) {
        
        std::vector<std::vector<double>> jacobian(numOutputs_, 
                                                 std::vector<double>(numInputs_));
        
        for (size_t j = 0; j < numInputs_; ++j) {
            std::vector<double> input_plus = input;
            std::vector<double> input_minus = input;
            
            double h = config_.finiteDiffBump;
            input_plus[j] += h;
            input_minus[j] -= h;
            
            auto f_plus = funcDouble_(input_plus);
            auto f_minus = funcDouble_(input_minus);
            
            for (size_t i = 0; i < numOutputs_; ++i) {
                if (config_.useRichardsonExtrapolation) {
                    // Richardson extrapolation for higher accuracy
                    std::vector<double> input_plus2 = input;
                    std::vector<double> input_minus2 = input;
                    input_plus2[j] += h/2.0;
                    input_minus2[j] -= h/2.0;
                    
                    auto f_plus2 = funcDouble_(input_plus2);
                    auto f_minus2 = funcDouble_(input_minus2);
                    
                    double D1 = (f_plus[i] - f_minus[i]) / (2.0 * h);
                    double D2 = (f_plus2[i] - f_minus2[i]) / h;
                    jacobian[i][j] = (4.0 * D2 - D1) / 3.0;
                } else {
                    jacobian[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * h);
                }
            }
        }
        
        return jacobian;
    }
    
    // Compute automatic differentiation Jacobian
    std::vector<std::vector<double>> computeAutoDiffJacobian(
        const std::vector<double>& input) {
        
        using namespace forge;
        std::vector<std::vector<double>> jacobian(numOutputs_, 
                                                 std::vector<double>(numInputs_));
        
        // For each output, we need to compute gradients w.r.t. all inputs
        for (size_t outputIdx = 0; outputIdx < numOutputs_; ++outputIdx) {
            GraphRecorder recorder;
            recorder.start();
            
            // Create inputs with gradient tracking
            std::vector<fdouble> tpInputs;
            for (size_t i = 0; i < numInputs_; ++i) {
                fdouble x(0.0);
                x.markInputAndDiff();
                tpInputs.push_back(x);
            }
            
            // Apply function
            auto tpOutputs = funcTP_(tpInputs);
            
            // Mark only the output we're interested in
            tpOutputs[outputIdx].markOutput();
            
            recorder.stop();
            Graph graph = recorder.graph();
            
            // Compile
            forge::ForgeEngine compiler;
            auto kernel = compiler.compile(graph);
            
            // Create NodeValueBuffer
            auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);
            
            // Set input values
            for (size_t i = 0; i < numInputs_; ++i) {
                buffer->setValue(graph.diff_inputs[i], input[i]);
            }
            buffer->clearGradients();
            
            // Execute to compute gradients
            kernel->execute(*buffer);
            
            // Extract gradients for this output w.r.t. all inputs
            for (size_t i = 0; i < numInputs_; ++i) {
                jacobian[outputIdx][i] = buffer->getGradient(graph.diff_inputs[i]);
            }
        }
        
        return jacobian;
    }
    
public:
    SanityMultiDimCheckerDiff(const std::string& name, FuncDouble funcDouble, FuncTP funcTP,
                             const std::vector<std::vector<double>>& inputs,
                             const MultiDimDiffConfig& config = MultiDimDiffConfig())
        : functionName_(name), funcDouble_(funcDouble), funcTP_(funcTP),
          testCases_(inputs), config_(config) {}
    
    bool RunTests() {
        using namespace std::chrono;
        
        detectDimensions();
        results_.clear();
        bool allPassed = true;
        
        std::cout << "\n=== Multi-Dim Sanity Check with Derivatives: " << functionName_ 
                  << " (R^" << numInputs_ << " -> R^" << numOutputs_ << ") ===" << std::endl;
        std::cout << "Testing " << testCases_.size() << " input vectors" << std::endl;
        
        // Section 1: Forward evaluation (same as regular multi-dim checker)
        std::cout << "\n--- Section 1: Forward Evaluation ---" << std::endl;
        if (config_.verbose) {
            PrintForwardHeader();
        }
        
        size_t totalOutputs = 0;
        size_t passedOutputs = 0;
        double sumSpeedup = 0.0;
        
        for (const auto& testCase : testCases_) {
            MultiDimDiffTestResult result;
            result.inputs = testCase;
            
            // Native evaluation with timing
            auto nativeStart = high_resolution_clock::now();
            for (int i = 0; i < config_.timingIterations; ++i) {
                result.nativeResults = funcDouble_(testCase);
            }
            auto nativeEnd = high_resolution_clock::now();
            result.nativeTimeUs = duration<double, std::micro>(nativeEnd - nativeStart).count() 
                                 / config_.timingIterations;
            
            // Graph evaluation with timing
            using namespace forge;
            GraphRecorder recorder;
            recorder.start();
            
            std::vector<fdouble> tpInputs;
            for (size_t i = 0; i < numInputs_; ++i) {
                fdouble x(0.0);
                x.markInputAndDiff();  // Mark for derivatives too
                tpInputs.push_back(x);
            }
            
            auto tpOutputs = funcTP_(tpInputs);
            for (auto& output : tpOutputs) {
                output.markOutput();
            }
            
            recorder.stop();
            Graph graph = recorder.graph();
            
            forge::ForgeEngine compiler;
            auto kernel = compiler.compile(graph);
            auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);
            
            // Warmup
            for (int i = 0; i < config_.warmupIterations; ++i) {
                for (size_t j = 0; j < numInputs_; ++j) {
                    buffer->setValue(graph.diff_inputs[j], testCase[j]);
                }
                kernel->execute(*buffer);
            }
            
            // Timed execution
            auto tapeStart = high_resolution_clock::now();
            for (int i = 0; i < config_.timingIterations; ++i) {
                for (size_t j = 0; j < numInputs_; ++j) {
                    buffer->setValue(graph.diff_inputs[j], testCase[j]);
                }
                kernel->execute(*buffer);
            }
            auto tapeEnd = high_resolution_clock::now();
            result.tapeTimeUs = duration<double, std::micro>(tapeEnd - tapeStart).count() 
                               / config_.timingIterations;
            
            // Get results
            result.tapeResults.resize(numOutputs_);
            for (size_t i = 0; i < numOutputs_; ++i) {
                result.tapeResults[i] = buffer->getValue(graph.outputs[i]);
            }
            
            // Compare results
            result.absoluteErrors.resize(numOutputs_);
            result.relativeErrors.resize(numOutputs_);
            result.valuesPassed.resize(numOutputs_);
            result.maxValueError = 0.0;
            
            for (size_t i = 0; i < numOutputs_; ++i) {
                result.absoluteErrors[i] = std::abs(result.tapeResults[i] - result.nativeResults[i]);
                if (std::abs(result.nativeResults[i]) > 1e-15) {
                    result.relativeErrors[i] = result.absoluteErrors[i] / std::abs(result.nativeResults[i]);
                } else {
                    result.relativeErrors[i] = result.absoluteErrors[i];
                }
                result.valuesPassed[i] = (result.absoluteErrors[i] <= config_.absoluteTolerance) ||
                                        (result.relativeErrors[i] <= config_.relativeTolerance);
                
                result.maxValueError = std::max(result.maxValueError, result.absoluteErrors[i]);
                totalOutputs++;
                if (result.valuesPassed[i]) passedOutputs++;
            }
            
            sumSpeedup += result.nativeTimeUs / result.tapeTimeUs;
            
            // Print forward results if verbose
            if (config_.verbose) {
                PrintForwardResult(result);
            }
            
            // Compute Jacobians
            result.fdJacobian = computeFiniteDifferenceJacobian(testCase);
            result.adJacobian = computeAutoDiffJacobian(testCase);
            
            // Compare Jacobians
            result.jacobianAbsErrors.resize(numOutputs_, std::vector<double>(numInputs_));
            result.jacobianRelErrors.resize(numOutputs_, std::vector<double>(numInputs_));
            result.jacobianPassed.resize(numOutputs_, std::vector<bool>(numInputs_));
            result.maxJacobianError = 0.0;
            
            for (size_t i = 0; i < numOutputs_; ++i) {
                for (size_t j = 0; j < numInputs_; ++j) {
                    double absErr = std::abs(result.adJacobian[i][j] - result.fdJacobian[i][j]);
                    result.jacobianAbsErrors[i][j] = absErr;
                    
                    if (std::abs(result.fdJacobian[i][j]) > 1e-15) {
                        result.jacobianRelErrors[i][j] = absErr / std::abs(result.fdJacobian[i][j]);
                    } else {
                        result.jacobianRelErrors[i][j] = absErr;
                    }
                    
                    result.jacobianPassed[i][j] = (absErr <= config_.derivativeAbsTolerance) ||
                                                  (result.jacobianRelErrors[i][j] <= config_.derivativeRelTolerance);
                    
                    result.maxJacobianError = std::max(result.maxJacobianError, absErr);
                }
            }
            
            // Check overall pass
            result.overallPassed = true;
            for (bool passed : result.valuesPassed) {
                if (!passed) result.overallPassed = false;
            }
            for (const auto& row : result.jacobianPassed) {
                for (bool passed : row) {
                    if (!passed) result.overallPassed = false;
                }
            }
            
            if (!result.overallPassed) allPassed = false;
            
            results_.push_back(result);
            
            if (!result.overallPassed && config_.stopOnFirstFailure) {
                std::cout << "\nStopping on first failure." << std::endl;
                break;
            }
        }
        
        // Print forward summary
        std::cout << "\n=== Forward Evaluation Summary ===" << std::endl;
        std::cout << "Total test vectors: " << testCases_.size() << std::endl;
        std::cout << "Total outputs tested: " << totalOutputs 
                  << " (" << testCases_.size() << " vectors × " << numOutputs_ << " dimensions)" << std::endl;
        std::cout << "Passed outputs: " << passedOutputs << "/" << totalOutputs << std::endl;
        
        double maxAbsErr = 0.0, maxRelErr = 0.0;
        for (const auto& result : results_) {
            for (size_t i = 0; i < numOutputs_; ++i) {
                maxAbsErr = std::max(maxAbsErr, result.absoluteErrors[i]);
                maxRelErr = std::max(maxRelErr, result.relativeErrors[i]);
            }
        }
        std::cout << "Max absolute error: " << std::scientific << maxAbsErr << std::endl;
        std::cout << "Max relative error: " << maxRelErr << std::endl;
        if (config_.showTimings) {
            std::cout << "Average speedup: " << std::fixed << std::setprecision(2) 
                      << (sumSpeedup / testCases_.size()) << "x" << std::endl;
        }
        
        // Section 2: Jacobian validation
        if (config_.showJacobian) {
            std::cout << "\n--- Section 2: Jacobian Validation ---" << std::endl;
            std::cout << "Finite difference bump size: " << std::scientific 
                      << config_.finiteDiffBump << std::endl;
            
            for (size_t idx = 0; idx < results_.size(); ++idx) {
                PrintJacobianResult(results_[idx], idx);
            }
            
            // Jacobian summary
            std::cout << "\n=== Jacobian Summary ===" << std::endl;
            double maxJacErr = 0.0;
            size_t totalJacElements = numOutputs_ * numInputs_ * testCases_.size();
            size_t passedJacElements = 0;
            
            for (const auto& result : results_) {
                maxJacErr = std::max(maxJacErr, result.maxJacobianError);
                for (const auto& row : result.jacobianPassed) {
                    for (bool passed : row) {
                        if (passed) passedJacElements++;
                    }
                }
            }
            
            std::cout << "Total Jacobian elements: " << totalJacElements << std::endl;
            std::cout << "Passed elements: " << passedJacElements << "/" << totalJacElements << std::endl;
            std::cout << "Max Jacobian error: " << std::scientific << maxJacErr << std::endl;
        }
        
        // Final verdict
        std::cout << std::fixed;
        if (allPassed) {
            std::cout << "\n✓ All tests PASSED" << std::endl;
        } else {
            std::cout << "\n✗ Some tests FAILED" << std::endl;
        }
        
        return allPassed;
    }
    
    const std::vector<MultiDimDiffTestResult>& GetResults() const { return results_; }
    MultiDimDiffConfig& Config() { return config_; }
    
private:
    void PrintForwardHeader() {
        std::cout << std::setw(25) << "Input Vector"
                  << std::setw(10) << "Output"
                  << std::setw(20) << "Native Result"
                  << std::setw(20) << "Graph Result"
                  << std::setw(15) << "Abs Error"
                  << std::setw(15) << "Rel Error";
        
        if (config_.showTimings) {
            std::cout << std::setw(15) << "Native(µs)"
                      << std::setw(15) << "Graph(µs)"
                      << std::setw(12) << "Speedup";
        }
        
        std::cout << std::setw(10) << "Status" << std::endl;
        std::cout << std::string(config_.showTimings ? 157 : 115, '-') << std::endl;
    }
    
    void PrintForwardResult(const MultiDimDiffTestResult& result) {
        // Format input vector
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < result.inputs.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << std::fixed << std::setprecision(2) << result.inputs[i];
        }
        ss << "]";
        std::string inputStr = ss.str();
        
        // Print results for each output
        for (size_t i = 0; i < numOutputs_; ++i) {
            std::cout << std::setw(25) << (i == 0 ? inputStr : "")
                      << std::setw(10) << ("[" + std::to_string(i) + "]")
                      << std::setw(20) << std::fixed << std::setprecision(12) 
                      << result.nativeResults[i]
                      << std::setw(20) << result.tapeResults[i]
                      << std::setw(15) << std::scientific << std::setprecision(12) 
                      << result.absoluteErrors[i]
                      << std::setw(15) << result.relativeErrors[i];
            
            if (config_.showTimings && i == 0) {
                std::cout << std::setw(15) << std::fixed << std::setprecision(3) 
                          << result.nativeTimeUs
                          << std::setw(15) << result.tapeTimeUs
                          << std::setw(12) << std::setprecision(2) 
                          << (result.nativeTimeUs / result.tapeTimeUs) << "x";
            } else if (config_.showTimings) {
                std::cout << std::setw(42) << "";
            }
            
            std::cout << std::setw(10) << (result.valuesPassed[i] ? "PASS" : "FAIL")
                      << std::endl;
        }
    }
    
    void PrintJacobianResult(const MultiDimDiffTestResult& result, size_t testIdx) {
        std::cout << "\nTest Vector " << (testIdx + 1) << ": [";
        for (size_t i = 0; i < result.inputs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(2) << result.inputs[i];
        }
        std::cout << "]" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        std::cout << "Jacobian Matrix (∂f_i/∂x_j):" << std::endl;
        
        // Header for columns
        std::cout << "     ";
        for (size_t j = 0; j < numInputs_; ++j) {
            std::cout << std::setw(30) << ("∂/∂x[" + std::to_string(j) + "]");
        }
        std::cout << std::endl;
        
        // Print each row
        for (size_t i = 0; i < numOutputs_; ++i) {
            std::cout << " f[" << i << "]: ";
            for (size_t j = 0; j < numInputs_; ++j) {
                std::cout << "FD=" << std::fixed << std::setprecision(3) 
                          << std::setw(7) << result.fdJacobian[i][j]
                          << " AD=" << std::setw(7) << result.adJacobian[i][j]
                          << (result.jacobianPassed[i][j] ? " ✓" : " ✗");
                if (j < numInputs_ - 1) std::cout << " | ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "Max Jacobian Error: " << std::scientific 
                  << result.maxJacobianError << std::endl;
    }
};

// Helper function for type deduction
template<typename FuncDouble, typename FuncTP>
auto makeSanityMultiDimCheckerDiff(const std::string& name, FuncDouble funcDouble, FuncTP funcTP,
                                   const std::vector<std::vector<double>>& inputs,
                                   const MultiDimDiffConfig& config = MultiDimDiffConfig()) {
    return SanityMultiDimCheckerDiff<FuncDouble, FuncTP>(name, funcDouble, funcTP, inputs, config);
}

} // namespace tools
} // namespace forge