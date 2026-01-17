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
#include <native/fdouble.hpp>
#include "../../src/graph/graph_recorder.hpp"
#include "../../src/compiler/forge_engine.hpp"
#include "../../src/compiler/interfaces/node_value_buffer.hpp"

namespace forge {
namespace tools {

struct MultiDimTestResult {
    std::vector<double> inputs;
    std::vector<double> nativeResults;
    std::vector<double> tapeResults;
    std::vector<double> absoluteErrors;
    std::vector<double> relativeErrors;
    std::vector<bool> outputsPassed;
    bool overallPassed;
    double nativeTimeUs;
    double tapeTimeUs;
};

struct MultiDimSanityConfig {
    double absoluteTolerance = 1e-10;
    double relativeTolerance = 1e-10;
    bool verbose = true;
    bool showTimings = true;
    bool stopOnFirstFailure = false;
    int warmupIterations = 10;
    int timingIterations = 100;
    
    // Compiler configuration for debugging tape compilation
    forge::CompilerConfig compilerConfig = forge::CompilerConfig::Default();
};

template<typename FuncDouble, typename FuncTP>
class SanityMultiDimChecker {
private:
    FuncDouble funcDouble_;
    FuncTP funcTP_;
    std::vector<std::vector<double>> testCases_;
    MultiDimSanityConfig config_;
    std::vector<MultiDimTestResult> results_;
    std::string functionName_;
    
    size_t numInputs_ = 0;
    size_t numOutputs_ = 0;
    bool dimensionsDetected_ = false;
    
    std::unique_ptr<forge::StitchedKernel> kernel_;
    std::unique_ptr<forge::INodeValueBuffer> buffer_;
    std::vector<forge::NodeId> inputNodes_;
    std::vector<forge::NodeId> outputNodes_;
    
    void detectDimensions() {
        if (dimensionsDetected_ || testCases_.empty()) return;
        
        numInputs_ = testCases_[0].size();
        auto result = funcDouble_(testCases_[0]);
        numOutputs_ = result.size();
        dimensionsDetected_ = true;
        
        if (config_.verbose) {
            std::cout << "Detected dimensions: R^" << numInputs_ << " -> R^" << numOutputs_ << std::endl;
        }
    }
    
    void compileKernel() {
        using namespace forge;
        
        GraphRecorder recorder;
        recorder.start();
        
        std::vector<fdouble> tpInputs;
        for (size_t i = 0; i < numInputs_; ++i) {
            fdouble x(0.0);
            x.markInput();
            tpInputs.push_back(x);
        }
        
        auto tpOutputs = funcTP_(tpInputs);
        
        for (auto& output : tpOutputs) {
            output.markOutput();
        }
        
        recorder.stop();
        Graph graph = recorder.graph();
        
        forge::ForgeEngine compiler(config_.compilerConfig);
        kernel_ = compiler.compile(graph);
        
        buffer_ = forge::NodeValueBufferFactory::create(graph, *kernel_);
        
        inputNodes_.clear();
        for (size_t i = 0; i < numInputs_; ++i) {
            inputNodes_.push_back(static_cast<NodeId>(i));
        }
        
        outputNodes_ = graph.outputs;
    }
    
public:
    SanityMultiDimChecker(const std::string& name, 
                          FuncDouble funcDouble, 
                          FuncTP funcTP,
                          const std::vector<std::vector<double>>& testCases,
                          const MultiDimSanityConfig& config = MultiDimSanityConfig())
        : functionName_(name), funcDouble_(funcDouble), funcTP_(funcTP), 
          testCases_(testCases), config_(config) {}
    
    bool RunTests() {
        using namespace std::chrono;
        
        if (testCases_.empty()) {
            std::cout << "No test cases provided for " << functionName_ << std::endl;
            return false;
        }
        
        detectDimensions();
        compileKernel();
        
        results_.clear();
        bool allPassed = true;
        
        std::cout << "\n=== Multi-Dim Sanity Check: " << functionName_ 
                  << " (R^" << numInputs_ << " -> R^" << numOutputs_ << ") ===" << std::endl;
        std::cout << "Testing " << testCases_.size() << " input vectors" << std::endl;
        
        // Print table header if verbose
        if (config_.verbose) {
            PrintTableHeader();
        }
        
        for (const auto& testInputs : testCases_) {
            MultiDimTestResult result;
            result.inputs = testInputs;
            
            for (int i = 0; i < config_.warmupIterations; ++i) {
                volatile auto dummy = funcDouble_(testInputs);
                (void)dummy;
            }
            
            auto nativeStart = high_resolution_clock::now();
            for (int i = 0; i < config_.timingIterations; ++i) {
                result.nativeResults = funcDouble_(testInputs);
            }
            auto nativeEnd = high_resolution_clock::now();
            result.nativeTimeUs = duration<double, std::micro>(nativeEnd - nativeStart).count() 
                                  / config_.timingIterations;
            
            for (size_t i = 0; i < numInputs_; ++i) {
                buffer_->setValue(inputNodes_[i], testInputs[i]);
            }
            
            for (int i = 0; i < config_.warmupIterations; ++i) {
                kernel_->execute(*buffer_);
            }
            
            auto tapeStart = high_resolution_clock::now();
            for (int i = 0; i < config_.timingIterations; ++i) {
                for (size_t j = 0; j < numInputs_; ++j) {
                    buffer_->setValue(inputNodes_[j], testInputs[j]);
                }
                kernel_->execute(*buffer_);
            }
            auto tapeEnd = high_resolution_clock::now();
            result.tapeTimeUs = duration<double, std::micro>(tapeEnd - tapeStart).count() 
                               / config_.timingIterations;
            
            result.tapeResults.clear();
            for (auto outputNode : outputNodes_) {
                result.tapeResults.push_back(buffer_->getValue(outputNode));
            }
            
            result.absoluteErrors.clear();
            result.relativeErrors.clear();
            result.outputsPassed.clear();
            result.overallPassed = true;
            
            for (size_t i = 0; i < numOutputs_; ++i) {
                double absError = std::abs(result.tapeResults[i] - result.nativeResults[i]);
                double relError;
                
                if (std::abs(result.nativeResults[i]) > 1e-15) {
                    relError = absError / std::abs(result.nativeResults[i]);
                } else {
                    relError = absError;
                }
                
                bool passed = (absError <= config_.absoluteTolerance) ||
                             (relError <= config_.relativeTolerance);
                
                if (std::isinf(result.nativeResults[i]) && std::isinf(result.tapeResults[i])) {
                    passed = (result.nativeResults[i] > 0) == (result.tapeResults[i] > 0);
                    absError = passed ? 0.0 : std::numeric_limits<double>::infinity();
                    relError = passed ? 0.0 : std::numeric_limits<double>::infinity();
                } else if (std::isnan(result.nativeResults[i]) && std::isnan(result.tapeResults[i])) {
                    passed = true;
                    absError = 0.0;
                    relError = 0.0;
                }
                
                result.absoluteErrors.push_back(absError);
                result.relativeErrors.push_back(relError);
                result.outputsPassed.push_back(passed);
                
                if (!passed) {
                    result.overallPassed = false;
                    allPassed = false;
                }
            }
            
            results_.push_back(result);
            
            if (config_.verbose) {
                PrintTestResult(result);
            }
            
            if (!result.overallPassed && config_.stopOnFirstFailure) {
                std::cout << "\nStopping on first failure." << std::endl;
                break;
            }
        }
        
        PrintSummary(allPassed);
        
        return allPassed;
    }
    
    const std::vector<MultiDimTestResult>& GetResults() const { return results_; }
    MultiDimSanityConfig& Config() { return config_; }
    
private:
    void PrintTableHeader() {
        std::cout << std::fixed << std::setprecision(12);
        std::cout << "\n" << std::setw(25) << "Input Vector" 
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
        
        int lineWidth = 140;
        if (config_.showTimings) {
            lineWidth = 157;
        }
        std::cout << std::string(lineWidth, '-') << std::endl;
    }
    
    void PrintTestResult(const MultiDimTestResult& result) {
        // Format input vector as string
        std::stringstream inputStr;
        inputStr << "[";
        for (size_t i = 0; i < result.inputs.size(); ++i) {
            if (i > 0) inputStr << ", ";
            inputStr << std::fixed << std::setprecision(2) << result.inputs[i];
        }
        inputStr << "]";
        
        // Print one row for each output dimension
        for (size_t i = 0; i < numOutputs_; ++i) {
            // First output shows the input vector, others show empty
            if (i == 0) {
                std::cout << std::setw(25) << inputStr.str();
            } else {
                std::cout << std::setw(25) << "";
            }
            
            // Output index
            std::cout << std::setw(10) << ("[" + std::to_string(i) + "]");
            
            // Results and errors
            std::cout << std::setw(20) << std::fixed << std::setprecision(12) 
                      << result.nativeResults[i];
            std::cout << std::setw(20) << result.tapeResults[i];
            std::cout << std::setw(15) << std::scientific << result.absoluteErrors[i];
            std::cout << std::setw(15) << result.relativeErrors[i];
            
            // Timings (only on first row)
            if (config_.showTimings) {
                if (i == 0) {
                    std::cout << std::setw(15) << std::fixed << std::setprecision(3) 
                              << result.nativeTimeUs;
                    std::cout << std::setw(15) << result.tapeTimeUs;
                    std::cout << std::setw(12) << std::setprecision(2) 
                              << (result.nativeTimeUs / result.tapeTimeUs) << "x";
                } else {
                    std::cout << std::setw(15) << ""
                              << std::setw(15) << ""
                              << std::setw(12) << "";
                }
            }
            
            // Status for this output
            std::cout << std::setw(10) << (result.outputsPassed[i] ? "PASS" : "FAIL");
            std::cout << std::endl;
        }
    }
    
    void PrintSummary(bool allPassed) {
        int totalOutputs = 0;
        int passedOutputs = 0;
        int failedVectors = 0;
        double maxAbsError = 0.0;
        double maxRelError = 0.0;
        double avgSpeedup = 0.0;
        
        for (const auto& result : results_) {
            if (!result.overallPassed) {
                failedVectors++;
            }
            
            for (size_t i = 0; i < result.outputsPassed.size(); ++i) {
                totalOutputs++;
                if (result.outputsPassed[i]) {
                    passedOutputs++;
                }
                maxAbsError = std::max(maxAbsError, result.absoluteErrors[i]);
                maxRelError = std::max(maxRelError, result.relativeErrors[i]);
            }
            
            if (config_.showTimings && result.tapeTimeUs > 0) {
                avgSpeedup += result.nativeTimeUs / result.tapeTimeUs;
            }
        }
        
        if (config_.showTimings && !results_.empty()) {
            avgSpeedup /= results_.size();
        }
        
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Total test vectors: " << results_.size() << std::endl;
        std::cout << "Failed vectors: " << failedVectors << std::endl;
        std::cout << "Total outputs tested: " << totalOutputs 
                  << " (" << results_.size() << " vectors × " << numOutputs_ << " dimensions)" << std::endl;
        std::cout << "Passed outputs: " << passedOutputs << "/" << totalOutputs << std::endl;
        std::cout << "Max absolute error: " << std::scientific << maxAbsError << std::endl;
        std::cout << "Max relative error: " << maxRelError << std::endl;
        
        if (config_.showTimings && !results_.empty()) {
            std::cout << "Average speedup: " << std::fixed << std::setprecision(2) 
                      << avgSpeedup << "x" << std::endl;
        }
        
        if (allPassed) {
            std::cout << "\n✓ All " << totalOutputs << " outputs PASSED" << std::endl;
        } else {
            std::cout << "\n✗ Some outputs FAILED" << std::endl;
        }
    }
};

template<typename FuncDouble, typename FuncTP>
auto makeSanityMultiDimChecker(const std::string& name, 
                                FuncDouble funcDouble, 
                                FuncTP funcTP,
                                const std::vector<std::vector<double>>& testCases,
                                const MultiDimSanityConfig& config = MultiDimSanityConfig()) {
    return SanityMultiDimChecker<FuncDouble, FuncTP>(name, funcDouble, funcTP, testCases, config);
}

} // namespace tools
} // namespace forge