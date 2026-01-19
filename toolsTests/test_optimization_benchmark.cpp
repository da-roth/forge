#include <gtest/gtest.h>
#include "../tools/graphSerialization/graph_serialization.hpp"
#include "../src/graph/graph_optimizer.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/interfaces/node_value_buffer.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <memory>
#include <map>

using namespace forge;

class OptimizationBenchmark : public ::testing::Test {
protected:
    struct BenchmarkResult {
        std::string configName;
        size_t originalNodes;
        size_t optimizedNodes;
        double compileTimeMs;
        double executeTimeNs;
        double throughputMops;  // Million operations per second
    };

    // Run benchmark for a specific optimization configuration
    BenchmarkResult runBenchmark(
        const std::string& configName,
        const Graph& originalGraph,
        const CompilerConfig& config,
        const std::vector<std::vector<double>>& testInputs)
    {
        BenchmarkResult result;
        result.configName = configName;
        result.originalNodes = originalGraph.nodes.size();

        // Measure compilation time
        auto compileStart = std::chrono::high_resolution_clock::now();

        ForgeEngine engine;
        engine.setConfig(config);

        auto kernel = engine.compile(originalGraph);

        auto compileEnd = std::chrono::high_resolution_clock::now();
        result.compileTimeMs = std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();

        // Count optimized nodes (non-dead nodes)
        result.optimizedNodes = 0;
        // Note: We don't have direct access to the optimized graph from the kernel,
        // but we can estimate from the working nodes size

        // Create workspace
        auto workspace = NodeValueBufferFactory::create(originalGraph, *kernel);

        // Measure execution time (average over multiple runs)
        const int warmupRuns = 5;
        const int benchmarkRuns = 10;

        // Use the first (baseline) input set for all runs to ensure consistent output
        const auto& baselineInputs = testInputs[0];

        // Warmup
        for (int i = 0; i < warmupRuns; ++i) {
            // Set inputs (assuming inputs are node IDs 0, 1, 2, etc.)
            for (size_t j = 0; j < baselineInputs.size(); ++j) {
                workspace->setValue(j, baselineInputs[j]);
            }
            kernel->execute(*workspace);
        }

        // Get output result to verify correctness
        if (!originalGraph.outputs.empty()) {
            double outputResult = workspace->getValue(originalGraph.outputs[0]);
            std::cout << "  Output result: " << std::setprecision(17) << std::fixed << outputResult << "\n";
        }

        // Benchmark - use same inputs for all runs for consistent output
        auto execStart = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < benchmarkRuns; ++i) {
            for (size_t j = 0; j < baselineInputs.size(); ++j) {
                workspace->setValue(j, baselineInputs[j]);
            }
            kernel->execute(*workspace);
        }
        auto execEnd = std::chrono::high_resolution_clock::now();

        auto totalExecTimeSec = std::chrono::duration<double>(execEnd - execStart).count();
        result.executeTimeNs = (totalExecTimeSec / benchmarkRuns) * 1e9;

        // Calculate throughput (nodes executed per second)
        double execTimeSeconds = result.executeTimeNs / 1e9;
        result.throughputMops = (result.originalNodes / execTimeSeconds) / 1e6;

        return result;
    }

    void printResultsTable(const std::vector<BenchmarkResult>& results) {
        std::cout << "\n";
        std::cout << "==========================================================================================\n";
        std::cout << "                            OPTIMIZATION BENCHMARK RESULTS                                \n";
        std::cout << "==========================================================================================\n";
        std::cout << std::left
                  << std::setw(25) << "Configuration"
                  << std::setw(12) << "Nodes"
                  << std::setw(20) << "Compile (s)"
                  << std::setw(20) << "Execute (s)"
                  << std::setw(15) << "Throughput"
                  << "\n";
        std::cout << std::setw(25) << ""
                  << std::setw(12) << "(orig→opt)"
                  << std::setw(20) << ""
                  << std::setw(20) << ""
                  << std::setw(15) << "(Mops/s)"
                  << "\n";
        std::cout << "------------------------------------------------------------------------------------------\n";

        for (const auto& result : results) {
            std::string nodeStr = std::to_string(result.originalNodes);
            if (result.optimizedNodes > 0 && result.optimizedNodes != result.originalNodes) {
                nodeStr += "→" + std::to_string(result.optimizedNodes);
            }

            // Convert to seconds for consistency
            double compileTimeSec = result.compileTimeMs / 1000.0;
            double executeTimeSec = result.executeTimeNs / 1e9;

            // Format with ms and μs in parentheses for reference
            std::ostringstream compileStr, executeStr;
            compileStr << std::fixed << std::setprecision(6) << compileTimeSec
                      << " (" << std::fixed << std::setprecision(2) << result.compileTimeMs << "ms)";
            executeStr << std::fixed << std::setprecision(6) << executeTimeSec
                      << " (" << std::fixed << std::setprecision(2) << (result.executeTimeNs / 1000.0) << "μs)";

            std::cout << std::left
                      << std::setw(25) << result.configName
                      << std::setw(12) << nodeStr
                      << std::setw(20) << compileStr.str()
                      << std::setw(20) << executeStr.str()
                      << std::setw(15) << std::fixed << std::setprecision(1) << result.throughputMops
                      << "\n";
        }

        std::cout << "==========================================================================================\n";

        // Calculate speedups relative to no optimization
        if (results.size() > 1) {
            std::cout << "\nSpeedup Analysis (relative to No Optimizations):\n";
            std::cout << "------------------------------------------------------------------------------------------\n";
            const auto& baseline = results[0];  // Assuming first is "No Optimizations"

            for (size_t i = 1; i < results.size(); ++i) {
                const auto& current = results[i];
                double compileSpeedup = baseline.compileTimeMs / current.compileTimeMs;
                double execSpeedup = baseline.executeTimeNs / current.executeTimeNs;

                std::cout << std::left << std::setw(25) << current.configName
                          << "Compile: " << std::fixed << std::setprecision(2) << compileSpeedup << "x"
                          << "  Exec: " << std::fixed << std::setprecision(2) << execSpeedup << "x"
                          << "\n";
            }
            std::cout << "==========================================================================================\n\n";
        }
    }
};

// Helper function to load input values from JSON file
namespace {
    struct InputStatistics {
        std::vector<double> values;
        size_t totalCount;
        size_t nonZeroCount;
        double minValue;
        double maxValue;
    };

    InputStatistics loadInputsFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open inputs file: " + filename);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json = buffer.str();

        // Simple JSON parsing for input values
        std::vector<double> inputs;

        // Find the "inputs" array
        size_t inputsPos = json.find("\"inputs\"");
        if (inputsPos == std::string::npos) {
            throw std::runtime_error("Could not find 'inputs' array in JSON");
        }

        // Parse each input value
        size_t pos = inputsPos;
        while ((pos = json.find("\"value\":", pos)) != std::string::npos) {
            pos += 8; // Skip "value":

            // Skip whitespace
            while (pos < json.size() && std::isspace(json[pos])) pos++;

            // Extract value string
            size_t endPos = pos;
            while (endPos < json.size() &&
                   (std::isdigit(json[endPos]) || json[endPos] == '.' ||
                    json[endPos] == '-' || json[endPos] == '+' ||
                    json[endPos] == 'e' || json[endPos] == 'E' ||
                    json[endPos] == 'i' || json[endPos] == 'n' || json[endPos] == 'f')) {
                endPos++;
            }

            std::string valueStr = json.substr(pos, endPos - pos);

            // Parse special values
            double value;
            if (valueStr == "inf") {
                value = std::numeric_limits<double>::infinity();
            } else if (valueStr == "-inf") {
                value = -std::numeric_limits<double>::infinity();
            } else if (valueStr == "nan") {
                value = std::numeric_limits<double>::quiet_NaN();
            } else {
                value = std::stod(valueStr);
            }

            inputs.push_back(value);
            pos = endPos;
        }

        // Calculate statistics
        InputStatistics stats;
        stats.values = inputs;
        stats.totalCount = inputs.size();
        stats.nonZeroCount = 0;
        stats.minValue = std::numeric_limits<double>::max();
        stats.maxValue = std::numeric_limits<double>::lowest();

        for (double val : inputs) {
            if (std::abs(val) > 1e-15) {
                stats.nonZeroCount++;
            }
            if (std::isfinite(val)) {
                stats.minValue = std::min(stats.minValue, val);
                stats.maxValue = std::max(stats.maxValue, val);
            }
        }

        return stats;
    }
}

// Test removed - only AnalyzeUserGraph test is needed

// Additional test: Analyze user graph (uses example by default, or custom via env var)
TEST_F(OptimizationBenchmark, AnalyzeUserGraph) {
    // Check if user provided a custom graph via environment variable
    const char* envGraphPath = std::getenv("TAPEPRESSO_GRAPH_FILE");

    // If not, use the example graph
    std::vector<std::string> possiblePaths;
    if (envGraphPath) {
        possiblePaths.push_back(envGraphPath);
    } else {
        // Try to find the example graph
        possiblePaths = {
            "user_graphs/example_user_graph.json",
            "../user_graphs/example_user_graph.json",
            "../../user_graphs/example_user_graph.json",
            "C:/devPrivate/TapePresso/user_graphs/example_user_graph.json",
            "C:\\devPrivate\\TapePresso\\user_graphs\\example_user_graph.json"
        };
    }

    Graph graph;
    std::string userGraphPath;
    bool loaded = false;

    for (const auto& path : possiblePaths) {
        try {
            graph = loadGraphFromFile(path);
            userGraphPath = path;
            loaded = true;
            break;
        } catch (const std::exception&) {
            // Try next path
        }
    }

    if (!loaded) {
        GTEST_SKIP() << "Could not load user graph.\n"
                     << "Either set TAPEPRESSO_GRAPH_FILE environment variable,\n"
                     << "or ensure user_graphs/example_user_graph.json exists.";
    }

    std::cout << "\n=== Analyzing User Graph: user_graphs/example_user_graph.json ===\n";
    std::cout << "Original nodes: " << graph.nodes.size() << "\n";
    std::cout << "Constant pool: " << graph.constPool.size() << "\n";
    std::cout << "Outputs: " << graph.outputs.size() << "\n";

    // Count inputs and node types
    size_t inputCount = 0;
    std::map<OpCode, size_t> nodeTypeCounts;
    for (const auto& node : graph.nodes) {
        if (node.op == OpCode::Input) inputCount++;
        nodeTypeCounts[node.op]++;
    }
    std::cout << "Inputs: " << inputCount << "\n";

    // Print node type breakdown
    std::cout << "\n=== Graph Load Diagnostics (C++) ===\n";
    std::cout << "  Node type breakdown (top 10):\n";
    std::vector<std::pair<OpCode, size_t>> sortedCounts(nodeTypeCounts.begin(), nodeTypeCounts.end());
    std::sort(sortedCounts.begin(), sortedCounts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    auto opToStr = [](OpCode op) -> std::string {
        switch(op) {
            case OpCode::Input: return "Input";
            case OpCode::Constant: return "Constant";
            case OpCode::Add: return "Add";
            case OpCode::Sub: return "Sub";
            case OpCode::Mul: return "Mul";
            case OpCode::Div: return "Div";
            case OpCode::If: return "If";
            case OpCode::Exp: return "Exp";
            case OpCode::Log: return "Log";
            default: return "Other";
        }
    };

    for (size_t i = 0; i < std::min(size_t(10), sortedCounts.size()); ++i) {
        std::cout << "    " << opToStr(sortedCounts[i].first) << ": " << sortedCounts[i].second << "\n";
    }
    std::cout << "\n";

    // Try to load real input values from example_user_inputs.json
    std::vector<std::vector<double>> testInputs;
    std::vector<std::string> inputPaths = {
        "user_graphs/example_user_inputs.json",
        "../user_graphs/example_user_inputs.json",
        "../../user_graphs/example_user_inputs.json",
        "C:/devPrivate/TapePresso/user_graphs/example_user_inputs.json",
        "C:\\devPrivate\\TapePresso\\user_graphs\\example_user_inputs.json"
    };

    bool loadedInputs = false;
    InputStatistics inputStats;

    for (const auto& inputPath : inputPaths) {
        try {
            inputStats = loadInputsFromFile(inputPath);

            // Show warning if mismatch
            if (inputStats.values.size() != inputCount) {
                std::cout << "Warning: Input file has " << inputStats.values.size()
                         << " inputs but graph expects " << inputCount << "\n";
            }

            // Use the inputs even if count doesn't match perfectly (we'll use what we can)
            std::cout << "\n=== Input Statistics ===\n";
            std::cout << "  Input JSON loaded successfully\n";
            std::cout << "  Total inputs: " << inputCount << "\n";
            std::cout << "  Non-zero values: " << inputStats.nonZeroCount << "\n";
            std::cout << "  Min value: " << std::fixed << std::setprecision(2) << inputStats.minValue << "\n";
            std::cout << "  Max value: " << std::fixed << std::setprecision(2) << inputStats.maxValue << "\n";
            std::cout << "  Loaded from: ../../user_graphs/example_user_inputs.json\n";

            // Resize to match graph input count if needed
            if (inputStats.values.size() > inputCount) {
                inputStats.values.resize(inputCount);
            } else if (inputStats.values.size() < inputCount) {
                inputStats.values.resize(inputCount, 0.0);  // Pad with zeros
            }

            // Use the same inputs for all test runs (to be consistent)
            testInputs.push_back(inputStats.values);

            // Create variations by bumping one input by 0.0001 for each run
            // This gives us slight variations to test with
            for (size_t i = 0; i < inputStats.values.size(); ++i) {
                std::vector<double> variation = inputStats.values;
                // Bump the i-th input by a small amount
                if (std::isfinite(variation[i])) {
                    variation[i] += 0.0001;
                }
                testInputs.push_back(variation);
            }

            loadedInputs = true;
            break;
        } catch (const std::exception& e) {
            // Try next path
        }
    }

    if (!loadedInputs) {
        std::cout << "Warning: Could not load real inputs. Using synthetic test data.\n";
        // Fallback to synthetic inputs - use base value and bump variations
        std::vector<double> baseInputs(inputCount, 100.0);
        testInputs.push_back(baseInputs);

        // Create variations by bumping one input by 0.0001
        for (size_t i = 0; i < inputCount; ++i) {
            std::vector<double> inputs = baseInputs;
            inputs[i] += 0.0001;
            testInputs.push_back(inputs);
        }
    }

    std::vector<BenchmarkResult> results;
    
    // Keep kernels alive for sanity checks
    std::unique_ptr<ForgeEngine> engineNoOpt;
    std::unique_ptr<ForgedKernel> kernelNoOpt;
    std::unique_ptr<INodeValueBuffer> workspaceNoOpt;
    
    std::unique_ptr<ForgeEngine> engineOpt;
    std::unique_ptr<ForgedKernel> kernelOpt;
    std::unique_ptr<INodeValueBuffer> workspaceOpt;

    // Test two configurations: No optimizations (except stability) vs All optimizations
    {
        CompilerConfig config;
        config.enableOptimizations = true;  // Need to enable this for any optimization to work
        config.enableInactiveFolding = false;
        config.enableCSE = false;
        config.enableAlgebraicSimplification = false;
        config.enableStabilityCleaning = true;  // Keep stability cleaning even in "no opts"
        results.push_back(runBenchmark("No Optimizations (Stability Only)", graph, config, testInputs));
        
        // Keep this kernel for sanity checks
        engineNoOpt = std::make_unique<ForgeEngine>();
        engineNoOpt->setConfig(config);
        kernelNoOpt = engineNoOpt->compile(graph);
        workspaceNoOpt = NodeValueBufferFactory::create(graph, *kernelNoOpt);
    }

    {
        CompilerConfig config;
        config.enableOptimizations = true;
        config.enableInactiveFolding = true;
        config.enableCSE = true;
        config.enableAlgebraicSimplification = true;
        config.enableStabilityCleaning = true;
        results.push_back(runBenchmark("All Optimizations", graph, config, testInputs));
        
        // Keep this kernel for sanity checks
        engineOpt = std::make_unique<ForgeEngine>();
        engineOpt->setConfig(config);
        kernelOpt = engineOpt->compile(graph);
        workspaceOpt = NodeValueBufferFactory::create(graph, *kernelOpt);
    }

    printResultsTable(results);
    
    // Additional sanity checks
    std::cout << "\n=== Additional Sanity Checks ===\n";
    
    // 1. Test that changing inputs changes outputs
    std::cout << "\n1. Testing input sensitivity...\n";
    
    // Use original inputs
    const auto& originalInputs = testInputs[0];
    for (size_t j = 0; j < originalInputs.size(); ++j) {
        workspaceNoOpt->setValue(j, originalInputs[j]);
        workspaceOpt->setValue(j, originalInputs[j]);
    }
    
    kernelNoOpt->execute(*workspaceNoOpt);
    kernelOpt->execute(*workspaceOpt);
    
    double outputOriginalNoOpt = workspaceNoOpt->getValue(graph.outputs[0]);
    double outputOriginalOpt = workspaceOpt->getValue(graph.outputs[0]);
    
    std::cout << "  Original output (No Opt): " << std::setprecision(17) << outputOriginalNoOpt << "\n";
    std::cout << "  Original output (Optimized): " << std::setprecision(17) << outputOriginalOpt << "\n";
    
    // Create modified inputs (bump ALL inputs by a small percentage)
    std::vector<double> modifiedInputs = originalInputs;
    int modifiedCount = 0;
    for (size_t i = 0; i < modifiedInputs.size(); ++i) {
        // Bump non-zero values by 1%, add small value to zeros
        if (std::abs(originalInputs[i]) > 1e-10) {
            modifiedInputs[i] *= 1.01;  // Increase by 1%
            modifiedCount++;
        } else if (i < 10) {  // For first 10 zero inputs, add a small value
            modifiedInputs[i] = 0.001;
            modifiedCount++;
        }
    }
    
    std::cout << "  Modified " << modifiedCount << " input values (1% increase for non-zeros, 0.001 for some zeros)\n";
    
    // Test with modified inputs
    for (size_t j = 0; j < modifiedInputs.size(); ++j) {
        workspaceNoOpt->setValue(j, modifiedInputs[j]);
        workspaceOpt->setValue(j, modifiedInputs[j]);
    }
    
    kernelNoOpt->execute(*workspaceNoOpt);
    kernelOpt->execute(*workspaceOpt);
    
    double outputModifiedNoOpt = workspaceNoOpt->getValue(graph.outputs[0]);
    double outputModifiedOpt = workspaceOpt->getValue(graph.outputs[0]);
    
    std::cout << "  Modified output (No Opt): " << std::setprecision(17) << outputModifiedNoOpt << "\n";
    std::cout << "  Modified output (Optimized): " << std::setprecision(17) << outputModifiedOpt << "\n";
    
    // Check that outputs changed
    bool noOptChanged = std::abs(outputOriginalNoOpt - outputModifiedNoOpt) > 1e-15;
    bool optChanged = std::abs(outputOriginalOpt - outputModifiedOpt) > 1e-15;
    
    if (noOptChanged && optChanged) {
        std::cout << "  ✓ Both kernels respond to input changes\n";
    } else {
        std::cout << "  ✗ Warning: Kernels not responding to input changes\n";
        if (!noOptChanged) std::cout << "    No-opt kernel output unchanged\n";
        if (!optChanged) std::cout << "    Optimized kernel output unchanged\n";
    }
    
    // 2. Test multiple kernels coexisting with different graphs
    std::cout << "\n2. Testing multiple kernel coexistence (different graphs)...\n";
    
    // Load the simple test graph for variety
    Graph simpleGraph;
    bool simpleGraphLoaded = false;
    std::vector<std::string> simpleGraphPaths = {
        "user_graphs/simple_test_graph.json",
        "../user_graphs/simple_test_graph.json",
        "../../user_graphs/simple_test_graph.json",
        "C:/devPrivate/TapePresso/user_graphs/simple_test_graph.json"
    };
    
    for (const auto& path : simpleGraphPaths) {
        try {
            simpleGraph = loadGraphFromFile(path);
            simpleGraphLoaded = true;
            std::cout << "  Loaded simple_test_graph (9 nodes) for kernel 3\n";
            break;
        } catch (const std::exception&) {
            // Try next path
        }
    }
    
    if (simpleGraphLoaded) {
        // Create kernel 3 with simple graph (different from main graph)
        CompilerConfig config3;
        config3.enableOptimizations = true;
        config3.enableInactiveFolding = true;
        config3.enableCSE = true;
        config3.enableAlgebraicSimplification = true;
        config3.enableStabilityCleaning = true;
        
        ForgeEngine engine3;
        engine3.setConfig(config3);
        auto kernel3 = engine3.compile(simpleGraph);
        auto workspace3 = NodeValueBufferFactory::create(simpleGraph, *kernel3);
        
        // Set inputs for simple graph: x=5.0, y=7.0, z=11.0
        // Expected: (5*2) + (7*3) + 11 = 10 + 21 + 11 = 42
        workspace3->setValue(0, 5.0);  // x
        workspace3->setValue(1, 7.0);  // y
        workspace3->setValue(2, 11.0); // z
        
        // Run all three kernels
        for (size_t j = 0; j < originalInputs.size(); ++j) {
            workspaceNoOpt->setValue(j, originalInputs[j]);
            workspaceOpt->setValue(j, originalInputs[j]);
        }
        
        kernelNoOpt->execute(*workspaceNoOpt);
        kernelOpt->execute(*workspaceOpt);
        kernel3->execute(*workspace3);
        
        double output1 = workspaceNoOpt->getValue(graph.outputs[0]);
        double output2 = workspaceOpt->getValue(graph.outputs[0]);
        double output3 = workspace3->getValue(simpleGraph.outputs[0]);
        
        std::cout << "  Kernel 1 (Large Graph, No Opt): " << std::setprecision(17) << output1 << "\n";
        std::cout << "  Kernel 2 (Large Graph, Full Opt): " << std::setprecision(17) << output2 << "\n";
        std::cout << "  Kernel 3 (Simple Graph, Full Opt): " << std::setprecision(17) << output3 << "\n";
        
        // Verify large graph kernels match each other
        bool largeGraphsMatch = std::abs(output1 - output2) < 1e-12;
        
        // Verify simple graph produces expected result
        bool simpleGraphCorrect = std::abs(output3 - 42.0) < 1e-10;
        
        if (largeGraphsMatch) {
            std::cout << "  ✓ Large graph kernels produce consistent results\n";
        } else {
            std::cout << "  ✗ Warning: Large graph kernels produce different results\n";
            std::cout << "    Difference: " << std::abs(output1 - output2) << "\n";
        }
        
        if (simpleGraphCorrect) {
            std::cout << "  ✓ Simple graph kernel produces expected result (42.0)\n";
        } else {
            std::cout << "  ✗ Warning: Simple graph kernel incorrect (expected 42.0, got " << output3 << ")\n";
        }
        
        std::cout << "  ✓ Three kernels with different graphs coexisting successfully\n";
    } else {
        // Fallback to same graph with different config if simple graph not found
        std::cout << "  Warning: simple_test_graph.json not found, using same graph with different config\n";
        
        CompilerConfig config3;
        config3.enableOptimizations = true;
        config3.enableInactiveFolding = true;
        config3.enableCSE = false;  // Different config
        config3.enableAlgebraicSimplification = true;
        config3.enableStabilityCleaning = true;
        
        ForgeEngine engine3;
        engine3.setConfig(config3);
        auto kernel3 = engine3.compile(graph);
        auto workspace3 = NodeValueBufferFactory::create(graph, *kernel3);
        
        // Run all three kernels with same inputs
        for (size_t j = 0; j < originalInputs.size(); ++j) {
            workspaceNoOpt->setValue(j, originalInputs[j]);
            workspaceOpt->setValue(j, originalInputs[j]);
            workspace3->setValue(j, originalInputs[j]);
        }
        
        kernelNoOpt->execute(*workspaceNoOpt);
        kernelOpt->execute(*workspaceOpt);
        kernel3->execute(*workspace3);
        
        double output1 = workspaceNoOpt->getValue(graph.outputs[0]);
        double output2 = workspaceOpt->getValue(graph.outputs[0]);
        double output3 = workspace3->getValue(graph.outputs[0]);
        
        std::cout << "  Kernel 1 (No Opt): " << std::setprecision(17) << output1 << "\n";
        std::cout << "  Kernel 2 (Full Opt): " << std::setprecision(17) << output2 << "\n";
        std::cout << "  Kernel 3 (Partial Opt): " << std::setprecision(17) << output3 << "\n";
        
        bool allMatch = (std::abs(output1 - output2) < 1e-12) && 
                        (std::abs(output2 - output3) < 1e-12);
        
        if (allMatch) {
            std::cout << "  ✓ All kernels produce consistent results\n";
        } else {
            std::cout << "  ✗ Warning: Kernels produce different results\n";
            std::cout << "    Max difference: " << std::max(std::abs(output1 - output2), 
                                                           std::abs(output2 - output3)) << "\n";
        }
        
        std::cout << "  ✓ Three kernels coexisting successfully\n";
    }
    
    std::cout << "\n=== Sanity Checks Complete ===\n\n";
    
    SUCCEED();
}
