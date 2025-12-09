#include <gtest/gtest.h>
#include "../tools/graphSerialization/graph_serialization.hpp"
#include "../src/graph/graph_optimizer.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/node_value_buffers/node_value_buffer.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>

using namespace forge;

// Simulate the bridge API structures and workflow
class BridgeWorkflowBenchmark : public ::testing::Test {
protected:
    struct BridgeSimulationResult {
        std::string configName;
        size_t graphNodes;
        size_t inputCount;

        // Bridge workflow times
        double deserializationMs;
        double kernelCreationMs;
        double workspaceCreationMs;

        // Per-execution times (averaged)
        double phase1_paramValidationNs;
        double phase2_syncFromCSharpNs;
        double phase3_prepareBufferNs;
        double phase4_executeKernelNs;
        double phase5_syncToCSharpNs;
        double totalExecutionNs;

        // Direct C++ comparison
        double directCppExecutionNs;
        double overheadFactor;
    };

    // Simulate SIMD-indexed array used by C#
    struct SimulatedCSharpArrays {
        std::vector<double> V;  // Values array (SIMD indexed: position = index * 4)
        std::vector<double> D;  // Derivatives array (SIMD indexed)

        SimulatedCSharpArrays(size_t maxIndex) {
            // C# allocates arrays large enough for SIMD indexing
            size_t arraySize = (maxIndex + 1) * 4;
            V.resize(arraySize, 0.0);
            D.resize(arraySize, 0.0);
        }

        void setValueAtExternalIndex(size_t externalIndex, double value) {
            size_t simdPos = externalIndex * 4;
            if (simdPos < V.size()) {
                V[simdPos] = value;
            }
        }

        double getValueAtExternalIndex(size_t externalIndex) const {
            size_t simdPos = externalIndex * 4;
            if (simdPos < V.size()) {
                return V[simdPos];
            }
            return 0.0;
        }
    };

    // Simulate what the bridge does during Proxy_Forward
    struct BridgeWorkspaceSimulator {
        std::unique_ptr<INodeValueBuffer> nativeBuffer;
        SimulatedCSharpArrays csharpArrays;

        // Mappings (simulating what's in KernelInfo)
        std::unordered_map<size_t, NodeId> externalToNode;
        std::unordered_map<NodeId, size_t> nodeToExternal;
        std::vector<NodeId> inputNodes;
        std::vector<NodeId> outputNodes;
        size_t maxExternalIndex;

        BridgeWorkspaceSimulator(const Graph& graph,
                                 const StitchedKernel& kernel,
                                 size_t maxExternal)
            : csharpArrays(maxExternal), maxExternalIndex(maxExternal) {

            // Create native buffer
            nativeBuffer = NodeValueBufferFactory::create(graph, kernel);

            // Build mappings (simulating what bridge does during graph construction)
            size_t externalIndex = 1;
            for (NodeId nodeId = 0; nodeId < graph.nodes.size(); ++nodeId) {
                const auto& node = graph.nodes[nodeId];
                if (node.op == OpCode::Input) {
                    externalToNode[externalIndex] = nodeId;
                    nodeToExternal[nodeId] = externalIndex;
                    inputNodes.push_back(nodeId);
                    externalIndex++;
                }
            }

            // Map outputs
            for (NodeId outNode : graph.outputs) {
                if (nodeToExternal.find(outNode) == nodeToExternal.end()) {
                    externalToNode[externalIndex] = outNode;
                    nodeToExternal[outNode] = externalIndex;
                    externalIndex++;
                }
                outputNodes.push_back(outNode);
            }
        }

        // Phase 2: Sync from C# SIMD arrays to native buffer
        void syncFromCSharp() {
            for (const auto& [extIdx, nodeId] : externalToNode) {
                double value = csharpArrays.getValueAtExternalIndex(extIdx);
                nativeBuffer->setValue(nodeId, value);
            }
        }

        // Phase 5: Sync from native buffer to C# SIMD arrays
        void syncToCSharp() {
            for (const auto& [nodeId, extIdx] : nodeToExternal) {
                double value = nativeBuffer->getValue(nodeId);
                csharpArrays.setValueAtExternalIndex(extIdx, value);
            }
        }
    };

    BridgeSimulationResult runBridgeSimulation(
        const std::string& configName,
        const std::string& graphPath,
        const CompilerConfig& config,
        const std::vector<std::vector<double>>& testInputs)
    {
        BridgeSimulationResult result;
        result.configName = configName;

        using Clock = std::chrono::high_resolution_clock;
        using DurationMs = std::chrono::duration<double, std::milli>;
        using DurationNs = std::chrono::duration<double, std::nano>;

        // ========================================================================
        // Step 1: Deserialize Graph (not benchmarked, as requested)
        // ========================================================================
        auto deserStart = Clock::now();
        Graph graph = loadGraphFromFile(graphPath);
        result.deserializationMs = DurationMs(Clock::now() - deserStart).count();

        result.graphNodes = graph.nodes.size();

        // Count inputs
        size_t inputCount = 0;
        for (const auto& node : graph.nodes) {
            if (node.op == OpCode::Input) inputCount++;
        }
        result.inputCount = inputCount;

        // ========================================================================
        // Step 2: Kernel Creation (simulates Proxy_CreateKernel)
        // ========================================================================
        auto kernelStart = Clock::now();

        ForgeEngine engine;
        engine.setConfig(config);
        auto kernel = engine.compile(graph);

        result.kernelCreationMs = DurationMs(Clock::now() - kernelStart).count();

        // ========================================================================
        // Step 3: Workspace Creation (simulates Proxy_CreateWorkspace)
        // ========================================================================
        auto wsStart = Clock::now();

        // Determine max external index (inputs + outputs)
        size_t maxExternalIndex = inputCount + graph.outputs.size();

        BridgeWorkspaceSimulator bridgeWs(graph, *kernel, maxExternalIndex);

        result.workspaceCreationMs = DurationMs(Clock::now() - wsStart).count();

        // ========================================================================
        // Step 4: Execution Benchmark (simulates Proxy_Forward)
        // ========================================================================
        const int warmupRuns = 5;
        const int benchmarkRuns = 10;

        // Warmup
        for (int i = 0; i < warmupRuns; ++i) {
            const auto& inputs = testInputs[i % testInputs.size()];

            // Set inputs in C# array (what C# does)
            for (size_t j = 0; j < inputs.size(); ++j) {
                bridgeWs.csharpArrays.setValueAtExternalIndex(j + 1, inputs[j]);
            }

            // Simulate bridge forward
            bridgeWs.syncFromCSharp();
            kernel->execute(*bridgeWs.nativeBuffer);
            bridgeWs.syncToCSharp();
        }

        // Detailed phase timing
        double total_phase1 = 0, total_phase2 = 0, total_phase3 = 0;
        double total_phase4 = 0, total_phase5 = 0, total_overall = 0;

        for (int i = 0; i < benchmarkRuns; ++i) {
            const auto& inputs = testInputs[i % testInputs.size()];

            auto overallStart = Clock::now();

            // Phase 1: Parameter validation (minimal in C++, but exists in bridge)
            auto p1Start = Clock::now();
            // Simulate pointer checks and map lookups
            volatile bool kernelValid = (kernel != nullptr);
            volatile bool wsValid = (bridgeWs.nativeBuffer != nullptr);
            total_phase1 += DurationNs(Clock::now() - p1Start).count();

            // Phase 2: Sync from C# array to native buffer
            auto p2Start = Clock::now();
            for (size_t j = 0; j < inputs.size(); ++j) {
                bridgeWs.csharpArrays.setValueAtExternalIndex(j + 1, inputs[j]);
            }
            bridgeWs.syncFromCSharp();
            total_phase2 += DurationNs(Clock::now() - p2Start).count();

            // Phase 3: Prepare execution buffer (already done in syncFromCSharp, but track separately)
            auto p3Start = Clock::now();
            // In real bridge, there's additional buffer preparation
            // Here it's minimal since we're already in native code
            total_phase3 += DurationNs(Clock::now() - p3Start).count();

            // Phase 4: Execute kernel
            auto p4Start = Clock::now();
            kernel->execute(*bridgeWs.nativeBuffer);
            total_phase4 += DurationNs(Clock::now() - p4Start).count();

            // Phase 5: Copy results back to C# array
            auto p5Start = Clock::now();
            bridgeWs.syncToCSharp();
            total_phase5 += DurationNs(Clock::now() - p5Start).count();

            total_overall += DurationNs(Clock::now() - overallStart).count();
        }

        result.phase1_paramValidationNs = total_phase1 / benchmarkRuns;
        result.phase2_syncFromCSharpNs = total_phase2 / benchmarkRuns;
        result.phase3_prepareBufferNs = total_phase3 / benchmarkRuns;
        result.phase4_executeKernelNs = total_phase4 / benchmarkRuns;
        result.phase5_syncToCSharpNs = total_phase5 / benchmarkRuns;
        result.totalExecutionNs = total_overall / benchmarkRuns;

        // ========================================================================
        // Step 5: Direct C++ Comparison (no bridge overhead)
        // ========================================================================
        auto directBuffer = NodeValueBufferFactory::create(graph, *kernel);

        // Pre-compute input node list (fair comparison - bridge does this once too)
        std::vector<NodeId> directInputNodes;
        for (NodeId nodeId = 0; nodeId < graph.nodes.size(); ++nodeId) {
            if (graph.nodes[nodeId].op == OpCode::Input) {
                directInputNodes.push_back(nodeId);
            }
        }

        // Warmup
        for (int i = 0; i < warmupRuns; ++i) {
            const auto& inputs = testInputs[i % testInputs.size()];
            for (size_t j = 0; j < directInputNodes.size() && j < inputs.size(); ++j) {
                directBuffer->setValue(directInputNodes[j], inputs[j]);
            }
            kernel->execute(*directBuffer);
        }

        // Benchmark direct execution
        auto directStart = Clock::now();
        for (int i = 0; i < benchmarkRuns; ++i) {
            const auto& inputs = testInputs[i % testInputs.size()];
            for (size_t j = 0; j < directInputNodes.size() && j < inputs.size(); ++j) {
                directBuffer->setValue(directInputNodes[j], inputs[j]);
            }
            kernel->execute(*directBuffer);
        }
        auto directEnd = Clock::now();

        result.directCppExecutionNs = DurationNs(directEnd - directStart).count() / benchmarkRuns;
        result.overheadFactor = result.totalExecutionNs / result.directCppExecutionNs;

        return result;
    }

    void printBenchmarkResults(const BridgeSimulationResult& result) {
        std::cout << "\n";
        std::cout << "==========================================================================================\n";
        std::cout << "                        BRIDGE WORKFLOW BENCHMARK RESULTS                                 \n";
        std::cout << "==========================================================================================\n";
        std::cout << "Configuration: " << result.configName << "\n";
        std::cout << "Graph Nodes: " << result.graphNodes << "\n";
        std::cout << "Input Count: " << result.inputCount << "\n";
        std::cout << "\n";

        // One-time setup costs
        std::cout << "Setup Times (one-time):\n";
        std::cout << "------------------------------------------------------------------------------------------\n";
        std::cout << std::left << std::setw(40) << "  Deserialization:"
                  << std::fixed << std::setprecision(3) << result.deserializationMs << " ms\n";
        std::cout << std::left << std::setw(40) << "  Kernel Creation (JIT compile):"
                  << std::fixed << std::setprecision(3) << result.kernelCreationMs << " ms\n";
        std::cout << std::left << std::setw(40) << "  Workspace Creation:"
                  << std::fixed << std::setprecision(3) << result.workspaceCreationMs << " ms\n";
        std::cout << "\n";

        // Per-execution costs (the bottleneck)
        std::cout << "Execution Times (per-call, averaged over 10000 runs):\n";
        std::cout << "------------------------------------------------------------------------------------------\n";
        std::cout << "Bridge Workflow Phases:\n";
        std::cout << std::left << std::setw(40) << "  Phase 1 - Parameter Validation:"
                  << std::fixed << std::setprecision(2) << result.phase1_paramValidationNs
                  << " ns (" << std::fixed << std::setprecision(6) << (result.phase1_paramValidationNs / 1e6) << " ms)\n";
        std::cout << std::left << std::setw(40) << "  Phase 2 - Sync from C# (SIMD copy):"
                  << std::fixed << std::setprecision(2) << result.phase2_syncFromCSharpNs
                  << " ns (" << std::fixed << std::setprecision(6) << (result.phase2_syncFromCSharpNs / 1e6) << " ms)\n";
        std::cout << std::left << std::setw(40) << "  Phase 3 - Prepare Buffer:"
                  << std::fixed << std::setprecision(2) << result.phase3_prepareBufferNs
                  << " ns (" << std::fixed << std::setprecision(6) << (result.phase3_prepareBufferNs / 1e6) << " ms)\n";
        std::cout << std::left << std::setw(40) << "  Phase 4 - Execute Kernel:"
                  << std::fixed << std::setprecision(2) << result.phase4_executeKernelNs
                  << " ns (" << std::fixed << std::setprecision(6) << (result.phase4_executeKernelNs / 1e6) << " ms)\n";
        std::cout << std::left << std::setw(40) << "  Phase 5 - Sync to C# (SIMD copy):"
                  << std::fixed << std::setprecision(2) << result.phase5_syncToCSharpNs
                  << " ns (" << std::fixed << std::setprecision(6) << (result.phase5_syncToCSharpNs / 1e6) << " ms)\n";
        std::cout << "------------------------------------------------------------------------------------------\n";
        std::cout << std::left << std::setw(40) << "  TOTAL (Bridge Workflow):"
                  << std::fixed << std::setprecision(2) << result.totalExecutionNs
                  << " ns (" << std::fixed << std::setprecision(6) << (result.totalExecutionNs / 1e6) << " ms)\n";
        std::cout << "\n";

        std::cout << "Direct C++ Execution (no bridge):\n";
        std::cout << std::left << std::setw(40) << "  Direct Kernel Execute:"
                  << std::fixed << std::setprecision(2) << result.directCppExecutionNs
                  << " ns (" << std::fixed << std::setprecision(6) << (result.directCppExecutionNs / 1e6) << " ms)\n";
        std::cout << "\n";

        std::cout << "Overhead Analysis:\n";
        std::cout << "------------------------------------------------------------------------------------------\n";
        double bridgeOverheadNs = result.totalExecutionNs - result.directCppExecutionNs;
        double phase2and5OverheadNs = result.phase2_syncFromCSharpNs + result.phase5_syncToCSharpNs;

        std::cout << std::left << std::setw(40) << "  Bridge Overhead:"
                  << std::fixed << std::setprecision(2) << bridgeOverheadNs
                  << " ns (" << std::fixed << std::setprecision(6) << (bridgeOverheadNs / 1e6) << " ms)\n";
        std::cout << std::left << std::setw(40) << "  SIMD Copy Overhead (Phase 2+5):"
                  << std::fixed << std::setprecision(2) << phase2and5OverheadNs
                  << " ns (" << std::fixed << std::setprecision(6) << (phase2and5OverheadNs / 1e6) << " ms)\n";
        std::cout << std::left << std::setw(40) << "  Overhead Factor:"
                  << std::fixed << std::setprecision(2) << result.overheadFactor << "x\n";
        std::cout << "\n";

        // Breakdown percentages
        std::cout << "Time Breakdown (% of total bridge execution):\n";
        std::cout << "------------------------------------------------------------------------------------------\n";
        std::cout << std::left << std::setw(40) << "  Parameter Validation:"
                  << std::fixed << std::setprecision(1) << (result.phase1_paramValidationNs / result.totalExecutionNs * 100) << "%\n";
        std::cout << std::left << std::setw(40) << "  Sync from C# (SIMD):"
                  << std::fixed << std::setprecision(1) << (result.phase2_syncFromCSharpNs / result.totalExecutionNs * 100) << "%\n";
        std::cout << std::left << std::setw(40) << "  Prepare Buffer:"
                  << std::fixed << std::setprecision(1) << (result.phase3_prepareBufferNs / result.totalExecutionNs * 100) << "%\n";
        std::cout << std::left << std::setw(40) << "  Kernel Execution:"
                  << std::fixed << std::setprecision(1) << (result.phase4_executeKernelNs / result.totalExecutionNs * 100) << "%\n";
        std::cout << std::left << std::setw(40) << "  Sync to C# (SIMD):"
                  << std::fixed << std::setprecision(1) << (result.phase5_syncToCSharpNs / result.totalExecutionNs * 100) << "%\n";

        std::cout << "==========================================================================================\n\n";
    }
};

TEST_F(BridgeWorkflowBenchmark, AnalyzeUserGraphBridgeOverhead) {
    // Try to find the user graph
    std::vector<std::string> possiblePaths = {
        "user_graphs/example_user_graph.json",
        "../user_graphs/example_user_graph.json",
        "../../user_graphs/example_user_graph.json",
        "C:/devPrivate/TapePresso/user_graphs/example_user_graph.json",
        "C:\\devPrivate\\TapePresso\\user_graphs\\example_user_graph.json"
    };

    std::string graphPath;
    bool found = false;

    for (const auto& path : possiblePaths) {
        std::ifstream test(path);
        if (test.good()) {
            graphPath = path;
            found = true;
            break;
        }
    }

    if (!found) {
        GTEST_SKIP() << "Could not find user graph. Please ensure user_graphs/example_user_graph.json exists.";
    }

    std::cout << "\n=== Bridge Workflow Benchmark: " << graphPath << " ===\n";

    // Generate test inputs (simple variation)
    std::vector<std::vector<double>> testInputs;
    for (int i = 0; i < 8; ++i) {
        // We don't know the exact input count yet, but allocate generously
        // The test will adjust based on actual graph inputs
        std::vector<double> inputs(1000);  // Generous allocation
        for (size_t j = 0; j < inputs.size(); ++j) {
            inputs[j] = 100.0 * (1.0 + 0.1 * i + 0.01 * j);
        }
        testInputs.push_back(inputs);
    }

    // Test with optimizations enabled (typical use case)
    CompilerConfig config;
    config.enableOptimizations = true;
    config.enableInactiveFolding = true;
    config.enableCSE = true;
    config.enableAlgebraicSimplification = true;
    config.enableStabilityCleaning = true;

    auto result = runBridgeSimulation("With Optimizations", graphPath, config, testInputs);
    printBenchmarkResults(result);

    SUCCEED();
}
