#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include "../src/graph/graph.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/compiler_config.hpp"
#include "../src/compiler/node_value_buffers/scalar_node_value_buffer.hpp"
#include "../src/compiler/node_value_buffers/avx2_node_value_buffer.hpp"
#include "test_graphs.hpp"

using namespace forge;
using namespace forge_tests;

// Helper to compare doubles with tolerance
static bool approxEqual(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

// Helper to format a compact result summary for PASS output
static std::string formatPassSummary(size_t numInputs, size_t numOutputs, size_t numInputSets, bool withGradient) {
    std::ostringstream oss;
    oss << numInputs << " inputs, " << numOutputs << " outputs, " << numInputSets << " input sets | ";
    oss << numInputSets << "/" << numInputSets << " results";
    if (withGradient) {
        oss << ", " << numInputSets << "/" << numInputSets << " gradients";
    }
    return oss.str();
}

// ============================================================================
// Scalar tests (SSE2)
// ============================================================================

TEST(ForgeEngineTest, CompileAndExecuteSimpleGraph) {
    int passed = 0, failed = 0;
    std::vector<std::string> failures;

    for (auto tg : createTestGraphs()) {
        try {
            ForgeEngine engine(CompilerConfig::Default());
            auto kernel = engine.compile(tg.graph);

            if (!kernel) {
                std::cout << "  [FAIL] " << tg.name << ": kernel is null" << std::endl;
                failures.push_back(tg.name + ": kernel is null");
                failed++;
                continue;
            }

            ScalarNodeValueBuffer buffer(tg.graph);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                buffer.setValue(tg.inputId, tc.input);
                kernel->execute(buffer);
                double result = buffer.getValue(tg.outputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] " << tg.name << ": input=" << tc.input
                              << ", got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
            }

            if (graphPassed) {
                std::cout << "  [PASS] " << tg.name << " | " << formatPassSummary(tg.numInputs, tg.numOutputs, tg.testCases.size(), false) << std::endl;
                passed++;
            } else {
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "  [CRASH] " << tg.name << ": " << e.what() << std::endl;
            failures.push_back(tg.name + ": " + e.what());
            failed++;
        } catch (...) {
            std::cout << "  [CRASH] " << tg.name << ": unknown exception" << std::endl;
            failures.push_back(tg.name + ": unknown exception");
            failed++;
        }
    }

    std::cout << "\n  Summary: " << passed << " passed, " << failed << " failed" << std::endl;
    EXPECT_EQ(failed, 0) << "Some graphs failed";
}

TEST(ForgeEngineTest, CompileAndExecuteWithGradient) {
    int passed = 0, failed = 0;
    std::vector<std::string> failures;

    for (auto tg : createTestGraphsWithGradient()) {
        try {
            ForgeEngine engine(CompilerConfig::Default());
            auto kernel = engine.compile(tg.graph);

            if (!kernel) {
                std::cout << "  [FAIL] " << tg.name << ": kernel is null" << std::endl;
                failures.push_back(tg.name + ": kernel is null");
                failed++;
                continue;
            }

            ScalarNodeValueBuffer buffer(tg.graph);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                buffer.setValue(tg.inputId, tc.input);
                buffer.clearGradients();
                kernel->execute(buffer);
                double result = buffer.getValue(tg.outputId);
                double gradient = buffer.getGradient(tg.inputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] " << tg.name << ": input=" << tc.input
                              << ", result got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
                if (!approxEqual(gradient, tc.expectedGradient)) {
                    std::cout << "  [FAIL] " << tg.name << ": input=" << tc.input
                              << ", gradient got=" << gradient << ", expected=" << tc.expectedGradient << std::endl;
                    failures.push_back(tg.name + ": wrong gradient for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
            }

            if (graphPassed) {
                std::cout << "  [PASS] " << tg.name << " | " << formatPassSummary(tg.numInputs, tg.numOutputs, tg.testCases.size(), true) << std::endl;
                passed++;
            } else {
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "  [CRASH] " << tg.name << ": " << e.what() << std::endl;
            failures.push_back(tg.name + ": " + e.what());
            failed++;
        } catch (...) {
            std::cout << "  [CRASH] " << tg.name << ": unknown exception" << std::endl;
            failures.push_back(tg.name + ": unknown exception");
            failed++;
        }
    }

    std::cout << "\n  Summary: " << passed << " passed, " << failed << " failed" << std::endl;
    EXPECT_EQ(failed, 0) << "Some graphs failed";
}

// ============================================================================
// AVX2 tests
// ============================================================================

TEST(ForgeEngineTestAVX2, CompileAndExecuteSimpleGraph) {
    int passed = 0, failed = 0;
    std::vector<std::string> failures;

    for (auto tg : createTestGraphs()) {
        try {
            CompilerConfig config = CompilerConfig::Default();
            config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;
            ForgeEngine engine(config);
            auto kernel = engine.compile(tg.graph);

            if (!kernel) {
                std::cout << "  [FAIL] " << tg.name << ": kernel is null" << std::endl;
                failures.push_back(tg.name + ": kernel is null");
                failed++;
                continue;
            }

            AVX2NodeValueBuffer buffer(tg.graph);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                buffer.setValue(tg.inputId, tc.input);
                kernel->execute(buffer);
                double result = buffer.getValue(tg.outputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [AVX2] " << tg.name << ": input=" << tc.input
                              << ", got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
            }

            if (graphPassed) {
                std::cout << "  [PASS] [AVX2] " << tg.name << " | " << formatPassSummary(tg.numInputs, tg.numOutputs, tg.testCases.size(), false) << std::endl;
                passed++;
            } else {
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "  [CRASH] [AVX2] " << tg.name << ": " << e.what() << std::endl;
            failures.push_back(tg.name + ": " + e.what());
            failed++;
        } catch (...) {
            std::cout << "  [CRASH] [AVX2] " << tg.name << ": unknown exception" << std::endl;
            failures.push_back(tg.name + ": unknown exception");
            failed++;
        }
    }

    std::cout << "\n  Summary: " << passed << " passed, " << failed << " failed" << std::endl;
    EXPECT_EQ(failed, 0) << "Some graphs failed";
}

TEST(ForgeEngineTestAVX2, CompileAndExecuteWithGradient) {
    int passed = 0, failed = 0;
    std::vector<std::string> failures;

    for (auto tg : createTestGraphsWithGradient()) {
        try {
            CompilerConfig config = CompilerConfig::Default();
            config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;
            ForgeEngine engine(config);
            auto kernel = engine.compile(tg.graph);

            if (!kernel) {
                std::cout << "  [FAIL] " << tg.name << ": kernel is null" << std::endl;
                failures.push_back(tg.name + ": kernel is null");
                failed++;
                continue;
            }

            AVX2NodeValueBuffer buffer(tg.graph);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                buffer.setValue(tg.inputId, tc.input);
                buffer.clearGradients();
                kernel->execute(buffer);
                double result = buffer.getValue(tg.outputId);
                double gradient = buffer.getGradient(tg.inputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [AVX2] " << tg.name << ": input=" << tc.input
                              << ", result got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
                if (!approxEqual(gradient, tc.expectedGradient)) {
                    std::cout << "  [FAIL] [AVX2] " << tg.name << ": input=" << tc.input
                              << ", gradient got=" << gradient << ", expected=" << tc.expectedGradient << std::endl;
                    failures.push_back(tg.name + ": wrong gradient for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
            }

            if (graphPassed) {
                std::cout << "  [PASS] [AVX2] " << tg.name << " | " << formatPassSummary(tg.numInputs, tg.numOutputs, tg.testCases.size(), true) << std::endl;
                passed++;
            } else {
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "  [CRASH] [AVX2] " << tg.name << ": " << e.what() << std::endl;
            failures.push_back(tg.name + ": " + e.what());
            failed++;
        } catch (...) {
            std::cout << "  [CRASH] [AVX2] " << tg.name << ": unknown exception" << std::endl;
            failures.push_back(tg.name + ": unknown exception");
            failed++;
        }
    }

    std::cout << "\n  Summary: " << passed << " passed, " << failed << " failed" << std::endl;
    EXPECT_EQ(failed, 0) << "Some graphs failed";
}

// ============================================================================
// Scalar with all optimizations enabled (CompilerConfig::Fast())
// ============================================================================

TEST(ForgeEngineTestOptimized, CompileAndExecuteSimpleGraph) {
    int passed = 0, failed = 0;
    std::vector<std::string> failures;

    for (auto tg : createTestGraphs()) {
        try {
            ForgeEngine engine(CompilerConfig::Fast());
            auto kernel = engine.compile(tg.graph);

            if (!kernel) {
                std::cout << "  [FAIL] " << tg.name << ": kernel is null" << std::endl;
                failures.push_back(tg.name + ": kernel is null");
                failed++;
                continue;
            }

            ScalarNodeValueBuffer buffer(tg.graph);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                buffer.setValue(tg.inputId, tc.input);
                kernel->execute(buffer);
                double result = buffer.getValue(tg.outputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [Opt] " << tg.name << ": input=" << tc.input
                              << ", got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
            }

            if (graphPassed) {
                std::cout << "  [PASS] [Opt] " << tg.name << " | " << formatPassSummary(tg.numInputs, tg.numOutputs, tg.testCases.size(), false) << std::endl;
                passed++;
            } else {
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "  [CRASH] [Opt] " << tg.name << ": " << e.what() << std::endl;
            failures.push_back(tg.name + ": " + e.what());
            failed++;
        } catch (...) {
            std::cout << "  [CRASH] [Opt] " << tg.name << ": unknown exception" << std::endl;
            failures.push_back(tg.name + ": unknown exception");
            failed++;
        }
    }

    std::cout << "\n  Summary: " << passed << " passed, " << failed << " failed" << std::endl;
    EXPECT_EQ(failed, 0) << "Some graphs failed";
}

TEST(ForgeEngineTestOptimized, CompileAndExecuteWithGradient) {
    int passed = 0, failed = 0;
    std::vector<std::string> failures;

    for (auto tg : createTestGraphsWithGradient()) {
        try {
            ForgeEngine engine(CompilerConfig::Fast());
            auto kernel = engine.compile(tg.graph);

            if (!kernel) {
                std::cout << "  [FAIL] " << tg.name << ": kernel is null" << std::endl;
                failures.push_back(tg.name + ": kernel is null");
                failed++;
                continue;
            }

            ScalarNodeValueBuffer buffer(tg.graph);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                buffer.setValue(tg.inputId, tc.input);
                buffer.clearGradients();
                kernel->execute(buffer);
                double result = buffer.getValue(tg.outputId);
                double gradient = buffer.getGradient(tg.inputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [Opt] " << tg.name << ": input=" << tc.input
                              << ", result got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
                if (!approxEqual(gradient, tc.expectedGradient)) {
                    std::cout << "  [FAIL] [Opt] " << tg.name << ": input=" << tc.input
                              << ", gradient got=" << gradient << ", expected=" << tc.expectedGradient << std::endl;
                    failures.push_back(tg.name + ": wrong gradient for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
            }

            if (graphPassed) {
                std::cout << "  [PASS] [Opt] " << tg.name << " | " << formatPassSummary(tg.numInputs, tg.numOutputs, tg.testCases.size(), true) << std::endl;
                passed++;
            } else {
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "  [CRASH] [Opt] " << tg.name << ": " << e.what() << std::endl;
            failures.push_back(tg.name + ": " + e.what());
            failed++;
        } catch (...) {
            std::cout << "  [CRASH] [Opt] " << tg.name << ": unknown exception" << std::endl;
            failures.push_back(tg.name + ": unknown exception");
            failed++;
        }
    }

    std::cout << "\n  Summary: " << passed << " passed, " << failed << " failed" << std::endl;
    EXPECT_EQ(failed, 0) << "Some graphs failed";
}

// ============================================================================
// AVX2 with all optimizations enabled (CompilerConfig::Fast() + AVX2)
// ============================================================================

TEST(ForgeEngineTestAVX2Optimized, CompileAndExecuteSimpleGraph) {
    int passed = 0, failed = 0;
    std::vector<std::string> failures;

    for (auto tg : createTestGraphs()) {
        try {
            CompilerConfig config = CompilerConfig::Fast();
            config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;
            ForgeEngine engine(config);
            auto kernel = engine.compile(tg.graph);

            if (!kernel) {
                std::cout << "  [FAIL] " << tg.name << ": kernel is null" << std::endl;
                failures.push_back(tg.name + ": kernel is null");
                failed++;
                continue;
            }

            AVX2NodeValueBuffer buffer(tg.graph);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                buffer.setValue(tg.inputId, tc.input);
                kernel->execute(buffer);
                double result = buffer.getValue(tg.outputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [AVX2+Opt] " << tg.name << ": input=" << tc.input
                              << ", got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
            }

            if (graphPassed) {
                std::cout << "  [PASS] [AVX2+Opt] " << tg.name << " | " << formatPassSummary(tg.numInputs, tg.numOutputs, tg.testCases.size(), false) << std::endl;
                passed++;
            } else {
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "  [CRASH] [AVX2+Opt] " << tg.name << ": " << e.what() << std::endl;
            failures.push_back(tg.name + ": " + e.what());
            failed++;
        } catch (...) {
            std::cout << "  [CRASH] [AVX2+Opt] " << tg.name << ": unknown exception" << std::endl;
            failures.push_back(tg.name + ": unknown exception");
            failed++;
        }
    }

    std::cout << "\n  Summary: " << passed << " passed, " << failed << " failed" << std::endl;
    EXPECT_EQ(failed, 0) << "Some graphs failed";
}

TEST(ForgeEngineTestAVX2Optimized, CompileAndExecuteWithGradient) {
    int passed = 0, failed = 0;
    std::vector<std::string> failures;

    for (auto tg : createTestGraphsWithGradient()) {
        try {
            CompilerConfig config = CompilerConfig::Fast();
            config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;
            ForgeEngine engine(config);
            auto kernel = engine.compile(tg.graph);

            if (!kernel) {
                std::cout << "  [FAIL] " << tg.name << ": kernel is null" << std::endl;
                failures.push_back(tg.name + ": kernel is null");
                failed++;
                continue;
            }

            AVX2NodeValueBuffer buffer(tg.graph);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                buffer.setValue(tg.inputId, tc.input);
                buffer.clearGradients();
                kernel->execute(buffer);
                double result = buffer.getValue(tg.outputId);
                double gradient = buffer.getGradient(tg.inputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [AVX2+Opt] " << tg.name << ": input=" << tc.input
                              << ", result got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
                if (!approxEqual(gradient, tc.expectedGradient)) {
                    std::cout << "  [FAIL] [AVX2+Opt] " << tg.name << ": input=" << tc.input
                              << ", gradient got=" << gradient << ", expected=" << tc.expectedGradient << std::endl;
                    failures.push_back(tg.name + ": wrong gradient for input " + std::to_string(tc.input));
                    graphPassed = false;
                    break;
                }
            }

            if (graphPassed) {
                std::cout << "  [PASS] [AVX2+Opt] " << tg.name << " | " << formatPassSummary(tg.numInputs, tg.numOutputs, tg.testCases.size(), true) << std::endl;
                passed++;
            } else {
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "  [CRASH] [AVX2+Opt] " << tg.name << ": " << e.what() << std::endl;
            failures.push_back(tg.name + ": " + e.what());
            failed++;
        } catch (...) {
            std::cout << "  [CRASH] [AVX2+Opt] " << tg.name << ": unknown exception" << std::endl;
            failures.push_back(tg.name + ": unknown exception");
            failed++;
        }
    }

    std::cout << "\n  Summary: " << passed << " passed, " << failed << " failed" << std::endl;
    EXPECT_EQ(failed, 0) << "Some graphs failed";
}
