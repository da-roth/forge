#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include "../src/graph/graph.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/compiler_config.hpp"
#include "../src/compiler/node_value_buffers/node_value_buffer.hpp"
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

// Helper to format inputs vector as string
static std::string formatInputs(const std::vector<double>& inputs) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << inputs[i];
    }
    oss << ")";
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

            auto buffer = NodeValueBufferFactory::create(tg.graph, *kernel);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                // Set all input values
                for (size_t i = 0; i < tg.inputIds.size(); ++i) {
                    buffer->setValue(tg.inputIds[i], tc.inputs[i]);
                }
                kernel->execute(*buffer);
                double result = buffer->getValue(tg.outputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for inputs " + formatInputs(tc.inputs));
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

            auto buffer = NodeValueBufferFactory::create(tg.graph, *kernel);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                // Set all input values
                for (size_t i = 0; i < tg.inputIds.size(); ++i) {
                    buffer->setValue(tg.inputIds[i], tc.inputs[i]);
                }
                buffer->clearGradients();
                kernel->execute(*buffer);
                double result = buffer->getValue(tg.outputId);
                double gradient = buffer->getGradient(tg.inputIds[0]); // Gradient w.r.t. first input (x)

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", result got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for inputs " + formatInputs(tc.inputs));
                    graphPassed = false;
                    break;
                }
                if (!approxEqual(gradient, tc.expectedGradient)) {
                    std::cout << "  [FAIL] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", gradient got=" << gradient << ", expected=" << tc.expectedGradient << std::endl;
                    failures.push_back(tg.name + ": wrong gradient for inputs " + formatInputs(tc.inputs));
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

            auto buffer = NodeValueBufferFactory::create(tg.graph, *kernel);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                // Set all input values
                for (size_t i = 0; i < tg.inputIds.size(); ++i) {
                    buffer->setValue(tg.inputIds[i], tc.inputs[i]);
                }
                kernel->execute(*buffer);
                double result = buffer->getValue(tg.outputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [AVX2] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for inputs " + formatInputs(tc.inputs));
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

            auto buffer = NodeValueBufferFactory::create(tg.graph, *kernel);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                // Set all input values
                for (size_t i = 0; i < tg.inputIds.size(); ++i) {
                    buffer->setValue(tg.inputIds[i], tc.inputs[i]);
                }
                buffer->clearGradients();
                kernel->execute(*buffer);
                double result = buffer->getValue(tg.outputId);
                double gradient = buffer->getGradient(tg.inputIds[0]); // Gradient w.r.t. first input (x)

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [AVX2] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", result got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for inputs " + formatInputs(tc.inputs));
                    graphPassed = false;
                    break;
                }
                if (!approxEqual(gradient, tc.expectedGradient)) {
                    std::cout << "  [FAIL] [AVX2] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", gradient got=" << gradient << ", expected=" << tc.expectedGradient << std::endl;
                    failures.push_back(tg.name + ": wrong gradient for inputs " + formatInputs(tc.inputs));
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

            auto buffer = NodeValueBufferFactory::create(tg.graph, *kernel);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                // Set all input values
                for (size_t i = 0; i < tg.inputIds.size(); ++i) {
                    buffer->setValue(tg.inputIds[i], tc.inputs[i]);
                }
                kernel->execute(*buffer);
                double result = buffer->getValue(tg.outputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [Opt] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for inputs " + formatInputs(tc.inputs));
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

            auto buffer = NodeValueBufferFactory::create(tg.graph, *kernel);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                // Set all input values
                for (size_t i = 0; i < tg.inputIds.size(); ++i) {
                    buffer->setValue(tg.inputIds[i], tc.inputs[i]);
                }
                buffer->clearGradients();
                kernel->execute(*buffer);
                double result = buffer->getValue(tg.outputId);
                double gradient = buffer->getGradient(tg.inputIds[0]); // Gradient w.r.t. first input (x)

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [Opt] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", result got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for inputs " + formatInputs(tc.inputs));
                    graphPassed = false;
                    break;
                }
                if (!approxEqual(gradient, tc.expectedGradient)) {
                    std::cout << "  [FAIL] [Opt] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", gradient got=" << gradient << ", expected=" << tc.expectedGradient << std::endl;
                    failures.push_back(tg.name + ": wrong gradient for inputs " + formatInputs(tc.inputs));
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

            auto buffer = NodeValueBufferFactory::create(tg.graph, *kernel);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                // Set all input values
                for (size_t i = 0; i < tg.inputIds.size(); ++i) {
                    buffer->setValue(tg.inputIds[i], tc.inputs[i]);
                }
                kernel->execute(*buffer);
                double result = buffer->getValue(tg.outputId);

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [AVX2+Opt] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for inputs " + formatInputs(tc.inputs));
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

            auto buffer = NodeValueBufferFactory::create(tg.graph, *kernel);
            bool graphPassed = true;

            for (const auto& tc : tg.testCases) {
                // Set all input values
                for (size_t i = 0; i < tg.inputIds.size(); ++i) {
                    buffer->setValue(tg.inputIds[i], tc.inputs[i]);
                }
                buffer->clearGradients();
                kernel->execute(*buffer);
                double result = buffer->getValue(tg.outputId);
                double gradient = buffer->getGradient(tg.inputIds[0]); // Gradient w.r.t. first input (x)

                if (!approxEqual(result, tc.expectedOutput)) {
                    std::cout << "  [FAIL] [AVX2+Opt] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", result got=" << result << ", expected=" << tc.expectedOutput << std::endl;
                    failures.push_back(tg.name + ": wrong result for inputs " + formatInputs(tc.inputs));
                    graphPassed = false;
                    break;
                }
                if (!approxEqual(gradient, tc.expectedGradient)) {
                    std::cout << "  [FAIL] [AVX2+Opt] " << tg.name << ": inputs=" << formatInputs(tc.inputs)
                              << ", gradient got=" << gradient << ", expected=" << tc.expectedGradient << std::endl;
                    failures.push_back(tg.name + ": wrong gradient for inputs " + formatInputs(tc.inputs));
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

// ============================================================================
// Register Allocator Unit Tests
// ============================================================================

#include "../src/compiler/xmm_register_allocator.hpp"
#include "../src/compiler/ymm_register_allocator.hpp"

// Test that exercises the LRU eviction path in allocateAvoiding()
// This happens when all registers are occupied and we need to evict one
TEST(RegisterAllocatorTest, LRUEvictionPath) {
    forge::XmmRegisterAllocator alloc;

    // Fill all 16 registers with nodes
    for (int i = 0; i < 16; ++i) {
        int reg = alloc.allocateAvoiding({});
        ASSERT_GE(reg, 0) << "Failed to allocate register " << i;
        alloc.setRegister(reg, static_cast<forge::NodeId>(i), false);
    }

    // Now all registers are occupied. The next allocation should trigger LRU eviction.
    // Allocate avoiding registers 0-5 to force eviction from 6-15
    int evictedReg = alloc.allocateAvoiding({0, 1, 2, 3, 4, 5});
    ASSERT_GE(evictedReg, 6) << "Should have evicted a register >= 6";
    ASSERT_LE(evictedReg, 15) << "Should have evicted a register <= 15";

    // The evicted register should now be empty (contents cleared)
    EXPECT_EQ(alloc.findNodeInRegister(static_cast<forge::NodeId>(evictedReg)), -1)
        << "Evicted register should no longer contain the old node";
}

// Test YMM allocator with blacklisted registers
TEST(RegisterAllocatorTest, YmmBlacklistAndLRU) {
    forge::YmmRegisterAllocator alloc;

    // YMM14 and YMM15 are blacklisted in constructor
    // Fill registers 0-13 (14 registers, skipping blacklisted ones)
    std::vector<int> allocatedRegs;
    for (int i = 0; i < 14; ++i) {
        int reg = alloc.allocateAvoiding({});
        ASSERT_GE(reg, 0) << "Failed to allocate register " << i;
        ASSERT_NE(reg, 14) << "Should not allocate blacklisted YMM14";
        ASSERT_NE(reg, 15) << "Should not allocate blacklisted YMM15";
        alloc.setRegister(reg, static_cast<forge::NodeId>(i), false);
        allocatedRegs.push_back(reg);
    }

    // Now all non-blacklisted registers are full. Next allocation triggers LRU eviction.
    int evictedReg = alloc.allocateAvoiding({0, 1, 2});
    ASSERT_GE(evictedReg, 0);
    ASSERT_NE(evictedReg, 14) << "Should not allocate blacklisted YMM14";
    ASSERT_NE(evictedReg, 15) << "Should not allocate blacklisted YMM15";
}

// Test getNodeInRegister and findNodeInRegister
TEST(RegisterAllocatorTest, NodeTracking) {
    forge::XmmRegisterAllocator alloc;

    int reg = alloc.allocateAvoiding({});
    ASSERT_GE(reg, 0);

    // Initially empty
    EXPECT_EQ(alloc.getNodeInRegister(reg), -1);
    EXPECT_EQ(alloc.findNodeInRegister(42), -1);

    // Set a node
    alloc.setRegister(reg, 42, false);
    EXPECT_EQ(alloc.getNodeInRegister(reg), 42);
    EXPECT_EQ(alloc.findNodeInRegister(42), reg);

    // Out of bounds should return -1
    EXPECT_EQ(alloc.getNodeInRegister(-1), -1);
    EXPECT_EQ(alloc.getNodeInRegister(100), -1);
}

// Test dirty flag tracking
TEST(RegisterAllocatorTest, DirtyTracking) {
    forge::XmmRegisterAllocator alloc;

    int reg = alloc.allocateAvoiding({});
    ASSERT_GE(reg, 0);

    // Initially not dirty
    EXPECT_FALSE(alloc.isDirty(reg));

    // Set with dirty flag
    alloc.setRegister(reg, 1, true);
    EXPECT_TRUE(alloc.isDirty(reg));

    // Set without dirty flag
    alloc.setRegister(reg, 2, false);
    EXPECT_FALSE(alloc.isDirty(reg));

    // Out of bounds should return false
    EXPECT_FALSE(alloc.isDirty(-1));
    EXPECT_FALSE(alloc.isDirty(100));
}

// Test volatile register invalidation
TEST(RegisterAllocatorTest, VolatileInvalidation) {
    forge::XmmRegisterAllocator alloc;

    // Allocate and fill some registers
    for (int i = 0; i < 10; ++i) {
        int reg = alloc.allocateAvoiding({});
        alloc.setRegister(reg, static_cast<forge::NodeId>(i), true);
    }

    // Invalidate volatile registers (0-5 on Windows)
    alloc.invalidateVolatileRegisters();

    // Volatile registers should be empty
    for (int i = alloc.getFirstVolatileReg(); i <= alloc.getLastVolatileReg(); ++i) {
        EXPECT_EQ(alloc.getNodeInRegister(i), -1)
            << "Volatile register " << i << " should be empty after invalidation";
        EXPECT_FALSE(alloc.isDirty(i))
            << "Volatile register " << i << " should not be dirty after invalidation";
    }
}

// Test markDirty and markClean
TEST(RegisterAllocatorTest, MarkDirtyClean) {
    forge::XmmRegisterAllocator alloc;

    int reg = alloc.allocateAvoiding({});
    alloc.setRegister(reg, 1, false);

    EXPECT_FALSE(alloc.isDirty(reg));

    alloc.markDirty(reg);
    EXPECT_TRUE(alloc.isDirty(reg));

    alloc.markClean(reg);
    EXPECT_FALSE(alloc.isDirty(reg));

    // Out of bounds should be safe (no crash)
    alloc.markDirty(-1);
    alloc.markDirty(100);
    alloc.markClean(-1);
    alloc.markClean(100);
}

// Test getNumRegisters
TEST(RegisterAllocatorTest, NumRegisters) {
    forge::XmmRegisterAllocator xmmAlloc;
    forge::YmmRegisterAllocator ymmAlloc;

    EXPECT_EQ(xmmAlloc.getNumRegisters(), 16);
    EXPECT_EQ(ymmAlloc.getNumRegisters(), 16);
}
