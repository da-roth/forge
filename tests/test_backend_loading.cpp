// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file test_backend_loading.cpp
 * @brief Tests for runtime backend loading
 *
 * These tests verify that backends can be loaded at runtime via loadBackend().
 * They are only meaningful when FORGE_BUNDLE_AVX2 is OFF and the backend
 * shared library is built separately.
 */

#include <gtest/gtest.h>
#include "../src/compiler/x86/common/instruction_set_factory.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/interfaces/node_value_buffer.hpp"
#include "../src/graph/graph.hpp"
#include "test_graphs.hpp"
#include <cstdlib>
#include <string>
#include <fstream>

using namespace forge;
using namespace forge_tests;

namespace {

// Get the path to the AVX2 backend library from environment variable
std::string getBackendPath() {
    const char* path = std::getenv("FORGE_AVX2_BACKEND_PATH");
    if (path) {
        return std::string(path);
    }
    // Default paths for CI
#ifdef _WIN32
    return "./forge_avx2.dll";
#else
    return "./libforge_avx2.so";
#endif
}

// Check if we should skip these tests (when AVX2 is bundled)
bool shouldSkipBackendLoadingTests() {
#ifdef FORGE_BUNDLE_AVX2
    return true;  // AVX2 is bundled, no need to test loading
#else
    return false;
#endif
}

// Check if a file exists
bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

} // anonymous namespace

// Test that we can check if AVX2 is available before loading
TEST(BackendLoadingTest, CheckAvailabilityBeforeLoad) {
    if (shouldSkipBackendLoadingTests()) {
        GTEST_SKIP() << "AVX2 is bundled, skipping runtime loading test";
    }

    // Before loading, AVX2-Packed should not be in the registry
    // (unless it was loaded by a previous test)
    // This test mainly verifies the hasInstructionSet API works
    bool hasAvx2 = forge::InstructionSetFactory::hasInstructionSet("AVX2-Packed");
    // We don't assert here because previous tests might have loaded it
    std::cout << "AVX2-Packed in registry before explicit load: "
              << (hasAvx2 ? "yes" : "no") << std::endl;
}

// Test loading the AVX2 backend
TEST(BackendLoadingTest, LoadAVX2Backend) {
    if (shouldSkipBackendLoadingTests()) {
        GTEST_SKIP() << "AVX2 is bundled, skipping runtime loading test";
    }

    std::string backendPath = getBackendPath();
    std::cout << "Attempting to load backend from: " << backendPath << std::endl;

    // Skip if the backend library doesn't exist (not built)
    if (!fileExists(backendPath)) {
        GTEST_SKIP() << "Backend library not found at: " << backendPath
                     << " (set FORGE_BUILD_AVX2_BACKEND=ON to build it)";
    }

    // Try to load the backend
    try {
        bool loaded = forge::InstructionSetFactory::loadBackend(backendPath);
        EXPECT_TRUE(loaded) << "Failed to load AVX2 backend from " << backendPath;

        // After loading, AVX2-Packed should be available
        EXPECT_TRUE(forge::InstructionSetFactory::hasInstructionSet("AVX2-Packed"))
            << "AVX2-Packed not registered after loading backend";

    } catch (const std::runtime_error& e) {
        FAIL() << "Exception while loading backend: " << e.what()
               << "\nBackend path: " << backendPath;
    }
}

// Test creating an instruction set after loading
TEST(BackendLoadingTest, CreateInstructionSetAfterLoad) {
    if (shouldSkipBackendLoadingTests()) {
        GTEST_SKIP() << "AVX2 is bundled, skipping runtime loading test";
    }

    // This test assumes LoadAVX2Backend ran first and succeeded
    if (!forge::InstructionSetFactory::hasInstructionSet("AVX2-Packed")) {
        GTEST_SKIP() << "AVX2 backend not loaded, skipping";
    }

    // Create instruction set by name
    auto instructionSet = forge::InstructionSetFactory::createByName("AVX2-Packed");
    ASSERT_NE(instructionSet, nullptr) << "createByName returned nullptr";

    EXPECT_EQ(instructionSet->getName(), "AVX2-Packed");
    EXPECT_EQ(instructionSet->getVectorWidth(), 4);
}

// Test compiling and running a simple kernel with loaded AVX2
TEST(BackendLoadingTest, CompileAndRunWithLoadedAVX2) {
    if (shouldSkipBackendLoadingTests()) {
        GTEST_SKIP() << "AVX2 is bundled, skipping runtime loading test";
    }

    // This test assumes LoadAVX2Backend ran first and succeeded
    if (!forge::InstructionSetFactory::hasInstructionSet("AVX2-Packed")) {
        GTEST_SKIP() << "AVX2 backend not loaded, skipping";
    }

    // Build a simple graph: f(x) = x * 2
    Graph graph;
    NodeId x = graph.addInput();
    NodeId two = graph.addConstant(2.0);
    NodeId result = addBinaryOp(graph, OpCode::Mul, x, two);
    graph.markOutput(result);

    // Compile with the loaded AVX2 backend
    forge::CompilerConfig config;
    config.useNamedInstructionSet = true;
    config.instructionSetName = "AVX2-Packed";

    forge::ForgeEngine engine(config);
    auto kernel = engine.compile(graph);

    ASSERT_NE(kernel, nullptr) << "Compilation failed";
    EXPECT_EQ(kernel->getVectorWidth(), 4) << "Expected AVX2 vector width of 4";
    EXPECT_EQ(kernel->getInstructionSetName(), "AVX2-Packed");

    // Create buffer and run
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);
    ASSERT_NE(buffer, nullptr) << "Buffer creation failed";

    // Set input value
    double inputValue = 3.5;
    buffer->setValue(x, inputValue);

    // Execute
    kernel->execute(*buffer);

    // Check result (x * 2 = 3.5 * 2 = 7.0)
    double outputValue = buffer->getValue(result);
    EXPECT_DOUBLE_EQ(outputValue, 7.0) << "Computation result incorrect";
}

// Test that SSE2-Scalar still works when AVX2 is not bundled
TEST(BackendLoadingTest, SSE2StillWorksWithoutBundledAVX2) {
    if (shouldSkipBackendLoadingTests()) {
        GTEST_SKIP() << "AVX2 is bundled, skipping runtime loading test";
    }

    // Build a simple graph: f(x) = x + 1
    Graph graph;
    NodeId x = graph.addInput();
    NodeId one = graph.addConstant(1.0);
    NodeId result = addBinaryOp(graph, OpCode::Add, x, one);
    graph.markOutput(result);

    // Compile with SSE2-Scalar (should always work)
    forge::CompilerConfig config;
    config.instructionSet = forge::CompilerConfig::InstructionSet::SSE2_SCALAR;

    forge::ForgeEngine engine(config);
    auto kernel = engine.compile(graph);

    ASSERT_NE(kernel, nullptr) << "SSE2 compilation failed";
    EXPECT_EQ(kernel->getVectorWidth(), 1) << "Expected SSE2 vector width of 1";

    // Create buffer and run
    auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);
    buffer->setValue(x, 5.0);
    kernel->execute(*buffer);

    double outputValue = buffer->getValue(result);
    EXPECT_DOUBLE_EQ(outputValue, 6.0) << "SSE2 computation result incorrect";
}
