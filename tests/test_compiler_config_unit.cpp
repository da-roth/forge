#include <gtest/gtest.h>
#include "../src/compiler/x86/common/compiler_config.hpp"
#include <cstdlib>
#ifdef _WIN32
#include <stdlib.h>  // For _putenv_s
#endif

using namespace forge;

class CompilerConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear environment variable if it exists
        #ifdef _WIN32
        _putenv_s("FORGE_INSTRUCTION_SET", "");
        #else
        unsetenv("FORGE_INSTRUCTION_SET");
        #endif
    }
};

TEST_F(CompilerConfigTest, DefaultConfiguration) {
    auto config = CompilerConfig::Default();
    
    // Default should only have stability cleaning enabled
    EXPECT_FALSE(config.enableOptimizations);
    EXPECT_FALSE(config.enableInactiveFolding);
    EXPECT_FALSE(config.enableCSE);
    EXPECT_FALSE(config.enableAlgebraicSimplification);
    EXPECT_TRUE(config.enableStabilityCleaning);
    
    // Default instruction set
    EXPECT_EQ(config.instructionSet, CompilerConfig::InstructionSet::SSE2_SCALAR);
}

TEST_F(CompilerConfigTest, DebugConfiguration) {
    auto config = CompilerConfig::Debug();
    
    // Debug should enable all diagnostic output
    EXPECT_TRUE(config.printOriginalGraph);
    EXPECT_TRUE(config.printOptimizedGraph);
    EXPECT_TRUE(config.printAssembly);
    EXPECT_TRUE(config.printOptimizationStats);
    EXPECT_TRUE(config.printGradientDebug);
    EXPECT_TRUE(config.printNodeFlags);
    EXPECT_TRUE(config.enableDebugRecording);
}

TEST_F(CompilerConfigTest, NoOptimizationConfiguration) {
    auto config = CompilerConfig::NoOptimization();
    
    // All optimizations should be disabled
    EXPECT_FALSE(config.enableOptimizations);
    EXPECT_FALSE(config.enableInactiveFolding);
    EXPECT_FALSE(config.enableCSE);
    EXPECT_FALSE(config.enableAlgebraicSimplification);
    EXPECT_FALSE(config.enableStabilityCleaning);
    EXPECT_EQ(config.maxOptimizationPasses, 0);
}

TEST_F(CompilerConfigTest, FastConfiguration) {
    auto config = CompilerConfig::Fast();
    
    // All optimizations should be enabled
    EXPECT_TRUE(config.enableOptimizations);
    EXPECT_TRUE(config.enableInactiveFolding);
    EXPECT_TRUE(config.enableCSE);
    EXPECT_TRUE(config.enableAlgebraicSimplification);
    EXPECT_TRUE(config.enableStabilityCleaning);
    EXPECT_GT(config.maxOptimizationPasses, 0);
}

TEST_F(CompilerConfigTest, ValidationConfiguration) {
    auto config = CompilerConfig::Validation();
    
    EXPECT_TRUE(config.validateGraph);
    EXPECT_TRUE(config.boundsChecking);
    EXPECT_TRUE(config.printOptimizationStats);
}

TEST_F(CompilerConfigTest, DebugTracingConfiguration) {
    auto config = CompilerConfig::DebugTracing();

    EXPECT_TRUE(config.printRuntimeTrace);
}

TEST_F(CompilerConfigTest, LoadFromEnvironmentSSE2) {
    #ifdef _WIN32
    _putenv_s("FORGE_INSTRUCTION_SET", "SSE2");
    #else
    setenv("FORGE_INSTRUCTION_SET", "SSE2", 1);
    #endif
    
    CompilerConfig config;
    config.loadFromEnvironment();
    
    EXPECT_EQ(config.instructionSet, CompilerConfig::InstructionSet::SSE2_SCALAR);
}

TEST_F(CompilerConfigTest, LoadFromEnvironmentSSE2Scalar) {
    #ifdef _WIN32
    _putenv_s("FORGE_INSTRUCTION_SET", "SSE2-Scalar");
    #else
    setenv("FORGE_INSTRUCTION_SET", "SSE2-Scalar", 1);
    #endif
    
    CompilerConfig config;
    config.loadFromEnvironment();
    
    EXPECT_EQ(config.instructionSet, CompilerConfig::InstructionSet::SSE2_SCALAR);
}

TEST_F(CompilerConfigTest, LoadFromEnvironmentAVX2) {
    #ifdef _WIN32
    _putenv_s("FORGE_INSTRUCTION_SET", "AVX2");
    #else
    setenv("FORGE_INSTRUCTION_SET", "AVX2", 1);
    #endif
    
    CompilerConfig config;
    config.loadFromEnvironment();
    
    EXPECT_EQ(config.instructionSet, CompilerConfig::InstructionSet::AVX2_PACKED);
}

TEST_F(CompilerConfigTest, LoadFromEnvironmentAVX2Packed) {
    #ifdef _WIN32
    _putenv_s("FORGE_INSTRUCTION_SET", "AVX2-Packed");
    #else
    setenv("FORGE_INSTRUCTION_SET", "AVX2-Packed", 1);
    #endif
    
    CompilerConfig config;
    config.loadFromEnvironment();
    
    EXPECT_EQ(config.instructionSet, CompilerConfig::InstructionSet::AVX2_PACKED);
}

TEST_F(CompilerConfigTest, DefaultValues) {
    CompilerConfig config;
    
    // Check default values
    EXPECT_FALSE(config.enableOptimizations);
    EXPECT_FALSE(config.enableInactiveFolding);
    EXPECT_FALSE(config.enableCSE);
    EXPECT_FALSE(config.enableAlgebraicSimplification);
    EXPECT_TRUE(config.enableStabilityCleaning);
    EXPECT_EQ(config.maxOptimizationPasses, 5);
    
    EXPECT_FALSE(config.printOriginalGraph);
    EXPECT_FALSE(config.printOptimizedGraph);
    EXPECT_FALSE(config.printAssembly);
    
    EXPECT_EQ(config.maxRegisterCount, 16);
    EXPECT_FALSE(config.validateGraph);
    EXPECT_FALSE(config.boundsChecking);
    
    EXPECT_EQ(config.instructionSet, CompilerConfig::InstructionSet::SSE2_SCALAR);
}

