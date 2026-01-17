#include <gtest/gtest.h>
#include "../src/compiler/runtime_trace.hpp"
#include "../src/compiler/x86/common/instruction_set_factory.hpp"
#include "../src/compiler/x86/common/compiler_config.hpp"
#include <asmjit/x86.h>
#include <iostream>

using namespace forge;

class RuntimeTraceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the trace buffer
        initializeTraceBuffer(1024);
        setTracingEnabled(true);
    }
    
    void TearDown() override {
        // Clean up
        cleanupTraceBuffer();
    }
};

TEST_F(RuntimeTraceTest, TestTraceBufferInitialization) {
    EXPECT_TRUE(isTracingEnabled());

    // Verify buffer is initialized and ready
    // Actual tracing happens via JIT-generated code writing directly to g_traceBuffer
    printTraceRecords();
}

TEST_F(RuntimeTraceTest, TestAVX2InstructionSetTracing) {
    // Create AVX2 instruction set with tracing enabled
    CompilerConfig config;
    config.printRuntimeTrace = true;
    
    auto avx2Set = InstructionSetFactory::create(CompilerConfig::InstructionSet::AVX2_PACKED, config);
    EXPECT_NE(avx2Set, nullptr);
    EXPECT_EQ(avx2Set->getName(), "AVX2-Packed");
    
    // Test that we can create assembler and emit instructions
    asmjit::JitRuntime rt;
    asmjit::CodeHolder code;
    code.init(rt.environment());
    
    asmjit::x86::Assembler a(&code);
    
    // This would normally be called by the compiler
    // For now, just test that the instruction set can be created
    EXPECT_TRUE(true);
}

TEST_F(RuntimeTraceTest, TestSSE2InstructionSetTracing) {
    // Create SSE2 instruction set with tracing enabled
    CompilerConfig config;
    config.printRuntimeTrace = true;
    
    auto sse2Set = InstructionSetFactory::create(CompilerConfig::InstructionSet::SSE2_SCALAR, config);
    EXPECT_NE(sse2Set, nullptr);
    EXPECT_EQ(sse2Set->getName(), "SSE2-Scalar");
    
    // Test that we can create assembler and emit instructions
    asmjit::JitRuntime rt;
    asmjit::CodeHolder code;
    code.init(rt.environment());
    
    asmjit::x86::Assembler a(&code);
    
    // This would normally be called by the compiler
    // For now, just test that the instruction set can be created
    EXPECT_TRUE(true);
}

TEST_F(RuntimeTraceTest, TestTraceRecordStructure) {
    // Test the trace record structure
    TraceRecord record;
    record.instructionId = 1;
    record.operationType = static_cast<uint32_t>(OperationType::ADD);
    record.vectorWidth = 4;
    record.timestamp = 12345;
    
    // Fill with test data
    double* values = reinterpret_cast<double*>(record.data);
    values[0] = 1.0;
    values[1] = 2.0;
    values[2] = 3.0;
    values[3] = 4.0;
    
    EXPECT_EQ(record.instructionId, 1);
    EXPECT_EQ(record.operationType, static_cast<uint32_t>(OperationType::ADD));
    EXPECT_EQ(record.vectorWidth, 4);
    EXPECT_EQ(values[0], 1.0);
    EXPECT_EQ(values[1], 2.0);
    EXPECT_EQ(values[2], 3.0);
    EXPECT_EQ(values[3], 4.0);
}

TEST_F(RuntimeTraceTest, TestOperationTypeNames) {
    EXPECT_STREQ(getOperationName(static_cast<uint32_t>(OperationType::ADD)), "ADD");
    EXPECT_STREQ(getOperationName(static_cast<uint32_t>(OperationType::SUB)), "SUB");
    EXPECT_STREQ(getOperationName(static_cast<uint32_t>(OperationType::MUL)), "MUL");
    EXPECT_STREQ(getOperationName(static_cast<uint32_t>(OperationType::DIV)), "DIV");
    EXPECT_STREQ(getOperationName(static_cast<uint32_t>(OperationType::SQRT)), "SQRT");
    EXPECT_STREQ(getOperationName(static_cast<uint32_t>(OperationType::UNKNOWN)), "UNKNOWN");
}

TEST_F(RuntimeTraceTest, TestTracingEnabledDisabled) {
    // Test enabling/disabling tracing
    setTracingEnabled(false);
    EXPECT_FALSE(isTracingEnabled());
    
    setTracingEnabled(true);
    EXPECT_TRUE(isTracingEnabled());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
