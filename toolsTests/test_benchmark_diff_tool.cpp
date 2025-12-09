#include <gtest/gtest.h>
#include "../tools/benchmarkTool/benchmark_diff_runner.hpp"
#include "../tools/testFunctions/oneToOne/all.hpp"

using namespace forge::tools;
using namespace forge::tools::test_functions::one_to_one;
using namespace forge;

class BenchmarkDiffToolTest : public ::testing::Test {
protected:
    BenchmarkDiffConfig config_;
    
    void SetUp() override {
        // Configure for comprehensive benchmarking with derivatives
        config_.iterations = 10;  // Reduced for faster tests
        config_.warmupRuns = 5;
        config_.finiteDiffBump = 1e-8;
        config_.useRichardsonExtrapolation = false;
        config_.absoluteTolerance = 1e-10;
        config_.relativeTolerance = 1e-10;
        config_.derivativeAbsTolerance = 1e-6;
        config_.derivativeRelTolerance = 1e-6;
    }
    
    void TearDown() override {}
};

TEST_F(BenchmarkDiffToolTest, SimpleBenchmark) {
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(config_);
    runner.AddFunction("Quadratic", quadratic<double>, quadratic<fdouble>, 
                       {-2.0, -1.0, 0.0, 1.0, 2.0});
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, SimpleBenchmarkAVX2) {
    // Simple test with AVX2 SIMD configuration
    // The benchmark runner now automatically detects AVX2 and uses vectorized workspace
    
    auto avx2Config = config_;
    avx2Config.compilerConfig.instructionSet = forge::CompilerConfig::InstructionSet::AVX2_PACKED;
    
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(avx2Config);
    runner.AddFunction("Quadratic", quadratic<double>, quadratic<fdouble>, 
                       {-2.0, -1.0, 0.0, 1.0, 2.0});
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, TrigonometricBenchmark) {
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(config_);
    runner.AddFunction("Sine", sine<double>, sine<fdouble>, 
                       getTrigonometricInputs());
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, ExponentialBenchmark) {
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(config_);
    runner.AddFunction("Exponential", expScaled<double>, expScaled<fdouble>, 
                       getSafeExponentialInputs());
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, ComprehensiveBenchmark) {
    // Test with reduced iterations for faster CI/CD
    auto fastConfig = config_;
    fastConfig.iterations = 10;
    fastConfig.warmupRuns = 5;
    
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(fastConfig);
    
    // Add multiple functions to benchmark with their specific test inputs
    runner.AddFunction("Linear", linear<double>, linear<fdouble>, {0.5, 1.0, 1.5, 2.0});
    runner.AddFunction("Quadratic", quadratic<double>, quadratic<fdouble>, {0.5, 1.0, 1.5, 2.0});
    runner.AddFunction("Cubic", cubic<double>, cubic<fdouble>, {0.5, 1.0, 1.5, 2.0});
    runner.AddFunction("Sine", sine<double>, sine<fdouble>, {0.5, 1.0, 1.5, 2.0});
    runner.AddFunction("Cosine", cosine<double>, cosine<fdouble>, {0.5, 1.0, 1.5, 2.0});
    runner.AddFunction("Sqrt", sqrtWithOps<double>, sqrtWithOps<fdouble>, {0.5, 1.0, 1.5, 2.0});
    
    // Run benchmarks and verify all pass
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, MinimalConfiguration) {
    // Test with minimal configuration for quick smoke test
    auto minimalConfig = config_;
    minimalConfig.iterations = 10;
    minimalConfig.warmupRuns = 5;
    
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(minimalConfig);
    runner.AddFunction("Inverse", inverse<double>, inverse<fdouble>, 
                       {0.5, 1.0, 2.0, 4.0});
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, RichardsonExtrapolation) {
    // Test with Richardson extrapolation for higher gradient accuracy
    auto richardsonConfig = config_;
    richardsonConfig.iterations = 10;
    richardsonConfig.useRichardsonExtrapolation = true;
    richardsonConfig.derivativeAbsTolerance = 1e-8;  // Tighter tolerance with Richardson
    
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(richardsonConfig);
    runner.AddFunction("Cubic", cubic<double>, cubic<fdouble>, 
                       {-1.0, 0.0, 0.5, 1.0, 2.0});
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, ComplexFunction) {
    // Test with a more complex function
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(config_);
    runner.AddFunction("Trig Combo", trigCombo<double>, trigCombo<fdouble>, 
                       {0.0, 0.5, 1.0, 1.5, 2.0});
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, RationalFunction) {
    // Test rational functions with derivative computation
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(config_);
    runner.AddFunction("Rational", rationalFunction<double>, rationalFunction<fdouble>, 
                       getRationalInputs());
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, HighPerformanceMode) {
    // Test with high iteration count for accurate performance measurement
    auto perfConfig = config_;
    perfConfig.iterations = 10;
    perfConfig.warmupRuns = 5;
    
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(perfConfig);
    runner.AddFunction("Linear", linear<double>, linear<fdouble>, 
                       {-1.0, 0.0, 1.0});
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, MultipleFunctionsSequential) {
    // Test running multiple functions sequentially
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(config_);
    
    // Add functions one by one
    runner.AddFunction("Squared", squared<double>, squared<fdouble>, {-2.0, -1.0, 0.0, 1.0, 2.0});
    runner.AddFunction("Absolute", absolute<double>, absolute<fdouble>, {-2.0, -1.0, 0.0, 1.0, 2.0});
    runner.AddFunction("Negation", negation<double>, negation<fdouble>, {-2.0, -1.0, 0.0, 1.0, 2.0});
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, AmericanOptionsBenchmark) {
    // Configure for options (relaxed derivative tolerances due to non-smooth payoffs)
    auto optionConfig = config_;
    optionConfig.derivativeAbsTolerance = 1e-3;
    optionConfig.derivativeRelTolerance = 1e-3;
    optionConfig.iterations = 10;  // Fewer iterations since options are complex
    optionConfig.warmupRuns = 5;
    
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(optionConfig);
    
    // Add American and European options
    runner.AddFunction("American Put", americanPut<double>, americanPut<fdouble>, 
                       {80.0, 90.0, 100.0, 110.0, 120.0});
    runner.AddFunction("American Call", americanCall<double>, americanCall<fdouble>, 
                       {80.0, 90.0, 100.0, 110.0, 120.0});
    runner.AddFunction("European Put", europeanPut<double>, europeanPut<fdouble>, 
                       {80.0, 90.0, 100.0, 110.0, 120.0});
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, AmericanOptionsBenchmarkAVX2) {
    // Configure for options with AVX2 SIMD vectorization
    // The benchmark runner now automatically detects AVX2 and uses vectorized workspace
    // Processing 4 inputs simultaneously with AVX2 SIMD instructions
    
    auto optionConfig = config_;
    optionConfig.derivativeAbsTolerance = 1e-3;
    optionConfig.derivativeRelTolerance = 1e-3;
    optionConfig.iterations = 10;  // Fewer iterations since options are complex
    optionConfig.warmupRuns = 5;
    
    // Configure compiler for AVX2 SIMD - processes 4 values in parallel
    optionConfig.compilerConfig.instructionSet = forge::CompilerConfig::InstructionSet::AVX2_PACKED;
    
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(optionConfig);
    
    // Add American and European options
    runner.AddFunction("American Put", americanPut<double>, americanPut<fdouble>, 
                       {80.0, 90.0, 100.0, 110.0, 120.0});
    runner.AddFunction("American Call", americanCall<double>, americanCall<fdouble>, 
                       {80.0, 90.0, 100.0, 110.0, 120.0});
    runner.AddFunction("European Put", europeanPut<double>, europeanPut<fdouble>, 
                       {80.0, 90.0, 100.0, 110.0, 120.0});
    
    // When AVX2_PACKED is detected, the benchmark runner should output additional SIMD comparison:
    // | Mode                    | Inputs | Forward(ns) | +Backward(ns) | Total(ns) | vs Native |
    // |-------------------------|--------|-------------|---------------|-----------|-----------|
    // | Native C++ (1x)         |      1 |      XXX.XX |        XXX.XX |   XXXX.XX |     1.00x |
    // | Native C++ (4x seq)     |      4 |     XXXX.XX |       XXXX.XX |   XXXX.XX |     1.00x |
    // | SSE2 JIT (scalar)       |      1 |       XX.XX |             - |     XX.XX |     X.XXx |
    // | SSE2 JIT (scalar+grad)  |      1 |           - |        XXX.XX |    XXX.XX |    XX.XXx |
    // | AVX2 JIT (4x SIMD)      |      4 |      XXX.XX |             - |    XXX.XX |     X.XXx |
    // | AVX2 JIT (4x SIMD+grad) |      4 |           - |        XXX.XX |    XXX.XX |    XX.XXx |
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, SmallIterativeGraphBenchmark) {
    // Configure for small iterative graph testing
    auto graphConfig = config_;
    graphConfig.iterations = 10;  // Reduce iterations since graph is complex
    graphConfig.warmupRuns = 5;
    graphConfig.derivativeAbsTolerance = 1e-5;  // Relaxed tolerance for complex graph
    graphConfig.derivativeRelTolerance = 1e-5;
    
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(graphConfig);
    
    // Add small iterative graph function
    runner.AddFunction("Small Iterative Graph", smallIterativeGraph<double>, smallIterativeGraph<fdouble>, 
                       getSmallGraphInputs());
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkDiffToolTest, MediumIterativeGraphBenchmark) {
    // Configure for medium iterative graph testing (~10K operations)
    auto graphConfig = config_;
    graphConfig.iterations = 10;  // Further reduce iterations for medium complexity
    graphConfig.warmupRuns = 5;
    graphConfig.derivativeAbsTolerance = 1e-4;  // More relaxed tolerance for medium graph
    graphConfig.derivativeRelTolerance = 1e-4;
    
    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(graphConfig);
    
    // Add medium iterative graph function
    runner.AddFunction("Medium Iterative Graph", mediumIterativeGraph<double>, mediumIterativeGraph<fdouble>, 
                       getBigGraphInputs());
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

//TEST_F(BenchmarkDiffToolTest, MassiveIterativeGraphBenchmark) {
//    // Configure for massive iterative graph testing (~1M operations)
//    auto graphConfig = config_;
//    graphConfig.iterations = 10;  // Very few iterations for massive graph
//    graphConfig.warmupRuns = 2;
//    graphConfig.derivativeAbsTolerance = 1e-3;  // Very relaxed tolerance for massive graph
//    graphConfig.derivativeRelTolerance = 1e-3;
//    
//    auto runner = makeBenchmarkDiffRunner<double(*)(double), fdouble(*)(fdouble)>(graphConfig);
//    
//    // Add massive iterative graph function
//    runner.AddFunction("Massive Iterative Graph", massiveIterativeGraph<double>, massiveIterativeGraph<fdouble>, 
//                       getBigGraphInputs());
//    
//    EXPECT_TRUE(runner.RunBenchmarks());
//}