#include <gtest/gtest.h>
#include "../tools/benchmarkTool/benchmark_runner.hpp"
#include "../tools/testFunctions/oneToOne/all.hpp"

using namespace forge::tools;
using namespace forge::tools::test_functions::one_to_one;
using namespace forge;

class BenchmarkToolTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(BenchmarkToolTest, SimpleBenchmark) {
    auto runner = makeBenchmarkRunner();
    runner.AddFunction("Quadratic", quadratic<double>, quadratic<fdouble>, 
                      {-2.0, -1.0, 0.0, 1.0, 2.0});
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkToolTest, TrigonometricBenchmark) {
    auto runner = makeBenchmarkRunner();
    runner.AddFunction("Sine", sine<double>, sine<fdouble>, 
                      getTrigonometricInputs());
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkToolTest, ExponentialBenchmark) {
    auto runner = makeBenchmarkRunner();
    runner.AddFunction("Exponential", expScaled<double>, expScaled<fdouble>, 
                      getSafeExponentialInputs());
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkToolTest, ComprehensiveBenchmark) {
    auto runner = makeBenchmarkRunner();
    
    // Add all functions to benchmark with their specific test inputs
    runner.AddFunction("Linear", linear<double>, linear<fdouble>, {0.5, 1.0, 1.5, 2.0});
    runner.AddFunction("Quadratic", quadratic<double>, quadratic<fdouble>, {0.5, 1.0, 1.5, 2.0});
    runner.AddFunction("Cubic", cubic<double>, cubic<fdouble>, {0.5, 1.0, 1.5, 2.0});
    runner.AddFunction("Sine", sine<double>, sine<fdouble>, {0.5, 1.0, 1.5, 2.0});
    runner.AddFunction("Cosine", cosine<double>, cosine<fdouble>, {0.5, 1.0, 1.5, 2.0});
    runner.AddFunction("Sqrt", sqrtWithOps<double>, sqrtWithOps<fdouble>, {0.5, 1.0, 1.5, 2.0});
    
    // Run benchmarks and verify all pass - one liner!
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkToolTest, MinimalConfiguration) {
    auto runner = makeBenchmarkRunner();
    runner.AddFunction("Inverse", inverse<double>, inverse<fdouble>, 
                      {0.5, 1.0, 2.0, 4.0});
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkToolTest, AmericanOptionsBenchmark) {
    auto runner = makeBenchmarkRunner();
    
    // Add American and European options
    runner.AddFunction("American Put", americanPut<double>, americanPut<fdouble>, 
                      {80.0, 90.0, 100.0, 110.0, 120.0});
    runner.AddFunction("American Call", americanCall<double>, americanCall<fdouble>, 
                      {80.0, 90.0, 100.0, 110.0, 120.0});
    runner.AddFunction("European Put", europeanPut<double>, europeanPut<fdouble>, 
                      {80.0, 90.0, 100.0, 110.0, 120.0});
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkToolTest, SmallIterativeGraphBenchmark) {
    auto runner = makeBenchmarkRunner();
    
    // Add small iterative graph function
    runner.AddFunction("Small Iterative Graph", smallIterativeGraph<double>, smallIterativeGraph<fdouble>, 
                      getSmallGraphInputs());
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkToolTest, MediumIterativeGraphBenchmark) {
    auto runner = makeBenchmarkRunner();
    
    // Add medium iterative graph function (~10K operations)
    runner.AddFunction("Medium Iterative Graph", mediumIterativeGraph<double>, mediumIterativeGraph<fdouble>, 
                      getBigGraphInputs());
    
    EXPECT_TRUE(runner.RunBenchmarks());
}