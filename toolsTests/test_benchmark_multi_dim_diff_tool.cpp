#include <gtest/gtest.h>
#include "../tools/benchmarkTool/benchmark_multi_dim_diff_runner.hpp"
#include "../tools/testFunctions/multiToMulti/all.hpp"

using namespace forge::tools;
using namespace forge::tools::test_functions::multi_to_multi;
using namespace forge;

// Test fixture for multi-dim benchmark differentiation tests
class BenchmarkMultiDimDiffToolTest : public ::testing::Test {
protected:
    BenchmarkMultiDimDiffConfig config_;
    
    void SetUp() override {
        // Configure for benchmarking (faster than sanity check defaults)
        config_.iterations = 10;  // Fewer iterations for benchmarking
        config_.warmupRuns = 5;
        config_.absoluteTolerance = 1e-10;
        config_.relativeTolerance = 1e-10;
        config_.jacobianAbsTolerance = 1e-6;  // Relaxed for finite differences
        config_.jacobianRelTolerance = 1e-6;
        config_.finiteDiffBump = 1e-8;
        config_.useRichardsonExtrapolation = false;
        config_.showJacobianDetails = false;  // Keep output concise
        config_.showScalingAnalysis = true;
    }
    
    void TearDown() override {}
};

// Test linear transformations (predictable performance baseline)
TEST_F(BenchmarkMultiDimDiffToolTest, LinearTransform2x3) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Linear Transform 2x3",
        linearTransform2x3<double>,
        linearTransform2x3<fdouble>,
        getLinearTransform2x3Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkMultiDimDiffToolTest, MatrixMultiply3x2) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Matrix Multiply 3x2", 
        matrixMultiply3x2<double>,
        matrixMultiply3x2<fdouble>,
        getMatrixMultiply3x2Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkMultiDimDiffToolTest, IdentityTransform3x3) {
    // Identity should have identity Jacobian - very fast
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Identity Transform 3x3",
        identityTransform3x3<double>,
        identityTransform3x3<fdouble>,
        getIdentityTransform3x3Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkMultiDimDiffToolTest, AffineTransform2x2) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Affine Transform 2x2",
        affineTransform2x2<double>,
        affineTransform2x2<fdouble>,
        getAffineTransform2x2Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Test coordinate transformations (moderate complexity)
TEST_F(BenchmarkMultiDimDiffToolTest, PolarToCartesian) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Polar to Cartesian",
        polarToCartesian<double>,
        polarToCartesian<fdouble>,
        getPolarToCartesianInputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}
TEST_F(BenchmarkMultiDimDiffToolTest, SphericalToCartesian) {
    // 3D coordinate transformation - higher dimensional Jacobian
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Spherical to Cartesian",
        sphericalToCartesian<double>,
        sphericalToCartesian<fdouble>,
        getSphericalToCartesianInputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Test nonlinear transformations (computational intensity)
TEST_F(BenchmarkMultiDimDiffToolTest, TrigonometricTransform2x3) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Trigonometric Transform 2x3",
        trigonometricTransform2x3<double>,
        trigonometricTransform2x3<fdouble>,
        getTrigonometricTransform2x3Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkMultiDimDiffToolTest, ExponentialTransform2x2) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Exponential Transform 2x2",
        exponentialTransform2x2<double>,
        exponentialTransform2x2<fdouble>,
        getExponentialTransform2x2Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkMultiDimDiffToolTest, NonlinearSystem2x2) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Nonlinear System 2x2",
        nonlinearSystem2x2<double>,
        nonlinearSystem2x2<fdouble>,
        getNonlinearSystem2x2Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkMultiDimDiffToolTest, PolynomialTransform2x3) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Polynomial Transform 2x3",
        polynomialTransform2x3<double>,
        polynomialTransform2x3<fdouble>,
        getPolynomialTransform2x3Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Test neural network activation functions
TEST_F(BenchmarkMultiDimDiffToolTest, SigmoidTransform2x2) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Sigmoid Transform 2x2",
        sigmoidTransform2x2<double>,
        sigmoidTransform2x2<fdouble>,
        getSigmoidTransform2x2Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkMultiDimDiffToolTest, SoftmaxTransform3x3) {
    // Softmax has special Jacobian structure - more expensive
    auto softmax_config = config_;
    softmax_config.jacobianAbsTolerance = 1e-5;  // Slightly relaxed for softmax
    
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(softmax_config);
    runner.AddFunction(
        "Softmax Transform 3x3",
        softmaxTransform3x3<double>,
        softmaxTransform3x3<fdouble>,
        getSoftmaxTransform3x3Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Test dimensionality extremes
TEST_F(BenchmarkMultiDimDiffToolTest, QuadraticForm2x1) {
    // Many inputs to single output - wide Jacobian
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Quadratic Form 2x1",
        quadraticForm2x1<double>,
        quadraticForm2x1<fdouble>,
        getQuadraticForm2x1Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkMultiDimDiffToolTest, ExpansionMap1x3) {
    // Single input to many outputs - tall Jacobian
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Expansion Map 1x3",
        expansionMap1x3<double>,
        expansionMap1x3<fdouble>,
        getExpansionMap1x3Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkMultiDimDiffToolTest, ProjectionMap4x2) {
    // Higher dimensional input space
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Projection Map 4x2",
        projectionMap4x2<double>,
        projectionMap4x2<fdouble>,
        getProjectionMap4x2Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Test special mathematical properties
TEST_F(BenchmarkMultiDimDiffToolTest, CrossProduct3x3) {
    // Cross product has antisymmetric properties
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Cross Product 3x3",
        crossProduct3x3<double>,
        crossProduct3x3<fdouble>,
        getCrossProduct3x3Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkMultiDimDiffToolTest, RotationTransform2x2) {
    // Orthogonal transformation - Jacobian has special structure
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Rotation Transform 2x2",
        rotationTransform2x2<double>,
        rotationTransform2x2<fdouble>,
        getRotationTransform2x2Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Test with Richardson extrapolation for higher accuracy
TEST_F(BenchmarkMultiDimDiffToolTest, HighAccuracyLinearTransform) {
    auto high_acc_config = config_;
    high_acc_config.useRichardsonExtrapolation = true;
    high_acc_config.jacobianAbsTolerance = 1e-6;  // Realistic tolerance for FD with Richardson
    high_acc_config.jacobianRelTolerance = 1e-6;
    high_acc_config.iterations = 10;  // Fewer iterations due to extra FD computations
    
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(high_acc_config);
    runner.AddFunction(
        "Linear Transform 2x3 (High Accuracy)",
        linearTransform2x3<double>,
        linearTransform2x3<fdouble>,
        getLinearTransform2x3Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Test rational functions (can be numerically challenging)
TEST_F(BenchmarkMultiDimDiffToolTest, RationalTransform2x2) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Rational Transform 2x2",
        rationalTransform2x2<double>,
        rationalTransform2x2<fdouble>,
        getRationalTransform2x2Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Test logarithmic transformations
TEST_F(BenchmarkMultiDimDiffToolTest, LogarithmicTransform2x2) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Logarithmic Transform 2x2",
        logarithmicTransform2x2<double>,
        logarithmicTransform2x2<fdouble>,
        getLogarithmicTransform2x2Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Test with detailed Jacobian output
TEST_F(BenchmarkMultiDimDiffToolTest, DetailedJacobianOutput) {
    auto detailed_config = config_;
    detailed_config.showJacobianDetails = true;
    detailed_config.iterations = 10;  // Fewer iterations for detailed output
    
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(detailed_config);
    runner.AddFunction(
        "Polar to Cartesian (Detailed)",
        polarToCartesian<double>,
        polarToCartesian<fdouble>,
        getPolarToCartesianInputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Test complex transformations
TEST_F(BenchmarkMultiDimDiffToolTest, MixedTransform3x4) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Mixed Transform 3x4",
        mixedTransform3x4<double>,
        mixedTransform3x4<fdouble>,
        getMixedTransform3x4Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

TEST_F(BenchmarkMultiDimDiffToolTest, NormalizationTransform3x3) {
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(config_);
    runner.AddFunction(
        "Normalization Transform 3x3",
        normalizationTransform3x3<double>,
        normalizationTransform3x3<fdouble>,
        getNormalizationTransform3x3Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Performance comparison test - multiple functions at once
TEST_F(BenchmarkMultiDimDiffToolTest, PerformanceComparison) {
    auto comparison_config = config_;
    comparison_config.showScalingAnalysis = true;
    comparison_config.iterations = 10;  // More iterations for better statistics
    
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(comparison_config);
    
    // Add multiple functions for direct comparison
    runner.AddFunction(
        "Linear 2x3 (Baseline)",
        linearTransform2x3<double>,
        linearTransform2x3<fdouble>,
        getLinearTransform2x3Inputs()
    );
    
    EXPECT_TRUE(runner.RunBenchmarks());
}

// Minimal configuration test
TEST_F(BenchmarkMultiDimDiffToolTest, MinimalConfiguration) {
    auto minimal_config = config_;
    minimal_config.iterations = 10;       // Fast execution
    minimal_config.warmupRuns = 5;
    minimal_config.showJacobianDetails = false;
    minimal_config.showScalingAnalysis = false;
    
    auto runner = makeBenchmarkMultiDimDiffRunner<
        std::vector<double>(*)(const std::vector<double>&),
        std::vector<fdouble>(*)(const std::vector<fdouble>&)
    >(minimal_config);
    runner.AddFunction(
        "Identity (Minimal)",
        identityTransform3x3<double>,
        identityTransform3x3<fdouble>,
        getIdentityTransform3x3Inputs()
    );
    EXPECT_TRUE(runner.RunBenchmarks());
}