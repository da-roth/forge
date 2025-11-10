#include <gtest/gtest.h>
#include "../tools/sanityTool/sanity_multi_dim_checker_diff.hpp"
#include "../tools/testFunctions/multiToMulti/all.hpp"

using namespace forge::tools;
using namespace forge::tools::test_functions::multi_to_multi;
using namespace forge;

// Test fixture for multi-dim sanity checker with derivatives tests
class SanityMultiDimDiffToolTest : public ::testing::Test {
protected:
    MultiDimDiffConfig config_;
    
    void SetUp() override {
        // Configure for derivative testing
        config_.absoluteTolerance = 1e-10;
        config_.relativeTolerance = 1e-10;
        config_.derivativeAbsTolerance = 1e-6;  // Relaxed for finite differences
        config_.derivativeRelTolerance = 1e-6;
        config_.finiteDiffBump = 1e-8;
        config_.verbose = true;  // Match original defaults
        config_.showTimings = true;
        config_.showJacobian = true;
        config_.useRichardsonExtrapolation = false;  // Start with simple FD
    }
    
    void TearDown() override {}
};

// Test linear transformations with known constant Jacobians
TEST_F(SanityMultiDimDiffToolTest, LinearTransform2x3) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Linear Transform 2x3",
        linearTransform2x3<double>,
        linearTransform2x3<fdouble>,
        getLinearTransform2x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, MatrixMultiply3x2) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Matrix Multiply 3x2",
        matrixMultiply3x2<double>,
        matrixMultiply3x2<fdouble>,
        getMatrixMultiply3x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, IdentityTransform3x3) {
    // Identity should have identity Jacobian
    auto checker = makeSanityMultiDimCheckerDiff(
        "Identity Transform 3x3",
        identityTransform3x3<double>,
        identityTransform3x3<fdouble>,
        getIdentityTransform3x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, AffineTransform2x2) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Affine Transform 2x2",
        affineTransform2x2<double>,
        affineTransform2x2<fdouble>,
        getAffineTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, ScalarMultiply2x2) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Scalar Multiply 2x2",
        scalarMultiply2x2<double>,
        scalarMultiply2x2<fdouble>,
        getScalarMultiply2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

// Test nonlinear transformations
TEST_F(SanityMultiDimDiffToolTest, PolarToCartesian) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Polar to Cartesian",
        polarToCartesian<double>,
        polarToCartesian<fdouble>,
        getPolarToCartesianInputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, TrigonometricTransform2x3) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Trigonometric Transform 2x3",
        trigonometricTransform2x3<double>,
        trigonometricTransform2x3<fdouble>,
        getTrigonometricTransform2x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, ExponentialTransform2x2) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Exponential Transform 2x2",
        exponentialTransform2x2<double>,
        exponentialTransform2x2<fdouble>,
        getExponentialTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, NonlinearSystem2x2) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Nonlinear System 2x2",
        nonlinearSystem2x2<double>,
        nonlinearSystem2x2<fdouble>,
        getNonlinearSystem2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, PolynomialTransform2x3) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Polynomial Transform 2x3",
        polynomialTransform2x3<double>,
        polynomialTransform2x3<fdouble>,
        getPolynomialTransform2x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

// Test functions with special Jacobian properties
TEST_F(SanityMultiDimDiffToolTest, CrossProduct3x3) {
    // Cross product has antisymmetric Jacobian properties
    auto checker = makeSanityMultiDimCheckerDiff(
        "Cross Product 3x3",
        crossProduct3x3<double>,
        crossProduct3x3<fdouble>,
        getCrossProduct3x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, QuadraticForm2x1) {
    // Quadratic form collapses to scalar
    auto checker = makeSanityMultiDimCheckerDiff(
        "Quadratic Form 2x1",
        quadraticForm2x1<double>,
        quadraticForm2x1<fdouble>,
        getQuadraticForm2x1Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

// Test dimension expansion and projection
TEST_F(SanityMultiDimDiffToolTest, ExpansionMap1x3) {
    // 1D to 3D expansion
    auto checker = makeSanityMultiDimCheckerDiff(
        "Expansion Map 1x3",
        expansionMap1x3<double>,
        expansionMap1x3<fdouble>,
        getExpansionMap1x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, ProjectionMap4x2) {
    // 4D to 2D projection
    auto checker = makeSanityMultiDimCheckerDiff(
        "Projection Map 4x2",
        projectionMap4x2<double>,
        projectionMap4x2<fdouble>,
        getProjectionMap4x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

// Test activation functions
TEST_F(SanityMultiDimDiffToolTest, SigmoidTransform2x2) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Sigmoid Transform 2x2",
        sigmoidTransform2x2<double>,
        sigmoidTransform2x2<fdouble>,
        getSigmoidTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, SoftmaxTransform3x3) {
    // Softmax has special Jacobian structure
    auto softmax_config = config_;
    softmax_config.derivativeAbsTolerance = 1e-5;  // Slightly relaxed for softmax
    
    auto checker = makeSanityMultiDimCheckerDiff(
        "Softmax Transform 3x3",
        softmaxTransform3x3<double>,
        softmaxTransform3x3<fdouble>,
        getSoftmaxTransform3x3Inputs(),
        softmax_config
    );
    EXPECT_TRUE(checker.RunTests());
}

// Test with Richardson extrapolation for higher accuracy
TEST_F(SanityMultiDimDiffToolTest, HighAccuracyLinearTransform) {
    auto high_acc_config = config_;
    high_acc_config.useRichardsonExtrapolation = true;
    high_acc_config.derivativeAbsTolerance = 1e-6;  // Realistic tolerance for FD with Richardson
    high_acc_config.derivativeRelTolerance = 1e-6;
    
    auto checker = makeSanityMultiDimCheckerDiff(
        "Linear Transform 2x3 (High Accuracy)",
        linearTransform2x3<double>,
        linearTransform2x3<fdouble>,
        getLinearTransform2x3Inputs(),
        high_acc_config
    );
    EXPECT_TRUE(checker.RunTests());
}

// Test with different bump sizes
TEST_F(SanityMultiDimDiffToolTest, DifferentBumpSizes) {
    // Test with larger bump
    auto large_bump_config = config_;
    large_bump_config.finiteDiffBump = 1e-5;
    large_bump_config.derivativeAbsTolerance = 1e-4;  // Relaxed for larger bump
    
    auto checker1 = makeSanityMultiDimCheckerDiff(
        "Polynomial (Large Bump)",
        polynomialTransform2x3<double>,
        polynomialTransform2x3<fdouble>,
        getPolynomialTransform2x3Inputs(),
        large_bump_config
    );
    EXPECT_TRUE(checker1.RunTests());
    
    // Test with smaller bump
    auto small_bump_config = config_;
    small_bump_config.finiteDiffBump = 1e-10;
    small_bump_config.derivativeAbsTolerance = 1e-5;  // Relaxed due to numerical precision
    
    auto checker2 = makeSanityMultiDimCheckerDiff(
        "Polynomial (Small Bump)",
        polynomialTransform2x3<double>,
        polynomialTransform2x3<fdouble>,
        getPolynomialTransform2x3Inputs(),
        small_bump_config
    );
    EXPECT_TRUE(checker2.RunTests());
}

// Test minimal verbosity
TEST_F(SanityMultiDimDiffToolTest, MinimalOutput) {
    auto quiet_config = config_;
    quiet_config.verbose = false;
    quiet_config.showTimings = false;
    quiet_config.showJacobian = false;
    
    auto checker = makeSanityMultiDimCheckerDiff(
        "Affine Transform (Quiet)",
        affineTransform2x2<double>,
        affineTransform2x2<fdouble>,
        getAffineTransform2x2Inputs(),
        quiet_config
    );
    EXPECT_TRUE(checker.RunTests());
}

// Test complex transformations
TEST_F(SanityMultiDimDiffToolTest, MixedTransform3x4) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Mixed Transform 3x4",
        mixedTransform3x4<double>,
        mixedTransform3x4<fdouble>,
        getMixedTransform3x4Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimDiffToolTest, NormalizationTransform3x3) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Normalization Transform 3x3",
        normalizationTransform3x3<double>,
        normalizationTransform3x3<fdouble>,
        getNormalizationTransform3x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

// Test rotation transformation
TEST_F(SanityMultiDimDiffToolTest, RotationTransform2x2) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Rotation Transform 2x2",
        rotationTransform2x2<double>,
        rotationTransform2x2<fdouble>,
        getRotationTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

// Test rational functions
TEST_F(SanityMultiDimDiffToolTest, RationalTransform2x2) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Rational Transform 2x2",
        rationalTransform2x2<double>,
        rationalTransform2x2<fdouble>,
        getRationalTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

// Test logarithmic transformation
TEST_F(SanityMultiDimDiffToolTest, LogarithmicTransform2x2) {
    auto checker = makeSanityMultiDimCheckerDiff(
        "Logarithmic Transform 2x2",
        logarithmicTransform2x2<double>,
        logarithmicTransform2x2<fdouble>,
        getLogarithmicTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}