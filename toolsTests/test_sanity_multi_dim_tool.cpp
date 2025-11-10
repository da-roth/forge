#include <gtest/gtest.h>
#include "../tools/sanityTool/sanity_multi_dim_checker.hpp"
#include "../tools/testFunctions/multiToMulti/all.hpp"

using namespace forge::tools;
using namespace forge::tools::test_functions::multi_to_multi;
using namespace forge;

class SanityMultiDimToolTest : public ::testing::Test {
protected:
    MultiDimSanityConfig config_;
    
    void SetUp() override {
        config_.absoluteTolerance = 1e-10;
        config_.relativeTolerance = 1e-10;
        config_.verbose = true;  // Match original sanity checker default
        config_.showTimings = true;  // Match original sanity checker default
    }
    
    void TearDown() override {}
};

TEST_F(SanityMultiDimToolTest, LinearTransform2x3) {
    auto checker = makeSanityMultiDimChecker(
        "Linear Transform 2x3",
        linearTransform2x3<double>,
        linearTransform2x3<fdouble>,
        getLinearTransform2x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, MatrixMultiply3x2) {
    auto checker = makeSanityMultiDimChecker(
        "Matrix Multiply 3x2",
        matrixMultiply3x2<double>,
        matrixMultiply3x2<fdouble>,
        getMatrixMultiply3x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, IdentityTransform3x3) {
    auto checker = makeSanityMultiDimChecker(
        "Identity Transform 3x3",
        identityTransform3x3<double>,
        identityTransform3x3<fdouble>,
        getIdentityTransform3x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, ScalarMultiply2x2) {
    auto checker = makeSanityMultiDimChecker(
        "Scalar Multiply 2x2",
        scalarMultiply2x2<double>,
        scalarMultiply2x2<fdouble>,
        getScalarMultiply2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, CrossProduct3x3) {
    auto checker = makeSanityMultiDimChecker(
        "Cross Product 3x3",
        crossProduct3x3<double>,
        crossProduct3x3<fdouble>,
        getCrossProduct3x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, AffineTransform2x2) {
    auto checker = makeSanityMultiDimChecker(
        "Affine Transform 2x2",
        affineTransform2x2<double>,
        affineTransform2x2<fdouble>,
        getAffineTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, QuadraticForm2x1) {
    auto checker = makeSanityMultiDimChecker(
        "Quadratic Form 2x1",
        quadraticForm2x1<double>,
        quadraticForm2x1<fdouble>,
        getQuadraticForm2x1Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, ExpansionMap1x3) {
    auto checker = makeSanityMultiDimChecker(
        "Expansion Map 1x3",
        expansionMap1x3<double>,
        expansionMap1x3<fdouble>,
        getExpansionMap1x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, ProjectionMap4x2) {
    auto checker = makeSanityMultiDimChecker(
        "Projection Map 4x2",
        projectionMap4x2<double>,
        projectionMap4x2<fdouble>,
        getProjectionMap4x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, RotationTransform2x2) {
    auto checker = makeSanityMultiDimChecker(
        "Rotation Transform 2x2",
        rotationTransform2x2<double>,
        rotationTransform2x2<fdouble>,
        getRotationTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, PolarToCartesian) {
    auto checker = makeSanityMultiDimChecker(
        "Polar to Cartesian",
        polarToCartesian<double>,
        polarToCartesian<fdouble>,
        getPolarToCartesianInputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}


TEST_F(SanityMultiDimToolTest, SphericalToCartesian) {
    auto checker = makeSanityMultiDimChecker(
        "Spherical to Cartesian",
        sphericalToCartesian<double>,
        sphericalToCartesian<fdouble>,
        getSphericalToCartesianInputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, NonlinearSystem2x2) {
    auto checker = makeSanityMultiDimChecker(
        "Nonlinear System 2x2",
        nonlinearSystem2x2<double>,
        nonlinearSystem2x2<fdouble>,
        getNonlinearSystem2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, TrigonometricTransform2x3) {
    auto checker = makeSanityMultiDimChecker(
        "Trigonometric Transform 2x3",
        trigonometricTransform2x3<double>,
        trigonometricTransform2x3<fdouble>,
        getTrigonometricTransform2x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, ExponentialTransform2x2) {
    auto checker = makeSanityMultiDimChecker(
        "Exponential Transform 2x2",
        exponentialTransform2x2<double>,
        exponentialTransform2x2<fdouble>,
        getExponentialTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, LogarithmicTransform2x2) {
    auto checker = makeSanityMultiDimChecker(
        "Logarithmic Transform 2x2",
        logarithmicTransform2x2<double>,
        logarithmicTransform2x2<fdouble>,
        getLogarithmicTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, PolynomialTransform2x3) {
    auto checker = makeSanityMultiDimChecker(
        "Polynomial Transform 2x3",
        polynomialTransform2x3<double>,
        polynomialTransform2x3<fdouble>,
        getPolynomialTransform2x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, RationalTransform2x2) {
    auto checker = makeSanityMultiDimChecker(
        "Rational Transform 2x2",
        rationalTransform2x2<double>,
        rationalTransform2x2<fdouble>,
        getRationalTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, MixedTransform3x4) {
    auto checker = makeSanityMultiDimChecker(
        "Mixed Transform 3x4",
        mixedTransform3x4<double>,
        mixedTransform3x4<fdouble>,
        getMixedTransform3x4Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, NormalizationTransform3x3) {
    auto checker = makeSanityMultiDimChecker(
        "Normalization Transform 3x3",
        normalizationTransform3x3<double>,
        normalizationTransform3x3<fdouble>,
        getNormalizationTransform3x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityMultiDimToolTest, SigmoidTransform2x2) {
    auto checker = makeSanityMultiDimChecker(
        "Sigmoid Transform 2x2",
        sigmoidTransform2x2<double>,
        sigmoidTransform2x2<fdouble>,
        getSigmoidTransform2x2Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}


TEST_F(SanityMultiDimToolTest, SoftmaxTransform3x3) {
    auto checker = makeSanityMultiDimChecker(
        "Softmax Transform 3x3",
        softmaxTransform3x3<double>,
        softmaxTransform3x3<fdouble>,
        getSoftmaxTransform3x3Inputs(),
        config_
    );
    EXPECT_TRUE(checker.RunTests());
}