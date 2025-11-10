#include <gtest/gtest.h>
#include "../tools/sanityTool/sanity_checker.hpp"
#include "../tools/testFunctions/oneToOne/all.hpp"

using namespace forge::tools;
using namespace forge::tools::test_functions::one_to_one;
using namespace forge;

// Test fixture for sanity checker tests
class SanityToolTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test polynomial functions
TEST_F(SanityToolTest, LinearFunction) {
    auto checker = makeSanityChecker("Linear", 
                                     linear<double>, 
                                     linear<fdouble>, 
                                     getPolynomialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, QuadraticFunction) {
    auto checker = makeSanityChecker("Quadratic", 
                                     quadratic<double>, 
                                     quadratic<fdouble>, 
                                     getPolynomialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, CubicFunction) {
    auto checker = makeSanityChecker("Cubic", 
                                     cubic<double>, 
                                     cubic<fdouble>, 
                                     getPolynomialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, SineApproxFunction) {
    auto checker = makeSanityChecker("Sine Approximation", 
                                     sineApprox<double>, 
                                     sineApprox<fdouble>, 
                                     getPolynomialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, CosineApproxFunction) {
    auto checker = makeSanityChecker("Cosine Approximation", 
                                     cosineApprox<double>, 
                                     cosineApprox<fdouble>, 
                                     getPolynomialInputs());
    EXPECT_TRUE(checker.RunTests());
}

// Test trigonometric functions
TEST_F(SanityToolTest, SineFunction) {
    auto checker = makeSanityChecker("Sine", 
                                     sine<double>, 
                                     sine<fdouble>, 
                                     getTrigonometricInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, CosineFunction) {
    auto checker = makeSanityChecker("Cosine", 
                                     cosine<double>, 
                                     cosine<fdouble>, 
                                     getTrigonometricInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, TangentFunction) {
    auto checker = makeSanityChecker("Tangent", 
                                     tangent<double>, 
                                     tangent<fdouble>, 
                                     getTangentInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, SinTestFunction) {
    auto checker = makeSanityChecker("Sin Test", 
                                     sinTest<double>, 
                                     sinTest<fdouble>, 
                                     getTrigonometricInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, CosTestFunction) {
    auto checker = makeSanityChecker("Cos Test", 
                                     cosTest<double>, 
                                     cosTest<fdouble>, 
                                     getTrigonometricInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, TanTestFunction) {
    auto checker = makeSanityChecker("Tan Test", 
                                     tanTest<double>, 
                                     tanTest<fdouble>, 
                                     getTangentInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, TrigComboFunction) {
    auto checker = makeSanityChecker("Trig Combo", 
                                     trigCombo<double>, 
                                     trigCombo<fdouble>, 
                                     getTrigonometricInputs());
    EXPECT_TRUE(checker.RunTests());
}

// Test exponential functions
TEST_F(SanityToolTest, ExponentialFunction) {
    auto checker = makeSanityChecker("Exponential", 
                                     expScaled<double>, 
                                     expScaled<fdouble>, 
                                     getSafeExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, LogarithmFunction) {
    auto checker = makeSanityChecker("Logarithm", 
                                     logConditioned<double>, 
                                     logConditioned<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, SquareRootFunction) {
    auto checker = makeSanityChecker("Square Root", 
                                     sqrtWithOps<double>, 
                                     sqrtWithOps<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ExpTestFunction) {
    auto checker = makeSanityChecker("Exp Test", 
                                     expTest<double>, 
                                     expTest<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, LogTestFunction) {
    auto checker = makeSanityChecker("Log Test", 
                                     logTest<double>, 
                                     logTest<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, SqrtTestFunction) {
    auto checker = makeSanityChecker("Sqrt Test", 
                                     sqrtTest<double>, 
                                     sqrtTest<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, TranscendentalComboFunction) {
    auto checker = makeSanityChecker("Transcendental Combo", 
                                     transcendentalCombo<double>, 
                                     transcendentalCombo<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

// Test power functions (now implemented)
TEST_F(SanityToolTest, PowerTestFunction) {
    auto checker = makeSanityChecker("Power Test", 
                                     powerTest<double>, 
                                     powerTest<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, PowerIntegerTestFunction) {
    auto checker = makeSanityChecker("Power Integer Test", 
                                     powerIntegerTest<double>, 
                                     powerIntegerTest<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, PowerFractionalTestFunction) {
    auto checker = makeSanityChecker("Power Fractional Test", 
                                     powerFractionalTest<double>, 
                                     powerFractionalTest<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, PowerComplexTestFunction) {
    auto checker = makeSanityChecker("Power Complex Test", 
                                     powerComplexTest<double>, 
                                     powerComplexTest<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

// New power tests with negative bases and extreme exponents
TEST_F(SanityToolTest, PowerNegativeBaseIntTestFunction) {
    auto checker = makeSanityChecker("Power Negative Base (Odd Int)", 
                                     powerNegativeBaseIntTest<double>, 
                                     powerNegativeBaseIntTest<fdouble>, 
                                     getPowerExtremeInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, PowerNegativeBaseEvenTestFunction) {
    auto checker = makeSanityChecker("Power Negative Base (Even Int)", 
                                     powerNegativeBaseEvenTest<double>, 
                                     powerNegativeBaseEvenTest<fdouble>, 
                                     getPowerExtremeInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, PowerSmallExponentTestFunction) {
    auto checker = makeSanityChecker("Power Small Exponent (0.01)", 
                                     powerSmallExponentTest<double>, 
                                     powerSmallExponentTest<fdouble>, 
                                     getPowerExtremeInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, PowerLargeBaseSmallExpTestFunction) {
    auto checker = makeSanityChecker("Power 40^0.01 Test", 
                                     powerLargeBaseSmallExpTest<double>, 
                                     powerLargeBaseSmallExpTest<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, PowerNegativeCubeRootTestFunction) {
    auto checker = makeSanityChecker("Power Negative Cube Root", 
                                     powerNegativeCubeRootTest<double>, 
                                     powerNegativeCubeRootTest<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, PowerVaryingBaseAndExpTestFunction) {
    auto checker = makeSanityChecker("Power Varying Base/Exp", 
                                     powerVaryingBaseAndExpTest<double>, 
                                     powerVaryingBaseAndExpTest<fdouble>, 
                                     getExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, PowerTowerTestFunction) {
    auto checker = makeSanityChecker("Power Tower x^(x^2+1)", 
                                     powerTowerTest<double>, 
                                     powerTowerTest<fdouble>, 
                                     getSafeExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

// Test massive graph functions (inspired by big computation patterns)
TEST_F(SanityToolTest, SmallIterativeGraphFunction) {
    auto checker = makeSanityChecker("Small Iterative Graph (~1K ops)", 
                                     smallIterativeGraph<double>, 
                                     smallIterativeGraph<fdouble>, 
                                     getSmallGraphInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, MediumIterativeGraphFunction) {
    auto checker = makeSanityChecker("Medium Iterative Graph (~10K ops)", 
                                     mediumIterativeGraph<double>, 
                                     mediumIterativeGraph<fdouble>, 
                                     getBigGraphInputs());
    EXPECT_TRUE(checker.RunTests());
}

// Test rational functions
TEST_F(SanityToolTest, InverseFunction) {
    auto checker = makeSanityChecker("Inverse", 
                                     inverse<double>, 
                                     inverse<fdouble>, 
                                     getSafeRationalInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, RationalFunction) {
    auto checker = makeSanityChecker("Rational", 
                                     rationalFunction<double>, 
                                     rationalFunction<fdouble>, 
                                     getRationalInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, GaussianLikeFunction) {
    auto checker = makeSanityChecker("Gaussian-like", 
                                     gaussianLike<double>, 
                                     gaussianLike<fdouble>, 
                                     getRationalInputs());
    EXPECT_TRUE(checker.RunTests());
}

// Test special functions
TEST_F(SanityToolTest, ClampFunction) {
    auto checker = makeSanityChecker("Clamp", 
                                     clamp<double>, 
                                     clamp<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ModuloFunction) {
    auto checker = makeSanityChecker("Modulo", 
                                     moduloAbs<double>, 
                                     moduloAbs<fdouble>, 
                                     getModuloInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, MixedOperations) {
    auto checker = makeSanityChecker("Mixed Operations", 
                                     mixedOperations<double>, 
                                     mixedOperations<fdouble>, 
                                     getSafeExponentialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, Compound1Function) {
    auto checker = makeSanityChecker("Compound 1", 
                                     compound1<double>, 
                                     compound1<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, Compound2Function) {
    auto checker = makeSanityChecker("Compound 2", 
                                     compound2<double>, 
                                     compound2<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, NegationFunction) {
    auto checker = makeSanityChecker("Negation", 
                                     negation<double>, 
                                     negation<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, SquaredFunction) {
    auto checker = makeSanityChecker("Squared", 
                                     squared<double>, 
                                     squared<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ReciprocalFunction) {
    auto checker = makeSanityChecker("Reciprocal", 
                                     reciprocal<double>, 
                                     reciprocal<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, MinTestFunction) {
    auto checker = makeSanityChecker("Min Test", 
                                     minTest<double>, 
                                     minTest<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, MaxTestFunction) {
    auto checker = makeSanityChecker("Max Test", 
                                     maxTest<double>, 
                                     maxTest<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, MinMaxComboFunction) {
    auto checker = makeSanityChecker("MinMax Combo", 
                                     minmaxCombo<double>, 
                                     minmaxCombo<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, Ops10Function) {
    auto checker = makeSanityChecker("Ops 10", 
                                     ops10<double>, 
                                     ops10<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, Ops50Function) {
    auto checker = makeSanityChecker("Ops 50", 
                                     ops50<double>, 
                                     ops50<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, MassiveComplexFunction) {
    auto checker = makeSanityChecker("Massive Complex", 
                                     massiveComplex<double>, 
                                     massiveComplex<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, UltraMassiveIterativeFunction) {
    auto checker = makeSanityChecker("Ultra Massive Iterative", 
                                     [](double x) { return ultraMassiveIterative<double>(x, 10); }, 
                                     [](fdouble x) { return ultraMassiveIterative<fdouble>(x, 10); }, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ExpNegativeStressFunction) {
    auto checker = makeSanityChecker("Exp Negative Stress", 
                                     expNegativeStress<double>, 
                                     expNegativeStress<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

// ===== Isolated exp() debugging tests =====

TEST_F(SanityToolTest, JustExp) {
    auto checker = makeSanityChecker("Just Exp", 
                                     justExp<double>, 
                                     justExp<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, JustAddition) {
    auto checker = makeSanityChecker("Just Addition", 
                                     justAddition<double>, 
                                     justAddition<fdouble>, 
                                     std::vector<double>{0.0, -1.0, 2.5});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ExpDivideConstant) {
    auto checker = makeSanityChecker("Exp Divide Constant", 
                                     expDivideConstant<double>, 
                                     expDivideConstant<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ExpSelfDivide) {
    auto checker = makeSanityChecker("Exp Self Divide", 
                                     expSelfDivide<double>, 
                                     expSelfDivide<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, TwoExpCalls) {
    auto checker = makeSanityChecker("Two Exp Calls", 
                                     twoExpCalls<double>, 
                                     twoExpCalls<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, TwoExpWithDiv) {
    auto checker = makeSanityChecker("Two Exp With Div", 
                                     twoExpWithDiv<double>, 
                                     twoExpWithDiv<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ExpComplexExpr) {
    auto checker = makeSanityChecker("Exp Complex Expr", 
                                     expComplexExpr<double>, 
                                     expComplexExpr<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ExpMultipleDivisions) {
    auto checker = makeSanityChecker("Exp Multiple Divisions", 
                                     expMultipleDivisions<double>, 
                                     expMultipleDivisions<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ExpWithStorage) {
    auto checker = makeSanityChecker("Exp With Storage", 
                                     expWithStorage<double>, 
                                     expWithStorage<fdouble>, 
                                     std::vector<double>{0.5, 1.0, 2.0, 3.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, MinimalAmericanPattern) {
    auto checker = makeSanityChecker("Minimal American Pattern", 
                                     minimalAmericanPattern<double>, 
                                     minimalAmericanPattern<fdouble>, 
                                     std::vector<double>{80, 90, 100, 110, 120});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, MinimalPatternNoExp) {
    auto checker = makeSanityChecker("Minimal Pattern No Exp", 
                                     minimalPatternNoExp<double>, 
                                     minimalPatternNoExp<fdouble>, 
                                     std::vector<double>{80, 90, 100, 110, 120});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ExpImmediateUse) {
    auto checker = makeSanityChecker("Exp Immediate Use", 
                                     expImmediateUse<double>, 
                                     expImmediateUse<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ExpStoredUse) {
    auto checker = makeSanityChecker("Exp Stored Use", 
                                     expStoredUse<double>, 
                                     expStoredUse<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, SimplestExpFail) {
    auto checker = makeSanityChecker("Simplest Exp Fail", 
                                     simplestExpFail<double>, 
                                     simplestExpFail<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ConstantAfterExp) {
    auto checker = makeSanityChecker("Constant After Exp", 
                                     constantAfterExp<double>, 
                                     constantAfterExp<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, RegisterCorruptionTest) {
    auto checker = makeSanityChecker("Register Corruption Test", 
                                     registerCorruptionTest<double>, 
                                     registerCorruptionTest<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, AbsoluteFunction) {
    auto checker = makeSanityChecker("Absolute", 
                                     absolute<double>, 
                                     absolute<fdouble>, 
                                     getSpecialInputs());
    EXPECT_TRUE(checker.RunTests());
}

// Note: Power functions have been removed as pow operation is not yet implemented in JIT compiler
// Test comparison-based functions
TEST_F(SanityToolTest, SignFunction) {
    auto checker = makeSanityChecker("Sign", 
                                     signFunc<double>, 
                                     signFunc<fdouble>, 
                                     getComparisonInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, PiecewiseLinearFunction) {
    auto checker = makeSanityChecker("Piecewise Linear", 
                                     piecewiseLinear<double>, 
                                     piecewiseLinear<fdouble>, 
                                     getPiecewiseInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, PiecewiseQuadraticFunction) {
    auto checker = makeSanityChecker("Piecewise Quadratic", 
                                     piecewiseQuadratic<double>, 
                                     piecewiseQuadratic<fdouble>, 
                                     getPiecewiseInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, RampFunction) {
    auto checker = makeSanityChecker("Ramp (ReLU)", 
                                     rampFunction<double>, 
                                     rampFunction<fdouble>, 
                                     getComparisonInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, LeakyReLUFunction) {
    auto checker = makeSanityChecker("Leaky ReLU", 
                                     [](double x) { return leakyReLU<double>(x, 0.1); }, 
                                     [](fdouble x) { return leakyReLU<fdouble>(x, fdouble(0.1)); }, 
                                     getComparisonInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, Min3Function) {
    auto checker = makeSanityChecker("Min of 3", 
                                     min3<double>, 
                                     min3<fdouble>, 
                                     getComparisonInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, Max3Function) {
    auto checker = makeSanityChecker("Max of 3", 
                                     max3<double>, 
                                     max3<fdouble>, 
                                     getComparisonInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, Median3Function) {
    auto checker = makeSanityChecker("Median of 3", 
                                     median3<double>, 
                                     median3<fdouble>, 
                                     getComparisonInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, StepFunction) {
    auto checker = makeSanityChecker("Step Function", 
                                     [](double x) { return stepFunction<double>(x, 0.0); }, 
                                     [](fdouble x) { return stepFunction<fdouble>(x, fdouble(0.0)); }, 
                                     getComparisonInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, ComplexPiecewiseFunction) {
    auto checker = makeSanityChecker("Complex Piecewise", 
                                     complexPiecewise<double>, 
                                     complexPiecewise<fdouble>, 
                                     getPiecewiseInputs());
    EXPECT_TRUE(checker.RunTests());
}

// Diagnostic tests to isolate comparison/select issues
TEST_F(SanityToolTest, DiagnosticSimpleSelect) {
    auto checker = makeSanityChecker("Diagnostic: Simple Select", 
                                     diagnosticSimpleSelect<double>, 
                                     diagnosticSimpleSelect<fdouble>, 
                                     std::vector<double>{-1, 0, 0.5, 1});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticMediumSmallLoop) {
    auto checker = makeSanityChecker("DiagnosticMedium: SmallLoop",
                                     diagnosticMedium_small_loop<double>,
                                     diagnosticMedium_small_loop<fdouble>,
                                     std::vector<double>{0.5, 1.0});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticComparisonOnly) {
    auto checker = makeSanityChecker("Diagnostic: Comparison Only", 
                                     diagnosticComparisonOnly<double>, 
                                     diagnosticComparisonOnly<fdouble>, 
                                     std::vector<double>{0, 0.25, 0.5, 0.75, 1});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticNestedSelect) {
    auto checker = makeSanityChecker("Diagnostic: Nested Select", 
                                     diagnosticNestedSelect<double>, 
                                     diagnosticNestedSelect<fdouble>, 
                                     std::vector<double>{-1, 0, 0.5, 1, 2});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticFailingSegment) {
    auto checker = makeSanityChecker("Diagnostic: Failing Segment", 
                                     diagnosticFailingSegment<double>, 
                                     diagnosticFailingSegment<fdouble>, 
                                     std::vector<double>{-0.5, 0, 0.5, 1, 1.5});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticConditionValues) {
    auto checker = makeSanityChecker("Diagnostic: Condition Values", 
                                     diagnosticConditionValues<double>, 
                                     diagnosticConditionValues<fdouble>, 
                                     std::vector<double>{-3, -1, 0, 0.5, 1, 2, 3});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticMinimalNesting) {
    auto checker = makeSanityChecker("Diagnostic: Minimal Nesting", 
                                     diagnosticMinimalNesting<double>, 
                                     diagnosticMinimalNesting<fdouble>, 
                                     std::vector<double>{-0.5, 0, 0.5, 1, 1.5});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticDirectIf) {
    auto checker = makeSanityChecker("Diagnostic: Direct If", 
                                     diagnosticDirectIf<double>, 
                                     diagnosticDirectIf<fdouble>, 
                                     std::vector<double>{0, 0.25, 0.5, 0.75, 1});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticThreeLevelNesting) {
    auto checker = makeSanityChecker("Diagnostic: Three Level Nesting", 
                                     diagnosticThreeLevelNesting<double>, 
                                     diagnosticThreeLevelNesting<fdouble>, 
                                     std::vector<double>{-2, -0.5, 0, 0.5, 1, 2});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticFourLevelNesting) {
    auto checker = makeSanityChecker("Diagnostic: Four Level Nesting", 
                                     diagnosticFourLevelNesting<double>, 
                                     diagnosticFourLevelNesting<fdouble>, 
                                     std::vector<double>{-3, -1, 0, 0.5, 1, 2, 3, 4});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticExactConstants) {
    auto checker = makeSanityChecker("Diagnostic: Exact Constants", 
                                     diagnosticExactConstants<double>, 
                                     diagnosticExactConstants<fdouble>, 
                                     std::vector<double>{-0.5, 0, 0.5, 1, 1.5});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticWithExpressions) {
    auto checker = makeSanityChecker("Diagnostic: With Expressions", 
                                     diagnosticWithExpressions<double>, 
                                     diagnosticWithExpressions<fdouble>, 
                                     std::vector<double>{-1, 0, 0.5, 1, 2});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticComplexDebug) {
    auto checker = makeSanityChecker("Diagnostic: Complex Debug", 
                                     diagnosticComplexDebug<double>, 
                                     diagnosticComplexDebug<fdouble>, 
                                     std::vector<double>{-3, -1, 0, 0.5, 1, 2, 3, 4});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticZeroSource) {
    auto checker = makeSanityChecker("Diagnostic: Zero Source", 
                                     diagnosticZeroSource<double>, 
                                     diagnosticZeroSource<fdouble>, 
                                     std::vector<double>{-1, 0, 0.5, 1});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticComplexNegativeOnly) {
    auto checker = makeSanityChecker("Diagnostic: Complex Negative Only", 
                                     diagnosticComplexNegativeOnly<double>, 
                                     diagnosticComplexNegativeOnly<fdouble>, 
                                     std::vector<double>{-3, -2, -1, 0, 0.5, 1});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticComplexFirstThree) {
    auto checker = makeSanityChecker("Diagnostic: Complex First Three", 
                                     diagnosticComplexFirstThree<double>, 
                                     diagnosticComplexFirstThree<fdouble>, 
                                     std::vector<double>{-3, -2, -1, 0, 0.5, 1, 2});
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, DiagnosticExactCopy) {
    auto checker = makeSanityChecker("Diagnostic: Exact Copy", 
                                     diagnosticExactCopy<double>, 
                                     diagnosticExactCopy<fdouble>, 
                                     std::vector<double>{-3, -1, 0, 0.5, 1, 2, 3});
    EXPECT_TRUE(checker.RunTests());
}

// Test massive expression functions (converted from test_functions_1d.hpp)
TEST_F(SanityToolTest, MassiveExpressionFunction) {
    auto checker = makeSanityChecker("Massive Expression", 
                                     massiveExpression<double>, 
                                     massiveExpression<fdouble>, 
                                     getMassiveExpressionInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, UltraMassiveIterative10Function) {
    auto checker = makeSanityChecker("Ultra Massive Iterative (10 iterations)", 
                                     ultraMassiveIterative10<double>, 
                                     ultraMassiveIterative10<fdouble>, 
                                     getUltraMassiveInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, UltraMassiveIterative100Function) {
    auto checker = makeSanityChecker("Ultra Massive Iterative (100 iterations)", 
                                     ultraMassiveIterative100<double>, 
                                     ultraMassiveIterative100<fdouble>, 
                                     getUltraMassiveInputs());
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, UltraMassiveIterative1000Function) {
    auto checker = makeSanityChecker("Ultra Massive Iterative (1000 iterations)", 
                                     ultraMassiveIterative1000<double>, 
                                     ultraMassiveIterative1000<fdouble>, 
                                     getUltraMassiveInputs());
    EXPECT_TRUE(checker.RunTests());
}

// Test American and European options
TEST_F(SanityToolTest, AmericanPutFunction) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American Put", 
                                     americanPut<double>, 
                                     americanPut<fdouble>, 
                                     getAmericanOptionInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, AmericanCallFunction) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American Call", 
                                     americanCall<double>, 
                                     americanCall<fdouble>, 
                                     getAmericanOptionInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, EuropeanPutFunction) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("European Put", 
                                     europeanPut<double>, 
                                     europeanPut<fdouble>, 
                                     getAmericanOptionInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Test simple conditional operations to isolate the bug
TEST_F(SanityToolTest, SimpleConditionalTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("Simple Conditional", 
                                     simpleConditionalTest<double>, 
                                     simpleConditionalTest<fdouble>, 
                                     getPiecewiseInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Test vector-like conditional operations 
TEST_F(SanityToolTest, VectorLikeConditionalTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("Vector-Like Conditional", 
                                     vectorLikeConditionalTest<double>, 
                                     vectorLikeConditionalTest<fdouble>, 
                                     getPiecewiseInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Test std::vector operations (the likely culprit!)
TEST_F(SanityToolTest, StdVectorTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("Std Vector Test", 
                                     stdVectorTest<double>, 
                                     stdVectorTest<fdouble>, 
                                     getPiecewiseInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Test to isolate the vectorization issue with SSE2
TEST_F(SanityToolTest, VectorizedMaxIssueTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("Vectorized Max Issue (SSE2)", 
                                     vectorizedMaxIssue<double>, 
                                     vectorizedMaxIssue<fdouble>, 
                                     std::vector<double>{100.0, 99.999, 100.001, 80.0, 120.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Test the exact American option pattern that should fail!
TEST_F(SanityToolTest, AmericanOptionPatternTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American Option Pattern", 
                                     americanOptionPattern<double>, 
                                     americanOptionPattern<fdouble>, 
                                     getAmericanOptionInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Test the EXACT American option code with transcendental functions!
TEST_F(SanityToolTest, ExactAmericanPatternTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("Exact American Pattern", 
                                     exactAmericanPattern<double>, 
                                     exactAmericanPattern<fdouble>, 
                                     getAmericanOptionInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Progressive isolation tests to find exact culprit

TEST_F(SanityToolTest, AmericanPatternNoSqrtTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American No Sqrt", 
                                     americanPatternNoSqrt<double>, 
                                     americanPatternNoSqrt<fdouble>, 
                                     getAmericanOptionInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, AmericanPatternNoExpTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American No Exp", 
                                     americanPatternNoExp<double>, 
                                     americanPatternNoExp<fdouble>, 
                                     getAmericanOptionInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, AmericanPatternNoDivisionTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American No Division", 
                                     americanPatternNoDivision<double>, 
                                     americanPatternNoDivision<fdouble>, 
                                     getAmericanOptionInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// SelectDiagnostic Test 1: Array assignment WITHOUT select (SSE2)
TEST_F(SanityToolTest, SelectDiagnosticArrayNoSelectTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Array No Select (SSE2)", 
                                     selectDiagnosticArrayNoSelect<double>, 
                                     selectDiagnosticArrayNoSelect<fdouble>, 
                                     std::vector<double>{100.0, 99.999, 100.001, 80.0, 120.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// SelectDiagnostic Test 2: Select with divergent lanes (SSE2)
TEST_F(SanityToolTest, SelectDiagnosticLaneDivergenceTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Lane Divergence (SSE2)", 
                                     selectDiagnosticLaneDivergence<double>, 
                                     selectDiagnosticLaneDivergence<fdouble>, 
                                     std::vector<double>{100.0, 99.9995, 100.0015, 100.0025, 100.003},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// SelectDiagnostic Test 3: Arrays WITH select (SSE2)
TEST_F(SanityToolTest, SelectDiagnosticArrayWithSelectTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Array With Select (SSE2)", 
                                     selectDiagnosticArrayWithSelect<double>, 
                                     selectDiagnosticArrayWithSelect<fdouble>, 
                                     std::vector<double>{100.0, 99.999, 100.001, 80.0, 120.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// SelectDiagnostic Test 4: Simple chained select (SSE2)
TEST_F(SanityToolTest, SelectDiagnosticSimpleChainedTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Simple Chained (SSE2)", 
                                     selectDiagnosticSimpleChained<double>, 
                                     selectDiagnosticSimpleChained<fdouble>, 
                                     std::vector<double>{100.0, 100.6, 101.0, 99.5, 102.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// SelectDiagnostic Test 5: Select divergence WITHOUT arrays (SSE2)
TEST_F(SanityToolTest, SelectDiagnosticDivergenceNoArrayTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Divergence No Array (SSE2)", 
                                     selectDiagnosticDivergenceNoArray<double>, 
                                     selectDiagnosticDivergenceNoArray<fdouble>, 
                                     std::vector<double>{100.0, 99.999, 100.001, 80.0, 120.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolTest, SimpleTranscendentalTest) {
    SanityConfig config;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("Simple Transcendental", 
                                     simpleTranscendentalTest<double>, 
                                     simpleTranscendentalTest<fdouble>, 
                                     getAmericanOptionInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}