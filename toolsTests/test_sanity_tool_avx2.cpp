#include <gtest/gtest.h>
#include "../tools/sanityTool/sanity_checker.hpp"
#include "../tools/testFunctions/oneToOne/all.hpp"
#include "../tools/testFunctions/oneToOne/comparisons_specialized.hpp"

using namespace forge::tools;
using namespace forge::tools::test_functions::one_to_one;
using namespace forge;

// Test fixture for AVX2 sanity checker tests
class SanityToolAVX2Test : public ::testing::Test {
protected:
    SanityConfig avx2Config_;
    
    void SetUp() override {
        // Configure to use AVX2 backend
        avx2Config_.instructionSet = forge::CompilerConfig::InstructionSet::AVX2_PACKED;
        // AVX2 might need more relaxed tolerances due to different instruction sequences
        avx2Config_.absoluteTolerance = 1e-9;
        avx2Config_.relativeTolerance = 1e-9;
        // For debugging, only run one iteration
        avx2Config_.warmupIterations = 0;
        avx2Config_.timingIterations = 1;
    }
    
    void TearDown() override {}
};

// Test polynomial functions
TEST_F(SanityToolAVX2Test, LinearFunction) {
    auto checker = makeSanityChecker("Linear", 
                                     linear<double>, 
                                     linear<fdouble>, 
                                     getPolynomialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, QuadraticFunction) {
    auto checker = makeSanityChecker("Quadratic", 
                                     quadratic<double>, 
                                     quadratic<fdouble>, 
                                     getPolynomialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, CubicFunction) {
    auto checker = makeSanityChecker("Cubic", 
                                     cubic<double>, 
                                     cubic<fdouble>, 
                                     getPolynomialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, SineApproxFunction) {
    auto checker = makeSanityChecker("Sine Approximation", 
                                     sineApprox<double>, 
                                     sineApprox<fdouble>, 
                                     getPolynomialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, CosineApproxFunction) {
    auto checker = makeSanityChecker("Cosine Approximation", 
                                     cosineApprox<double>, 
                                     cosineApprox<fdouble>, 
                                     getPolynomialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

// Test trigonometric functions
TEST_F(SanityToolAVX2Test, SineFunction) {
    auto checker = makeSanityChecker("Sine", 
                                     sine<double>, 
                                     sine<fdouble>, 
                                     getTrigonometricInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, CosineFunction) {
    auto checker = makeSanityChecker("Cosine", 
                                     cosine<double>, 
                                     cosine<fdouble>, 
                                     getTrigonometricInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, TangentFunction) {
    auto checker = makeSanityChecker("Tangent", 
                                     tangent<double>, 
                                     tangent<fdouble>, 
                                     getTangentInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, TangentFunction2) {
    auto checker = makeSanityChecker("Tangent", 
                                     tangent<double>, 
                                     tangent<fdouble>, 
                                     getTangentInputsShort(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, SinTestFunction) {
    auto checker = makeSanityChecker("Sin Test", 
                                     sinTest<double>, 
                                     sinTest<fdouble>, 
                                     getTrigonometricInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, CosTestFunction) {
    auto checker = makeSanityChecker("Cos Test", 
                                     cosTest<double>, 
                                     cosTest<fdouble>, 
                                     getTrigonometricInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, TanTestFunction) {
    auto checker = makeSanityChecker("Tan Test", 
                                     tanTest<double>, 
                                     tanTest<fdouble>, 
                                     getTangentInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, TrigComboFunction) {
    auto checker = makeSanityChecker("Trig Combo", 
                                     trigCombo<double>, 
                                     trigCombo<fdouble>, 
                                     getTrigonometricInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

// Test exponential functions
TEST_F(SanityToolAVX2Test, ExponentialFunction) {
    auto checker = makeSanityChecker("Exponential", 
                                     expScaled<double>, 
                                     expScaled<fdouble>, 
                                     getSafeExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, LogarithmFunction) {
    auto checker = makeSanityChecker("Logarithm", 
                                     logConditioned<double>, 
                                     logConditioned<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, SquareRootFunction) {
    auto checker = makeSanityChecker("Square Root", 
                                     sqrtWithOps<double>, 
                                     sqrtWithOps<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ExpTestFunction) {
    auto checker = makeSanityChecker("Exp Test", 
                                     expTest<double>, 
                                     expTest<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, LogTestFunction) {
    auto checker = makeSanityChecker("Log Test", 
                                     logTest<double>, 
                                     logTest<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, SqrtTestFunction) {
    auto checker = makeSanityChecker("Sqrt Test", 
                                     sqrtTest<double>, 
                                     sqrtTest<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, TranscendentalComboFunction) {
    auto checker = makeSanityChecker("Transcendental Combo", 
                                     transcendentalCombo<double>, 
                                     transcendentalCombo<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

// Test power functions (now implemented)
TEST_F(SanityToolAVX2Test, PowerTestFunction) {
    auto checker = makeSanityChecker("Power Test", 
                                     powerTest<double>, 
                                     powerTest<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, PowerIntegerTestFunction) {
    auto checker = makeSanityChecker("Power Integer Test", 
                                     powerIntegerTest<double>, 
                                     powerIntegerTest<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, PowerFractionalTestFunction) {
    auto checker = makeSanityChecker("Power Fractional Test", 
                                     powerFractionalTest<double>, 
                                     powerFractionalTest<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, PowerComplexTestFunction) {
    auto checker = makeSanityChecker("Power Complex Test", 
                                     powerComplexTest<double>, 
                                     powerComplexTest<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

// New power tests with negative bases and extreme exponents
TEST_F(SanityToolAVX2Test, PowerNegativeBaseIntTestFunction) {
    auto checker = makeSanityChecker("Power Negative Base (Odd Int)", 
                                     powerNegativeBaseIntTest<double>, 
                                     powerNegativeBaseIntTest<fdouble>, 
                                     getPowerExtremeInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, PowerNegativeBaseEvenTestFunction) {
    auto checker = makeSanityChecker("Power Negative Base (Even Int)", 
                                     powerNegativeBaseEvenTest<double>, 
                                     powerNegativeBaseEvenTest<fdouble>, 
                                     getPowerExtremeInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, PowerSmallExponentTestFunction) {
    auto checker = makeSanityChecker("Power Small Exponent (0.01)", 
                                     powerSmallExponentTest<double>, 
                                     powerSmallExponentTest<fdouble>, 
                                     getPowerExtremeInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, PowerLargeBaseSmallExpTestFunction) {
    auto checker = makeSanityChecker("Power 40^0.01 Test", 
                                     powerLargeBaseSmallExpTest<double>, 
                                     powerLargeBaseSmallExpTest<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, PowerNegativeCubeRootTestFunction) {
    auto checker = makeSanityChecker("Power Negative Cube Root", 
                                     powerNegativeCubeRootTest<double>, 
                                     powerNegativeCubeRootTest<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, PowerVaryingBaseAndExpTestFunction) {
    auto checker = makeSanityChecker("Power Varying Base/Exp", 
                                     powerVaryingBaseAndExpTest<double>, 
                                     powerVaryingBaseAndExpTest<fdouble>, 
                                     getExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, PowerTowerTestFunction) {
    auto checker = makeSanityChecker("Power Tower x^(x^2+1)", 
                                     powerTowerTest<double>, 
                                     powerTowerTest<fdouble>, 
                                     getSafeExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

// Test massive graph functions (inspired by big computation patterns)
TEST_F(SanityToolAVX2Test, SmallIterativeGraphFunction) {
    auto checker = makeSanityChecker("Small Iterative Graph (~1K ops)", 
                                     smallIterativeGraph<double>, 
                                     smallIterativeGraph<fdouble>, 
                                     std::vector<double>{0.5, 1.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, MediumIterativeGraphFunction) {
    SanityConfig config = avx2Config_;
    // AVX2 path now uses vectorized IF; loosen tolerances for this large graph
    config.absoluteTolerance = 1e-2;
    config.relativeTolerance = 2e-2;
    auto checker = makeSanityChecker("Medium Iterative Graph (~10K ops)", 
                                     mediumIterativeGraph<double>, 
                                     mediumIterativeGraph<fdouble>, 
                                     std::vector<double>{0.5, 1.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Diagnostic medium-size tests focused on IF/select and tan/exp
TEST_F(SanityToolAVX2Test, DiagnosticMediumIfChain) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-2; // allow small relative diff for tan
    auto checker = makeSanityChecker("DiagnosticMedium: IfChain",
                                     diagnosticMedium_if_chain<double>,
                                     diagnosticMedium_if_chain<fdouble>,
                                     getDiagnosticMediumInputs(),
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticMediumMaskReuse) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-2;
    auto checker = makeSanityChecker("DiagnosticMedium: MaskReuse",
                                     diagnosticMedium_mask_reuse<double>,
                                     diagnosticMedium_mask_reuse<fdouble>,
                                     std::vector<double>{0.5, 1.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticMediumSmallLoop) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-2;
    auto checker = makeSanityChecker("DiagnosticMedium: SmallLoop",
                                     diagnosticMedium_small_loop<double>,
                                     diagnosticMedium_small_loop<fdouble>,
                                     std::vector<double>{0.5, 1.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Test rational functions
TEST_F(SanityToolAVX2Test, InverseFunction) {
    auto checker = makeSanityChecker("Inverse", 
                                     inverse<double>, 
                                     inverse<fdouble>, 
                                     getSafeRationalInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, RationalFunction) {
    auto checker = makeSanityChecker("Rational", 
                                     rationalFunction<double>, 
                                     rationalFunction<fdouble>, 
                                     getRationalInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, GaussianLikeFunction) {
    auto checker = makeSanityChecker("Gaussian-like", 
                                     gaussianLike<double>, 
                                     gaussianLike<fdouble>, 
                                     getRationalInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

// Test special functions
TEST_F(SanityToolAVX2Test, ClampFunction) {
    auto checker = makeSanityChecker("Clamp", 
                                     clamp<double>, 
                                     clamp<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ModuloFunction) {
    auto checker = makeSanityChecker("Modulo", 
                                     moduloAbs<double>, 
                                     moduloAbs<fdouble>, 
                                     getModuloInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, MixedOperations) {
    auto checker = makeSanityChecker("Mixed Operations", 
                                     mixedOperations<double>, 
                                     mixedOperations<fdouble>, 
                                     getSafeExponentialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, Compound1Function) {
    auto checker = makeSanityChecker("Compound 1", 
                                     compound1<double>, 
                                     compound1<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, Compound2Function) {
    auto checker = makeSanityChecker("Compound 2", 
                                     compound2<double>, 
                                     compound2<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, NegationFunction) {
    auto checker = makeSanityChecker("Negation", 
                                     negation<double>, 
                                     negation<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, SquaredFunction) {
    auto checker = makeSanityChecker("Squared", 
                                     squared<double>, 
                                     squared<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ReciprocalFunction) {
    auto checker = makeSanityChecker("Reciprocal", 
                                     reciprocal<double>, 
                                     reciprocal<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, MinTestFunction) {
    auto checker = makeSanityChecker("Min Test", 
                                     minTest<double>, 
                                     minTest<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, MaxTestFunction) {
    auto checker = makeSanityChecker("Max Test", 
                                     maxTest<double>, 
                                     maxTest<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, MinMaxComboFunction) {
    auto checker = makeSanityChecker("MinMax Combo", 
                                     minmaxCombo<double>, 
                                     minmaxCombo<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, Ops10Function) {
    auto checker = makeSanityChecker("Ops 10", 
                                     ops10<double>, 
                                     ops10<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, Ops50Function) {
    auto checker = makeSanityChecker("Ops 50", 
                                     ops50<double>, 
                                     ops50<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, MassiveComplexFunction) {
    auto checker = makeSanityChecker("Massive Complex", 
                                     massiveComplex<double>, 
                                     massiveComplex<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, UltraMassiveIterativeFunction) {
    auto checker = makeSanityChecker("Ultra Massive Iterative", 
                                     [](double x) { return ultraMassiveIterative<double>(x, 10); }, 
                                     [](fdouble x) { return ultraMassiveIterative<fdouble>(x, 10); }, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ExpNegativeStressFunction) {
    auto checker = makeSanityChecker("Exp Negative Stress", 
                                     expNegativeStress<double>, 
                                     expNegativeStress<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

// ===== Isolated exp() debugging tests =====

TEST_F(SanityToolAVX2Test, JustExp) {
    auto checker = makeSanityChecker("Just Exp", 
                                     justExp<double>, 
                                     justExp<fdouble>, 
                                     std::vector<double>{-0.5, 0.0, 0.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustAddition) {
    auto checker = makeSanityChecker("Just Addition", 
                                     justAddition<double>, 
                                     justAddition<fdouble>, 
                                     std::vector<double>{-2.0, 0.0, 1.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustSubtraction) {
    auto checker = makeSanityChecker("Just Subtraction", 
                                     justSubtraction<double>, 
                                     justSubtraction<fdouble>, 
                                     std::vector<double>{-1.0, 0.0, 3.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustMultiplication) {
    auto checker = makeSanityChecker("Just Multiplication", 
                                     justMultiplication<double>, 
                                     justMultiplication<fdouble>, 
                                     std::vector<double>{-1.5, 0.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustDivision) {
    auto checker = makeSanityChecker("Just Division", 
                                     justDivision<double>, 
                                     justDivision<fdouble>, 
                                     std::vector<double>{-4.0, 1.0, 6.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustNegation) {
    auto checker = makeSanityChecker("Just Negation", 
                                     justNegation<double>, 
                                     justNegation<fdouble>, 
                                     std::vector<double>{-2.5, 0.0, 1.8},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustAbsolute) {
    auto checker = makeSanityChecker("Just Absolute", 
                                     justAbsolute<double>, 
                                     justAbsolute<fdouble>, 
                                     std::vector<double>{-3.2, 0.0, 2.1},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustReciprocal) {
    auto checker = makeSanityChecker("Just Reciprocal", 
                                     justReciprocal<double>, 
                                     justReciprocal<fdouble>, 
                                     std::vector<double>{-2.0, 0.5, 4.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustSquareRoot) {
    auto checker = makeSanityChecker("Just Square Root", 
                                     justSquareRoot<double>, 
                                     justSquareRoot<fdouble>, 
                                     std::vector<double>{1.0, 4.0, 9.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustLogarithm) {
    auto checker = makeSanityChecker("Just Logarithm", 
                                     justLogarithm<double>, 
                                     justLogarithm<fdouble>, 
                                     std::vector<double>{0.5, 1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustLessThan) {
    auto checker = makeSanityChecker("Just Less Than", 
                                     justLessThan_double, 
                                     justLessThan_doubleTP, 
                                     std::vector<double>{0.5, 1.0, 1.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustLessEqual) {
    auto checker = makeSanityChecker("Just Less Equal", 
                                     justLessEqual_double, 
                                     justLessEqual_doubleTP, 
                                     std::vector<double>{0.5, 1.0, 1.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustGreaterThan) {
    auto checker = makeSanityChecker("Just Greater Than", 
                                     justGreaterThan_double, 
                                     justGreaterThan_doubleTP, 
                                     std::vector<double>{0.5, 1.0, 1.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustGreaterEqual) {
    auto checker = makeSanityChecker("Just Greater Equal", 
                                     justGreaterEqual_double, 
                                     justGreaterEqual_doubleTP, 
                                     std::vector<double>{0.5, 1.0, 1.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustEqual) {
    auto checker = makeSanityChecker("Just Equal", 
                                     justEqual_double, 
                                     justEqual_doubleTP, 
                                     std::vector<double>{0.5, 1.0, 1.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustNotEqual) {
    auto checker = makeSanityChecker("Just Not Equal", 
                                     justNotEqual_double, 
                                     justNotEqual_doubleTP, 
                                     std::vector<double>{0.5, 1.0, 1.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustPower) {
    auto checker = makeSanityChecker("Just Power", 
                                     justPower<double>, 
                                     justPower<fdouble>, 
                                     std::vector<double>{1.5, 2.0, 3.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustModulo) {
    auto checker = makeSanityChecker("Just Modulo", 
                                     justModulo<double>, 
                                     justModulo<fdouble>, 
                                     std::vector<double>{2.5, 5.0, 8.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustIf) {
    auto checker = makeSanityChecker("Just If", 
                                     justIf<double>, 
                                     justIf<fdouble>, 
                                     std::vector<double>{-1.0, 0.0, 1.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustAddition3) {
    auto checker = makeSanityChecker("Just Addition 3-operand", 
                                     justAddition3<double>, 
                                     justAddition3<fdouble>, 
                                     std::vector<double>{-2.0, 0.0, 1.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustSubtraction3) {
    auto checker = makeSanityChecker("Just Subtraction 3-operand", 
                                     justSubtraction3<double>, 
                                     justSubtraction3<fdouble>, 
                                     std::vector<double>{-1.0, 0.0, 5.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustMultiplication3) {
    auto checker = makeSanityChecker("Just Multiplication 3-operand", 
                                     justMultiplication3<double>, 
                                     justMultiplication3<fdouble>, 
                                     std::vector<double>{-0.5, 1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustDivision3) {
    auto checker = makeSanityChecker("Just Division 3-operand", 
                                     justDivision3<double>, 
                                     justDivision3<fdouble>, 
                                     std::vector<double>{-12.0, 6.0, 18.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustSquare) {
    auto checker = makeSanityChecker("Just Square", 
                                     justSquare<double>, 
                                     justSquare<fdouble>, 
                                     std::vector<double>{-3.0, 0.0, 2.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustSine) {
    auto checker = makeSanityChecker("Just Sine", 
                                     justSine<double>, 
                                     justSine<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustCosine) {
    auto checker = makeSanityChecker("Just Cosine", 
                                     justCosine<double>, 
                                     justCosine<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustTangent) {
    auto checker = makeSanityChecker("Just Tangent", 
                                     justTangent<double>, 
                                     justTangent<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustMinimum) {
    auto checker = makeSanityChecker("Just Minimum", 
                                     justMinimum<double>, 
                                     justMinimum<fdouble>, 
                                     std::vector<double>{1.0, 2.0, 3.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, JustMaximum) {
    auto checker = makeSanityChecker("Just Maximum", 
                                     justMaximum<double>, 
                                     justMaximum<fdouble>, 
                                     std::vector<double>{1.0, 2.0, 3.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}


TEST_F(SanityToolAVX2Test, ExpDivideConstant) {
    auto checker = makeSanityChecker("Exp Divide Constant", 
                                     expDivideConstant<double>, 
                                     expDivideConstant<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ExpSelfDivide) {
    auto checker = makeSanityChecker("Exp Self Divide", 
                                     expSelfDivide<double>, 
                                     expSelfDivide<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, TwoExpCalls) {
    auto checker = makeSanityChecker("Two Exp Calls", 
                                     twoExpCalls<double>, 
                                     twoExpCalls<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, TwoExpWithDiv) {
    auto checker = makeSanityChecker("Two Exp With Div", 
                                     twoExpWithDiv<double>, 
                                     twoExpWithDiv<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ExpComplexExpr) {
    auto checker = makeSanityChecker("Exp Complex Expr", 
                                     expComplexExpr<double>, 
                                     expComplexExpr<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ExpMultipleDivisions) {
    auto checker = makeSanityChecker("Exp Multiple Divisions", 
                                     expMultipleDivisions<double>, 
                                     expMultipleDivisions<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ExpWithStorage) {
    auto checker = makeSanityChecker("Exp With Storage", 
                                     expWithStorage<double>, 
                                     expWithStorage<fdouble>, 
                                     std::vector<double>{0.5, 1.0, 2.0, 3.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, MinimalAmericanPattern) {
    auto checker = makeSanityChecker("Minimal American Pattern", 
                                     minimalAmericanPattern<double>, 
                                     minimalAmericanPattern<fdouble>, 
                                     std::vector<double>{80, 90, 100, 110, 120},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, MinimalPatternNoExp) {
    auto checker = makeSanityChecker("Minimal Pattern No Exp", 
                                     minimalPatternNoExp<double>, 
                                     minimalPatternNoExp<fdouble>, 
                                     std::vector<double>{80, 90, 100, 110, 120},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ExpImmediateUse) {
    auto checker = makeSanityChecker("Exp Immediate Use", 
                                     expImmediateUse<double>, 
                                     expImmediateUse<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ExpStoredUse) {
    auto checker = makeSanityChecker("Exp Stored Use", 
                                     expStoredUse<double>, 
                                     expStoredUse<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, SimplestExpFail) {
    auto checker = makeSanityChecker("Simplest Exp Fail", 
                                     simplestExpFail<double>, 
                                     simplestExpFail<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ConstantAfterExp) {
    auto checker = makeSanityChecker("Constant After Exp", 
                                     constantAfterExp<double>, 
                                     constantAfterExp<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, RegisterCorruptionTest) {
    auto checker = makeSanityChecker("Register Corruption Test", 
                                     registerCorruptionTest<double>, 
                                     registerCorruptionTest<fdouble>, 
                                     std::vector<double>{0.0, 0.5, 1.0, -1.0, 2.0},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, AbsoluteFunction) {
    auto checker = makeSanityChecker("Absolute", 
                                     absolute<double>, 
                                     absolute<fdouble>, 
                                     getSpecialInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

// Note: Power functions have been removed as pow operation is not yet implemented in JIT compiler
// Test comparison-based functions
TEST_F(SanityToolAVX2Test, SignFunction) {
    auto checker = makeSanityChecker("Sign", 
                                     signFunc<double>, 
                                     signFunc<fdouble>, 
                                     getComparisonInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, PiecewiseLinearFunction) {
    auto checker = makeSanityChecker("Piecewise Linear", 
                                     piecewiseLinear<double>, 
                                     piecewiseLinear<fdouble>, 
                                     getPiecewiseInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, PiecewiseQuadraticFunction) {
    auto checker = makeSanityChecker("Piecewise Quadratic", 
                                     piecewiseQuadratic<double>, 
                                     piecewiseQuadratic<fdouble>, 
                                     getPiecewiseInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, RampFunction) {
    auto checker = makeSanityChecker("Ramp (ReLU)", 
                                     rampFunction<double>, 
                                     rampFunction<fdouble>, 
                                     getComparisonInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, LeakyReLUFunction) {
    auto checker = makeSanityChecker("Leaky ReLU", 
                                     [](double x) { return leakyReLU<double>(x, 0.1); }, 
                                     [](fdouble x) { return leakyReLU<fdouble>(x, fdouble(0.1)); }, 
                                     getComparisonInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, Min3Function) {
    auto checker = makeSanityChecker("Min of 3", 
                                     min3<double>, 
                                     min3<fdouble>, 
                                     getComparisonInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, Max3Function) {
    auto checker = makeSanityChecker("Max of 3", 
                                     max3<double>, 
                                     max3<fdouble>, 
                                     getComparisonInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, Median3Function) {
    auto checker = makeSanityChecker("Median of 3", 
                                     median3<double>, 
                                     median3<fdouble>, 
                                     getComparisonInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, StepFunction) {
    auto checker = makeSanityChecker("Step Function", 
                                     [](double x) { return stepFunction<double>(x, 0.0); }, 
                                     [](fdouble x) { return stepFunction<fdouble>(x, fdouble(0.0)); }, 
                                     getComparisonInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, ComplexPiecewiseFunction) {
    auto checker = makeSanityChecker("Complex Piecewise", 
                                     complexPiecewise<double>, 
                                     complexPiecewise<fdouble>, 
                                     getPiecewiseInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

// Diagnostic tests to isolate comparison/select issues
TEST_F(SanityToolAVX2Test, DiagnosticSimpleSelect) {
    auto checker = makeSanityChecker("Diagnostic: Simple Select", 
                                     diagnosticSimpleSelect<double>, 
                                     diagnosticSimpleSelect<fdouble>, 
                                     std::vector<double>{-1, 0, 0.5, 1},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticComparisonOnly) {
    auto checker = makeSanityChecker("Diagnostic: Comparison Only", 
                                     diagnosticComparisonOnly<double>, 
                                     diagnosticComparisonOnly<fdouble>, 
                                     std::vector<double>{0, 0.25, 0.5, 0.75, 1},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticNestedSelect) {
    auto checker = makeSanityChecker("Diagnostic: Nested Select", 
                                     diagnosticNestedSelect<double>, 
                                     diagnosticNestedSelect<fdouble>, 
                                     std::vector<double>{-1, 0, 0.5, 1, 2},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticFailingSegment) {
    auto checker = makeSanityChecker("Diagnostic: Failing Segment", 
                                     diagnosticFailingSegment<double>, 
                                     diagnosticFailingSegment<fdouble>, 
                                     std::vector<double>{-0.5, 0, 0.5, 1, 1.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticConditionValues) {
    auto checker = makeSanityChecker("Diagnostic: Condition Values", 
                                     diagnosticConditionValues<double>, 
                                     diagnosticConditionValues<fdouble>, 
                                     std::vector<double>{-3, -1, 0, 0.5, 1, 2, 3},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticMinimalNesting) {
    auto checker = makeSanityChecker("Diagnostic: Minimal Nesting", 
                                     diagnosticMinimalNesting<double>, 
                                     diagnosticMinimalNesting<fdouble>, 
                                     std::vector<double>{-0.5, 0, 0.5, 1, 1.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticDirectIf) {
    auto checker = makeSanityChecker("Diagnostic: Direct If", 
                                     diagnosticDirectIf<double>, 
                                     diagnosticDirectIf<fdouble>, 
                                     std::vector<double>{0, 0.25, 0.5, 0.75, 1},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticThreeLevelNesting) {
    auto checker = makeSanityChecker("Diagnostic: Three Level Nesting", 
                                     diagnosticThreeLevelNesting<double>, 
                                     diagnosticThreeLevelNesting<fdouble>, 
                                     std::vector<double>{-2, -0.5, 0, 0.5, 1, 2},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticFourLevelNesting) {
    auto checker = makeSanityChecker("Diagnostic: Four Level Nesting", 
                                     diagnosticFourLevelNesting<double>, 
                                     diagnosticFourLevelNesting<fdouble>, 
                                     std::vector<double>{-3, -1, 0, 0.5, 1, 2, 3, 4},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticExactConstants) {
    auto checker = makeSanityChecker("Diagnostic: Exact Constants", 
                                     diagnosticExactConstants<double>, 
                                     diagnosticExactConstants<fdouble>, 
                                     std::vector<double>{-0.5, 0, 0.5, 1, 1.5},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticWithExpressions) {
    auto checker = makeSanityChecker("Diagnostic: With Expressions", 
                                     diagnosticWithExpressions<double>, 
                                     diagnosticWithExpressions<fdouble>, 
                                     std::vector<double>{-1, 0, 0.5, 1, 2},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticComplexDebug) {
    auto checker = makeSanityChecker("Diagnostic: Complex Debug", 
                                     diagnosticComplexDebug<double>, 
                                     diagnosticComplexDebug<fdouble>, 
                                     std::vector<double>{-3,0.5, 1},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticComplexDebug2) {
    auto checker = makeSanityChecker("Diagnostic: Complex Debug2", 
                                     diagnosticComplexDebug2<double>, 
                                     diagnosticComplexDebug2<fdouble>, 
                                     std::vector<double>{-3, 0.5, 1},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticComplexDebug3) {
    auto checker = makeSanityChecker("Diagnostic: Complex Debug", 
                                     diagnosticComplexDebug3<double>, 
                                     diagnosticComplexDebug3<fdouble>, 
                                     std::vector<double>{-3, 0.5, 1},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticZeroSource) {
    auto checker = makeSanityChecker("Diagnostic: Zero Source", 
                                     diagnosticZeroSource<double>, 
                                     diagnosticZeroSource<fdouble>, 
                                     std::vector<double>{-1, 0, 0.5, 1},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticComplexNegativeOnly) {
    auto checker = makeSanityChecker("Diagnostic: Complex Negative Only", 
                                     diagnosticComplexNegativeOnly<double>, 
                                     diagnosticComplexNegativeOnly<fdouble>, 
                                     std::vector<double>{-3, -2, -1, 0, 0.5, 1},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticComplexFirstThree) {
    auto checker = makeSanityChecker("Diagnostic: Complex First Three", 
                                     diagnosticComplexFirstThree<double>, 
                                     diagnosticComplexFirstThree<fdouble>, 
                                     std::vector<double>{-3, -2, -1, 0, 0.5, 1, 2},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, DiagnosticExactCopy) {
    auto checker = makeSanityChecker("Diagnostic: Exact Copy", 
                                     diagnosticExactCopy<double>, 
                                     diagnosticExactCopy<fdouble>, 
                                     std::vector<double>{-3, -1, 0, 0.5, 1, 2, 3},
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

// Test massive expression functions (converted from test_functions_1d.hpp)
TEST_F(SanityToolAVX2Test, MassiveExpressionFunction) {
    auto checker = makeSanityChecker("Massive Expression", 
                                     massiveExpression<double>, 
                                     massiveExpression<fdouble>, 
                                     getMassiveExpressionInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, UltraMassiveIterative10Function) {
    auto checker = makeSanityChecker("Ultra Massive Iterative (10 iterations)", 
                                     ultraMassiveIterative10<double>, 
                                     ultraMassiveIterative10<fdouble>, 
                                     getUltraMassiveInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, UltraMassiveIterative100Function) {
    auto checker = makeSanityChecker("Ultra Massive Iterative (100 iterations)", 
                                     ultraMassiveIterative100<double>, 
                                     ultraMassiveIterative100<fdouble>, 
                                     getUltraMassiveInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, UltraMassiveIterative1000Function) {
    auto checker = makeSanityChecker("Ultra Massive Iterative (1000 iterations)", 
                                     ultraMassiveIterative1000<double>, 
                                     ultraMassiveIterative1000<fdouble>, 
                                     getUltraMassiveInputs(),
                                     avx2Config_);
    EXPECT_TRUE(checker.RunTests());
}

// Test American and European options
TEST_F(SanityToolAVX2Test, AmericanPutFunction) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American Put", 
                                     americanPut<double>, 
                                     americanPut<fdouble>, 
                                     std::vector<double>{100.0, 110.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, AmericanCallFunction) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American Call", 
                                     americanCall<double>, 
                                     americanCall<fdouble>, 
                                     std::vector<double>{100.0, 110.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, EuropeanPutFunction) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("European Put", 
                                     europeanPut<double>, 
                                     europeanPut<fdouble>, 
                                     std::vector<double>{100.0, 110.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Test simple conditional operations to isolate the bug
TEST_F(SanityToolAVX2Test, SimpleConditionalTest) {
    SanityConfig config = avx2Config_;
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
TEST_F(SanityToolAVX2Test, VectorLikeConditionalTest) {
    SanityConfig config = avx2Config_;
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
TEST_F(SanityToolAVX2Test, StdVectorTest) {
    SanityConfig config = avx2Config_;
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

// Simplified test to isolate the vectorization issue
TEST_F(SanityToolAVX2Test, VectorizedMaxIssueTest) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("Vectorized Max Issue", 
                                     vectorizedMaxIssue<double>, 
                                     vectorizedMaxIssue<fdouble>, 
                                     std::vector<double>{100.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// SelectDiagnostic Test 1: Array assignment WITHOUT select
TEST_F(SanityToolAVX2Test, SelectDiagnosticArrayNoSelectTest) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;  // Show all to see if it works
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Array No Select", 
                                     selectDiagnosticArrayNoSelect<double>, 
                                     selectDiagnosticArrayNoSelect<fdouble>, 
                                     std::vector<double>{100.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// // SelectDiagnostic Test 2: Select with divergent lanes (no arrays)
// TEST_F(SanityToolAVX2Test, SelectDiagnosticLaneDivergenceTest) {
//     SanityConfig config = avx2Config_;
//     config.absoluteTolerance = 1e-6;
//     config.relativeTolerance = 1e-6;
//     config.verbose = true;
//     config.showOnlyFailures = false;  // Show all to see lane behavior
//     config.timingIterations = 10;
    
//     auto checker = makeSanityChecker("SelectDiagnostic: Lane Divergence", 
//                                      selectDiagnosticLaneDivergence<double>, 
//                                      selectDiagnosticLaneDivergence<fdouble>, 
//                                      std::vector<double>{100.0, 99.9995, 100.0015, 100.0025, 100.003},
//                                      config);
//     EXPECT_TRUE(checker.RunTests());
// }

// SelectDiagnostic Test 3: Arrays WITH select - should show the issue
TEST_F(SanityToolAVX2Test, SelectDiagnosticArrayWithSelectTest) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;  // Show all to diagnose
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Array With Select", 
                                     selectDiagnosticArrayWithSelect<double>, 
                                     selectDiagnosticArrayWithSelect<fdouble>, 
                                     std::vector<double>{100.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// SelectDiagnostic Test 4: Simple chained select
TEST_F(SanityToolAVX2Test, SelectDiagnosticSimpleChainedTest) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;  // Show all to see behavior
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Simple Chained", 
                                     selectDiagnosticSimpleChained<double>, 
                                     selectDiagnosticSimpleChained<fdouble>, 
                                     std::vector<double>{100.0, 100.6, 101.0, 99.5, 102.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// SelectDiagnostic Test 5: Select divergence WITHOUT arrays
TEST_F(SanityToolAVX2Test, SelectDiagnosticDivergenceNoArrayTest) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;  // Show all to confirm it works
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Divergence No Array", 
                                     selectDiagnosticDivergenceNoArray<double>, 
                                     selectDiagnosticDivergenceNoArray<fdouble>, 
                                     std::vector<double>{100.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// New diagnostic tests for isolating AVX2 issue
TEST_F(SanityToolAVX2Test, SelectDiagnosticArrayWithSelectTest2) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Array With Select 2 (100 ops)", 
                                     selectDiagnosticArrayWithSelect2<double>, 
                                     selectDiagnosticArrayWithSelect2<fdouble>, 
                                     std::vector<double>{100.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, SelectDiagnosticArrayWithSelectTest3) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Array With Select 3 (no mul)", 
                                     selectDiagnosticArrayWithSelect3<double>, 
                                     selectDiagnosticArrayWithSelect3<fdouble>, 
                                     std::vector<double>{100.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, SelectDiagnosticArrayWithSelectTest4) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Array With Select 4 (no if)", 
                                     selectDiagnosticArrayWithSelect4<double>, 
                                     selectDiagnosticArrayWithSelect4<fdouble>, 
                                     std::vector<double>{100.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, SelectDiagnosticArrayWithSelectTest5) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Array With Select 5 (mul after select)", 
                                     selectDiagnosticArrayWithSelect5<double>, 
                                     selectDiagnosticArrayWithSelect5<fdouble>, 
                                     std::vector<double>{100.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, SelectDiagnosticArrayWithSelectTest6) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = false;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("SelectDiagnostic: Array With Select 6 (conditional mul)", 
                                     selectDiagnosticArrayWithSelect6<double>, 
                                     selectDiagnosticArrayWithSelect6<fdouble>, 
                                     std::vector<double>{100.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Test the exact American option pattern that should fail!
TEST_F(SanityToolAVX2Test, AmericanOptionPatternTest) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American Option Pattern", 
                                     americanOptionPattern<double>, 
                                     americanOptionPattern<fdouble>, 
                                     std::vector<double>{100.0, 110.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Test the EXACT American option code with transcendental functions!
TEST_F(SanityToolAVX2Test, ExactAmericanPatternTest) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;  // Only show failing entries
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("Exact American Pattern", 
                                     exactAmericanPattern<double>, 
                                     exactAmericanPattern<fdouble>, 
                                     std::vector<double>{100.0, 110.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

// Progressive isolation tests to find exact culprit

TEST_F(SanityToolAVX2Test, AmericanPatternNoSqrtTest) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American No Sqrt", 
                                     americanPatternNoSqrt<double>, 
                                     americanPatternNoSqrt<fdouble>, 
                                     std::vector<double>{100.0, 110.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, AmericanPatternNoExpTest) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American No Exp", 
                                     americanPatternNoExp<double>, 
                                     americanPatternNoExp<fdouble>, 
                                     std::vector<double>{100.0, 110.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, AmericanPatternNoDivisionTest) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("American No Division", 
                                     americanPatternNoDivision<double>, 
                                     americanPatternNoDivision<fdouble>, 
                                     std::vector<double>{100.0, 110.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}

TEST_F(SanityToolAVX2Test, SimpleTranscendentalTest) {
    SanityConfig config = avx2Config_;
    config.absoluteTolerance = 1e-6;
    config.relativeTolerance = 1e-6;
    config.verbose = true;
    config.showOnlyFailures = true;
    config.timingIterations = 10;
    
    auto checker = makeSanityChecker("Simple Transcendental", 
                                     simpleTranscendentalTest<double>, 
                                     simpleTranscendentalTest<fdouble>, 
                                     std::vector<double>{100.0, 110.0},
                                     config);
    EXPECT_TRUE(checker.RunTests());
}