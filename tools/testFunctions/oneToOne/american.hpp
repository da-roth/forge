#pragma once

#include <cmath>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <algorithm>
#include "../select_helper.hpp"

namespace forge {
namespace tools {
namespace test_functions {
namespace one_to_one {

using namespace forge::tools::test_functions;

// ============================================================================
// COMPREHENSIVE AMERICAN OPTION PRICING WITH BINOMIAL TREE
// Based on the test implementation from tests/American/test_american_option.hpp
// This provides a more realistic example that includes:
// - Term structure of interest rates
// - Time-dependent volatility
// - Proper binomial tree construction
// - Virtual function calls to stress the JIT compiler
// ============================================================================

// -------------------- Financial Curves --------------------
template<typename T>
class FinancialCurve {
public:
    virtual ~FinancialCurve() = default;
    virtual T GetValue(const T& t) const = 0;
    virtual std::string Name() const = 0;
};

// Flat curve - constant rate/vol
template<typename T>
class FlatCurve : public FinancialCurve<T> {
private:
    std::string name_;
    double level_;
    
public:
    FlatCurve(const std::string& name, double level) 
        : name_(name), level_(level) {}
    
    T GetValue(const T& t) const override { 
        return T(level_); 
    }
    
    std::string Name() const override { 
        return name_; 
    }
};

// Two-point curve with linear interpolation
template<typename T>
class TwoPointCurve : public FinancialCurve<T> {
private:
    std::string name_;
    double tenor1_;
    double tenor2_;
    double value1_;
    double value2_;
    
public:
    TwoPointCurve(const std::string& name, 
                  double tenor1, double tenor2,
                  double value1, double value2)
        : name_(name), tenor1_(tenor1), tenor2_(tenor2), 
          value1_(value1), value2_(value2) {}
    
    T GetValue(const T& t) const override {
        T t1(tenor1_);
        T t2(tenor2_);
        T v1(value1_);
        T v2(value2_);
        
        // Compute alpha for interpolation
        T alpha = (t - t1) / (t2 - t1);
        T interpolated = v1 * (T(1.0) - alpha) + v2 * alpha;
        
        // Handle boundary conditions
        auto is_before = t <= t1;
        auto is_after = t >= t2;
        
        // Use select helper for conditional logic
        return select(is_before, v1, select(is_after, v2, interpolated));
    }
    
    std::string Name() const override { 
        return name_; 
    }
};

// Volatility smile curve - time-dependent vol
template<typename T>
class VolatilitySmileCurve : public FinancialCurve<T> {
private:
    std::string name_;
    double base_vol_;
    double smile_factor_;
    
public:
    VolatilitySmileCurve(const std::string& name, double base_vol, double smile_factor)
        : name_(name), base_vol_(base_vol), smile_factor_(smile_factor) {}
    
    T GetValue(const T& t) const override {
        // Vol increases with time (term structure effect)
        return T(base_vol_) * (T(1.0) + t * T(0.1));
    }
    
    std::string Name() const override { 
        return name_; 
    }
};

// -------------------- Market Data Repository --------------------
template<typename T>
class MarketDataRepository {
public:
    virtual ~MarketDataRepository() = default;
    virtual std::shared_ptr<FinancialCurve<T>> GetCurve(const std::string& key) const = 0;
};

template<typename T>
class MapMarketDataRepository : public MarketDataRepository<T> {
private:
    std::map<std::string, std::shared_ptr<FinancialCurve<T>>> curves_;
    
public:
    void Add(const std::string& key, std::shared_ptr<FinancialCurve<T>> curve) {
        curves_[key] = curve;
    }
    
    std::shared_ptr<FinancialCurve<T>> GetCurve(const std::string& key) const override {
        auto it = curves_.find(key);
        if (it == curves_.end()) {
            return std::make_shared<FlatCurve<T>>("default", 0.0);
        }
        return it->second;
    }
};

// -------------------- Payoff Interface --------------------
template<typename T>
class Payoff {
public:
    virtual ~Payoff() = default;
    virtual T Compute(const T& spot) const = 0;
    virtual std::string Name() const = 0;
};

template<typename T>
class AmericanPutPayoff : public Payoff<T> {
private:
    double K_;
    
public:
    explicit AmericanPutPayoff(double K) : K_(K) {}
    
    T Compute(const T& spot) const override {
        T payoff_value = T(K_) - spot;
        auto is_positive = payoff_value > T(0.0);
        return select(is_positive, payoff_value, T(0.0));
    }
    
    std::string Name() const override {
        return "AmericanPut";
    }
};

template<typename T>
class AmericanCallPayoff : public Payoff<T> {
private:
    double K_;
    
public:
    explicit AmericanCallPayoff(double K) : K_(K) {}
    
    T Compute(const T& spot) const override {
        T payoff_value = spot - T(K_);
        auto is_positive = payoff_value > T(0.0);
        return select(is_positive, payoff_value, T(0.0));
    }
    
    std::string Name() const override {
        return "AmericanCall";
    }
};

// -------------------- Exercise Policy --------------------
template<typename T>
class ExercisePolicy {
public:
    virtual ~ExercisePolicy() = default;
    virtual T ShouldExercise(const T& t, const T& spot, 
                            const T& continuation, const T& intrinsic) const = 0;
};

template<typename T>
class DefaultAmericanPolicy : public ExercisePolicy<T> {
public:
    T ShouldExercise(const T& t, const T& spot, 
                    const T& continuation, const T& intrinsic) const override {
        // Returns 1.0 if should exercise, 0.0 otherwise
        auto should_ex = intrinsic >= continuation;
        return select(should_ex, T(1.0), T(0.0));
    }
};

// -------------------- Binomial Parameters Provider --------------------
template<typename T>
class BinomialParametersProvider {
public:
    virtual ~BinomialParametersProvider() = default;
    
    struct Parameters {
        T u, d, p, disc;
    };
    
    virtual Parameters Compute(const T& t, const T& dt, 
                               const MarketDataRepository<T>& repo,
                               const T& spot) const = 0;
};

template<typename T>
class CRRParametersProvider : public BinomialParametersProvider<T> {
private:
    std::string rateKey_;
    std::string volKey_;
    
public:
    CRRParametersProvider(const std::string& rateKey, const std::string& volKey)
        : rateKey_(rateKey), volKey_(volKey) {}
    
    typename BinomialParametersProvider<T>::Parameters 
    Compute(const T& t, const T& dt,
            const MarketDataRepository<T>& repo,
            const T& spot) const override {
        // Virtual calls through repo
        auto rCurve = repo.GetCurve(rateKey_);
        auto vCurve = repo.GetCurve(volKey_);
        
        T r = rCurve->GetValue(t);
        T sigma = vCurve->GetValue(t);
        
        // Cox-Ross-Rubinstein formulas
        T a = std::exp(sigma * std::sqrt(dt));
        typename BinomialParametersProvider<T>::Parameters params;
        params.u = a;
        params.d = T(1.0) / a;
        T erdt = std::exp(r * dt);
        params.p = (erdt - params.d) / (params.u - params.d);
        params.disc = T(1.0) / erdt;
        
        return params;
    }
};

// -------------------- Main American Option Functions --------------------

// American Put with comprehensive binomial tree
template<typename T>
T americanPut(T spot) {
    // Create market data repository
    auto repo = std::make_shared<MapMarketDataRepository<T>>();
    
    // Two-point term structure: rate at t=0 is 1%, rate at t=1 is 2%
    repo->Add("IR.risk_free", 
              std::make_shared<TwoPointCurve<T>>("IR.risk_free", 0.0, 1.0, 0.01, 0.02));
    
    // Time-dependent volatility
    repo->Add("VOL.equity", 
              std::make_shared<VolatilitySmileCurve<T>>("VOL.equity", 0.25, 0.1));
    
    // Create components
    auto payoff = std::make_shared<AmericanPutPayoff<T>>(100.0);
    auto policy = std::make_shared<DefaultAmericanPolicy<T>>();
    auto params_provider = std::make_shared<CRRParametersProvider<T>>("IR.risk_free", "VOL.equity");
    
    // Configuration
    const int steps = 5;  // More steps than the simple version for better accuracy
    const T maturity(1.0);
    const T dt = maturity / T(steps);
    
    // Build binomial tree
    std::vector<T> current_level;
    std::vector<T> next_level;
    
    // Initialize terminal nodes
    current_level.resize(steps + 1);
    for (int j = 0; j <= steps; ++j) {
        T S = spot;
        
        // Get parameters at final time
        T t_final = maturity - dt;
        auto bin_params = params_provider->Compute(t_final, dt, *repo, S);
        
        // Apply up and down moves
        for (int k = 0; k < j; ++k) {
            S = S * bin_params.u;
        }
        for (int k = 0; k < (steps - j); ++k) {
            S = S * bin_params.d;
        }
        
        // Terminal payoff
        current_level[j] = payoff->Compute(S);
    }
    
    // Backward induction
    for (int i = steps - 1; i >= 0; --i) {
        T t = T(i) * dt;
        next_level = current_level;
        current_level.resize(i + 1);
        
        for (int j = 0; j <= i; ++j) {
            T S = spot;
            
            // Recompute parameters (virtual calls add complexity)
            auto bin_params = params_provider->Compute(t, dt, *repo, S);
            
            // Apply moves
            for (int k = 0; k < j; ++k) {
                S = S * bin_params.u;
            }
            for (int k = 0; k < (i - j); ++k) {
                S = S * bin_params.d;
            }
            
            // Continuation value
            T cont_up = next_level[j + 1];
            T cont_down = next_level[j];
            T continuation = bin_params.disc * 
                (bin_params.p * cont_up + (T(1.0) - bin_params.p) * cont_down);
            
            // Early exercise value
            T intrinsic = payoff->Compute(S);
            
            // Exercise decision using policy
            T should_ex = policy->ShouldExercise(t, S, continuation, intrinsic);
            auto should_exercise = should_ex > T(0.5);
            current_level[j] = select(should_exercise, intrinsic, continuation);
            
            // Add small transaction cost
            current_level[j] = current_level[j] * T(0.9999);
        }
    }
    
    return current_level[0];
}

// American Call with comprehensive binomial tree
template<typename T>
T americanCall(T spot) {
    // Create market data repository
    auto repo = std::make_shared<MapMarketDataRepository<T>>();
    
    // Two-point term structure
    repo->Add("IR.risk_free", 
              std::make_shared<TwoPointCurve<T>>("IR.risk_free", 0.0, 1.0, 0.01, 0.02));
    
    // Time-dependent volatility
    repo->Add("VOL.equity", 
              std::make_shared<VolatilitySmileCurve<T>>("VOL.equity", 0.25, 0.1));
    
    // Create components
    auto payoff = std::make_shared<AmericanCallPayoff<T>>(100.0);
    auto policy = std::make_shared<DefaultAmericanPolicy<T>>();
    auto params_provider = std::make_shared<CRRParametersProvider<T>>("IR.risk_free", "VOL.equity");
    
    // Configuration
    const int steps = 5;
    const T maturity(1.0);
    const T dt = maturity / T(steps);
    
    // Build binomial tree
    std::vector<T> current_level;
    std::vector<T> next_level;
    
    // Initialize terminal nodes
    current_level.resize(steps + 1);
    for (int j = 0; j <= steps; ++j) {
        T S = spot;
        
        T t_final = maturity - dt;
        auto bin_params = params_provider->Compute(t_final, dt, *repo, S);
        
        for (int k = 0; k < j; ++k) {
            S = S * bin_params.u;
        }
        for (int k = 0; k < (steps - j); ++k) {
            S = S * bin_params.d;
        }
        
        current_level[j] = payoff->Compute(S);
    }
    
    // Backward induction
    for (int i = steps - 1; i >= 0; --i) {
        T t = T(i) * dt;
        next_level = current_level;
        current_level.resize(i + 1);
        
        for (int j = 0; j <= i; ++j) {
            T S = spot;
            
            auto bin_params = params_provider->Compute(t, dt, *repo, S);
            
            for (int k = 0; k < j; ++k) {
                S = S * bin_params.u;
            }
            for (int k = 0; k < (i - j); ++k) {
                S = S * bin_params.d;
            }
            
            T cont_up = next_level[j + 1];
            T cont_down = next_level[j];
            T continuation = bin_params.disc * 
                (bin_params.p * cont_up + (T(1.0) - bin_params.p) * cont_down);
            
            T intrinsic = payoff->Compute(S);
            
            T should_ex = policy->ShouldExercise(t, S, continuation, intrinsic);
            auto should_exercise = should_ex > T(0.5);
            current_level[j] = select(should_exercise, intrinsic, continuation);
            
            current_level[j] = current_level[j] * T(0.9999);
        }
    }
    
    return current_level[0];
}

// European Put for comparison (no early exercise)
template<typename T>
T europeanPut(T spot) {
    // Create market data repository
    auto repo = std::make_shared<MapMarketDataRepository<T>>();
    
    repo->Add("IR.risk_free", 
              std::make_shared<TwoPointCurve<T>>("IR.risk_free", 0.0, 1.0, 0.01, 0.02));
    repo->Add("VOL.equity", 
              std::make_shared<VolatilitySmileCurve<T>>("VOL.equity", 0.25, 0.1));
    
    auto payoff = std::make_shared<AmericanPutPayoff<T>>(100.0);
    auto params_provider = std::make_shared<CRRParametersProvider<T>>("IR.risk_free", "VOL.equity");
    
    const int steps = 5;
    const T maturity(1.0);
    const T dt = maturity / T(steps);
    
    std::vector<T> current_level;
    std::vector<T> next_level;
    
    // Terminal payoffs
    current_level.resize(steps + 1);
    for (int j = 0; j <= steps; ++j) {
        T S = spot;
        
        T t_final = maturity - dt;
        auto bin_params = params_provider->Compute(t_final, dt, *repo, S);
        
        for (int k = 0; k < j; ++k) {
            S = S * bin_params.u;
        }
        for (int k = 0; k < (steps - j); ++k) {
            S = S * bin_params.d;
        }
        
        current_level[j] = payoff->Compute(S);
    }
    
    // Backward induction (no early exercise)
    for (int i = steps - 1; i >= 0; --i) {
        T t = T(i) * dt;
        next_level = current_level;
        current_level.resize(i + 1);
        
        for (int j = 0; j <= i; ++j) {
            T S = spot;
            
            auto bin_params = params_provider->Compute(t, dt, *repo, S);
            
            for (int k = 0; k < j; ++k) {
                S = S * bin_params.u;
            }
            for (int k = 0; k < (i - j); ++k) {
                S = S * bin_params.d;
            }
            
            T cont_up = next_level[j + 1];
            T cont_down = next_level[j];
            current_level[j] = bin_params.disc * 
                (bin_params.p * cont_up + (T(1.0) - bin_params.p) * cont_down);
            
            // Transaction cost
            current_level[j] = current_level[j] * T(0.9999);
        }
    }
    
    return current_level[0];
}

// Test input sets for American options
inline std::vector<double> getAmericanOptionInputs() {
    // Spot prices around the strike (100)
    return {80, 85, 90, 95, 100, 105, 110, 115, 120};
}

} // namespace one_to_one
} // namespace test_functions
} // namespace tools
} // namespace forge