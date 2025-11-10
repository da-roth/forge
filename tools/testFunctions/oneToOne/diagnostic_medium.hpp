#pragma once

#include <cmath>
#include <vector>
#include "../select_helper.hpp"

namespace forge {
namespace tools {
namespace test_functions {
namespace one_to_one {

// Smaller iterative graph focusing on IF/select and tan/exp usage
// Aim: replicate MediumIterativeGraph issues on a compact case

template<typename T>
T diagnosticMedium_if_chain(T x) {
	T a = x;
	T b = std::exp(x * T(0.2)) - T(1.0);
	T c = std::tan(x * T(0.3));
	T d = a + b - c;
	// Three nested selects with near-threshold comparisons
	auto c1 = (d > T(0.0));
	auto c2 = (b > c);
	auto c3 = (a + b < c + T(0.1));
	T r1 = select(c1, d * T(1.1), d * T(0.9));
	T r2 = select(c2, r1 + b * T(0.25), r1 - c * T(0.25));
	T r3 = select(c3, r2 * T(0.95) + a * T(0.05), r2 * T(1.05) - a * T(0.05));
	return r3;
}

// Same as above but with explicit mask reuse pattern (to stress vblendvpd path)

template<typename T>
T diagnosticMedium_mask_reuse(T x) {
	T e = std::exp(x * T(0.5));
	T t = std::tan(x * T(0.4));
	T s = e - t;
	auto m = (x > T(0.1));
	T y1 = select(m, s + T(0.2), s - T(0.2));
	// Reuse of m in a second select
	T y2 = select(m, y1 * e, y1 * t);
	// Different condition combining intermediate results
	auto n = (y2 < T(0.0));
	T y3 = select(n, -y2, y2 + T(0.1));
	return y3;
}

// A compact loop doing a handful of iterations with select and tan

template<typename T>
T diagnosticMedium_small_loop(T x) {
	T u = x * T(0.3);
	for (int i = 0; i < 6; ++i) {
		T e = std::exp(u);
		T t = std::tan(u);
		T z = e - t + T(0.05) * T(i);
		auto c = (z > T(0.2));
		u = select(c, z * T(0.8) + u * T(0.2), z * T(1.1) - u * T(0.1));
	}
	return u;
}

inline std::vector<double> getDiagnosticMediumInputs() {
	return {-1.0, -0.5, 0.0, 0.5, 1.0};
}

} // namespace one_to_one
} // namespace test_functions
} // namespace tools
} // namespace forge
