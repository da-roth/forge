#pragma once

// Include all optimization passes
#include "inactive_folding.hpp"
#include "common_subexpression_elimination.hpp"
#include "algebraic_simplification.hpp"
#include "stability_cleaning.hpp"
#include "constant_cleanup.hpp"

namespace forge {
namespace optimizations {

// This header provides access to all optimization passes
// Individual optimizations can be used independently or through the main GraphOptimizer

} // namespace optimizations
} // namespace forge
