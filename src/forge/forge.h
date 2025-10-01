#pragma once

// Main Forge include file
// This is the primary header for the Forge library

// Core components
#include "forge/core/opcodes.h"
#include "forge/core/computation_graph.h"

// X86 instruction sets and registers
#include "forge/x86/instruction_set.h"
#include "forge/x86/register_allocator.h"
#include "forge/x86/xmm_register_allocator.h"
#include "forge/x86/ymm_register_allocator.h"

// Runtime components
#include "forge/runtime/runtime.h"

// Compiler components
#include "forge/compiler/analysis/stability_cleaner.h"
#include "forge/compiler/utils/compilation_timer.h"
#include "forge/compiler/generators/constant_pool_manager.h"
#include "forge/compiler/generators/operation_utils.h"
#include "forge/compiler/generators/instruction_set_factory.h"
#include "forge/compiler/generators/register_utils.h"
#include "forge/compiler/operations/arithmetic_operations.h"
#include "forge/compiler/operations/math_functions.h"
#include "forge/compiler/operations/comparison_control.h"
#include "forge/compiler/operations/boolean_operations.h"
#include "forge/compiler/operations/integer_operations.h"
#include "forge/compiler/forward_compiler.h"
#include "forge/compiler/reverse_gradient_compiler.h"
#include "forge/compiler/forge_engine.h"

namespace forge {

// Version information
constexpr const char* VERSION = "0.1.0";

} // namespace forge