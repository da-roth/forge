#pragma once

#include "forge/core/opcodes.h"
#include "forge/compiler/utils/compilation_timer.h"
#include <string>

namespace forge::compiler::generators {

/**
 * Utility functions for working with operation codes in assembly generation.
 * These functions provide debugging and diagnostic support for the ASM stitcher.
 */

/**
 * Convert an OpCode to its string representation for debugging and logging.
 * @param op The operation code to convert
 * @return String name of the operation (e.g., "Add", "Mul", "Exp")
 */
inline std::string getOpName(forge::core::OpCode op) {
    return forge::compiler::utils::CompilationTimer::getOpName(op);
}

} // namespace forge::compiler::generators