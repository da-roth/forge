// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file avx2_backend.cpp
 * @brief Dynamic backend registration for AVX2
 *
 * This file is compiled into a shared library (libforge_avx2.so / forge_avx2.dll)
 * that can be loaded at runtime via InstructionSetFactory::loadBackend().
 *
 * Usage:
 *   InstructionSetFactory::loadBackend("./libforge_avx2.so");
 *   auto avx2 = InstructionSetFactory::createByName("AVX2-Packed");
 *
 * Build this file separately as a shared library with AVX2 flags:
 *   g++ -shared -fPIC -mavx2 -o libforge_avx2.so avx2_backend.cpp avx2_instruction_set.cpp ...
 */

#include "avx2_instruction_set.hpp"
#include "compiler/x86/common/instruction_set_factory.hpp"

/**
 * @brief Entry point for dynamic backend loading
 *
 * This function is called by InstructionSetFactory::loadBackend() when
 * the shared library is loaded. It registers the AVX2 instruction set
 * with the factory.
 *
 * The function must be exported with C linkage to avoid name mangling.
 */
extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
void forge_register_backend() {
    forge::InstructionSetFactory::registerInstructionSet(
        "AVX2-Packed",
        []() { return std::make_unique<forge::AVX2InstructionSet>(); }
    );
}

} // extern "C"
