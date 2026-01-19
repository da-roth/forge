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
#include "avx2_node_value_buffer.hpp"
#include "compiler/x86/common/instruction_set_factory.hpp"
#include "compiler/interfaces/node_value_buffer.hpp"

namespace {

// AVX2 buffer creator function for dynamic loading
std::unique_ptr<forge::INodeValueBuffer> createAVX2Buffer(
    const forge::Graph& optimizedTape,
    const std::vector<forge::NodeId>& mapping,
    size_t requiredNodes) {
    return std::make_unique<forge::AVX2NodeValueBuffer>(optimizedTape, mapping, requiredNodes);
}

} // anonymous namespace

/**
 * @brief Entry point for dynamic backend loading
 *
 * This function is called by InstructionSetFactory::loadBackend() when
 * the shared library is loaded. It registers the AVX2 instruction set
 * and buffer creator with the factory.
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
    // Register instruction set
    forge::InstructionSetFactory::registerInstructionSet(
        "AVX2-Packed",
        []() { return std::make_unique<forge::AVX2InstructionSet>(); }
    );

    // Register buffer creator
    forge::NodeValueBufferFactory::registerAVX2BufferCreator(createAVX2Buffer);
}

} // extern "C"
