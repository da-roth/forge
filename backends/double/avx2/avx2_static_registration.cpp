// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file avx2_static_registration.cpp
 * @brief Static registration of AVX2 backend when bundled
 *
 * This file is compiled only when FORGE_BUNDLE_AVX2 is ON.
 * It registers the AVX2 instruction set and buffer creator with the factories
 * at static initialization time.
 */

#include "avx2_instruction_set.hpp"
#include "avx2_node_value_buffer.hpp"
#include "compiler/x86/common/instruction_set_factory.hpp"
#include "compiler/interfaces/node_value_buffer.hpp"

namespace forge {
namespace internal {

// AVX2 buffer creator function (defined in .cpp, not header)
static std::unique_ptr<INodeValueBuffer> createAVX2Buffer(
    const Graph& optimizedTape,
    const std::vector<NodeId>& mapping,
    size_t requiredNodes) {
    return std::make_unique<AVX2NodeValueBuffer>(optimizedTape, mapping, requiredNodes);
}

// Combined registrar for both instruction set and buffer creator
// Using a struct ensures both registrations happen together
struct AVX2BackendRegistrar {
    AVX2BackendRegistrar() {
        // Register the instruction set
        InstructionSetFactory::registerInstructionSet(
            "AVX2-Packed",
            []() { return std::make_unique<AVX2InstructionSet>(); }
        );

        // Register the buffer creator for vector width 4
        NodeValueBufferFactory::registerBufferCreator(4, createAVX2Buffer);
    }
};

// Static instance ensures registration happens at program startup
static AVX2BackendRegistrar s_avx2BackendRegistrar;

} // namespace internal
} // namespace forge

// Force the linker to include this object file even if nothing references it
// This function is exported with C linkage to ensure the symbol is visible
// and prevents the linker from stripping the object file
extern "C" void forge_force_avx2_registration() {
    // Reference the static to prevent dead code elimination
    (void)&forge::internal::s_avx2BackendRegistrar;
}
