// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file avx2_static_registration.cpp
 * @brief Static registration of AVX2 backend when bundled
 *
 * This file is compiled only when FORGE_BUNDLE_AVX2 is ON.
 * It registers the AVX2 instruction set with the factory at static
 * initialization time, making it available via createByName("AVX2-Packed").
 * It also registers the AVX2 buffer creator with the NodeValueBufferFactory.
 */

#include "avx2_instruction_set.hpp"
#include "avx2_node_value_buffer.hpp"
#include "compiler/x86/common/instruction_set_factory.hpp"
#include "compiler/interfaces/node_value_buffer.hpp"

namespace {

// Static registration of AVX2 instruction set
// This runs before main() and registers AVX2 in the factory's registry
static forge::InstructionSetRegistrar<forge::AVX2InstructionSet>
    s_avx2Registrar("AVX2-Packed");

// AVX2 buffer creator function
std::unique_ptr<forge::INodeValueBuffer> createAVX2Buffer(
    const forge::Graph& optimizedTape,
    const std::vector<forge::NodeId>& mapping,
    size_t requiredNodes) {
    return std::make_unique<forge::AVX2NodeValueBuffer>(optimizedTape, mapping, requiredNodes);
}

// Static registration of AVX2 buffer creator
struct AVX2BufferRegistrar {
    AVX2BufferRegistrar() {
        forge::NodeValueBufferFactory::registerAVX2BufferCreator(createAVX2Buffer);
    }
};
static AVX2BufferRegistrar s_avx2BufferRegistrar;

} // anonymous namespace
