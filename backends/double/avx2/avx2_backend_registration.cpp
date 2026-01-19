// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file avx2_backend_registration.cpp
 * @brief Minimal buffer factory registration for loadable backends
 *
 * This file provides a self-contained implementation of NodeValueBufferFactory
 * for use in dynamically loadable backends. It avoids dependencies on the full
 * forge library to allow the DLL/shared library to be self-contained.
 *
 * Only compiled when building the loadable backend (FORGE_LOADABLE_BACKEND).
 */

#ifdef FORGE_LOADABLE_BACKEND

#include "compiler/interfaces/node_value_buffer.hpp"
#include <unordered_map>

namespace forge {

// Self-contained registry for loadable backends
namespace {
    std::unordered_map<int, NodeValueBufferFactory::BufferCreatorFunc>& getBufferCreatorRegistry() {
        static std::unordered_map<int, NodeValueBufferFactory::BufferCreatorFunc> registry;
        return registry;
    }
}

void NodeValueBufferFactory::registerBufferCreator(int vectorWidth, BufferCreatorFunc creator) {
    getBufferCreatorRegistry()[vectorWidth] = creator;
}

bool NodeValueBufferFactory::hasBufferCreator(int vectorWidth) {
    auto& registry = getBufferCreatorRegistry();
    return registry.find(vectorWidth) != registry.end();
}

// Note: NodeValueBufferFactory::create() is NOT implemented here because
// loadable backends only need to register their buffer creator. The create()
// function is called from the main forge library which has the full implementation.

} // namespace forge

#endif // FORGE_LOADABLE_BACKEND
