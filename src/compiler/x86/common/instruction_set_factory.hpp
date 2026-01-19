// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file instruction_set_factory.hpp
 * @brief Factory for creating instruction set implementations
 *
 * Provides a factory pattern for instantiating SIMD instruction set backends.
 * Supports both static selection (via enum) and dynamic registration by name.
 *
 * Thread Safety: Static methods are thread-safe for reading. Registration
 * methods should only be called during startup (not thread-safe).
 */

#pragma once

#include "../../interfaces/instruction_set.hpp"
#include "../../interfaces/node_value_buffer.hpp"
#include "../../../graph/graph.hpp"
#include "../double/scalar/sse2_scalar_instruction_set.hpp"
#ifdef FORGE_BUNDLE_AVX2
#include "../../../../backends/double/avx2/avx2_instruction_set.hpp"
#endif
#include "compiler_config.hpp"
#include <memory>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <vector>

// Platform-specific dynamic loading
#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace forge {

/**
 * @brief API struct passed to dynamically loaded backends
 *
 * This struct contains function pointers that backends use to register
 * themselves with the main program. This solves the Windows DLL issue
 * where static variables are duplicated between the main exe and DLL.
 */
struct ForgeBackendAPI {
    void (*registerInstructionSet)(const char* name, std::unique_ptr<IInstructionSet>(*)());
    void (*registerBufferCreator)(int vectorWidth, std::unique_ptr<INodeValueBuffer>(*)(
        const Graph&, const std::vector<NodeId>&, size_t));
};

/**
 * @brief Factory for creating instruction set implementations
 *
 * Provides two ways to create instruction set backends:
 * 1. Static creation via CompilerConfig::InstructionSet enum
 * 2. Dynamic registration via name strings
 *
 * The registration system allows adding new instruction sets without
 * modifying existing code.
 *
 * API Stability: Stable
 *
 * Example (static):
 * @code
 * auto sse2 = InstructionSetFactory::create(
 *     CompilerConfig::InstructionSet::SSE2_SCALAR);
 * @endcode
 *
 * Example (dynamic):
 * @code
 * InstructionSetFactory::registerInstructionSet("MyISA",
 *     []() { return std::make_unique<MyInstructionSet>(); });
 * auto custom = InstructionSetFactory::createByName("MyISA");
 * @endcode
 */
class InstructionSetFactory {
public:
    /** @brief Function type for creating instruction set instances */
    using CreateFunc = std::function<std::unique_ptr<IInstructionSet>()>;

    /**
     * @brief Create instruction set from enum type
     *
     * @param type Instruction set type from CompilerConfig
     * @param config Compiler configuration (passed to instruction set)
     * @return New instruction set instance
     */
    static std::unique_ptr<IInstructionSet> create(CompilerConfig::InstructionSet type, const CompilerConfig& config = CompilerConfig::Default()) {
        switch(type) {
            case CompilerConfig::InstructionSet::SSE2_SCALAR:
                return std::make_unique<SSE2ScalarInstructionSet>(config);

            case CompilerConfig::InstructionSet::AVX2_PACKED:
#ifdef FORGE_BUNDLE_AVX2
                return std::make_unique<AVX2InstructionSet>(config);
#else
                // AVX2 not bundled - try to get from registry (loaded at runtime)
                if (hasInstructionSet("AVX2-Packed")) {
                    return createByName("AVX2-Packed", config, false);
                }
                // Fall back to SSE2-Scalar if AVX2 not available
                return std::make_unique<SSE2ScalarInstructionSet>(config);
#endif

            // Future instruction sets will be added here by contributors
            // No modification to existing cases needed

            default:
                // Fallback to SSE2-Scalar if unknown
                return std::make_unique<SSE2ScalarInstructionSet>(config);
        }
    }

    /**
     * @brief Register a custom instruction set
     *
     * Allows dynamic registration of new instruction set implementations.
     * Call this during static initialization.
     *
     * @param name Unique name for the instruction set
     * @param factory Function that creates instruction set instances
     *
     * Thread Safety: Not thread-safe - call during startup only
     */
    static void registerInstructionSet(const std::string& name, CreateFunc factory) {
        getRegistry()[name] = factory;
    }

    /**
     * @brief Create instruction set by name
     *
     * @param name Name of registered instruction set
     * @param config Compiler configuration
     * @param checkVersion If true, throws if API version doesn't match
     * @return New instruction set instance, or SSE2-Scalar if name not found
     * @throws std::runtime_error If checkVersion is true and versions don't match
     */
    static std::unique_ptr<IInstructionSet> createByName(const std::string& name, const CompilerConfig& config = CompilerConfig::Default(), bool checkVersion = true) {
        auto& registry = getRegistry();
        auto it = registry.find(name);

        if (it != registry.end()) {
            auto instance = it->second();

            if (checkVersion && instance->apiVersion() != INSTRUCTION_SET_API_VERSION) {
                throw std::runtime_error(
                    "Instruction set '" + name + "' was built against API version " +
                    std::to_string(instance->apiVersion()) + ", but core expects version " +
                    std::to_string(INSTRUCTION_SET_API_VERSION));
            }

            return instance;
        }

        // Default to SSE2-Scalar if not found, but pass the config!
        return std::make_unique<SSE2ScalarInstructionSet>(config);
    }

    /**
     * @brief Check if an instruction set is registered
     * @param name Name to check
     * @return true if registered
     */
    static bool hasInstructionSet(const std::string& name) {
        return getRegistry().find(name) != getRegistry().end();
    }

    /**
     * @brief Get list of all available instruction sets
     * @return Vector of instruction set names (includes built-in and registered)
     */
    static std::vector<std::string> getAvailableInstructionSets() {
        std::vector<std::string> names;
        names.push_back("SSE2-Scalar"); // Always available

        for (const auto& [name, _] : getRegistry()) {
            names.push_back(name);
        }

        return names;
    }

    /**
     * @brief Load a backend from a shared library at runtime
     *
     * Loads the specified shared library and calls its registration function.
     * The library should export a C function with signature:
     *     extern "C" void forge_register_backend();
     *
     * This function should call InstructionSetFactory::registerInstructionSet()
     * to register the custom instruction set(s) provided by the library.
     *
     * @param libraryPath Path to the shared library (.so on Linux, .dll on Windows)
     * @return true if library was loaded and registration function called successfully
     * @throws std::runtime_error If library cannot be loaded or has no registration function
     *
     * Example:
     * @code
     * // Load a custom AVX-512 backend
     * InstructionSetFactory::loadBackend("./libforge_avx512.so");
     *
     * // Now the instruction set is available
     * auto avx512 = InstructionSetFactory::createByName("AVX512-Packed");
     * @endcode
     *
     * Writing a backend library:
     * @code
     * // In your backend library (e.g., avx512_backend.cpp):
     * #include "instruction_set_factory.hpp"
     *
     * class AVX512InstructionSet : public forge::IInstructionSet {
     *     // ... implementation ...
     * };
     *
     * extern "C" void forge_register_backend_v2(ForgeBackendAPI* api) {
     *     api->registerInstructionSet("AVX512-Packed",
     *         []() { return std::make_unique<AVX512InstructionSet>(); });
     * }
     * @endcode
     */
    static bool loadBackend(const std::string& libraryPath) {
        // V2 API uses callbacks to avoid Windows DLL static variable issues
        using RegisterFuncV2 = void(*)(ForgeBackendAPI*);

#ifdef _WIN32
        HMODULE handle = LoadLibraryA(libraryPath.c_str());
        if (!handle) {
            DWORD error = GetLastError();
            throw std::runtime_error(
                "Failed to load library '" + libraryPath + "': error code " + std::to_string(error));
        }

        // Try V2 API first (required for Windows)
        RegisterFuncV2 registerFuncV2 = reinterpret_cast<RegisterFuncV2>(
            GetProcAddress(handle, "forge_register_backend_v2"));

        if (!registerFuncV2) {
            FreeLibrary(handle);
            throw std::runtime_error(
                "Library '" + libraryPath + "' does not export 'forge_register_backend_v2'");
        }

        getLibraryHandles().push_back(handle);
#else
        void* handle = dlopen(libraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            throw std::runtime_error(
                "Failed to load library '" + libraryPath + "': " + dlerror());
        }

        dlerror(); // Clear any existing error

        // Try V2 API first (required for Windows, also works on Linux)
        RegisterFuncV2 registerFuncV2 = reinterpret_cast<RegisterFuncV2>(
            dlsym(handle, "forge_register_backend_v2"));

        if (!registerFuncV2) {
            dlclose(handle);
            throw std::runtime_error(
                "Library '" + libraryPath + "' does not export 'forge_register_backend_v2'");
        }

        getLibraryHandles().push_back(handle);
#endif

        // Create API callbacks that register into this process's registry
        ForgeBackendAPI api;
        api.registerInstructionSet = &registerInstructionSetCallback;
        api.registerBufferCreator = &registerBufferCreatorCallback;

        registerFuncV2(&api);
        return true;
    }

private:
    // Callback wrappers for the backend API
    static void registerInstructionSetCallback(const char* name, std::unique_ptr<IInstructionSet>(*factory)()) {
        getRegistry()[name] = factory;
    }

    static void registerBufferCreatorCallback(int vectorWidth,
        std::unique_ptr<INodeValueBuffer>(*creator)(const Graph&, const std::vector<NodeId>&, size_t)) {
        NodeValueBufferFactory::registerBufferCreator(vectorWidth, creator);
    }

public:

    /**
     * @brief Unload all dynamically loaded backend libraries
     *
     * Clears the registry of dynamically loaded backends and unloads
     * their shared libraries. Built-in instruction sets remain available.
     */
    static void unloadAllBackends() {
#ifdef _WIN32
        for (HMODULE handle : getLibraryHandles()) {
            FreeLibrary(handle);
        }
#else
        for (void* handle : getLibraryHandles()) {
            dlclose(handle);
        }
#endif
        getLibraryHandles().clear();
        getRegistry().clear();
    }

private:
    // Registry for dynamically registered instruction sets
    static std::unordered_map<std::string, CreateFunc>& getRegistry() {
        static std::unordered_map<std::string, CreateFunc> registry;
        return registry;
    }

    // Storage for loaded library handles (prevents unloading while in use)
#ifdef _WIN32
    static std::vector<HMODULE>& getLibraryHandles() {
        static std::vector<HMODULE> handles;
        return handles;
    }
#else
    static std::vector<void*>& getLibraryHandles() {
        static std::vector<void*> handles;
        return handles;
    }
#endif
};

/**
 * @brief Helper class for automatic instruction set registration
 *
 * Use this in your implementation file to automatically register a custom
 * instruction set during static initialization.
 *
 * Example:
 * @code
 * // In my_instruction_set.cpp:
 * static InstructionSetRegistrar<MyInstructionSet> registrar("MyISA");
 * @endcode
 *
 * @tparam T Instruction set implementation class
 */
template<typename T>
class InstructionSetRegistrar {
public:
    /**
     * @brief Register instruction set during static initialization
     * @param name Unique name for the instruction set
     */
    InstructionSetRegistrar(const std::string& name) {
        InstructionSetFactory::registerInstructionSet(name, []() {
            return std::make_unique<T>();
        });
    }
};

/**
 * @brief Macro for easy instruction set registration
 *
 * Declares a static registrar instance that automatically registers
 * the instruction set during program startup.
 *
 * Usage:
 * @code
 * // In my_instruction_set.cpp:
 * REGISTER_INSTRUCTION_SET(MyInstructionSet, "MyISA")
 * @endcode
 */
#define REGISTER_INSTRUCTION_SET(Class, Name) \
    static forge::InstructionSetRegistrar<Class> _##Class##_registrar(Name);

} // namespace forge
