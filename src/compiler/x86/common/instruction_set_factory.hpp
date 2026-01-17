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
#include "../double/scalar/sse2_scalar_instruction_set.hpp"
#include "../double/avx2/avx2_instruction_set.hpp"
#include "compiler_config.hpp"
#include <memory>
#include <unordered_map>
#include <functional>
#include <stdexcept>

namespace forge {

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
                return std::make_unique<AVX2InstructionSet>(config);
            
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
    
private:
    // Registry for dynamically registered instruction sets
    static std::unordered_map<std::string, CreateFunc>& getRegistry() {
        static std::unordered_map<std::string, CreateFunc> registry;
        return registry;
    }
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