#pragma once

#include "forge/x86/instruction_set.h"
#include "forge/x86/sse2_scalar_instruction_set.h"
#include "forge/x86/avx2_instruction_set.h"
#include "forge/x86/compiler_config.h"
#include <memory>
#include <unordered_map>
#include <functional>
#include <stdexcept>

namespace forge::compiler::generators {

// Factory class for creating instruction set implementations
// This design allows contributors to register new instruction sets at runtime
// without modifying any existing code
class InstructionSetFactory {
public:
    using CreateFunc = std::function<std::unique_ptr<forge::x86::IInstructionSet>()>;
    
    // Create an instruction set based on the configuration
    static std::unique_ptr<forge::x86::IInstructionSet> create(forge::x86::CompilerConfig::InstructionSet type, const forge::x86::CompilerConfig& config = forge::x86::CompilerConfig::Default()) {
        switch(type) {
            case forge::x86::CompilerConfig::InstructionSet::SSE2_SCALAR:
                return std::make_unique<forge::x86::SSE2ScalarInstructionSet>(config);
            
            case forge::x86::CompilerConfig::InstructionSet::AVX2_PACKED:
                return std::make_unique<forge::x86::AVX2InstructionSet>(config);
            
            // Future instruction sets will be added here by contributors
            // No modification to existing cases needed
            
            default:
                // Fallback to SSE2-Scalar if unknown
                return std::make_unique<forge::x86::SSE2ScalarInstructionSet>(config);
        }
    }
    
    // Plugin registration system for dynamic instruction set loading
    // Contributors can register custom instruction sets without modifying this file
    static void registerInstructionSet(const std::string& name, CreateFunc factory) {
        getRegistry()[name] = factory;
    }
    
    // Create instruction set by name (for plugin system)
    static std::unique_ptr<forge::x86::IInstructionSet> createByName(const std::string& name, const forge::x86::CompilerConfig& config = forge::x86::CompilerConfig::Default()) {
        auto& registry = getRegistry();
        auto it = registry.find(name);
        
        if (it != registry.end()) {
            return it->second();
        }
        
        // Default to SSE2-Scalar if not found, but pass the config!
        return std::make_unique<forge::x86::SSE2ScalarInstructionSet>(config);
    }
    
    // Check if an instruction set is registered
    static bool hasInstructionSet(const std::string& name) {
        return getRegistry().find(name) != getRegistry().end();
    }
    
    // Get list of all registered instruction sets
    static std::vector<std::string> getAvailableInstructionSets() {
        std::vector<std::string> names;
        names.push_back("SSE2-Scalar"); // Always available
        
        for (const auto& [name, _] : getRegistry()) {
            names.push_back(name);
        }
        
        return names;
    }
    
private:
    // Registry for dynamically loaded instruction sets
    static std::unordered_map<std::string, CreateFunc>& getRegistry() {
        static std::unordered_map<std::string, CreateFunc> registry;
        return registry;
    }
};

// Helper class for automatic registration of instruction sets
// Contributors can use this in their implementation files
template<typename T>
class InstructionSetRegistrar {
public:
    InstructionSetRegistrar(const std::string& name) {
        InstructionSetFactory::registerInstructionSet(name, []() {
            return std::make_unique<T>();
        });
    }
};

// Macro for easy registration (optional)
// Usage: REGISTER_INSTRUCTION_SET(MyInstructionSet, "MyISA")
#define REGISTER_INSTRUCTION_SET(Class, Name) \
    static tapepresso::InstructionSetRegistrar<Class> _##Class##_registrar(Name);

} // namespace forge::compiler::generators