#pragma once

#include "forge/core/computation_graph.h"
#include "forge/x86/register_allocator.h"  // Use IRegisterAllocator interface
#include "forge/x86/compiler_config.h"
#include "generators/constant_pool_manager.h"  // For ConstantInfo
#include "forge/x86/instruction_set.h"
#include <asmjit/x86.h>
#include <unordered_map>

namespace forge::compiler {

class ReverseGradientCompiler {
public:
    // Generate backward pass code for a single operation
    static void generateGradientOperation(
        asmjit::x86::Assembler& a,
        const forge::core::Node& node,
        forge::core::NodeId nodeId,
        forge::x86::IRegisterAllocator& regState,  // Changed to use interface
        const forge::core::ComputationGraph& graph,
        const std::unordered_map<forge::core::NodeId, generators::ConstantInfo>& constantMap,
        const asmjit::Label& constPoolLabel,
        forge::x86::IInstructionSet* instructionSet,
        const forge::x86::CompilerConfig* config = nullptr
    );
    
    // Generate complete gradient computation
    static void stitchGradientPass(
        asmjit::x86::Assembler& a,
        const forge::core::ComputationGraph& graph,
        const std::unordered_map<forge::core::NodeId, generators::ConstantInfo>& constantMap,
        const asmjit::Label& constPoolLabel,
        forge::x86::IRegisterAllocator& regState,  // Changed to use interface
        forge::x86::IInstructionSet* instructionSet,
        const forge::x86::CompilerConfig* config = nullptr
    );
    
    // No private helper methods - all operations go through instruction set abstraction
};

} // namespace forge::compiler