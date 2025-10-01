#pragma once

#include "forge/core/computation_graph.h"
#include "forge/x86/register_allocator.h"
#include "forge/x86/instruction_set.h"
#include "forge/x86/compiler_config.h"
#include "generators/constant_pool_manager.h"
#include <asmjit/x86.h>
#include <unordered_map>
#include <unordered_set>

namespace forge::compiler {

/**
 * Handles forward pass code generation for mathematical operations.
 * Extracted from AsmStitcher::generateOperation for better organization.
 */
class ForwardCompiler {
public:
    ForwardCompiler(const forge::x86::CompilerConfig& config, forge::x86::IInstructionSet* instructionSet);
    
    /**
     * Generate assembly code for a single operation node
     * This is the main entry point for forward pass code generation
     */
    void generateOperation(asmjit::x86::Assembler& a, 
                          const forge::core::Node& node,
                          forge::core::NodeId nodeId,
                          const forge::core::ComputationGraph& graph,
                          const std::unordered_map<forge::core::NodeId, generators::ConstantInfo>& constantMap,
                          const asmjit::Label& constPoolLabel,
                          forge::x86::IRegisterAllocator& regState);

private:
    const forge::x86::CompilerConfig& config_;
    forge::x86::IInstructionSet* instructionSet_;
};

} // namespace forge::compiler