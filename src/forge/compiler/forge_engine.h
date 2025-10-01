#pragma once

#include "forge/core/computation_graph.h"
#include "forge/runtime/compiled_kernel.h"
#include "forge/x86/register_allocator.h"
#include "forge/x86/xmm_register_allocator.h"
#include "forge/x86/ymm_register_allocator.h"
#include "forge/x86/compiler_config.h"
#include "forge/x86/instruction_set.h"
#include "generators/instruction_set_factory.h"
#include "forge/x86/runtime_trace.h"
#include "generators/constant_pool_manager.h"
#include "generators/register_utils.h"
#include "forward_compiler.h"
#include "analysis/stability_cleaner.h"
#include <asmjit/x86.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace forge {
namespace compiler {

// Main JIT compilation engine
// Orchestrates the entire compilation process from ComputationGraph to executable kernel
class ForgeEngine {
public:
    ForgeEngine();
    explicit ForgeEngine(const forge::x86::CompilerConfig& config);
    ~ForgeEngine();
    
    // Main compilation entry point
    // Takes a Graph and produces executable kernel
    std::unique_ptr<forge::runtime::CompiledKernel> compile(const forge::core::ComputationGraph& graph);
    
    // Get/Set configuration
    const forge::x86::CompilerConfig& getConfig() const { return config_; }
    void setConfig(const forge::x86::CompilerConfig& config) { config_ = config; }
    
    // Get the shared JitRuntime (for testing/debugging)
    static asmjit::JitRuntime& getRuntime();
    
private:
    // Compiler configuration
    forge::x86::CompilerConfig config_;
    
    // Instruction set implementation (based on config)
    std::unique_ptr<forge::x86::IInstructionSet> instructionSet_;
    
    // Shared JitRuntime for all compilers - long-lived per Design v3
    // This ensures executable memory remains valid after compiler destruction
    static asmjit::JitRuntime s_runtime;
    
    // Register tracking for optimization
    // Creates appropriate allocator based on instruction set
    std::unique_ptr<forge::x86::IRegisterAllocator> createRegisterAllocator() const;
    
    // ============================================================================
    // COMPILATION PHASE METHODS
    // ============================================================================
    
    // Phase result structures for clean data flow
    struct StabilityResult {
        forge::core::ComputationGraph cleanedGraph;
        analysis::StabilityCleaner::CleaningResult stabilityData;
    };
    
    struct ConstantPoolData {
        generators::ConstantPoolManager::ConstantPoolResult poolResult;
        std::unique_ptr<generators::ConstantPoolManager> poolManager;
    };
    
    struct AssemblyContext {
        asmjit::CodeHolder code;
        asmjit::x86::Assembler assembler;
        std::unique_ptr<forge::x86::IRegisterAllocator> regState;
        forge::core::NodeId maxNodeIdAccessed;
        
        AssemblyContext(asmjit::JitRuntime& runtime) : assembler(&code), maxNodeIdAccessed(0) {
            code.init(runtime.environment(), runtime.cpuFeatures());
            assembler.addDiagnosticOptions(asmjit::DiagnosticOptions::kValidateAssembler);
        }
    };
    
    // Phase extraction methods
    StabilityResult performStabilityCleaning(const forge::core::ComputationGraph& graph);
    ConstantPoolData createConstantPool(const forge::core::ComputationGraph& graph, AssemblyContext& ctx);
    void generateForwardPass(AssemblyContext& ctx, const forge::core::ComputationGraph& graph, 
                           const ConstantPoolData& constants);
    void generateReversePass(AssemblyContext& ctx, const forge::core::ComputationGraph& graph,
                           const ConstantPoolData& constants);
    std::unique_ptr<forge::runtime::CompiledKernel> finalizeKernel(AssemblyContext& ctx, 
                                                                  const StabilityResult& stability);
    
    // Code generation phases
    void generatePrologue(asmjit::x86::Assembler& a);
    void generateEpilogue(asmjit::x86::Assembler& a);
    
    // Register allocation plan (fixed, no dynamic allocation)
    // RDI = values pointer (base for all memory access)
    // XMM0-XMM3 = working registers for operations
    static constexpr int kNumWorkingRegs = 4;
};

} // namespace compiler
} // namespace forge