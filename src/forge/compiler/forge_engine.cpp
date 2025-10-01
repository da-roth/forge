#include "forge_engine.h"
#include "generators/operation_utils.h"
#include "generators/constant_pool_manager.h"
#include "generators/register_utils.h"
#include "operations/arithmetic_operations.h"
#include "operations/math_functions.h"
#include "operations/comparison_control.h"
#include "operations/boolean_operations.h"
#include "operations/integer_operations.h"
#include "analysis/stability_cleaner.h"
#include "reverse_gradient_compiler.h"
#include "forge/x86/sse2_scalar_instruction_set.h"
#include "utils/compilation_timer.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>  // For std::swap
#include <set>        // For std::set
#include <unordered_set>  // For std::unordered_set
#include <cmath>      // For std::exp, std::log
#include <chrono>     // For timing
#include <asmjit/core.h>

namespace forge {
namespace compiler {

using namespace asmjit;
using namespace forge::core;

// ========================================================================
// JIT Compilation Configuration Options
// ========================================================================


// Define the static JitRuntime (shared across all compilers)
asmjit::JitRuntime ForgeEngine::s_runtime;

ForgeEngine::ForgeEngine() : config_(forge::x86::CompilerConfig::Default()) {
    // Create instruction set based on config - MUST pass the full config!
    instructionSet_ = generators::InstructionSetFactory::create(config_.instructionSet, config_);
}

ForgeEngine::ForgeEngine(const forge::x86::CompilerConfig& config) : config_(config) {
    // Create instruction set based on config - MUST pass the full config!
    instructionSet_ = generators::InstructionSetFactory::create(config_.instructionSet, config_);
}

ForgeEngine::~ForgeEngine() = default;

asmjit::JitRuntime& ForgeEngine::getRuntime() {
    return s_runtime;
}

std::unique_ptr<forge::x86::IRegisterAllocator> ForgeEngine::createRegisterAllocator() const {
    // Create appropriate allocator based on instruction set
    switch (config_.instructionSet) {
        case forge::x86::CompilerConfig::InstructionSet::AVX2_PACKED:
            return std::make_unique<forge::x86::YmmRegisterAllocator>();
        case forge::x86::CompilerConfig::InstructionSet::SSE2_SCALAR:
        default:
            return std::make_unique<forge::x86::XmmRegisterAllocator>();
    }
}

std::unique_ptr<forge::runtime::CompiledKernel> ForgeEngine::compile(const forge::core::ComputationGraph& graph) {
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    
    auto totalStart = Clock::now();
    
    // Validate outputs exist
    if (graph.outputs.empty()) {
        throw std::runtime_error("No outputs were marked on the graph. Ensure markOutput() is called.");
    }

    // ============================================================================
    // FORGE JIT COMPILATION PIPELINE
    // ============================================================================
    // 1. STABILITY CLEANING    - Numerical safety transformations
    // 2. CONSTANT POOL         - Deduplicate constants and memory layout  
    // 3. FORWARD PASS          - Generate computation assembly
    // 4. REVERSE PASS (AAD)    - Generate gradient assembly (if needed)
    // 5. FINALIZATION          - Finalize kernel and create executable
    // ============================================================================

    // Phase 1: Stability cleaning for numerical safety
    // Input: graph (original computation graph), config_ (stability settings)
    // Output: stabilityResult (numerically safe graph + mapping + timing)
    auto stabilityResult = performStabilityCleaning(graph);
    
    forge::core::ComputationGraph stabilizedGraph = stabilityResult.cleanedGraph;
    
    // Print stability cleaning statistics
    if (config_.printAssembly || config_.printOriginalGraph || config_.printStabilizedGraph) {
        std::cout << "\n=== Stability Cleaning Info ===" << std::endl;
        std::cout << "  Stability fixes applied: " << stabilityResult.stabilityData.stabilityFixesApplied << std::endl;
        std::cout << "  Stability cleaning time: " << std::fixed << std::setprecision(2) 
                  << stabilityResult.stabilityData.cleaningTimeMs << " ms" << std::endl;
        
        // Helper lambda to print node details
        auto printNode = [](const forge::core::Node& node, size_t idx) {
            std::cout << "    Node " << idx << ": ";
            switch(node.op) {
                case forge::core::OpCode::Input: std::cout << "Input"; break;
                case forge::core::OpCode::Constant: std::cout << "Constant"; break;
                case forge::core::OpCode::Add: std::cout << "Add"; break;
                case forge::core::OpCode::Sub: std::cout << "Sub"; break;
                case forge::core::OpCode::Mul: std::cout << "Mul"; break;
                case forge::core::OpCode::Div: std::cout << "Div"; break;
                case forge::core::OpCode::Neg: std::cout << "Neg"; break;
                case forge::core::OpCode::Abs: std::cout << "Abs"; break;
                case forge::core::OpCode::Square: std::cout << "Square"; break;
                case forge::core::OpCode::Recip: std::cout << "Recip"; break;
                case forge::core::OpCode::Mod: std::cout << "Mod"; break;
                case forge::core::OpCode::Exp: std::cout << "Exp"; break;
                case forge::core::OpCode::Log: std::cout << "Log"; break;
                case forge::core::OpCode::Pow: std::cout << "Pow"; break;
                case forge::core::OpCode::Sqrt: std::cout << "Sqrt"; break;
                case forge::core::OpCode::Sin: std::cout << "Sin"; break;
                case forge::core::OpCode::Cos: std::cout << "Cos"; break;
                case forge::core::OpCode::Tan: std::cout << "Tan"; break;
                case forge::core::OpCode::Min: std::cout << "Min"; break;
                case forge::core::OpCode::Max: std::cout << "Max"; break;
                case forge::core::OpCode::If: std::cout << "If"; break;
                case forge::core::OpCode::CmpLT: std::cout << "CmpLT"; break;
                case forge::core::OpCode::CmpLE: std::cout << "CmpLE"; break;
                case forge::core::OpCode::CmpGT: std::cout << "CmpGT"; break;
                case forge::core::OpCode::CmpGE: std::cout << "CmpGE"; break;
                case forge::core::OpCode::CmpEQ: std::cout << "CmpEQ"; break;
                case forge::core::OpCode::CmpNE: std::cout << "CmpNE"; break;
                // Boolean operations
                case forge::core::OpCode::BoolConstant: std::cout << "BoolConstant"; break;
                case forge::core::OpCode::BoolAnd: std::cout << "BoolAnd"; break;
                case forge::core::OpCode::BoolOr: std::cout << "BoolOr"; break;
                case forge::core::OpCode::BoolNot: std::cout << "BoolNot"; break;
                case forge::core::OpCode::BoolEq: std::cout << "BoolEq"; break;
                case forge::core::OpCode::BoolNe: std::cout << "BoolNe"; break;
                // Integer operations
                case forge::core::OpCode::IntConstant: std::cout << "IntConstant"; break;
                case forge::core::OpCode::IntAdd: std::cout << "IntAdd"; break;
                case forge::core::OpCode::IntSub: std::cout << "IntSub"; break;
                case forge::core::OpCode::IntMul: std::cout << "IntMul"; break;
                case forge::core::OpCode::IntDiv: std::cout << "IntDiv"; break;
                case forge::core::OpCode::IntMod: std::cout << "IntMod"; break;
                case forge::core::OpCode::IntNeg: std::cout << "IntNeg"; break;
                case forge::core::OpCode::IntCmpLT: std::cout << "IntCmpLT"; break;
                case forge::core::OpCode::IntCmpLE: std::cout << "IntCmpLE"; break;
                case forge::core::OpCode::IntCmpGT: std::cout << "IntCmpGT"; break;
                case forge::core::OpCode::IntCmpGE: std::cout << "IntCmpGE"; break;
                case forge::core::OpCode::IntCmpEQ: std::cout << "IntCmpEQ"; break;
                case forge::core::OpCode::IntCmpNE: std::cout << "IntCmpNE"; break;
                case forge::core::OpCode::IntIf: std::cout << "IntIf"; break;
                case forge::core::OpCode::ArrayIndex: std::cout << "ArrayIndex"; break;
                default: std::cout << "Op" << static_cast<int>(node.op); break;
            }
            std::cout << "(";
            if (node.a != UINT32_MAX) std::cout << node.a;
            if (node.b != UINT32_MAX) std::cout << "," << node.b;
            if (node.c != UINT32_MAX) std::cout << "," << node.c;
            std::cout << ")";
            if (node.op == forge::core::OpCode::Constant) {
                std::cout << " imm=" << node.imm;
            }
            std::cout << " [active=" << node.isActive << ", dead=" << node.isDead << "]";
            std::cout << std::endl;
        };
        
        // Print original graph if requested
        if (config_.printOriginalGraph) {
            std::cout << "\n  Original Graph:" << std::endl;
            for (size_t i = 0; i < graph.nodes.size(); ++i) {
                printNode(graph.nodes[i], i);
            }
        }
        
        // Print stabilized graph if requested
        if (config_.printStabilizedGraph) {
            std::cout << "\n  Stability-Cleaned Graph:" << std::endl;
            for (size_t i = 0; i < stabilizedGraph.nodes.size(); ++i) {
                printNode(stabilizedGraph.nodes[i], i);
            }
        }
        
        if (config_.printNodeFlags) {
            std::cout << "\n  Node flags in stability-cleaned graph:" << std::endl;
            for (size_t i = 0; i < stabilizedGraph.nodes.size(); ++i) {
                std::cout << "    Node " << i << ": needsGradient=" << stabilizedGraph.nodes[i].needsGradient 
                          << " isActive=" << stabilizedGraph.nodes[i].isActive << std::endl;
            }
        }
    }
    
    // Use the stability-cleaned graph for compilation
    const ComputationGraph& workingGraph = stabilizedGraph;
    
    // AAD: Check if any node needs gradients and validate flags
    bool needsGradient = false;
    for (const auto& node : workingGraph.nodes) {
        if (node.needsGradient) {
            needsGradient = true;
            // Validation: needsGradient implies isActive
            if (!node.isActive) {
                throw std::runtime_error("Gradient validation failed: node with needsGradient=true must have isActive=true");
            }
        }
    }
    
    if (needsGradient) {
        if (config_.printGradientDebug) {
            std::cout << "  AAD: Gradient computation enabled (" << workingGraph.diff_inputs.size() << " differentiated inputs)" << std::endl;
            // Count operations that will generate gradient code
            int gradientOpsCount = 0;
            for (const auto& node : workingGraph.nodes) {
                if (node.needsGradient && !node.isDead) {
                    gradientOpsCount++;
                }
            }
            std::cout << "  Gradient operations to generate: " << gradientOpsCount << std::endl;
        }
    }
    
    // Start timing kernel stitching phase
    auto stitchingStart = Clock::now();
    
    // Detailed timing for stitching phases
    double constantPoolTime = 0.0;
    double codeGenerationTime = 0.0;
    double assemblyFinalizationTime = 0.0;
    std::unordered_map<std::string, double> opTypeTime;
    std::unordered_map<std::string, int> opTypeCounts;
    
    // Create code holder and assembler
    auto codeInitStart = Clock::now();
    CodeHolder code;
    // CRITICAL: Initialize with both environment AND CPU features for AVX2 support
    code.init(s_runtime.environment(), s_runtime.cpuFeatures());
    
    // Use x86::Assembler directly - NO Compiler abstraction!
    asmjit::x86::Assembler a(&code);
    
    // Enable validation to catch assembly errors (as suggested by specialist)
    a.addDiagnosticOptions(asmjit::DiagnosticOptions::kValidateAssembler);
    Duration codeInitTime = Clock::now() - codeInitStart;
    
    // Phase 2: Constant pool management
    // Input: workingGraph (stabilized graph), config_ (instruction set, optimization flags)
    // Output: constPoolResult (deduplicated constants + memory layout + mappings + timing)
    auto constantPoolStart = Clock::now();

    generators::ConstantPoolManager constPoolManager(config_);
    auto constPoolResult = constPoolManager.createConstantPool(workingGraph, a);
    
    // Extract results for compatibility with existing code
    auto& constPoolLabel = constPoolResult.constPoolLabel;
    auto& constantMap = constPoolResult.constantMap;
    
    constantPoolTime = constPoolResult.creationTimeMs;
    
    // Generate function prologue
    auto prologueStart = Clock::now();
    generatePrologue(a);
    Duration prologueTime = Clock::now() - prologueStart;
    
    // Phase 2.3: Initialize register tracking state
    // Create appropriate allocator based on instruction set
    auto regStatePtr = createRegisterAllocator();
    forge::x86::IRegisterAllocator& regState = *regStatePtr;
    
    // Track maximum node ID accessed for proper buffer allocation
    NodeId maxNodeIdAccessed = 0;
    
    // Use ConstantPoolManager to preload hot constants (simplified for now)
    constPoolManager.preloadHotConstants(a, constPoolResult, instructionSet_.get());
    
    // Phase 3: Forward pass code generation
    // Input: workingGraph (stabilized graph), constantMap, constPoolLabel, config_ (instruction set, registers)
    // Behavior: Generates x86 assembly for forward computation (same code regardless of gradient needs)
    // Output: Assembly code with forward computation logic embedded in assembler
    ForwardCompiler forwardGen(config_, instructionSet_.get());
    
    auto codeGenStart = Clock::now();
    int nodesProcessed = 0;
    
    // Process nodes one by one
    for (NodeId nodeId = 0; nodeId < workingGraph.nodes.size(); ++nodeId) {
        const Node& node = workingGraph.nodes[nodeId];
        if (node.isDead) continue;  // Skip dead nodes from optimization
        
        // Track operation type timing only if profiling is enabled
        // Use printAssembly as the profiling enable flag for compilation stats
        bool enableProfiling = config_.printAssembly || config_.printOriginalGraph || config_.printStabilizedGraph;
        utils::OperationTimer timer(generators::getOpName(node.op), opTypeTime, opTypeCounts, enableProfiling);
        
        forwardGen.generateOperation(a, node, nodeId, workingGraph, constantMap, constPoolLabel, regState);
        
        // Track maximum node ID
        maxNodeIdAccessed = std::max(maxNodeIdAccessed, nodeId);
        nodesProcessed++;
    }
    
    // Generate function epilogue
    codeGenerationTime = Duration(Clock::now() - codeGenStart).count();
    
    // Phase 4: Optional reverse gradient computation (AAD)
    // Input: assembler (with forward code), workingGraph, constantMap, constPoolLabel, config_ (gradient flags)
    // Behavior: Reads forward results from memory and generates gradient accumulation code
    // Output: Assembly extended with backward pass that computes gradients via chain rule
    if (needsGradient) {
        // Check if gradients pointer is not null at runtime
        // Note: After prologue, RSI contains the gradients pointer (moved from RDX)
        Label skipGradient = a.newLabel();
        a.test(asmjit::x86::rsi, asmjit::x86::rsi);  // RSI = gradients pointer (already set in prologue)
        a.jz(skipGradient);  // Jump if gradients == nullptr
        
        // Generate gradient code (RSI already points to gradients)
        ReverseGradientCompiler::stitchGradientPass(a, workingGraph, constantMap, constPoolLabel, regState, instructionSet_.get(), &config_);
        
        a.bind(skipGradient);
    }
    
    // Generate function epilogue
    auto epilogueStart = Clock::now();
    generateEpilogue(a);
    Duration epilogueTime = Clock::now() - epilogueStart;
    
    // Phase 5: Assembly finalization and JIT compilation
    // Input: assembler (complete code), constPoolResult, config_ (runtime settings), stabilityResult.mapping
    // Behavior: Embeds constants, links assembly, creates executable kernel function
    // Output: CompiledKernel (executable function pointer + metadata + node mappings)
    double embedTime = constPoolManager.embedConstantPool(a, constPoolResult);
    
    auto finalizeStart = Clock::now();
    forge::runtime::CompiledKernel::KernelFunc func = nullptr;
    Error err = s_runtime.add(&func, &code);
    if (err) {
        // Get detailed error information
        const char* errMsg = DebugUtils::errorAsString(err);
        std::string errorStr = "Failed to compile kernel: ";
        errorStr += errMsg ? errMsg : "Unknown error";
        throw std::runtime_error(errorStr);
    }
    assemblyFinalizationTime = Duration(Clock::now() - finalizeStart).count();
    
    auto stitchingEnd = Clock::now();
    Duration stitchingTime = stitchingEnd - stitchingStart;
    
    auto totalEnd = Clock::now();
    Duration totalTime = totalEnd - totalStart;
    
    if (maxNodeIdAccessed >= workingGraph.nodes.size()) {
        std::cout << "[WARNING] Buffer overflow risk! Node " << maxNodeIdAccessed 
                  << " exceeds tape size!" << std::endl;
    }
    
    // Print timing information if assembly output is requested
    if (config_.printAssembly || config_.printOriginalGraph || config_.printStabilizedGraph) {
        std::cout << "\n=== JIT Compilation Timing ===" << std::endl;
        std::cout << "  Maximum node ID accessed: " << maxNodeIdAccessed 
                  << " (tape size: " << workingGraph.nodes.size() << ")" << std::endl;
        if (maxNodeIdAccessed >= workingGraph.nodes.size()) {
            std::cout << "  WARNING: Kernel accesses node " << maxNodeIdAccessed 
                      << " but tape only has " << workingGraph.nodes.size() << " nodes!" << std::endl;
        }
        std::cout << "  Stability cleaning: " << std::fixed << std::setprecision(2) 
                  << stabilityResult.stabilityData.cleaningTimeMs << " ms" << std::endl;
    std::cout << "  Kernel stitching: " << std::fixed << std::setprecision(2) 
              << stitchingTime.count() << " ms" << std::endl;
    std::cout << "    - Constant pool: " << std::fixed << std::setprecision(2)
              << constantPoolTime << " ms" << std::endl;
    std::cout << "    - Code generation: " << std::fixed << std::setprecision(2)
              << codeGenerationTime << " ms (" << nodesProcessed << " nodes)" << std::endl;
    std::cout << "    - Assembly finalization: " << std::fixed << std::setprecision(2)
              << assemblyFinalizationTime << " ms" << std::endl;
    
    // Print top operation types by time
    std::cout << "\n  Top operations by time:" << std::endl;
    std::vector<std::pair<std::string, double>> sortedOps;
    for (const auto& [op, time] : opTypeTime) {
        sortedOps.push_back({op, time});
    }
    std::sort(sortedOps.begin(), sortedOps.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    int shown = 0;
    for (const auto& [op, time] : sortedOps) {
        if (shown++ >= 5) break;  // Show top 5
        int count = opTypeCounts[op];
        double avgTime = count > 0 ? time / count : 0.0;
        std::cout << "    - " << op << ": " << std::fixed << std::setprecision(2)
                  << time << " ms (" << count << " ops, " 
                  << std::setprecision(3) << avgTime << " ms/op)" << std::endl;
    }
    
    std::cout << "  Total compilation: " << std::fixed << std::setprecision(2) 
              << totalTime.count() << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(0)
              << (workingGraph.nodes.size() * 1000.0 / totalTime.count()) << " nodes/sec" << std::endl;
    }
    
    return std::make_unique<forge::runtime::CompiledKernel>(func, s_runtime, stabilizedGraph.nodes.size(), instructionSet_.get(), config_, stabilityResult.stabilityData.originalToCleanedMapping, maxNodeIdAccessed, workingGraph.nodes.size());
}

void ForgeEngine::generatePrologue(asmjit::x86::Assembler& a) {
    // Delegate to instruction set implementation
    instructionSet_->emitPrologue(a);
}

void ForgeEngine::generateEpilogue(asmjit::x86::Assembler& a) {
    // Delegate to instruction set implementation
    instructionSet_->emitEpilogue(a);
}

// ============================================================================
// COMPILATION PHASE METHOD IMPLEMENTATIONS
// ============================================================================

ForgeEngine::StabilityResult ForgeEngine::performStabilityCleaning(const forge::core::ComputationGraph& graph) {
    // Phase 1: Stability cleaning for numerical safety
    // Input: graph (original computation graph), config_ (stability settings)
    // Output: stabilityResult (numerically safe graph + mapping + timing)
    auto stabilityData = analysis::StabilityCleaner::clean(graph, config_.enableStabilityCleaning);
    
    StabilityResult result;
    result.cleanedGraph = stabilityData.cleanedGraph;
    result.stabilityData = std::move(stabilityData);
    
    return result;
}

ForgeEngine::ConstantPoolData ForgeEngine::createConstantPool(const forge::core::ComputationGraph& graph, AssemblyContext& ctx) {
    // Phase 2: Constant pool management
    // Input: graph (stabilized graph), config_ (instruction set, optimization flags)
    // Output: constPoolData (deduplicated constants + memory layout + mappings + timing)
    ConstantPoolData result;
    result.poolManager = std::make_unique<generators::ConstantPoolManager>(config_);
    result.poolResult = result.poolManager->createConstantPool(graph, ctx.assembler);
    
    return result;
}

} // namespace compiler
} // namespace forge