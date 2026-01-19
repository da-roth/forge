// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file forge_engine.cpp
 * @brief Implementation of the ForgeEngine JIT compiler
 *
 * Implements the main compilation pipeline: graph optimization, forward pass
 * code generation, gradient pass generation, and JIT assembly via AsmJit.
 */

#include "forge_engine.hpp"
#include "../graph/graph_optimizer.hpp"
#include "backward_forging.hpp"
#include "forward_forging.hpp"
#include "x86/double/scalar/sse2_scalar_instruction_set.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>  // For std::swap
#include <cmath>      // For std::exp, std::log
#include <chrono>     // For timing
#include <asmjit/core.h>

// Force link AVX2 registration when bundled
#ifdef FORGE_BUNDLE_AVX2
extern "C" void forge_force_avx2_registration();
#endif

namespace forge {

using namespace asmjit;
using namespace forge;

// Define the static JitRuntime (shared across all compilers)
asmjit::JitRuntime ForgeEngine::s_runtime;

ForgeEngine::ForgeEngine() : config_(CompilerConfig::Default()) {
#ifdef FORGE_BUNDLE_AVX2
    // Ensure AVX2 registration is linked (prevents dead code elimination)
    forge_force_avx2_registration();
#endif
    // Create instruction set based on config - MUST pass the full config!
    instructionSet_ = InstructionSetFactory::create(config_.instructionSet, config_);
    // Initialize with default policy
    policy_ = std::make_unique<DefaultCompilationPolicy>();
}

ForgeEngine::ForgeEngine(const CompilerConfig& config) : config_(config) {
#ifdef FORGE_BUNDLE_AVX2
    // Ensure AVX2 registration is linked (prevents dead code elimination)
    forge_force_avx2_registration();
#endif
    // Create instruction set based on config
    if (config_.useNamedInstructionSet && !config_.instructionSetName.empty()) {
        // Use dynamically registered instruction set by name
        instructionSet_ = InstructionSetFactory::createByName(config_.instructionSetName, config_);
    } else {
        // Use enum-based selection (built-in instruction sets)
        instructionSet_ = InstructionSetFactory::create(config_.instructionSet, config_);
    }
    // Initialize with default policy
    policy_ = std::make_unique<DefaultCompilationPolicy>();
}

ForgeEngine::~ForgeEngine() = default;

asmjit::JitRuntime& ForgeEngine::getRuntime() {
    return s_runtime;
}

std::unique_ptr<IRegisterAllocator> ForgeEngine::createRegisterAllocator() const {
    // For dynamically loaded backends, use vector width to determine allocator
    if (config_.useNamedInstructionSet && instructionSet_) {
        int vectorWidth = instructionSet_->getVectorWidth();
        if (vectorWidth >= 4) {
#ifdef FORGE_BUNDLE_AVX2
            return std::make_unique<YmmRegisterAllocator>();
#else
            throw std::runtime_error("AVX2 register allocator not available. AVX2 backend not bundled.");
#endif
        } else {
            return std::make_unique<XmmRegisterAllocator>();
        }
    }

    // Create appropriate allocator based on instruction set enum
    switch (config_.instructionSet) {
        case CompilerConfig::InstructionSet::AVX2_PACKED:
#ifdef FORGE_BUNDLE_AVX2
            return std::make_unique<YmmRegisterAllocator>();
#else
            throw std::runtime_error("AVX2 register allocator not available. AVX2 backend not bundled.");
#endif
        case CompilerConfig::InstructionSet::SSE2_SCALAR:
        default:
            return std::make_unique<XmmRegisterAllocator>();
    }
}

// Helper function to get operation name for timing
static std::string getOpName(forge::OpCode op) {
    switch(op) {
        case forge::OpCode::Input: return "Input";
        case forge::OpCode::Constant: return "Constant";
        case forge::OpCode::Add: return "Add";
        case forge::OpCode::Sub: return "Sub";
        case forge::OpCode::Mul: return "Mul";
        case forge::OpCode::Div: return "Div";
        case forge::OpCode::Neg: return "Neg";
        case forge::OpCode::Abs: return "Abs";
        case forge::OpCode::Square: return "Square";
        case forge::OpCode::Recip: return "Recip";
        case forge::OpCode::Sqrt: return "Sqrt";
        case forge::OpCode::Pow: return "Pow";
        case forge::OpCode::Exp: return "Exp";
        case forge::OpCode::Log: return "Log";
        case forge::OpCode::Sin: return "Sin";
        case forge::OpCode::Cos: return "Cos";
        case forge::OpCode::Tan: return "Tan";
        case forge::OpCode::Mod: return "Mod";
        case forge::OpCode::Min: return "Min";
        case forge::OpCode::Max: return "Max";
        case forge::OpCode::If: return "If";
        default: return "Unknown";
    }
}

std::unique_ptr<StitchedKernel> ForgeEngine::compile(const Graph& graph) {
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    
    auto totalStart = Clock::now();
    
    // Phase 0: Apply graph-level optimizations before JIT compilation
    auto optimizationStart = Clock::now();
    
    // Configure the tape optimizer based on our config
    GraphOptimizer optimizer;
    GraphOptimizer::OptimizationConfig optConfig;
    optConfig.enableInactiveFolding = config_.enableInactiveFolding;
    optConfig.enableCSE = config_.enableCSE;
    optConfig.enableAlgebraicSimplification = config_.enableAlgebraicSimplification;
    optConfig.enableStabilityCleaning = config_.enableStabilityCleaning;
    optConfig.maxOptimizationPasses = config_.maxOptimizationPasses;
    optConfig.printOriginalGraph = config_.printOriginalGraph;
    optConfig.printOptimizedGraph = config_.printOptimizedGraph;
    optConfig.printStepByStepDebug = config_.printStepByStepDebug;
    optimizer.setConfig(optConfig);
    
    // Validate outputs exist
    if (graph.outputs.empty()) {
        throw std::runtime_error("No outputs were marked on the graph. Ensure markOutput() is called.");
    }

    // Use the new mapping-based optimization
    GraphOptimizer::OptimizationResult optResult;
    if (config_.enableOptimizations) {
        optResult = optimizer.optimizeWithMapping(graph);
    } else {
        // When optimizations are disabled, supply identity mapping
        std::vector<forge::NodeId> identity(graph.nodes.size());
        for (size_t i = 0; i < identity.size(); ++i) identity[i] = static_cast<forge::NodeId>(i);
        optResult = GraphOptimizer::OptimizationResult{graph, std::move(identity)};
    }
    
    Graph optimizedGraph = optResult.optimizedTape;
    
    // Store the mapping for later use by NodeValueBuffer
    // This will be used when creating the StitchedKernel
    
    auto optimizationEnd = Clock::now();
    Duration optimizationTime = optimizationEnd - optimizationStart;
    
    // Print optimization statistics
    const auto& stats = optimizer.getLastStats();
    
    // Only print optimization info if debug flags are set
    if (config_.printOptimizationStats || config_.printOriginalGraph || config_.printOptimizedGraph) {
        std::cout << "\n=== Graph Optimization Info ===" << std::endl;
        std::cout << "  Original nodes: " << stats.originalNodeCount << std::endl;
        std::cout << "  Inactive subgraphs folded: " << stats.inactiveNodesFolded << std::endl;
        std::cout << "  Duplicates eliminated (CSE): " << stats.duplicatesEliminated << std::endl;
        std::cout << "  Algebraic simplifications: " << stats.algebraicSimplifications << std::endl;
        std::cout << "  Stability fixes applied: " << stats.stabilityFixes << std::endl;
        
        // Count actual dead nodes in the graph
        int actualDeadCount = 0;
        for (const auto& node : optimizedGraph.nodes) {
            if (node.isDead) actualDeadCount++;
        }
        std::cout << "  Dead nodes marked: " << actualDeadCount << std::endl;
        std::cout << "  Effective nodes (not dead): " << (stats.originalNodeCount - actualDeadCount) << std::endl;
        std::cout << "  Optimization ratio: " << std::fixed << std::setprecision(1) 
                  << (100.0 * actualDeadCount / stats.originalNodeCount) << "% nodes eliminated" << std::endl;
        std::cout << "  Optimization time: " << std::fixed << std::setprecision(2) 
                  << optimizationTime.count() << " ms" << std::endl;
        
        // Helper lambda to print node details
        auto printNode = [](const forge::Node& node, size_t idx) {
            std::cout << "    Node " << idx << ": ";
            switch(node.op) {
                case forge::OpCode::Input: std::cout << "Input"; break;
                case forge::OpCode::Constant: std::cout << "Constant"; break;
                case forge::OpCode::Add: std::cout << "Add"; break;
                case forge::OpCode::Sub: std::cout << "Sub"; break;
                case forge::OpCode::Mul: std::cout << "Mul"; break;
                case forge::OpCode::Div: std::cout << "Div"; break;
                case forge::OpCode::Neg: std::cout << "Neg"; break;
                case forge::OpCode::Abs: std::cout << "Abs"; break;
                case forge::OpCode::Square: std::cout << "Square"; break;
                case forge::OpCode::Recip: std::cout << "Recip"; break;
                case forge::OpCode::Mod: std::cout << "Mod"; break;
                case forge::OpCode::Exp: std::cout << "Exp"; break;
                case forge::OpCode::Log: std::cout << "Log"; break;
                case forge::OpCode::Pow: std::cout << "Pow"; break;
                case forge::OpCode::Sqrt: std::cout << "Sqrt"; break;
                case forge::OpCode::Sin: std::cout << "Sin"; break;
                case forge::OpCode::Cos: std::cout << "Cos"; break;
                case forge::OpCode::Tan: std::cout << "Tan"; break;
                case forge::OpCode::Min: std::cout << "Min"; break;
                case forge::OpCode::Max: std::cout << "Max"; break;
                case forge::OpCode::If: std::cout << "If"; break;
                case forge::OpCode::CmpLT: std::cout << "CmpLT"; break;
                case forge::OpCode::CmpLE: std::cout << "CmpLE"; break;
                case forge::OpCode::CmpGT: std::cout << "CmpGT"; break;
                case forge::OpCode::CmpGE: std::cout << "CmpGE"; break;
                case forge::OpCode::CmpEQ: std::cout << "CmpEQ"; break;
                case forge::OpCode::CmpNE: std::cout << "CmpNE"; break;
                // Boolean operations
                case forge::OpCode::BoolConstant: std::cout << "BoolConstant"; break;
                case forge::OpCode::BoolAnd: std::cout << "BoolAnd"; break;
                case forge::OpCode::BoolOr: std::cout << "BoolOr"; break;
                case forge::OpCode::BoolNot: std::cout << "BoolNot"; break;
                case forge::OpCode::BoolEq: std::cout << "BoolEq"; break;
                case forge::OpCode::BoolNe: std::cout << "BoolNe"; break;
                // Integer operations
                case forge::OpCode::IntConstant: std::cout << "IntConstant"; break;
                case forge::OpCode::IntAdd: std::cout << "IntAdd"; break;
                case forge::OpCode::IntSub: std::cout << "IntSub"; break;
                case forge::OpCode::IntMul: std::cout << "IntMul"; break;
                case forge::OpCode::IntDiv: std::cout << "IntDiv"; break;
                case forge::OpCode::IntMod: std::cout << "IntMod"; break;
                case forge::OpCode::IntNeg: std::cout << "IntNeg"; break;
                case forge::OpCode::IntCmpLT: std::cout << "IntCmpLT"; break;
                case forge::OpCode::IntCmpLE: std::cout << "IntCmpLE"; break;
                case forge::OpCode::IntCmpGT: std::cout << "IntCmpGT"; break;
                case forge::OpCode::IntCmpGE: std::cout << "IntCmpGE"; break;
                case forge::OpCode::IntCmpEQ: std::cout << "IntCmpEQ"; break;
                case forge::OpCode::IntCmpNE: std::cout << "IntCmpNE"; break;
                case forge::OpCode::IntIf: std::cout << "IntIf"; break;
                case forge::OpCode::ArrayIndex: std::cout << "ArrayIndex"; break;
                default: std::cout << "Op" << static_cast<int>(node.op); break;
            }
            std::cout << "(";
            if (node.a != UINT32_MAX) std::cout << node.a;
            if (node.b != UINT32_MAX) std::cout << "," << node.b;
            if (node.c != UINT32_MAX) std::cout << "," << node.c;
            std::cout << ")";
            if (node.op == forge::OpCode::Constant) {
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
        
        // Print optimized graph if requested
        if (config_.printOptimizedGraph) {
            std::cout << "\n  Optimized Graph:" << std::endl;
            for (size_t i = 0; i < optimizedGraph.nodes.size(); ++i) {
                printNode(optimizedGraph.nodes[i], i);
            }
        }
        
        if (config_.printNodeFlags) {
            std::cout << "\n  Node flags in optimized graph:" << std::endl;
            for (size_t i = 0; i < optimizedGraph.nodes.size(); ++i) {
                std::cout << "    Node " << i << ": needsGradient=" << optimizedGraph.nodes[i].needsGradient 
                          << " isActive=" << optimizedGraph.nodes[i].isActive << std::endl;
            }
        }
    }
    
    // Use the optimized graph for compilation
    const Graph& workingGraph = optimizedGraph;
    
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
    x86::Assembler a(&code);
    
    // Enable validation to catch assembly errors (as suggested by specialist)
    a.addDiagnosticOptions(asmjit::DiagnosticOptions::kValidateAssembler);
    Duration codeInitTime = Clock::now() - codeInitStart;
    
    // Phase 2.2: Create constant pool
    auto constantPoolStart = Clock::now();
    // Use a separate Zone as shown in the specialist's example
    Zone zone(1024);
    ConstPool constPool(&zone);
    
    // Create the label for the constant pool BEFORE adding constants (as per specialist's example)
    Label constPoolLabel = a.newLabel();
    
    std::unordered_map<NodeId, ConstantInfo> constantMap;
    
    // First pass: collect all Constant nodes and add to pool
    for (NodeId nodeId = 0; nodeId < workingGraph.nodes.size(); ++nodeId) {
        const Node& node = workingGraph.nodes[nodeId];
        if (node.isDead) continue;  // Skip dead nodes from optimization
        if (node.op == OpCode::Constant) {
            // Get the actual constant value from workingGraph.constPool
            size_t constIndex = static_cast<size_t>(node.imm);
            if (constIndex >= workingGraph.constPool.size()) {
                throw std::runtime_error("Invalid constant index");
            }
            
            // CRITICAL FIX: Pass the address directly from workingGraph.constPool, not a local variable!
            // The pool needs to copy the data from a valid address
            const double* constantPtr = &workingGraph.constPool[constIndex];
            double constantValue = *constantPtr;  // Keep for debug output
            
            // Add constant to pool (bit-exact deduplication happens automatically)
            size_t offset = 0;
            constPool.add(constantPtr, sizeof(double), offset);
            constantMap[nodeId] = {offset, constantValue};
        }
    }
    
    constantPoolTime = Duration(Clock::now() - constantPoolStart).count();
    
    // Generate function prologue
    auto prologueStart = Clock::now();
    instructionSet_->emitPrologue(a);
    Duration prologueTime = Clock::now() - prologueStart;
    
    // Phase 2.3: Initialize register tracking state
    // Create appropriate allocator based on instruction set
    auto regStatePtr = createRegisterAllocator();
    IRegisterAllocator& regState = *regStatePtr;
    
    // Track maximum node ID accessed for proper buffer allocation
    NodeId maxNodeIdAccessed = 0;
    
    // CONSTANT POOLING: Analyze constant usage frequency and preload hot constants
    std::unordered_map<double, int> constantFrequency;
    std::unordered_map<double, std::vector<NodeId>> constantNodes;
    
    // Count frequency of each constant value
    for (NodeId nodeId = 0; nodeId < workingGraph.nodes.size(); ++nodeId) {
        const Node& node = workingGraph.nodes[nodeId];
        if (node.isDead) continue;
        
        // Count references to constant nodes
        if (node.a != static_cast<NodeId>(-1)) {
            const Node& aNode = workingGraph.nodes[node.a];
            if (aNode.op == OpCode::Constant && !aNode.isDead) {
                auto it = constantMap.find(node.a);
                if (it != constantMap.end()) {
                    constantFrequency[it->second.value]++;
                    constantNodes[it->second.value].push_back(node.a);
                }
            }
        }
        if (node.b != static_cast<NodeId>(-1)) {
            const Node& bNode = workingGraph.nodes[node.b];
            if (bNode.op == OpCode::Constant && !bNode.isDead) {
                auto it = constantMap.find(node.b);
                if (it != constantMap.end()) {
                    constantFrequency[it->second.value]++;
                    constantNodes[it->second.value].push_back(node.b);
                }
            }
        }
    }
    
    // Find the top 4 most frequently used constants for XMM12-XMM15
    std::vector<std::pair<double, int>> sortedConstants;
    for (const auto& [value, freq] : constantFrequency) {
        if (freq > 1) {  // Only pool constants used more than once
            sortedConstants.push_back({value, freq});
        }
    }
    std::sort(sortedConstants.begin(), sortedConstants.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Debug: Print constant pooling info (optional - can be enabled for debugging)
    const bool debugConstantPooling = config_.printAssembly; // Debug constant pooling when printing assembly
    if (!sortedConstants.empty() && debugConstantPooling) {
        std::cout << "\n=== Constant Pooling Analysis ===" << std::endl;
        std::cout << "  Total unique constants: " << constantFrequency.size() << std::endl;
        std::cout << "  Constants used >1 time: " << sortedConstants.size() << std::endl;
        if (sortedConstants.size() > 0) {
            std::cout << "  Top constants to pin:" << std::endl;
            for (int i = 0; i < std::min(4, static_cast<int>(sortedConstants.size())); ++i) {
                std::cout << "    XMM" << (12 + i) << ": value=" << sortedConstants[i].first 
                         << " (used " << sortedConstants[i].second << " times)" << std::endl;
            }
        }
    }
    
    // Preload up to 4 hot constants into XMM12-XMM15
    int pinnedRegStart = 12;
    int numPinnedConstants = std::min(4, static_cast<int>(sortedConstants.size()));
    std::unordered_map<double, int> pinnedConstants;  // value -> register index
    
    for (int i = 0; i < numPinnedConstants; ++i) {
        double value = sortedConstants[i].first;
        int regIdx = pinnedRegStart + i;
        
        // Load constant into dedicated register
        
        // Find the constant's pool offset
        for (const auto& nodeId : constantNodes[value]) {
            auto it = constantMap.find(nodeId);
            if (it != constantMap.end()) {
                // Load from constant pool using instruction set abstraction
                instructionSet_->emitLoadFromConstantPool(a, regIdx, constPoolLabel, it->second.poolOffset);
                pinnedConstants[value] = regIdx;
                
                // Mark all nodes with this constant value to use the pinned register
                for (const auto& nid : constantNodes[value]) {
                    regState.setRegister(regIdx, nid, false);
                }
                
                // Lock the register to prevent eviction
                regState.lock(regIdx);
                break;
            }
        }
    }
    
    // Main code generation phase - process nodes sequentially
    auto codeGenStart = Clock::now();
    int nodesProcessed = 0;

    // Notify policy that compilation is beginning
    policy_->onCompileBegin(workingGraph, a);

    for (NodeId nodeId = 0; nodeId < workingGraph.nodes.size(); ++nodeId) {
        const Node& node = workingGraph.nodes[nodeId];
        if (node.isDead) continue;  // Skip dead nodes from optimization

        // Notify policy before node processing
        policy_->onNodeBegin(nodeId, a);

        // Track operation type timing
        auto opStart = Clock::now();
        std::string opName = getOpName(node.op);

        // Get store decision from policy (inverted: requiresStore=true means deferStore=false)
        bool deferStore = !policy_->requiresStore(nodeId, workingGraph);

        // Generate forward operation code
        ForwardForging::generateForwardOperation(a, node, nodeId, workingGraph, constantMap, constPoolLabel, regState, instructionSet_.get(), policy_.get(), deferStore);

        // Track maximum node ID
        maxNodeIdAccessed = std::max(maxNodeIdAccessed, nodeId);

        // Notify policy after node processing
        int resultReg = regState.findNodeInRegister(nodeId);
        policy_->onNodeEnd(nodeId, resultReg, a);

        double opTime = Duration(Clock::now() - opStart).count();
        opTypeTime[opName] += opTime;
        opTypeCounts[opName]++;
        nodesProcessed++;
    }

    // Notify policy that compilation is ending
    policy_->onCompileEnd(a);
    
    // Generate function epilogue
    codeGenerationTime = Duration(Clock::now() - codeGenStart).count();
    
    // Generate backward pass if needed (reuse needsGradient flag from earlier check)
    if (needsGradient) {
        // Check if gradients pointer is not null at runtime
        // Note: After prologue, RSI contains the gradients pointer (moved from RDX)
        Label skipGradient = a.newLabel();
        a.test(x86::rsi, x86::rsi);  // RSI = gradients pointer (already set in prologue)
        a.jz(skipGradient);  // Jump if gradients == nullptr
        
        // Generate gradient code (RSI already points to gradients)
        BackwardForging::forgeBackwardPass(a, workingGraph, constantMap, constPoolLabel, regState, instructionSet_.get(), &config_);
        
        a.bind(skipGradient);
    }
    
    // Generate function epilogue
    auto epilogueStart = Clock::now();
    instructionSet_->emitEpilogue(a);
    Duration epilogueTime = Clock::now() - epilogueStart;
    
    // Phase 2.2: Embed constant pool after code with proper alignment
    auto embedStart = Clock::now();
    if (constPool.size() > 0) {
        // CRITICAL FIX: Don't manually bind! embedConstPool does align→bind→emit for us
        // a.align(AlignMode::kData, 32);  // Optional, embedConstPool handles alignment
        a.embedConstPool(constPoolLabel, constPool);  // This does align→bind→emit
        // Successfully embedded constant pool with size: constPool.size() bytes
    }
    
    Duration embedTime = Clock::now() - embedStart;
    
    // Add the compiled function to runtime
    auto finalizeStart = Clock::now();
    StitchedKernel::KernelFunc func = nullptr;
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
    
    // Print timing information if optimization stats are requested
    if (config_.printOptimizationStats || config_.printOriginalGraph || config_.printOptimizedGraph) {
        std::cout << "\n=== JIT Compilation Timing ===" << std::endl;
        std::cout << "  Maximum node ID accessed: " << maxNodeIdAccessed 
                  << " (tape size: " << workingGraph.nodes.size() << ")" << std::endl;
        if (maxNodeIdAccessed >= workingGraph.nodes.size()) {
            std::cout << "  WARNING: Kernel accesses node " << maxNodeIdAccessed 
                      << " but tape only has " << workingGraph.nodes.size() << " nodes!" << std::endl;
        }
        std::cout << "  Graph optimization: " << std::fixed << std::setprecision(2) 
                  << optimizationTime.count() << " ms" << std::endl;
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
    
    return std::make_unique<StitchedKernel>(func, s_runtime, optimizedGraph.nodes.size(), instructionSet_.get(), config_, optResult.originalToOptimizedMapping, maxNodeIdAccessed, workingGraph.nodes.size(), workingGraph.outputs);
}

} // namespace forge
