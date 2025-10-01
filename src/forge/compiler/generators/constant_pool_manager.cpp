#include "constant_pool_manager.h"
#include "forge/x86/instruction_set.h"
#include <iostream>
#include <algorithm>
#include <chrono>

namespace forge::compiler::generators {

using namespace asmjit;
using namespace forge::core;

ConstantPoolManager::ConstantPoolManager(const forge::x86::CompilerConfig& config)
    : config_(config) {
}

ConstantPoolManager::ConstantPoolResult ConstantPoolManager::createConstantPool(
    const ComputationGraph& graph,
    asmjit::x86::Assembler& assembler) {
    
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    
    auto start = Clock::now();
    
    // Create zone and constant pool - store them in the manager
    zone_ = std::make_unique<asmjit::Zone>(1024);
    constPool_ = std::make_unique<asmjit::ConstPool>(zone_.get());
    
    asmjit::Label constPoolLabel = assembler.newLabel();
    std::unordered_map<NodeId, ConstantInfo> constantMap;
    
    // First pass: collect all Constant nodes and add to pool
    for (NodeId nodeId = 0; nodeId < graph.nodes.size(); ++nodeId) {
        const Node& node = graph.nodes[nodeId];
        if (node.isDead) continue;  // Skip dead nodes from optimization
        if (node.op == OpCode::Constant) {
            // Get the actual constant value from graph.constPool
            size_t constIndex = static_cast<size_t>(node.imm);
            if (constIndex >= graph.constPool.size()) {
                throw std::runtime_error("Invalid constant index");
            }
            
            // CRITICAL FIX: Pass the address directly from graph.constPool, not a local variable!
            // The pool needs to copy the data from a valid address
            const double* constantPtr = &graph.constPool[constIndex];
            double constantValue = *constantPtr;  // Keep for debug output
            
            // Add constant to pool (bit-exact deduplication happens automatically)
            size_t offset = 0;
            constPool_->add(constantPtr, sizeof(double), offset);
            constantMap[nodeId] = {offset, constantValue};
        }
    }
    
    double creationTime = Duration(Clock::now() - start).count();
    
    // Return result structure - frequency analysis and preloading will be done separately
    ConstantPoolResult result;
    result.constPool = constPool_.get();  // Return pointer
    result.constPoolLabel = constPoolLabel;
    result.constantMap = std::move(constantMap);
    result.pinnedConstants.clear();  // To be filled by preloadHotConstants
    result.numPinnedConstants = 0;
    result.creationTimeMs = creationTime;
    
    return result;
}

double ConstantPoolManager::embedConstantPool(
    asmjit::x86::Assembler& assembler,
    const ConstantPoolResult& result) {
    
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    
    auto start = Clock::now();
    
    if (result.constPool && result.constPool->size() > 0) {
        // CRITICAL FIX: Don't manually bind! embedConstPool does align→bind→emit for us
        // assembler.align(AlignMode::kData, 32);  // Optional, embedConstPool handles alignment
        assembler.embedConstPool(result.constPoolLabel, *result.constPool);  // This does align→bind→emit
        // Successfully embedded constant pool with size: constPool.size() bytes
    }
    
    return Duration(Clock::now() - start).count();
}

void ConstantPoolManager::preloadHotConstants(
    asmjit::x86::Assembler& assembler,
    ConstantPoolResult& result,
    forge::x86::IInstructionSet* instructionSet) {
    
    // CONSTANT POOLING: Analyze constant usage frequency and preload hot constants
    std::unordered_map<double, int> constantFrequency;
    std::unordered_map<double, std::vector<NodeId>> constantNodes;
    
    // This would need the graph - we'll pass it as parameter later
    // For now, create empty maps to maintain interface
    result.pinnedConstants.clear();
    result.numPinnedConstants = 0;
}

void ConstantPoolManager::analyzeConstantFrequency(
    const ComputationGraph& graph,
    const std::unordered_map<NodeId, ConstantInfo>& constantMap,
    std::unordered_map<double, int>& constantFrequency,
    std::unordered_map<double, std::vector<NodeId>>& constantNodes) {
    
    // Count frequency of each constant value
    for (NodeId nodeId = 0; nodeId < graph.nodes.size(); ++nodeId) {
        const Node& node = graph.nodes[nodeId];
        if (node.isDead) continue;
        
        // Count references to constant nodes
        if (node.a != static_cast<NodeId>(-1)) {
            const Node& aNode = graph.nodes[node.a];
            if (aNode.op == OpCode::Constant && !aNode.isDead) {
                auto it = constantMap.find(node.a);
                if (it != constantMap.end()) {
                    constantFrequency[it->second.value]++;
                    constantNodes[it->second.value].push_back(node.a);
                }
            }
        }
        if (node.b != static_cast<NodeId>(-1)) {
            const Node& bNode = graph.nodes[node.b];
            if (bNode.op == OpCode::Constant && !bNode.isDead) {
                auto it = constantMap.find(node.b);
                if (it != constantMap.end()) {
                    constantFrequency[it->second.value]++;
                    constantNodes[it->second.value].push_back(node.b);
                }
            }
        }
    }
}

void ConstantPoolManager::printConstantPoolingInfo(
    const std::unordered_map<double, int>& constantFrequency,
    const std::vector<std::pair<double, int>>& sortedConstants) {
    
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
}

} // namespace forge::compiler::generators