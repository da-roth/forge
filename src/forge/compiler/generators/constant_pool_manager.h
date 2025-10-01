#pragma once

#include "forge/core/computation_graph.h"
#include "forge/x86/compiler_config.h"
#include "forge/x86/instruction_set.h"
#include <asmjit/x86.h>
#include <unordered_map>
#include <vector>
#include <memory>

namespace forge::compiler::generators {

// ConstantInfo struct definition
struct ConstantInfo {
    size_t poolOffset;  // Offset within the constant pool
    double value;       // The constant value
};

/**
 * Manages constant pool creation, optimization, and embedding for JIT compilation.
 * Handles constant deduplication, frequency analysis, and register pre-loading.
 */
class ConstantPoolManager {
public:
    /**
     * Result of constant pool creation containing pool data and node mappings
     */
    struct ConstantPoolResult {
        asmjit::ConstPool* constPool;  // Pointer to avoid copy issues
        asmjit::Label constPoolLabel;
        std::unordered_map<forge::core::NodeId, ConstantInfo> constantMap;
        std::unordered_map<double, int> pinnedConstants;  // value -> register index
        int numPinnedConstants;
        double creationTimeMs;
    };

    /**
     * Create constant pool manager with given configuration
     */
    explicit ConstantPoolManager(const forge::x86::CompilerConfig& config);

    /**
     * Create constant pool from computation graph
     * @param graph The computation graph containing constants
     * @param assembler The assembler to create labels with
     * @return Complete constant pool result with mappings
     */
    ConstantPoolResult createConstantPool(
        const forge::core::ComputationGraph& graph,
        asmjit::x86::Assembler& assembler);

    /**
     * Embed constant pool into assembled code
     * @param assembler The assembler to embed into
     * @param result The constant pool result from createConstantPool
     * @return Embedding time in milliseconds
     */
    double embedConstantPool(
        asmjit::x86::Assembler& assembler,
        const ConstantPoolResult& result);

    /**
     * Preload hot constants into dedicated registers (XMM12-XMM15)
     * @param assembler The assembler to generate code with
     * @param result The constant pool result
     * @param instructionSet The instruction set to use for loading
     */
    void preloadHotConstants(
        asmjit::x86::Assembler& assembler,
        ConstantPoolResult& result,
        forge::x86::IInstructionSet* instructionSet);

private:
    const forge::x86::CompilerConfig& config_;
    std::unique_ptr<asmjit::ConstPool> constPool_;  // Store the pool
    std::unique_ptr<asmjit::Zone> zone_;  // Store the zone

    /**
     * Analyze constant usage frequency for optimization
     */
    void analyzeConstantFrequency(
        const forge::core::ComputationGraph& graph,
        const std::unordered_map<forge::core::NodeId, ConstantInfo>& constantMap,
        std::unordered_map<double, int>& constantFrequency,
        std::unordered_map<double, std::vector<forge::core::NodeId>>& constantNodes);

    /**
     * Print constant pooling debug information
     */
    void printConstantPoolingInfo(
        const std::unordered_map<double, int>& constantFrequency,
        const std::vector<std::pair<double, int>>& sortedConstants);
};

} // namespace forge::compiler::generators