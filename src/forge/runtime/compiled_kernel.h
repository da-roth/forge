#pragma once

#include <asmjit/x86.h>
#include "forge/core/computation_graph.h"
#include "forge/x86/instruction_set.h"
#include "forge/x86/compiler_config.h"
#include "forge/x86/runtime_trace.h"
#include "forge/runtime/kernel_requirements.h"
#include "forge/runtime/node_buffer.h"
#include <memory>
#include <vector>
#include <string>

namespace forge {
namespace runtime {

// Compiled kernel - holds the JIT function and manages its lifetime
class CompiledKernel {
public:
    // Extended signature: accepts optional gradient pointer
    using KernelFunc = void(*)(double* values, double* gradients, size_t count);
    //                                         ^^^^^^^^^^^^^^^^ NEW parameter
    
    CompiledKernel(KernelFunc func, asmjit::JitRuntime& runtime, size_t num_nodes, const forge::x86::IInstructionSet* instructionSet, const forge::x86::CompilerConfig& config, size_t max_node_id = 0, size_t working_nodes = 0)
        : func_(func), runtime_(&runtime), num_nodes_(num_nodes), instructionSet_(instructionSet), config_(config), max_node_id_(max_node_id), working_nodes_(working_nodes > 0 ? working_nodes : num_nodes) {}
    
    // Constructor with node ID mapping
    CompiledKernel(KernelFunc func, asmjit::JitRuntime& runtime, size_t num_nodes, const forge::x86::IInstructionSet* instructionSet, const forge::x86::CompilerConfig& config,
                   const std::vector<forge::core::NodeId>& originalToOptimizedMapping, size_t max_node_id = 0, size_t working_nodes = 0)
        : func_(func), runtime_(&runtime), num_nodes_(num_nodes), instructionSet_(instructionSet), config_(config),
          originalToOptimizedMapping_(originalToOptimizedMapping), max_node_id_(max_node_id), working_nodes_(working_nodes > 0 ? working_nodes : num_nodes) {
        // std::cout << "[KERNEL CONSTRUCTOR] num_nodes=" << num_nodes_ 
        //           << ", max_node_id=" << max_node_id_ 
        //           << ", working_nodes=" << working_nodes_ 
        //           << ", getRequiredNodes()=" << getRequiredNodes() 
        //           << ", getMaxNodeId()=" << getMaxNodeId() << std::endl;
    }
    
    ~CompiledKernel() {
        if (func_ && runtime_) {
            runtime_->release(func_);
        }
    }
    
    // Execute the kernel with a NodeValueBuffer
    void execute(INodeBuffer& buffer) {
        // std::cout << "\n[KERNEL] ==> execute() called" << std::endl;
        // std::cout << "[KERNEL] Instruction set: " << instructionSet_->getName() << std::endl;
        // std::cout << "[KERNEL] func_ is " << (func_ ? "valid" : "NULL") << std::endl;
        // std::cout << "[KERNEL] buffer.getNumNodes() = " << buffer.getNumNodes() 
        //           << ", num_nodes_ = " << num_nodes_ 
        //           << ", getRequiredNodes() = " << getRequiredNodes() << std::endl;
        // std::cout << "[KERNEL] Using getRequiredNodes() for size check (which is what buffer was sized with)" << std::endl;
        
        if (func_ && buffer.getNumNodes() >= getRequiredNodes()) {
            // std::cout << "[KERNEL] Conditions met, executing kernel function..." << std::endl;
            
            // Pass gradient pointer if available, otherwise nullptr
            // std::cout << "[KERNEL] Calling func_ (the compiled kernel)..." << std::endl;
            func_(buffer.getValuesPtr(), buffer.getGradientsPtr(), buffer.getNumNodes());
            // std::cout << "[KERNEL] func_ returned successfully" << std::endl;
            
            // Print runtime trace after forward pass if tracing is enabled
            // std::cout << "[KERNEL] Checking if tracing is enabled..." << std::endl;
            bool tracingEnabled = forge::x86::isTracingEnabled();
            // std::cout << "[KERNEL] isTracingEnabled() returned: " << (tracingEnabled ? "TRUE" : "FALSE") << std::endl;
            // std::cout << "[KERNEL] g_traceBuffer.enabled = " << (forge::x86::g_traceBuffer.enabled ? "true" : "false") << std::endl;
            // std::cout << "[KERNEL] g_traceBuffer.records = " << (forge::x86::g_traceBuffer.records ? "not null" : "NULL") << std::endl;
            
            if (tracingEnabled) {
                // std::cout << "[KERNEL] ==> ENTERING TRACE PRINT BLOCK" << std::endl;
                // std::cout << "[TRACE] Starting runtime trace records print..." << std::endl;
                // std::cout << "[TRACE] Instruction set: " << instructionSet_->getName() << std::endl;
                // std::cout << "[TRACE] Trace buffer enabled: " << (forge::runtime::g_traceBuffer.enabled ? "yes" : "no") << std::endl;
                // std::cout << "[TRACE] Records captured: " << forge::runtime::g_traceBuffer.index.load() << std::endl;
                
                // std::cout << "[KERNEL] About to call printTraceRecords()..." << std::endl;
                forge::x86::printTraceRecords();
                // std::cout << "[KERNEL] printTraceRecords() returned" << std::endl;
                
                // std::cout << "[TRACE] Finished runtime trace records print." << std::endl;
                // std::cout << "[KERNEL] ==> EXITING TRACE PRINT BLOCK" << std::endl;
            } else {
                // std::cout << "[KERNEL] ==> SKIPPING TRACE PRINT (tracing not enabled)" << std::endl;
                // std::cout << "[TRACE] Runtime tracing is NOT enabled (isTracingEnabled() returned false)" << std::endl;
            }
        } else {
            // std::cout << "[KERNEL] Conditions NOT met, skipping execution" << std::endl;
            // if (!func_) std::cout << "[KERNEL] Reason: func_ is NULL" << std::endl;
            // if (buffer.getNumNodes() < getRequiredNodes()) {
            //     std::cout << "[KERNEL] Reason: buffer too small (" << buffer.getNumNodes() << " < " << getRequiredNodes() << ")" << std::endl;
            //     std::cout << "[KERNEL] Note: num_nodes_=" << num_nodes_ << " but buffer sized for getRequiredNodes()=" << getRequiredNodes() << std::endl;
            // }
        }
        
        // std::cout << "[KERNEL] ==> execute() completed" << std::endl;
    }
    
    // Get the raw function pointer (for benchmarking)
    KernelFunc getFunction() const { return func_; }
    
    // Get instruction set information (for workspace compatibility)
    int getVectorWidth() const { return instructionSet_->getVectorWidth(); }
    std::string getInstructionSetName() const { return instructionSet_->getName(); }
    
    // Get the maximum node ID accessed by this kernel
    size_t getMaxNodeId() const { return max_node_id_ > 0 ? max_node_id_ : working_nodes_ - 1; }
    size_t getRequiredNodes() const { 
        return getMaxNodeId() + 1;
    }
    
    // Get the node ID mapping (for NodeValueBuffer creation)
    const std::vector<forge::core::NodeId>& getOriginalToOptimizedMapping() const { 
        return originalToOptimizedMapping_; 
    }
    
    // Get kernel requirements for buffer creation (new method)
    KernelRequirements getRequirements() const {
        // Determine memory alignment based on instruction set
        size_t alignment = 64;  // Default cache-line alignment
        if (instructionSet_) {
            int width = instructionSet_->getVectorWidth();
            if (width == 4) {
                alignment = 32;  // AVX2 requires 32-byte alignment
            } else if (width == 2) {
                alignment = 16;  // SSE requires 16-byte alignment
            }
        }
        
        // Convert NodeId vector to uint32_t vector
        std::vector<uint32_t> mapping;
        mapping.reserve(originalToOptimizedMapping_.size());
        for (auto nodeId : originalToOptimizedMapping_) {
            mapping.push_back(static_cast<uint32_t>(nodeId));
        }
        
        // Create requirements using struct initialization
        KernelRequirements req;
        req.vectorWidth = instructionSet_ ? instructionSet_->getVectorWidth() : 1;
        req.requiredNodes = getRequiredNodes();
        req.memoryAlignment = alignment;
        req.nodeMapping = std::move(mapping);
        req.needsGradients = true;  // Kernels always support gradients (may be nullptr at runtime)
        
        return req;
    }
    
    // Disable copy
    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;
    
    // Enable move
    CompiledKernel(CompiledKernel&& other) noexcept
        : func_(other.func_), runtime_(other.runtime_), num_nodes_(other.num_nodes_), 
          instructionSet_(other.instructionSet_), config_(other.config_), originalToOptimizedMapping_(std::move(other.originalToOptimizedMapping_)),
          max_node_id_(other.max_node_id_), working_nodes_(other.working_nodes_) {
        other.func_ = nullptr;
        other.runtime_ = nullptr;
        other.instructionSet_ = nullptr;
        other.max_node_id_ = 0;
        other.working_nodes_ = 0;
    }
    
private:
    KernelFunc func_;
    asmjit::JitRuntime* runtime_;  // Points to shared static runtime
    size_t num_nodes_;              // Original graph size (for buffer compatibility)
    const forge::x86::IInstructionSet* instructionSet_;  // Instruction set used for compilation
    forge::x86::CompilerConfig config_;         // Compiler configuration
    size_t max_node_id_;           // Maximum node ID accessed during compilation
    size_t working_nodes_;         // Working graph size (after optimizations)
    std::vector<forge::core::NodeId> originalToOptimizedMapping_;  // Node ID mapping
};


} // namespace runtime
} // namespace forge