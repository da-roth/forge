#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace forge {
namespace runtime {

/**
 * Requirements for kernel execution - describes what a compiled kernel needs
 * This struct serves as the contract between compilation and runtime execution.
 * 
 * This is a pure data structure with no dependencies, allowing clean separation
 * between the kernel (CompiledKernel) and the buffer system (INodeBuffer).
 */
struct KernelRequirements {
    // Instruction set requirements
    int vectorWidth;              // Number of values processed in parallel (1=scalar, 4=AVX2, etc.)
    
    // Memory requirements
    size_t requiredNodes;         // Number of nodes the buffer must accommodate
    size_t memoryAlignment;       // Required memory alignment in bytes (16 for SSE, 32 for AVX2)
    
    // Optimization mapping
    std::vector<uint32_t> nodeMapping;  // Maps original node IDs to optimized node IDs
                                        // Empty if no optimization was performed
    
    // Gradient computation requirements
    bool needsGradients;          // Whether gradient buffer is required
    
    // Default constructor
    KernelRequirements() 
        : vectorWidth(1)
        , requiredNodes(0)
        , memoryAlignment(64)    // Default to cache-line alignment
        , needsGradients(false) {}
    
    // Constructor with all fields
    KernelRequirements(int width, size_t nodes, size_t align, 
                      const std::vector<uint32_t>& mapping, bool gradients)
        : vectorWidth(width)
        , requiredNodes(nodes)
        , memoryAlignment(align)
        , nodeMapping(mapping)
        , needsGradients(gradients) {}
};

} // namespace runtime
} // namespace forge