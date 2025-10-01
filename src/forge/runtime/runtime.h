#pragma once

// Forge Runtime Components
// This header provides all runtime execution components for JIT-compiled kernels

#include "forge/runtime/kernel_requirements.h"
#include "forge/runtime/compiled_kernel.h"
#include "forge/runtime/node_buffer.h"
#include "forge/runtime/scalar_buffer.h"
#include "forge/runtime/avx2_buffer.h"

namespace forge {
namespace runtime {

// Convenience using declarations
using CompiledKernelPtr = std::unique_ptr<CompiledKernel>;
using NodeBufferPtr = std::unique_ptr<INodeBuffer>;

} // namespace runtime
} // namespace forge
