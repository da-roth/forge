# Forge Backends

Forge's backend system enables custom instruction set implementations with full control over machine code generation, register allocation, and memory layouts. Backends can be bundled at compile time or loaded dynamically at runtime.

**Key interfaces** (see `src/compiler/interfaces/`):
- `IInstructionSet` — Emits machine code for all supported operations
- `INodeValueBuffer` — Manages value/gradient storage with custom alignment
- `IRegisterAllocator` — Controls register allocation strategy
- `ICompilationPolicy` — Decides when to store values to memory vs keep in registers (forward-optimized vs backward-compatible)

## Structure

```
backends/
└── double/           # Double-precision backends
    └── avx2/         # AVX2 4-wide packed implementation
```

## AVX2 Backend

The AVX2 backend processes 4 doubles in parallel using 256-bit YMM registers.

| File | Description |
|------|-------------|
| `avx2_instruction_set.hpp/cpp` | Code generation for all AVX2 operations |
| `avx2_node_value_buffer.hpp` | 32-byte aligned buffer for AVX2 data |
| `ymm_register_allocator.hpp` | YMM register allocation |
| `avx2_static_registration.cpp` | Registration when bundled into library |
| `avx2_backend.cpp` | Registration for loadable shared library |

**Build options:**
- `FORGE_BUNDLE_AVX2=ON` (default) — Include AVX2 in the main library
- `FORGE_BUILD_AVX2_BACKEND=ON` — Build as loadable `libforge_avx2.so`

## Implementing a Custom Backend

Custom backends allow you to add new SIMD instruction sets that can be loaded at runtime.

### Core Interfaces

Your backend must implement these interfaces from `src/compiler/interfaces/`:

**1. `IInstructionSet`** — Code generation for all operations

```cpp
class IInstructionSet {
public:
    virtual uint32_t apiVersion() const { return INSTRUCTION_SET_API_VERSION; }
    virtual std::string getName() const = 0;
    virtual int getVectorWidth() const = 0;

    // Arithmetic
    virtual void emitAdd(asmjit::x86::Assembler& a, int dst, int src) = 0;
    virtual void emitSub(asmjit::x86::Assembler& a, int dst, int src) = 0;
    virtual void emitMul(asmjit::x86::Assembler& a, int dst, int src) = 0;
    virtual void emitDiv(asmjit::x86::Assembler& a, int dst, int src) = 0;

    // Transcendentals
    virtual void emitExp(asmjit::x86::Assembler& a, int dst, int src) = 0;
    virtual void emitLog(asmjit::x86::Assembler& a, int dst, int src) = 0;
    virtual void emitSin(asmjit::x86::Assembler& a, int dst, int src) = 0;
    // ... etc
};
```

**2. `INodeValueBuffer`** — Storage for node values and gradients

Extend `NodeValueBufferBase<VectorWidth, Alignment>` which handles most of the implementation:

```cpp
class MyNodeValueBuffer : public NodeValueBufferBase<4, 32> {
public:
    using NodeValueBufferBase::NodeValueBufferBase;
};
```

**3. `IRegisterAllocator`** — Register allocation strategy

Extend `RegisterAllocatorBase<NumRegisters>`:

```cpp
class MyRegisterAllocator : public RegisterAllocatorBase<16> {
    // Override if you need custom allocation logic
};
```

### Step-by-Step Implementation

**Step 1: Create your instruction set class**

```cpp
// my_backend_instruction_set.hpp
#include "compiler/interfaces/instruction_set.hpp"
#include "compiler/x86/common/x86_instruction_set_base.hpp"

namespace forge {

class MyInstructionSet : public X86InstructionSetBase {
public:
    MyInstructionSet(const CompilerConfig& config = CompilerConfig::Default())
        : X86InstructionSetBase(config) {}

    std::string getName() const override { return "My-Backend"; }
    int getVectorWidth() const override { return 4; }

    void emitAdd(asmjit::x86::Assembler& a, int dst, int src) override {
        // Your implementation
    }

    // Implement all required operations...
};

} // namespace forge
```

**Step 2: Create buffer and register allocator**

```cpp
// my_backend_buffer.hpp
#include "compiler/interfaces/node_value_buffer.hpp"

namespace forge {

class MyNodeValueBuffer : public NodeValueBufferBase<4, 32> {
public:
    using NodeValueBufferBase::NodeValueBufferBase;
};

} // namespace forge
```

**Step 3: Create the registration entry point**

For loadable backends, export a registration function using the V2 callback API:

```cpp
// my_backend.cpp
#include "my_backend_instruction_set.hpp"
#include "my_backend_buffer.hpp"
#include "compiler/x86/common/instruction_set_factory.hpp"

namespace {

std::unique_ptr<forge::IInstructionSet> createMyInstructionSet() {
    return std::make_unique<forge::MyInstructionSet>();
}

std::unique_ptr<forge::INodeValueBuffer> createMyBuffer(
    const forge::Graph& graph,
    const std::vector<forge::NodeId>& mapping,
    size_t requiredNodes) {
    return std::make_unique<forge::MyNodeValueBuffer>(graph, mapping, requiredNodes);
}

} // namespace

extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
void forge_register_backend_v2(forge::ForgeBackendAPI* api) {
    api->registerInstructionSet("My-Backend", createMyInstructionSet);
    api->registerBufferCreator(4, createMyBuffer);  // 4 = vector width
}

} // extern "C"
```

**Step 4: Build as shared library**

```cmake
add_library(forge_mybackend SHARED
    my_backend_instruction_set.cpp
    my_backend.cpp
)
target_link_libraries(forge_mybackend PRIVATE asmjit)
target_compile_options(forge_mybackend PRIVATE -mavx2)  # or your flags
```

**Step 5: Load and use at runtime**

```cpp
#include "compiler/x86/common/instruction_set_factory.hpp"

// Load the backend
forge::InstructionSetFactory::loadBackend("./libforge_mybackend.so");

// Verify it's available
if (forge::InstructionSetFactory::hasInstructionSet("My-Backend")) {
    auto instructionSet = forge::InstructionSetFactory::createByName("My-Backend");
    // Use with ForgeEngine...
}
```

### API Version Compatibility

Backends include an API version to prevent crashes from interface mismatches:

```cpp
uint32_t apiVersion() const override {
    return INSTRUCTION_SET_API_VERSION;
}
```

If you build a backend against API version 1, it only works with Forge expecting version 1. Loading a mismatched backend throws an error with a clear message.

### Reference Implementation

The AVX2 backend in this directory is a complete reference:

- `avx2_instruction_set.hpp/cpp` — Full implementation of all operations
- `avx2_static_registration.cpp` — How to register when bundled
- `avx2_backend.cpp` — How to register for runtime loading

Study these files to understand the patterns and requirements for a complete backend.
