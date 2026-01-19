# Forge Backends

SIMD instruction set implementations for Forge.

## Structure

```
backends/
└── double/           # Double-precision backends
    └── avx2/         # AVX2 4-wide packed implementation
```

## AVX2 Backend

The AVX2 backend processes 4 doubles in parallel using 256-bit YMM registers.

**Key files:**
- `avx2_instruction_set.hpp/cpp` — Code generation for AVX2 instructions
- `avx2_node_value_buffer.hpp` — 32-byte aligned buffer for AVX2 operations
- `ymm_register_allocator.hpp` — YMM register allocation

**Build options:**
- `FORGE_BUNDLE_AVX2=ON` (default) — Include AVX2 in the main library
- `FORGE_BUILD_AVX2_BACKEND=ON` — Build as loadable `libforge_avx2.so`

## Implementing a Custom Backend

To add a new SIMD backend:

1. Implement `IInstructionSet` interface
2. Implement `NodeValueBufferBase<VectorWidth, Alignment>`
3. Register via `InstructionSetFactory::registerInstructionSet()`
4. Register buffer creator via `NodeValueBufferFactory::registerBufferCreator()`

See `avx2_static_registration.cpp` for bundled registration or `avx2_backend.cpp` for loadable backend registration.
