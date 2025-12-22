# Forge C API

Stable C interface for the Forge JIT compiler library, enabling binary compatibility across different C++ compilers and standard library implementations.

## Why C API?

The Forge library is written in C++ and uses features like `std::vector`, `std::unique_ptr`, and virtual functions. These have different ABIs across:
- Different compilers (GCC, Clang, MSVC)
- Different compiler versions
- Different C++ standard library implementations (libstdc++, libc++, MSVC STL)
- Different C++ standards (C++11 vs C++17)

The C API provides a stable binary interface that works with any compiler on a given platform. This means:
- **One package per platform** instead of one per compiler/version combo
- **Easy integration** with any language that supports C FFI (Python, Rust, Go, etc.)
- **Future-proof** - the ABI won't break with new compiler versions

## Building

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
cmake --install . --prefix /path/to/install
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `FORGE_CAPI_BUILD_TESTS` | ON | Build C API tests |

## Output

The build produces:
- `forge_capi.dll` / `libforge_capi.so` / `libforge_capi.dylib` - Shared library
- `forge_c_api.h` - Public C header

## Usage

```c
#include "forge_c_api.h"

int main() {
    // Create a graph: f(x) = x^2
    ForgeGraphHandle graph = forge_graph_create();
    uint32_t x = forge_graph_add_input(graph);
    uint32_t x_sq = forge_graph_add_unary(graph, FORGE_OP_SQUARE, x);
    forge_graph_mark_output(graph, x_sq);

    // Compile
    ForgeKernelHandle kernel = forge_compile(graph, NULL);

    // Create buffer and execute
    ForgeBufferHandle buffer = forge_buffer_create(graph, kernel);
    forge_buffer_set_value(buffer, x, 3.0);
    forge_execute(kernel, buffer);

    double result;
    forge_buffer_get_value(buffer, x_sq, &result);
    printf("3^2 = %f\n", result);  // Output: 9.0

    // Clean up
    forge_buffer_destroy(buffer);
    forge_kernel_destroy(kernel);
    forge_graph_destroy(graph);

    return 0;
}
```

## API Overview

### Graph Construction
- `forge_graph_create()` - Create empty graph
- `forge_graph_add_input()` - Add input node
- `forge_graph_add_constant(value)` - Add constant
- `forge_graph_add_unary(op, a)` - Add unary operation
- `forge_graph_add_binary(op, a, b)` - Add binary operation
- `forge_graph_add_ternary(op, a, b, c)` - Add ternary operation
- `forge_graph_mark_output(node)` - Mark node as output
- `forge_graph_mark_diff_input(node)` - Mark for differentiation

### Compilation
- `forge_config_create_default()` - Default config (stability cleaning)
- `forge_config_create_fast()` - All optimizations enabled
- `forge_config_set_instruction_set(config, set)` - SSE2_SCALAR or AVX2_PACKED
- `forge_compile(graph, config)` - Compile to kernel

### Execution
- `forge_buffer_create(graph, kernel)` - Create value buffer
- `forge_buffer_set_value(buffer, node, value)` - Set input
- `forge_execute(kernel, buffer)` - Run forward+backward
- `forge_buffer_get_value(buffer, node, &result)` - Get output
- `forge_buffer_get_gradient(buffer, node, &grad)` - Get gradient

### Error Handling
- All functions return `ForgeError` or NULL on failure
- `forge_get_last_error()` - Get detailed error message
- `forge_error_string(err)` - Get error description

## Thread Safety

- **Graph building**: NOT thread-safe (use one graph per thread)
- **Kernel execution**: Thread-safe (same kernel can run on multiple threads)
- **Buffer operations**: NOT thread-safe (use one buffer per thread)

## Supported Platforms

| Platform | Architecture | Library |
|----------|--------------|---------|
| Windows | x64 | `forge_capi.dll` |
| Linux | x64 | `libforge_capi.so` |
| macOS | x64 | `libforge_capi.dylib` |
| macOS | ARM64 | `libforge_capi.dylib` |
