<div align="center">
  <img src="forgeGPT.png" alt="Forge Logo" width="170"/>
  <h3>FORGE — Forward & Reverse Gradient Engine</h3>
  <p>JIT compiler for mathematical expressions with automatic differentiation</p>
</div>

<br/>

FORGE compiles computation graphs into optimized x86-64 machine code. Record once, compile once, evaluate many times at native speed — ideal for Monte Carlo methods, scenario analysis, and model calibration where the same computation runs repeatedly with different inputs.

## Quick Example

```cpp
#include <forge.hpp>

using namespace forge;

int main() {
    // Record computation: f(x) = x² + sin(x)
    GraphRecorder recorder;
    recorder.start();

    fdouble x(0.0);
    x.markInputAndDiff();
    fdouble result = square(x) + sin(x);
    result.markOutput();

    recorder.stop();
    Graph graph = recorder.graph();

    // Compile to machine code
    ForgeEngine compiler;
    auto kernel = compiler.compile(graph);
    auto buffer = NodeValueBufferFactory::create(graph, *kernel);

    // Evaluate with different inputs
    buffer->setValue(graph.diff_inputs[0], 2.0);
    kernel->execute(*buffer);

    double f_x = buffer->getValue(graph.outputs[0]);           // f(2.0)
    double df_dx = buffer->getGradient(graph.diff_inputs[0]);  // f'(2.0)
}
```

See [examples/](examples/) for more and [api/native/](api/native/) for the full operator overloading API.

## Getting Started

```bash
git clone https://github.com/da-roth/forge.git
cd forge && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
ctest  # Run tests
```

**Requirements:** C++17, CMake 3.20+. Dependencies (AsmJit, SLEEF, nlohmann/json, GoogleTest) are fetched automatically.

**CMake integration:**
```cmake
add_subdirectory(forge)
target_link_libraries(your_target PRIVATE forge::forge)
```

## Key Features

- **JIT Compilation** — Native x86-64 machine code via [AsmJit](https://github.com/asmjit/asmjit)
- **Reverse-mode AD** — Automatic gradient computation
- **Graph Optimizations** — CSE, constant folding, algebraic simplification
- **Modular Backends** — SSE2 (default) + AVX2, extensible for custom SIMD

## Backends

Forge separates the compiler from platform-specific code generation through interfaces (`IInstructionSet`, `IRegisterAllocator`, `INodeValueBuffer`). SSE2 is always available; AVX2 is bundled by default.

```cpp
// Use AVX2 (bundled by default)
CompilerConfig config;
config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;
ForgeEngine compiler(config);
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `FORGE_BUNDLE_AVX2` | ON | Include AVX2 in the library |
| `FORGE_BUILD_AVX2_BACKEND` | OFF | Build AVX2 as loadable `libforge_avx2.so` |

### Runtime Loading

Backends can be loaded at runtime for optional acceleration or custom SIMD:

```cpp
InstructionSetFactory::loadBackend("./libforge_avx2.so");
auto kernel = compiler.compile(graph);  // Now uses loaded backend
```

See [backends/double/avx2/](backends/double/avx2/) for implementation examples and [tests/test_backend_loading.cpp](tests/test_backend_loading.cpp) for usage.

## Documentation

| Resource | Description |
|----------|-------------|
| [api/native/](api/native/) | `fdouble`, `fbool`, `fint` operator overloading |
| [src/graph/graph.hpp](src/graph/graph.hpp) | Direct Graph API and `OpCode` definitions |
| [examples/](examples/) | Working demonstrations |
| [backends/double/avx2/](backends/double/avx2/) | Backend implementation reference |

## License

Zlib License. See [LICENSE.md](LICENSE.md).

## Acknowledgments

- [AsmJit](https://github.com/asmjit/asmjit) — Machine code generation
- [SLEEF](https://github.com/shibatch/sleef) — Vectorized math functions
