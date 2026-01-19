<div align="center">
  <img src="forgeGPT.png" alt="Forge Logo" width="170"/>
  <h3>FORGE — Forward & Reverse Gradient Engine</h3>
  <p>High-performance JIT compilation for mathematical expressions with automatic differentiation</p>
</div>

<br/>

Forge compiles mathematical expressions to optimized x86-64 machine code with automatic gradient computation. It follows a **record-once, compile-once, evaluate-many** paradigm designed for workloads where the same computation is repeated with varying inputs.

## Key Features

- **JIT Compilation**: Generates native x86-64 machine code via [AsmJit](https://github.com/asmjit/asmjit)
- **Reverse-mode AD**: Automatic gradient computation for all recorded operations
- **Graph Optimizations**: Common subexpression elimination, constant folding, algebraic simplification
- **SIMD Backends**: SSE2 scalar (default) and AVX2 packed (4-wide), with extensible backend interface
- **Branching Support**: Record-time conditional evaluation via `fbool` and `If()` for data-dependent control flow

## When to Use Forge

Forge is designed for **repeated evaluation** scenarios:

- **Monte Carlo methods**: Pricing, XVA, path-dependent calculations
- **Scenario analysis**: Stress testing, what-if analysis, parameter sweeps
- **Sensitivities**: Fast gradient computation across input variations
- **Model calibration**: Repeated function/gradient evaluation during optimization

**Trade-off**: Forge incurs upfront compilation cost. For single evaluations, tape-based AD is faster. Break-even typically occurs after 10–50 evaluations depending on graph complexity.

## Overview

<table>
<tr>
  <th>Phase</th>
  <th>Description</th>
</tr>
<tr>
  <td><a href="src/graph/"><b>1. Graph API</b></a></td>
  <td>Define computation graph using Direct API, operator overloading (<code>fdouble</code>), or transform from external sources (e.g., <a href="https://github.com/da-roth/xad-forge">xad-forge</a>)</td>
</tr>
<tr>
  <td><a href="src/graph/optimizations/"><b>2. Graph Pre-processing</b></a></td>
  <td>ForgeEngine applies graph optimizations: common subexpression elimination, constant folding, algebraic simplification, and stability cleaning</td>
</tr>
<tr>
  <td><a href="backends/"><b>3. Kernel Forging</b></a></td>
  <td>ForgeEngine compiles optimized graph to native machine code for forward pass and optional backward pass (gradients) using pluggable instruction set backends</td>
</tr>
<tr>
  <td><a href="examples/"><b>4. Execution</b></a></td>
  <td>Execute the compiled kernel repeatedly with varying inputs; retrieve computed values and gradients</td>
</tr>
</table>

*Extensibility: Phases 1-3 support custom extensions — graph transformations, optimization passes, and instruction set backends respectively.*

## Example

```cpp
#include <graph/graph.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/interfaces/node_value_buffer.hpp>

using namespace forge;

int main() {
    // 1. Input Graph — Build f(x) = x² + sin(x) using Direct API
    Graph graph;
    NodeId x = graph.addInput();
    graph.diff_inputs.push_back(x);                        // Mark x for gradient computation

    NodeId x_squared = graph.addNode({OpCode::Mul, x, x}); // x²
    NodeId sin_x = graph.addNode({OpCode::Sin, x});        // sin(x)
    NodeId result = graph.addNode({OpCode::Add, x_squared, sin_x});
    graph.markOutput(result);

    // 2. Forging — Compile graph (includes optimization + code generation)
    ForgeEngine engine;
    auto kernel = engine.compile(graph);
    auto buffer = NodeValueBufferFactory::create(graph, *kernel);

    // 3. Execution — Evaluate repeatedly with different inputs
    buffer->setValue(x, 2.0);
    kernel->execute(*buffer);

    double f_x = buffer->getValue(result);    // f(2.0)
    double df_dx = buffer->getGradient(x);    // f'(2.0)
}
```

## Getting Started

```bash
git clone https://github.com/da-roth/forge.git
cd forge && mkdir build && cd build
cmake .. && cmake --build .
```

**CMake integration:**
```cmake
add_subdirectory(forge)
target_link_libraries(your_target PRIVATE forge::forge)
```

Requires C++17 and CMake 3.20+. All dependencies are fetched automatically.

## SIMD Backends

Forge supports multiple instruction set backends:

```cpp
CompilerConfig config;
config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;
ForgeEngine compiler(config);
```

| Option | Default | Description |
|--------|---------|-------------|
| `FORGE_BUNDLE_AVX2` | ON | Bundle AVX2 backend into library |
| `FORGE_BUILD_AVX2_BACKEND` | OFF | Build AVX2 as loadable shared library |

Backends can also be loaded at runtime:
```cpp
InstructionSetFactory::loadBackend("./libforge_avx2.so");
```

## License

Zlib License. See [LICENSE.md](LICENSE.md).

## Related Projects

- [xad-forge](https://github.com/da-roth/xad-forge) — Forge JIT backend for [XAD](https://github.com/auto-differentiation/xad)

## Acknowledgments

- [AsmJit](https://github.com/asmjit/asmjit) — Machine code generation
- [MathPresso](https://github.com/kobalicek/mathpresso) — JIT expression compilation inspiration
- [SLEEF](https://github.com/shibatch/sleef) — Vectorized transcendental functions
