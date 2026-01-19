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
- **Instruction Set Backends**: SSE2 scalar (default) and AVX2 packed (4-wide SIMD), with extensible backend interface
- **Branching Support**: Record-time conditional evaluation via `fbool` and `If()` for data-dependent control flow

### Pluggable Backend Architecture

Forge is designed to be **backend-agnostic** — the core compiler is decoupled from specific instruction sets, number types, and hardware. The AVX2 backend demonstrates this: it can be bundled at compile time (`FORGE_BUNDLE_AVX2=ON`) or loaded dynamically at runtime via `InstructionSetFactory::loadBackend()`. This architecture enables custom backends with their own register allocation strategies, machine code generation, and memory layouts. The compilation policy (`ICompilationPolicy`) controls whether intermediate values are stored to memory or kept in registers — enabling forward-optimized execution when gradients aren't needed, or storing values for backward forging when they are. See [backends/](backends/) for implementation details and a step-by-step guide to creating custom backends.

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
  <td>ForgeEngine compiles optimized graph to native machine code via forward forging and optional backward forging (for gradients) using pluggable instruction set backends</td>
</tr>
<tr>
  <td><a href="examples/"><b>4. Execution</b></a></td>
  <td>Execute the ForgedKernel repeatedly with varying inputs; retrieve computed values and gradients</td>
</tr>
</table>

*Extensibility: Custom graph transformations (1), optimization passes (2), instruction set backends with custom machine code and register management (3).*

## Example

```cpp
#include <graph/graph.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/interfaces/node_value_buffer.hpp>

using namespace forge;

int main() {
    // 1. Graph API — Define f(x) = x² + sin(x) using Direct API
    Graph graph;
    NodeId x = graph.addInput();
    graph.diff_inputs.push_back(x);                        // Mark x for gradient computation

    NodeId x_squared = graph.addNode({OpCode::Mul, x, x}); // x²
    NodeId sin_x = graph.addNode({OpCode::Sin, x});        // sin(x)
    NodeId result = graph.addNode({OpCode::Add, x_squared, sin_x});
    graph.markOutput(result);

    // 2. Graph Pre-processing + 3. Kernel Forging — ForgeEngine compiles graph
    ForgeEngine engine;
    auto kernel = engine.compile(graph);
    auto buffer = NodeValueBufferFactory::create(graph, *kernel);

    // 4. Execution — Run ForgedKernel repeatedly with different inputs
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

## License

FORGE is licensed under the Zlib License. See [LICENSE.md](LICENSE.md) for details.

## Related Projects

- [xad-forge](https://github.com/da-roth/xad-forge) — Forge JIT backend for [XAD](https://github.com/auto-differentiation/xad)
- [QuantLib-Risks-Cpp-Forge](https://github.com/da-roth/QuantLib-Risks-Cpp-Forge) — [QuantLib-Risks](https://github.com/auto-differentiation/QuantLib-Risks-Cpp) with Forge JIT integration

## Authors & Maintainers

- [da-roth](https://github.com/da-roth)

## Acknowledgments

- [AsmJit](https://github.com/asmjit/asmjit) — High-performance machine code generation
- [MathPresso](https://github.com/kobalicek/mathpresso) — Mathematical expression JIT compilation inspiration
- [AutoDiffSharp](https://github.com/naasking/AutoDiffSharp) — Automatic differentiation design influence
- [SLEEF](https://github.com/shibatch/sleef) — Vectorized math functions for SIMD operations
