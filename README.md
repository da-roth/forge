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

```
   1. Input          2. Graph Pre-processing    3. Kernel Forging        4. Evaluation

┌────────────┐      ┌────────────────────────────────────┐      ┌──────────────┐
│ Graph via: │      │ ┌────────────┐   ┌──────────────┐ │      │Forged Kernel:│
│- Direct API│      │ │- CSE       │   │- Forward     │ │      │- Execute     │
│- Overload  │─────▶│ │- Const Fold│──▶│- Backward    │ │─────▶│- Values      │
│+ Custom    │      │ │- Simplify  │   │  (optional)  │ │      │- Gradients   │
└────────────┘      │ │- Stability │   │+ Custom ISA  │ │      └──────────────┘
                    │ │+ Custom    │   └──────────────┘ │
                    │ └────────────┘                    │
                    │          ForgeEngine              │
                    └────────────────────────────────────┘
```

| Phase | What happens | Extensibility | Reference |
|-------|--------------|---------------|-----------|
| **1. Input** | Build computation graph from expressions | Custom graph transformations | [Graph](src/graph/) |
| **2. Graph Pre-processing** | Optimize: CSE, constant folding, simplification, stability | Custom optimization passes | [Optimizations](src/graph/optimizations/) |
| **3. Kernel Forging** | Generate forward + (optional) backward machine code | Custom instruction set backends | [Backends](backends/) |
| **4. Evaluation** | Execute kernel repeatedly with different inputs | — | [Examples](examples/) |

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
