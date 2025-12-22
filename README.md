<div align="center">
  <img src="forgeGPT.png" alt="Forge Logo" width="170"/>
  <h3>FORGE — Forward & Reverse Gradient Engine</h3>
</div>

<br/>

FORGE is a JIT compilation engine for mathematical expressions with automatic differentiation. It follows a **record-once, compile-once, evaluate-many** paradigm: the computation graph is recorded once, compiled to optimized x86-64 machine code, and the resulting kernel can be re-evaluated with different inputs at native speed.

This approach is beneficial when the same computation must be repeated many times with varying inputs — such as Monte Carlo methods, scenario evaluation, or model calibration — where the JIT compilation overhead is amortized across evaluations.

## When to Use Forge

Forge is designed for **repeated evaluation** scenarios where the computation structure remains constant but inputs vary:

- **Monte Carlo methods**: Pricing, XVA, or any path-dependent calculation
- **Scenario evaluation**: Stress testing, what-if analysis, or parameter sweeps
- **Greeks and sensitivities**: Fast gradient computation across market moves
- **Model calibration**: Repeated function and gradient evaluation during parameter fitting

**Trade-off**: Forge incurs an upfront compilation cost. For single evaluations, traditional tape-based AD is faster. The break-even depends on graph complexity, but typically occurs after 10–50 repeated evaluations.

**Important**: The recorded computation graph must have the same structure for all inputs. For functions with branches, see [api/native/](api/native/) for how to record both paths using `fbool` and `If()`.

## Key Features

- **JIT Compilation**: Native x86-64 machine code eliminates interpretation overhead
- **Reverse-mode AD**: Automatic gradient computation with full reverse-mode differentiation
- **SIMD Vectorization**: AVX2 and SSE2 support for parallel evaluation
- **Graph Optimization**: CSE, algebraic simplification, constant folding, stability cleaning

## Integration

Forge provides two ways to build computation graphs:

### Option 1: Direct Graph API (For framework integration)

Build graphs programmatically — useful for transforming existing computation graphs (e.g., from other AD frameworks) into Forge format:

```cpp
Graph graph;
NodeId x = graph.addInput();
NodeId x_sq = graph.addNode({OpCode::Square, 0, x});
NodeId sin_x = graph.addNode({OpCode::Sin, 0, x});
NodeId result = graph.addNode({OpCode::Add, 0, x_sq, sin_x});
graph.markOutput(result);
graph.diff_inputs.push_back(x);

ForgeEngine compiler;
auto kernel = compiler.compile(graph);
```

See [src/graph/graph.hpp](src/graph/graph.hpp) for the full Graph API and available `OpCode` values.

### Option 2: Operator Overloading

Use `fdouble`, `fbool`, `fint` types for natural C++ syntax:

```cpp
GraphRecorder recorder;
recorder.start();

fdouble x(0.0);
x.markInputAndDiff();
fdouble result = square(x) + sin(x);
result.markOutput();

recorder.stop();
Graph graph = recorder.graph();

ForgeEngine compiler;
auto kernel = compiler.compile(graph);
```

See [api/native/](api/native/) for supported operations and [examples/](examples/) for complete examples.

## Architecture Overview

FORGE uses a three-stage pipeline:

1. **Graph Construction**: Build computation graph via operator overloading (`fdouble`, `fbool`, `fint`) or direct API (`Graph::addNode`)
2. **Graph Optimization**: Applies CSE, algebraic simplification, constant folding, stability cleaning
3. **JIT Compilation**: Generates optimized forward + gradient x86-64 machine code via AsmJit

The resulting kernel can be executed repeatedly with different input values at native performance.

## Build Instructions

### Prerequisites

- **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake 3.20+**
- Dependencies auto-fetched via CMake FetchContent:
  - AsmJit (JIT code generation)
  - nlohmann/json (graph serialization)
  - SLEEF (vectorized math functions)
  - GoogleTest (testing framework)

### Building

```bash
git clone https://github.com/da-roth/forge.git
cd forge
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
ctest  # Run tests
```

### CMake Integration

```cmake
# Option 1: As subdirectory / git submodule
add_subdirectory(forge)
target_link_libraries(your_target PRIVATE forge::forge)

# Option 2: Via FetchContent
include(FetchContent)
FetchContent_Declare(forge
    GIT_REPOSITORY https://github.com/da-roth/forge.git
    GIT_TAG main)
FetchContent_MakeAvailable(forge)
target_link_libraries(your_target PRIVATE forge::forge)
```

## Examples & Testing

The `examples/` directory contains working demonstrations:

| Example | Description |
|---------|-------------|
| `basic_gradient.cpp` | Single-variable gradient computation |
| `multi_variable.cpp` | Multi-variable functions with partial derivatives |
| `performance_demo.cpp` | Performance comparison: optimized vs. unoptimized |

```bash
cd build
./bin/basic_gradient
./bin/multi_variable
./bin/performance_demo
ctest                    # Run all tests
ctest -R derivatives     # Gradient correctness tests
```

## Compiler Configuration

FORGE provides configuration options for optimization and debugging:

```cpp
// Default: only stability cleaning (1/exp(x) → exp(-x))
ForgeEngine compiler(CompilerConfig::Default());

// Fast: all optimizations enabled (higher compilation time, faster evaluation)
ForgeEngine compiler(CompilerConfig::Fast());

// No optimizations (useful for debugging)
ForgeEngine compiler(CompilerConfig::NoOptimization());
```

For fine-grained control:

```cpp
CompilerConfig config;
config.enableCSE = true;                     // Common subexpression elimination
config.enableAlgebraicSimplification = true; // x*0→0, x*1→x, etc.
config.enableInactiveFolding = true;         // Constant folding
config.enableStabilityCleaning = true;       // Numerical stability fixes
ForgeEngine compiler(config);
```

## Documentation

- **Operator Overloading API**: See [api/native/](api/native/) for `fdouble`, `fbool`, `fint` documentation
- **Graph API**: See [src/graph/graph.hpp](src/graph/graph.hpp) for `OpCode` definitions and direct graph construction
- **API Headers**: All public headers include Doxygen documentation

## License

FORGE is licensed under the Zlib License. See [LICENSE.md](LICENSE.md) for details.

## Authors & Maintainers

- [da-roth](https://github.com/da-roth)

## Acknowledgments

- [AsmJit](https://github.com/asmjit/asmjit) - High-performance machine code generation
- [MathPresso](https://github.com/kobalicek/mathpresso) - Mathematical expression JIT compilation inspiration
- [AutoDiffSharp](https://github.com/naasking/AutoDiffSharp) - Automatic differentiation design influence
- [SLEEF](https://github.com/shibatch/sleef) - Vectorized math functions for SIMD operations