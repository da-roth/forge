<img src="forgeGPT.png" alt="Forge Logo" width="170" align="left"/>

<br/>
<br/>

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FORGE — Forward & Reverse Gradient Engine

<br clear="left"/>
<br/>

FORGE is a high-performance JIT compilation library built on [AsmJit](https://asmjit.com) that generates optimized x86-64 machine code from mathematical expressions with automatic differentiation support.

- **Forward evaluation**: Fast re-evaluation of expressions with varying inputs
- **Reverse-mode AD**: Automatic gradient computation via reverse-mode differentiation
- **JIT compilation**: Both kernels compiled to optimized native machine code at runtime

Designed for high-throughput scenarios like Monte Carlo simulations, stress testing, and live risk calculations.

## Acknowledgments

- [AsmJit](https://github.com/asmjit/asmjit) - High-performance machine code generation
- [MathPresso](https://github.com/kobalicek/mathpresso) - Mathematical expression JIT compilation inspiration
- [AutoDiffSharp](https://github.com/naasking/AutoDiffSharp) - Automatic differentiation design influence
- [SLEEF](https://github.com/shibatch/sleef) - Vectorized math functions for SIMD operations

FORGE combines AsmJit's foundation for optimized x86-64 code generation with expression compilation patterns inspired by MathPresso and automatic differentiation techniques influenced by AutoDiffSharp.

## Key Features

- **JIT Compilation**: Native machine code generation eliminates interpretation overhead
- **Automatic Differentiation**: Full reverse-mode AD support for gradient computation
- **SIMD Optimization**: Leverages AVX2 and SSE2 instruction sets for vectorization
- **Graph Optimization**: CSE, algebraic simplification, constant folding, and stability improvements
- **Operator Overloading**: Natural C++ syntax for mathematical expressions
- **Rich Operations**: Arithmetic, transcendental functions, comparisons, and control flow

## Example Usage

```cpp
#include <types/fdouble.hpp>
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>
#include <iostream>
#include <cmath>

using namespace forge;

int main() {
    // Start recording the computation graph
    GraphRecorder recorder;
    recorder.start();

    // Create input variable: f(x) = x^2 + sin(x)
    fdouble x(0.0);
    x.markInputAndDiff();  // Mark for automatic differentiation

    // Build expression using natural C++ operators
    fdouble result = square(x) + sin(x);
    result.markOutput();

    // Stop recording and retrieve the graph
    recorder.stop();
    Graph graph = recorder.graph();

    // Compile to optimized machine code
    ForgeEngine compiler;
    auto kernel = compiler.compile(graph);

    // Create execution buffer
    auto buffer = NodeValueBufferFactory::create(graph, *kernel);

    // Evaluate at x = 2.0
    double x_val = 2.0;
    buffer->setValue(graph.diff_inputs[0], x_val);
    buffer->clearGradients();

    // Execute (automatically computes both forward and gradient)
    kernel->execute(*buffer);

    // Retrieve results
    double f_x = buffer->getValue(graph.outputs[0]);
    double df_dx = buffer->getGradient(graph.diff_inputs[0]);

    std::cout << "f(" << x_val << ") = " << f_x << std::endl;
    std::cout << "f'(" << x_val << ") = " << df_dx << std::endl;

    // Analytical gradient: f'(x) = 2x + cos(x)
    double analytical = 2 * x_val + std::cos(x_val);
    std::cout << "Analytical f'(" << x_val << ") = " << analytical << std::endl;

    return 0;
}
```

## Supported Operations

FORGE provides comprehensive mathematical operation support through the `fdouble`, `fbool`, and `fint` types:

### Arithmetic Operations
- **Basic**: `+`, `-`, `*`, `/`, unary `-`
- **Functions**: `abs()`, `square()`, `recip()`, `mod()`, `min()`, `max()`

### Mathematical Functions
- **Exponential**: `exp()`, `log()`, `sqrt()`, `pow()`
- **Trigonometric**: `sin()`, `cos()`, `tan()`

### Comparison Operations
- **Operators**: `<`, `<=`, `>`, `>=`, `==`, `!=` (return `fbool`)
- **Functions**: `cmpLT()`, `cmpLE()`, `cmpGT()`, `cmpGE()`, `cmpEQ()`, `cmpNE()`

### Control Flow
- **Conditional**: `If()` - ternary operator for branching
- **Boolean**: `&&`, `||`, `!` via `fbool` type
- **Integer**: Full arithmetic via `fint` type

### Data Types
- **`fdouble`**: Double-precision floating-point (primary type for AD)
- **`fbool`**: Boolean values with logical operations
- **`fint`**: Integer arithmetic with truncating division

## Build Instructions

### Prerequisites

- **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake 3.20+**
- Dependencies auto-fetched via CMake FetchContent:
  - AsmJit (JIT code generation)
  - nlohmann/json (graph serialization)
  - SLEEF (vectorized math functions)
  - GoogleTest (testing framework)

### Building Standalone

```bash
# Clone the repository
git clone https://github.com/da-roth/forge.git
cd forge

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . -j

# Run tests
ctest
```

### Building on Windows with PowerShell

```powershell
# Clone the repository
git clone https://github.com/da-roth/forge.git
cd forge

# Run the build test script
.\build-test.ps1
```

### CMake Integration as Submodule

```cmake
# Add forge as a subdirectory (e.g., if using as git submodule)
add_subdirectory(forge)

# Link to your target
target_link_libraries(your_target PRIVATE forge::forge)
```

### CMake Integration via FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    forge
    GIT_REPOSITORY https://github.com/da-roth/forge.git
    GIT_TAG main
)
FetchContent_MakeAvailable(forge)

target_link_libraries(your_target PRIVATE forge::forge)
```

## Compiler Configuration

FORGE provides extensive configuration options for optimization and debugging:

```cpp
// Default configuration (recommended for production)
ForgeEngine compiler(CompilerConfig::Default());

// No optimizations (useful for debugging)
ForgeEngine compiler(CompilerConfig::NoOptimization());

// Custom configuration
CompilerConfig config;
config.enableOptimizations = true;
config.enableCSE = true;                    // Common subexpression elimination
config.enableAlgebraicSimplification = true; // x*0→0, x*1→x, etc.
config.enableInactiveFolding = true;         // Constant folding
config.enableStabilityCleaning = true;       // 1/exp(x) → exp(-x)
config.printOptimizationStats = true;        // Show optimization metrics

ForgeEngine compiler(config);
```

## Performance Characteristics

- **JIT Compilation**: Optimized x86-64 assembly code generation
- **SIMD Vectorization**: AVX2 support for 4-wide double operations
- **Graph Optimization**: Multiple passes reduce node count 20-40% on typical graphs
- **Zero-overhead AD**: Gradients computed with minimal overhead via reverse-mode
- **Thread Safety**: Compiled kernels are thread-safe and reentrant

## Architecture Overview

FORGE uses a three-stage pipeline:

1. **Graph Recording**: Captures computation via operator overloading (`fdouble`, `fbool`, `fint`)
2. **Graph Optimization**: Applies CSE, algebraic simplification, constant folding, stability cleaning
3. **JIT Compilation**: Generates optimized forward + gradient x86-64 machine code via AsmJit

The resulting `StitchedKernel` can be executed repeatedly with different input values at native performance.

## Testing

FORGE includes comprehensive test suites:

```bash
# Run all tests
cd build
ctest

# Run specific test categories
ctest -R derivatives  # Gradient correctness tests
ctest -R benchmarks   # Performance benchmarks
```

Tests cover:
- Gradient correctness (verified against finite differences)
- Graph optimization correctness
- Multi-dimensional functions
- Edge cases and numerical stability

## Examples

The `examples/` directory contains working demonstrations:

- **basic_gradient.cpp**: Single-variable gradient computation with quadratic, trigonometric, and composite functions
- **multi_variable.cpp**: Multi-variable functions with partial derivatives and chain rule examples
- **performance_demo.cpp**: Performance comparison between optimized and unoptimized compilation with throughput measurements

Build and run examples:
```bash
cd build
./bin/basic_gradient
./bin/multi_variable
./bin/performance_demo
```

The `toolsTests/` directory contains comprehensive test suites for gradient correctness, benchmarking, and graph optimization verification

## Documentation

- **API Headers**: All public headers in `src/` and `tools/` include comprehensive Doxygen documentation
- **Graph Operations**: See `src/graph/graph.hpp` for OpCode definitions
- **Type System**: See `tools/types/fdouble.hpp`, `fbool.hpp`, `fint.hpp` for operator overloading API

## License

FORGE is licensed under the Zlib License. See [LICENSE.md](LICENSE.md) for details.

## Authors & Maintainers

- [da-roth](https://github.com/da-roth)