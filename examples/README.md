# FORGE Examples

This directory contains example programs demonstrating the use of FORGE for JIT compilation and automatic differentiation.

## Quick Start

The easiest way to build and run all examples:

**Windows (PowerShell):**
```powershell
.\build-run-examples.ps1
```

**Linux/Mac (Bash):**
```bash
./build-run-examples.sh
```

This will build and run all three examples with nicely formatted output.

## Building the Examples

### Platform-Specific Scripts

**Windows (PowerShell):**
```powershell
.\build-run-examples.ps1          # Build and run all examples
.\build-run-examples.ps1 -Clean   # Clean build from scratch
.\build-run-examples.ps1 -Debug   # Build in Debug mode
```

**Linux/Mac (Bash):**
```bash
./build-run-examples.sh           # Build and run all examples
./build-run-examples.sh --clean   # Clean build from scratch
./build-run-examples.sh --debug   # Build in Debug mode
```

### Manual Build (All Platforms)

From the forge root directory:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

The example executables will be in `build/bin/Release/` (Windows) or `build/bin/` (Linux/Mac).

## Available Examples

### 1. basic_gradient.cpp
Demonstrates single-variable gradient computation:
- **Example 1**: Quadratic function f(x) = x² + 3x + 2
- **Example 2**: Trigonometric function f(x) = sin(x) + cos(x)
- **Example 3**: Composite function f(x) = exp(x) * sin(x)

Shows how FORGE automatically computes derivatives using reverse-mode automatic differentiation.

```bash
# Run the example
./build/bin/Release/basic_gradient
```

### 2. multi_variable.cpp
Multi-variable gradient computation with partial derivatives:
- **Example 1**: Two-variable function f(x,y) = x² + y² + xy
- **Example 2**: Three-variable with transcendentals f(x,y,z) = exp(x) + sin(y) + z²
- **Example 3**: Chain rule demonstration f(x,y) = sin(xy) + exp(x-y)

Demonstrates computing gradients for functions with multiple inputs simultaneously.

```bash
# Run the example
./build/bin/Release/multi_variable
```

### 3. performance_demo.cpp
Performance benchmarking and optimization comparison:
- Builds a complex mathematical function
- Compares optimized vs unoptimized compilation
- Measures execution throughput (1 million evaluations)
- Shows memory efficiency metrics

Demonstrates the performance benefits of FORGE's graph optimization passes.

```bash
# Run the example
./build/bin/Release/performance_demo
```

## Writing Your Own Examples

To create a new example using FORGE:

### 1. Include the necessary headers

```cpp
#include <native/fdouble.hpp>
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/interfaces/node_value_buffer.hpp>
```

### 2. Record your computation graph

```cpp
using namespace forge;

GraphRecorder recorder;
recorder.start();

fdouble x(0.0);
x.markInputAndDiff();  // Mark as input with gradient

fdouble y = sin(x) + square(x);
y.markOutput();

recorder.stop();
Graph graph = recorder.graph();
```

### 3. Compile and execute

```cpp
// Compile to native code
ForgeEngine compiler;
auto kernel = compiler.compile(graph);

// Create execution buffer
auto buffer = NodeValueBufferFactory::create(graph, *kernel);

// Set input value
buffer->setValue(graph.diff_inputs[0], 2.0);
buffer->clearGradients();

// Execute (automatically computes both function and gradient)
kernel->execute(*buffer);

// Get results
double result = buffer->getValue(graph.outputs[0]);
double gradient = buffer->getGradient(graph.diff_inputs[0]);

std::cout << "f(2.0) = " << result << "\n";
std::cout << "f'(2.0) = " << gradient << "\n";
```

## Key Concepts

- **GraphRecorder**: Captures mathematical expressions as computation graphs
- **fdouble/fbool/fint**: Operator-overloaded types that build the graph during recording
- **ForgeEngine**: JIT compiler that generates optimized x86-64 machine code
- **StitchedKernel**: Compiled executable that can be run repeatedly with different inputs
- **NodeValueBuffer**: Memory buffer for inputs, outputs, and gradients
- **Automatic Differentiation**: Gradients computed automatically via reverse-mode AD

## Compiler Configuration

```cpp
// Default configuration (recommended)
ForgeEngine compiler;

// With optimizations
CompilerConfig config;
config.enableOptimizations = true;
config.enableCSE = true;                    // Common subexpression elimination
config.enableAlgebraicSimplification = true; // Simplify x*0, x*1, etc.
config.enableInactiveFolding = true;         // Constant folding
ForgeEngine compiler(config);

// No optimizations (debugging)
CompilerConfig config;
config.enableOptimizations = false;
ForgeEngine compiler(config);
```

## Performance Tips

1. **Reuse compiled kernels**: Compile once, execute many times with different inputs
2. **Enable optimizations**: Use default config for production (20-40% node reduction)
3. **Batch operations**: Process multiple data points sequentially for cache efficiency
4. **Use AVX2**: FORGE automatically uses SIMD when available for 4-wide double operations

## Troubleshooting

### Build Issues

If examples fail to build:
- Ensure C++17 compiler support (GCC 7+, Clang 5+, MSVC 2017+)
- Check CMake version is 3.20 or higher
- On Windows, ensure `_USE_MATH_DEFINES` is defined before `<cmath>` for M_PI

### Runtime Issues

- **CPU support**: Check CPU supports SSE2 (required) or AVX2 (optional)
- **Node indices**: Access inputs via `graph.diff_inputs[i]` and outputs via `graph.outputs[i]`
- **Gradients**: Call `buffer->clearGradients()` before each execution
- **Memory**: Gradients are computed automatically, no manual seeding required

## Learn More

- See `../README.md` for full FORGE documentation
- Check `../toolsTests/` for comprehensive test suites
- Review the source code in `../src/` and `../tools/` for API details
