<img src="forgeGPT.png" alt="Forge Logo" width="170" align="left"/>

<br/>
<br/>

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FORGE — Forward & Reverse Gradient Engine

<br clear="left"/>
<br/>

FORGE is a high-performance JIT compilation library built on [AsmJit](https://asmjit.com) that takes computation graphs and produces two key outputs:
- A **forward kernel** for fast re-evaluation of the graph with varying inputs
- A **reverse kernel** using automatic differentiation to compute derivatives

Both kernels are compiled into optimized x86-64 machine code at runtime, enabling high-throughput scenarios like Monte Carlo simulations, stress testing, and live risk calculations.

## Acknowledgments

- [AsmJit](https://github.com/asmjit/asmjit) - High-performance machine code generation
- [MathPresso](https://github.com/kobalicek/mathpresso) - Mathematical expression JIT compilation
- [AutoDiffSharp](https://github.com/naasking/AutoDiffSharp) - Automatic differentiation inspiration

FORGE brings these technologies together: AsmJit provides the foundation for generating optimized x86-64 machine code, MathPresso's approach inspired the expression compilation pipeline, and AutoDiffSharp's design influenced the automatic differentiation implementation. This combination enables FORGE to deliver both high-performance forward evaluation and efficient reverse-mode gradient computation from a single computation graph.

## Key Features

- **JIT Compilation**: Native machine code generation eliminates interpretation overhead
- **Automatic Differentiation**: Full reverse-mode AD support for gradient computation
- **SIMD Optimization**: Leverages AVX2 and SSE2 instruction sets
- **Rich Operations**: Arithmetic, transcendental functions, comparisons, and control flow

## Example Usage

```cpp
#include <forge/forge.h>
#include <iostream>

int main() {
    using namespace forge;
    
    // Create graph for f(x) = x^2 + sin(x)
    core::ComputationGraph graph;
    
    auto x = graph.addInput();
    graph.diff_inputs.push_back(x);  // Mark for gradient computation
    
    // x^2
    core::Node squareNode;
    squareNode.op = core::OpCode::Square;
    squareNode.a = x;
    auto x2 = graph.addNode(squareNode);
    
    // sin(x)
    core::Node sinNode;
    sinNode.op = core::OpCode::Sin;
    sinNode.a = x;
    auto sinx = graph.addNode(sinNode);
    
    // x^2 + sin(x)
    core::Node addNode;
    addNode.op = core::OpCode::Add;
    addNode.a = x2;
    addNode.b = sinx;
    auto result = graph.addNode(addNode);
    
    graph.markOutput(result);
    
    // Compile forward and reverse kernels
    compiler::ForwardCompiler fwdCompiler;
    auto fwdKernel = fwdCompiler.compile(graph);
    
    compiler::ReverseGradientCompiler revCompiler;
    auto revKernel = revCompiler.compile(graph);
    
    // Test at x = 2.0
    double xval = 2.0;
    
    // Forward pass
    auto fwdBuffer = runtime::NodeBufferFactory::create(graph, *fwdKernel);
    fwdBuffer->setValue(x, xval);
    fwdKernel->execute(*fwdBuffer);
    double fx = fwdBuffer->getValue(result);
    
    // Reverse pass for gradient
    auto revBuffer = runtime::NodeBufferFactory::create(graph, *revKernel);
    revBuffer->setValue(x, xval);
    revBuffer->setGradient(result, 1.0);  // Seed gradient
    revKernel->execute(*revBuffer);
    double dfdx = revBuffer->getGradient(x);
    
    std::cout << "f(" << xval << ") = " << fx << std::endl;
    std::cout << "f'(" << xval << ") = " << dfdx << std::endl;
    
    // Analytical gradient: f'(x) = 2x + cos(x)
    double analytical = 2*xval + cos(xval);
    std::cout << "Analytical f'(" << xval << ") = " << analytical << std::endl;
    
    return 0;
}
```

## Supported Operations

FORGE provides comprehensive mathematical operation support:

### Arithmetic Operations
- Basic: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`
- Advanced: `Square`, `Recip`, `Mod`, `Min`, `Max`

### Mathematical Functions  
- Exponential: `Exp`, `Log`, `Sqrt`, `Pow`
- Trigonometric: `Sin`, `Cos`, `Tan`
- Comparison: `CmpLT`, `CmpLE`, `CmpGT`, `CmpGE`, `CmpEQ`, `CmpNE`

### Control Flow
- Conditional: `If` (ternary operator)
- Boolean: `BoolAnd`, `BoolOr`, `BoolNot`
- Integer: Full integer arithmetic with separate opcodes

### Data Types
- `double`: Double-precision floating-point (primary)
- `bool`: Boolean values with logical operations
- `int`: Integer arithmetic with truncating division

## Build Instructions

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.16+
- AsmJit (automatically fetched via FetchContent)

### Building

```bash
# Clone the repository
git clone https://github.com/your-org/forge.git
cd forge

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . -j

# Optional: Install
cmake --install .
```

### CMake Integration

```cmake
# Use FetchContent
include(FetchContent)
FetchContent_Declare(
    forge
    GIT_REPOSITORY https://github.com/your-org/forge.git
    GIT_TAG main
)
FetchContent_MakeAvailable(forge)

# Link to your target
target_link_libraries(your_target PRIVATE forge::forge)
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- **Basic Math**: Simple arithmetic and function evaluation
- **Gradients**: Automatic differentiation examples
- **Performance**: Benchmarking and optimization demonstrations
- **Integration**: Using Forge with other libraries

## Documentation

- [API Documentation](docs/api.md) - Complete API reference
- [Architecture Guide](docs/architecture.md) - Internal design details
- [Performance Guide](docs/performance.md) - Optimization tips
- [Examples](examples/) - Usage examples and tutorials

## License

FORGE is licensed under the Zlib License. See [LICENSE.md](LICENSE.md) for details.

## Authors & Maintainers

- [da-roth](https://github.com/da-roth)