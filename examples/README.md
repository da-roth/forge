# Forge Examples

Working demonstrations of Forge's JIT compilation and automatic differentiation.

## Building

From the forge root directory:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

Executables are placed in `build/bin/`.

## Examples

### basic_gradient.cpp

Single-variable gradient computation:
- `f(x) = x² + 3x + 2` with derivative `f'(x) = 2x + 3`
- `f(x) = sin(x) + cos(x)` with derivative `f'(x) = cos(x) - sin(x)`
- `f(x) = exp(x) * sin(x)` with derivative via product rule

```bash
./build/bin/basic_gradient
```

### multi_variable.cpp

Multi-variable gradient computation with partial derivatives:
- `f(x,y) = x² + y² + xy` — partial derivatives ∂f/∂x, ∂f/∂y
- `f(x,y,z) = exp(x) + sin(y) + z²` — three-variable function
- `f(x,y) = sin(xy) + exp(x-y)` — chain rule demonstration

```bash
./build/bin/multi_variable
```

### performance_demo.cpp

Performance benchmarking:
- Compares optimized vs unoptimized compilation
- Measures execution throughput (100,000 evaluations)
- Shows memory efficiency metrics

```bash
./build/bin/performance_demo
```

## Writing Your Own

```cpp
#include <native/fdouble.hpp>
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/interfaces/node_value_buffer.hpp>

using namespace forge;

int main() {
    // 1. Record computation graph
    GraphRecorder recorder;
    recorder.start();

    fdouble x(0.0);
    x.markInputAndDiff();

    fdouble y = sin(x) + square(x);
    y.markOutput();

    recorder.stop();
    Graph graph = recorder.graph();

    // 2. Compile to native code
    ForgeEngine compiler;
    auto kernel = compiler.compile(graph);

    // 3. Execute
    auto buffer = NodeValueBufferFactory::create(graph, *kernel);
    buffer->setValue(graph.diff_inputs[0], 2.0);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double result = buffer->getValue(graph.outputs[0]);
    double gradient = buffer->getGradient(graph.diff_inputs[0]);
}
```

## See Also

- [api/native/](../api/native/) — `fdouble`, `fbool`, `fint` operator overloading API
- [src/graph/graph.hpp](../src/graph/graph.hpp) — Direct Graph API
