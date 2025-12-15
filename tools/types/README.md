# Forge Types — Operator Overloading API

This directory provides convenience types (`fdouble`, `fbool`, `fint`) that enable natural C++ syntax for building computation graphs via operator overloading.

## Quick Start

```cpp
#include <types/fdouble.hpp>
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/node_value_buffers/node_value_buffer.hpp>

using namespace forge;

int main() {
    // 1. Record the computation graph
    GraphRecorder recorder;
    recorder.start();

    fdouble x(0.0);
    x.markInputAndDiff();  // Mark for automatic differentiation

    fdouble result = square(x) + sin(x);
    result.markOutput();

    recorder.stop();
    Graph graph = recorder.graph();

    // 2. Compile to machine code
    ForgeEngine compiler;
    auto kernel = compiler.compile(graph);

    // 3. Execute with different inputs
    auto buffer = NodeValueBufferFactory::create(graph, *kernel);

    buffer->setValue(graph.diff_inputs[0], 2.0);
    buffer->clearGradients();
    kernel->execute(*buffer);

    double f_x = buffer->getValue(graph.outputs[0]);      // f(2.0)
    double df_dx = buffer->getGradient(graph.diff_inputs[0]); // f'(2.0)

    return 0;
}
```

## Supported Operations

### fdouble — Double-Precision Floating-Point

The primary type for automatic differentiation.

**Arithmetic:**
| Operation | Syntax |
|-----------|--------|
| Addition | `a + b` |
| Subtraction | `a - b` |
| Multiplication | `a * b` |
| Division | `a / b` |
| Negation | `-a` |

**Functions:**
| Function | Description |
|----------|-------------|
| `abs(x)` | Absolute value |
| `square(x)` | x² (optimized) |
| `recip(x)` | 1/x |
| `mod(x, y)` | Modulo |
| `min(x, y)` | Minimum |
| `max(x, y)` | Maximum |
| `exp(x)` | Exponential |
| `log(x)` | Natural logarithm |
| `sqrt(x)` | Square root |
| `pow(x, y)` | Power |
| `sin(x)` | Sine |
| `cos(x)` | Cosine |
| `tan(x)` | Tangent |

**Comparisons** (return `fbool`):
| Operator | Function |
|----------|----------|
| `a < b` | `cmpLT(a, b)` |
| `a <= b` | `cmpLE(a, b)` |
| `a > b` | `cmpGT(a, b)` |
| `a >= b` | `cmpGE(a, b)` |
| `a == b` | `cmpEQ(a, b)` |
| `a != b` | `cmpNE(a, b)` |

### fbool — Boolean Type

Used for conditions and control flow.

**Operators:**
- Logical AND: `a && b`
- Logical OR: `a || b`
- Logical NOT: `!a`

**Conditional selection:**
```cpp
fbool condition = x > 0.0;
fdouble result = If(condition, trueValue, falseValue);
```

The `If()` function records both branches in the graph. At runtime, the correct path is selected based on input values — enabling JIT re-evaluation even when inputs take different branches.

### fint — Integer Type

Integer arithmetic with truncating division.

**Operators:** `+`, `-`, `*`, `/`, `%`, unary `-`

**Comparisons:** Same as `fdouble`, return `fbool`

**Conditional:** `If(condition, trueInt, falseInt)`

## Input/Output Marking

```cpp
fdouble x(0.0);
x.markInputAndDiff();  // Input variable, compute gradients w.r.t. this
// or
x.markInput();         // Input variable, no gradient needed

result.markOutput();   // Mark as output node
```

## See Also

- [examples/](../../examples/) — Working demonstrations
- [src/graph/graph.hpp](../../src/graph/graph.hpp) — Direct Graph API for programmatic graph construction
