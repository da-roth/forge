# Forge Types — Operator Overloading API

This directory provides convenience types (`fdouble`, `fbool`, `fint`) that enable natural C++ syntax for building computation graphs via operator overloading.

## Quick Start

```cpp
#include <native/fdouble.hpp>
#include <graph/graph_recorder.hpp>
#include <compiler/forge_engine.hpp>
#include <compiler/interfaces/node_value_buffer.hpp>

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

**Conditional selection with `If()`:**

Using native C++ `if` statements only records the branch taken during graph construction:

```cpp
fdouble x(5.0);  // x > 0 during recording
x.markInputAndDiff();

fdouble result;
if (x.value() > 0)      // Native C++ if - evaluated at recording time
    result = fdouble(1.0);
else
    result = fdouble(2.0);

result.markOutput();
// The kernel will ALWAYS return 1.0, even when re-evaluated with x = -10
```

With `fbool` and `If()`, both branches are recorded in the graph:

```cpp
fdouble x(5.0);
x.markInputAndDiff();

fbool condition = x > 0.0;
fdouble result = If(condition, fdouble(1.0), fdouble(2.0));

result.markOutput();
// The kernel correctly returns 1.0 for x > 0, and 2.0 for x <= 0
```

The compiled kernel can now be re-evaluated with any input value and will select the correct branch at runtime.

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
