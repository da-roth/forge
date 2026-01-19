# Graph API

Forge computation graphs represent mathematical expressions as directed acyclic graphs (DAGs) of operations. This directory contains the core graph data structures and recording mechanisms.

## Creating Graphs

### Operator Overloading (Recommended)

Use `fdouble`, `fbool`, and `fint` for natural mathematical syntax:

```cpp
#include <native/fdouble.hpp>
#include <graph/graph_recorder.hpp>

GraphRecorder recorder;
recorder.start();

fdouble x(0.0);
x.markInputAndDiff();  // Mark as input with gradient tracking

fdouble y = sin(x) + square(x);
y.markOutput();

recorder.stop();
Graph graph = recorder.graph();
```

See [api/native/](../../api/native/) for the full operator overloading API.

### Direct Graph API

Build graphs programmatically for maximum control:

```cpp
#include <graph/graph.hpp>

Graph graph;
NodeId x = graph.addInput();
graph.diff_inputs.push_back(x);  // Mark for gradient computation

NodeId x_squared = graph.addNode({OpCode::Mul, x, x});
NodeId sin_x = graph.addNode({OpCode::Sin, x});
NodeId result = graph.addNode({OpCode::Add, x_squared, sin_x});
graph.markOutput(result);
```

### External Transformation

Convert tapes from other AD libraries. See [xad-forge](https://github.com/da-roth/xad-forge) for an example transforming XAD tapes to Forge graphs.

## Files

| File | Description |
|------|-------------|
| `graph.hpp` | Core `Graph` and `Node` structures, `OpCode` definitions |
| `graph_recorder.hpp` | Thread-local recording context for operator overloading |
| `graph_optimizer.hpp` | Graph optimization orchestrator |
| `optimizations/` | Individual optimization passes |

## Graph Structure

```cpp
struct Graph {
    std::vector<Node> nodes;       // All operations
    std::vector<double> constPool; // Constant values
    std::vector<NodeId> outputs;   // Output nodes
    std::vector<NodeId> diff_inputs; // Inputs marked for differentiation
};

struct Node {
    OpCode op;      // Operation type
    NodeId a, b, c; // Operand references
    double imm;     // Immediate value (for constants)
    bool isActive;  // True if depends on inputs
};
```

## Supported Operations

| Category | Operations |
|----------|------------|
| Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`, `Square`, `Recip`, `Mod` |
| Transcendental | `Exp`, `Log`, `Sqrt`, `Pow`, `Sin`, `Cos`, `Tan` |
| Comparison | `CmpLT`, `CmpLE`, `CmpGT`, `CmpGE`, `CmpEQ`, `CmpNE` |
| Control flow | `If`, `Min`, `Max` |
| Boolean | `BoolAnd`, `BoolOr`, `BoolNot`, `BoolEq`, `BoolNe` |
| Integer | `IntAdd`, `IntSub`, `IntMul`, `IntDiv`, `IntMod`, `IntNeg`, `IntIf` |
| Indexing | `ArrayIndex` |

## See Also

- [optimizations/](optimizations/) — Graph optimization passes
- [api/native/](../../api/native/) — `fdouble`, `fbool`, `fint` API
- [examples/](../../examples/) — Working demonstrations
