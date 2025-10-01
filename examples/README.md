# Forge Examples

This directory contains examples demonstrating various features of the Forge JIT compilation library.

## Examples

### graph_from_json

Demonstrates loading a computation graph from JSON format and executing it with both forward and reverse (gradient) computation.

**Features:**
- Loading graphs from JSON files
- Forward execution with multiple inputs
- Reverse-mode automatic differentiation
- Gradient verification using finite differences

**Usage:**
```bash
# Run with default JSON file (ops10_function_graph.json)
./graph_from_json

# Run with custom JSON file
./graph_from_json path/to/your/graph.json
```

## JSON Graph Format

The JSON format for computation graphs follows this structure:

```json
{
  "constants": [
    // Array of constant values used in the graph
    2.0,
    0.5
  ],
  "nodes": [
    // Array of computation nodes
    { "op": "Input" },
    { "op": "Constant", "imm": 0 },
    { "op": "Mul", "a": 0, "b": 1 },
    { "op": "Sin", "a": 2 },
    // ... more nodes
  ],
  "outputs": [
    // Indices of output nodes
    5
  ],
  "diff_inputs": [
    // Indices of nodes that need gradients
    0
  ]
}
```

### Supported Operations

The following operations are supported in JSON graphs:

**Basic Arithmetic:**
- `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`, `Square`, `Recip`, `Mod`

**Mathematical Functions:**
- `Exp`, `Log`, `Sqrt`, `Pow`, `Sin`, `Cos`, `Tan`

**Comparison & Control:**
- `Min`, `Max`, `If`
- `CmpLT`, `CmpLE`, `CmpGT`, `CmpGE`, `CmpEQ`, `CmpNE`

**Special Nodes:**
- `Input`: Graph input node
- `Constant`: Constant value (references constant pool)

## Building Examples

Examples are built automatically when building Forge with CMake:

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

To disable building examples:
```bash
cmake .. -DFORGE_BUILD_EXAMPLES=OFF
```

## Sample Graphs

### ops10_function_graph.json

This computation graph was generated from the following C++ function:

```cpp
template<typename T>
T ops10(T x) {
    T a = x * T(2.0);           // 1: mul
    T p = T(1.0) + T(1.0);
    T u = p * T(2.0);
    T v = u * T(2.0);
    T b = x * x + v;            // 2: mul, 3: add
    T c = a + b + u;            // 4: add, 5: add
    T d = c * T(3.0);           // 6: mul
    T e = d + x;                // 7: add
    T f = e * T(1.5);           // 8: mul
    T g = f - b;                // 9: sub
    T h = g + T(10.0);          // 10: add
    T i = h * T(0.5);           // 11: mul
    T j = x + T(1.0);           // intermediate for div
    return i / j;               // 12: div
}
```

This function demonstrates:
- Multiple mathematical operations (add, mul, sub, div)
- Constant folding opportunities (p, u, v are compile-time constants)
- Use of both input values and constants
- Single input (x), single output configuration

The resulting graph contains 27 nodes after recording with automatic differentiation, including:
- 1 Input node
- 10 Constant nodes
- 16 Operation nodes (Add, Mul, Sub, Div)

Note: Be careful with input values near -1.0, as the function divides by (x + 1) which would cause division by zero.