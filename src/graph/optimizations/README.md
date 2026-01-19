# Graph Optimizations

Pre-processing passes applied to computation graphs before code generation. All optimizations are O(n) complexity to maintain fast compile times.

## Optimization Passes

### Inactive Folding

Evaluates constant subgraphs at compile time. Nodes that don't depend on inputs (`isActive=false`) are computed and replaced with constants.

```
Before: y = 2 + 3; z = y / 5; result = x + z
After:  result = x + 1.0
```

### Common Subexpression Elimination (CSE)

Identifies duplicate computations and redirects references to a single canonical node.

```
Before: a = x + y; b = x + y; c = a * b
After:  a = x + y; c = a * a
```

### Algebraic Simplification

Applies algebraic identities and strength reduction:

| Pattern | Simplification |
|---------|---------------|
| `x * 1.0` | `x` |
| `x + 0.0` | `x` |
| `x - x` | `0.0` |
| `x * x` | `Square(x)` |
| `x * 2.0` | `x + x` |

### Stability Cleaning

Transforms numerically unstable patterns into stable equivalents:

| Unstable | Stable |
|----------|--------|
| `1.0 / exp(x)` | `exp(-x)` |
| `exp(x) / exp(y)` | `exp(x - y)` |

### Constant Cleanup

Removes unused constants from the constant pool after other optimizations have run, reducing memory usage and improving cache locality.

## Configuration

```cpp
GraphOptimizer optimizer;
auto config = optimizer.getConfig();

config.enableInactiveFolding = true;        // Default: true
config.enableCSE = true;                    // Default: true
config.enableAlgebraicSimplification = true; // Default: true
config.enableStabilityCleaning = true;      // Default: true
config.enableConstantCleanup = true;        // Default: true
config.maxOptimizationPasses = 5;           // Iterate until fixed point

optimizer.setConfig(config);
Graph optimized = optimizer.optimize(graph);
```

## Statistics

After optimization, retrieve statistics:

```cpp
auto stats = optimizer.getLastStats();
// stats.originalNodeCount, stats.optimizedNodeCount
// stats.duplicatesEliminated, stats.algebraicSimplifications
// stats.stabilityFixes, stats.totalOptimizationTimeMs
```

## Files

| File | Description |
|------|-------------|
| `inactive_folding.hpp/cpp` | Constant subgraph evaluation |
| `common_subexpression_elimination.hpp/cpp` | Duplicate computation removal |
| `algebraic_simplification.hpp/cpp` | Identity and strength reduction |
| `stability_cleaning.hpp/cpp` | Numerical stability transforms |
| `constant_cleanup.hpp/cpp` | Unused constant removal |
| `optimizations.hpp` | Convenience header including all passes |

## Adding Custom Passes

Optimization passes follow a consistent pattern:

```cpp
class MyOptimization {
public:
    static Graph apply(const Graph& graph,
                       GraphOptimizer::OptimizationStats& stats);
};
```

To integrate with `GraphOptimizer`, add your pass to the optimization loop in `graph_optimizer.cpp`.

## See Also

- [graph_optimizer.hpp](../graph_optimizer.hpp) — Optimization orchestrator
- [backends/](../../../backends/) — Code generation after optimization
