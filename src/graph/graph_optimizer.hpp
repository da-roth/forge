#pragma once

#include "graph.hpp"  // For Graph structure
#include <string>

namespace forge {

/**
 * Graph-level optimizer for mathematical expression graphs.
 * Performs optimization passes on the graph structure before JIT compilation.
 * 
 * Design principles:
 * - Takes const Graph& input, returns optimized Graph copy
 * - Multiple optimization passes can be chained
 * - Each pass preserves correctness while improving performance
 * - Operates on graph structure, not generated code
 * 
 * Optimization passes:
 * - Inactive folding: Evaluates and folds entire constant subgraphs (nodes with isActive=false)
 *   Example: y = 2 + 3; z = y / 5; result = x + z → result = x + 1.0
 * - Placeholder optimization - Reserved for future high-impact optimization
 * 
 * IMPORTANT: All optimizations must be O(nodes) complexity to maintain fast compile times.
 * Avoid algorithms that are O(nodes²) or worse, especially for large expression graphs.
 */
class GraphOptimizer {
public:
    GraphOptimizer();
    ~GraphOptimizer() = default;

    /**
     * Main optimization entry point.
     * Takes a recorded graph and applies all enabled optimization passes.
     * 
     * @param input The original graph from recorder.stop()
     * @return Optimized graph ready for JIT compilation
     */
    forge::Graph optimize(const forge::Graph& input);
    
    /**
     * Optimization result containing both optimized tape and node ID mapping
     */
    struct OptimizationResult {
        forge::Graph optimizedTape;
        std::vector<forge::NodeId> originalToOptimizedMapping;
    };
    
    /**
     * Optimize with mapping - returns both optimized tape and node ID mapping
     * 
     * @param input The original graph from recorder.stop()
     * @return Optimization result with optimized tape and mapping
     */
    OptimizationResult optimizeWithMapping(const forge::Graph& input);

    // Configuration for optimization passes - SINGLE SOURCE OF TRUTH
    struct OptimizationConfig {
        // Optimization passes (enable/disable as needed)
        bool enableInactiveFolding = true;       // Fold constant subgraphs (isActive=false nodes)
        bool enableCSE = true;                   // Common Subexpression Elimination
        bool enableAlgebraicSimplification = true; // Algebraic simplifications and strength reduction
        bool enableStabilityCleaning = true;    // Fix numerical stability issues (1/exp(x) -> exp(-x)) - TEMPORARY: Disabled to isolate JIT input issue
        bool enablePlaceholderOptimization = false;  // Reserved for future high-impact optimization
        bool enableConstantCleanup = true;      // Remove unused constants from const pool
        
        // Performance vs. compile time trade-offs
        int maxOptimizationPasses = 5;  // Iterate until no changes or max passes (O(k*n) where k≤5)
        
        // Debug output controls (enable/disable as needed for investigation)
        bool printStepByStepDebug = false;  // Print graph after each optimization step
        bool printOriginalGraph = false;    // Print original graph before optimization
        bool printOptimizedGraph = false;   // Print final optimized graph
    };
    
    void setConfig(const OptimizationConfig& config) { config_ = config; }
    const OptimizationConfig& getConfig() const { return config_; }

    // Statistics for analysis and debugging
    struct OptimizationStats {
        size_t originalNodeCount = 0;
        size_t optimizedNodeCount = 0;
        size_t deadNodeCount = 0;        // Nodes marked as dead after optimization
        size_t inactiveNodesFolded = 0;  // Number of inactive subgraphs folded
        size_t duplicatesEliminated = 0; // Number of duplicate subexpressions eliminated
        size_t algebraicSimplifications = 0; // Number of algebraic simplifications applied
        size_t stabilityFixes = 0;       // Number of stability improvements applied
        size_t constantsRemoved = 0;     // Number of unused constants removed
        int passesPerformed = 0;
        bool changesApplied = false;
        
        // Timing information for each optimization pass (in milliseconds)
        double inactiveFoldingTimeMs = 0.0;
        double cseTimeMs = 0.0;
        double algebraicTimeMs = 0.0;
        double stabilityTimeMs = 0.0;
        double totalOptimizationTimeMs = 0.0;
        
        void clear() {
            originalNodeCount = 0;
            optimizedNodeCount = 0; 
            deadNodeCount = 0;
            inactiveNodesFolded = 0;
            duplicatesEliminated = 0;
            algebraicSimplifications = 0;
            stabilityFixes = 0;
            constantsRemoved = 0;
            passesPerformed = 0;
            changesApplied = false;
            inactiveFoldingTimeMs = 0.0;
            cseTimeMs = 0.0;
            algebraicTimeMs = 0.0;
            stabilityTimeMs = 0.0;
            totalOptimizationTimeMs = 0.0;
        }
    };
    
    const OptimizationStats& getLastStats() const { return stats_; }

    // Disable copy (optimizer should be lightweight and stateless)
    GraphOptimizer(const GraphOptimizer&) = delete;
    GraphOptimizer& operator=(const GraphOptimizer&) = delete;

private:
    OptimizationConfig config_;
    mutable OptimizationStats stats_;
    
    // Individual optimization passes are now in separate files in optimizations/ directory
    
    // Helper methods
    bool graphsEqual(const forge::Graph& a, const forge::Graph& b) const;
    void printGraphDebug(const forge::Graph& graph, const std::string& title);
    std::string getOpCodeName(forge::OpCode op) const;
};

} // namespace forge