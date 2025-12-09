#include "graph_optimizer.hpp"
#include "optimizations/optimizations.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace forge {

void GraphOptimizer::printGraphDebug(const forge::Graph& graph, const std::string& title) {
    std::cout << "\n=== " << title << " ===" << std::endl;
    std::cout << "Nodes: " << graph.nodes.size() << std::endl;
    
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        std::printf("  %zu: %s", i, getOpCodeName(node.op).c_str());
        
        if (node.a != UINT32_MAX) std::printf(" %u", node.a);
        if (node.b != UINT32_MAX) std::printf(" %u", node.b);
        if (node.c != UINT32_MAX) std::printf(" %u", node.c);
        
        if (node.op == forge::OpCode::Constant) {
            std::printf(" %.1f", node.imm);
        }
        std::printf(" [active=%d, dead=%d]\n", node.isActive, node.isDead);
    }
}

GraphOptimizer::GraphOptimizer() {
    // Use default values from header file - don't override them here
    // The header file sets the defaults, constructor should respect them
}

forge::Graph GraphOptimizer::optimize(const forge::Graph& input) {
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    
    stats_.clear();
    stats_.originalNodeCount = input.nodes.size();
    
    forge::Graph current = input;  // Copy constructor
    
    // Timing for individual optimization passes
    double inactiveFoldingTime = 0.0;
    double cseTime = 0.0;
    double algebraicTime = 0.0;
    double stabilityTime = 0.0;
    double deadCodeTime = 0.0;
    
    auto totalOptStart = Clock::now();
    
    // IMPORTANT: Apply stability cleaning BEFORE any other optimization
    // This ensures 1/exp(x) patterns are transformed to exp(-x) before constant folding
    if (config_.enableStabilityCleaning) {
        auto start = Clock::now();
        current = optimizations::StabilityCleaning::apply(current, stats_);
        Duration elapsed = Clock::now() - start;
        stabilityTime += elapsed.count();
    }
    
    // Apply optimization passes - up to maxOptimizationPasses iterations
    // Each pass is O(n), so total is O(k*n) where k = maxOptimizationPasses
    for (int pass = 0; pass < config_.maxOptimizationPasses; ++pass) {
        size_t changesMadeThisPass = 0;
        size_t previousFolded = stats_.inactiveNodesFolded;
        size_t previousCSE = stats_.duplicatesEliminated;
        size_t previousSimplified = stats_.algebraicSimplifications;
        size_t previousStability = stats_.stabilityFixes;
        
        if (config_.enableInactiveFolding) {
            auto start = Clock::now();
            current = optimizations::InactiveFolding::apply(current, stats_);
            Duration elapsed = Clock::now() - start;
            inactiveFoldingTime += elapsed.count();
            changesMadeThisPass += (stats_.inactiveNodesFolded - previousFolded);
            if (config_.printStepByStepDebug) {
                printGraphDebug(current, "After Inactive Folding");
            }
        }
        
        if (config_.enableCSE) {
            auto start = Clock::now();
            current = optimizations::CommonSubexpressionElimination::apply(current, stats_);
            Duration elapsed = Clock::now() - start;
            cseTime += elapsed.count();
            changesMadeThisPass += (stats_.duplicatesEliminated - previousCSE);
            if (config_.printStepByStepDebug) {
                printGraphDebug(current, "After CSE");
            }
        }
        
        if (config_.enableAlgebraicSimplification) {
            auto start = Clock::now();
            current = optimizations::AlgebraicSimplification::apply(current, stats_);
            Duration elapsed = Clock::now() - start;
            algebraicTime += elapsed.count();
            changesMadeThisPass += (stats_.algebraicSimplifications - previousSimplified);
            if (config_.printStepByStepDebug) {
                printGraphDebug(current, "After Algebraic Simplification");
            }
        }
        
        // Also run stability cleaning after other optimizations (may expose new patterns)
        if (config_.enableStabilityCleaning) {
            current = optimizations::StabilityCleaning::apply(current, stats_);
            changesMadeThisPass += (stats_.stabilityFixes - previousStability);
            if (config_.printStepByStepDebug) {
                printGraphDebug(current, "After Stability Cleaning");
            }
        }
        
        stats_.passesPerformed = pass + 1;
        
        // Stop early if no changes were made in this pass
        if (changesMadeThisPass == 0) {
            break;
        } else {
            stats_.changesApplied = true;
        }
    }
    
    // Apply constant cleanup as final step
    if (config_.enableConstantCleanup) {
        auto start = Clock::now();
        current = optimizations::ConstantCleanup::apply(current, stats_);
        Duration elapsed = Clock::now() - start;
        if (config_.printStepByStepDebug) {
            printGraphDebug(current, "After Constant Cleanup");
        }
    }
    
    stats_.optimizedNodeCount = current.nodes.size();
    
    // Calculate total optimization time
    Duration totalOptTime = Clock::now() - totalOptStart;
    
    // Store timing information in stats
    stats_.inactiveFoldingTimeMs = inactiveFoldingTime;
    stats_.cseTimeMs = cseTime;
    stats_.algebraicTimeMs = algebraicTime;
    stats_.stabilityTimeMs = stabilityTime;
    stats_.totalOptimizationTimeMs = totalOptTime.count();
    
    // Print detailed optimization timing only if debug flags are set
    if (config_.printOriginalGraph || config_.printOptimizedGraph) {
        std::cout << "\n=== Optimization Pass Timing ===" << std::endl;
        if (inactiveFoldingTime > 0.0)
            std::cout << "  Inactive folding: " << std::fixed << std::setprecision(2) 
                      << inactiveFoldingTime << " ms" << std::endl;
        if (cseTime > 0.0)
            std::cout << "  Common subexpression elimination: " << std::fixed << std::setprecision(2) 
                      << cseTime << " ms" << std::endl;
        if (algebraicTime > 0.0)
            std::cout << "  Algebraic simplification: " << std::fixed << std::setprecision(2) 
                      << algebraicTime << " ms" << std::endl;
        if (stabilityTime > 0.0)
            std::cout << "  Stability cleaning: " << std::fixed << std::setprecision(2) 
                      << stabilityTime << " ms" << std::endl;
        std::cout << "  Total optimization time: " << std::fixed << std::setprecision(2) 
                  << totalOptTime.count() << " ms" << std::endl;
    }
    
    // Print optimization statistics
    if (config_.printOriginalGraph || config_.printOptimizedGraph) {
        std::cout << "\n=== Optimization Statistics ===" << std::endl;
        std::cout << "  Original nodes: " << stats_.originalNodeCount << std::endl;
        std::cout << "  Optimized nodes: " << stats_.optimizedNodeCount << std::endl;
        std::cout << "  Dead nodes: " << stats_.deadNodeCount << std::endl;
        std::cout << "  Inactive nodes folded: " << stats_.inactiveNodesFolded << std::endl;
        std::cout << "  Duplicates eliminated: " << stats_.duplicatesEliminated << std::endl;
        std::cout << "  Algebraic simplifications: " << stats_.algebraicSimplifications << std::endl;
        std::cout << "  Stability fixes: " << stats_.stabilityFixes << std::endl;
        std::cout << "  Constants removed: " << stats_.constantsRemoved << std::endl;
        std::cout << "  Passes performed: " << stats_.passesPerformed << std::endl;
        std::cout << "  Changes applied: " << (stats_.changesApplied ? "Yes" : "No") << std::endl;
    }
    
    // Print graphs if requested
    if (config_.printOriginalGraph) {
        printGraphDebug(input, "Original Graph");
    }
    if (config_.printOptimizedGraph) {
        printGraphDebug(current, "Optimized Graph");
    }
    
    // Constant pool debugging removed
    
    return current;
}

GraphOptimizer::OptimizationResult GraphOptimizer::optimizeWithMapping(const forge::Graph& input) {
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;
    
    stats_.clear();
    stats_.originalNodeCount = input.nodes.size();
    
    // Initialize mapping as identity - maps original node ID to optimized node ID
    std::vector<forge::NodeId> currentMapping(input.nodes.size(), UINT32_MAX);
    for (size_t i = 0; i < input.nodes.size(); ++i) {
        currentMapping[i] = i;
    }
    
    forge::Graph current = input;  // Copy constructor
    
    // Timing for individual optimization passes
    double inactiveFoldingTime = 0.0;
    double cseTime = 0.0;
    double algebraicTime = 0.0;
    double stabilityTime = 0.0;
    double deadCodeTime = 0.0;
    
    auto totalOptStart = Clock::now();
    
    // IMPORTANT: Apply stability cleaning BEFORE any other optimization
    // This ensures 1/exp(x) patterns are transformed to exp(-x) before constant folding
    if (config_.enableStabilityCleaning) {
        auto start = Clock::now();
        current = optimizations::StabilityCleaning::apply(current, stats_);
        Duration elapsed = Clock::now() - start;
        stabilityTime += elapsed.count();
    }
    
    // Apply optimization passes - up to maxOptimizationPasses iterations
    // Each pass is O(n), so total is O(k*n) where k = maxOptimizationPasses
    for (int pass = 0; pass < config_.maxOptimizationPasses; ++pass) {
        size_t changesMadeThisPass = 0;
        size_t previousFolded = stats_.inactiveNodesFolded;
        size_t previousCSE = stats_.duplicatesEliminated;
        size_t previousSimplified = stats_.algebraicSimplifications;
        size_t previousStability = stats_.stabilityFixes;
        
        if (config_.enableInactiveFolding) {
            auto start = Clock::now();
            current = optimizations::InactiveFolding::apply(current, stats_);
            Duration elapsed = Clock::now() - start;
            inactiveFoldingTime += elapsed.count();
            changesMadeThisPass += (stats_.inactiveNodesFolded - previousFolded);
            if (config_.printStepByStepDebug) {
                printGraphDebug(current, "After Inactive Folding");
            }
        }
        
        if (config_.enableCSE) {
            auto start = Clock::now();
            current = optimizations::CommonSubexpressionElimination::apply(current, stats_);
            Duration elapsed = Clock::now() - start;
            cseTime += elapsed.count();
            changesMadeThisPass += (stats_.duplicatesEliminated - previousCSE);
            if (config_.printStepByStepDebug) {
                printGraphDebug(current, "After CSE");
            }
        }
        
        if (config_.enableAlgebraicSimplification) {
            auto start = Clock::now();
            current = optimizations::AlgebraicSimplification::apply(current, stats_);
            Duration elapsed = Clock::now() - start;
            algebraicTime += elapsed.count();
            changesMadeThisPass += (stats_.algebraicSimplifications - previousSimplified);
            if (config_.printStepByStepDebug) {
                printGraphDebug(current, "After Algebraic Simplification");
            }
        }
        
        // Also run stability cleaning after other optimizations (may expose new patterns)
        if (config_.enableStabilityCleaning) {
            current = optimizations::StabilityCleaning::apply(current, stats_);
            changesMadeThisPass += (stats_.stabilityFixes - previousStability);
            if (config_.printStepByStepDebug) {
                printGraphDebug(current, "After Stability Cleaning");
            }
        }
        
        stats_.passesPerformed = pass + 1;
        
        // Stop early if no changes were made in this pass
        if (changesMadeThisPass == 0) {
            break;
        } else {
            stats_.changesApplied = true;
        }
    }
    
    // Apply constant cleanup as final step
    if (config_.enableConstantCleanup) {
        auto start = Clock::now();
        current = optimizations::ConstantCleanup::apply(current, stats_);
        Duration elapsed = Clock::now() - start;
        if (config_.printStepByStepDebug) {
            printGraphDebug(current, "After Constant Cleanup");
        }
    }
    
    stats_.optimizedNodeCount = current.nodes.size();
    
    // Calculate total optimization time
    Duration totalOptTime = Clock::now() - totalOptStart;
    
    // Store timing information in stats
    stats_.inactiveFoldingTimeMs = inactiveFoldingTime;
    stats_.cseTimeMs = cseTime;
    stats_.algebraicTimeMs = algebraicTime;
    stats_.stabilityTimeMs = stabilityTime;
    stats_.totalOptimizationTimeMs = totalOptTime.count();
    
    // Print detailed optimization timing only if debug flags are set
    if (config_.printOriginalGraph || config_.printOptimizedGraph) {
        std::cout << "\n=== Optimization Pass Timing ===" << std::endl;
        if (inactiveFoldingTime > 0.0)
            std::cout << "  Inactive folding: " << std::fixed << std::setprecision(2) 
                      << inactiveFoldingTime << " ms" << std::endl;
        if (cseTime > 0.0)
            std::cout << "  Common subexpression elimination: " << std::fixed << std::setprecision(2) 
                      << cseTime << " ms" << std::endl;
        if (algebraicTime > 0.0)
            std::cout << "  Algebraic simplification: " << std::fixed << std::setprecision(2) 
                      << algebraicTime << " ms" << std::endl;
        if (stabilityTime > 0.0)
            std::cout << "  Stability cleaning: " << std::fixed << std::setprecision(2) 
                      << stabilityTime << " ms" << std::endl;
        std::cout << "  Total optimization time: " << std::fixed << std::setprecision(2) 
                  << totalOptTime.count() << " ms" << std::endl;
    }
    
    // Print optimization statistics
    if (config_.printOriginalGraph || config_.printOptimizedGraph) {
        std::cout << "\n=== Optimization Statistics ===" << std::endl;
        std::cout << "  Original nodes: " << stats_.originalNodeCount << std::endl;
        std::cout << "  Optimized nodes: " << stats_.optimizedNodeCount << std::endl;
        std::cout << "  Dead nodes: " << stats_.deadNodeCount << std::endl;
        std::cout << "  Inactive nodes folded: " << stats_.inactiveNodesFolded << std::endl;
        std::cout << "  Duplicates eliminated: " << stats_.duplicatesEliminated << std::endl;
        std::cout << "  Algebraic simplifications: " << stats_.algebraicSimplifications << std::endl;
        std::cout << "  Stability fixes: " << stats_.stabilityFixes << std::endl;
        std::cout << "  Constants removed: " << stats_.constantsRemoved << std::endl;
        std::cout << "  Passes performed: " << stats_.passesPerformed << std::endl;
        std::cout << "  Changes applied: " << (stats_.changesApplied ? "Yes" : "No") << std::endl;
    }
    
    // Print graphs if requested
    if (config_.printOriginalGraph) {
        printGraphDebug(input, "Original Graph");
    }
    if (config_.printOptimizedGraph) {
        printGraphDebug(current, "Optimized Graph");
    }
    
    // The current implementation doesn't properly track mappings through optimizations
    // For now, we need to implement a proper mapping system
    // This is a temporary solution that maintains the original node IDs
    
    OptimizationResult result;
    result.optimizedTape = current;
    
    // Create a mapping from original to optimized node IDs
    // Minimal correct mapping: map all inputs by order and all outputs by position
    std::vector<forge::NodeId> finalMapping(input.nodes.size(), UINT32_MAX);
    
    // Map inputs by ordinal occurrence of OpCode::Input
    if (!input.nodes.empty()) {
        // Collect input indices in original and optimized tapes
        std::vector<forge::NodeId> originalInputs;
        originalInputs.reserve(input.nodes.size());
        for (forge::NodeId i = 0; i < input.nodes.size(); ++i) {
            if (input.nodes[i].op == forge::OpCode::Input) {
                originalInputs.push_back(i);
            }
        }
        std::vector<forge::NodeId> optimizedInputs;
        optimizedInputs.reserve(current.nodes.size());
        for (forge::NodeId i = 0; i < current.nodes.size(); ++i) {
            if (current.nodes[i].op == forge::OpCode::Input) {
                optimizedInputs.push_back(i);
            }
        }
        const size_t numInputsToMap = std::min(originalInputs.size(), optimizedInputs.size());
        for (size_t k = 0; k < numInputsToMap; ++k) {
            finalMapping[originalInputs[k]] = optimizedInputs[k];
        }
        
        // Map outputs by position
        const size_t numOutputsToMap = std::min(input.outputs.size(), current.outputs.size());
        for (size_t k = 0; k < numOutputsToMap; ++k) {
            const auto origOutNode = input.outputs[k];
            const auto optOutNode = current.outputs[k];
            if (origOutNode < finalMapping.size()) {
                finalMapping[origOutNode] = optOutNode;
            }
        }
    }
    
        // Debug output (commented out for cleaner test output)
        // std::cout << "[MAPPING DEBUG] Original tape size: " << input.nodes.size() << ", Optimized tape size: " << current.nodes.size() << std::endl;
        // for (size_t i = 0; i < finalMapping.size(); ++i) {
        //     std::cout << "[MAPPING DEBUG] " << i << " -> " << finalMapping[i] << std::endl;
        // }
    
    result.originalToOptimizedMapping = finalMapping;
    
    return result;
}

bool GraphOptimizer::graphsEqual(const forge::Graph& a, const forge::Graph& b) const {
    // Simple equality check for optimization iteration
    if (a.nodes.size() != b.nodes.size()) {
        return false;
    }
    
    // Compare node by node
    for (size_t i = 0; i < a.nodes.size(); ++i) {
        const auto& nodeA = a.nodes[i];
        const auto& nodeB = b.nodes[i];
        
        if (nodeA.op != nodeB.op || 
            nodeA.a != nodeB.a || 
            nodeA.b != nodeB.b ||
            nodeA.c != nodeB.c ||
            nodeA.imm != nodeB.imm ||
            nodeA.isActive != nodeB.isActive ||
            nodeA.isDead != nodeB.isDead) {
            return false;
        }
    }
    
    return true;
}

std::string GraphOptimizer::getOpCodeName(forge::OpCode op) const {
    switch (op) {
        case forge::OpCode::Input: return "Input";
        case forge::OpCode::Add: return "Add";
        case forge::OpCode::Sub: return "Sub";
        case forge::OpCode::Mul: return "Mul";
        case forge::OpCode::Div: return "Div";
        case forge::OpCode::Pow: return "Pow";
        case forge::OpCode::Exp: return "Exp";
        case forge::OpCode::Log: return "Log";
        case forge::OpCode::Sin: return "Sin";
        case forge::OpCode::Cos: return "Cos";
        case forge::OpCode::Tan: return "Tan";
        case forge::OpCode::Sqrt: return "Sqrt";
        case forge::OpCode::Abs: return "Abs";
        case forge::OpCode::Neg: return "Neg";
        case forge::OpCode::Constant: return "Constant";
        default: return "Unknown";
    }
}

} // namespace forge
