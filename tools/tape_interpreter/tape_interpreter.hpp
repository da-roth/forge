#pragma once

#include "../../src/graph/graph.hpp"
#include "../../src/graph/graph_debug.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <map>

namespace forge {
namespace tools {

// Simple interpreter for testing graph correctness
// This is only for debugging/testing, not for production use
class TapeInterpreter {
private:
    const Graph& graph_;
    std::vector<double> values_;
    std::vector<NodeId> inputNodeIds_;  // Track which nodes are inputs for easy access
    
public:
    explicit TapeInterpreter(const Graph& graph) : graph_(graph) {
        values_.resize(graph.nodes.size());
        
        // Build list of input node IDs for easy indexing
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            if (graph.nodes[i].op == OpCode::Input) {
                inputNodeIds_.push_back(static_cast<NodeId>(i));
            }
        }
    }
    
    void setInput(NodeId nodeId, double value) {
        if (nodeId >= graph_.nodes.size()) {
            throw std::runtime_error("Invalid node ID for input");
        }
        if (graph_.nodes[nodeId].op != OpCode::Input) {
            throw std::runtime_error("Node is not an input node");
        }
        values_[nodeId] = value;
    }
    
    // Simplified input setting by index (much cleaner for bridge)
    void setInputByIndex(size_t inputIndex, double value) {
        if (inputIndex >= inputNodeIds_.size()) {
            throw std::runtime_error("Input index out of range");
        }
        NodeId nodeId = inputNodeIds_[inputIndex];
        values_[nodeId] = value;
    }
    
    // Get number of inputs
    size_t getInputCount() const {
        return inputNodeIds_.size();
    }
    
    double getOutput(NodeId nodeId) const {
        if (nodeId >= values_.size()) {
            throw std::runtime_error("Invalid node ID for output");
        }
        return values_[nodeId];
    }
    
    void execute() {
        // Process nodes in order (they're already topologically sorted)
        for (size_t i = 0; i < graph_.nodes.size(); ++i) {
            const auto& node = graph_.nodes[i];
            
            // Add safety check for large tapes - report progress periodically
            if (i % 100000 == 0) {
                // Use printf to avoid C++ stream issues
                printf("[INTERPRETER] Processing node %zu / %zu (%.1f%%)\n", 
                       i, graph_.nodes.size(), (double)i / graph_.nodes.size() * 100.0);
                fflush(stdout);
            }
            
            // Bounds checking - this might be the issue
            if (node.a >= values_.size() || node.b >= values_.size() || node.c >= values_.size()) {
                printf("[INTERPRETER] ERROR: Node %zu references out-of-bounds index (a=%u, b=%u, c=%u, values_size=%zu)\n",
                       i, node.a, node.b, node.c, values_.size());
                throw std::runtime_error("Node references out-of-bounds value array index");
            }
            
            switch (node.op) {
                case OpCode::Input:
                    // Value already set via setInput
                    break;
                    
                case OpCode::Constant:
                    // Load constant from pool
                    {
                        size_t constIdx = static_cast<size_t>(node.imm);
                        if (constIdx >= graph_.constPool.size()) {
                            throw std::runtime_error("Invalid constant pool index");
                        }
                        values_[i] = graph_.constPool[constIdx];
                    }
                    break;
                    
                case OpCode::Add:
                    values_[i] = values_[node.a] + values_[node.b];
                    break;
                    
                case OpCode::Sub:
                    values_[i] = values_[node.a] - values_[node.b];
                    break;
                    
                case OpCode::Mul:
                    values_[i] = values_[node.a] * values_[node.b];
                    break;
                    
                case OpCode::Div:
                    values_[i] = values_[node.a] / values_[node.b];
                    break;
                    
                case OpCode::Neg:
                    values_[i] = -values_[node.a];
                    break;
                    
                case OpCode::Abs:
                    values_[i] = std::abs(values_[node.a]);
                    break;
                    
                case OpCode::Square:
                    values_[i] = values_[node.a] * values_[node.a];
                    break;
                    
                case OpCode::Recip:
                    values_[i] = 1.0 / values_[node.a];
                    break;
                    
                case OpCode::Mod:
                    values_[i] = std::fmod(values_[node.a], values_[node.b]);
                    break;
                    
                case OpCode::Exp:
                    values_[i] = std::exp(values_[node.a]);
                    break;
                    
                case OpCode::Log:
                    values_[i] = std::log(values_[node.a]);
                    break;
                    
                case OpCode::Sqrt:
                    values_[i] = std::sqrt(values_[node.a]);
                    break;
                    
                case OpCode::Pow:
                    values_[i] = std::pow(values_[node.a], values_[node.b]);
                    break;
                    
                case OpCode::Sin:
                    values_[i] = std::sin(values_[node.a]);
                    break;
                    
                case OpCode::Cos:
                    values_[i] = std::cos(values_[node.a]);
                    break;
                    
                case OpCode::Tan:
                    values_[i] = std::tan(values_[node.a]);
                    break;
                    
                case OpCode::Min:
                    values_[i] = std::fmin(values_[node.a], values_[node.b]);
                    break;
                    
                case OpCode::Max:
                    values_[i] = std::fmax(values_[node.a], values_[node.b]);
                    break;
                    
                case OpCode::CmpLT:
                    values_[i] = (values_[node.a] < values_[node.b]) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::CmpLE:
                    values_[i] = (values_[node.a] <= values_[node.b]) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::CmpGT:
                    values_[i] = (values_[node.a] > values_[node.b]) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::CmpGE:
                    values_[i] = (values_[node.a] >= values_[node.b]) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::CmpEQ:
                    values_[i] = (values_[node.a] == values_[node.b]) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::CmpNE:
                    values_[i] = (values_[node.a] != values_[node.b]) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::If:
                    // Conditional selection: condition ? true_val : false_val
                    // node.a = condition (0.0 or 1.0)
                    // node.b = true value
                    // node.c = false value
                    values_[i] = (values_[node.a] != 0.0) ? values_[node.b] : values_[node.c];
                    break;
                    
                case OpCode::BoolConstant:
                    // Boolean constant stored in imm field
                    values_[i] = node.imm;
                    break;
                    
                case OpCode::BoolAnd:
                    // Logical AND: both must be non-zero
                    values_[i] = ((values_[node.a] != 0.0) && (values_[node.b] != 0.0)) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::BoolOr:
                    // Logical OR: at least one must be non-zero
                    values_[i] = ((values_[node.a] != 0.0) || (values_[node.b] != 0.0)) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::BoolNot:
                    // Logical NOT: flip the boolean value
                    values_[i] = (values_[node.a] == 0.0) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::BoolEq:
                    // Boolean equality: both same (both 0 or both non-0)
                    values_[i] = ((values_[node.a] == 0.0) == (values_[node.b] == 0.0)) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::BoolNe:
                    // Boolean inequality: different (one 0, one non-0)
                    values_[i] = ((values_[node.a] == 0.0) != (values_[node.b] == 0.0)) ? 1.0 : 0.0;
                    break;
                    
                // Integer operations
                case OpCode::IntConstant:
                    // Integer constant stored as double in imm field
                    values_[i] = node.imm;
                    break;
                    
                case OpCode::IntAdd:
                    // Integer addition (truncate to remove floating point errors)
                    values_[i] = std::trunc(values_[node.a]) + std::trunc(values_[node.b]);
                    break;
                    
                case OpCode::IntSub:
                    // Integer subtraction
                    values_[i] = std::trunc(values_[node.a]) - std::trunc(values_[node.b]);
                    break;
                    
                case OpCode::IntMul:
                    // Integer multiplication
                    values_[i] = std::trunc(values_[node.a]) * std::trunc(values_[node.b]);
                    break;
                    
                case OpCode::IntDiv:
                    // Integer division (truncate toward zero)
                    values_[i] = std::trunc(std::trunc(values_[node.a]) / std::trunc(values_[node.b]));
                    break;
                    
                case OpCode::IntMod:
                    // Integer modulo
                    {
                        double a = std::trunc(values_[node.a]);
                        double b = std::trunc(values_[node.b]);
                        values_[i] = std::fmod(a, b);
                    }
                    break;
                    
                case OpCode::IntNeg:
                    // Integer negation
                    values_[i] = -std::trunc(values_[node.a]);
                    break;
                    
                // No conversions - fint is purely integer-only
                    
                // Integer comparisons (return Bool as 0.0/1.0)
                case OpCode::IntCmpLT:
                    values_[i] = (std::trunc(values_[node.a]) < std::trunc(values_[node.b])) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::IntCmpLE:
                    values_[i] = (std::trunc(values_[node.a]) <= std::trunc(values_[node.b])) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::IntCmpGT:
                    values_[i] = (std::trunc(values_[node.a]) > std::trunc(values_[node.b])) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::IntCmpGE:
                    values_[i] = (std::trunc(values_[node.a]) >= std::trunc(values_[node.b])) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::IntCmpEQ:
                    values_[i] = (std::trunc(values_[node.a]) == std::trunc(values_[node.b])) ? 1.0 : 0.0;
                    break;
                    
                case OpCode::IntCmpNE:
                    values_[i] = (std::trunc(values_[node.a]) != std::trunc(values_[node.b])) ? 1.0 : 0.0;
                    break;
                    
                // Integer conditional selection
                case OpCode::IntIf:
                    // Conditional selection: condition ? int_true : int_false
                    // node.a = condition (0.0 or 1.0)
                    // node.b = true integer value
                    // node.c = false integer value
                    values_[i] = (values_[node.a] != 0.0) ? std::trunc(values_[node.b]) : std::trunc(values_[node.c]);
                    break;
                    
                default:
                    throw std::runtime_error("Unsupported operation in interpreter");
            }
        }
    }
    
    // Standalone execution - completely independent of any workspace/kernel infrastructure
    std::vector<double> executeStandalone(const std::vector<double>& inputValues) {
        // Debug: Check what the graph reference actually contains
        printf("[INTERPRETER] DEBUG: graph_ reference has %zu nodes, %zu constants, %zu outputs\n", 
               graph_.nodes.size(), graph_.constPool.size(), graph_.outputs.size());
        
        // Check if we have recorded results for comparison
        auto* debugRecorder = DebugRecorderManager::get();
        bool hasRecordingResults = debugRecorder && !debugRecorder->recordingResults.empty();
        int divergenceCount = 0;
        size_t firstDivergence = SIZE_MAX;
        if (hasRecordingResults) {
            printf("[INTERPRETER] COMPARISON MODE: Found %zu recorded results for step-by-step comparison\n", 
                   debugRecorder->recordingResults.size());
        }
        
        // Memory requirement
        size_t memoryMB = (graph_.nodes.size() * sizeof(double)) / (1024 * 1024);
        printf("[INTERPRETER] Standalone execution: %zu nodes, ~%zu MB memory\n", 
               graph_.nodes.size(), memoryMB);
        
        // Clear and initialize our own value array
        values_.clear();
        values_.resize(graph_.nodes.size(), 0.0);
        
        // Set input values directly by finding input nodes
        size_t inputIndex = 0;
        for (size_t i = 0; i < graph_.nodes.size() && inputIndex < inputValues.size(); ++i) {
            if (graph_.nodes[i].op == OpCode::Input) {
                values_[i] = inputValues[inputIndex];
                printf("[INTERPRETER] Set input node %zu = %f\n", i, inputValues[inputIndex]);
                
                // Update reference values for input nodes if recording results are available
                if (hasRecordingResults && i < debugRecorder->recordingResults.size()) {
                    debugRecorder->recordingResults[i] = inputValues[inputIndex];
                    printf("[INTERPRETER] Updated reference value for input node %zu to %f\n", i, inputValues[inputIndex]);
                }
                
                inputIndex++;
            }
        }
        
        printf("[INTERPRETER] Starting execution loop for %zu nodes\n", graph_.nodes.size());
        
        // Execute the computation - completely self-contained loop
        for (size_t i = 0; i < graph_.nodes.size(); ++i) {
            const auto& node = graph_.nodes[i];
            
            // Progress reporting
            if (i % 100000 == 0) {
                printf("[INTERPRETER] Processing node %zu / %zu (%.1f%%)\n", 
                       i, graph_.nodes.size(), (double)i / graph_.nodes.size() * 100.0);
                fflush(stdout);
            }
            
            // Store previous value for comparison
            double previousValue = values_[i];
            
            switch (node.op) {
                case OpCode::Input:
                    // Already set above
                    break;
                    
                case OpCode::Constant:
                    // Load from constant pool
                    {
                        size_t constIdx = static_cast<size_t>(node.imm);
                        if (constIdx < graph_.constPool.size()) {
                            values_[i] = graph_.constPool[constIdx];
                        }
                    }
                    break;
                    
                case OpCode::Add:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = values_[node.a] + values_[node.b];
                    }
                    break;
                    
                case OpCode::Sub:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = values_[node.a] - values_[node.b];
                    }
                    break;
                    
                case OpCode::Mul:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = values_[node.a] * values_[node.b];
                    }
                    break;
                    
                case OpCode::Div:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = values_[node.a] / values_[node.b];
                    }
                    break;
                    
                case OpCode::Neg:
                    if (node.a < values_.size()) {
                        values_[i] = -values_[node.a];
                    }
                    break;
                    
                case OpCode::Abs:
                    if (node.a < values_.size()) {
                        values_[i] = std::abs(values_[node.a]);
                    }
                    break;
                    
                case OpCode::Square:
                    if (node.a < values_.size()) {
                        values_[i] = values_[node.a] * values_[node.a];
                    }
                    break;
                    
                case OpCode::Exp:
                    if (node.a < values_.size()) {
                        values_[i] = std::exp(values_[node.a]);
                    }
                    break;
                    
                case OpCode::Log:
                    if (node.a < values_.size()) {
                        values_[i] = std::log(values_[node.a]);
                    }
                    break;
                    
                case OpCode::Sqrt:
                    if (node.a < values_.size()) {
                        values_[i] = std::sqrt(values_[node.a]);
                    }
                    break;
                    
                case OpCode::Pow:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = std::pow(values_[node.a], values_[node.b]);
                    }
                    break;
                    
                case OpCode::Max:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = std::fmax(values_[node.a], values_[node.b]);
                    }
                    break;
                    
                case OpCode::Min:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = std::fmin(values_[node.a], values_[node.b]);
                    }
                    break;
                    
                case OpCode::CmpLT:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = (values_[node.a] < values_[node.b]) ? 1.0 : 0.0;
                    }
                    break;
                    
                case OpCode::CmpLE:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = (values_[node.a] <= values_[node.b]) ? 1.0 : 0.0;
                    }
                    break;
                    
                case OpCode::CmpGT:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = (values_[node.a] > values_[node.b]) ? 1.0 : 0.0;
                    }
                    break;
                    
                case OpCode::CmpGE:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = (values_[node.a] >= values_[node.b]) ? 1.0 : 0.0;
                    }
                    break;
                    
                case OpCode::CmpEQ:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = (values_[node.a] == values_[node.b]) ? 1.0 : 0.0;
                    }
                    break;
                    
                case OpCode::If:
                    if (node.a < values_.size() && node.b < values_.size() && node.c < values_.size()) {
                        values_[i] = (values_[node.a] != 0.0) ? values_[node.b] : values_[node.c];
                    }
                    break;
                    
                case OpCode::BoolConstant:
                    // Boolean constant stored in imm field
                    values_[i] = node.imm;
                    break;
                    
                case OpCode::BoolAnd:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = ((values_[node.a] != 0.0) && (values_[node.b] != 0.0)) ? 1.0 : 0.0;
                    }
                    break;
                    
                case OpCode::BoolOr:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = ((values_[node.a] != 0.0) || (values_[node.b] != 0.0)) ? 1.0 : 0.0;
                    }
                    break;
                    
                case OpCode::BoolNot:
                    if (node.a < values_.size()) {
                        values_[i] = (values_[node.a] == 0.0) ? 1.0 : 0.0;
                    }
                    break;
                    
                case OpCode::BoolEq:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = ((values_[node.a] == 0.0) == (values_[node.b] == 0.0)) ? 1.0 : 0.0;
                    }
                    break;
                    
                case OpCode::BoolNe:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = ((values_[node.a] == 0.0) != (values_[node.b] == 0.0)) ? 1.0 : 0.0;
                    }
                    break;
                    
                case OpCode::Sin:
                    if (node.a < values_.size()) {
                        values_[i] = std::sin(values_[node.a]);
                    }
                    break;
                    
                case OpCode::Cos:
                    if (node.a < values_.size()) {
                        values_[i] = std::cos(values_[node.a]);
                    }
                    break;
                    
                case OpCode::Tan:
                    if (node.a < values_.size()) {
                        values_[i] = std::tan(values_[node.a]);
                    }
                    break;
                    
                case OpCode::Recip:
                    if (node.a < values_.size()) {
                        values_[i] = 1.0 / values_[node.a];
                    }
                    break;
                    
                case OpCode::Mod:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = std::fmod(values_[node.a], values_[node.b]);
                    }
                    break;
                    
                case OpCode::CmpNE:
                    if (node.a < values_.size() && node.b < values_.size()) {
                        values_[i] = (values_[node.a] != values_[node.b]) ? 1.0 : 0.0;
                    }
                    break;
                    
                default:
                    // Skip unknown operations rather than crashing, but count them
                    static std::map<int, int> unsupportedOpCounts;
                    unsupportedOpCounts[static_cast<int>(node.op)]++;
                    
                    if (unsupportedOpCounts[static_cast<int>(node.op)] <= 5) {  // Only log first 5 of each type
                        printf("[INTERPRETER] WARNING: Unsupported operation %d at node %zu (occurrence #%d)\n", 
                               static_cast<int>(node.op), i, unsupportedOpCounts[static_cast<int>(node.op)]);
                    } else if (unsupportedOpCounts[static_cast<int>(node.op)] == 6) {
                        printf("[INTERPRETER] WARNING: Suppressing further warnings for operation %d (occurs frequently)\n", 
                               static_cast<int>(node.op));
                    }
                    
                    // Print summary at end of execution (only once)
                    if (i == graph_.nodes.size() - 1 && !unsupportedOpCounts.empty()) {
                        printf("[INTERPRETER] SUMMARY: Unsupported operations encountered:\n");
                        for (const auto& pair : unsupportedOpCounts) {
                            printf("  Operation %d: %d occurrences\n", pair.first, pair.second);
                        }
                    }
                    break;
            }
            
            // Compare with recorded result if available
            // Skip input nodes as they don't have values during recording
            if (hasRecordingResults && i < debugRecorder->recordingResults.size() && 
                node.op != OpCode::Input) {
                double expected = debugRecorder->recordingResults[i];
                double actual = values_[i];
                double diff = std::abs(actual - expected);
                
                if (diff > 1e-12) {  // Tolerance for floating point comparison
                    divergenceCount++;
                    if (firstDivergence == SIZE_MAX) {
                        firstDivergence = i;
                        printf("[INTERPRETER] FIRST DIVERGENCE (non-input) at node %zu:\n", i);
                        printf("  Operation: %d (%s)\n", static_cast<int>(node.op),
                               node.op == OpCode::Constant ? "Constant" :
                               node.op == OpCode::Add ? "Add" :
                               node.op == OpCode::Sub ? "Sub" :
                               node.op == OpCode::Mul ? "Mul" :
                               node.op == OpCode::Div ? "Div" :
                               node.op == OpCode::Neg ? "Neg" :
                               node.op == OpCode::Abs ? "Abs" :
                               node.op == OpCode::Sqrt ? "Sqrt" :
                               node.op == OpCode::Exp ? "Exp" :
                               node.op == OpCode::Log ? "Log" :
                               node.op == OpCode::Sin ? "Sin" :
                               node.op == OpCode::Cos ? "Cos" :
                               node.op == OpCode::Max ? "Max" :
                               node.op == OpCode::Min ? "Min" :
                               node.op == OpCode::CmpLT ? "CmpLT" :
                               node.op == OpCode::CmpGT ? "CmpGT" :
                               node.op == OpCode::If ? "If" :
                               "Unknown");
                        printf("  Expected (recording): %.17g\n", expected);
                        printf("  Actual (interpreter): %.17g\n", actual);
                        printf("  Difference: %.17g\n", diff);
                        if (node.op != OpCode::Constant) {
                            printf("  Input a: node %u = %.17g\n", node.a, 
                                   node.a < values_.size() ? values_[node.a] : 0.0);
                            if (node.b < values_.size()) {
                                printf("  Input b: node %u = %.17g\n", node.b, values_[node.b]);
                            }
                            if (node.c < values_.size()) {
                                printf("  Input c: node %u = %.17g\n", node.c, values_[node.c]);
                            }
                        }
                    } else if (divergenceCount <= 50) {  // Show more divergences
                        const char* opName = 
                            node.op == OpCode::Constant ? "Const" :
                            node.op == OpCode::Add ? "Add" :
                            node.op == OpCode::Sub ? "Sub" :
                            node.op == OpCode::Mul ? "Mul" :
                            node.op == OpCode::Div ? "Div" :
                            node.op == OpCode::If ? "If" :
                            node.op == OpCode::Max ? "Max" :
                            node.op == OpCode::Min ? "Min" :
                            node.op == OpCode::CmpLT ? "LT" :
                            node.op == OpCode::CmpGT ? "GT" :
                            "Op";
                        printf("[INTERPRETER] DIV #%d at node %zu (%s): exp %.17g, got %.17g (diff: %.17g)\n", 
                               divergenceCount, i, opName, expected, actual, diff);
                        
                        // Show input values for operations to compare recorded vs computed
                        if (node.op != OpCode::Constant && node.op != OpCode::BoolConstant) {
                            if (node.a < debugRecorder->recordingResults.size()) {
                                printf("    Input a[%u]: recorded=%.17g, computed=%.17g\n", 
                                       node.a, debugRecorder->recordingResults[node.a], values_[node.a]);
                            }
                            if (node.b < values_.size() && node.b < debugRecorder->recordingResults.size()) {
                                printf("    Input b[%u]: recorded=%.17g, computed=%.17g\n", 
                                       node.b, debugRecorder->recordingResults[node.b], values_[node.b]);
                            }
                            if (node.c < values_.size() && node.c < debugRecorder->recordingResults.size() && 
                                (node.op == OpCode::If || node.op == OpCode::IntIf)) {
                                printf("    Input c[%u]: recorded=%.17g, computed=%.17g\n", 
                                       node.c, debugRecorder->recordingResults[node.c], values_[node.c]);
                            }
                        }
                    }
                }
            }
        }
        
        printf("[INTERPRETER] Execution complete, returning %zu values\n", values_.size());
        return values_;
    }
    
    // Convenience function for single input, single output graphs
    double evaluate(double input) {
        std::vector<double> inputs = {input};
        auto results = executeStandalone(inputs);
        
        if (graph_.outputs.empty()) {
            throw std::runtime_error("No output nodes marked");
        }
        
        NodeId outputNode = graph_.outputs[0];
        if (outputNode >= results.size()) {
            throw std::runtime_error("Output node ID out of range");
        }
        
        return results[outputNode];
    }
};

} // namespace tools
} // namespace forge