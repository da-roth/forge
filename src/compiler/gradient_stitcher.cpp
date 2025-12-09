// This file is part of Forge <https://github.com/da-roth/forge>
//
// See LICENSE.md for license and copyright information
// SPDX-License-Identifier: Zlib

/**
 * @file gradient_stitcher.cpp
 * @brief Implementation of gradient pass code generation for automatic differentiation
 *
 * Generates x86/x64 assembly code for computing gradients via reverse-mode
 * automatic differentiation (backpropagation).
 */

#include "gradient_stitcher.hpp"
#include "forge_engine.hpp"
#include <stdexcept>
#include <iostream>
#include <unordered_map>

namespace forge {

using namespace asmjit;
using namespace forge;

// Helper function to convert OpCode to string for debugging
const char* opName(OpCode op) {
    switch(op) {
        case OpCode::Input: return "Input";
        case OpCode::Constant: return "Constant";
        case OpCode::Add: return "Add";
        case OpCode::Sub: return "Sub";
        case OpCode::Mul: return "Mul";
        case OpCode::Div: return "Div";
        case OpCode::Neg: return "Neg";
        case OpCode::Exp: return "Exp";
        case OpCode::Log: return "Log";
        case OpCode::Pow: return "Pow";
        case OpCode::Sqrt: return "Sqrt";
        case OpCode::Square: return "Square";
        case OpCode::Recip: return "Recip";
        default: return "Unknown";
    }
}

// All helper methods have been removed - using instruction set abstraction instead

void GradientStitcher::generateGradientOperation(
    x86::Assembler& a,
    const Node& node,
    NodeId nodeId,
    IRegisterAllocator& regState,
    const forge::Graph& graph,
    const std::unordered_map<NodeId, ForgeEngine::ConstantInfo>& constantMap,
    const Label& constPoolLabel,
    IInstructionSet* instructionSet,
    const CompilerConfig* config) {
    
    // Only process if node needs gradient
    if (!node.needsGradient) return;
    
    switch(node.op) {
        case OpCode::Add:
            // grad[a] += grad[nodeId]
            // grad[b] += grad[nodeId]
            
            if (config && config->printGradientDebug) {
                std::cout << "    Add: node.a=" << node.a << " (needsGrad=" 
                         << (node.a < graph.nodes.size() ? graph.nodes[node.a].needsGradient : 0) 
                         << "), node.b=" << node.b << " (needsGrad=" 
                         << (node.b < graph.nodes.size() ? graph.nodes[node.b].needsGradient : 0) 
                         << ")" << std::endl;
            }
            
            instructionSet->emitLoadGradient(a, 0, nodeId);  // Load gradient into XMM0
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitAccumulateGradient(a, 0, node.a);  // Accumulate XMM0 to gradient[node.a]
                if (config && config->printGradientDebug) {
                    std::cout << "      Accumulating gradient to node.a (" << node.a << ")" << std::endl;
                }
            }
            if (node.b < graph.nodes.size() && graph.nodes[node.b].needsGradient) {
                instructionSet->emitAccumulateGradient(a, 0, node.b);  // Accumulate XMM0 to gradient[node.b]
                if (config && config->printGradientDebug) {
                    std::cout << "      Accumulating gradient to node.b (" << node.b << ")" << std::endl;
                }
            }
            break;
            
        case OpCode::Sub:
        {
            // grad[a] += grad[nodeId]
            // grad[b] -= grad[nodeId]
            instructionSet->emitLoadGradient(a, 0, nodeId);  // Load gradient into XMM0
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitAccumulateGradient(a, 0, node.a);
            }
            // For subtraction, negate before accumulating to b
            instructionSet->emitMove(a, 1, 0);  // Copy XMM0 to XMM1
            instructionSet->emitNeg(a, 1, 2);   // Negate XMM1, using XMM2 as temp
            if (node.b < graph.nodes.size() && graph.nodes[node.b].needsGradient) {
                instructionSet->emitAccumulateGradient(a, 1, node.b);
            }
            break;
        }
            
        case OpCode::Mul:
            // grad[a] += grad[nodeId] * value[b]
            // grad[b] += grad[nodeId] * value[a]
            instructionSet->emitLoadGradient(a, 0, nodeId);
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitLoadValueForGradient(a, 1, node.b, graph, &constantMap, constPoolLabel);
                instructionSet->emitMul(a, 1, 0);  // xmm1 = grad[nodeId] * value[b]
                instructionSet->emitAccumulateGradient(a, 1, node.a);
            }
            
            if (node.b < graph.nodes.size() && graph.nodes[node.b].needsGradient) {
                instructionSet->emitLoadValueForGradient(a, 1, node.a, graph, &constantMap, constPoolLabel);
                instructionSet->emitMul(a, 1, 0);  // xmm1 = grad[nodeId] * value[a]
                instructionSet->emitAccumulateGradient(a, 1, node.b);
            }
            break;
            
        case OpCode::Div:
        {
            // grad[a] += grad[nodeId] / value[b]
            // grad[b] -= grad[nodeId] * value[a] / (value[b] * value[b])
            
            if (config && config->printGradientDebug) {
                std::cout << "    Div: node.a=" << node.a << " (needsGrad=" 
                         << (node.a < graph.nodes.size() ? graph.nodes[node.a].needsGradient : 0) 
                         << "), node.b=" << node.b << " (needsGrad=" 
                         << (node.b < graph.nodes.size() ? graph.nodes[node.b].needsGradient : 0) 
                         << ")" << std::endl;
            }
            
            instructionSet->emitLoadGradient(a, 0, nodeId);
            instructionSet->emitLoadValueForGradient(a, 1, node.b, graph, &constantMap, constPoolLabel);
            
            // Gradient for a
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitMove(a, 2, 0);  // Copy grad[nodeId] to XMM2
                instructionSet->emitDiv(a, 2, 1);   // xmm2 = grad[nodeId] / value[b]
                instructionSet->emitAccumulateGradient(a, 2, node.a);
                
                if (config && config->printGradientDebug) {
                    std::cout << "      Accumulating gradient to node.a (" << node.a << ")" << std::endl;
                }
            } else if (config && config->printGradientDebug) {
                std::cout << "      NOT accumulating to node.a (" << node.a 
                         << ") - needsGradient=false or out of bounds" << std::endl;
            }
            
            // Gradient for b
            if (node.b < graph.nodes.size() && graph.nodes[node.b].needsGradient) {
                instructionSet->emitLoadValueForGradient(a, 2, node.a, graph, &constantMap, constPoolLabel);  // Load value[a] (numerator)
                instructionSet->emitMul(a, 2, 0);    // xmm2 = grad[nodeId] * value[a]
                instructionSet->emitMul(a, 1, 1);    // xmm1 = value[b] * value[b]
                instructionSet->emitDiv(a, 2, 1);    // xmm2 = grad[nodeId] * value[a] / (value[b]^2)
                // Negate and accumulate
                instructionSet->emitNeg(a, 2, 3);    // Negate xmm2, using xmm3 as temp
                instructionSet->emitAccumulateGradient(a, 2, node.b);
                
                if (config && config->printGradientDebug) {
                    std::cout << "      Accumulating gradient to node.b (" << node.b << ")" << std::endl;
                }
            } else if (config && config->printGradientDebug) {
                std::cout << "      NOT accumulating to node.b (" << node.b 
                         << ") - needsGradient=false or out of bounds" << std::endl;
            }
            break;
        }
            
        case OpCode::Neg:
        {
            // grad[a] -= grad[nodeId]
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitLoadGradient(a, 0, nodeId);
                instructionSet->emitNeg(a, 0, 1);  // Negate xmm0, using xmm1 as temp
                instructionSet->emitAccumulateGradient(a, 0, node.a);
            }
            break;
        }

        case OpCode::Abs:
            // grad[a] += sign(value[a]) * grad[nodeId]
            // sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
            // Using bit manipulation approach (same as forward stitcher for Abs)
            // This works correctly for both SSE2 and AVX2
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitLoadValueForGradient(a, 1, node.a, graph, &constantMap, constPoolLabel);
                instructionSet->emitLoadGradient(a, 0, nodeId);

                // Compute sign(x) using bit manipulation, with sign(0) = 0
                // Approach: sign(x) = x / |x| for x != 0, and 0 for x == 0
                // This naturally handles the x == 0 case (0/0 = NaN, but we avoid it)

                // Step 1: Compute |x| using bit manipulation (clear sign bit)
                instructionSet->emitCreateAllOnes(a, 2);      // reg 2 = all 1s
                instructionSet->emitShiftRight(a, 2, 1);      // reg 2 = 0x7FFFFFFFFFFFFFFF (sign bit cleared)
                instructionSet->emitMove(a, 3, 1);            // reg 3 = value[a]
                instructionSet->emitAndPD(a, 3, 2);           // reg 3 = |value[a]|

                // Step 2: Compute sign = x / |x|, but handle x == 0 specially
                // Add a tiny epsilon to |x| to avoid division by zero
                // This gives sign(0) ≈ 0 since 0 / epsilon ≈ 0
                instructionSet->emitLoadImmediate(a, 4, 1e-300); // reg 4 = tiny epsilon
                instructionSet->emitAdd(a, 3, 4);             // reg 3 = |x| + epsilon
                instructionSet->emitMove(a, 5, 1);            // reg 5 = x
                instructionSet->emitDiv(a, 5, 3);             // reg 5 = x / (|x| + epsilon) ≈ sign(x)

                // Step 3: Round to exactly -1, 0, or +1 by truncating tiny values
                // For normal values: x / (|x| + epsilon) ≈ ±1 (very close)
                // For x = 0: 0 / epsilon = 0 (exactly)
                // We can use this directly since the error is negligible

                // Multiply gradient by sign
                instructionSet->emitMul(a, 0, 5);  // grad[nodeId] * sign(value[a])
                instructionSet->emitAccumulateGradient(a, 0, node.a);
            }
            break;
            
        case OpCode::Square:
            // grad[a] += 2 * value[a] * grad[nodeId]
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitLoadGradient(a, 0, nodeId);
                instructionSet->emitLoadValueForGradient(a, 1, node.a, graph, &constantMap, constPoolLabel);
                instructionSet->emitAdd(a, 1, 1);  // xmm1 = 2 * value[a]
                instructionSet->emitMul(a, 1, 0);  // xmm1 = 2 * value[a] * grad[nodeId]
                instructionSet->emitAccumulateGradient(a, 1, node.a);
            }
            break;
            
        case OpCode::Sqrt:
            // grad[a] += grad[nodeId] / (2 * value[nodeId])
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitLoadGradient(a, 0, nodeId);
                instructionSet->emitLoadValueForGradient(a, 1, nodeId, graph, &constantMap, constPoolLabel);  // Load sqrt result
                instructionSet->emitAdd(a, 1, 1);    // xmm1 = 2 * sqrt(x)
                instructionSet->emitDiv(a, 0, 1);    // xmm0 = grad / (2 * sqrt(x))
                instructionSet->emitAccumulateGradient(a, 0, node.a);
            }
            break;
            
        case OpCode::Exp:
            // grad[a] += grad[nodeId] * value[nodeId]
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitLoadGradient(a, 0, nodeId);
                instructionSet->emitLoadValueForGradient(a, 1, nodeId, graph, &constantMap, constPoolLabel);  // exp(x) result
                instructionSet->emitMul(a, 0, 1);
                instructionSet->emitAccumulateGradient(a, 0, node.a);
            }
            break;
            
        case OpCode::Log:
            // grad[a] += grad[nodeId] / value[a]
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitLoadGradient(a, 0, nodeId);
                instructionSet->emitLoadValueForGradient(a, 1, node.a, graph, &constantMap, constPoolLabel);
                instructionSet->emitDiv(a, 0, 1);
                instructionSet->emitAccumulateGradient(a, 0, node.a);
            }
            break;
            
        case OpCode::Pow:
        {
            // pow(x, y) = x^y
            // grad[x] += grad[nodeId] * y * x^(y-1) = grad[nodeId] * y * pow(x, y-1)
            // grad[y] += grad[nodeId] * x^y * log(x) = grad[nodeId] * pow(x, y) * log(x)
            
            // Load common values
            instructionSet->emitLoadGradient(a, 0, nodeId);  // grad[nodeId] in reg 0
            instructionSet->emitLoadValueForGradient(a, 1, node.a, graph, &constantMap, constPoolLabel);  // x in reg 1
            instructionSet->emitLoadValueForGradient(a, 2, node.b, graph, &constantMap, constPoolLabel);  // y in reg 2
            
            // Gradient for x (base): grad[nodeId] * y * x^(y-1)
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                // Compute y-1
                instructionSet->emitLoadImmediate(a, 3, 1.0);
                instructionSet->emitMove(a, 4, 2);  // Copy y to reg 4
                instructionSet->emitSub(a, 4, 3);   // reg 4 = y - 1
                
                // Compute x^(y-1) using pow
                instructionSet->emitPow(a, 5, 1, 4, regState);  // reg 5 = pow(x, y-1)
                
                // Reload values after pow call (which may have clobbered registers)
                instructionSet->emitLoadGradient(a, 0, nodeId);
                instructionSet->emitLoadValueForGradient(a, 2, node.b, graph, &constantMap, constPoolLabel);  // Reload y
                
                // Compute grad[nodeId] * y * x^(y-1)
                instructionSet->emitMul(a, 5, 2);  // reg 5 = x^(y-1) * y
                instructionSet->emitMul(a, 5, 0);  // reg 5 = grad[nodeId] * y * x^(y-1)
                
                instructionSet->emitAccumulateGradient(a, 5, node.a);
            }
            
            // Gradient for y (exponent): grad[nodeId] * x^y * log(x)
            if (node.b < graph.nodes.size() && graph.nodes[node.b].needsGradient) {
                // Reload base value
                instructionSet->emitLoadValueForGradient(a, 1, node.a, graph, &constantMap, constPoolLabel);  // x in reg 1
                
                // Compute log(x)
                instructionSet->emitLog(a, 6, 1, regState);  // reg 6 = log(x)
                
                // Reload gradient and result after log call
                instructionSet->emitLoadGradient(a, 0, nodeId);
                instructionSet->emitLoadValueForGradient(a, 7, nodeId, graph, &constantMap, constPoolLabel);  // x^y (result) in reg 7
                
                // Compute grad[nodeId] * x^y * log(x)
                instructionSet->emitMul(a, 7, 6);  // reg 7 = x^y * log(x)
                instructionSet->emitMul(a, 7, 0);  // reg 7 = grad[nodeId] * x^y * log(x)
                
                instructionSet->emitAccumulateGradient(a, 7, node.b);
            }
            break;
        }
            
        case OpCode::Sin:
            // grad[a] += grad[nodeId] * cos(value[a])
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitLoadValueForGradient(a, 1, node.a, graph, &constantMap, constPoolLabel);
                instructionSet->emitCos(a, 2, 1, regState);  // xmm2 = cos(value[a])
                instructionSet->emitLoadGradient(a, 0, nodeId);  // Load gradient after cos call
                instructionSet->emitMul(a, 0, 2);  // xmm0 = grad[nodeId] * cos(value[a])
                instructionSet->emitAccumulateGradient(a, 0, node.a);
            }
            break;
            
        case OpCode::Cos:
        {
            // grad[a] -= grad[nodeId] * sin(value[a])
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitLoadValueForGradient(a, 1, node.a, graph, &constantMap, constPoolLabel);
                instructionSet->emitSin(a, 2, 1, regState);  // xmm2 = sin(value[a])
                instructionSet->emitLoadGradient(a, 0, nodeId);  // Load gradient after sin call
                instructionSet->emitMul(a, 0, 2);  // xmm0 = grad[nodeId] * sin(value[a])
                instructionSet->emitNeg(a, 0, 3);  // xmm0 = -grad[nodeId] * sin(value[a]), using xmm3 as temp
                instructionSet->emitAccumulateGradient(a, 0, node.a);
            }
            break;
        }
            
        case OpCode::Tan:
        {
            // grad[a] += grad[nodeId] * sec²(value[a])
            // sec²(x) = 1 + tan²(x)
            // We can get tan(x) from the already computed value[nodeId]
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                // Load the already computed tan(x) from forward pass
                instructionSet->emitLoadValueForGradient(a, 1, nodeId, graph, &constantMap, constPoolLabel);

                // Compute tan²(x)
                instructionSet->emitMove(a, 2, 1);  // reg2 = tan(x)
                instructionSet->emitMul(a, 2, 1);   // reg2 = tan²(x)

                // Compute 1 + tan²(x) = sec²(x)
                instructionSet->emitLoadImmediate(a, 3, 1.0);
                instructionSet->emitAdd(a, 2, 3);  // reg2 = 1 + tan²(x) = sec²(x)

                // Load gradient of current node
                instructionSet->emitLoadGradient(a, 0, nodeId);

                // Multiply gradient by sec²(x)
                instructionSet->emitMul(a, 0, 2);  // reg0 = grad[nodeId] * sec²(x)

                // Accumulate to grad[a]
                instructionSet->emitAccumulateGradient(a, 0, node.a);
            }
            break;
        }
            
        case OpCode::If:
        case OpCode::IntIf:
        {
            // SPECIAL HANDLING FOR CONDITIONALS
            // The gradient must flow only through the branch that was taken.
            // We check values[node.a] to determine which branch was taken:
            // - Comparison ops store 0.0 for false, 1.0 for true
            // - The If operation used this to select the branch
            
            // Load condition value from forward pass
            instructionSet->emitLoadValueForGradient(a, 0, node.a, graph, &constantMap, constPoolLabel);  // 0.0 or 1.0
            instructionSet->emitLoadGradient(a, 1, nodeId);  // gradient to propagate
            
            // Gradient for true branch: condition * grad[result]
            instructionSet->emitMove(a, 2, 0);
            instructionSet->emitMul(a, 2, 1);
            instructionSet->emitAccumulateGradient(a, 2, node.b);
            
            // Gradient for false branch: (1 - condition) * grad[result]
            instructionSet->emitLoadImmediate(a, 2, 1.0);  // Load 1.0 into register 2
            instructionSet->emitSub(a, 2, 0);  // 1.0 - condition  
            instructionSet->emitMul(a, 2, 1);
            instructionSet->emitAccumulateGradient(a, 2, node.c);
            break;
        }
            
        case OpCode::Min:
            // grad flows through the minimum value
            // grad[a] += (value[a] <= value[b]) ? grad[nodeId] : 0
            // grad[b] += (value[b] < value[a]) ? grad[nodeId] : 0
            {
                instructionSet->emitLoadValueForGradient(a, 0, node.a, graph, &constantMap, constPoolLabel);
                instructionSet->emitLoadValueForGradient(a, 1, node.b, graph, &constantMap, constPoolLabel);
                instructionSet->emitLoadGradient(a, 2, nodeId);
                
                // Compare and create mask: a <= b
                instructionSet->emitCmpLE(a, 3, 0, 1, regState);  // Compare a with b, store result in reg 3
                instructionSet->emitAndPD(a, 2, 3);  // Mask gradient with comparison result
                instructionSet->emitAccumulateGradient(a, 2, node.a);
                
                // For b: b < a (opposite condition)
                instructionSet->emitLoadValueForGradient(a, 0, node.a, graph, &constantMap, constPoolLabel);
                instructionSet->emitLoadValueForGradient(a, 1, node.b, graph, &constantMap, constPoolLabel);
                instructionSet->emitLoadGradient(a, 2, nodeId);
                instructionSet->emitCmpLT(a, 3, 1, 0, regState);  // Compare b with a, store result in reg 3
                instructionSet->emitAndPD(a, 2, 3);
                instructionSet->emitAccumulateGradient(a, 2, node.b);
            }
            break;
            
        case OpCode::Max:
            // grad flows through the maximum value
            // grad[a] += (value[a] >= value[b]) ? grad[nodeId] : 0
            // grad[b] += (value[b] > value[a]) ? grad[nodeId] : 0
            {
                instructionSet->emitLoadValueForGradient(a, 0, node.a, graph, &constantMap, constPoolLabel);
                instructionSet->emitLoadValueForGradient(a, 1, node.b, graph, &constantMap, constPoolLabel);
                instructionSet->emitLoadGradient(a, 2, nodeId);
                
                // Compare and create mask for a: a >= b
                instructionSet->emitCmpGE(a, 3, 0, 1, regState);  // Compare a with b, store result in reg 3
                instructionSet->emitAndPD(a, 2, 3);  // Mask gradient with comparison result
                instructionSet->emitAccumulateGradient(a, 2, node.a);
                
                // For b: b > a (opposite condition)
                instructionSet->emitLoadValueForGradient(a, 0, node.a, graph, &constantMap, constPoolLabel);
                instructionSet->emitLoadValueForGradient(a, 1, node.b, graph, &constantMap, constPoolLabel);
                instructionSet->emitLoadGradient(a, 2, nodeId);
                instructionSet->emitCmpGT(a, 3, 1, 0, regState);  // Compare b with a, store result in reg 3
                instructionSet->emitAndPD(a, 2, 3);
                instructionSet->emitAccumulateGradient(a, 2, node.b);
            }
            break;
            
        case OpCode::Recip:
            // grad[a] += -grad[nodeId] / (value[a] * value[a])
            // Reciprocal: f(x) = 1/x, f'(x) = -1/x²
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitLoadGradient(a, 0, nodeId);
                instructionSet->emitLoadValueForGradient(a, 1, node.a, graph, &constantMap, constPoolLabel);
                instructionSet->emitMul(a, 1, 1);  // xmm1 = value[a] * value[a]
                instructionSet->emitDiv(a, 0, 1);  // xmm0 = grad[nodeId] / (value[a]²)
                instructionSet->emitNeg(a, 0, 2);  // xmm0 = -grad[nodeId] / (value[a]²), using xmm2 as temp
                instructionSet->emitAccumulateGradient(a, 0, node.a);
            }
            break;
            
        case OpCode::Mod:
            // grad[a] += grad[nodeId]  (derivative w.r.t. dividend is 1)
            // grad[b] += -floor(value[a]/value[b]) * grad[nodeId]  (derivative w.r.t. divisor)
            // Note: This is approximate due to discontinuities in modulo
            if (node.a < graph.nodes.size() && graph.nodes[node.a].needsGradient) {
                instructionSet->emitLoadGradient(a, 0, nodeId);
                instructionSet->emitAccumulateGradient(a, 0, node.a);  // grad[a] += grad[nodeId]
            }
            
            if (node.b < graph.nodes.size() && graph.nodes[node.b].needsGradient) {
                // For now, don't propagate gradient to divisor due to complexity and discontinuity
                // This is a reasonable approximation for most use cases
                // A more accurate implementation would require floor(a/b) computation
            }
            break;
            
        // Comparison operations don't backpropagate gradients
        case OpCode::CmpLT:
        case OpCode::CmpLE:
        case OpCode::CmpGT:
        case OpCode::CmpGE:
        case OpCode::CmpEQ:
        case OpCode::CmpNE:
            // No gradient flow through comparisons
            break;
            
        default:
            // Skip unhandled operations
            break;
    }
}

void GradientStitcher::stitchGradientPass(
    x86::Assembler& a,
    const Graph& graph,
    const std::unordered_map<NodeId, ForgeEngine::ConstantInfo>& constantMap,
    const Label& constPoolLabel,
    IRegisterAllocator& regState,
    IInstructionSet* instructionSet,
    const CompilerConfig* config) {
    
    // First, set gradient of output nodes to 1.0
    for (NodeId outputNode : graph.outputs) {
        if (outputNode < graph.nodes.size() && graph.nodes[outputNode].needsGradient) {
            instructionSet->emitLoadImmediate(a, 0, 1.0);  // Load 1.0 into register 0
            instructionSet->emitStoreGradient(a, 0, outputNode);
            
            if (config && config->printGradientDebug) {
                std::cout << "  Setting initial gradient for output node " << outputNode << " to 1.0" << std::endl;
            }
        }
    }
    
    // Process nodes in reverse topological order (backward pass)
    // For simplicity, we iterate backwards through all nodes
    // A more efficient implementation would use proper topological sorting
    for (int64_t i = graph.nodes.size() - 1; i >= 0; --i) {
        const Node& node = graph.nodes[i];
        NodeId nodeId = static_cast<NodeId>(i);
        
        // Skip if this node doesn't need gradient
        if (!node.needsGradient) {
            if (config && config->printGradientDebug) {
                std::cout << "  Skipping node " << nodeId << " (needsGradient=false)" << std::endl;
            }
            continue;
        }
        
        // Skip dead nodes
        if (node.isDead) {
            if (config && config->printGradientDebug) {
                std::cout << "  Skipping node " << nodeId << " (isDead=true)" << std::endl;
            }
            continue;
        }
        
        if (config && config->printGradientDebug) {
            std::cout << "  Processing gradient for node " << nodeId << " (" << opName(node.op) << ")" << std::endl;
        }
        
        // Generate gradient operation
        generateGradientOperation(a, node, nodeId, regState, graph, constantMap, constPoolLabel, instructionSet, config);
    }
}

} // namespace forge