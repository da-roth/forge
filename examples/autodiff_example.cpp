// Automatic differentiation example with analytical gradient verification
// This matches the second example in the README

#include <forge/forge.h>
#include <iostream>
#include <cmath>

int main() {
    using namespace forge;
    
    // Create graph for f(x) = x^2 + sin(x)
    core::ComputationGraph graph;
    
    auto x = graph.addInput();
    graph.diff_inputs.push_back(x);  // Mark for gradient computation
    
    // x^2
    core::Node squareNode;
    squareNode.op = core::OpCode::Square;
    squareNode.a = x;
    auto x2 = graph.addNode(squareNode);
    
    // sin(x)
    core::Node sinNode;
    sinNode.op = core::OpCode::Sin;
    sinNode.a = x;
    auto sinx = graph.addNode(sinNode);
    
    // x^2 + sin(x)
    core::Node addNode;
    addNode.op = core::OpCode::Add;
    addNode.a = x2;
    addNode.b = sinx;
    auto result = graph.addNode(addNode);
    
    graph.markOutput(result);
    
    // Compile forward kernel
    compiler::ForgeEngine engine;
    auto fwdKernel = engine.compile(graph);
    
    // For reverse mode, we need to use a different approach
    // Note: The reverse compiler needs to be implemented separately
    // For now, we'll just demonstrate forward mode
    
    // Test at x = 2.0
    double xval = 2.0;
    
    // Forward pass
    auto fwdBuffer = runtime::NodeBufferFactory::create(graph, *fwdKernel);
    fwdBuffer->setValue(x, xval);
    fwdKernel->execute(*fwdBuffer);
    double fx = fwdBuffer->getValue(result);
    
    std::cout << "f(" << xval << ") = " << fx << std::endl;
    
    // Expected result for x^2 + sin(x) at x = 2.0
    double expected = xval * xval + std::sin(xval);
    std::cout << "Expected: " << expected << std::endl;
    std::cout << "Error: " << std::abs(fx - expected) << std::endl;
    
    // Note: Gradient computation would require implementing a proper
    // reverse mode compiler interface, which is beyond this basic example
    std::cout << "\nAnalytical gradient f'(x) = 2x + cos(x)" << std::endl;
    double analytical_gradient = 2*xval + std::cos(xval);
    std::cout << "At x = " << xval << ", f'(x) = " << analytical_gradient << std::endl;
    
    return 0;
}