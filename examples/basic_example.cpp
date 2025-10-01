// Basic example demonstrating forward computation with Forge
// This matches the first example in the README

#include <forge/forge.h>
#include <iostream>

int main() {
    using namespace forge;
    
    // Create a computation graph
    core::ComputationGraph graph;
    
    // Add inputs
    auto x = graph.addInput();
    auto y = graph.addInput();
    
    // Build computation: z = x*y + sin(x)
    core::Node mulNode;
    mulNode.op = core::OpCode::Mul;
    mulNode.a = x;
    mulNode.b = y;
    auto xy = graph.addNode(mulNode);
    
    core::Node sinNode;
    sinNode.op = core::OpCode::Sin;
    sinNode.a = x;
    auto sinx = graph.addNode(sinNode);
    
    core::Node addNode;
    addNode.op = core::OpCode::Add;
    addNode.a = xy;
    addNode.b = sinx;
    auto result = graph.addNode(addNode);
    
    // Mark output
    graph.markOutput(result);
    
    // Compile to machine code
    compiler::ForgeEngine engine;
    auto kernel = engine.compile(graph);
    
    // Create execution buffer
    auto buffer = runtime::NodeBufferFactory::create(graph, *kernel);
    
    // Set inputs and execute
    buffer->setValue(x, 1.5);  // x = 1.5
    buffer->setValue(y, 2.0);  // y = 2.0
    kernel->execute(*buffer);
    
    // Get result
    double output = buffer->getValue(result);
    std::cout << "f(1.5, 2.0) = " << output << std::endl;
    // Output: f(1.5, 2.0) = 3.99749 (which is 1.5*2.0 + sin(1.5))
    
    return 0;
}