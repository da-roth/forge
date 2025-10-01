// Example demonstrating loading and executing graphs from JSON
// This matches the third example in the README

#include <forge/forge.h>
#include <tools/graph_serialization_service.h>
#include <iostream>
#include <fstream>

int main() {
    using namespace forge;
    
    try {
        // Load graph from JSON file
        std::ifstream file("ops10_function_graph.json");
        if (!file.is_open()) {
            std::cerr << "Error: Could not open ops10_function_graph.json" << std::endl;
            std::cerr << "Make sure the file exists in the current directory" << std::endl;
            return 1;
        }
        
        std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
        
        core::ComputationGraph graph = 
            tools::GraphSerializationService::fromJson(json);
        
        std::cout << "Graph loaded successfully!" << std::endl;
        std::cout << "  Nodes: " << graph.nodes.size() << std::endl;
        std::cout << "  Constants: " << graph.constPool.size() << std::endl;
        std::cout << "  Outputs: " << graph.outputs.size() << std::endl << std::endl;
        
        // Compile and execute as usual
        compiler::ForgeEngine engine;
        auto kernel = engine.compile(graph);
        
        auto buffer = runtime::NodeBufferFactory::create(graph, *kernel);
        buffer->setValue(0, 3.14);  // Set first input
        kernel->execute(*buffer);
        
        // Get output from first output node
        double result = buffer->getValue(graph.outputs[0]);
        std::cout << "f(3.14) = " << result << std::endl;
        
        // Try a few more values
        std::cout << "\nTesting with different inputs:" << std::endl;
        double test_values[] = {0.0, 1.0, 2.0, -0.5};
        
        for (double val : test_values) {
            buffer->setValue(0, val);
            kernel->execute(*buffer);
            result = buffer->getValue(graph.outputs[0]);
            std::cout << "f(" << val << ") = " << result << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}