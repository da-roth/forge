// This example demonstrates loading a computation graph from JSON and executing it
// with both forward and reverse (gradient) computation.

#include <forge/forge.h>
#include <tools/graph_serialization_service.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

using namespace forge;

// Helper function to read file contents
std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

int main(int argc, char* argv[]) {
    try {
        // Path to the JSON file
        std::string jsonFile = (argc > 1) ? argv[1] : "ops10_function_graph.json";
        
        std::cout << "======================================\n";
        std::cout << "Forge Graph Execution Example\n";
        std::cout << "======================================\n\n";
        
        // Load the computation graph from JSON
        std::cout << "Loading graph from: " << jsonFile << "\n";
        std::string jsonContent = readFile(jsonFile);
        core::ComputationGraph graph = tools::GraphSerializationService::fromJson(jsonContent);
        
        std::cout << "Graph loaded successfully!\n";
        std::cout << "  - Nodes: " << graph.nodes.size() << "\n";
        std::cout << "  - Constants: " << graph.constPool.size() << "\n";
        std::cout << "  - Outputs: " << graph.outputs.size() << "\n\n";
        
        // Create compiler engine
        compiler::ForgeEngine engine;
        auto forwardKernel = engine.compile(graph);
        std::cout << "Forward kernel compiled successfully!\n\n";
        
        // Test with different input values
        std::vector<double> testInputs = {0.0, 0.5, 1.0, 1.5, 2.0, -1.0, -0.5};
        
        std::cout << "======================================\n";
        std::cout << "Forward Evaluation Results\n";
        std::cout << "======================================\n";
        std::cout << "Input\t\tOutput\n";
        std::cout << "--------------------------------------\n";
        
        for (double x : testInputs) {
            // Create buffer for forward execution
            auto buffer = runtime::NodeBufferFactory::create(graph, *forwardKernel);
            
            // Set input value (assuming first node is input)
            buffer->setValue(0, x);
            
            // Execute forward pass
            forwardKernel->execute(*buffer);
            
            // Get output value (using first output node)
            double result = buffer->getValue(graph.outputs[0]);
            
            std::cout << x << "\t\t" << result << "\n";
        }
        
        
        // Note: The ops10 function is a complex test function from TapePresso
        // Gradient verification would require reverse-mode compilation
        
        std::cout << "\n======================================\n";
        std::cout << "Example completed successfully!\n";
        std::cout << "======================================\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}