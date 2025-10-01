#pragma once

#include <forge/core/computation_graph.h>
#include <string>

namespace forge {
namespace tools {

/**
 * Service for serializing and deserializing computation graphs to/from JSON format.
 * This allows saving and loading computation graphs for testing and example purposes.
 */
class GraphSerializationService {
public:
    /**
     * Serialize a computation graph to JSON string.
     * @param graph The graph to serialize
     * @return JSON string representation
     */
    static std::string toJson(const core::ComputationGraph& graph);
    
    /**
     * Deserialize a computation graph from JSON string.
     * @param json JSON string representation
     * @return Reconstructed computation graph
     */
    static core::ComputationGraph fromJson(const std::string& json);
    
    /**
     * Save a computation graph to a JSON file.
     * @param graph The graph to save
     * @param filename Path to output file
     * @return true if successful, false otherwise
     */
    static bool saveToFile(const core::ComputationGraph& graph, const std::string& filename);
    
    /**
     * Load a computation graph from a JSON file.
     * @param filename Path to input file
     * @return Loaded computation graph
     */
    static core::ComputationGraph loadFromFile(const std::string& filename);

private:
    // Helper functions for string conversion
    static std::string opCodeToString(core::OpCode op);
    static core::OpCode stringToOpCode(const std::string& str);
};

} // namespace tools
} // namespace forge