#pragma once

#include "../../src/graph/graph.hpp"
#include <string>

namespace forge {

/**
 * @brief Serialize a Graph to JSON format
 *
 * Produces a JSON representation containing:
 * - All nodes with their complete state (op, operands, flags, etc.)
 * - Constant pool values
 * - Output node IDs
 * - Differentiation input node IDs
 *
 * @param graph The graph to serialize
 * @param pretty If true, output formatted JSON with indentation
 * @return JSON string representation of the graph
 */
std::string serializeGraphToJson(const Graph& graph, bool pretty = true);

/**
 * @brief Deserialize a Graph from JSON format
 *
 * Reconstructs a Graph from JSON produced by serializeGraphToJson.
 * Validates the JSON structure and throws std::runtime_error on failure.
 *
 * @param json JSON string to parse
 * @return Reconstructed graph
 * @throws std::runtime_error if JSON is invalid or malformed
 */
Graph deserializeGraphFromJson(const std::string& json);

/**
 * @brief Save a Graph to a JSON file
 *
 * @param graph The graph to save
 * @param filename Path to output file
 * @param pretty If true, output formatted JSON with indentation
 * @return true if successful, false on I/O error
 */
bool saveGraphToFile(const Graph& graph, const std::string& filename, bool pretty = true);

/**
 * @brief Load a Graph from a JSON file
 *
 * @param filename Path to input file
 * @return Loaded graph
 * @throws std::runtime_error if file cannot be read or JSON is invalid
 */
Graph loadGraphFromFile(const std::string& filename);

} // namespace forge
