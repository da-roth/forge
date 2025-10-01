#include "graph_serialization_service.h"
#include <sstream>
#include <fstream>
#include <iostream>

namespace forge {
namespace tools {

std::string GraphSerializationService::opCodeToString(core::OpCode op) {
    switch(op) {
        case core::OpCode::Input: return "Input";
        case core::OpCode::Constant: return "Constant";
        case core::OpCode::Add: return "Add";
        case core::OpCode::Sub: return "Sub";
        case core::OpCode::Mul: return "Mul";
        case core::OpCode::Div: return "Div";
        case core::OpCode::Neg: return "Neg";
        case core::OpCode::Abs: return "Abs";
        case core::OpCode::Square: return "Square";
        case core::OpCode::Recip: return "Recip";
        case core::OpCode::Mod: return "Mod";
        case core::OpCode::Exp: return "Exp";
        case core::OpCode::Log: return "Log";
        case core::OpCode::Sqrt: return "Sqrt";
        case core::OpCode::Pow: return "Pow";
        case core::OpCode::Sin: return "Sin";
        case core::OpCode::Cos: return "Cos";
        case core::OpCode::Tan: return "Tan";
        case core::OpCode::Min: return "Min";
        case core::OpCode::Max: return "Max";
        case core::OpCode::If: return "If";
        case core::OpCode::CmpLT: return "CmpLT";
        case core::OpCode::CmpLE: return "CmpLE";
        case core::OpCode::CmpGT: return "CmpGT";
        case core::OpCode::CmpGE: return "CmpGE";
        case core::OpCode::CmpEQ: return "CmpEQ";
        case core::OpCode::CmpNE: return "CmpNE";
        default: return "Unknown";
    }
}

core::OpCode GraphSerializationService::stringToOpCode(const std::string& str) {
    if (str == "Input") return core::OpCode::Input;
    if (str == "Constant") return core::OpCode::Constant;
    if (str == "Add") return core::OpCode::Add;
    if (str == "Sub") return core::OpCode::Sub;
    if (str == "Mul") return core::OpCode::Mul;
    if (str == "Div") return core::OpCode::Div;
    if (str == "Neg") return core::OpCode::Neg;
    if (str == "Abs") return core::OpCode::Abs;
    if (str == "Square") return core::OpCode::Square;
    if (str == "Recip") return core::OpCode::Recip;
    if (str == "Mod") return core::OpCode::Mod;
    if (str == "Exp") return core::OpCode::Exp;
    if (str == "Log") return core::OpCode::Log;
    if (str == "Sqrt") return core::OpCode::Sqrt;
    if (str == "Pow") return core::OpCode::Pow;
    if (str == "Sin") return core::OpCode::Sin;
    if (str == "Cos") return core::OpCode::Cos;
    if (str == "Tan") return core::OpCode::Tan;
    if (str == "Min") return core::OpCode::Min;
    if (str == "Max") return core::OpCode::Max;
    if (str == "If") return core::OpCode::If;
    if (str == "CmpLT") return core::OpCode::CmpLT;
    if (str == "CmpLE") return core::OpCode::CmpLE;
    if (str == "CmpGT") return core::OpCode::CmpGT;
    if (str == "CmpGE") return core::OpCode::CmpGE;
    if (str == "CmpEQ") return core::OpCode::CmpEQ;
    if (str == "CmpNE") return core::OpCode::CmpNE;
    return core::OpCode::Add; // Default fallback
}

std::string GraphSerializationService::toJson(const core::ComputationGraph& graph) {
    std::ostringstream json;
    json << "{\n";
    
    // Constants
    json << "  \"constants\": [\n";
    for (size_t i = 0; i < graph.constPool.size(); ++i) {
        json << "    " << graph.constPool[i];
        if (i < graph.constPool.size() - 1) json << ",";
        json << "\n";
    }
    json << "  ],\n";
    
    // Nodes
    json << "  \"nodes\": [\n";
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const core::Node& node = graph.nodes[i];
        json << "    { \"op\": \"" << opCodeToString(node.op) << "\"";
        
        // Always write ALL operands for complete graph representation
        json << ", \"a\": " << node.a;
        json << ", \"b\": " << node.b;
        json << ", \"c\": " << node.c;
        
        // Keep flags and imm conditional since they're metadata
        if (node.flags != 0) json << ", \"flags\": " << node.flags;
        if (node.imm != 0.0) json << ", \"imm\": " << node.imm;
        
        json << " }";
        if (i < graph.nodes.size() - 1) json << ",";
        json << "\n";
    }
    json << "  ],\n";
    
    // Outputs
    json << "  \"outputs\": [\n";
    for (size_t i = 0; i < graph.outputs.size(); ++i) {
        json << "    " << graph.outputs[i];
        if (i < graph.outputs.size() - 1) json << ",";
        json << "\n";
    }
    json << "  ],\n";
    
    // Diff inputs
    json << "  \"diff_inputs\": [\n";
    for (size_t i = 0; i < graph.diff_inputs.size(); ++i) {
        json << "    " << graph.diff_inputs[i];
        if (i < graph.diff_inputs.size() - 1) json << ",";
        json << "\n";
    }
    json << "  ]\n";
    
    json << "}\n";
    return json.str();
}

core::ComputationGraph GraphSerializationService::fromJson(const std::string& json) {
    core::ComputationGraph graph;
    
    // Simple JSON parser - in production, use a proper JSON library like nlohmann/json
    std::istringstream stream(json);
    std::string line;
    bool in_constants = false;
    bool in_nodes = false;
    bool in_outputs = false;
    bool in_diff_inputs = false;
    
    while (std::getline(stream, line)) {
        // Remove whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
        
        if (line.empty() || line == "]" || line == "[") continue;
        
        // Skip opening and closing braces only when not in nodes/outputs/etc sections
        if (!in_nodes && !in_outputs && !in_diff_inputs && !in_constants && (line[0] == '{' || line[0] == '}')) continue;
        
        if (line.find("\"constants\"") != std::string::npos) {
            in_constants = true;
            in_nodes = false;
            in_outputs = false;
            in_diff_inputs = false;
            continue;
        }
        
        if (line.find("\"nodes\"") != std::string::npos) {
            in_constants = false;
            in_nodes = true;
            in_outputs = false;
            in_diff_inputs = false;
            continue;
        }
        
        if (line.find("\"outputs\"") != std::string::npos) {
            in_constants = false;
            in_nodes = false;
            in_outputs = true;
            in_diff_inputs = false;
            continue;
        }
        
        if (line.find("\"diff_inputs\"") != std::string::npos) {
            in_constants = false;
            in_nodes = false;
            in_outputs = false;
            in_diff_inputs = true;
            continue;
        }
        
        if (in_constants && line.find_first_of("0123456789.-") != std::string::npos) {
            // Parse constant value
            size_t start = line.find_first_of("0123456789.-");
            size_t end = line.find_last_of("0123456789.");
            if (start != std::string::npos && end != std::string::npos) {
                std::string numStr = line.substr(start, end - start + 1);
                if (numStr.back() == ',') numStr.pop_back();
                graph.constPool.push_back(std::stod(numStr));
            }
        }
        
        if (in_nodes && line.find("\"op\"") != std::string::npos) {
            // Parse node - all operands (a,b,c) should now always be present
            // since the saver writes them unconditionally
            core::Node node{};
            
            // Extract operation
            size_t opPos = line.find("\"op\": \"");
            if (opPos != std::string::npos) {
                size_t opStart = opPos + 7;
                size_t opEnd = line.find("\"", opStart);
                if (opEnd != std::string::npos) {
                    std::string opStr = line.substr(opStart, opEnd - opStart);
                    node.op = stringToOpCode(opStr);
                }
            }
            
            // Extract operands if present
            size_t aPos = line.find("\"a\": ");
            if (aPos != std::string::npos) {
                size_t aStart = aPos + 5;
                size_t aEnd = line.find_first_of(",}", aStart);
                if (aEnd != std::string::npos) {
                    node.a = std::stoi(line.substr(aStart, aEnd - aStart));
                }
            }
            
            size_t bPos = line.find("\"b\": ");
            if (bPos != std::string::npos) {
                size_t bStart = bPos + 5;
                size_t bEnd = line.find_first_of(",}", bStart);
                if (bEnd != std::string::npos) {
                    node.b = std::stoi(line.substr(bStart, bEnd - bStart));
                }
            }
            
            size_t cPos = line.find("\"c\": ");
            if (cPos != std::string::npos) {
                size_t cStart = cPos + 5;
                size_t cEnd = line.find_first_of(",}", cStart);
                if (cEnd != std::string::npos) {
                    node.c = std::stoi(line.substr(cStart, cEnd - cStart));
                }
            }
            
            size_t flagsPos = line.find("\"flags\": ");
            if (flagsPos != std::string::npos) {
                size_t flagsStart = flagsPos + 9;
                size_t flagsEnd = line.find_first_of(",}", flagsStart);
                if (flagsEnd != std::string::npos) {
                    node.flags = std::stoi(line.substr(flagsStart, flagsEnd - flagsStart));
                }
            }
            
            size_t immPos = line.find("\"imm\": ");
            if (immPos != std::string::npos) {
                size_t immStart = immPos + 7;
                size_t immEnd = line.find_first_of(",}", immStart);
                if (immEnd != std::string::npos) {
                    node.imm = std::stod(line.substr(immStart, immEnd - immStart));
                }
            }
            
            node.isActive = (node.op != core::OpCode::Constant);
            graph.addNode(node);
        }
        
        if ((in_outputs || in_diff_inputs) && line.find_first_of("0123456789") != std::string::npos) {
            // Parse index
            size_t start = line.find_first_of("0123456789");
            size_t end = line.find_last_of("0123456789");
            if (start != std::string::npos && end != std::string::npos) {
                std::string numStr = line.substr(start, end - start + 1);
                if (numStr.back() == ',') numStr.pop_back();
                core::NodeId nodeId = std::stoi(numStr);
                
                if (in_outputs) {
                    graph.outputs.push_back(nodeId);
                } else if (in_diff_inputs) {
                    graph.diff_inputs.push_back(nodeId);
                }
            }
        }
    }
    
    return graph;
}

bool GraphSerializationService::saveToFile(const core::ComputationGraph& graph, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << toJson(graph);
    return true;
}

core::ComputationGraph GraphSerializationService::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return core::ComputationGraph(); // Return empty graph on error
    }
    
    std::string json((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    
    return fromJson(json);
}

} // namespace tools
} // namespace forge