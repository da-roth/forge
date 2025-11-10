#include "graph_serialization.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <cctype>
#include <cmath>
#include <limits>

using json = nlohmann::json;

namespace forge {

namespace {

// Convert OpCode to string
std::string opCodeToString(OpCode op) {
    switch (op) {
        case OpCode::Input: return "Input";
        case OpCode::Constant: return "Constant";
        case OpCode::Add: return "Add";
        case OpCode::Sub: return "Sub";
        case OpCode::Mul: return "Mul";
        case OpCode::Div: return "Div";
        case OpCode::Neg: return "Neg";
        case OpCode::Abs: return "Abs";
        case OpCode::Square: return "Square";
        case OpCode::Recip: return "Recip";
        case OpCode::Mod: return "Mod";
        case OpCode::Exp: return "Exp";
        case OpCode::Log: return "Log";
        case OpCode::Sqrt: return "Sqrt";
        case OpCode::Pow: return "Pow";
        case OpCode::Sin: return "Sin";
        case OpCode::Cos: return "Cos";
        case OpCode::Tan: return "Tan";
        case OpCode::Min: return "Min";
        case OpCode::Max: return "Max";
        case OpCode::If: return "If";
        case OpCode::CmpLT: return "CmpLT";
        case OpCode::CmpLE: return "CmpLE";
        case OpCode::CmpGT: return "CmpGT";
        case OpCode::CmpGE: return "CmpGE";
        case OpCode::CmpEQ: return "CmpEQ";
        case OpCode::CmpNE: return "CmpNE";
        case OpCode::BoolConstant: return "BoolConstant";
        case OpCode::BoolAnd: return "BoolAnd";
        case OpCode::BoolOr: return "BoolOr";
        case OpCode::BoolNot: return "BoolNot";
        case OpCode::BoolEq: return "BoolEq";
        case OpCode::BoolNe: return "BoolNe";
        case OpCode::IntConstant: return "IntConstant";
        case OpCode::IntAdd: return "IntAdd";
        case OpCode::IntSub: return "IntSub";
        case OpCode::IntMul: return "IntMul";
        case OpCode::IntDiv: return "IntDiv";
        case OpCode::IntMod: return "IntMod";
        case OpCode::IntNeg: return "IntNeg";
        case OpCode::IntCmpLT: return "IntCmpLT";
        case OpCode::IntCmpLE: return "IntCmpLE";
        case OpCode::IntCmpGT: return "IntCmpGT";
        case OpCode::IntCmpGE: return "IntCmpGE";
        case OpCode::IntCmpEQ: return "IntCmpEQ";
        case OpCode::IntCmpNE: return "IntCmpNE";
        case OpCode::IntIf: return "IntIf";
        case OpCode::ArrayIndex: return "ArrayIndex";
        default: return "Unknown";
    }
}

// Convert string to OpCode
OpCode stringToOpCode(const std::string& str) {
    if (str == "Input") return OpCode::Input;
    if (str == "Constant") return OpCode::Constant;
    if (str == "Add") return OpCode::Add;
    if (str == "Sub") return OpCode::Sub;
    if (str == "Mul") return OpCode::Mul;
    if (str == "Div") return OpCode::Div;
    if (str == "Neg") return OpCode::Neg;
    if (str == "Abs") return OpCode::Abs;
    if (str == "Square") return OpCode::Square;
    if (str == "Recip") return OpCode::Recip;
    if (str == "Mod") return OpCode::Mod;
    if (str == "Exp") return OpCode::Exp;
    if (str == "Log") return OpCode::Log;
    if (str == "Sqrt") return OpCode::Sqrt;
    if (str == "Pow") return OpCode::Pow;
    if (str == "Sin") return OpCode::Sin;
    if (str == "Cos") return OpCode::Cos;
    if (str == "Tan") return OpCode::Tan;
    if (str == "Min") return OpCode::Min;
    if (str == "Max") return OpCode::Max;
    if (str == "If") return OpCode::If;
    if (str == "CmpLT") return OpCode::CmpLT;
    if (str == "CmpLE") return OpCode::CmpLE;
    if (str == "CmpGT") return OpCode::CmpGT;
    if (str == "CmpGE") return OpCode::CmpGE;
    if (str == "CmpEQ") return OpCode::CmpEQ;
    if (str == "CmpNE") return OpCode::CmpNE;
    if (str == "BoolConstant") return OpCode::BoolConstant;
    if (str == "BoolAnd") return OpCode::BoolAnd;
    if (str == "BoolOr") return OpCode::BoolOr;
    if (str == "BoolNot") return OpCode::BoolNot;
    if (str == "BoolEq") return OpCode::BoolEq;
    if (str == "BoolNe") return OpCode::BoolNe;
    if (str == "IntConstant") return OpCode::IntConstant;
    if (str == "IntAdd") return OpCode::IntAdd;
    if (str == "IntSub") return OpCode::IntSub;
    if (str == "IntMul") return OpCode::IntMul;
    if (str == "IntDiv") return OpCode::IntDiv;
    if (str == "IntMod") return OpCode::IntMod;
    if (str == "IntNeg") return OpCode::IntNeg;
    if (str == "IntCmpLT") return OpCode::IntCmpLT;
    if (str == "IntCmpLE") return OpCode::IntCmpLE;
    if (str == "IntCmpGT") return OpCode::IntCmpGT;
    if (str == "IntCmpGE") return OpCode::IntCmpGE;
    if (str == "IntCmpEQ") return OpCode::IntCmpEQ;
    if (str == "IntCmpNE") return OpCode::IntCmpNE;
    if (str == "IntIf") return OpCode::IntIf;
    if (str == "ArrayIndex") return OpCode::ArrayIndex;
    throw std::runtime_error("Unknown OpCode: " + str);
}

// Serialize a double with full precision, handling special values
std::string serializeDouble(double value) {
    if (std::isnan(value)) {
        return "\"NaN\"";
    } else if (std::isinf(value)) {
        return value > 0 ? "\"Infinity\"" : "\"-Infinity\"";
    } else {
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        return oss.str();
    }
}

// Serialize a node to JSON
std::string serializeNode(const Node& node, const std::string& indent) {
    std::ostringstream oss;
    oss << indent << "{\n";
    oss << indent << "  \"op\": \"" << opCodeToString(node.op) << "\",\n";
    oss << indent << "  \"dst\": " << node.dst << ",\n";
    oss << indent << "  \"a\": " << node.a << ",\n";
    oss << indent << "  \"b\": " << node.b << ",\n";
    oss << indent << "  \"c\": " << node.c << ",\n";
    oss << indent << "  \"flags\": " << node.flags << ",\n";
    oss << indent << "  \"imm\": " << serializeDouble(node.imm) << ",\n";
    oss << indent << "  \"isActive\": " << (node.isActive ? "true" : "false") << ",\n";
    oss << indent << "  \"isDead\": " << (node.isDead ? "true" : "false") << ",\n";
    oss << indent << "  \"needsGradient\": " << (node.needsGradient ? "true" : "false") << "\n";
    oss << indent << "}";
    return oss.str();
}

// Simple JSON parsing helpers
class JsonParser {
public:
    explicit JsonParser(const std::string& json) : json_(json), pos_(0) {
        skipWhitespace();
    }

    void expectChar(char c) {
        if (pos_ >= json_.size() || json_[pos_] != c) {
            throw std::runtime_error(std::string("Expected '") + c + "' at position " + std::to_string(pos_));
        }
        pos_++;
        skipWhitespace();
    }

    std::string parseString() {
        expectChar('"');
        size_t start = pos_;
        while (pos_ < json_.size() && json_[pos_] != '"') {
            if (json_[pos_] == '\\') pos_++; // Skip escaped char
            pos_++;
        }
        std::string result = json_.substr(start, pos_ - start);
        expectChar('"');
        return result;
    }

    double parseNumber() {
        // Check if it's a special value string
        if (peek('"')) {
            std::string str = parseString();
            if (str == "NaN") {
                return std::numeric_limits<double>::quiet_NaN();
            } else if (str == "Infinity") {
                return std::numeric_limits<double>::infinity();
            } else if (str == "-Infinity") {
                return -std::numeric_limits<double>::infinity();
            }
            throw std::runtime_error("Unknown special double value: " + str);
        }

        // Parse regular number
        size_t start = pos_;
        if (pos_ < json_.size() && (json_[pos_] == '-' || json_[pos_] == '+')) pos_++;
        while (pos_ < json_.size() && (std::isdigit(json_[pos_]) || json_[pos_] == '.' ||
               json_[pos_] == 'e' || json_[pos_] == 'E' || json_[pos_] == '-' || json_[pos_] == '+')) {
            pos_++;
        }
        std::string numStr = json_.substr(start, pos_ - start);
        skipWhitespace();
        return std::stod(numStr);
    }

    uint32_t parseUint32() {
        return static_cast<uint32_t>(parseNumber());
    }

    bool parseBool() {
        if (json_.substr(pos_, 4) == "true") {
            pos_ += 4;
            skipWhitespace();
            return true;
        } else if (json_.substr(pos_, 5) == "false") {
            pos_ += 5;
            skipWhitespace();
            return false;
        }
        throw std::runtime_error("Expected boolean at position " + std::to_string(pos_));
    }

    void skipWhitespace() {
        while (pos_ < json_.size() && std::isspace(json_[pos_])) {
            pos_++;
        }
    }

    bool peek(char c) const {
        return pos_ < json_.size() && json_[pos_] == c;
    }

private:
    std::string json_;
    size_t pos_;
};

} // anonymous namespace

std::string serializeGraphToJson(const Graph& graph, bool pretty) {
    std::ostringstream oss;
    const std::string indent1 = pretty ? "  " : "";
    const std::string indent2 = pretty ? "    " : "";
    const std::string nl = pretty ? "\n" : "";

    oss << "{" << nl;
    oss << indent1 << "\"version\": \"1.0\"," << nl;

    // Serialize nodes
    oss << indent1 << "\"nodes\": [" << nl;
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        oss << serializeNode(graph.nodes[i], indent2);
        if (i + 1 < graph.nodes.size()) oss << ",";
        oss << nl;
    }
    oss << indent1 << "]," << nl;

    // Serialize constant pool
    oss << indent1 << "\"constPool\": [";
    for (size_t i = 0; i < graph.constPool.size(); ++i) {
        oss << serializeDouble(graph.constPool[i]);
        if (i + 1 < graph.constPool.size()) oss << ", ";
    }
    oss << "]," << nl;

    // Serialize outputs
    oss << indent1 << "\"outputs\": [";
    for (size_t i = 0; i < graph.outputs.size(); ++i) {
        oss << graph.outputs[i];
        if (i + 1 < graph.outputs.size()) oss << ", ";
    }
    oss << "]," << nl;

    // Serialize diff_inputs
    oss << indent1 << "\"diff_inputs\": [";
    for (size_t i = 0; i < graph.diff_inputs.size(); ++i) {
        oss << graph.diff_inputs[i];
        if (i + 1 < graph.diff_inputs.size()) oss << ", ";
    }
    oss << "]" << nl;

    oss << "}";
    return oss.str();
}

Graph deserializeGraphFromJson(const std::string& jsonStr) {
    Graph graph;

    try {
        // Parse JSON using nlohmann/json (much more robust for large files)
        auto j = json::parse(jsonStr);

        // Check version
        if (j.contains("version")) {
            std::string version = j["version"].get<std::string>();
            if (version != "1.0") {
                throw std::runtime_error("Unsupported version: " + version);
            }
        }

        // Parse nodes
        if (j.contains("nodes") && j["nodes"].is_array()) {
            for (const auto& nodeJson : j["nodes"]) {
                Node node;

                if (nodeJson.contains("op")) {
                    node.op = stringToOpCode(nodeJson["op"].get<std::string>());
                }
                if (nodeJson.contains("dst")) {
                    node.dst = nodeJson["dst"].get<uint32_t>();
                }
                if (nodeJson.contains("a")) {
                    node.a = nodeJson["a"].get<uint32_t>();
                }
                if (nodeJson.contains("b")) {
                    node.b = nodeJson["b"].get<uint32_t>();
                }
                if (nodeJson.contains("c")) {
                    node.c = nodeJson["c"].get<uint32_t>();
                }
                if (nodeJson.contains("flags")) {
                    node.flags = nodeJson["flags"].get<uint32_t>();
                }
                if (nodeJson.contains("imm")) {
                    // Handle special double values
                    if (nodeJson["imm"].is_string()) {
                        std::string val = nodeJson["imm"].get<std::string>();
                        if (val == "NaN") {
                            node.imm = std::numeric_limits<double>::quiet_NaN();
                        } else if (val == "Infinity") {
                            node.imm = std::numeric_limits<double>::infinity();
                        } else if (val == "-Infinity") {
                            node.imm = -std::numeric_limits<double>::infinity();
                        } else {
                            node.imm = std::stod(val);
                        }
                    } else {
                        node.imm = nodeJson["imm"].get<double>();
                    }
                }
                if (nodeJson.contains("isActive")) {
                    node.isActive = nodeJson["isActive"].get<bool>();
                }
                if (nodeJson.contains("isDead")) {
                    node.isDead = nodeJson["isDead"].get<bool>();
                }
                if (nodeJson.contains("needsGradient")) {
                    node.needsGradient = nodeJson["needsGradient"].get<bool>();
                }

                graph.nodes.push_back(node);
            }
        }

        // Parse constant pool
        if (j.contains("constPool") && j["constPool"].is_array()) {
            for (const auto& constVal : j["constPool"]) {
                if (constVal.is_string()) {
                    std::string val = constVal.get<std::string>();
                    if (val == "NaN") {
                        graph.constPool.push_back(std::numeric_limits<double>::quiet_NaN());
                    } else if (val == "Infinity") {
                        graph.constPool.push_back(std::numeric_limits<double>::infinity());
                    } else if (val == "-Infinity") {
                        graph.constPool.push_back(-std::numeric_limits<double>::infinity());
                    } else {
                        graph.constPool.push_back(std::stod(val));
                    }
                } else {
                    graph.constPool.push_back(constVal.get<double>());
                }
            }
        }

        // Parse outputs
        if (j.contains("outputs") && j["outputs"].is_array()) {
            for (const auto& output : j["outputs"]) {
                graph.outputs.push_back(output.get<uint32_t>());
            }
        }

        // Parse diff_inputs
        if (j.contains("diff_inputs") && j["diff_inputs"].is_array()) {
            for (const auto& diffInput : j["diff_inputs"]) {
                graph.diff_inputs.push_back(diffInput.get<uint32_t>());
            }
        }

    } catch (const json::exception& e) {
        throw std::runtime_error(std::string("JSON parsing error: ") + e.what());
    }

    return graph;
}

bool saveGraphToFile(const Graph& graph, const std::string& filename, bool pretty) {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        file << serializeGraphToJson(graph, pretty);
        return true;
    } catch (...) {
        return false;
    }
}

Graph loadGraphFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string json((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());

    return deserializeGraphFromJson(json);
}

} // namespace forge
