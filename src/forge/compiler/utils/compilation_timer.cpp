#include "compilation_timer.h"
#include <iostream>
#include <iomanip>

namespace forge::compiler::utils {

std::string CompilationTimer::getOpName(forge::core::OpCode op) {
    switch(op) {
        case forge::core::OpCode::Input: return "Input";
        case forge::core::OpCode::Constant: return "Constant";
        case forge::core::OpCode::Add: return "Add";
        case forge::core::OpCode::Sub: return "Sub";
        case forge::core::OpCode::Mul: return "Mul";
        case forge::core::OpCode::Div: return "Div";
        case forge::core::OpCode::Neg: return "Neg";
        case forge::core::OpCode::Abs: return "Abs";
        case forge::core::OpCode::Square: return "Square";
        case forge::core::OpCode::Recip: return "Recip";
        case forge::core::OpCode::Sqrt: return "Sqrt";
        case forge::core::OpCode::Pow: return "Pow";
        case forge::core::OpCode::Exp: return "Exp";
        case forge::core::OpCode::Log: return "Log";
        case forge::core::OpCode::Sin: return "Sin";
        case forge::core::OpCode::Cos: return "Cos";
        case forge::core::OpCode::Tan: return "Tan";
        case forge::core::OpCode::Mod: return "Mod";
        case forge::core::OpCode::Min: return "Min";
        case forge::core::OpCode::Max: return "Max";
        case forge::core::OpCode::If: return "If";
        case forge::core::OpCode::CmpLT: return "CmpLT";
        case forge::core::OpCode::CmpLE: return "CmpLE";
        case forge::core::OpCode::CmpGT: return "CmpGT";
        case forge::core::OpCode::CmpGE: return "CmpGE";
        case forge::core::OpCode::CmpEQ: return "CmpEQ";
        case forge::core::OpCode::CmpNE: return "CmpNE";
        // Boolean operations
        case forge::core::OpCode::BoolConstant: return "BoolConstant";
        case forge::core::OpCode::BoolAnd: return "BoolAnd";
        case forge::core::OpCode::BoolOr: return "BoolOr";
        case forge::core::OpCode::BoolNot: return "BoolNot";
        case forge::core::OpCode::BoolEq: return "BoolEq";
        case forge::core::OpCode::BoolNe: return "BoolNe";
        // Integer operations
        case forge::core::OpCode::IntConstant: return "IntConstant";
        case forge::core::OpCode::IntAdd: return "IntAdd";
        case forge::core::OpCode::IntSub: return "IntSub";
        case forge::core::OpCode::IntMul: return "IntMul";
        case forge::core::OpCode::IntDiv: return "IntDiv";
        case forge::core::OpCode::IntMod: return "IntMod";
        case forge::core::OpCode::IntNeg: return "IntNeg";
        case forge::core::OpCode::IntCmpLT: return "IntCmpLT";
        case forge::core::OpCode::IntCmpLE: return "IntCmpLE";
        case forge::core::OpCode::IntCmpGT: return "IntCmpGT";
        case forge::core::OpCode::IntCmpGE: return "IntCmpGE";
        case forge::core::OpCode::IntCmpEQ: return "IntCmpEQ";
        case forge::core::OpCode::IntCmpNE: return "IntCmpNE";
        default: return "Unknown";
    }
}

void CompilationTimer::printTimingSummary(const TimingData& timing, bool verbose) {
    if (!verbose) return;
    
    std::cout << "\n=== Compilation Timing Summary ===" << std::endl;
    std::cout << "  Graph optimization: " << std::fixed << std::setprecision(2) 
              << timing.optimizationTimeMs << " ms" << std::endl;
    std::cout << "  Analysis phase: " << std::fixed << std::setprecision(2) 
              << timing.analysisTimeMs << " ms" << std::endl;
    std::cout << "  Code generation: " << std::fixed << std::setprecision(2) 
              << timing.codeGenerationTimeMs << " ms" << std::endl;
    std::cout << "  Total compilation: " << std::fixed << std::setprecision(2) 
              << timing.totalTimeMs << " ms" << std::endl;
    std::cout << "  Nodes processed: " << timing.originalNodeCount << " -> " 
              << timing.optimizedNodeCount << " (eliminated " << timing.deadNodeCount << ")" << std::endl;
}


// OperationTimer Implementation
OperationTimer::OperationTimer(const std::string& opName,
                               std::unordered_map<std::string, double>& timeMap,
                               std::unordered_map<std::string, int>& countMap,
                               bool enabled) 
    : enabled_(enabled), timeMap_(&timeMap), countMap_(&countMap) {
    if (enabled_) {
        opName_ = opName;
        start_ = CompilationTimer::Clock::now();
    }
}

OperationTimer::~OperationTimer() {
    if (enabled_) {
        auto end = CompilationTimer::Clock::now();
        double elapsed = CompilationTimer::Duration(end - start_).count();
        (*timeMap_)[opName_] += elapsed;
        (*countMap_)[opName_]++;
    }
}

} // namespace forge::compiler::utils
