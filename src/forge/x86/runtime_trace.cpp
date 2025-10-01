#include "runtime_trace.h"
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc and _aligned_free
#endif

namespace forge {
namespace x86 {

// Global trace buffer instance
TraceBuffer g_traceBuffer = {nullptr, 0, {0}, false};
RuntimeFilterConfig g_filterConfig;

void initializeTraceBuffer(size_t bufferSize) {
    // Only initialize once - check if already initialized
    if (g_traceBuffer.records != nullptr) {
        // Already initialized, just reset the index for a fresh start
        g_traceBuffer.index.store(0);
        g_traceBuffer.enabled = true;
        // Clear the buffer to avoid showing old garbage data
        std::memset(g_traceBuffer.records, 0, (g_traceBuffer.mask + 1) * sizeof(TraceRecord));
        return;
    }
    
    // Ensure buffer size is a power of 2
    size_t actualSize = 1;
    while (actualSize < bufferSize) {
        actualSize <<= 1;
    }
    
    // Allocate aligned memory for trace records
    // Use platform-specific aligned allocation since std::aligned_alloc may not be available
#ifdef _WIN32
    g_traceBuffer.records = static_cast<TraceRecord*>(_aligned_malloc(actualSize * sizeof(TraceRecord), 32));
#else
    g_traceBuffer.records = static_cast<TraceRecord*>(aligned_alloc(32, actualSize * sizeof(TraceRecord)));
#endif
    g_traceBuffer.mask = static_cast<uint32_t>(actualSize - 1);
    g_traceBuffer.index.store(0);
    g_traceBuffer.enabled = true;
    
    if (!g_traceBuffer.records) {
        std::cerr << "Failed to allocate trace buffer" << std::endl;
        g_traceBuffer.enabled = false;
    } else {
        // Clear the newly allocated buffer
        std::memset(g_traceBuffer.records, 0, actualSize * sizeof(TraceRecord));
    }
}

void cleanupTraceBuffer() {
    if (g_traceBuffer.records) {
#ifdef _WIN32
        _aligned_free(g_traceBuffer.records);
#else
        free(g_traceBuffer.records);
#endif
        g_traceBuffer.records = nullptr;
    }
    g_traceBuffer.mask = 0;
    g_traceBuffer.index.store(0);
    g_traceBuffer.enabled = false;
}

void setTracingEnabled(bool enabled) {
    g_traceBuffer.enabled = enabled;
}

bool isTracingEnabled() {
    bool enabled = g_traceBuffer.enabled;
    bool hasRecords = (g_traceBuffer.records != nullptr);
    bool result = enabled && hasRecords;
    return result;
}

void configureSmartFiltering(const RuntimeFilterConfig& config) {
    g_filterConfig = config;
}

// Smart corruption detection at runtime
bool isVectorDataCorrupted(const double* data, uint32_t vectorWidth) {
    if (!g_filterConfig.enableSmartFilter) return false;
    
    int validLanes = 0;
    bool foundCorruption = false;
    
    for (uint32_t i = 0; i < vectorWidth; i++) {
        double val = data[i];
        
        // NaN/Inf detection
        if (g_filterConfig.detectNaN && std::isnan(val)) {
            foundCorruption = true;
        } else if (g_filterConfig.detectInf && std::isinf(val)) {
            foundCorruption = true;
        } else {
            validLanes++;
            
            // Known corruption patterns: 0.002, 0.003 etc.
            if (g_filterConfig.detectKnownPatterns) {
                if (std::abs(val - 0.002) < 1e-12 || std::abs(val - 0.003) < 1e-12) {
                    foundCorruption = true;
                }
            }
            
            // Zero corruption (lanes 2-3 being zero in AVX2)
            if (g_filterConfig.detectZeroCorruption && vectorWidth == 4 && i >= 2 && val == 0.0) {
                // Only suspicious if earlier lanes have non-zero values
                bool earlierLanesNonZero = false;
                for (uint32_t j = 0; j < i; j++) {
                    if (data[j] != 0.0) {
                        earlierLanesNonZero = true;
                        break;
                    }
                }
                if (earlierLanesNonZero) {
                    foundCorruption = true;
                }
            }
        }
    }
    
    // Partial corruption detection (some lanes work, others don't)
    if (g_filterConfig.detectPartialCorruption && vectorWidth > 1) {
        if (validLanes > 0 && validLanes < static_cast<int>(vectorWidth)) {
            foundCorruption = true;
        }
    }
    
    return foundCorruption;
}

const char* getOperationName(uint32_t operationType) {
    switch (static_cast<OperationType>(operationType)) {
        case OperationType::ADD: return "ADD";
        case OperationType::SUB: return "SUB";
        case OperationType::MUL: return "MUL";
        case OperationType::DIV: return "DIV";
        case OperationType::NEG: return "NEG";
        case OperationType::ABS: return "ABS";
        case OperationType::SQRT: return "SQRT";
        case OperationType::RECIP: return "RECIP";
        case OperationType::EXP: return "EXP";
        case OperationType::LOG: return "LOG";
        case OperationType::SIN: return "SIN";
        case OperationType::COS: return "COS";
        case OperationType::TAN: return "TAN";
        case OperationType::POW: return "POW";
        case OperationType::MOD: return "MOD";
        case OperationType::MIN: return "MIN";
        case OperationType::MAX: return "MAX";
        case OperationType::CMP_LT: return "CMP_LT";
        case OperationType::CMP_LE: return "CMP_LE";
        case OperationType::CMP_GT: return "CMP_GT";
        case OperationType::CMP_GE: return "CMP_GE";
        case OperationType::CMP_EQ: return "CMP_EQ";
        case OperationType::CMP_NE: return "CMP_NE";
        case OperationType::LOAD: return "LOAD";
        case OperationType::STORE: return "STORE";
        case OperationType::LOAD_CONST: return "LOAD_CONST";
        case OperationType::MOVE: return "MOVE";
        case OperationType::ZERO: return "ZERO";
        case OperationType::SQUARE: return "SQUARE";
        case OperationType::AND: return "AND";
        case OperationType::XOR: return "XOR";
        case OperationType::OR: return "OR";
        case OperationType::ANDNOT: return "ANDNOT";
        case OperationType::BLEND: return "BLEND";
        case OperationType::CREATE_MASK: return "CREATE_MASK";
        case OperationType::CREATE_ALL_ONES: return "CREATE_ALL_ONES";
        case OperationType::SHIFT_LEFT: return "SHIFT_LEFT";
        case OperationType::SHIFT_RIGHT: return "SHIFT_RIGHT";
        case OperationType::ROUND: return "ROUND";
        case OperationType::IF: return "IF";
        default: return "UNKNOWN";
    }
}

void printTraceRecords() {
    if (!g_traceBuffer.records) {
        std::cout << "Trace buffer not initialized" << std::endl;
        return;
    }
    
    uint32_t currentIndex = g_traceBuffer.index.load();
    uint32_t numRecords = std::min(currentIndex, g_traceBuffer.mask + 1);
    
    std::cout << "\n=== Trace Records (" << numRecords << " records) ===" << std::endl;
    
    for (uint32_t i = 0; i < numRecords; i++) {
        uint32_t idx = i & g_traceBuffer.mask;
        TraceRecord& record = g_traceBuffer.records[idx];
        
        std::cout << "[" << std::setw(4) << i << "] ";
        std::cout << "ID=" << std::setw(4) << record.instructionId << " ";
        std::cout << "Op=" << std::setw(12) << getOperationName(record.operationType) << " ";
        std::cout << "Width=" << record.vectorWidth << " ";
        
        // Decode register info from timestamp
        uint32_t regInfo = static_cast<uint32_t>(record.timestamp & 0xFFFFFFFF);
        int dstReg = (regInfo >> 16) & 0xFFFF;
        int srcReg = regInfo & 0xFFFF;
        
        if (dstReg != 0xFFFE || srcReg != 0xFFFE) {
            std::cout << "Regs=[";
            if (dstReg != 0xFFFE) std::cout << "dst:" << dstReg;
            if (srcReg != 0xFFFE) {
                if (dstReg != 0xFFFE) std::cout << ",";
                std::cout << "src:" << srcReg;
            }
            std::cout << "] ";
        }
        
        std::cout << "Data=";
        double* values = reinterpret_cast<double*>(record.data);
        for (uint32_t j = 0; j < record.vectorWidth; j++) {
            if (j > 0) std::cout << ", ";
            if (std::isnan(values[j])) {
                std::cout << "NaN";
            } else if (std::isinf(values[j])) {
                std::cout << (values[j] > 0 ? "+Inf" : "-Inf");
            } else {
                std::cout << std::fixed << std::setprecision(6) << values[j];
            }
        }
        
        // Check for corruption
        if (isVectorDataCorrupted(values, record.vectorWidth)) {
            std::cout << " [CORRUPTED]";
        }
        
        std::cout << std::endl;
    }
    
    std::cout << "=== End Trace Records ===" << std::endl;
}

// Tracer function called from JIT code
extern "C" void traceVectorData(const void* data, uint32_t instructionId, uint32_t operationType, uint32_t vectorWidth) {
    if (!g_traceBuffer.enabled || !g_traceBuffer.records) {
        return;
    }
    
    // Smart filtering: Skip if not corrupted (when enabled)
    if (g_filterConfig.enableSmartFilter && g_filterConfig.traceCorruptedOnly) {
        const double* doubleData = static_cast<const double*>(data);
        if (!isVectorDataCorrupted(doubleData, vectorWidth)) {
            return;
        }
    }
    
    // Get next index atomically
    uint32_t index = g_traceBuffer.index.fetch_add(1);
    uint32_t recordIdx = index & g_traceBuffer.mask;
    
    TraceRecord& record = g_traceBuffer.records[recordIdx];
    record.instructionId = instructionId;
    record.operationType = operationType;
    record.vectorWidth = vectorWidth;
    record.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // Copy vector data
    std::memcpy(record.data, data, std::min(vectorWidth * sizeof(double), sizeof(record.data)));
}

} // namespace x86
} // namespace forge