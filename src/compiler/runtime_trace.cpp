#include "runtime_trace.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc and _aligned_free
#endif

namespace forge {

// Global trace buffer instance
TraceBuffer g_traceBuffer = {nullptr, 0, {0}, false};

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

    // std::cout << "[isTracingEnabled] enabled=" << (enabled ? "true" : "false")
    //           << ", hasRecords=" << (hasRecords ? "true" : "false")
    //           << ", returning=" << (result ? "true" : "false") << std::endl;

    return result;
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
    if (!isTracingEnabled()) {
        return;
    }
    
    uint32_t currentIndex = g_traceBuffer.index.load();
    uint32_t recordCount = std::min(currentIndex, g_traceBuffer.mask + 1);
    
    // Limit runtime trace to 1000 entries
    const uint32_t maxDisplayRecords = 1000;
    uint32_t displayCount = std::min(recordCount, maxDisplayRecords);
    
    std::cout << "\n=== Runtime Trace Records ===" << std::endl;
    std::cout << "Total records: " << recordCount;
    if (recordCount > maxDisplayRecords) {
        std::cout << " (showing first " << maxDisplayRecords << ")";
    }
    std::cout << std::endl;
    std::cout << "Buffer size: " << (g_traceBuffer.mask + 1) << std::endl;
    std::cout << "Current index: " << currentIndex << std::endl;
    std::cout << "sizeof(TraceRecord): " << sizeof(TraceRecord) << std::endl;
    std::cout << "offsetof(data): " << offsetof(TraceRecord, data) << std::endl;
    std::cout << "Note: Bitwise ops (CREATE_ALL_ONES, SHIFT_*) show hex patterns" << std::endl;
    std::cout << "=============================" << std::endl;
    
    for (uint32_t i = 0; i < displayCount; ++i) {
        const TraceRecord& record = g_traceBuffer.records[i];
        
        // Debug: Always print first record for debugging
        if (i == 0) {
            std::cout << "[DEBUG] First record: op=" << record.operationType 
                      << ", width=" << record.vectorWidth 
                      << ", id=" << record.instructionId;
            const double* values = reinterpret_cast<const double*>(record.data);
            std::cout << ", data=" << values[0] << std::endl;
        }
        
        // Debug: show all records with full details
        if (record.operationType == 0 && record.vectorWidth == 0) {
            std::cout << "[" << i << "] EMPTY record (op=" << record.operationType 
                      << ", width=" << record.vectorWidth 
                      << ", id=" << record.instructionId << ")" << std::endl;
            continue;
        }
        
        std::cout << "[" << i << "] ";
        
        // Format based on operation type
        const char* opName = getOperationName(record.operationType);
        
        // Extract register info from timestamp field
        uint32_t regInfo = static_cast<uint32_t>(record.timestamp & 0xFFFFFFFF);
        uint32_t srcRegRaw = regInfo & 0xFFFF;
        uint32_t dstRegRaw = (regInfo >> 16) & 0xFFFF;
        
        // Convert back from encoded format (0xFFFE means no register)
        int srcReg = (srcRegRaw == 0xFFFE) ? -1 : static_cast<int>(srcRegRaw);
        int dstReg = (dstRegRaw == 0xFFFE) ? -1 : static_cast<int>(dstRegRaw);
        
        // Helper function to format register names (handles -1 as "none")
        auto formatReg = [&](int regId) {
            if (regId < 0) {
                std::cout << "none";
            } else {
                const char* regPrefix = (record.vectorWidth == 1) ? "xmm" : "ymm";
                std::cout << regPrefix << regId;
            }
        };
        
        if (record.operationType == static_cast<uint32_t>(OperationType::LOAD)) {
            std::cout << "LOAD(node#" << record.instructionId << "->";
            formatReg(dstReg);
            std::cout << ")";
        } else if (record.operationType == static_cast<uint32_t>(OperationType::STORE)) {
            std::cout << "STORE(";
            formatReg(srcReg);
            std::cout << "->node#" << record.instructionId << ")";
        } else if (record.operationType == static_cast<uint32_t>(OperationType::ADD)) {
            std::cout << "ADD(";
            formatReg(dstReg);
            std::cout << "+";
            formatReg(srcReg);
            std::cout << ")";
        } else if (record.operationType == static_cast<uint32_t>(OperationType::SUB)) {
            std::cout << "SUB(";
            formatReg(dstReg);
            std::cout << "-";
            formatReg(srcReg);
            std::cout << ")";
        } else if (record.operationType == static_cast<uint32_t>(OperationType::MUL)) {
            std::cout << "MUL(";
            formatReg(dstReg);
            std::cout << "*";
            formatReg(srcReg);
            std::cout << ")";
        } else if (record.operationType == static_cast<uint32_t>(OperationType::DIV)) {
            std::cout << "DIV(";
            formatReg(dstReg);
            std::cout << "/";
            formatReg(srcReg);
            std::cout << ")";
        } else if (record.operationType == static_cast<uint32_t>(OperationType::EXP)) {
            std::cout << "EXP(";
            formatReg(dstReg);
            std::cout << ")";
        } else if (record.operationType == static_cast<uint32_t>(OperationType::LOG)) {
            std::cout << "LOG(";
            formatReg(dstReg);
            std::cout << ")";
        } else if (record.operationType == static_cast<uint32_t>(OperationType::SQRT)) {
            std::cout << "SQRT(";
            formatReg(dstReg);
            std::cout << ")";
        } else {
            std::cout << opName << "(op=" << record.operationType << ",regs=" << dstReg << "," << srcReg << ")";
        }
        
        std::cout << " = ";
        
        // Check if this is a bitwise operation that should show hex patterns
        bool isBitwiseOp = (record.operationType == static_cast<uint32_t>(OperationType::CREATE_ALL_ONES) ||
                           record.operationType == static_cast<uint32_t>(OperationType::SHIFT_LEFT) ||
                           record.operationType == static_cast<uint32_t>(OperationType::SHIFT_RIGHT) ||
                           record.operationType == static_cast<uint32_t>(OperationType::CREATE_MASK));
        
        // Print vector values - only print the actual number of lanes
        const double* values = reinterpret_cast<const double*>(record.data);
        const uint64_t* bitPatterns = reinterpret_cast<const uint64_t*>(record.data);
        
        for (uint32_t lane = 0; lane < std::min(record.vectorWidth, 4u); ++lane) {
            if (lane > 0) std::cout << ", ";
            
            if (isBitwiseOp) {
                // Show as hex pattern for bit operations
                std::cout << "0x" << std::hex << std::setw(16) << std::setfill('0') << bitPatterns[lane] << std::dec;
            } else if (std::isnan(values[lane])) {
                // For non-bitwise ops, NaN is unexpected - show warning
                std::cout << "NaN";
            } else {
                // Normal double value
                std::cout << std::fixed << std::setprecision(3) << values[lane];
            }
        }
        std::cout << std::endl;
    }
    
    // Show truncation message if needed
    if (recordCount > maxDisplayRecords) {
        std::cout << "... (" << (recordCount - maxDisplayRecords) << " more records omitted)" << std::endl;
    }
    
    std::cout << "=============================" << std::endl;
}

} // namespace forge
