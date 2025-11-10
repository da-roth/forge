#include "runtime_trace.hpp"
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

    // std::cout << "[isTracingEnabled] enabled=" << (enabled ? "true" : "false")
    //           << ", hasRecords=" << (hasRecords ? "true" : "false")
    //           << ", returning=" << (result ? "true" : "false") << std::endl;

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

// Enhanced corruption detection for comparison mask issues
bool isComparisonMaskCorrupted(const double* data, uint32_t vectorWidth, uint32_t operationType) {
    // Check if this is a comparison operation
    bool isComparison = (operationType == static_cast<uint32_t>(OperationType::CMP_LT) ||
                        operationType == static_cast<uint32_t>(OperationType::CMP_LE) ||
                        operationType == static_cast<uint32_t>(OperationType::CMP_GT) ||
                        operationType == static_cast<uint32_t>(OperationType::CMP_GE) ||
                        operationType == static_cast<uint32_t>(OperationType::CMP_EQ) ||
                        operationType == static_cast<uint32_t>(OperationType::CMP_NE));
    
    if (!isComparison) return false;
    
    // For comparison operations, check if result contains -nan values
    // vcmppd produces 0xFFFFFFFFFFFFFFFF (true) or 0x0000000000000000 (false)
    // When interpreted as double: 0xFFFFFFFFFFFFFFFF = -nan(ind), 0x0000000000000000 = 0.0
    bool hasComparisonMasks = false;
    for (uint32_t i = 0; i < vectorWidth; i++) {
        double val = data[i];
        // Check for comparison mask patterns: -nan (true) or 0.0 (false)
        if (std::isnan(val) || val == 0.0) {
            hasComparisonMasks = true;
        } else {
            // If it's not a comparison mask pattern, it's suspicious
            return true;
        }
    }
    
    return false; // Comparison masks are expected for comparison operations
}

// Enhanced corruption detection for IF operations expecting booleans
bool isIfOperationCorrupted(const double* data, uint32_t vectorWidth, uint32_t operationType) {
    // Check if this is an IF operation
    if (operationType != static_cast<uint32_t>(OperationType::IF)) return false;
    
    // IF operations should receive boolean values (0.0 or 1.0), not comparison masks (-nan or 0.0)
    // If we see -nan in an IF operation, it means comparison masks weren't converted to booleans
    for (uint32_t i = 0; i < vectorWidth; i++) {
        double val = data[i];
        if (std::isnan(val)) {
            return true; // -nan in IF operation indicates mask-to-boolean conversion problem
        }
    }
    
    return false;
}

const char* getCorruptionDescription(const double* data, uint32_t vectorWidth) {
    if (!isVectorDataCorrupted(data, vectorWidth)) return "NO_CORRUPTION";
    
    for (uint32_t i = 0; i < vectorWidth; i++) {
        double val = data[i];
        if (std::isnan(val)) return "NaN_CORRUPTION";
        if (std::isinf(val)) return "INF_CORRUPTION";
        if (std::abs(val - 0.002) < 1e-12 || std::abs(val - 0.003) < 1e-12) return "PATTERN_CORRUPTION";
    }
    
    if (vectorWidth == 4) {
        if (data[2] == 0.0 || data[3] == 0.0) return "ZERO_CORRUPTION";
    }
    
    return "PARTIAL_CORRUPTION";
}

extern "C" void traceVectorData(const void* data, uint32_t instructionId, uint32_t operationType, uint32_t vectorWidth) {
    if (!isTracingEnabled()) {
        return;
    }
    
    const double* values = reinterpret_cast<const double*>(data);
    
    // Enhanced corruption detection
    bool isGeneralCorrupted = isVectorDataCorrupted(values, vectorWidth);
    bool isMaskCorrupted = isComparisonMaskCorrupted(values, vectorWidth, operationType);
    bool isIfCorrupted = isIfOperationCorrupted(values, vectorWidth, operationType);
    
    bool anyCorruption = isGeneralCorrupted || isMaskCorrupted || isIfCorrupted;
    
    // Apply smart filtering with enhanced detection
    if (g_filterConfig.enableSmartFilter && g_filterConfig.traceCorruptedOnly) {
        // Special focus on comparison->IF transition problems
        bool shouldTrace = anyCorruption;
        
        // Always trace IF operations that receive -nan (comparison mask problem)
        if (operationType == static_cast<uint32_t>(OperationType::IF)) {
            shouldTrace = true; // Always trace IF operations to see mask conversion issues
        }
        
        // Always trace comparison operations to see mask generation
        if (operationType == static_cast<uint32_t>(OperationType::CMP_LT) ||
            operationType == static_cast<uint32_t>(OperationType::CMP_LE) ||
            operationType == static_cast<uint32_t>(OperationType::CMP_GT) ||
            operationType == static_cast<uint32_t>(OperationType::CMP_GE) ||
            operationType == static_cast<uint32_t>(OperationType::CMP_EQ) ||
            operationType == static_cast<uint32_t>(OperationType::CMP_NE)) {
            shouldTrace = true; // Always trace comparison operations
        }
        
        if (!shouldTrace) {
            return;  // Skip this trace
        }
    }
    
    // Print with enhanced corruption detection
    if (isIfCorrupted) {
        std::cout << "[🚨 IF_MASK_BUG] ";
    } else if (isMaskCorrupted) {
        std::cout << "[🚨 MASK_CORRUPT] ";
    } else if (isGeneralCorrupted) {
        std::cout << "[🚨 CORRUPTION] ";
    } else {
        // Special markers for operations we're monitoring
        if (operationType == static_cast<uint32_t>(OperationType::IF)) {
            std::cout << "[🔍 IF_CHECK] ";
        } else if (operationType == static_cast<uint32_t>(OperationType::CMP_LT) ||
                   operationType == static_cast<uint32_t>(OperationType::CMP_LE) ||
                   operationType == static_cast<uint32_t>(OperationType::CMP_GT) ||
                   operationType == static_cast<uint32_t>(OperationType::CMP_GE) ||
                   operationType == static_cast<uint32_t>(OperationType::CMP_EQ) ||
                   operationType == static_cast<uint32_t>(OperationType::CMP_NE)) {
            std::cout << "[🔍 CMP_MASK] ";
        } else {
            std::cout << "[✓ CLEAN] ";
        }
    }
    
    std::cout << "ID:" << instructionId << " OP:" << getOperationName(operationType) << " LANES:" << vectorWidth;
    
    // Enhanced corruption type reporting
    if (isIfCorrupted) {
        std::cout << " TYPE:IF_RECEIVES_COMPARISON_MASK";
    } else if (isMaskCorrupted) {
        std::cout << " TYPE:INVALID_COMPARISON_MASK";
    } else if (isGeneralCorrupted) {
        std::cout << " TYPE:" << getCorruptionDescription(values, vectorWidth);
    }
    
    std::cout << " VALUES: ";
    for (uint32_t i = 0; i < vectorWidth; ++i) {
        if (std::isnan(values[i])) {
            std::cout << "🔥-nan🔥 ";
        } else if (std::isinf(values[i])) {
            std::cout << "⚠️" << std::fixed << std::setprecision(4) << values[i] << "⚠️ ";
        } else {
            std::cout << std::fixed << std::setprecision(4) << values[i] << " ";
        }
    }
    std::cout << std::endl;
    
    // Get current index and increment atomically
    uint32_t currentIndex = g_traceBuffer.index.fetch_add(1);
    uint32_t bufferIndex = currentIndex & g_traceBuffer.mask;
    
    TraceRecord& record = g_traceBuffer.records[bufferIndex];
    
    // Fill the record
    record.instructionId = instructionId;
    record.operationType = operationType;
    record.vectorWidth = vectorWidth;
    
    // Get high-resolution timestamp
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    record.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    
    // Copy vector data (up to 32 bytes)
    size_t dataSize = vectorWidth * sizeof(double);
    if (dataSize > sizeof(record.data)) {
        dataSize = sizeof(record.data);
    }
    std::memcpy(record.data, data, dataSize);
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
