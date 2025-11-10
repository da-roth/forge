#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>

namespace forge {

// Runtime tracing system for AVX/SSE2 instruction execution
// This provides safe, non-corrupting tracing of vector register values

// Trace record structure for storing vector data
struct TraceRecord {
    uint32_t instructionId;     // Unique ID for the instruction being traced
    uint32_t operationType;     // Type of operation (add, mul, etc.)
    uint32_t vectorWidth;       // Number of lanes (1 for SSE2 scalar, 4 for AVX2)
    uint64_t timestamp;         // High-resolution timestamp
    alignas(32) uint8_t data[32]; // Vector data (32 bytes for YMM, 16 bytes for XMM)
};

// Ring buffer for trace records
struct TraceBuffer {
    TraceRecord* records;
    uint32_t mask;              // Ring buffer size mask (must be power of 2 - 1)
    std::atomic<uint32_t> index; // Current write position
    bool enabled;
};

// Smart filtering configuration for runtime
struct RuntimeFilterConfig {
    bool enableSmartFilter = false;
    bool traceCorruptedOnly = false;
    bool detectNaN = true;
    bool detectInf = true;
    bool detectZeroCorruption = true;
    bool detectKnownPatterns = true;
    bool detectPartialCorruption = true;
};

// Global trace buffer instance
extern TraceBuffer g_traceBuffer;
extern RuntimeFilterConfig g_filterConfig;

// Initialize the trace buffer
void initializeTraceBuffer(size_t bufferSize = 1024);

// Configure smart filtering
void configureSmartFiltering(const RuntimeFilterConfig& config);

// Cleanup the trace buffer
void cleanupTraceBuffer();

// Enable/disable tracing
void setTracingEnabled(bool enabled);

// Check if tracing is enabled
bool isTracingEnabled();

// Tracer function to be called from JIT code
// This function is called with a pointer to stored vector data
extern "C" void traceVectorData(const void* data, uint32_t instructionId, uint32_t operationType, uint32_t vectorWidth);

// Helper function to get operation name from type
const char* getOperationName(uint32_t operationType);

// Print all trace records (for debugging)
void printTraceRecords();

// Operation type constants
enum class OperationType : uint32_t {
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,
    NEG = 5,
    ABS = 6,
    SQRT = 7,
    RECIP = 8,
    EXP = 9,
    LOG = 10,
    SIN = 11,
    COS = 12,
    TAN = 13,
    POW = 14,
    MOD = 15,
    MIN = 16,
    MAX = 17,
    CMP_LT = 18,
    CMP_LE = 19,
    CMP_GT = 20,
    CMP_GE = 21,
    CMP_EQ = 22,
    CMP_NE = 23,
    LOAD = 24,
    STORE = 25,
    LOAD_CONST = 26,
    MOVE = 27,
    ZERO = 28,
    SQUARE = 29,
    AND = 30,
    XOR = 31,
    OR = 32,
    ANDNOT = 33,
    BLEND = 34,
    CREATE_MASK = 35,
    CREATE_ALL_ONES = 36,
    SHIFT_LEFT = 37,
    SHIFT_RIGHT = 38,
    ROUND = 39,
    IF = 40,
    UNKNOWN = 0
};

} // namespace forge
