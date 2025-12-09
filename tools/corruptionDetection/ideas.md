# Corruption Detection for JIT-Compiled Kernels

This document describes techniques for detecting data corruption in JIT-compiled SIMD kernels. These patterns were developed during debugging of AVX2/SSE2 code generation issues.

## Overview

When debugging JIT-compiled vector code, corruption can manifest in subtle ways:
- Only some SIMD lanes contain wrong values
- Specific operations produce NaN/Inf unexpectedly
- Known garbage patterns appear (e.g., 0.002, 0.003)
- Comparison masks aren't properly converted to boolean values

## Common Corruption Patterns

### 1. NaN/Inf Corruption

**Symptom:** Operations that should produce valid numbers return NaN or Inf.

**Causes:**
- Division by zero
- `sqrt()` of negative number
- Uninitialized register data
- Stack/register clobbering

**Detection:**
```cpp
for (uint32_t i = 0; i < vectorWidth; i++) {
    if (std::isnan(values[i])) {
        std::cout << "NaN detected in lane " << i << std::endl;
    }
    if (std::isinf(values[i])) {
        std::cout << "Inf detected in lane " << i << std::endl;
    }
}
```

### 2. Zero Corruption (AVX2 Lanes 2-3)

**Symptom:** In AVX2 mode (4 lanes), lanes 0-1 work correctly but lanes 2-3 are zero.

**Causes:**
- Using 128-bit (XMM) operations instead of 256-bit (YMM)
- Incorrect `vbroadcastsd` usage
- Memory alignment issues with upper lanes

**Detection:**
```cpp
if (vectorWidth == 4) {
    bool earlierLanesNonZero = (data[0] != 0.0 || data[1] != 0.0);
    bool laterLanesZero = (data[2] == 0.0 && data[3] == 0.0);

    if (earlierLanesNonZero && laterLanesZero) {
        std::cout << "Suspicious: lanes 0-1 have data, lanes 2-3 are zero" << std::endl;
    }
}
```

### 3. Known Pattern Corruption

**Symptom:** Specific garbage values appear repeatedly (e.g., 0.002, 0.003).

**Causes:**
- Reading from wrong memory location
- Stack frame corruption
- Register allocation bugs

**Detection:**
```cpp
// These were actual corruption signatures observed during development
const double knownPatterns[] = {0.002, 0.003};
for (double pattern : knownPatterns) {
    if (std::abs(value - pattern) < 1e-12) {
        std::cout << "Known corruption pattern detected: " << pattern << std::endl;
    }
}
```

### 4. Comparison Mask Corruption

**Symptom:** IF operations receive raw comparison masks (-nan/0x0) instead of boolean values (1.0/0.0).

**Background:** x86 SIMD comparisons (`vcmppd`) produce bitmasks:
- True: `0xFFFFFFFFFFFFFFFF` (displays as `-nan` when interpreted as double)
- False: `0x0000000000000000` (displays as `0.0`)

**Causes:**
- Missing mask-to-boolean conversion after comparison
- Using `vandpd` result directly instead of converting

**Detection:**
```cpp
// For IF operations, check if we're receiving masks instead of booleans
if (operationType == OperationType::IF) {
    for (uint32_t i = 0; i < vectorWidth; i++) {
        if (std::isnan(values[i])) {
            std::cout << "IF operation received comparison mask (-nan) instead of boolean!" << std::endl;
            std::cout << "Fix: Add mask-to-boolean conversion after comparison" << std::endl;
        }
    }
}
```

### 5. Partial Corruption

**Symptom:** Some lanes work, others don't (but aren't simply zero).

**Causes:**
- Incorrect lane shuffling
- Partial register overwrites
- Mixed scalar/vector operations

**Detection:**
```cpp
int validLanes = 0;
int corruptedLanes = 0;

for (int i = 0; i < vectorWidth; i++) {
    if (std::isnan(values[i]) || std::isinf(values[i])) {
        corruptedLanes++;
    } else {
        validLanes++;
    }
}

if (validLanes > 0 && corruptedLanes > 0) {
    std::cout << "Partial corruption: " << validLanes << " valid, "
              << corruptedLanes << " corrupted lanes" << std::endl;
}
```

## Adding Temporary Tracing

### Step 1: Enable Runtime Tracing

In your test or application:
```cpp
CompilerConfig config = CompilerConfig::Default();
config.printRuntimeTrace = true;  // Enable tracing
config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;

ForgeEngine engine(config);
auto kernel = engine.compile(graph);
```

### Step 2: Add Corruption Detection to traceVectorData

Temporarily modify `src/compiler/runtime_trace.cpp`:

```cpp
extern "C" void traceVectorData(const void* data, uint32_t instructionId,
                                 uint32_t operationType, uint32_t vectorWidth) {
    if (!isTracingEnabled()) return;

    const double* values = reinterpret_cast<const double*>(data);

    // === ADD YOUR CORRUPTION DETECTION HERE ===

    bool hasCorruption = false;
    std::string corruptionType;

    for (uint32_t i = 0; i < vectorWidth; ++i) {
        if (std::isnan(values[i])) {
            hasCorruption = true;
            corruptionType = "NaN";
            break;
        }
        if (std::isinf(values[i])) {
            hasCorruption = true;
            corruptionType = "Inf";
            break;
        }
        // Add your custom pattern detection here
    }

    // Only print if corruption detected (reduces noise)
    if (hasCorruption) {
        std::cout << "[CORRUPTION:" << corruptionType << "] ";
        std::cout << "ID:" << instructionId << " OP:" << getOperationName(operationType);
        std::cout << " VALUES: ";
        for (uint32_t i = 0; i < vectorWidth; ++i) {
            std::cout << values[i] << " ";
        }
        std::cout << std::endl;
    }

    // ... rest of function
}
```

### Step 3: Add Smart Filtering (Optional)

For large kernels, you may want to filter which operations get traced:

```cpp
// Add to CompilerConfig temporarily:
bool traceOnlyCorrupted = true;  // Only trace when corruption detected
bool traceComparisons = true;     // Always trace CMP operations
bool traceIfOperations = true;    // Always trace IF operations

// In traceVectorData:
bool shouldTrace = !traceOnlyCorrupted;  // Default: trace everything

if (hasCorruption) {
    shouldTrace = true;
}

// Always trace operations that commonly cause issues
if (operationType == OperationType::CMP_LT ||
    operationType == OperationType::CMP_LE ||
    operationType == OperationType::IF) {
    shouldTrace = true;
}

if (!shouldTrace) return;
```

### Step 4: Trace Context (Before/After)

Sometimes you need to see what happened before corruption:

```cpp
// Keep a ring buffer of recent operations
static std::vector<std::string> recentOps;
static const size_t contextSize = 5;

std::ostringstream oss;
oss << "ID:" << instructionId << " OP:" << getOperationName(operationType);
for (uint32_t i = 0; i < vectorWidth; ++i) {
    oss << " " << values[i];
}

recentOps.push_back(oss.str());
if (recentOps.size() > contextSize) {
    recentOps.erase(recentOps.begin());
}

if (hasCorruption) {
    std::cout << "=== CORRUPTION CONTEXT ===" << std::endl;
    for (const auto& op : recentOps) {
        std::cout << op << std::endl;
    }
    std::cout << "==========================" << std::endl;
}
```

## Debugging Workflow

1. **Identify the symptom**: Wrong output values, NaN results, partial failures

2. **Enable tracing**: Set `config.printRuntimeTrace = true`

3. **Run the test**: Look for patterns in the trace output

4. **Narrow down**: Add corruption detection to filter noise

5. **Find the operation**: Identify which operation produces wrong values

6. **Check the codegen**: Look at the generated assembly for that operation
   - Use `config.printAssembly = true` if available
   - Or use a debugger to step through JIT code

7. **Fix and verify**: Make the fix, run tests, remove tracing

## Example: Debugging Comparison Mask Issue

This was an actual bug where IF operations received comparison masks instead of booleans.

**Symptom:** Conditional operations produced wrong results in some lanes.

**Trace output showed:**
```
[TRACE] ID:15 OP:CMP_LT VALUES: -nan 0.0000 -nan -nan
[TRACE] ID:16 OP:IF VALUES: -nan 0.0000 -nan -nan   <-- IF receiving masks!
```

**Root cause:** The comparison `vcmppd` produces `0xFFFFFFFF...` for true, but IF expected `1.0`.

**Fix:** Add mask-to-boolean conversion:
```asm
; After vcmppd:
vandpd ymm_result, ymm_mask, ymm_one  ; Convert 0xFFFF... to 1.0
```

## Cleanup Reminder

After debugging, remember to:
1. Remove any temporary corruption detection code
2. Set `config.printRuntimeTrace = false`
3. Remove debug print statements
4. Run full test suite to verify fix
