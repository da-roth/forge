# Tape Interpreter

A debugging tool for interpreting and analyzing tape execution, useful for:
- Step-by-step execution of tape operations
- Comparing recorded vs computed values
- Debugging tape generation issues
- Performance analysis without JIT compilation

## Usage

The interpreter can operate in two modes:

### 1. Standard Execution
Simply executes the tape and returns results.

### 2. Comparison Mode
When the tape has `recordingResults` populated, the interpreter will compare computed values against recorded values and report divergences.

## Features

- Complete tape operation support
- Bounds checking and error reporting
- Progress reporting for large tapes
- Divergence detection with detailed output
- Memory-efficient execution

## Note

This is a debugging tool and should not be used in production. For production use, the JIT-compiled kernels provide much better performance.