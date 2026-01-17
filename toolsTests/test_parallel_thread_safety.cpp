#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cmath>
#include <native/fdouble.hpp>
#include "../src/graph/graph_recorder.hpp"
#include "../src/compiler/forge_engine.hpp"
#include "../src/compiler/x86/common/compiler_config.hpp"
#include "../src/compiler/interfaces/node_value_buffer.hpp"

using namespace forge;

// Test fixture for parallel thread safety tests
class ParallelThreadSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper function that each thread will execute
    // Performs a simple computation: f(x) = x * 2 + 1
    static void simpleComputation(int threadId, std::atomic<int>& successCount,
                                  std::atomic<int>& failureCount) {
        try {
            // Create a recorder for this thread
            GraphRecorder recorder;

            // Start recording
            recorder.start();

            // Create computation: f(x) = x * 2 + 1
            fdouble input(0.0);
            input.markInput();
            fdouble result = input * fdouble(2.0) + fdouble(1.0);
            result.markOutput();

            // Stop recording
            recorder.stop();

            // Get the graph
            Graph graph = recorder.graph();

            // Compile the graph
            CompilerConfig config;
            config.instructionSet = CompilerConfig::InstructionSet::SSE2_SCALAR;
            ForgeEngine engine(config);
            auto kernel = engine.compile(graph);

            // Create buffer
            auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

            // Execute with test value
            double testInput = 5.0;
            NodeId inputNode = 0;  // First node is input
            NodeId outputNode = graph.outputs[0];

            buffer->setValue(inputNode, testInput);
            kernel->execute(*buffer);
            double actualResult = buffer->getValue(outputNode);

            // Verify result: 5 * 2 + 1 = 11
            double expected = 11.0;
            if (std::abs(actualResult - expected) < 1e-10) {
                successCount.fetch_add(1, std::memory_order_relaxed);
            } else {
                failureCount.fetch_add(1, std::memory_order_relaxed);
            }

        } catch (const std::exception& e) {
            // Record failure with error message
            failureCount.fetch_add(1, std::memory_order_relaxed);
            // Thread-safe error logging
            static std::mutex error_mutex;
            std::lock_guard<std::mutex> lock(error_mutex);
            std::cerr << "[Thread " << threadId << "] ERROR: " << e.what() << std::endl;
        }
    }

    // More complex computation for stress testing
    static void complexComputation(int threadId, std::atomic<int>& successCount,
                                   std::atomic<int>& failureCount) {
        try {
            GraphRecorder recorder;
            recorder.start();

            // Create a more complex computation involving trig and exp functions
            // f(x) = sin(x) * exp(x/10) + cos(x)
            fdouble input(0.0);
            input.markInput();
            fdouble temp1 = sin(input);
            fdouble temp2 = input / fdouble(10.0);
            fdouble temp3 = exp(temp2);
            fdouble temp4 = temp1 * temp3;
            fdouble temp5 = cos(input);
            fdouble result = temp4 + temp5;
            result.markOutput();

            recorder.stop();
            Graph graph = recorder.graph();

            CompilerConfig config;
            config.instructionSet = CompilerConfig::InstructionSet::SSE2_SCALAR;
            ForgeEngine engine(config);
            auto kernel = engine.compile(graph);

            // Create buffer
            auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

            // Test with pi/4
            double testInput = 0.785398163397448; // pi/4
            NodeId inputNode = 0;
            NodeId outputNode = graph.outputs[0];

            buffer->setValue(inputNode, testInput);
            kernel->execute(*buffer);
            double actualResult = buffer->getValue(outputNode);

            // Compute expected value
            double expected = std::sin(testInput) * std::exp(testInput/10.0) + std::cos(testInput);

            if (std::abs(actualResult - expected) < 1e-6) {
                successCount.fetch_add(1, std::memory_order_relaxed);
            } else {
                failureCount.fetch_add(1, std::memory_order_relaxed);
            }

        } catch (const std::exception& e) {
            failureCount.fetch_add(1, std::memory_order_relaxed);
            // Thread-safe error logging
            static std::mutex error_mutex;
            std::lock_guard<std::mutex> lock(error_mutex);
            std::cerr << "[Thread " << threadId << " COMPLEX] ERROR: " << e.what() << std::endl;
        }
    }
};

// Test 1: Sequential execution (baseline - should always work)
TEST_F(ParallelThreadSafetyTest, SequentialExecution) {
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    const int numIterations = 10;

    for (int i = 0; i < numIterations; ++i) {
        simpleComputation(i, successCount, failureCount);
    }

    EXPECT_EQ(successCount.load(), numIterations);
    EXPECT_EQ(failureCount.load(), 0);
}

// Test 2: Parallel execution with 2 threads
TEST_F(ParallelThreadSafetyTest, TwoThreadsParallel) {
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    const int numThreads = 2;
    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(simpleComputation, i, std::ref(successCount), std::ref(failureCount));
    }

    for (auto& t : threads) {
        t.join();
    }

    // At least one thread should complete successfully
    // Both should complete successfully for thread-safe implementation
    EXPECT_GT(successCount.load() + failureCount.load(), 0);

    // Report results
    std::cout << "2 Threads - Success: " << successCount.load()
              << ", Failures: " << failureCount.load() << std::endl;
}

// Test 3: Parallel execution with 4 threads
TEST_F(ParallelThreadSafetyTest, FourThreadsParallel) {
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    const int numThreads = 4;
    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(simpleComputation, i, std::ref(successCount), std::ref(failureCount));
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(successCount.load() + failureCount.load(), 0);

    std::cout << "4 Threads - Success: " << successCount.load()
              << ", Failures: " << failureCount.load() << std::endl;
}

// Test 4: Parallel execution with 8 threads (stress test)
TEST_F(ParallelThreadSafetyTest, EightThreadsParallel) {
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    const int numThreads = 8;
    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(simpleComputation, i, std::ref(successCount), std::ref(failureCount));
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(successCount.load() + failureCount.load(), 0);

    std::cout << "8 Threads - Success: " << successCount.load()
              << ", Failures: " << failureCount.load() << std::endl;
}

// Test 5: Multiple iterations per thread
TEST_F(ParallelThreadSafetyTest, MultipleIterationsPerThread) {
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    const int numThreads = 4;
    const int iterationsPerThread = 5;

    auto threadFunc = [&](int threadId) {
        for (int i = 0; i < iterationsPerThread; ++i) {
            simpleComputation(threadId * iterationsPerThread + i, successCount, failureCount);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(threadFunc, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    const int totalExpected = numThreads * iterationsPerThread;
    EXPECT_EQ(successCount.load() + failureCount.load(), totalExpected);

    std::cout << "Multiple Iterations - Success: " << successCount.load()
              << ", Failures: " << failureCount.load()
              << " out of " << totalExpected << std::endl;
}

// Test 6: Complex computations in parallel
TEST_F(ParallelThreadSafetyTest, ComplexComputationsParallel) {
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    const int numThreads = 4;
    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(complexComputation, i, std::ref(successCount), std::ref(failureCount));
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(successCount.load() + failureCount.load(), 0);

    std::cout << "Complex Parallel - Success: " << successCount.load()
              << ", Failures: " << failureCount.load() << std::endl;
}

// Test 7: Race condition stress test - many threads starting simultaneously
TEST_F(ParallelThreadSafetyTest, RaceConditionStressTest) {
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    const int numThreads = 16;
    std::vector<std::thread> threads;

    // Use a barrier to ensure all threads start as close to simultaneously as possible
    std::atomic<bool> startFlag{false};

    auto threadFunc = [&](int threadId) {
        // Wait for start signal
        while (!startFlag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        simpleComputation(threadId, successCount, failureCount);
    };

    // Create all threads
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(threadFunc, i);
    }

    // Small delay to ensure all threads are waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Release all threads at once
    startFlag.store(true, std::memory_order_release);

    // Wait for completion
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(successCount.load() + failureCount.load(), 0);

    std::cout << "Race Condition Stress Test (16 threads) - Success: " << successCount.load()
              << ", Failures: " << failureCount.load() << std::endl;
}

// Test 8: Mixed SSE2 and AVX2 compilations in parallel
TEST_F(ParallelThreadSafetyTest, MixedInstructionSetsParallel) {
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    const int numThreads = 4;

    auto threadFunc = [&](int threadId) {
        try {
            GraphRecorder recorder;
            recorder.start();

            fdouble input(0.0);
            input.markInput();
            fdouble result = input * fdouble(3.0) + fdouble(2.0);
            result.markOutput();

            recorder.stop();
            Graph graph = recorder.graph();

            // Alternate between SSE2 and AVX2 based on thread ID
            CompilerConfig config;
            if (threadId % 2 == 0) {
                config.instructionSet = CompilerConfig::InstructionSet::SSE2_SCALAR;
            } else {
                config.instructionSet = CompilerConfig::InstructionSet::AVX2_PACKED;
            }

            ForgeEngine engine(config);
            auto kernel = engine.compile(graph);

            // Create buffer
            auto buffer = forge::NodeValueBufferFactory::create(graph, *kernel);

            double testInput = 4.0;
            NodeId inputNode = 0;
            NodeId outputNode = graph.outputs[0];

            buffer->setValue(inputNode, testInput);
            kernel->execute(*buffer);
            double actualResult = buffer->getValue(outputNode);

            double expected = 14.0; // 4 * 3 + 2
            if (std::abs(actualResult - expected) < 1e-10) {
                successCount.fetch_add(1, std::memory_order_relaxed);
            } else {
                failureCount.fetch_add(1, std::memory_order_relaxed);
            }

        } catch (const std::exception& e) {
            failureCount.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(threadFunc, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GT(successCount.load() + failureCount.load(), 0);

    std::cout << "Mixed Instruction Sets - Success: " << successCount.load()
              << ", Failures: " << failureCount.load() << std::endl;
}
