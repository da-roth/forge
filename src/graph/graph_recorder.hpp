#pragma once

#include "graph.hpp"
#include <memory>

namespace forge {

// Forward declaration
class GraphRecorder;

// Thread-local recorder registry - each thread has its own active recorder
class RecorderRegistry {
private:
    static thread_local GraphRecorder* activeRecorder_;
    
public:
    static void setActive(GraphRecorder* recorder);
    static GraphRecorder* getActive();
    static void clearActive();
};

class GraphRecorder {
private:
    Graph graph_;
    bool recording_ = false;
    
public:
    GraphRecorder() = default;
    ~GraphRecorder();
    
    GraphRecorder(const GraphRecorder&) = delete;
    GraphRecorder& operator=(const GraphRecorder&) = delete;
    
    void start();
    void stop();
    
    Graph& graph() { return graph_; }
    const Graph& graph() const { return graph_; }
    
    bool isRecording() const { return recording_; }
    
    static bool isAnyRecording() { return RecorderRegistry::getActive() != nullptr; }
    static GraphRecorder* active() { return RecorderRegistry::getActive(); }
};


} // namespace forge