#include "graph_recorder.hpp"
#include <stdexcept>
#include <mutex>

namespace forge {

// Thread-local state - each thread gets its own active recorder
thread_local GraphRecorder* RecorderRegistry::activeRecorder_ = nullptr;

void RecorderRegistry::setActive(GraphRecorder* recorder) {
    activeRecorder_ = recorder;
}

GraphRecorder* RecorderRegistry::getActive() {
    return activeRecorder_;
}

void RecorderRegistry::clearActive() {
    activeRecorder_ = nullptr;
}

GraphRecorder::~GraphRecorder() {
    if (recording_) {
        // Clean up if still recording
        recording_ = false;
        if (RecorderRegistry::getActive() == this) {
            RecorderRegistry::clearActive();
        }
    }
}

void GraphRecorder::start() {
    // Check if another recorder is already active
    if (RecorderRegistry::getActive() != nullptr) {
        throw std::runtime_error("Another GraphRecorder is already active");
    }
    
    if (recording_) {
        throw std::runtime_error("This recorder is already recording");
    }
    
    graph_.clear();
    recording_ = true;
    RecorderRegistry::setActive(this);
}

void GraphRecorder::stop() {
    if (!recording_) {
        throw std::runtime_error("GraphRecorder::stop() called without matching start()");
    }
    
    // Enforce that at least one output was marked
    if (graph_.outputs.empty()) {
        recording_ = false;
        if (RecorderRegistry::getActive() == this) {
            RecorderRegistry::clearActive();
        }
        throw std::runtime_error("No outputs were marked. Call markOutput() on at least one result before stopping the recorder.");
    }

    recording_ = false;
    if (RecorderRegistry::getActive() == this) {
        RecorderRegistry::clearActive();
    }
}

} // namespace forge