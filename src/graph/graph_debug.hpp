#pragma once

#include "graph.hpp"
#include <vector>
#include <memory>

namespace forge {

// Separate debug recording structure that can be optionally attached to a tape
// This avoids adding overhead to the core Graph structure
class GraphDebugRecorder {
public:
    std::vector<double> recordingResults;
    bool enabled{false};
    
    void captureIntermediateResult(NodeId nodeId, double value) {
        if (!enabled) return;
        
        // Ensure recordingResults has enough space
        if (recordingResults.size() <= nodeId) {
            recordingResults.resize(nodeId + 1, 0.0);
        }
        recordingResults[nodeId] = value;
    }
    
    void clear() {
        recordingResults.clear();
        enabled = false;
    }
};

// Global debug recorder that can be attached when needed
// This is a singleton to avoid passing it everywhere
class DebugRecorderManager {
private:
    static std::unique_ptr<GraphDebugRecorder> instance_;
    
public:
    static GraphDebugRecorder* get() {
        if (!instance_) {
            instance_ = std::make_unique<GraphDebugRecorder>();
        }
        return instance_.get();
    }
    
    static void enable() {
        get()->enabled = true;
    }
    
    static void disable() {
        if (instance_) {
            instance_->enabled = false;
            instance_->clear();
        }
    }
    
    static bool isEnabled() {
        return instance_ && instance_->enabled;
    }
    
    static void reset() {
        instance_.reset();
    }
};

// Initialize the static member
inline std::unique_ptr<GraphDebugRecorder> DebugRecorderManager::instance_ = nullptr;

} // namespace forge