#pragma once

#include "graph.hpp"

namespace forge {

struct InputHandle {
    NodeId node;
    explicit InputHandle(NodeId n) : node(n) {}
};

struct ResultHandle {
    NodeId node;
    explicit ResultHandle(NodeId n) : node(n) {}
};

} // namespace forge