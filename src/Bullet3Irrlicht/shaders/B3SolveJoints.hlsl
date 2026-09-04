// PREFER_FLOW_CONTROL - joint kernels branch per type/limit state; flattening silently evaluates
// every disabled row. Must sit in the top-level file: the flag is picked before includes resolve.
// Single-precision wrapper; the kernels live in the shared body.

#include "B3SolveJointsBody.hlsli"
