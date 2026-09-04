// Single-precision wrapper; the kernels live in the shared body.
// PREFER_FLOW_CONTROL - the marker is only read from the top-level source, never an .hlsli, and the
// body's wake/island loops change results if Release flattens them.

#include "B3SleepBody.hlsli"
