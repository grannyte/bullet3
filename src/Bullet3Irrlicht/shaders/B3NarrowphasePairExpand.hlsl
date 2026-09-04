// PREFER_FLOW_CONTROL - data-dependent loops here must stay real loops; Release flattens
// branches by default, which breaks stack traversal and runtime-bounded clipping.
// Single-precision wrapper; the kernel lives in the shared body.

#include "B3NarrowphasePairExpandBody.hlsli"
