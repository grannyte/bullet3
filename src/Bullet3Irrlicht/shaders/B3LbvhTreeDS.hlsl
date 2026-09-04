// PREFER_FLOW_CONTROL - data-dependent loops here must stay real loops; Release flattens
// branches by default, which breaks stack traversal and runtime-bounded clipping.
// Emulated-double (df64) wrapper; the kernels live in the shared body.

#define OS_DS 1
#include "B3LbvhTreeBody.hlsli"
