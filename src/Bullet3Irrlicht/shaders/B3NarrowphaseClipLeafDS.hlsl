// PREFER_FLOW_CONTROL - data-dependent loops here must stay real loops; Release flattens
// branches by default, which breaks stack traversal and runtime-bounded clipping.
// Emulated-double (df64) LEAF-PAIR wrapper; see B3NarrowphaseClipLeaf.hlsl.

#define OS_DS 1
#define B3_LEAF_PAIRS 1
#include "B3NarrowphaseClipBody.hlsli"
