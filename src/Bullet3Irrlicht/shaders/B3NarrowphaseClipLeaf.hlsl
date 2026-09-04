// PREFER_FLOW_CONTROL - data-dependent loops here must stay real loops; Release flattens
// branches by default, which breaks stack traversal and runtime-bounded clipping.
// Single-precision LEAF-PAIR wrapper: consumes (bodyA, bodyB, childA, childB) from the pair
// expansion kernel and composes compound child transforms onto the body transforms.

#define B3_LEAF_PAIRS 1
#include "B3NarrowphaseClipBody.hlsli"
