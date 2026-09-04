// PREFER_FLOW_CONTROL - the ancestor walk must stay a real loop; Release flattens branches by
// default, which breaks a runtime-bounded traversal.
// Single-precision wrapper; the kernels live in the shared body.

#include "B3LbvhRefitBody.hlsli"
