// PREFER_FLOW_CONTROL - the ancestor walk must stay a real loop; Release flattens branches by
// default, which breaks a runtime-bounded traversal.
// Emulated-double (df64) wrapper; the kernels live in the shared body.

#define OS_DS 1
#include "B3LbvhRefitBody.hlsli"
