// PREFER_FLOW_CONTROL - the per-kind dispatch and the brute-force AABB ray loop must stay real
// branches/loops; Release flattens them by default and changes what they compute.
// Emulated-double (df64) wrapper; the kernels live in the shared body.

#define OS_DS 1
#include "B3ActuatorsBody.hlsli"
