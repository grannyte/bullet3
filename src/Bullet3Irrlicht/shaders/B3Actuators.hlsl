// PREFER_FLOW_CONTROL - the per-kind dispatch and the brute-force AABB ray loop must stay real
// branches/loops; Release flattens them by default and changes what they compute.
// Single-precision wrapper; the kernels live in the shared body.

#include "B3ActuatorsBody.hlsli"
