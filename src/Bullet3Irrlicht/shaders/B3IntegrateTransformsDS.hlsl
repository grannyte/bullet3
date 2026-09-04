// PREFER_FLOW_CONTROL - the per-body gravity/damping selects must stay branches in Release.
// Emulated-double (df64) wrapper; the kernels live in the shared body.

#define OS_DS 1
#include "B3IntegrateTransformsBody.hlsli"
