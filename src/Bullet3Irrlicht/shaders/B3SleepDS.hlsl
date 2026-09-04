// Emulated-double (df64) wrapper; the kernels live in the shared body.
// PREFER_FLOW_CONTROL - the marker is only read from the top-level source, never an .hlsli, and the
// body's wake/island loops change results if Release flattens them.

#define OS_DS 1
#include "B3SleepBody.hlsli"
