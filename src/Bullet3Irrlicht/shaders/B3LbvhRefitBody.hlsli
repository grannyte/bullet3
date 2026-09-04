// Sparse LBVH refit: only the root-paths of leaves that actually moved are re-fitted, instead of
// every internal node at every depth. Own file so the kernels get a clean register space.

// Included by B3LbvhRefit.hlsl (f32) and B3LbvhRefitDS.hlsl (df64) - see B3Precision.hlsli.

#include "B3Precision.hlsli"

#define WG_SIZE 128
#define B3_PLBVH_ROOT_NODE_MARKER -1
#define SLEEP_ASLEEP_BIT 0x80000000u

struct RefitParams
{
	uint numLeafNodes;
	uint numInternalNodes;
	int processedDistance;
	uint frameStamp;
};

StructuredBuffer<RefitParams> Params : register(t0);
StructuredBuffer<uint2> MortonCodes : register(t1);    // (key, leaf row), sorted
StructuredBuffer<int> LeafParents : register(t2);
StructuredBuffer<int> InternalParents : register(t3);
StructuredBuffer<uint> SleepStates : register(t4);     // bit31 = asleep
StructuredBuffer<uint> LeafToBody : register(t5);      // leaf row -> caller/body index
StructuredBuffer<uint4> DirtyCount : register(t6);     // .x patched from the append counter
StructuredBuffer<uint> DirtyNodes : register(t7);
StructuredBuffer<int> ChildNodes : register(t8);
StructuredBuffer<int> DistanceFromRoot : register(t9);
StructuredBuffer<b3Aabb> LeafAabbs : register(t10);

// A frame stamp rather than a flag: nothing has to be cleared between steps.
RWStructuredBuffer<uint> DirtyStamps : register(u0);
AppendStructuredBuffer<uint> OutDirtyNodes : register(u1);
// Read back through the UAV: D3D11 nulls one binding if a resource is both SRV and UAV.
RWStructuredBuffer<b3Aabb> OutInternalAabbs : register(u2);

int isLeafNode(int index) { return (index >> 31) == 0; }
int indexMarkerRemoved(int index) { return index & (~0x80000000); }

//! One thread per leaf slot; an awake leaf stamps every node up to the root and appends the ones
//! it stamped first. Breaking on an already-stamped node is safe: that stamper walks the rest.
[numthreads(WG_SIZE, 1, 1)]
void CSMarkMovedLeafAncestors(uint3 tid : SV_DispatchThreadID)
{
	const uint numLeafNodes = Params[0].numLeafNodes;
	const uint slot = tid.x;
	if (slot >= numLeafNodes)
		return;

	if ((SleepStates[LeafToBody[MortonCodes[slot].y]] & SLEEP_ASLEEP_BIT) != 0)
		return;

	const uint stamp = Params[0].frameStamp;
	const int numInternalNodes = (int)Params[0].numInternalNodes;

	int node = LeafParents[slot];
	int guard = 0;
	while (node >= 0 && node < numInternalNodes && guard++ <= numInternalNodes)
	{
		uint previous;
		InterlockedExchange(DirtyStamps[node], stamp, previous);
		if (previous == stamp)
			break;

		OutDirtyNodes.Append((uint)node);
		node = InternalParents[node];
	}
}

//! CSBuildTreeAabbs restricted to the dirty set. A node whose subtree holds no moved leaf keeps
//! last step's bound, which is still exact - nothing under it changed.
[numthreads(WG_SIZE, 1, 1)]
void CSRefitDirtyAabbs(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= DirtyCount[0].x)
		return;

	const int numInternalNodes = (int)Params[0].numInternalNodes;
	const int nodeIndex = (int)DirtyNodes[tid.x];
	if (nodeIndex < 0 || nodeIndex >= numInternalNodes)
		return;

	if (DistanceFromRoot[nodeIndex] != Params[0].processedDistance)
		return;

	int leftChild = ChildNodes[2 * nodeIndex + 0];
	int rightChild = ChildNodes[2 * nodeIndex + 1];

	const bool leftIsLeaf = isLeafNode(leftChild) != 0;
	const bool rightIsLeaf = isLeafNode(rightChild) != 0;

	leftChild = indexMarkerRemoved(leftChild);
	rightChild = indexMarkerRemoved(rightChild);

	b3Aabb leftAabb;
	if (leftIsLeaf)
		leftAabb = LeafAabbs[MortonCodes[leftChild].y];
	else
		leftAabb = OutInternalAabbs[leftChild];

	b3Aabb rightAabb;
	if (rightIsLeaf)
		rightAabb = LeafAabbs[MortonCodes[rightChild].y];
	else
		rightAabb = OutInternalAabbs[rightChild];

	OutInternalAabbs[nodeIndex] = osAabbUnion(leftAabb, rightAabb);
}
