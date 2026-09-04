// Port of Bullet3OpenCL parallelLinearBvh.cl - Karras binary radix tree construction.
// Phase 4b: adjacent-pair common prefixes, leaf/internal node linking, depth, AABB fitting.
//
// The original keys are 64-bit (morton << 32 | leafIndex, which breaks duplicate morton codes
// and keeps the sequence strictly increasing). SM5.0 has no 64-bit ints, so a key is a uint2:
// .x = high 32 (morton), .y = low 32 (leaf index). All prefix maths below is emulated on that.

// Included by B3LbvhTree.hlsl (f32) and B3LbvhTreeDS.hlsl (df64) - see B3Precision.hlsli.

#include "B3Precision.hlsli"

#define WG_SIZE 128
#define B3_PLBVH_INVALID_COMMON_PREFIX 128
#define B3_PLBVH_ROOT_NODE_MARKER -1

struct TreeParams
{
	uint numLeafNodes;
	uint numInternalNodes;
	int processedDistance;
	int pad0;
};

StructuredBuffer<TreeParams> Params : register(t0);
StructuredBuffer<uint2> MortonCodes : register(t1);       // (key, leafIndex), sorted
StructuredBuffer<uint2> CommonPrefixes : register(t2);    // 64-bit as uint2
StructuredBuffer<int> CommonPrefixLengths : register(t3);
StructuredBuffer<int> ChildNodesIn : register(t4);        // 2 ints per internal node
StructuredBuffer<int> InternalParentsIn : register(t5);
StructuredBuffer<int> DistanceFromRootIn : register(t6);
StructuredBuffer<b3Aabb> LeafAabbs : register(t7);
// No separate SRV for internal AABBs: D3D11 refuses the same resource as SRV and UAV in one
// dispatch (it nulls one binding), so children are read back through the UAV below instead.

RWStructuredBuffer<uint2> OutCommonPrefixes : register(u0);
RWStructuredBuffer<int> OutCommonPrefixLengths : register(u1);
RWStructuredBuffer<int> OutChildNodes : register(u2);
RWStructuredBuffer<int> OutLeafParents : register(u3);
RWStructuredBuffer<int> OutInternalParents : register(u4);
RWStructuredBuffer<int> OutRootIndex : register(u5);
RWStructuredBuffer<int> OutDistanceFromRoot : register(u6);
RWStructuredBuffer<b3Aabb> OutInternalAabbs : register(u7);

int isLeafNode(int index) { return (index >> 31) == 0; }
int indexMarkerRemoved(int index) { return index & (~0x80000000); }
int indexWithMarker(bool isLeaf, int index) { return isLeaf ? index : (index | 0x80000000); }

uint clz32(uint v)
{
	// firstbithigh returns 0xFFFFFFFF for 0, so the zero case is handled separately.
	return (v == 0) ? 32u : (31u - firstbithigh(v));
}

uint commonPrefixLength64(uint2 a, uint2 b)
{
	const uint2 d = uint2(a.x ^ b.x, a.y ^ b.y);
	return (d.x != 0) ? clz32(d.x) : (32u + clz32(d.y));
}

//! Shared high bits, with everything after the common prefix masked off. Only (a & b) is
//! strictly required by the algorithm; the mask keeps the values readable when debugging.
uint2 commonPrefix64(uint2 a, uint2 b)
{
	const uint len = commonPrefixLength64(a, b);
	const uint2 sharedBits = uint2(a.x & b.x, a.y & b.y);

	// bitmask = ~0 << (64 - len), built per half because HLSL shifts of >=32 are undefined.
	const uint s = 64u - len;
	uint2 mask;
	if (s == 0u)
		mask = uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
	else if (s < 32u)
		mask = uint2(0xFFFFFFFFu, 0xFFFFFFFFu << s);
	else if (s < 64u)
		mask = uint2(0xFFFFFFFFu << (s - 32u), 0u);
	else
		mask = uint2(0u, 0u);

	return uint2(sharedBits.x & mask.x, sharedBits.y & mask.y);
}

int getSharedPrefixLength(uint2 prefixA, int lengthA, uint2 prefixB, int lengthB)
{
	return min((int)commonPrefixLength64(prefixA, prefixB), min(lengthA, lengthB));
}

//! One prefix per adjacent leaf pair. Appending the leaf index to the morton code is what makes
//! duplicate morton codes safe - the tree build is undefined with duplicates.
[numthreads(WG_SIZE, 1, 1)]
void CSComputeAdjacentPairCommonPrefix(uint3 tid : SV_DispatchThreadID)
{
	const uint numInternalNodes = Params[0].numInternalNodes;
	const uint i = tid.x;
	if (i >= numInternalNodes)
		return;

	// (i + 1) is always in range: numInternalNodes == numLeafNodes - 1.
	const uint2 left = uint2(MortonCodes[i].x, i);
	const uint2 right = uint2(MortonCodes[i + 1].x, i + 1);

	OutCommonPrefixes[i] = commonPrefix64(left, right);
	OutCommonPrefixLengths[i] = (int)commonPrefixLength64(left, right);
}

//! Each leaf attaches to whichever adjacent split has the higher common prefix.
[numthreads(WG_SIZE, 1, 1)]
void CSBuildLeafNodes(uint3 tid : SV_DispatchThreadID)
{
	const int numLeafNodes = (int)Params[0].numLeafNodes;
	const int numInternalNodes = numLeafNodes - 1;
	const int leaf = (int)tid.x;
	if (leaf >= numLeafNodes)
		return;

	const int leftSplit = leaf - 1;
	const int rightSplit = leaf;

	const int leftPrefix = (leftSplit >= 0) ? CommonPrefixLengths[leftSplit] : B3_PLBVH_INVALID_COMMON_PREFIX;
	const int rightPrefix = (rightSplit < numInternalNodes) ? CommonPrefixLengths[rightSplit] : B3_PLBVH_INVALID_COMMON_PREFIX;

	bool leftIsHigher = (leftPrefix > rightPrefix);
	// Edge nodes: exactly one side is invalid, never both.
	if (leftPrefix == B3_PLBVH_INVALID_COMMON_PREFIX) leftIsHigher = false;
	if (rightPrefix == B3_PLBVH_INVALID_COMMON_PREFIX) leftIsHigher = true;

	const int parent = leftIsHigher ? leftSplit : rightSplit;
	OutLeafParents[leaf] = parent;

	// If the left split is the parent, this leaf is its right child, and vice versa.
	const int isRightChild = leftIsHigher ? 1 : 0;
	OutChildNodes[2 * parent + isRightChild] = indexWithMarker(true, leaf);
}

//! Links internal nodes by binary-searching left and right for the nearest node whose shared
//! prefix with this one is shorter. The node with no such neighbour on either side is the root.
[numthreads(WG_SIZE, 1, 1)]
void CSBuildInternalNodes(uint3 tid : SV_DispatchThreadID)
{
	const int numInternalNodes = (int)Params[0].numInternalNodes;
	const int nodeIndex = (int)tid.x;
	if (nodeIndex >= numInternalNodes)
		return;

	const uint2 nodePrefix = CommonPrefixes[nodeIndex];
	const int nodePrefixLength = CommonPrefixLengths[nodeIndex];

	int leftIndex = -1;
	{
		int lower = 0;
		int upper = nodeIndex - 1;
		while (lower <= upper)
		{
			const int mid = (lower + upper) / 2;
			const int midShared = getSharedPrefixLength(nodePrefix, nodePrefixLength, CommonPrefixes[mid], CommonPrefixLengths[mid]);
			if (midShared < nodePrefixLength)
			{
				const int right = mid + 1;
				if (right < nodeIndex)
				{
					const int rightShared = getSharedPrefixLength(nodePrefix, nodePrefixLength, CommonPrefixes[right], CommonPrefixLengths[right]);
					if (rightShared < nodePrefixLength)
					{
						lower = right;
						leftIndex = right;
					}
					else
					{
						leftIndex = mid;
						break;
					}
				}
				else
				{
					leftIndex = mid;
					break;
				}
			}
			else
			{
				upper = mid - 1;
			}
		}
	}

	int rightIndex = -1;
	{
		int lower = nodeIndex + 1;
		int upper = numInternalNodes - 1;
		while (lower <= upper)
		{
			const int mid = (lower + upper) / 2;
			const int midShared = getSharedPrefixLength(nodePrefix, nodePrefixLength, CommonPrefixes[mid], CommonPrefixLengths[mid]);
			if (midShared < nodePrefixLength)
			{
				const int left = mid - 1;
				if (left > nodeIndex)
				{
					const int leftShared = getSharedPrefixLength(nodePrefix, nodePrefixLength, CommonPrefixes[left], CommonPrefixLengths[left]);
					if (leftShared < nodePrefixLength)
					{
						upper = left;
						rightIndex = left;
					}
					else
					{
						rightIndex = mid;
						break;
					}
				}
				else
				{
					rightIndex = mid;
					break;
				}
			}
			else
			{
				lower = mid + 1;
			}
		}
	}

	if (leftIndex != -1 || rightIndex != -1)
	{
		int parent;
		if (leftIndex != -1 && rightIndex != -1)
			parent = (CommonPrefixLengths[leftIndex] > CommonPrefixLengths[rightIndex]) ? leftIndex : rightIndex;
		else
			parent = (leftIndex != -1) ? leftIndex : rightIndex;

		OutInternalParents[nodeIndex] = parent;

		const int isRightChild = (parent == leftIndex) ? 1 : 0;
		OutChildNodes[2 * parent + isRightChild] = indexWithMarker(false, nodeIndex);
	}
	else
	{
		OutInternalParents[nodeIndex] = B3_PLBVH_ROOT_NODE_MARKER;
		OutRootIndex[0] = indexWithMarker(false, nodeIndex);
	}
}

//! Depth of each internal node, walked up to the root. Needed because AABB fitting has to run
//! deepest-first, one depth level per dispatch.
[numthreads(WG_SIZE, 1, 1)]
void CSFindDistanceFromRoot(uint3 tid : SV_DispatchThreadID)
{
	const int numInternalNodes = (int)Params[0].numInternalNodes;
	const int nodeIndex = (int)tid.x;
	if (nodeIndex >= numInternalNodes)
		return;

	int distance = 0;
	int parent = InternalParentsIn[nodeIndex];
	// Bounded by the tree depth; the guard stops a malformed tree hanging the GPU.
	while (parent != B3_PLBVH_ROOT_NODE_MARKER && distance < numInternalNodes)
	{
		parent = InternalParentsIn[parent];
		++distance;
	}

	OutDistanceFromRoot[nodeIndex] = distance;
}

//! Merges child AABBs into each internal node at exactly one depth. The host dispatches this
//! from the deepest level up to 0, so a node's children are always already fitted.
[numthreads(WG_SIZE, 1, 1)]
void CSBuildTreeAabbs(uint3 tid : SV_DispatchThreadID)
{
	const int numInternalNodes = (int)Params[0].numInternalNodes;
	const int processedDistance = Params[0].processedDistance;
	const int nodeIndex = (int)tid.x;
	if (nodeIndex >= numInternalNodes)
		return;

	if (DistanceFromRootIn[nodeIndex] != processedDistance)
		return;

	int leftChild = ChildNodesIn[2 * nodeIndex + 0];
	int rightChild = ChildNodesIn[2 * nodeIndex + 1];

	const bool leftIsLeaf = isLeafNode(leftChild) != 0;
	const bool rightIsLeaf = isLeafNode(rightChild) != 0;

	leftChild = indexMarkerRemoved(leftChild);
	rightChild = indexMarkerRemoved(rightChild);

	// A leaf's AABB is indexed by the rigid body it came from, not by the sorted leaf slot.
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
