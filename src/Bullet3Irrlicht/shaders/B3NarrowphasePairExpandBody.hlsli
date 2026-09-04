// Expands broadphase body pairs into LEAF pairs (bodyA, bodyB, childA, childB): one per child
// combination when either side is a compound, -1 for a body colliding as its own shape.
// A triangle-mesh leaf path plugs in here: append (bodyA, bodyB, -1, triangleIndex) per
// candidate triangle and let the leaf clip kernel resolve the right-hand side by shape type.

// Included by B3NarrowphasePairExpand.hlsl (f32) and B3NarrowphasePairExpandDS.hlsl (df64); the
// body stride differs between the two, which is the only reason there are two top-level files.
#include "B3Precision.hlsli"

#define WG_SIZE 64

#define SHAPE_CONVEX_HULL 3
#define SHAPE_COMPOUND_OF_CONVEX_HULLS 6

struct b3Collidable
{
	int numChildShapes;
	float radius;
	int shapeType;
	int shapeIndex;   // first child in ChildShapes for a compound
};

struct ExpandParams
{
	uint numPairs;
	uint pad0;
	uint pad1;
	uint pad2;
};

StructuredBuffer<ExpandParams> Params : register(t0);
StructuredBuffer<b3RigidBodyData> Bodies : register(t1);
StructuredBuffer<b3Collidable> Collidables : register(t2);
StructuredBuffer<uint2> Pairs : register(t3);

// Append buffers must own u0 - see the narrowphase notes.
AppendStructuredBuffer<int4> OutLeafPairs : register(u0);

[numthreads(WG_SIZE, 1, 1)]
void CSExpandPairs(uint3 tid : SV_DispatchThreadID)
{
	const uint pairIndex = tid.x;
	if (pairIndex >= Params[0].numPairs)
		return;

	const uint2 pair = Pairs[pairIndex];
	const b3Collidable ca = Collidables[Bodies[pair.x].collidableIdx];
	const b3Collidable cb = Collidables[Bodies[pair.y].collidableIdx];

	const bool compoundA = ca.shapeType == SHAPE_COMPOUND_OF_CONVEX_HULLS;
	const bool compoundB = cb.shapeType == SHAPE_COMPOUND_OF_CONVEX_HULLS;
	const int countA = compoundA ? ca.numChildShapes : 1;
	const int countB = compoundB ? cb.numChildShapes : 1;

	// No local array is indexed here, so these stay real runtime loops (PREFER_FLOW_CONTROL).
	for (int i = 0; i < countA; ++i)
	{
		for (int j = 0; j < countB; ++j)
		{
			int4 leaf;
			leaf.x = (int)pair.x;
			leaf.y = (int)pair.y;
			leaf.z = compoundA ? ca.shapeIndex + i : -1;
			leaf.w = compoundB ? cb.shapeIndex + j : -1;
			OutLeafPairs.Append(leaf);
		}
	}
}
