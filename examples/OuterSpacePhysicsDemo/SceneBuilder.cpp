/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#include "PhysicsBackend.h"

#include <cmath>
#include <cstdlib>

double DemoWorldOffset()
{
	static const double offset = getenv("OSDEMO_WORLD_OFFSET") ? atof(getenv("OSDEMO_WORLD_OFFSET")) : 0.0;
	return offset;
}

namespace
{
/// Boxes are 1 unit across; 1.2 leaves a small gap so a stack settles under gravity instead of
/// starting interpenetrated, which would measure the solver's recovery rather than steady state.
const float kSpacing = kBoxHalfExtent * 2.4f;
const float kDropHeight = 2.f;

void AddBox(SceneSpec& scene, float x, float y, float z)
{
	scene.BoxPositions.push_back(x);
	scene.BoxPositions.push_back(y);
	scene.BoxPositions.push_back(z);
}

/// Islands of stacks, built from an explicit grid side rather than a single cubic knob.
SceneSpec BuildIslandGrid(int islandsPerSide)
{
	SceneSpec scene;
	scene.Name = "Island stacks";

	const int height = IslandTowerHeight();
	const float gap = IslandSpacing();
	const float origin = -gap * (islandsPerSide - 1) * 0.5f;

	scene.IslandCount = islandsPerSide * islandsPerSide;
	scene.BoxPositions.reserve((size_t)scene.IslandCount * height * 3);
	for (int tx = 0; tx < islandsPerSide; ++tx)
		for (int tz = 0; tz < islandsPerSide; ++tz)
			for (int y = 0; y < height; ++y)
				AddBox(scene, origin + gap * tx, kDropHeight + kSpacing * y, origin + gap * tz);

	// Overhanging the default slab would free-fall instead of settling, so pairs would read 0.
	const float half = gap * (islandsPerSide - 1) * 0.5f + kSpacing * 2.f;
	scene.GroundHalfExtent = half > kDefaultGroundHalfExtent ? half : kDefaultGroundHalfExtent;
	return scene;
}

/// Clusters of island grids: two levels of separation, so the broadphase sees sparse groups of
/// dense groups rather than one uniform lattice.
SceneSpec BuildClusteredIslands(int clustersPerSide)
{
	SceneSpec scene;
	scene.Name = "Clustered islands";

	const int height = IslandTowerHeight();
	const int perCluster = ClusterIslandsPerSide();
	const float gap = IslandSpacing();
	const float clusterSpan = gap * (perCluster - 1);
	const float clusterGap = clusterSpan + ClusterGap();
	const float clusterOrigin = -clusterGap * (clustersPerSide - 1) * 0.5f;
	const float innerOrigin = -clusterSpan * 0.5f;

	scene.IslandCount = clustersPerSide * clustersPerSide * perCluster * perCluster;
	scene.BoxPositions.reserve((size_t)scene.IslandCount * height * 3);

	for (int cx = 0; cx < clustersPerSide; ++cx)
		for (int cz = 0; cz < clustersPerSide; ++cz)
		{
			const float baseX = clusterOrigin + clusterGap * cx + innerOrigin;
			const float baseZ = clusterOrigin + clusterGap * cz + innerOrigin;
			for (int tx = 0; tx < perCluster; ++tx)
				for (int tz = 0; tz < perCluster; ++tz)
					for (int y = 0; y < height; ++y)
						AddBox(scene, baseX + gap * tx, kDropHeight + kSpacing * y, baseZ + gap * tz);
		}

	const float half = clusterGap * (clustersPerSide - 1) * 0.5f + clusterSpan * 0.5f + kSpacing * 2.f;
	scene.GroundHalfExtent = half > kDefaultGroundHalfExtent ? half : kDefaultGroundHalfExtent;
	return scene;
}
}  // namespace

int IslandTowerHeight()
{
	const char* env = getenv("OSDEMO_TOWER_HEIGHT");
	const int h = env ? atoi(env) : 10;
	return h < 1 ? 1 : h;
}

int ClusterIslandsPerSide()
{
	const char* env = getenv("OSDEMO_CLUSTER_ISLANDS");
	const int n = env ? atoi(env) : 8;
	return n < 1 ? 1 : n;
}

float ClusterGap()
{
	// Clear air BETWEEN clusters, on top of the span each cluster already occupies.
	const char* env = getenv("OSDEMO_CLUSTER_GAP");
	const float g = env ? (float)atof(env) : kSpacing * 8.f;
	return g < kSpacing ? kSpacing : g;
}

float IslandSpacing()
{
	// 3.6 m between 1 m boxes: 2.6 m of clear air, so settled towers are genuinely separate islands.
	const char* env = getenv("OSDEMO_ISLAND_SPACING");
	const float s = env ? (float)atof(env) : kSpacing * 3.f;
	return s < kSpacing ? kSpacing : s;
}

SceneSpec BuildIslandStackScene(int targetBodies)
{
	if (targetBodies < 1)
		targetBodies = 1;

	const int height = IslandTowerHeight();
	const int islands = (targetBodies + height - 1) / height;
	int side = 1;
	while (side * side < islands)
		++side;
	return BuildIslandGrid(side);
}

SceneSpec BuildClusteredIslandScene(int targetBodies)
{
	if (targetBodies < 1)
		targetBodies = 1;

	const int perCluster = ClusterIslandsPerSide();
	const int bodiesPerCluster = perCluster * perCluster * IslandTowerHeight();
	const int clusters = (targetBodies + bodiesPerCluster - 1) / bodiesPerCluster;
	int side = 1;
	while (side * side < clusters)
		++side;
	return BuildClusteredIslands(side);
}

SceneSpec BuildSleepDropScene(int towersPerSide, int towerHeight, float dropHeight)
{
	if (towersPerSide < 1) towersPerSide = 1;
	if (towerHeight < 1) towerHeight = 1;

	SceneSpec scene;
	scene.Name = "Sleep drop";
	scene.IslandCount = towersPerSide * towersPerSide;

	// Near-touching and already on the ground: the tower must settle and DEACTIVATE well before
	// the dropper reaches it, or the test proves nothing about waking.
	const float stackStep = kBoxHalfExtent * 2.f + 0.02f;
	const float gap = IslandSpacing();
	const float origin = -gap * (towersPerSide - 1) * 0.5f;
	const float top = kBoxHalfExtent + stackStep * (towerHeight - 1);

	for (int tx = 0; tx < towersPerSide; ++tx)
		for (int tz = 0; tz < towersPerSide; ++tz)
			for (int y = 0; y < towerHeight; ++y)
				AddBox(scene, origin + gap * tx, kBoxHalfExtent + stackStep * y, origin + gap * tz);

	// Droppers come after every tower box, so a caller can split the two groups by index alone.
	for (int tx = 0; tx < towersPerSide; ++tx)
		for (int tz = 0; tz < towersPerSide; ++tz)
			AddBox(scene, origin + gap * tx, top + dropHeight, origin + gap * tz);

	const float half = gap * (towersPerSide - 1) * 0.5f + kSpacing * 2.f;
	scene.GroundHalfExtent = half > kDefaultGroundHalfExtent ? half : kDefaultGroundHalfExtent;
	return scene;
}

void DemoKinematicPose(const DemoKinematicBody& body, double time, float outPos[3],
					   float outQuat[4], float outLinVel[3], float outAngVel[3])
{
	const double t = time - (double)body.startDelay;
	const double active = t > 0.0 ? t : 0.0;

	double s = 0.0;
	double dsdt = 0.0;
	if (body.period > 0.f && active > 0.0)
	{
		// Triangle, not sine: a sine's turn-around velocity is zero exactly where the attached
		// body is meant to be dragged hardest, which would hide the behaviour this drives.
		const double phase = active / (double)body.period;
		const double frac = phase - floor(phase);
		s = frac < 0.5 ? 2.0 * frac : 2.0 * (1.0 - frac);
		dsdt = (frac < 0.5 ? 2.0 : -2.0) / (double)body.period;
	}

	for (int a = 0; a < 3; ++a)
	{
		outPos[a] = (float)((double)body.position[a] + (double)body.travel[a] * s);
		outLinVel[a] = (float)((double)body.travel[a] * dsdt);
	}

	const double angle = (double)body.spinRate * active;
	outQuat[0] = 0.f;
	outQuat[1] = (float)sin(angle * 0.5);
	outQuat[2] = 0.f;
	outQuat[3] = (float)cos(angle * 0.5);
	outAngVel[0] = 0.f;
	outAngVel[1] = active > 0.0 ? body.spinRate : 0.f;
	outAngVel[2] = 0.f;
}

SceneSpec BuildScene(int kind, int scale)
{
	SceneSpec scene;
	scene.Name = "";

	if (scale < 2)
		scale = 2;

	switch (kind)
	{
	case SCENE_MULTIPLE_STACKS:
	{
		// Separate towers rather than one block: the broadphase sees several disjoint clusters,
		// which is a different tree shape from one dense island and is where an LBVH's spatial
		// partitioning actually has to work.
		scene.Name = "Multiple stacks";
		const int towersPerSide = scale / 2 < 2 ? 2 : scale / 2;
		const int towerHeight = scale;
		const float towerGap = kSpacing * 3.f;
		const float origin = -towerGap * (towersPerSide - 1) * 0.5f;

		for (int tx = 0; tx < towersPerSide; ++tx)
			for (int tz = 0; tz < towersPerSide; ++tz)
				for (int y = 0; y < towerHeight; ++y)
					AddBox(scene, origin + towerGap * tx, kDropHeight + kSpacing * y,
						   origin + towerGap * tz);
		break;
	}

	case SCENE_ISLAND_STACKS:
		// Tower height is fixed here, so scale only grows the island COUNT - the per-island physics
		// stays identical as the scene scales, which is what makes a scaling sweep comparable.
		return BuildIslandGrid(scale);

	case SCENE_CLUSTERED_ISLANDS:
		return BuildClusteredIslands(scale);

	case SCENE_PYRAMID:
	{
		// Contact count per body climbs toward the base, so the solver sees a real dependency
		// chain rather than the uniform load a cube gives it.
		scene.Name = "Pyramid";
		for (int layer = 0; layer < scale; ++layer)
		{
			const int side = scale - layer;
			const float offset = -kSpacing * (side - 1) * 0.5f;
			for (int x = 0; x < side; ++x)
				for (int z = 0; z < side; ++z)
					AddBox(scene, offset + kSpacing * x, kDropHeight + kSpacing * layer,
						   offset + kSpacing * z);
		}
		break;
	}

	case SCENE_WALL:
	{
		// Two bricks deep and tall: topples readily, so it exercises sustained resolution rather
		// than settling once and going quiet.
		scene.Name = "Wall";
		const int width = scale * 2;
		const int height = scale;
		const float offset = -kSpacing * (width - 1) * 0.5f;
		for (int y = 0; y < height; ++y)
			for (int x = 0; x < width; ++x)
				for (int d = 0; d < 2; ++d)
					AddBox(scene, offset + kSpacing * x + (y % 2 ? kSpacing * 0.5f : 0.f),
						   kDropHeight + kSpacing * y, kSpacing * d);
		break;
	}

	case SCENE_CHAINS:
	{
		// Hanging chains: every link is constrained to the one above, so this scene is only
		// meaningful on a backend that solves joints.
		scene.Name = "Chains (constraints)";
		const int chains = scale;
		const int links = scale;
		const float chainGap = kSpacing * 2.f;
		const float origin = -chainGap * (chains - 1) * 0.5f;

		// A bit-exactly vertical column of ball joints is a degenerate inverted pendulum: an
		// order-independent solver finds no asymmetry to tip it and it stands forever. The default
		// 0.1 mm/link lean makes the scene well-posed; OSDEMO_CHAIN_PERTURB overrides it.
		const char* perturbEnv = getenv("OSDEMO_CHAIN_PERTURB");
		const float perturb = perturbEnv ? (float)atof(perturbEnv) : 1e-4f;

		for (int c = 0; c < chains; ++c)
		{
			const int first = scene.BoxCount();
			for (int l = 0; l < links; ++l)
				AddBox(scene, origin + chainGap * c + perturb * (float)(links - l),
					   kDropHeight + kSpacing * (links - l), 0.f);

			// Link each box to the one above it. The topmost is left free rather than pinned to
			// the world, so the whole chain falls and swings instead of hanging statically.
			for (int l = 1; l < links; ++l)
			{
				DemoConstraint c2;
				c2.bodyA = first + l - 1;
				c2.bodyB = first + l;
				c2.pivotA[0] = 0.f; c2.pivotA[1] = -kBoxHalfExtent * 1.2f; c2.pivotA[2] = 0.f;
				c2.pivotB[0] = 0.f; c2.pivotB[1] = kBoxHalfExtent * 1.2f;  c2.pivotB[2] = 0.f;
				scene.Constraints.push_back(c2);
			}
		}
		break;
	}

	case SCENE_KINEMATIC_ANCHORS:
	{
		// Every path an infinite-mass body can take through this pipeline, in one scene: joints
		// anchored to the static ground, a kinematic sweeper, and a jointed body that must WAKE.
		scene.Name = "Kinematic + anchors";
		const int links = scale < 4 ? 4 : (scale > 8 ? 8 : scale);
		const float pivotUp = kBoxHalfExtent * 1.2f;
		const float linkStep = pivotUp * 2.f;
		// Chosen so the bottom link lands at 2.6 m whatever the link count: a free-fall failure
		// then reads as 0.5 m and cannot be confused with a longer chain simply reaching lower.
		const float anchorY = 2.f + 1.2f * (float)links;

		// 1. Chains hung from the STATIC ground. Ground-local pivots, so the y offset carries the
		// slab's own -kGroundThickness centre.
		for (int c = 0; c < 2; ++c)
		{
			const float x = -6.f + 4.f * (float)c;
			const float z = -6.f;
			const int first = scene.BoxCount();
			for (int l = 0; l < links; ++l)
				AddBox(scene, x, anchorY - pivotUp - linkStep * (float)l, z);

			DemoConstraint anchor;
			anchor.bodyA = kDemoAnchorGround;
			anchor.bodyB = first;
			anchor.pivotA[0] = x; anchor.pivotA[1] = anchorY + kGroundThickness; anchor.pivotA[2] = z;
			anchor.pivotB[0] = 0.f; anchor.pivotB[1] = pivotUp; anchor.pivotB[2] = 0.f;
			scene.Constraints.push_back(anchor);

			for (int l = 1; l < links; ++l)
			{
				DemoConstraint link;
				link.bodyA = first + l - 1;
				link.bodyB = first + l;
				link.pivotA[0] = 0.f; link.pivotA[1] = -pivotUp; link.pivotA[2] = 0.f;
				link.pivotB[0] = 0.f; link.pivotB[1] = pivotUp;  link.pivotB[2] = 0.f;
				scene.Constraints.push_back(link);
			}
		}

		// 2. Loose boxes in the sweeper's path; they must be pushed and the sweeper must not be.
		for (int i = 0; i < 5; ++i)
			AddBox(scene, -4.f + 2.f * (float)i, kBoxHalfExtent, 0.f);

		// 3. A box already at rest on the ground, jointed to a kinematic body that only starts
		// moving once the box has had time to deactivate. This is the joint-wake case.
		const int sleeper = scene.BoxCount();
		AddBox(scene, 0.f, kBoxHalfExtent, -12.f);

		// 4. A chain hung from a kinematic body, i.e. all three mechanisms at once.
		const int hung = scene.BoxCount();
		for (int l = 0; l < 3; ++l)
			AddBox(scene, 0.f, anchorY - pivotUp - linkStep * (float)l, 8.f);

		DemoKinematicBody sweeper;
		sweeper.position[0] = -10.f; sweeper.position[1] = kBoxHalfExtent; sweeper.position[2] = 0.f;
		sweeper.travel[0] = 20.f;
		sweeper.period = 8.f;
		scene.Kinematics.push_back(sweeper);

		// Held still for longer than the 120-step deactivation window, so the box it holds is
		// provably asleep before the anchor ever moves.
		DemoKinematicBody holder;
		holder.position[0] = 0.f; holder.position[1] = 4.f; holder.position[2] = -12.f;
		holder.travel[0] = 6.f;
		holder.period = 12.f;
		holder.startDelay = 3.f;
		scene.Kinematics.push_back(holder);

		DemoKinematicBody carrier;
		carrier.position[0] = 0.f; carrier.position[1] = anchorY + pivotUp; carrier.position[2] = 8.f;
		carrier.travel[0] = 8.f;
		carrier.period = 10.f;
		scene.Kinematics.push_back(carrier);

		DemoConstraint hold;
		hold.bodyA = DemoAnchorKinematic(1);
		hold.bodyB = sleeper;
		hold.pivotA[0] = 0.f; hold.pivotA[1] = kBoxHalfExtent - 4.f; hold.pivotA[2] = 0.f;
		hold.pivotB[0] = 0.f; hold.pivotB[1] = 0.f; hold.pivotB[2] = 0.f;
		scene.Constraints.push_back(hold);

		DemoConstraint carry;
		carry.bodyA = DemoAnchorKinematic(2);
		carry.bodyB = hung;
		carry.pivotA[0] = 0.f; carry.pivotA[1] = -pivotUp; carry.pivotA[2] = 0.f;
		carry.pivotB[0] = 0.f; carry.pivotB[1] = pivotUp; carry.pivotB[2] = 0.f;
		scene.Constraints.push_back(carry);

		for (int l = 1; l < 3; ++l)
		{
			DemoConstraint link;
			link.bodyA = hung + l - 1;
			link.bodyB = hung + l;
			link.pivotA[0] = 0.f; link.pivotA[1] = -pivotUp; link.pivotA[2] = 0.f;
			link.pivotB[0] = 0.f; link.pivotB[1] = pivotUp;  link.pivotB[2] = 0.f;
			scene.Constraints.push_back(link);
		}
		break;
	}

	case SCENE_SINGLE_STACK:
	default:
	{
		scene.Name = "Single stack";
		const float offset = -kSpacing * scale * 0.5f;
		for (int y = 0; y < scale; ++y)
			for (int x = 0; x < scale; ++x)
				for (int z = 0; z < scale; ++z)
					AddBox(scene, offset + kSpacing * x, kDropHeight + kSpacing * y,
						   offset + kSpacing * z);
		break;
	}
	}

	return scene;
}
