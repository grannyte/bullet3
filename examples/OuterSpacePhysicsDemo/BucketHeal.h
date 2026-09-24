/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#ifndef OUTERSPACE_PHYSICS_DEMO_BUCKET_HEAL_H
#define OUTERSPACE_PHYSICS_DEMO_BUCKET_HEAL_H

class GpuPhysicsBackend;
struct SceneSpec;

/**
 * @brief Proves a desync is narrowed by bucket hash and then healed back to a bit-exact match.
 *
 * Runs both peers inside one backend by swapping full snapshots between them, so the healing side
 * never sees a state the authority did not actually reach.
 *
 * @param backend Backend to exercise; left Reset to scene on return.
 * @param scene Scene both peers build.
 * @param settleSteps Steps run before the divergence is injected.
 * @param healSteps Ticks the heal is given to re-match before it is called a failure.
 * @return 0 if every check passed, otherwise the number of failures.
 */
int RunBucketHealTest(GpuPhysicsBackend& backend, const SceneSpec& scene, int settleSteps,
					  int healSteps);

#endif  //OUTERSPACE_PHYSICS_DEMO_BUCKET_HEAL_H
