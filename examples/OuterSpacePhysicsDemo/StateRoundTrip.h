/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#ifndef OUTERSPACE_PHYSICS_DEMO_STATE_ROUNDTRIP_H
#define OUTERSPACE_PHYSICS_DEMO_STATE_ROUNDTRIP_H

class PhysicsBackend;
struct SceneSpec;

/**
 * @brief Proves CaptureState/ApplyState restore a simulation exactly, in one process.
 *
 * Hash equality alone only shows the bytes came back; the third leg re-runs the same steps from the
 * restored state so a restore that is equal but not equivalent still fails.
 *
 * @param backend Backend to exercise; left Reset to scene on return.
 * @param scene Scene every leg builds.
 * @param captureStep Steps before the snapshot is taken.
 * @param divergeSteps Extra steps run before the restore, so the state provably differs first.
 * @param trackSteps Steps run after the restore and compared against the undisturbed run.
 * @return 0 if every check passed, otherwise the number of failures.
 */
int RunStateRoundTrip(PhysicsBackend& backend, const SceneSpec& scene, int captureStep,
					  int divergeSteps, int trackSteps);

#endif  //OUTERSPACE_PHYSICS_DEMO_STATE_ROUNDTRIP_H
