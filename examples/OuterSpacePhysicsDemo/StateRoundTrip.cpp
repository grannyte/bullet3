/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
#include "StateRoundTrip.h"

#include "PhysicsBackend.h"

#include <cstdio>
#include <vector>

namespace
{
const float kRoundTripStep = 1.f / 60.f;

/**
 * @brief Resets and steps a backend, returning the state hash.
 * @param backend Backend to run.
 * @param scene Scene to build.
 * @param steps Steps to run after the reset.
 * @param out Receives the hash.
 * @return False if the reset or the hash failed.
 */
bool RunLeg(PhysicsBackend& backend, const SceneSpec& scene, int steps, unsigned long long& out)
{
	if (!backend.Reset(scene))
		return false;
	for (int i = 0; i < steps; ++i)
		backend.Step(kRoundTripStep);
	return backend.HashState(out);
}

/**
 * @brief Steps an already-built backend and returns the state hash.
 * @param backend Backend to run.
 * @param steps Steps to run.
 * @param out Receives the hash.
 * @return False if the hash failed.
 */
bool StepAndHash(PhysicsBackend& backend, int steps, unsigned long long& out)
{
	for (int i = 0; i < steps; ++i)
		backend.Step(kRoundTripStep);
	return backend.HashState(out);
}

/**
 * @brief Prints one check and counts it.
 * @param label What was checked.
 * @param ok Whether it held.
 * @param detail Extra text, printed as-is.
 * @param failures Failure tally, incremented when ok is false.
 */
void Check(const char* label, bool ok, const char* detail, int& failures)
{
	printf("  %-34s %s  %s\n", label, ok ? "PASS" : "FAIL", detail);
	if (!ok)
		++failures;
	fflush(stdout);
}
}  // namespace

int RunStateRoundTrip(PhysicsBackend& backend, const SceneSpec& scene, int captureStep,
					  int divergeSteps, int trackSteps)
{
	if (captureStep < 1)
		captureStep = 1;
	if (divergeSteps < 1)
		divergeSteps = 1;
	if (trackSteps < 0)
		trackSteps = 0;

	printf("\nState round-trip: %s on %s (%d bodies), capture@%d diverge+%d track+%d\n",
		   backend.Name(), scene.Name, scene.BoxCount(), captureStep, divergeSteps, trackSteps);
	fflush(stdout);

	int failures = 0;
	char detail[256];

	// Leg A: the undisturbed trajectory. Its snapshot is what the other two legs restore, and its
	// end hash is what a restored run has to reproduce.
	unsigned long long captureHash = 0;
	if (!RunLeg(backend, scene, captureStep, captureHash))
	{
		printf("  backend cannot hash or reset - round-trip not applicable\n");
		return 1;
	}

	std::vector<unsigned char> snapshot;
	if (!backend.CaptureState(snapshot) || snapshot.empty())
	{
		printf("  CaptureState declined - backend does not support snapshots\n");
		return 1;
	}
	// Awake tally at capture: with everything awake the snapshot never exercises its sleep block, so
	// a pass would say nothing about whether sleep state has to be in it.
	backend.RefreshAwakeCount();
	sprintf(detail, "%d bytes for %d bodies, %d awake at capture", (int)snapshot.size(),
			backend.BodyCount() + 1, backend.AwakeBodyCount());
	Check("CaptureState", true, detail, failures);

	unsigned long long undisturbedHash = 0;
	StepAndHash(backend, trackSteps, undisturbedHash);

	// Leg B: restore onto the very state that was captured. Isolates the restore's own cost from
	// the intervening steps, and is the apples-to-apples reference for leg C.
	unsigned long long legBStart = 0;
	RunLeg(backend, scene, captureStep, legBStart);
	Check("re-run reaches the same state", legBStart == captureHash, "", failures);
	const bool bApplied = backend.ApplyState(&snapshot[0], snapshot.size());
	unsigned long long legBRestored = 0;
	backend.HashState(legBRestored);
	unsigned long long selfAppliedHash = 0;
	StepAndHash(backend, trackSteps, selfAppliedHash);

	// Leg C: the real case - restore over a state that has genuinely moved on.
	unsigned long long divergedHash = 0;
	RunLeg(backend, scene, captureStep + divergeSteps, divergedHash);
	// A scene that has already settled and slept produces a bit-identical state no matter how long
	// it runs, and restoring onto an identical state would prove nothing.
	int extraSteps = 0;
	while (divergedHash == captureHash && extraSteps < divergeSteps * 9)
	{
		StepAndHash(backend, divergeSteps, divergedHash);
		extraSteps += divergeSteps;
	}
	sprintf(detail, "%016llx vs %016llx after +%d%s", divergedHash, captureHash,
			divergeSteps + extraSteps,
			divergedHash == captureHash ? " (scene is asleep - capture earlier)" : "");
	Check("state actually diverged first", divergedHash != captureHash, detail, failures);

	const bool cApplied = backend.ApplyState(&snapshot[0], snapshot.size());
	Check("ApplyState accepted", bApplied && cApplied, "", failures);

	unsigned long long restoredHash = 0;
	backend.HashState(restoredHash);
	sprintf(detail, "%016llx vs %016llx", restoredHash, captureHash);
	Check("hash after restore == at capture", restoredHash == captureHash && legBRestored == captureHash,
		  detail, failures);

	unsigned long long trackedHash = 0;
	StepAndHash(backend, trackSteps, trackedHash);
	sprintf(detail, "%016llx vs %016llx", trackedHash, selfAppliedHash);
	Check("dynamic equivalence (vs restore)", trackedHash == selfAppliedHash, detail, failures);
	sprintf(detail, "%016llx vs %016llx", trackedHash, undisturbedHash);
	// A separate line from the one above: the undisturbed run never took the forced tree rebuild a
	// restore does, so this one also measures that, not just the restore.
	Check("dynamic equivalence (vs undisturbed)", trackedHash == undisturbedHash, detail, failures);

	// Refusals. Each must leave the simulation untouched, so the hash is re-read after all three.
	unsigned long long beforeRefusals = 0;
	backend.HashState(beforeRefusals);

	Check("refuses a truncated payload",
		  !backend.ApplyState(&snapshot[0], snapshot.size() - 1)
			  && !backend.ApplyState(&snapshot[0], 8), "", failures);

	std::vector<unsigned char> flipped = snapshot;
	// Byte 20 is the header's flags word; bit 0 is the df64 marker, so this is a payload claiming
	// the precision this backend is not running.
	flipped[20] ^= 0x01;
	Check("refuses a precision mismatch", !backend.ApplyState(&flipped[0], flipped.size()), "",
		  failures);

	unsigned long long afterRefusals = 0;
	backend.HashState(afterRefusals);
	Check("refusals left the state untouched", beforeRefusals == afterRefusals, "", failures);

	// Body-count mismatch needs a differently sized world, so it runs last and rebuilds after.
	SceneSpec smaller = scene;
	smaller.BoxPositions.resize((smaller.BoxPositions.size() / 6) * 3);
	smaller.Constraints.clear();
	if (smaller.BoxCount() > 0 && smaller.BoxCount() != scene.BoxCount()
		&& backend.Reset(smaller))
		Check("refuses a different body count", !backend.ApplyState(&snapshot[0], snapshot.size()),
			  "", failures);

	backend.Reset(scene);
	printf("  %s\n", failures ? "STATE ROUND-TRIP FAILED" : "State round-trip: every check passed.");
	fflush(stdout);
	return failures;
}
