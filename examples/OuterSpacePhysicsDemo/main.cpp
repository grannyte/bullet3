/*******************************************************
* Copyright (C) 2026 David Delisle granyte@hotmail.com
*
* This file is part of Outerspace.
*
* Outerspace can not be copied and/or distributed without the express
* permission of David Delisle
*******************************************************/
//
// Physics backend comparison harness. Deliberately does NOT boot IEngineFramework - it opens a
// bare Irrlicht device so the numbers measure physics, not the engine's per-frame work.

#include "CpuBulletBackend.h"
#include "DemoNetSync.h"
#include "GpuInstanceCuller.h"
#include "GpuPhysicsBackend.h"
#include "HealStubBackend.h"

#include <irrlicht.h>
#include <IComputebuffer.h>
#include <IInstancedMeshSceneNode.h>

#include <windows.h>   // SetCurrentDirectoryW/GetFileAttributesW for the Data directory
#include <ppl.h>       // the repo's parallel primitive; Bullet MT here uses the PPL scheduler too

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <map>
#include <string>
#include <vector>

#pragma comment(lib, "Irrlicht.lib")

using namespace irr;

namespace
{
const int kDefaultBoxesPerSide = 10;
const float kFixedStep = 1.f / 60.f;

/// Below this the serial loop already costs less than the task overhead.
const size_t kParallelTransformThreshold = 4096;
const size_t kTransformGrain = 1024;

/// Rolling mean over the last N steps. A single frame's timing is dominated by scheduling noise;
/// the comparison is only meaningful averaged.
class RollingAverage
{
public:
	RollingAverage() : m_next(0), m_count(0), m_sum(0.0) { }

	void Add(double value)
	{
		if (m_count < kWindow)
		{
			m_samples[m_count++] = value;
			m_sum += value;
		}
		else
		{
			m_sum -= m_samples[m_next];
			m_samples[m_next] = value;
			m_sum += value;
			m_next = (m_next + 1) % kWindow;
		}
	}

	double Mean() const { return m_count ? m_sum / m_count : 0.0; }
	void Clear() { m_next = 0; m_count = 0; m_sum = 0.0; }

private:
	static const int kWindow = 120;
	double m_samples[kWindow];
	int m_next;
	int m_count;
	double m_sum;
};

/// Button ids the UI and the event receiver agree on.
enum DemoButtonId
{
	BUTTON_BACKEND_BASE = 100,   // + backend index
	BUTTON_SCENE_BASE = 150,     // + scene kind
	BUTTON_RESET = 200,
	BUTTON_MORE_BOXES = 201,
	BUTTON_FEWER_BOXES = 202,
	BUTTON_NET_HOST = 203,
	BUTTON_NET_JOIN = 204,
	BUTTON_NET_DISCONNECT = 205,
	BUTTON_NET_MODE = 206
};

/// Keeps keyboard and GUI state the main loop polls; Irrlicht delivers both through a receiver.
class DemoInput : public IEventReceiver
{
public:
	DemoInput() : m_clickedId(-1) { for (u32 i = 0; i < KEY_KEY_CODES_COUNT; ++i) m_pressed[i] = false; }

	virtual bool OnEvent(const SEvent& event)
	{
		if (event.EventType == EET_KEY_INPUT_EVENT && event.KeyInput.PressedDown)
			m_pressed[event.KeyInput.Key] = true;

		if (event.EventType == EET_GUI_EVENT
			&& event.GUIEvent.EventType == gui::EGET_BUTTON_CLICKED
			&& event.GUIEvent.Caller)
		{
			m_clickedId = event.GUIEvent.Caller->getID();
		}
		return false;
	}

	/// One-shot: reading a key consumes it, so a held key does not retrigger every frame.
	bool Consume(EKEY_CODE key)
	{
		const bool was = m_pressed[key];
		m_pressed[key] = false;
		return was;
	}

	/// One-shot for the same reason; -1 when nothing was clicked this frame.
	s32 ConsumeClick()
	{
		const s32 id = m_clickedId;
		m_clickedId = -1;
		return id;
	}

private:
	bool m_pressed[KEY_KEY_CODES_COUNT];
	s32 m_clickedId;
};

/// Feeds the instancing shader its view-projection and light direction each draw.
class InstancingCallback : public video::IShaderConstantSetCallBack
{
public:
	core::vector3df LightDir;

	InstancingCallback() : LightDir(0.4f, 0.8f, -0.45f) { }

	virtual void OnSetConstants(video::IMaterialRendererServices* services, s32 userData)
	{
		video::IVideoDriver* driver = services->getVideoDriver();

		// World stays identity: the per-instance matrix comes in through the vertex stream, so
		// folding the node transform in here would apply it twice.
		core::matrix4 viewProj = driver->getTransform(video::ETS_PROJECTION);
		viewProj *= driver->getTransform(video::ETS_VIEW);

		services->setVertexShaderConstant("ViewProj", viewProj.pointer(), 16);

		const float light[4] = {LightDir.X, LightDir.Y, LightDir.Z, 0.f};
		services->setVertexShaderConstant("LightDir", light, 4);
		services->setPixelShaderConstant("LightDir", light, 4);
	}
};

/// The exe runs from x64\Debug|Release while the shaders live in x64\Data, so every
/// "media/shaders/..." path the compute pipeline uses is relative to Data, not to the exe.
bool EnterDataDirectory()
{
	const wchar_t* sentinel = L"media\\shaders\\B3Narrowphase.hlsl";
	if (GetFileAttributesW(sentinel) != INVALID_FILE_ATTRIBUTES)
		return true;
	if (SetCurrentDirectoryW(L"..\\Data") && GetFileAttributesW(sentinel) != INVALID_FILE_ATTRIBUTES)
		return true;
	if (SetCurrentDirectoryW(L"x64\\Data") && GetFileAttributesW(sentinel) != INVALID_FILE_ATTRIBUTES)
		return true;
	return false;
}

scene::IMesh* MakeCubeMesh(const std::shared_ptr<scene::ISceneManager>& smgr, float halfExtent)
{
	// Irrlicht's cube is 1 unit across, so scale it to the collision half-extent rather than
	// scaling the node - instanced nodes take their scale per instance.
	scene::IMesh* cube = smgr->getGeometryCreator()->createCubeMesh(
		core::vector3df(halfExtent * 2.f, halfExtent * 2.f, halfExtent * 2.f));
	return cube;
}
}  // namespace

/// Runs every backend over the same scene and prints a comparison table. Headless, so it works over
/// a remote shell with no message loop, and it isolates physics cost from the renderer.
/**
 * @brief Per-tick state hashes for one scene, written to a file and optionally diffed against one.
 *
 * The point is cross-MACHINE comparison: run it on two GPUs and diff, which is the only way to
 * answer whether the pipeline is lockstep-safe. Same-machine it just re-confirms determinism.
 *
 * @param backends Every available backend.
 * @param scene Scene to run.
 * @param steps Steps to run.
 * @param interval Steps between hash samples.
 * @param outPath File to write; empty writes none.
 * @param comparePath Reference file to diff against; empty skips the diff.
 * @return 0 if nothing diverged, 1 otherwise.
 */
int RunDeterminismTest(std::vector<std::unique_ptr<PhysicsBackend> >& backends,
					   const SceneSpec& scene, int steps, int interval,
					   const std::string& outPath, const std::string& comparePath)
{
	if (interval < 1)
		interval = 1;

	// backend name + tick -> hash
	std::map<std::string, unsigned long long> reference;
	if (!comparePath.empty())
	{
		FILE* f = fopen(comparePath.c_str(), "r");
		if (!f)
		{
			printf("Determinism: cannot open reference '%s'\n", comparePath.c_str());
			return 1;
		}
		char name[128];
		long long tick;
		unsigned long long hash;
		while (fscanf(f, "%127[^\t]\t%lld\t%llx\n", name, &tick, &hash) == 3)
		{
			char key[160];
			sprintf(key, "%s\t%lld", name, tick);
			reference[key] = hash;
		}
		fclose(f);
		printf("Determinism: %d reference samples from '%s'\n", (int)reference.size(),
			   comparePath.c_str());
	}

	printf("\nDeterminism run: %s (%d bodies), %d steps, hashing every %d\n", scene.Name,
		   scene.BoxCount(), steps, interval);

	std::string report;
	int failures = 0;
	for (size_t b = 0; b < backends.size(); ++b)
	{
		if (!backends[b]->Reset(scene))
		{
			printf("  %-30s RESET FAILED\n", backends[b]->Name());
			++failures;
			continue;
		}

		long long firstDivergentTick = -1;
		int samples = 0, matched = 0, missing = 0;
		for (int i = 0; i < steps; ++i)
		{
			backends[b]->Step(kFixedStep);
			if (((i + 1) % interval) != 0)
				continue;

			unsigned long long hash = 0;
			if (!backends[b]->HashState(hash))
				continue;
			++samples;

			char line[256];
			sprintf(line, "%s\t%d\t%016llx\n", backends[b]->Name(), i + 1, hash);
			report += line;

			if (reference.empty())
				continue;
			char key[160];
			sprintf(key, "%s\t%d", backends[b]->Name(), i + 1);
			std::map<std::string, unsigned long long>::const_iterator it = reference.find(key);
			if (it == reference.end())
				++missing;
			else if (it->second == hash)
				++matched;
			else if (firstDivergentTick < 0)
				firstDivergentTick = i + 1;   // keep running: where it diverges is the useful part
		}

		printf("  %-30s %d samples", backends[b]->Name(), samples);
		if (!reference.empty())
		{
			printf(", %d matched, %d missing", matched, missing);
			if (firstDivergentTick >= 0)
			{
				printf("  DIVERGED first at tick %lld", firstDivergentTick);
				++failures;
			}
			else if (matched > 0)
				printf("  IDENTICAL");
		}
		printf("\n");
		fflush(stdout);
	}

	if (!outPath.empty())
	{
		FILE* f = fopen(outPath.c_str(), "w");
		if (!f)
			printf("Determinism: cannot write '%s'\n", outPath.c_str());
		else
		{
			fwrite(report.c_str(), 1, report.size(), f);
			fclose(f);
			printf("Determinism: wrote '%s'\n", outPath.c_str());
		}
	}

	if (!reference.empty())
		printf("\n%s\n", failures ? "DETERMINISM CHECK FAILED" : "Every hash matched the reference.");
	return failures ? 1 : 0;
}

int RunBenchmark(std::vector<std::unique_ptr<PhysicsBackend> >& backends, const SceneSpec& scene,
				 int steps)
{
	printf("\nScene: %s  (%d bodies%s)\n", scene.Name, scene.BoxCount(),
		   scene.NeedsConstraints() ? ", constrained" : "");

	// minY/maxY are a correctness check, not decoration: a backend whose bodies sank through the
	// ground or exploded outward reports a fast step time for a scene that fell apart.
	printf("\n%-30s %8s %10s %11s %8s %8s %9s %9s\n", "Backend", "bodies", "step ms", "transfer ms",
		   "pairs", "awake", "minY", "maxY");
	printf("%s\n", std::string(101, '-').c_str());

	for (size_t b = 0; b < backends.size(); ++b)
	{
		if (scene.NeedsConstraints() && !backends[b]->SupportsConstraints())
		{
			printf("%-30s  SKIPPED - no constraint solver\n", backends[b]->Name());
			continue;
		}
		if (scene.NeedsKinematics() && !backends[b]->SupportsKinematics())
		{
			printf("%-30s  SKIPPED - no kinematic bodies\n", backends[b]->Name());
			continue;
		}
		if (!backends[b]->Reset(scene))
		{
			printf("%-30s  RESET FAILED\n", backends[b]->Name());
			continue;
		}

		// Discard the first steps: the stack starts separated, so early steps do almost no
		// collision work and would flatter whichever backend ran first.
		const int warmup = steps / 4;
		double stepSum = 0.0, transferSum = 0.0;
		int measured = 0;
		for (int i = 0; i < steps; ++i)
		{
			backends[b]->Step(kFixedStep);
			if (i >= warmup)
			{
				stepSum += backends[b]->LastStepMs();
				transferSum += backends[b]->LastTransferMs();
				++measured;
			}
		}

		const double stepMs = measured ? stepSum / measured : 0.0;
		const double transferMs = measured ? transferSum / measured : 0.0;

		float minY = 1e30f, maxY = -1e30f;
		const std::vector<DemoTransform>& finalT = backends[b]->Transforms();
		for (size_t i = 0; i < finalT.size(); ++i)
		{
			minY = finalT[i].position[1] < minY ? finalT[i].position[1] : minY;
			maxY = finalT[i].position[1] > maxY ? finalT[i].position[1] : maxY;
		}

		// The awake tally is read on an interval, so pull a fresh one now rather than reporting a
		// value up to that interval stale in the one row that gets published.
		backends[b]->RefreshAwakeCount();

		char awake[16];
		if (backends[b]->AwakeBodyCount() >= 0)
			sprintf_s(awake, "%d", backends[b]->AwakeBodyCount());
		else
			strcpy_s(awake, "-");

		printf("%-30s %8d %10.3f %11.3f %8d %8s %9.2f %9.2f\n", backends[b]->Name(),
			   backends[b]->BodyCount(), stepMs, transferMs, backends[b]->LastPairCount(),
			   awake, minY, maxY);
		fflush(stdout);
	}

	printf("\nTransfer time is CPU<->GPU only and is INCLUDED in step ms. The GPU path is\n"
		   "resident by default - bodies are uploaded once and only packed transforms come\n"
		   "back. OSDEMO_GPU_RESIDENT=0 forces the older per-phase upload path.\n"
		   "'awake' is bodies not deactivated at the final step; OSDEMO_NO_SLEEP=1 disables\n"
		   "deactivation on every backend.\n");
	return 0;
}

/**
 * @brief Checks the three things an infinite-mass body has to be able to do here.
 *
 * Static anchors, a driven kinematic sweeper, and a SLEEPING body woken through a joint - the
 * last only passes with joint-aware wake, so OSDEMO_JOINT_WAKE=0 is the negative control.
 *
 * @param backends Every available backend.
 * @param scale Link count knob, clamped by the scene itself.
 * @param steps Total steps; the holder starts moving at 3 s, so this must exceed 180.
 * @return 0 when every backend that accepted the scene passed, 1 otherwise.
 */
int RunKinematicTest(std::vector<std::unique_ptr<PhysicsBackend> >& backends, int scale, int steps)
{
	const SceneSpec scene = BuildScene(SCENE_KINEMATIC_ANCHORS, scale);
	const int links = scale < 4 ? 4 : (scale > 8 ? 8 : scale);
	const int firstTarget = links * 2;
	const int sleeper = firstTarget + 5;
	const int hung = sleeper + 1;
	const int boxes = scene.BoxCount();
	const int kinSweeper = boxes + 0;
	const int kinHolder = boxes + 1;
	const int kinCarrier = boxes + 2;
	const int settleStep = 170;   // the holder is still at rest here; it starts moving at 3 s

	if (boxes != hung + 3 || scene.KinematicCount() != 3)
	{
		printf("\nKinematic test: scene layout changed - indices no longer match.\n");
		return 1;
	}

	printf("\nKinematic + anchor test: %d bodies, %d kinematic, %d joints, %d steps\n", boxes,
		   scene.KinematicCount(), (int)scene.Constraints.size(), steps);
	printf("  anchored chains 0..%d, targets %d..%d, jointed sleeper %d, carried chain %d..%d\n",
		   firstTarget - 1, firstTarget, sleeper - 1, sleeper, hung, hung + 2);

	int failures = 0;
	for (size_t b = 0; b < backends.size(); ++b)
	{
		if ((scene.NeedsConstraints() && !backends[b]->SupportsConstraints())
			|| (scene.NeedsKinematics() && !backends[b]->SupportsKinematics()))
		{
			printf("  %-30s SKIPPED - cannot represent this scene\n", backends[b]->Name());
			continue;
		}
		if (!backends[b]->Reset(scene))
		{
			printf("  %-30s RESET FAILED\n", backends[b]->Name());
			++failures;
			continue;
		}

		float sleeperAtSettle = 0.f, holderAtSettle = 0.f;
		float targetMaxStart = -1e30f;
		double sweeperMaxErr = 0.0;
		{
			const std::vector<DemoTransform>& t0 = backends[b]->Transforms();
			for (int k = 0; k < 5; ++k)
				targetMaxStart = t0[firstTarget + k].position[0] > targetMaxStart
							   ? t0[firstTarget + k].position[0] : targetMaxStart;
		}

		for (int i = 0; i < steps; ++i)
		{
			backends[b]->Step(kFixedStep);
			const std::vector<DemoTransform>& t = backends[b]->Transforms();

			// i, not i+1: the deferred readback publishes the kinematic pose packed one step earlier.
			float pos[3], quat[4], lin[3], ang[3];
			DemoKinematicPose(scene.Kinematics[0], (double)i * (double)kFixedStep, pos, quat, lin, ang);
			for (int a = 0; a < 3; ++a)
			{
				const double err = fabs((double)t[kinSweeper].position[a] - (double)pos[a]);
				sweeperMaxErr = err > sweeperMaxErr ? err : sweeperMaxErr;
			}

			if (i == settleStep)
			{
				sleeperAtSettle = t[sleeper].position[0];
				holderAtSettle = t[kinHolder].position[0];
			}
		}

		const std::vector<DemoTransform>& t = backends[b]->Transforms();

		float anchoredMinY = 1e30f;
		for (int k = 0; k < firstTarget; ++k)
			anchoredMinY = t[k].position[1] < anchoredMinY ? t[k].position[1] : anchoredMinY;

		float targetMaxEnd = -1e30f;
		for (int k = 0; k < 5; ++k)
			targetMaxEnd = t[firstTarget + k].position[0] > targetMaxEnd
						 ? t[firstTarget + k].position[0] : targetMaxEnd;

		const float holderX = t[kinHolder].position[0];
		const float sleeperX = t[sleeper].position[0];
		const float carrierX = t[kinCarrier].position[0];
		const float carriedX = t[hung].position[0];

		// The bottom link hangs at 2.60; a dropped static endpoint lands it at 0.50.
		const bool anchorHeld = anchoredMinY > 1.5f;
		const bool sweeperExact = sweeperMaxErr < 1e-4;
		const bool targetsPushed = targetMaxEnd > targetMaxStart + 1.f;
		const bool sleptStill = fabs((double)sleeperAtSettle) < 0.05;
		const bool sleeperFollowed = fabs((double)(sleeperX - holderX)) < 1.0
								  && fabs((double)sleeperX) > 0.5;
		const bool carriedFollowed = fabs((double)(carriedX - carrierX)) < 1.5;

		const bool pass = anchorHeld && sweeperExact && targetsPushed && sleptStill
					   && sleeperFollowed && carriedFollowed;
		if (!pass)
			++failures;

		printf("  %-30s %s\n", backends[b]->Name(), pass ? "PASS" : "FAIL");
		printf("      anchored chain lowest y %.3f (hangs at 2.60, free-falls to 0.50)\n",
			   anchoredMinY);
		printf("      sweeper max deviation from its script %.3g m; targets max x %.3f -> %.3f\n",
			   sweeperMaxErr, targetMaxStart, targetMaxEnd);
		printf("      jointed sleeper x %.3f at step %d -> %.3f final, holder %.3f -> %.3f\n",
			   sleeperAtSettle, settleStep, sleeperX, holderAtSettle, holderX);
		printf("      carried chain top x %.3f, its kinematic carrier %.3f\n", carriedX, carrierX);
		if (!anchorHeld)
			printf("      FAIL: the ground-anchored chains fell - the static endpoint was dropped.\n");
		if (!sweeperExact)
			printf("      FAIL: the sweeper left its scripted path - something wrote to it.\n");
		if (!targetsPushed)
			printf("      FAIL: the sweeper passed through its targets - its AABB is not tracking.\n");
		if (!sleptStill)
			printf("      FAIL: the jointed box was not at rest before its anchor moved.\n");
		if (!sleeperFollowed)
			printf("      FAIL: the jointed box never followed its anchor - it was not woken.\n");
		if (!carriedFollowed)
			printf("      FAIL: the chain hung from a kinematic body did not follow it.\n");
		fflush(stdout);
	}

	printf("\nJoint wake is what makes the sleeper follow; OSDEMO_JOINT_WAKE=0 reproduces the\n"
		   "original bug, and OSDEMO_NO_SLEEP=1 shows it is the sleep gate at fault.\n");
	return failures == 0 ? 0 : 1;
}

/**
 * @brief Drops a box onto a settled, SLEEPING tower and checks the tower actually reacts.
 *
 * The single most important sleeping check: a stack that sleeps must still be woken by a new
 * impact, and must neither be passed through nor stay frozen while something rests on it.
 *
 * @param backends Every available backend.
 * @param towersPerSide Towers along each ground axis.
 * @param towerHeight Boxes per tower.
 * @param steps Total steps; the drop must land well before the end.
 * @return 0 when every backend passed, 1 otherwise.
 */
int RunSleepTest(std::vector<std::unique_ptr<PhysicsBackend> >& backends, int towersPerSide,
				 int towerHeight, int steps)
{
	// 25 m is ~2.2 s of fall, so the tower is asleep long before impact, and the impact speed stays
	// under half a box per step - a miss is a real wake failure, not tunnelling.
	const float dropHeight = 25.f;
	const SceneSpec scene = BuildSleepDropScene(towersPerSide, towerHeight, dropHeight);
	const int towers = towersPerSide * towersPerSide;
	const int towerBoxes = towers * towerHeight;
	const int preImpact = 100;   // the tower is settled and asleep well before this; impact is ~134

	printf("\nSleep drop test: %d towers of %d, one box falling %.0f m onto each\n", towers,
		   towerHeight, dropHeight);
	printf("  %d steps; tower boxes are indices 0..%d, droppers %d..%d\n", steps, towerBoxes - 1,
		   towerBoxes, scene.BoxCount() - 1);

	int failures = 0;
	for (size_t b = 0; b < backends.size(); ++b)
	{
		if (!backends[b]->Reset(scene))
		{
			printf("  %-30s RESET FAILED\n", backends[b]->Name());
			++failures;
			continue;
		}

		float settledTop = -1e30f, settledMin = 1e30f;
		int awakeAtPreImpact = -1;
		int peakAwakeAfterImpact = -1;
		double preMs = 0.0, postMs = 0.0;
		int preSamples = 0, postSamples = 0;

		for (int i = 0; i < steps; ++i)
		{
			backends[b]->Step(kFixedStep);
			const std::vector<DemoTransform>& t = backends[b]->Transforms();

			if (i == preImpact - 1)
			{
				backends[b]->RefreshAwakeCount();
				awakeAtPreImpact = backends[b]->AwakeBodyCount();
				for (int k = 0; k < towerBoxes && k < (int)t.size(); ++k)
				{
					settledTop = t[k].position[1] > settledTop ? t[k].position[1] : settledTop;
					settledMin = t[k].position[1] < settledMin ? t[k].position[1] : settledMin;
				}
			}
			// Sampled either side of the drop so the sleeping saving and the woken cost are both
			// visible in one run rather than inferred from two.
			if (i >= preImpact - 30 && i < preImpact) { preMs += backends[b]->LastStepMs(); ++preSamples; }
			if (i >= steps - 30) { postMs += backends[b]->LastStepMs(); ++postSamples; }

			if (i >= preImpact)
			{
				backends[b]->RefreshAwakeCount();
				const int awake = backends[b]->AwakeBodyCount();
				peakAwakeAfterImpact = awake > peakAwakeAfterImpact ? awake : peakAwakeAfterImpact;
			}
		}

		const std::vector<DemoTransform>& t = backends[b]->Transforms();
		float dropperMin = 1e30f, towerMin = 1e30f, towerMax = -1e30f;
		for (int k = 0; k < (int)t.size(); ++k)
		{
			if (k < towerBoxes)
			{
				towerMin = t[k].position[1] < towerMin ? t[k].position[1] : towerMin;
				towerMax = t[k].position[1] > towerMax ? t[k].position[1] : towerMax;
			}
			else
				dropperMin = t[k].position[1] < dropperMin ? t[k].position[1] : dropperMin;
		}

		// A dropper resting ON the tower sits about a box above its top face; one that tunnelled
		// through ends up at or below it.
		const bool noPassThrough = dropperMin > settledTop + kBoxHalfExtent;
		// Frozen-stack detection: the tower must have taken the impact, i.e. compressed slightly,
		// rather than the dropper balancing on an inert stack at exactly its settled height.
		const bool towerHeld = towerMin > settledMin - 0.35f;
		const bool sleptBeforeImpact = awakeAtPreImpact >= 0 && awakeAtPreImpact <= towers;
		const bool wokeOnImpact = peakAwakeAfterImpact > awakeAtPreImpact;

		const bool pass = noPassThrough && towerHeld && (awakeAtPreImpact < 0
														 || (sleptBeforeImpact && wokeOnImpact));
		if (!pass)
			++failures;

		printf("  %-30s %s\n", backends[b]->Name(), pass ? "PASS" : "FAIL");
		printf("      awake before impact %d of %d, peak after %d, final %d\n", awakeAtPreImpact,
			   scene.BoxCount(), peakAwakeAfterImpact, backends[b]->AwakeBodyCount());
		printf("      settled tower top %.3f  final tower %.3f..%.3f  lowest dropper %.3f\n",
			   settledTop, towerMin, towerMax, dropperMin);
		printf("      step ms: %.3f asleep (pre-impact) -> %.3f final\n",
			   preSamples ? preMs / preSamples : 0.0, postSamples ? postMs / postSamples : 0.0);
		if (!noPassThrough)
			printf("      FAIL: a dropper ended at or below the settled tower top - passed through.\n");
		if (!towerHeld)
			printf("      FAIL: the tower sank more than 0.35 m - it did not hold the impact.\n");
		if (awakeAtPreImpact >= 0 && !sleptBeforeImpact)
			printf("      FAIL: the tower never went to sleep before the drop landed.\n");
		if (awakeAtPreImpact >= 0 && !wokeOnImpact)
			printf("      FAIL: nothing woke on impact - the stack stayed deactivated.\n");
		fflush(stdout);
	}

	printf("\n%s\n", failures ? "SLEEP DROP TEST FAILED" : "Sleep drop test passed on every backend.");
	return failures ? 1 : 0;
}

/**
 * @brief Ascending body-count sweep over the island-stacks scene, one table for every size.
 * @param backends Every available backend; CPU ones drop out once they exceed the budget.
 * @param ladder Ascending target body counts.
 * @param steps Steps per size; the second half is measured, so it must outlast the drop.
 * @param cpuBudgetMs Step time past which a CPU backend stops being run at larger sizes.
 * @return 0 always; a failure is reported in the table rather than as an exit code.
 */
int RunScalingSweep(std::vector<std::unique_ptr<PhysicsBackend> >& backends,
					const std::vector<int>& ladder, int steps, double cpuBudgetMs)
{
	// Landing takes ~35 steps, but a tower needs ~300 to COMPRESS from its 1.2 drop spacing to
	// contact: measured, 120 steps reports only 70% of the settled contacts, at every size.
	if (steps < 300)
		steps = 300;
	const int measureFrom = steps / 2 > 40 ? steps / 2 : 40;

	printf("\nScaling sweep: islands of stacks\n");
	printf("  tower height %d, island spacing %.2f m (OSDEMO_TOWER_HEIGHT / OSDEMO_ISLAND_SPACING)\n",
		   IslandTowerHeight(), IslandSpacing());
	printf("  %d steps per size, timings averaged over steps %d..%d (post-landing)\n", steps,
		   measureFrom, steps - 1);
	printf("  CPU budget %.0f ms/step - a CPU backend past it is skipped at every larger size\n",
		   cpuBudgetMs);

	printf("\n%9s %9s %8s %10s %8s %-30s %10s %11s %9s %9s %9s\n", "target", "bodies", "islands",
		   "pair cap", "buf MB", "Backend", "step ms", "transfer ms", "pairs", "minY", "maxY");
	printf("%s\n", std::string(140, '-').c_str());

	std::vector<bool> overBudget(backends.size(), false);
	bool sawZeroPairs = false;

	for (size_t s = 0; s < ladder.size(); ++s)
	{
		const SceneSpec scene = BuildIslandStackScene(ladder[s]);

		for (size_t b = 0; b < backends.size(); ++b)
		{
			const bool isGpu = backends[b]->GpuBufferBytes() >= 0 || backends[b]->PairCapacity() >= 0;

			printf("%9d %9d %8d ", ladder[s], scene.BoxCount(), scene.IslandCount);

			if (overBudget[b])
			{
				printf("%10s %8s %-30s %s\n", "-", "-", backends[b]->Name(),
					   "SKIPPED (over budget)");
				fflush(stdout);
				continue;
			}
			if (!backends[b]->Reset(scene))
			{
				printf("%10s %8s %-30s   RESET FAILED\n", "-", "-", backends[b]->Name());
				fflush(stdout);
				continue;
			}

			double stepSum = 0.0, transferSum = 0.0;
			int measured = 0;
			for (int i = 0; i < steps; ++i)
			{
				backends[b]->Step(kFixedStep);
				if (i >= measureFrom)
				{
					stepSum += backends[b]->LastStepMs();
					transferSum += backends[b]->LastTransferMs();
					++measured;
				}
			}

			const double stepMs = measured ? stepSum / measured : 0.0;
			const double transferMs = measured ? transferSum / measured : 0.0;

			float minY = 1e30f, maxY = -1e30f;
			const std::vector<DemoTransform>& finalT = backends[b]->Transforms();
			for (size_t i = 0; i < finalT.size(); ++i)
			{
				minY = finalT[i].position[1] < minY ? finalT[i].position[1] : minY;
				maxY = finalT[i].position[1] > maxY ? finalT[i].position[1] : maxY;
			}

			const int cap = backends[b]->PairCapacity();
			const long long bytes = backends[b]->GpuBufferBytes();
			if (cap >= 0)
				printf("%10d ", cap);
			else
				printf("%10s ", "-");
			if (bytes >= 0)
				printf("%8.1f ", (double)bytes / (1024.0 * 1024.0));
			else
				printf("%8s ", "-");

			printf("%-30s %10.3f %11.3f %9d %9.2f %9.2f\n", backends[b]->Name(), stepMs, transferMs,
				   backends[b]->LastPairCount(), minY, maxY);
			fflush(stdout);

			if (backends[b]->LastPairCount() == 0)
				sawZeroPairs = true;
			if (!isGpu && stepMs > cpuBudgetMs)
			{
				overBudget[b] = true;
				// Drop the world now: a retired 40k-body Bullet scene would otherwise hold its
				// memory for the rest of the sweep, against the sizes that actually need it.
				backends[b]->Reset(BuildIslandStackScene(1));
			}
		}
		printf("\n");
	}

	if (sawZeroPairs)
		printf("WARNING: a row reported 0 pairs - that size measured free-fall, not contacts.\n");
	printf("Contacts were present wherever pairs > 0. A pair count equal to the cap means the\n"
		   "broadphase overflowed and that row's physics is not trustworthy.\n");
	return 0;
}

int main(int argc, char** argv)
{
	bool benchmark = false;
	bool scaling = false;
	bool sleepTest = false;
	int sleepTestSteps = 400;
	bool kinematicTest = false;
	int kinematicTestSteps = 300;
	int sleepTowerHeight = 6;
	int benchSteps = 240;
	int scalingSteps = 0;   // 0 = the sweep's own default, so --steps stays a bench knob
	int benchBoxes = kDefaultBoxesPerSide;
	int benchScene = -1;   // -1 = sweep every scene
	int scalingMax = 600000;
	double cpuBudgetMs = 100.0;
	bool gpuOnly = false;
	bool stubBackend = false;
	bool determinism = false;
	int determinismInterval = 50;
	std::string determinismOut = "determinism.txt";
	std::string determinismCompare;
	bool net = false;
	bool netInteractive = false;
	bool netModeExplicit = false;
	DemoNetConfig netConfig;
	std::vector<int> scalingList;
	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "--net-host") == 0)
		{
			net = true;
			netConfig.Host = true;
		}
		else if (strcmp(argv[i], "--net-join") == 0 && i + 1 < argc)
		{
			net = true;
			netConfig.Host = false;
			netConfig.PeerHost = argv[++i];
		}
		else if (strcmp(argv[i], "--net-port") == 0 && i + 1 < argc)
			netConfig.Port = atoi(argv[++i]);
		else if (strcmp(argv[i], "--net-mode") == 0 && i + 1 < argc)
		{
			++i;
			netConfig.Mode = strcmp(argv[i], "authoritative") == 0 ? DEMO_NET_AUTHORITATIVE
							 : strcmp(argv[i], "heal") == 0       ? DEMO_NET_HEALING
																  : DEMO_NET_LOCKSTEP;
			netModeExplicit = true;
		}
		else if (strcmp(argv[i], "--net-interval") == 0 && i + 1 < argc)
			netConfig.Interval = atoi(argv[++i]);
		else if (strcmp(argv[i], "--net-correct-every") == 0 && i + 1 < argc)
			netConfig.CorrectEvery = atoi(argv[++i]);
		else if (strcmp(argv[i], "--net-timeout") == 0 && i + 1 < argc)
			netConfig.ConnectTimeoutSeconds = atoi(argv[++i]);
		else if (strcmp(argv[i], "--bench") == 0)
			benchmark = true;
		else if (strcmp(argv[i], "--scaling") == 0)
			scaling = true;
		else if (strcmp(argv[i], "--gpu-only") == 0)
			gpuOnly = true;
		else if (strcmp(argv[i], "--stub-backend") == 0)
			stubBackend = true;
		else if (strcmp(argv[i], "--net-interactive") == 0)
			netInteractive = true;
		else if (strcmp(argv[i], "--determinism") == 0)
			determinism = true;
		else if (strcmp(argv[i], "--determinism-interval") == 0 && i + 1 < argc)
			determinismInterval = atoi(argv[++i]);
		else if (strcmp(argv[i], "--determinism-out") == 0 && i + 1 < argc)
			determinismOut = argv[++i];
		else if (strcmp(argv[i], "--determinism-compare") == 0 && i + 1 < argc)
		{
			determinismCompare = argv[++i];
			determinism = true;
		}
		else if (strcmp(argv[i], "--sleep-test") == 0)
			sleepTest = true;
		else if (strcmp(argv[i], "--kinematic-test") == 0)
			kinematicTest = true;
		else if (strcmp(argv[i], "--kinematic-steps") == 0 && i + 1 < argc)
			kinematicTestSteps = atoi(argv[++i]);
		else if (strcmp(argv[i], "--sleep-steps") == 0 && i + 1 < argc)
			sleepTestSteps = atoi(argv[++i]);
		else if (strcmp(argv[i], "--tower-height") == 0 && i + 1 < argc)
			sleepTowerHeight = atoi(argv[++i]);
		else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
		{
			benchSteps = atoi(argv[++i]);
			scalingSteps = benchSteps;
		}
		else if (strcmp(argv[i], "--boxes") == 0 && i + 1 < argc)
			benchBoxes = atoi(argv[++i]);
		else if (strcmp(argv[i], "--scene") == 0 && i + 1 < argc)
			benchScene = atoi(argv[++i]);
		else if (strcmp(argv[i], "--scaling-max") == 0 && i + 1 < argc)
			scalingMax = atoi(argv[++i]);
		else if (strcmp(argv[i], "--cpu-budget-ms") == 0 && i + 1 < argc)
			cpuBudgetMs = atof(argv[++i]);
		else if (strcmp(argv[i], "--scaling-list") == 0 && i + 1 < argc)
		{
			const char* p = argv[++i];
			while (*p)
			{
				const int v = atoi(p);
				if (v > 0)
					scalingList.push_back(v);
				while (*p && *p != ',')
					++p;
				if (*p == ',')
					++p;
			}
		}
		else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
		{
			printf("OuterSpacePhysicsDemo\n"
				   "  (no args)                 interactive comparison\n"
				   "  --gpu-only                skip both CPU backends (any mode)\n"
			   "  --determinism             log a state hash every N steps; run on two machines\n"
			   "                            and diff to answer whether lockstep is viable\n"
			   "    --determinism-interval N  steps between hashes (default 50)\n"
			   "    --determinism-out FILE    where to write (default determinism.txt)\n"
			   "    --determinism-compare FILE  diff against a reference and name the first\n"
			   "                            divergent tick; implies --determinism\n"
			   "  --net-host                run connected to a second instance; this side listens\n"
			   "  --net-join HOST           ...and this side dials it (127.0.0.1 on one machine)\n"
			   "                            both peers need identical --scene/--boxes/--steps\n"
			   "    --net-port N            TCP port (default 45654)\n"
			   "    --net-mode M            lockstep (compare hashes, correct nothing),\n"
			   "                            heal (lockstep until a root mismatch, then narrow to\n"
			   "                            the diverged buckets and converge) or\n"
			   "                            authoritative (host snapshots, client applies)\n"
			   "    --stub-backend          swap every backend for a deterministic CPU stub that\n"
			   "                            implements healing, so the protocol can be tested with\n"
			   "                            no GPU. Env: OSDEMO_STUB_DIVERGE_TICK/_AT/_COUNT/_EPS,\n"
			   "                            OSDEMO_STUB_BUCKET, OSDEMO_STUB_COUPLING\n"
			   "    --net-interval N        ticks between exchanges (default 50)\n"
			   "    --net-interactive       windowed instead of headless, so a desync can be\n"
			   "                            SEEN; lockstep/heal only, scene locked while connected\n"
			   "    --net-correct-every N   exchanges between snapshots, authoritative (default 2)\n"
			   "    --net-timeout N         seconds to wait for the peer (default 60)\n"
			   "  --bench                   time every backend over one or all scenes\n"
				   "    --scene N               0 single, 1 multi-stack, 2 pyramid, 3 wall,\n"
				   "                            4 chains, 5 island stacks, 6 clustered islands,\n"
				   "                            7 kinematic + anchors; omit to sweep all\n"
				   "    --boxes N               scene scale knob (island stacks: islands per side)\n"
				   "    --steps N               steps per backend\n"
				   "  --scaling                 body-count sweep over the island-stacks scene\n"
				   "    --scaling-max N         top of the ladder (default 600000)\n"
				   "    --scaling-list a,b,c    explicit ladder instead of the default one\n"
				   "    --cpu-budget-ms X       CPU backends stop above this step time (default 100)\n"
				   "    --steps N               steps per size (min 300; towers need ~300 to settle)\n"
				   "Env: OSDEMO_TOWER_HEIGHT (10)  OSDEMO_ISLAND_SPACING (3.6 m)\n"
				   "     OSDEMO_CLUSTER_ISLANDS (8)  OSDEMO_CLUSTER_GAP (8 m)\n"
				   "     OSDEMO_PAIR_RATIO (4.0)   OSDEMO_MAX_PAIRS (pins the cap)\n"
				   "     OSDEMO_GPU_RESIDENT (1)   0 = old per-phase upload path (f32 only)\n"
				   "     OSDEMO_WORLD_OFFSET (0)   translates the whole scene, in metres. 6371000\n"
				   "                               is Earth's radius, where f32 breaks down\n"
				   "  --kinematic-test          drive kinematic bodies, anchor joints to the ground,\n"
				   "                            and check a SLEEPING jointed body wakes and follows\n"
				   "    --kinematic-steps N     steps to run (default 300)\n"
				   "  --sleep-test              drop a box onto a SLEEPING tower and check it wakes\n"
				   "    --boxes N               towers per side (default 10)\n"
				   "    --tower-height N        boxes per tower (default 6)\n"
				   "    --sleep-steps N         steps to run (default 400)\n"
				   "     OSDEMO_SOLVER_ITERATIONS  OSDEMO_THREADS  OSDEMO_STATS  OSDEMO_GPU_DEBUG\n"
				   "     OSDEMO_NO_SLEEP (0)       1 disables deactivation on every backend\n"
				   "     OSDEMO_SLEEP_STEPS (120)  still steps before a GPU body sleeps\n"
				   "     OSDEMO_SLEEP_MOVE (0.05)  metres a GPU body may drift and still be still\n"
				   "     OSDEMO_SLEEP_ANGULAR (1.0) rad/s\n"
				   "     OSDEMO_AWAKE_INTERVAL (30) steps between awake-count readbacks\n");
			return 0;
		}
	}

	netConfig.StepSeconds = (double)kFixedStep;

	// 0.5 s instead of the 2 s default, so a 20 m drop still lands on a tower that is already
	// asleep. Set before any backend is constructed, since Init reads it.
	if (sleepTest && !getenv("OSDEMO_SLEEP_STEPS"))
		_putenv_s("OSDEMO_SLEEP_STEPS", "30");

	if (scalingList.empty())
	{
		const int base[] = {1000, 2500, 5000, 10000, 25000, 50000, 100000, 200000, 400000, 600000};
		for (int i = 0; i < (int)(sizeof(base) / sizeof(base[0])); ++i)
			if (base[i] <= scalingMax)
				scalingList.push_back(base[i]);
		if (scalingList.empty() || scalingList.back() != scalingMax)
			scalingList.push_back(scalingMax);
	}

	const bool dataFound = EnterDataDirectory();

	SIrrlichtCreationParameters params;
	params.DriverType = video::EDT_DIRECT3D11;
	params.WindowSize = core::dimension2d<u32>(1280, 720);
	params.AntiAlias = 0;
	params.Vsync = false;   // vsync would cap the frame rate and hide the physics cost

	IrrlichtDevice* device = createDeviceEx(params);
	if (!device)
	{
		params.DriverType = video::EDT_DIRECT3D9;
		device = createDeviceEx(params);
	}
	if (!device)
	{
		printf("Device creation failed for both D3D11 and D3D9.\n");
		return 1;
	}

	// A WM_QUIT left in this thread's queue makes the first run() return false and the demo exit
	// before drawing a frame; run() itself does not drain it. Same fix the engine's boot path uses.
	{
		MSG quitMessage;
		while (PeekMessage(&quitMessage, nullptr, WM_QUIT, WM_QUIT, PM_REMOVE))
		{
		}
	}

	DemoInput input;
	device->setEventReceiver(&input);
	device->setWindowCaption(L"OuterSpace physics backend comparison");

	// This Irrlicht fork hands out shared_ptr for the scene manager and every scene node, but raw
	// pointers for the driver, filesystem and GUI - hence the mixed styles below.
	video::IVideoDriver* driver = device->getVideoDriver();
	auto smgr = device->getSceneManager();
	gui::IGUIEnvironment* gui = device->getGUIEnvironment();

	auto camera = smgr->addCameraSceneNodeFPS(0, 60.f, 0.05f);
	camera->setPosition(core::vector3df(0, 18, -30));
	camera->setTarget(core::vector3df(0, 4, 0));
	camera->setFarValue(500.f);

	// Start in UI mode: the FPS camera grabs the mouse for look, which makes the buttons
	// unclickable, so camera input is off until Tab hands the mouse back to it.
	bool flyMode = false;
	camera->setInputReceiverEnabled(false);
	device->getCursorControl()->setVisible(true);

	smgr->addLightSceneNode(0, core::vector3df(40, 80, -40), video::SColorf(1.f, 1.f, 1.f), 300.f);
	smgr->setAmbientLight(video::SColorf(0.35f, 0.35f, 0.4f));

	// Ground drawn as a plain node - it is one static body and does not belong in the instanced
	// batch, whose whole point is the thousands of identical dynamic boxes.
	scene::IMesh* groundMesh = MakeCubeMesh(smgr, 1.f);
	auto ground = smgr->addMeshSceneNode(groundMesh);
	// MakeCubeMesh already returns a half-extent-1 cube, so scale by the half-extents directly -
	// doubling here drew the slab twice as tall and buried the bottom cube layer.
	ground->setScale(core::vector3df(kGroundHalfExtent, kGroundThickness, kGroundHalfExtent));
	ground->setPosition(core::vector3df(0, -kGroundThickness, 0));
	ground->setMaterialFlag(video::EMF_LIGHTING, true);
	ground->getMaterial(0).DiffuseColor = video::SColor(255, 60, 70, 80);

	scene::IMesh* boxMesh = MakeCubeMesh(smgr, kBoxHalfExtent);

	// Without this material the instanced draw feeds TEXCOORD1..4 to a shader that does not
	// declare them, which takes the D3D11 driver down on the first frame.
	InstancingCallback* instancingCallback = new InstancingCallback();
	const irr::s32 instancingMaterial =
		driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles(
			"media/shaders/DemoInstancing.hlsl", "VSMain", video::EVST_VS_5_0,
			"media/shaders/DemoInstancing.hlsl", "PSMain", video::EPST_PS_5_0,
			instancingCallback, video::EMT_SOLID);
	instancingCallback->drop();

	std::vector<std::unique_ptr<PhysicsBackend> > backends;
	bool gpuAvailable = false;
	bool gpuDsAvailable = false;
	if (stubBackend)
	{
		// Alone in the list on purpose: the handshake compares backend names, and this mode exists to
		// exercise the sync protocol rather than any real physics.
		backends.push_back(std::unique_ptr<PhysicsBackend>(new HealStubBackend()));
	}
	else
	{
		if (!gpuOnly)
		{
			backends.push_back(std::unique_ptr<PhysicsBackend>(new CpuBulletBackend(false, 1)));
			backends.push_back(std::unique_ptr<PhysicsBackend>(new CpuBulletBackend(true, 16)));
		}

		std::unique_ptr<GpuPhysicsBackend> gpu(new GpuPhysicsBackend(driver, device->getFileSystem()));
		gpuAvailable = gpu->Init();
		if (gpuAvailable)
			backends.push_back(std::unique_ptr<PhysicsBackend>(gpu.release()));

		// The emulated-double pipeline is a fourth backend rather than a mode on the third: the two are
		// distinct kernel sets, and comparing them side by side is the whole point.
		std::unique_ptr<GpuPhysicsBackend> gpuDs(
			new GpuPhysicsBackend(driver, device->getFileSystem(), true));
		gpuDsAvailable = gpuDs->Init();
		if (gpuDsAvailable)
			backends.push_back(std::unique_ptr<PhysicsBackend>(gpuDs.release()));
	}

	int boxesPerSide = kDefaultBoxesPerSide;
	// The knob means different things per scene (islands per side vs. a cube edge), so the real
	// ceiling is a BODY count - 245 islands/side is 600k bodies, but 245 cubed is 14 billion.
	int maxBoxesPerSide = 245;
	int maxBodies = 600250;
	if (const char* env = getenv("OSDEMO_MAX_BOXES_PER_SIDE"))
	{
		const int parsed = atoi(env);
		if (parsed >= 4)
			maxBoxesPerSide = parsed;
	}
	if (const char* env = getenv("OSDEMO_MAX_BODIES"))
	{
		const int parsed = atoi(env);
		if (parsed >= 8)
			maxBodies = parsed;
	}
	size_t active = 0;
	backends[active]->Reset(BuildScene(SCENE_SINGLE_STACK, boxesPerSide));

	printf("Data directory: %s\n", dataFound ? "found" : "NOT FOUND");
	if (stubBackend)
		printf("Backends: 1  (deterministic stub only - --stub-backend, no GPU backend built)\n");
	else
		printf("Backends: %u  (GPU f32 %s, GPU df64 %s)\n", (unsigned)backends.size(),
			   gpuAvailable ? "available" : "UNAVAILABLE - a compute shader failed to compile",
			   gpuDsAvailable ? "available" : "UNAVAILABLE - a compute shader failed to compile");
	if (DemoWorldOffset() != 0.0)
		printf("World offset: %.1f m per axis (OSDEMO_WORLD_OFFSET). Reported positions are\n"
			   "  scene-local, so a backend that cannot hold the offset shows the damage here.\n",
			   DemoWorldOffset());
	printf("Bodies: %d\n", backends[active]->BodyCount());
	fflush(stdout);

	if (kinematicTest)
	{
		const int result = RunKinematicTest(backends, benchBoxes, kinematicTestSteps);
		device->closeDevice();
		device->drop();
		return result;
	}

	if (sleepTest)
	{
		const int result = RunSleepTest(backends, benchBoxes, sleepTowerHeight, sleepTestSteps);
		device->closeDevice();
		device->drop();
		return result;
	}

	if (net && !netInteractive)
	{
		const int result = RunNetSync(backends, BuildScene(benchScene < 0 ? 5 : benchScene,
														   benchBoxes),
									  benchSteps, kFixedStep, netConfig);
		device->closeDevice();
		device->drop();
		return result;
	}

	if (determinism)
	{
		const int result = RunDeterminismTest(backends, BuildScene(benchScene < 0 ? 5 : benchScene,
															   benchBoxes),
											  benchSteps, determinismInterval, determinismOut,
											  determinismCompare);
		device->closeDevice();
		device->drop();
		return result;
	}

	if (scaling)
	{
		const int result = RunScalingSweep(backends, scalingList, scalingSteps, cpuBudgetMs);
		device->closeDevice();
		device->drop();
		return result;
	}

	if (benchmark)
	{
		// --scene -1 (default) sweeps every scene, which is the useful default for a benchmark:
		// one layout tells you little about how a backend handles a different contact topology.
		int result = 0;
		if (benchScene < 0)
			for (int s = 0; s < SCENE_COUNT; ++s)
				result |= RunBenchmark(backends, BuildScene(s, benchBoxes), benchSteps);
		else
			result = RunBenchmark(backends, BuildScene(benchScene, benchBoxes), benchSteps);
		device->closeDevice();
		device->drop();
		return result;
	}

	// GPU cull/sort/matrix-build over the physics pipeline's own transform buffer. Declines
	// (leaving the node path below in charge) when a kernel will not compile.
	GpuInstanceCuller culler(driver, device->getFileSystem());
	const bool cullerAvailable = culler.Init();
	int cullBucket = -1;
	bool phaseTimingOn = false;
	bool gpuCullActive = false;
	// One uint per body: 0 for the static ground at index 0, 1 for every drawn box. The bucket's
	// mask/value pair is the whole filter mechanism - a renderer with more meshes adds buckets.
	std::vector<unsigned int> filterKeys;
	scene::ComputeBuffer<unsigned int>* filterKeyBuffer = 0;
	bool cullVerified = false;

	// One instanced node rebuilt on reset; instance count is fixed for a given scene size.
	std::shared_ptr<scene::IInstancedMeshSceneNode> instanced;
	std::vector<std::shared_ptr<scene::ISceneNode> > instances;

	auto rebuildInstances = [&]() {
		if (instanced)
		{
			instanced->remove();
			instanced.reset();
		}
		instances.clear();
		instanced = smgr->addInstancedMeshSceneNode(boxMesh);
		if (!instanced)
			return;
		if (instancingMaterial >= 0)
			instanced->setMaterialType((video::E_MATERIAL_TYPE)instancingMaterial);
		instanced->setMaterialFlag(video::EMF_LIGHTING, false);
		// The instance batch spans the whole stack; per-node culling would drop it wholesale.
		instanced->setAutomaticCulling(scene::EAC_OFF);

		const std::vector<DemoTransform>& transforms = backends[active]->Transforms();

		// World offset is not a gate: DemoTransform::position is float too, so the node path renders
		// the same float precision at an offset - camera-relative transforms are a later project.
		gpuCullActive = cullerAvailable && instancingMaterial >= 0
						&& !getenv("OSDEMO_FORCE_NODE_PATH")
						&& backends[active]->ProducesGpuTransforms();

		culler.ClearBuckets();
		cullBucket = -1;
		cullVerified = false;

		if (gpuCullActive)
		{
			GpuInstanceCuller::BucketDesc desc;
			desc.FilterMask = 1;
			desc.FilterValue = 1;
			desc.Sort = true;               // solids: front to back, for early-Z
			desc.SortDescending = false;
			desc.IndexCount = boxMesh->getMeshBuffer(0)->getIndexBuffer()->getIndexCount();
			for (int a = 0; a < 3; ++a)
			{
				desc.BoundsMin[a] = -kBoxHalfExtent;
				desc.BoundsMax[a] = kBoxHalfExtent;
			}
			cullBucket = culler.AddBucket(desc);
			gpuCullActive = cullBucket >= 0;
		}

		if (gpuCullActive)
		{
			filterKeys.assign(transforms.size() + 1, 1u);
			filterKeys[0] = 0u;   // the static ground is drawn as its own node
			if (filterKeyBuffer)
				filterKeyBuffer->drop();
			filterKeyBuffer = new scene::ComputeBuffer<unsigned int>();
			filterKeyBuffer->setHardwareMappingHint(scene::EHM_DYNAMIC);
			filterKeyBuffer->set_used((u32)filterKeys.size());
			memcpy(filterKeyBuffer->getBufferPointer(), &filterKeys[0],
				   filterKeys.size() * sizeof(unsigned int));
			filterKeyBuffer->setDirty();

			// Nothing per instance exists on the CPU any more: no nodes, no instance vector.
			instanced->setVisible(false);
			return;
		}

		instances.reserve(transforms.size());
		for (size_t i = 0; i < transforms.size(); ++i)
		{
			instances.push_back(instanced->addInstance(
				core::vector3df(transforms[i].position[0], transforms[i].position[1],
								transforms[i].position[2])));
		}
	};
	// No call here: selectBackend(0) below performs the first build, and doing it twice would
	// create and discard a thousand instance nodes.

	RollingAverage stepAvg;
	RollingAverage transferAvg;
	RollingAverage frameAvg;

	gui::IGUIStaticText* hud = gui->addStaticText(L"", core::rect<s32>(10, 10, 620, 190), true);
	hud->setBackgroundColor(video::SColor(160, 0, 0, 0));
	hud->setOverrideColor(video::SColor(255, 235, 235, 235));

	// One button per backend, built from the backend list so a backend that failed to initialise
	// (shaders that will not compile) simply has no button rather than a dead one.
	int sceneKind = SCENE_SINGLE_STACK;
	std::vector<gui::IGUIButton*> sceneButtons;
	std::vector<gui::IGUIButton*> backendButtons;
	gui::IGUIEditBox* netAddressBox = 0;
	gui::IGUIButton* netModeButton = 0;
	// Heal is the interactive default: the UI exists to watch a divergence get corrected, and
	// lockstep would only ever report one. An explicit --net-mode still wins.
	DemoNetMode uiNetMode = netModeExplicit ? netConfig.Mode : DEMO_NET_HEALING;
	{
		const s32 top = 198;
		const s32 height = 26;
		s32 x = 10;
		for (size_t b = 0; b < backends.size(); ++b)
		{
			const s32 width = 176;
			core::stringw label(backends[b]->Name());
			gui::IGUIButton* button = gui->addButton(
				core::rect<s32>(x, top, x + width, top + height), 0,
				(s32)(BUTTON_BACKEND_BASE + b), label.c_str(), L"Switch to this physics backend");
			backendButtons.push_back(button);
			x += width + 6;
		}

		// Scene row. Names come from BuildScene so the labels cannot drift from what is built.
		const s32 sceneTop = top + height + 6;
		s32 sx = 10;
		for (int s = 0; s < SCENE_COUNT; ++s)
		{
			const SceneSpec probe = BuildScene(s, 4);
			const s32 width = 176;
			sceneButtons.push_back(gui->addButton(
				core::rect<s32>(sx, sceneTop, sx + width, sceneTop + height), 0,
				BUTTON_SCENE_BASE + s, core::stringw(probe.Name).c_str(),
				probe.NeedsConstraints() ? L"Needs a constraint solver"
										 : L"Switch scene"));
			sx += width + 6;
		}

		const s32 ctrlTop = sceneTop + height + 6;
		gui->addButton(core::rect<s32>(10, ctrlTop, 110, ctrlTop + height), 0,
					   BUTTON_RESET, L"Reset [R]", L"Rebuild the scene");
		gui->addButton(core::rect<s32>(116, ctrlTop, 216, ctrlTop + height), 0,
					   BUTTON_MORE_BOXES, L"More [+]", L"Larger scene");
		gui->addButton(core::rect<s32>(222, ctrlTop, 322, ctrlTop + height), 0,
					   BUTTON_FEWER_BOXES, L"Fewer [-]", L"Smaller scene");

		// Sync row: both peers must be on the same scene, so connecting locks it until disconnect.
		const s32 netTop = ctrlTop + height + 6;
		gui->addStaticText(L"Peer:", core::rect<s32>(10, netTop + 4, 50, netTop + height), false);
		netAddressBox = gui->addEditBox(core::stringw(netConfig.PeerHost.c_str()).c_str(),
										core::rect<s32>(52, netTop, 192, netTop + height), true);
		gui->addButton(core::rect<s32>(198, netTop, 298, netTop + height), 0,
					   BUTTON_NET_HOST, L"Host", L"Listen for a second instance");
		gui->addButton(core::rect<s32>(304, netTop, 404, netTop + height), 0,
					   BUTTON_NET_JOIN, L"Join", L"Connect to the peer address");
		gui->addButton(core::rect<s32>(410, netTop, 530, netTop + height), 0,
					   BUTTON_NET_DISCONNECT, L"Disconnect", L"Drop the link and unlock the scene");
		// Without this the buttons could only ever connect in the config default, and lockstep
		// detects a divergence without ever correcting it - which reads as "healing is broken".
		netModeButton = gui->addButton(
			core::rect<s32>(536, netTop, 686, netTop + height), 0, BUTTON_NET_MODE,
			uiNetMode == DEMO_NET_HEALING ? L"Mode: Heal" : L"Mode: Lockstep",
			L"Heal corrects a divergence; lockstep only reports it. Both peers must match.");
	}

	{
		// Proof the UI exists and sits on screen - "I see no UI" is otherwise indistinguishable
		// between not built, not created, positioned off-screen, and not drawn.
		const core::dimension2du screen = driver->getScreenSize();
		printf("[ui] screen %ux%u, font=%s\n", screen.Width, screen.Height,
			   gui->getSkin() && gui->getSkin()->getFont() ? "yes" : "NONE");
		const core::list<gui::IGUIElement*>& children = gui->getRootGUIElement()->getChildren();
		printf("[ui] %u root elements\n", (unsigned)children.getSize());
		for (core::list<gui::IGUIElement*>::ConstIterator it = children.begin(); it != children.end(); ++it)
		{
			const core::rect<s32> r = (*it)->getAbsolutePosition();
			printf("[ui]   id=%d visible=%d rect=(%d,%d)-(%d,%d)\n", (*it)->getID(),
				   (*it)->isVisible() ? 1 : 0, r.UpperLeftCorner.X, r.UpperLeftCorner.Y,
				   r.LowerRightCorner.X, r.LowerRightCorner.Y);
		}
		fflush(stdout);
	}

	// Non-empty when the active backend refused the current scene, so the HUD can say why instead
	// of showing an empty world and plausible-looking timings.
	core::stringw sceneRefusal;

	// Declared ahead of rebuildScene so it can refuse a scene change: both peers fingerprint the
	// scene at handshake, so switching one side's would report as a desync rather than a mismatch.
	std::unique_ptr<DemoNetInteractiveSession> netSession;
	unsigned long long netTick = 0;
	/// Barrier tick whose GPU hash was dispatched last frame and is collected this one.
	unsigned long long netHashPendingTick = 0;

	auto rebuildScene = [&]() {
		if (netSession)
		{
			sceneRefusal = L"Scene is locked while a network session is connected.";
			return;
		}
		netTick = 0;
		const SceneSpec scene = BuildScene(sceneKind, boxesPerSide);
		sceneRefusal = L"";

		if (scene.NeedsConstraints() && !backends[active]->SupportsConstraints())
		{
			sceneRefusal = core::stringw(backends[active]->Name());
			sceneRefusal += L" has no constraint solver - this scene needs one.";
		}
		else if (scene.NeedsKinematics() && !backends[active]->SupportsKinematics())
		{
			sceneRefusal = core::stringw(backends[active]->Name());
			sceneRefusal += L" cannot drive kinematic bodies - this scene needs them.";
		}

		// Switching backend restarts the scene: identical initial conditions keep timings comparable.
		if (!backends[active]->Reset(scene) && sceneRefusal.size() == 0)
			sceneRefusal = L"Backend refused this scene.";

		stepAvg.Clear();
		transferAvg.Clear();
		rebuildInstances();
	};

	auto selectBackend = [&](size_t index) {
		if (index >= backends.size())
			return;
		active = index;
		rebuildScene();
		for (size_t b = 0; b < backendButtons.size(); ++b)
			if (backendButtons[b])
				backendButtons[b]->setEnabled(b != active);
	};

	auto selectScene = [&](int kind) {
		if (kind < 0 || kind >= SCENE_COUNT)
			return;
		sceneKind = kind;
		rebuildScene();
		for (size_t s = 0; s < sceneButtons.size(); ++s)
			if (sceneButtons[s])
				sceneButtons[s]->setEnabled((int)s != sceneKind);
	};

	// Start knobs, so an automated run can reach a large scene without driving the UI.
	if (const char* env = getenv("OSDEMO_START_BACKEND"))
		selectBackend((size_t)atoi(env));
	else
		selectBackend(0);
	if (const char* env = getenv("OSDEMO_START_SCALE"))
	{
		const int parsed = atoi(env);
		if (parsed >= 4)
			boxesPerSide = parsed;
	}
	selectScene(getenv("OSDEMO_START_SCENE") ? atoi(getenv("OSDEMO_START_SCENE"))
											 : SCENE_SINGLE_STACK);

	// --net-host/--net-join only PRESET the buttons in interactive mode; this starts the link
	// straight away so a scripted two-window run needs no clicks.
	if (net && netInteractive)
		netSession.reset(DemoNetInteractiveSession::Begin(
			netConfig, BuildScene(sceneKind, boxesPerSide), backends));

	u32 lastTime = device->getTimer()->getTime();
	unsigned long long framesRun = 0;

	// Optional cap so an automated smoke run terminates on its own instead of needing a kill.
	const char* frameLimitEnv = getenv("OSDEMO_MAX_FRAMES");
	const unsigned long long frameLimit = frameLimitEnv ? strtoull(frameLimitEnv, nullptr, 10) : 0;

	while (device->run())
	{
		++framesRun;
		if (frameLimit && framesRun > frameLimit)
			break;

		const u32 now = device->getTimer()->getTime();
		const double frameMs = (double)(now - lastTime);
		lastTime = now;
		frameAvg.Add(frameMs);

		// Buttons and keys drive the same helpers, so the two paths cannot drift apart.
		const s32 clicked = input.ConsumeClick();
		if (clicked >= BUTTON_BACKEND_BASE
			&& clicked < BUTTON_BACKEND_BASE + (s32)backends.size())
		{
			selectBackend((size_t)(clicked - BUTTON_BACKEND_BASE));
		}
		if (clicked >= BUTTON_SCENE_BASE && clicked < BUTTON_SCENE_BASE + SCENE_COUNT)
			selectScene(clicked - BUTTON_SCENE_BASE);

		if (clicked == BUTTON_NET_MODE && !netSession)
		{
			uiNetMode = uiNetMode == DEMO_NET_HEALING ? DEMO_NET_LOCKSTEP : DEMO_NET_HEALING;
			if (netModeButton)
				netModeButton->setText(uiNetMode == DEMO_NET_HEALING ? L"Mode: Heal"
																	 : L"Mode: Lockstep");
		}
		if ((clicked == BUTTON_NET_HOST || clicked == BUTTON_NET_JOIN) && !netSession)
		{
			netConfig.Host = clicked == BUTTON_NET_HOST;
			netConfig.Mode = uiNetMode;
			if (netAddressBox)
			{
				const core::stringc typed(netAddressBox->getText());
				if (typed.size() > 0)
					netConfig.PeerHost = typed.c_str();
			}
			// The peer's tick count starts at zero, so ours has to as well.
			netTick = 0;
			netHashPendingTick = 0;
			backends[active]->Reset(BuildScene(sceneKind, boxesPerSide));
			netSession.reset(DemoNetInteractiveSession::Begin(
				netConfig, BuildScene(sceneKind, boxesPerSide), backends));
		}
		if (clicked == BUTTON_NET_DISCONNECT && netSession)
		{
			netSession.reset();
			sceneRefusal = L"";
		}

		if (input.Consume(KEY_F1)) selectScene(SCENE_SINGLE_STACK);
		if (input.Consume(KEY_F2)) selectScene(SCENE_MULTIPLE_STACKS);
		if (input.Consume(KEY_F3)) selectScene(SCENE_PYRAMID);
		if (input.Consume(KEY_F4)) selectScene(SCENE_WALL);
		if (input.Consume(KEY_F5)) selectScene(SCENE_CHAINS);
		if (input.Consume(KEY_F6)) selectScene(SCENE_ISLAND_STACKS);
		if (input.Consume(KEY_F7)) selectScene(SCENE_CLUSTERED_ISLANDS);

		if (input.Consume(KEY_KEY_1)) selectBackend(0);
		if (input.Consume(KEY_KEY_2)) selectBackend(1);
		if (input.Consume(KEY_KEY_3)) selectBackend(2);
		if (input.Consume(KEY_KEY_4)) selectBackend(3);

		// P: per-phase cost attribution. Serialises the pipeline, so step time gets worse while on.
		if (input.Consume(KEY_KEY_P))
		{
			phaseTimingOn = !phaseTimingOn;
			for (size_t b = 0; b < backends.size(); ++b)
				backends[b]->SetPhaseTiming(phaseTimingOn);
		}

		if (input.Consume(KEY_KEY_R) || clicked == BUTTON_RESET)
			rebuildScene();

		// Deliberately uncapped: the interactive path grows until something actually breaks, which is
		// the only honest ceiling. A failed Reset reverts rather than leaving a half-built scene.
		if (input.Consume(KEY_ADD) || input.Consume(KEY_PLUS) || clicked == BUTTON_MORE_BOXES)
		{
			const int step = boxesPerSide < 16 ? 2 : boxesPerSide / 8;
			const int previous = boxesPerSide;
			boxesPerSide += step > 0 ? step : 1;
			rebuildScene();
			if (sceneRefusal.size() != 0)
			{
				boxesPerSide = previous;
				rebuildScene();
			}
		}
		if (input.Consume(KEY_SUBTRACT) || input.Consume(KEY_MINUS) || clicked == BUTTON_FEWER_BOXES)
		{
			const int step = boxesPerSide <= 16 ? 2 : boxesPerSide / 9;
			boxesPerSide = boxesPerSide - step >= 4 ? boxesPerSide - (step > 0 ? step : 1) : 4;
			rebuildScene();
		}

		// Tab hands the mouse to the camera and back; the buttons need a visible cursor and the
		// FPS look needs a captured one, so they cannot both be live.
		if (input.Consume(KEY_TAB))
		{
			flyMode = !flyMode;
			camera->setInputReceiverEnabled(flyMode);
			device->getCursorControl()->setVisible(!flyMode);
		}

		if (input.Consume(KEY_ESCAPE))
			break;

		// The barrier below gates stepping only, not drawing/message pumps. Hash publish stays
		// outside it too, or both peers would wait on each other forever.
		if (netSession && netHashPendingTick != 0)
		{
			unsigned long long stateHash = 0;
			if (backends[active]->FetchHashState(stateHash))
				netSession->PublishHash(netHashPendingTick, stateHash);
			netHashPendingTick = 0;
		}

		// Also outside the gate, and for the same reason: the heal handshake is what CLEARS the
		// barrier, so running it only once stepping is allowed would deadlock both peers.
		if (netSession)
			netSession->PumpHeal(backends[active].get());

		if (!netSession || netSession->MayStep(netTick))
		{
			backends[active]->Step(kFixedStep);
			stepAvg.Add(backends[active]->LastStepMs());
			transferAvg.Add(backends[active]->LastTransferMs());
			++netTick;
			if (netSession && (netTick % (unsigned long long)netSession->Interval()) == 0)
			{
				// Only dispatched here; collected next frame, once this step's work has retired.
				if (backends[active]->BeginHashState())
					netHashPendingTick = netTick;
				else
				{
					unsigned long long stateHash = 0;
					if (backends[active]->HashState(stateHash))
						netSession->PublishHash(netTick, stateHash);
				}
			}
		}
		const std::vector<DemoTransform>& transforms = backends[active]->Transforms();
		const size_t count = instances.size() < transforms.size() ? instances.size() : transforms.size();

		// Each iteration writes only its own node's relative translation/rotation, so there is no
		// shared state; the quaternion->Euler conversion is what makes this worth splitting.
		auto applyTransform = [&](size_t i) {
			instances[i]->setPosition(core::vector3df(transforms[i].position[0],
													  transforms[i].position[1],
													  transforms[i].position[2]));
			core::quaternion q(transforms[i].orientation[0], transforms[i].orientation[1],
							   transforms[i].orientation[2], transforms[i].orientation[3]);
			core::vector3df euler;
			q.toEuler(euler);
			instances[i]->setRotation(euler * core::RADTODEG);
		};

		if (gpuCullActive)
		{
			// Nothing to do: the instance matrices are built by compute from the physics
			// pipeline's own transform buffer, further down.
		}
		else if (count < kParallelTransformThreshold)
		{
			for (size_t i = 0; i < count; ++i)
				applyTransform(i);
		}
		else
		{
			const size_t chunks = (count + kTransformGrain - 1) / kTransformGrain;
			concurrency::parallel_for(size_t(0), chunks, [&](size_t c) {
				const size_t begin = c * kTransformGrain;
				const size_t end = begin + kTransformGrain < count ? begin + kTransformGrain : count;
				for (size_t i = begin; i < end; ++i)
					applyTransform(i);
			});
		}
		// beginScene CLEARS the default back buffer but never BINDS it, so without the
		// setRenderTarget below every draw lands on whatever target was last bound.
		driver->beginScene(false, false, video::SColor(255, 18, 20, 26));
		driver->setRenderTarget(video::ERT_FRAME_BUFFER, true, true, video::SColor(255, 18, 20, 26));

		smgr->drawAll();

		if (gpuCullActive && instanced)
		{
			core::matrix4 viewProj = driver->getTransform(video::ETS_PROJECTION);
			viewProj *= driver->getTransform(video::ETS_VIEW);

			GpuInstanceCuller::CameraState cam;
			GpuInstanceCuller::MakeCameraState(viewProj.pointer(),
											   camera->getAbsolutePosition().X,
											   camera->getAbsolutePosition().Y,
											   camera->getAbsolutePosition().Z, cam);

			scene::IComputeBuffer* gpuTransforms = backends[active]->GpuTransformBuffer();
			const unsigned int objectCount = (unsigned int)filterKeys.size();
			if (gpuTransforms && culler.Run(gpuTransforms, filterKeyBuffer, objectCount, cam))
			{
				video::SMaterial mat;
				mat.MaterialType = (video::E_MATERIAL_TYPE)instancingMaterial;
				mat.Lighting = false;
				driver->setMaterial(mat);
				driver->setTransform(video::ETS_WORLD, core::matrix4());
				driver->drawMeshBufferInstancedIndirect(boxMesh->getMeshBuffer(0),
														culler.InstanceBuffer(cullBucket),
														GpuInstanceCuller::InstanceStride(),
														culler.DrawArgsBuffer(cullBucket), 0);

				// One-shot correctness check against the CPU reference. Stalls, so it runs once.
				if (!cullVerified && getenv("OSDEMO_CULL_VERIFY"))
				{
					cullVerified = true;
					std::vector<unsigned int> gpuOrder;
					std::vector<unsigned int> cpuOrder;
					culler.DownloadOrder(cullBucket, gpuOrder);

					// The reference reads the same transforms the GPU did, ground included.
					std::vector<float> flat(objectCount * 7, 0.f);
					for (size_t i = 0; i + 1 < objectCount && i < transforms.size(); ++i)
					{
						memcpy(&flat[(i + 1) * 7], transforms[i].position, 3 * sizeof(float));
						memcpy(&flat[(i + 1) * 7 + 3], transforms[i].orientation, 4 * sizeof(float));
					}
					GpuInstanceCuller::BucketDesc desc;
					desc.FilterMask = 1;
					desc.FilterValue = 1;
					desc.Sort = true;
					for (int a = 0; a < 3; ++a)
					{
						desc.BoundsMin[a] = -kBoxHalfExtent;
						desc.BoundsMax[a] = kBoxHalfExtent;
					}
					GpuInstanceCuller::CpuReference(&flat[0], filterKeys.empty() ? 0 : &filterKeys[0],
													objectCount, desc, cam, cpuOrder);

					size_t firstDiff = gpuOrder.size() == cpuOrder.size() ? gpuOrder.size() : 0;
					for (size_t i = 0; i < gpuOrder.size() && i < cpuOrder.size(); ++i)
						if (gpuOrder[i] != cpuOrder[i]) { firstDiff = i; break; }

					// Same set, and monotone depth, are the real bars: two instances at the same
					// distance may legitimately come back in either order.
					std::vector<unsigned int> gpuSet(gpuOrder), cpuSet(cpuOrder);
					std::sort(gpuSet.begin(), gpuSet.end());
					std::sort(cpuSet.begin(), cpuSet.end());
					size_t positionDiffs = 0;
					float worstTie = 0.f;
					for (size_t i = 0; i < gpuOrder.size() && i < cpuOrder.size(); ++i)
					{
						if (gpuOrder[i] == cpuOrder[i])
							continue;
						++positionDiffs;
						const float* g = &flat[gpuOrder[i] * 7];
						const float* c = &flat[cpuOrder[i] * 7];
						const float gd = core::vector3df(g[0] - cam.Position[0], g[1] - cam.Position[1],
														 g[2] - cam.Position[2]).getLength();
						const float cd = core::vector3df(c[0] - cam.Position[0], c[1] - cam.Position[1],
														 c[2] - cam.Position[2]).getLength();
						const float delta = gd > cd ? gd - cd : cd - gd;
						if (delta > worstTie)
							worstTie = delta;
					}
					printf("[cull-verify] same set: %s, positions differing: %u, worst depth delta "
						   "at those: %.6f m\n", gpuSet == cpuSet ? "YES" : "NO",
						   (unsigned)positionDiffs, worstTie);
					printf("[cull-verify] gpu=%u cpu=%u of %u objects, first order mismatch at %u\n",
						   (unsigned)gpuOrder.size(), (unsigned)cpuOrder.size(), objectCount,
						   (unsigned)firstDiff);
					for (size_t i = 0; i < 6 && i < gpuOrder.size() && i < cpuOrder.size(); ++i)
					{
						const float* g = &flat[gpuOrder[i] * 7];
						const float* c = &flat[cpuOrder[i] * 7];
						const float gd = core::vector3df(g[0] - cam.Position[0], g[1] - cam.Position[1],
														 g[2] - cam.Position[2]).getLength();
						const float cd = core::vector3df(c[0] - cam.Position[0], c[1] - cam.Position[1],
														 c[2] - cam.Position[2]).getLength();
						printf("[cull-verify]   %u: gpu idx %u d=%.4f | cpu idx %u d=%.4f\n",
							   (unsigned)i, gpuOrder[i], gd, cpuOrder[i], cd);
					}
					fflush(stdout);
				}
			}
		}

		core::stringw text;
		text += L"Backend: ";
		text += backends[active]->Name();
		text += L"     Mouse: ";
		text += (flyMode ? L"FLY (Tab for buttons)" : L"UI (Tab to fly)");
		text += L"\nScene [F1-F7]: ";
		text += BuildScene(sceneKind, boxesPerSide).Name;
		text += L"     Bodies: ";
		text += backends[active]->BodyCount();
		text += L"   (scale ";
		text += boxesPerSide;
		text += L")";
		if (sceneRefusal.size() > 0)
		{
			text += L"\n!! ";
			text += sceneRefusal;
		}
		text += L"\n\nPhysics step: ";
		text += core::stringw(stepAvg.Mean());
		text += L" ms  (rolling mean of 120 steps)";
		if (backends[active]->LastTransferMs() > 0.0)
		{
			text += L"\n  of which CPU<->GPU transfer: ";
			text += core::stringw(transferAvg.Mean());
			text += L" ms";
			text += L"\n  NOT representative of a GPU-resident pipeline - this composition";
			text += L"\n  uploads and reads back per phase.";
		}
		double phase[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		if (backends[active]->LastPhaseMs(phase))
		{
			const wchar_t* names[6] = { L"aabb", L"broad", L"paircompact",
										L"narrow", L"sleep+solve", L"integrate" };
			double sum = 0.0;
			for (int i = 0; i < 6; ++i)
				sum += phase[i];
			text += L"\nPhase ms/step (SERIALISED - shares, not totals):";
			for (int i = 0; i < 6; ++i)
			{
				text += L"\n  ";
				text += names[i];
				text += L": ";
				text += core::stringw(phase[i]);
				if (sum > 0.0)
				{
					text += L"  (";
					text += (int)(phase[i] * 100.0 / sum + 0.5);
					text += L"%)";
				}
			}
		}

		text += L"\nContact pairs: ";
		text += backends[active]->LastPairCount();
		if (backends[active]->AwakeBodyCount() >= 0)
		{
			text += L"   Awake bodies: ";
			text += backends[active]->AwakeBodyCount();
			text += L" / ";
			text += backends[active]->BodyCount();
		}
		text += L"\nFrame: ";
		text += core::stringw(frameAvg.Mean());
		text += L" ms  (includes render)";
		if (netSession)
		{
			text += L"\nNet: ";
			text += core::stringw(netSession->StatusText());
			if (netSession->Connected())
			{
				text += L"  tick ";
				text += (s32)netTick;
				text += L"  peer ";
				text += (s32)netSession->PeerTick();
				if (netSession->Desynced())
				{
					text += L"  *** DESYNCED at tick ";
					text += (s32)netSession->FirstDivergentTick();
					text += L" ***";
				}
				else
				{
					text += L"  in sync (";
					text += netSession->Matched();
					text += L" barriers)";
					if (!netSession->MayStep(netTick))
						text += L"  waiting for peer";
				}

				const DemoHealStats heal = netSession->HealStats();
				if (heal.Unsupported)
					text += L"\nHeal: unavailable - the backend does not bucket its state hash.";
				else if (heal.Divergences || heal.Rounds)
				{
					text += L"\nHeal: ";
					if (netSession->Diverged())
					{
						text += netSession->HealingNow() ? L"correcting" : L"diverged";
						text += L" since tick ";
						text += (s32)heal.DivergedSinceTick;
					}
					else
						text += L"converged";
					text += L"   healing bodies: ";
					text += netSession->HealingBodies();
					text += L"   buckets ";
					text += heal.BucketsCorrected;
					text += L"/";
					text += heal.BucketsCompared;
					text += L" over ";
					text += heal.Rounds;
					text += L" round(s)";
					if (heal.Recoveries)
					{
						text += L"\n      recovered ";
						text += heal.Recoveries;
						text += L"x, last in ";
						text += (s32)heal.LastTicksToRecovery;
						text += L" ticks (worst ";
						text += (s32)heal.WorstTicksToRecovery;
						text += L")";
					}
				}
			}
		}
		if (!dataFound)
			text += L"\n\nx64\\Data not found - run from the build output folder.";
		else if (!gpuAvailable)
			text += L"\n\nGPU backend unavailable: a compute shader failed to compile.";
		hud->setText(text.c_str());
		gui->drawAll();
		driver->endScene();

		// OSDEMO_SCREENSHOT=<path> captures one frame, so a "nothing renders" report can be
		// checked directly instead of reasoned about second-hand.
		if (framesRun == 30)
		{
			const char* shotPath = getenv("OSDEMO_SCREENSHOT");
			if (shotPath)
			{
				video::IImage* shot = driver->createScreenShot();
				if (shot)
				{
					driver->writeImageToFile(shot, io::path(shotPath));
					shot->drop();
					printf("[shot] wrote %s\n", shotPath);
					fflush(stdout);
				}
				else
				{
					printf("[shot] createScreenShot returned null\n");
					fflush(stdout);
				}
			}
		}
	}

	culler.ClearBuckets();
	if (filterKeyBuffer)
		filterKeyBuffer->drop();

	printf("Frames rendered: %llu\n", framesRun);
	printf("Final step time (%s): %.3f ms over %d bodies\n", backends[active]->Name(),
		   stepAvg.Mean(), backends[active]->BodyCount());
	printf("Final frame time: %.3f ms  (instance path: %s)\n", frameAvg.Mean(),
		   gpuCullActive ? "GPU cull + indirect draw" : "per-instance scene nodes");
	// Non-zero when a divergence is still outstanding, matching the headless verdict exactly - the
	// interactive path used to exit 0 on a desync, so the same run disagreed with itself.
	int exitCode = framesRun > 0 ? 0 : 2;
	if (netSession)
	{
		const DemoHealStats heal = netSession->HealStats();
		const bool healing = netConfig.Mode == DEMO_NET_HEALING;
		const bool outstanding = healing ? netSession->Diverged() : netSession->Desynced();
		printf("Net: ended at tick %llu, %d barriers matched, %s\n", netTick,
			   netSession->Matched(),
			   outstanding ? "DESYNCED" : (netSession->Connected() ? "in sync" : "link down"));
		if (healing)
		{
			printf("Net heal: %d divergence(s), %d recovered, %d round(s), %d/%d buckets corrected,"
				   " %llu bytes\n", heal.Divergences, heal.Recoveries, heal.Rounds,
				   heal.BucketsCorrected, heal.BucketsCompared, heal.BytesSent);
			if (heal.Recoveries)
				printf("Net heal: ticks to recovery last %llu, worst %llu\n",
					   heal.LastTicksToRecovery, heal.WorstTicksToRecovery);
			if (heal.Unsupported)
				printf("Net heal: UNAVAILABLE - the backend does not bucket its state hash.\n");
			if (heal.Divergences == 0 && heal.Rounds != 0)
				printf("Net heal: *** %d round(s) ran with no divergence - healing is not free.\n",
					   heal.Rounds);
		}
		if (outstanding && exitCode == 0)
			exitCode = 1;
	}
	fflush(stdout);

	device->closeDevice();
	device->drop();
	return exitCode;
}
