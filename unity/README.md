# MotionBricks → Unity

This folder holds the Unity integration for [MotionBricks](https://github.com/NVlabs/motionbricks):
a C# port of the SOMA runtime (generation, chaining, BVH clips, VRM retarget) plus parity tooling
against the Python reference engine.

```
unity/
  com.nvlabs.motionbricks/     UPM package (runtime + editor + golden tests)
tools/unity_golden/            parity fixtures + ONNX preparation (run from repo root)
```

The sample scenes live in the `With U` project (`Assets/Samples/MotionBricks`), driving its VRM
avatars. See `com.nvlabs.motionbricks/README.md` for API usage.

## What was ported

| Python reference | C# port |
| --- | --- |
| `motionbricks/model/soma/kernels.py` (encode/decode kernels) | `SomaGenerator` (encode → decode → endpoint correction) |
| `motionbricks/model/generation.py`, `config.py` | `SomaGenerator`, `MotionBricksInferenceConfig`, `TargetPoseInferenceParams` |
| `motionbricks/runtime/motion_recorder.py`, `motion_player.py` | `SomaMotionRecorder`, `GeneratedSomaMotionPlayer` |
| `motionbricks/animation/*`, `motionbricks/io/bvh.py` | `SomaAnimationBuffer`, `SomaClip`, `SomaBvhLoader` |
| `motionbricks/runtime/target_pose.py` (placement/speed) | `SomaTargetPlacement` |
| Newton/MuJoCo viewer + SOMA-X mesh | `SomaVrmRetargeter` (Unity humanoid / VRM) |

Coordinate space: Unity's Y-up world *is* the SOMA model space, so `model_to_runtime` is the
identity — the Python demos' Y-up→Z-up conversion (for their MuJoCo viewer) is intentionally absent.

## Parity tooling

Run from the repository root:

```bash
# 1. Export golden fixtures + the skeleton asset from the reference (TensorRT) engine
uv run python tools/unity_golden/export_golden.py

# 2. Rewire the ONNX token-count path for the Unity Inference Engine (see below)
uv run --with onnx --with onnxruntime python tools/unity_golden/simplify_token_output.py
```

`export_golden.py` writes `Tests/Fixtures/golden_soma.json` (two chained generations with packed
input tensors, raw ONNX outputs, and final corrected frames), `bvh_samples.json`, `mini_walk.mbvh`,
and the runtime skeleton. The package's EditMode golden tests validate C# encode, decode, BVH loading
and SOMA IK against those fixtures. They replay recorded model output, not live Unity inference.
Retargeting tests also use the project's Takagi Katsumi prefab when available and skip that fixture
in projects that do not contain it.

### Why the ONNX needs rewiring

The exported graph computes `pred_num_tokens` through an INT64 sentinel chain
(`ConstantOfShape`/`Reshape`/`Where`) and a `ScatterND` masking chain. InferenceEngine
mis-executes both on the GPU backend (the count came out as `198 = 18*11`, the masked logits read
back as NaN), and because `pred_num_tokens` feeds back into the internal decoder, later motion
frames were corrupted too.

`simplify_token_output.py` copies each model to `motionbricks/assets/onnx/soma/unity/` with the
count path rewired to plain `Slice`/`Equal`/`Where`/`ArgMax` ops over the raw token logits, and
exposes those logits as `pred_num_token_logits`. It validates the rewired models against the golden
fixture with onnxruntime (CPU): token counts match the TensorRT reference exactly (8 and 14), the
rewire is semantics-preserving (0.000000 vs the original on the same backend), and the C# argmax
rule (`SomaInferenceEngineBackend.ArgMaxTokenCount`) is checked in the same run.

`SomaInferenceEngineBackend` computes the token count in C# from `pred_num_token_logits` and falls
back to the CPU backend automatically if the GPU backend returns invalid logits (~80 ms per
generation — comfortably real-time for transition-driven generation). Invalid logits are probed on
every run until the first valid one, not just the first generation.

Note that some GPU drivers do not merely mis-execute this model — they drop a command out of its
compute stream: Unity's D3D12 fence then waits forever for it (`D3D12Fence::Wait(N) error: got
N-1`), the watchdog resets the device and the editor dies with
`Device removed reason (887a0006)` before any C# fallback can run. In practice this surfaces on
the **second** press of Play that uses GPUCompute: the first session runs, and the next session's
first generation hangs the device (the failure is in the GPU dispatch stream itself, not in
dispose/lock handling — generation is synchronous and its readback is blocking, and the locks are
main-thread only). The samples therefore default `MbSampleBootstrap._backendType` to CPU and leave
GPUCompute opt-in; `SomaInferenceEngineBackend.Dispose` also frees the inference engine's
still-queued deferred GPU buffer disposals so they cannot leak across play sessions.

## Samples (in the `With U` project)

| Scene | Mirrors | What it shows |
| --- | --- | --- |
| `Assets/Samples/MotionBricks/Scenes/GenerateAndChain.unity` | `docs/examples/generate_once.py` | Generate toward placed targets, play on a VRM avatar, chain a second generation from the last played poses |
| `Assets/Samples/MotionBricks/Scenes/Locomotion.unity` | `docs/examples/locomotion_loop.py` | WASD/arrow steering; speed-scaled clip targets placed ahead of the avatar; regeneration at the transition point or on sharp input changes |
| `Assets/Samples/MotionBricks/Scenes/CharPickupObject.unity` | Smart Primitives concepts | Fully automatic sequence: walk to a cube → pick it up (hand IK-pinned to the cube) → carry 3 steps → put it down → backpedal 3 steps → jump → celebrate |
| `Assets/Samples/MotionBricks/Scenes/CharJump.unity` | Showcase interaction boxes (no-ground area) | Fully automatic: jog to a floor gap → jump across the forbidden area box marking the pit → celebrate → turn → jump back, forever. The takeoff distance is learned from each arc, so the landing keeps clearing the far rim |
| `Assets/Samples/MotionBricks/Scenes/CharRoll.unity` | Showcase interaction boxes (allowed area) | Fully automatic: walk to a green allowed area box → roll inside it (jump-clip stand-in until `Clips/roll.mbvh` exists) → celebrate → walk back, repeat |

All five scenes show a status overlay (provider, state, frames, stage timings) and a foot-contact
gizmo, and keep the camera behind the avatar (`MbCameraFollow`).

### Area boxes

The new scenes' green boxes demonstrate area-constrained motion. The SOMA generator itself has no
area constraints — `MbAreaBox` (`MbAreaMode.Allowed` / `Forbidden`) is a sample component that:

- draws a green wireframe over the game view (red while violated),
- reports the violation depth the demos show in their overlay, and
- hardens the promise through the demos' **Root Clamp** toggle (on by default): the bootstrap's
  `PoseFilter` hook runs after IK and before retargeting/recording, snapping the root into the
  allowed footprint or out of the forbidden volume. The corrected pose feeds the next generation's
  context, exactly like IK pins do.

The constraint is realized by authoring, not by a solver: `CharJump` places the jump target on the
far rim so the endpoint correction pulls a real arc across the pit, and learns the takeoff distance
from the measured arc; `CharRoll` places the roll target on the box plot and lets the footprint
clamp hold the root inside it.

The area box owns the gap's geometry: the demos read the takeoff and landing edges from the box
(`MbAreaBox.NearEdgeZ` / `FarEdgeZ`), so moving or resizing the box moves the trajectory with it —
the constraint and the motion cannot drift apart.

In MotionBricks itself the boxes on Smart Primitives only trigger interactions; the "confinement"
seen in the showcase comes from the authored target poses — these scenes reproduce that with
takeoff/landing and roll placement math.

Changes to scripts do not require rebuilding the scenes. The **Tools → MotionBricks → Build Sample
Scenes** command recreates and saves all five scenes, so use it only when intentionally replacing
their settings. The builder uses the project's `Takagi Katsumi` VRM prefab.

## IK pins

`TargetPoseIkCorrector` can capture hand and foot pins from the first target pose and apply two-bone
IK to a SOMA pose. Setting `TargetPoseInferenceParams.IkPinMask` alone does not automatically
capture or apply pins; the caller must invoke the corrector. Unreachable pins are clamped.

The easing matches the reference character: `smoothstep(0.5, 1.0, playback progress)` — pins do
nothing for the first half of a segment and reach full strength at the transition point. The
pipeline order matters and mirrors the reference: **sample → IK-correct → retarget → record** — the
corrected pose is what shows and what the next generation's context holds.

`CharPickupObject` aims at the cube's top and optionally solves the avatar's own arm after retargeting,
because a VRM's limb proportions need not match SOMA's. Avatar-space correction does not feed back
into SOMA history. At pickup the sample logs the actual hand-to-grab-point distance and preserves the
cube's world position when parenting it. Unreachable targets and imperfect contact remain possible.
The sample exposes **Interaction Stand Back** and **Interaction Crouch Drop** for adjusting target
reach. The extra hip drop preserves the SOMA target's foot positions with two-bone IK; different VRM
foot proportions can still require tuning. Enable **Log State Transitions** to inspect the scene-space
root and target positions for each generated segment.

## Retarget Debugging

Select the avatar with Gizmos enabled during Play mode. On `SomaVrmRetargeter`, enable **Draw Debug
Joints** and choose **Avatar Anchored** to align source hips with avatar hips, or **Raw** for source
coordinates. Enable **Draw Avatar Bones** and **Draw Bone Error Lines** for comparison. Different
limb lengths and rest poses mean source and avatar joints need not overlap exactly. The Inspector
also reports the captured anchor, placement yaw, height and root travel scales, missing bones, and
actual avatar pin error. The toggle for avatar limb IK isolates the post-retarget correction.
On the sample's `MbSampleBootstrap`, **Enable Source Ik** separately toggles pre-retarget SOMA IK.
The bootstrap's **Backend Type** defaults to CPU because GPU compute on this model can hang some
drivers into a D3D12 device-removed editor crash; switch it to GPUCompute only where that is known
to work. The backend's automatic fallback probes invalid logits on every run until the first valid
one.

Initialize while the avatar is in its intended rest pose. The optional yaw alignment is captured at
initialization; toggle it before entering Play mode. **Rebase Current Poses** aligns the current
source and avatar poses without changing height scale. **Reinitialize (apply toggles)** rebuilds the
mapping live. Two knobs for correspondence debugging: **Swap Left Right Bones** (on by default — the
default SOMA skeleton's left/right names are mirrored relative to its geometry, its toes face +Z
while its "Left" joints sit at +X) and **Source Facing Offset Deg** (extra placement yaw if the
overlay is rotated relative to the avatar). Debug joint colors follow the side each joint drives:
green = avatar left, orange = avatar right. `SomaSceneMap` converts root positions and
directions between source and scene space; limb offsets use height scale independently of root travel
scale. Disable and re-enable the retargeter to apply changes to RigBuilder/offscreen-update options;
their prior states are restored when the retargeter is disabled.

## Authoring motion clips (.mbvh)

Clips are not playback animations — they are **style exemplars**. The generator transitions toward a
clip *window* (e.g. the middle of a walk clip); everything else (locomotion to a point, reach, carry)
is procedural: clip windows + placement + IK pins. One short clip per style is enough; the five
bundled clips drive the whole `CharPickupObject` sequence. A `.mbvh` file is plain standard BVH text
(`HIERARCHY` + `MOTION`) with SOMA joint names; the extension just selects the package's `BvhTextAssetImporter`.

### Format requirements

| Requirement | Detail |
|---|---|
| Joint names | Exact SOMA names, case-sensitive (`namespace:` prefixes are stripped). Missing joints keep reference transforms, so a body-only rig is fine. |
| Units | **Centimeters** — the loader scales offsets/positions by `0.01`. A correct file has `Hips OFFSET` ≈ 100-ish. |
| Rotation channels | Degrees, composed in the declared order (`Zrotation Yrotation Xrotation` is Blender's default and matches the bundled clips). |
| Up axis | BVH is Y-up; Blender's exporter converts from Z-up automatically. |
| Frame rate | From `Frame Time:` (the bundled clips use ~30 fps). |

Name traps: SOMA's thigh is `LeftLeg` and the shin is `LeftShin` — not Mixamo's
`LeftUpLeg`/`LeftLeg`. The spine is `Spine1 → Spine2 → Chest` and the neck `Neck1 → Neck2`. Body
set: `Hips, Spine1, Spine2, Chest, Neck1, Neck2, Head, LeftShoulder, LeftArm, LeftForeArm, LeftHand,
Right…, LeftLeg, LeftShin, LeftFoot, LeftToeBase, Right…` (fingers/jaw/eyes optional).

### Blender workflow

There is no SOMA FBX in the repo — the skeleton ships as USD
(`motionbricks/assets/human/soma_base_skel_minimal_no_root.usd`) and as the joint hierarchies inside the bundled `.mbvh` clips.

- **Route A (metric rig):** import the USD (Blender 4.x has built-in USD import) or rename an
  armature to the SOMA names → animate (2–4 s per style is typical) → export BVH with
  **Scale = 100** → rename `.bvh` to `.mbvh`.
- **Route B (import an existing `.mbvh`):** import a bundled clip to get a correctly named armature.
  Its offsets are centimeters, so Blender shows a 100×-size rig (hips ≈ 101 "meters") — **scale the
  armature ×0.01 first** to normalize to meters, then follow route A.

Scale rule: meters-scale rig → export Scale = 100; untouched imported cm-scale rig → export
Scale = 1. Never combine them. Sanity-check the output: `Hips OFFSET` ≈ 100-ish and
`Frames:` / `Frame Time:` present.

### Looping animations (idle)

MotionBricks does not play clips as loops — targets are windows the model generates toward. Two
options for a looping idle:

1. **Generative idle (recommended for the main character).** Author a seamless clip (first frame ≈
   last frame, 2–4 s at 30 fps, root travel ≤ 3 cm so placement uses root facing). Each time
   `player.TimeToTransition` hits 0, regenerate toward the same idle window at the character's
   current position and facing — the pattern `MbCharPickupDemo` uses for its locomotion states. The
   result is continuous standing motion with natural micro-variation, continuous across segments
   because generations are endpoint-corrected to the recorded history. Cost: one inference per
   segment.
2. **Cyclic clip playback (zero inference, exact loop).** Sample `SomaClip.Sample(time % duration)`
   each frame into `SomaVrmRetargeter.ApplyPose`. Clips share the generated motion's SOMA pose
   space, so blending idle ↔ generated motion is a simple pose slerp — no Animator handoff needed.
   Good for background NPCs or true loops.

Practical combo: option 2 for long-idle/background states, option 1 whenever the character acts;
transition into MotionBricks by seeding history from the current clip pose (`SeedHistoryFromPose`).

### Using clips alongside a normal Animator

The retargeter re-applies its pose every `LateUpdate` (order 10000) and owns its 22 humanoid bones,
so an Animator driving those same bones is wasted work and fights the anchor — keep
`applyRootMotion` off and no locomotion controller on the body during MotionBricks segments.
Animators (or VRM expressions/uLipSync) driving **fingers, face, jaw, hair** — bones outside the 22
bound ones — coexist cleanly, and MotionBricks does not drive fingers at all. Mixing Mecanim for
*body* idle with MotionBricks for dynamic states needs a handoff plus reverse-retarget seeding
(avatar pose → SOMA history), which the package does not implement; prefer the two loop options
above.

> The bundled sample clips are evaluation-only (`LICENSES/NOTICE.md`) — fine for prototyping; ship
> your own exports.

## Skeleton conventions (gotchas)

- **Body forward is the root's local +Y** — the SOMA root frame is tilted (its rest rotation is a
  120° diagonal). `root.rotation * Vector3.forward` is *not* the facing; it reads as a 90° error.
  Use `root.rotation * Vector3.up` as a rig-specific proxy; the model's heading is derived from hip
  positions and can differ during unusual poses.
- **Static clips have no travel direction.** `CanonicalizeTarget` derives the canonical yaw from the
  window's travel; for a static clip that travel is sub-centimeter jitter and its yaw is pure noise
  (a crouch idle drifted 10 mm and came out rotated 88°, which threw the hand's pin out of reach).
  The implementation falls back to the root's forward when the travel is under 3 cm.
- **A jump style target must come from the clip's airborne apex.** The model takes the target's root
  height as its cue to leave the ground. A window taken from the clip's run-up (standing height)
  asks for a run at the landing spot, and the character jogs flat across the pit no matter how wide
  it is. `MbCharJumpDemo` and `MbCharPickupDemo` center their jump window on the clip's apex
  (`ApexWindowTime`), which makes the arc rise ~30 cm over standing height at any gap width.
- **Seeding matters.** The recorder starts filled with the skeleton's T-pose reference; call
  `MbSampleBootstrap.Seed(pose)` with a natural pose before the first generation or every generation
  starts from a T-pose context.

> **No text prompts**: the SOMA ONNX takes only pose tensors — there is no text conditioning in the
> graph. "Prompting" movement means authoring target poses (procedurally or from clips) and
> constraining them with pins. The Python reference's Smart Primitives (authored pose sequences with
> translation/rotation constraints) are the larger-scale version of this authoring model.

## Platform support

Windows x64 Editor and Standalone. WebGL/Android cannot run this: the model is 380 MB and needs
GPU/CPU inference through the Unity Inference Engine (the package compiles everywhere; only the
model load is Windows-focused). The `With U` project's other build targets are unaffected — the
sample assemblies guard their code and the model asset is only referenced by the sample scenes.
