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
generation — comfortably real-time for transition-driven generation).

## Samples (in the `With U` project)

| Scene | Mirrors | What it shows |
| --- | --- | --- |
| `Assets/Samples/MotionBricks/Scenes/GenerateAndChain.unity` | `docs/examples/generate_once.py` | Generate toward placed targets, play on a VRM avatar, chain a second generation from the last played poses |
| `Assets/Samples/MotionBricks/Scenes/Locomotion.unity` | `docs/examples/locomotion_loop.py` | WASD/arrow steering; speed-scaled clip targets placed ahead of the avatar; regeneration at the transition point or on sharp input changes |
| `Assets/Samples/MotionBricks/Scenes/CharPickupObject.unity` | Smart Primitives concepts | Fully automatic sequence: walk to a cube → pick it up (hand IK-pinned to the cube) → carry 3 steps → put it down → backpedal 3 steps → jump → celebrate |

Both interaction scenes show a status overlay (provider, state, frames, stage timings) and a
foot-contact gizmo, and keep the camera behind the avatar (`MbCameraFollow`).

Changes to scripts do not require rebuilding the scenes. The **Tools → MotionBricks → Build Sample
Scenes** command recreates and saves all three scenes, so use it only when intentionally replacing
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
The bootstrap's **Backend Type** can be set to CPU before Play mode to isolate GPU-specific problems;
the backend's automatic fallback only detects invalid logits on its initial probe.

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

## Skeleton conventions (gotchas)

- **Body forward is the root's local +Y** — the SOMA root frame is tilted (its rest rotation is a
  120° diagonal). `root.rotation * Vector3.forward` is *not* the facing; it reads as a 90° error.
  Use `root.rotation * Vector3.up` as a rig-specific proxy; the model's heading is derived from hip
  positions and can differ during unusual poses.
- **Static clips have no travel direction.** `CanonicalizeTarget` derives the canonical yaw from the
  window's travel; for a static clip that travel is sub-centimeter jitter and its yaw is pure noise
  (a crouch idle drifted 10 mm and came out rotated 88°, which threw the hand's pin out of reach).
  The implementation falls back to the root's forward when the travel is under 3 cm.
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
