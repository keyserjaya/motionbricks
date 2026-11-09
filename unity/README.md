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
and the runtime skeleton. The package's EditMode tests (`MotionBricks.Tests.GoldenParityTests`)
validate the C# encode/decode/correction against those fixtures — 4/4 green.

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

Both scenes show a status overlay (provider, frames, stage timings) and a foot-contact gizmo, and
keep the camera behind the avatar (`MbCameraFollow`).

Rebuild the scenes after changing the sample scripts with **Tools → MotionBricks → Build Sample
Scenes** (the builder instantiates the project's `Takagi Katsumi` VRM prefab; point it at another
VRM prefab to use a different avatar).

## Platform support

Windows x64 Editor and Standalone. WebGL/Android cannot run this: the model is 380 MB and needs
GPU/CPU inference through the Unity Inference Engine (the package compiles everywhere; only the
model load is Windows-focused). The `With U` project's other build targets are unaffected — the
sample assemblies guard their code and the model asset is only referenced by the sample scenes.
