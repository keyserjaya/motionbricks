# Package license and notices

This package combines content with different licenses. Review each before distribution.

| Content | License | Notes |
| --- | --- | --- |
| All C# code in this package | Apache-2.0 | A port of the MIT-like Apache-2.0 licensed MotionBricks runtime from https://github.com/NVlabs/motionbricks (code Apache-2.0). |
| ONNX models (`mmm_v6_soma_fp16.onnx`, `mmm_v6_soma_fp32.onnx`) | [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) | Permits commercial use and redistribution of modified models subject to the agreement. You must add these files yourself; this package does not include them. |
| Onnx Runtime native libraries (`onnxruntime.dll`, `DirectML.dll`, …) | MIT | © Microsoft Corporation. From the `Microsoft.ML.OnnxRuntime.DirectML` NuGet distribution. |
| SOMA skeleton JSON (`Runtime/Skeleton/soma_skeleton.json`) | Apache-2.0 | Derived from the bundled `soma_base_skel_minimal_no_root.usd` of the MotionBricks repository. |
| SOMA BVH sample clips (used by the tests/samples) | [NVIDIA Sample Data Evaluation License](https://github.com/NVlabs/motionbricks/blob/main/assets/motions/LICENSE.txt) | Evaluation/testing of NVIDIA technologies only. **Do not ship these clips in a commercial build.** |
| VRM avatars used by the samples | Their own licenses | Loaded from the host project (`With U`). Not distributed with this package. |

# Third-party notices

- Onnx Runtime — MIT License. Copyright (c) Microsoft Corporation.
- MotionBricks (Python reference) — Apache License 2.0. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
- The two-bone IK solver (`MbTwoBoneIk`) is derived from Daniel Holden's `ik_two_bone` implementation
  in Motion Matching (https://github.com/orangeduck/Motion-Matching), MIT License — see
  `motionbricks/licenses/Daniel-Holden-MIT-LICENSE.txt` in the MotionBricks repository.
- This package is not an NVIDIA product and is not affiliated with NVIDIA.
