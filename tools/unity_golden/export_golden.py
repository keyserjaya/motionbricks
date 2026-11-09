# SPDX-License-Identifier: Apache-2.0
"""Export Unity-side parity fixtures and the SOMA skeleton asset for the Unity integration.

Run from the repository root:

    uv run python tools/unity_golden/export_golden.py

Outputs:
    unity/com.nvlabs.motionbricks/Runtime/Skeleton/soma_skeleton.json
        The skeleton the Unity runtime uses (names, parents, reference pose).
    unity/com.nvlabs.motionbricks/Tests/Fixtures/golden_soma.json
        Fixture: two chained generations run by the reference TensorRT engine, with
        packed input tensors and raw ONNX outputs plus the final corrected frames.
    unity/com.nvlabs.motionbricks/Tests/Fixtures/mini_walk.bvh
        A truncated copy of the bundled walk clip, for the C# BVH loader test.
    unity/com.nvlabs.motionbricks/Tests/Fixtures/bvh_samples.json
        Expected sampled poses (with root offsets) for the mini clip.

The engine runs with model_to_runtime = identity, which is the Unity situation:
Unity's Y-up world *is* the SOMA model space. All poses are local joint transforms
stored as {"px","py","pz","qx","qy","qz","qw"} objects (Unity Quaternion order).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import warp as wp

from motionbricks.animation.animation_buffer import AnimationBuffer
from motionbricks.io.bvh import load_bvh
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.model.contract import MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE
from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
from motionbricks.model.soma.engine import MotionBricksSOMAEngine
from motionbricks.runtime.target_pose import (
    TargetPoseInferenceParams,
    adjust_target_pose_speed,
    place_target_poses,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "unity" / "com.nvlabs.motionbricks"
FIXTURES = PACKAGE / "Tests" / "Fixtures"
CONTEXT_END = 1.0
TARGET_TIME_A = 1.8
TARGET_TIME_B = 3.4
JOG = "locomotion/bs_1d_linear/neutral_jog_ff_180_R_001__A534"
WALK = "locomotion/bs_1d_linear/Neutral_walk_forward_002__A057"
MINI_BVH_FRAMES = 150
MODEL_JOINT_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye",
    "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd",
    "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot",
    "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
]


def fp(value) -> float:
    """Round a float for compact, deterministic JSON (Unity-parsable widths)."""
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"Non-finite value in export: {x}")
    rounded = round(x, 6)
    return 0.0 if rounded == 0.0 else rounded


def pose7(pose: wp.array, joint: int) -> dict:
    """One wp.transform as an XYZW-ordered pose object."""
    d = pose.numpy()[joint]
    return {"px": fp(d[0]), "py": fp(d[1]), "pz": fp(d[2]),
            "qx": fp(d[3]), "qy": fp(d[4]), "qz": fp(d[5]), "qw": fp(d[6])}


def transform7(t) -> list:
    """A wp.transform or a 7-float row as [p, q(XYZW)] list."""
    if isinstance(t, wp.transform):
        return [fp(t.p[0]), fp(t.p[1]), fp(t.p[2]), fp(t.q[0]), fp(t.q[1]), fp(t.q[2]), fp(t.q[3])]
    return [fp(v) for v in t]


def pose_batch(poses: wp.array) -> list:
    return [pose7(poses, i) for i in range(poses.shape[0])]


def sample_window(clip: AnimationBuffer, start_time: float, poses: wp.array, root_offset=None) -> None:
    """Sample four clip poses at 30 Hz from start_time into poses, then place them."""
    for index in range(MODEL_POSES_PER_TOKEN):
        clip.sample(
            start_time + index / MODEL_SAMPLE_RATE,
            poses[index],
            root_offset=root_offset if root_offset is not None else wp.transform_identity(),
        )


def main() -> None:
    wp.init()
    wp.set_device("cuda:0")
    FIXTURES.mkdir(parents=True, exist_ok=True)

    usd_path = get_asset_file("human", "soma_base_skel_minimal_no_root.usd")
    skeleton = load_skeleton_from_usd(usd_path, SOMA_SKELETON_PRIM_PATH)
    print(f"skeleton joints: {skeleton.num_joints}")

    clip_path = REPO_ROOT / "assets" / "motions" / "soma" / f"{WALK}.bvh"
    clip = load_bvh(str(clip_path), skeleton)
    print(f"walk clip: {clip.num_frames} frames at {clip.sample_rate} Hz")

    # --- skeleton asset -------------------------------------------------------
    reference = skeleton.reference_local_transforms.numpy()
    skeleton_doc = {
        "sample_rate": MODEL_SAMPLE_RATE,
        "joint_names": list(skeleton.joint_names),
        "parent_indices": [int(skeleton.joint_parent(i)) for i in range(skeleton.num_joints)],
        "reference_local": [
            {"px": fp(reference[i, 0]), "py": fp(reference[i, 1]), "pz": fp(reference[i, 2]),
             "qx": fp(reference[i, 3]), "qy": fp(reference[i, 4]), "qz": fp(reference[i, 5]), "qw": fp(reference[i, 6])}
            for i in range(skeleton.num_joints)
        ],
    }
    skeleton_path = PACKAGE / "Runtime" / "Skeleton" / "soma_skeleton.json"
    skeleton_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton_path.write_text(json.dumps(skeleton_doc), encoding="utf-8")
    print(f"wrote {skeleton_path}")

    cs_path = PACKAGE / "Runtime" / "Skeleton" / "SomaSkeletonDefault.cs"
    def cfloat(v: float) -> str:
        return f"{fp(v):.6f}"
    names = skeleton_doc["joint_names"]
    parents = skeleton_doc["parent_indices"]
    ref_values = ",".join(
        cfloat(v) + "f"
        for row in skeleton_doc["reference_local"]
        for v in (row["px"], row["py"], row["pz"], row["qx"], row["qy"], row["qz"], row["qw"])
    )
    joint_list = ", ".join(f'"{n}"' for n in names)
    parent_list = ", ".join(str(p) for p in parents)
    cs = f"""// <auto-generated>
// Generated by tools/unity_golden/export_golden.py from the bundled SOMA USD skeleton
// (soma_base_skel_minimal_no_root.usd). Do not edit by hand.
// Reference pose values are local joint transforms: px,py,pz,qx,qy,qz,qw (Unity order).
// </auto-generated>
namespace MotionBricks
{{
    /// <summary>The SOMA skeleton bundled with the MotionBricks repository.</summary>
    public static class SomaSkeletonDefault
    {{
        public const float SampleRate = {fp(MODEL_SAMPLE_RATE)}f;
        public static readonly string[] JointNames = {{ {joint_list} }};
        public static readonly int[] ParentIndices = {{ {parent_list} }};
        public static readonly float[] ReferenceLocal = new float[]
        {{
            {ref_values}
        }};
    }}
}}
"""
    cs_path.write_text(cs, encoding="utf-8")
    print(f"wrote {cs_path}")

    # --- BVH loader fixture ---------------------------------------------------
    mini_text = clip_path.read_text(encoding="utf-8").splitlines()
    motion_index = next(i for i, line in enumerate(mini_text) if line.strip().startswith("MOTION"))
    motion_lines = mini_text[motion_index + 1:]
    frames_header = next(i for i, line in enumerate(motion_lines) if line.strip().startswith("Frames:"))
    keep = min(MINI_BVH_FRAMES, int(motion_lines[frames_header].split()[1]))
    assert motion_lines[frames_header + 1].strip().startswith("Frame Time:")
    first_frame = frames_header + 2
    mini_lines = mini_text[:motion_index] + ["MOTION"] + \
        [f"Frames: {keep}"] + [motion_lines[frames_header + 1]] + \
        motion_lines[first_frame:first_frame + keep]
    mini_path = FIXTURES / "mini_walk.mbvh"
    mini_path.write_text("\n".join(mini_lines) + "\n", encoding="utf-8")
    print(f"wrote {mini_path} ({keep} frames)")

    placement = wp.transform(
        wp.vec3(1.5, 0.25, -0.6),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(25.0)),
    )
    sample_times = [round(i * 0.05, 2) for i in range(21)]
    sample_poses = []
    pose = wp.empty(skeleton.num_joints, dtype=wp.transform)
    for offset_index, offset in ((0, wp.transform_identity()), (1, placement)):
        for time in sample_times:
            clip.sample(time, pose, root_offset=offset)
            sample_poses.append(
                {"time": time, "root_offset": offset_index, "pose": pose_batch(pose)}
            )
    bvh_doc = {
        "bvh_frames": keep,
        "bvh_sample_rate": clip.sample_rate,
        "root_offsets": [transform7(wp.transform_identity()), transform7(placement)],
        "samples": sample_poses,
    }
    (FIXTURES / "bvh_samples.json").write_text(json.dumps(bvh_doc), encoding="utf-8")
    print("wrote bvh_samples.json")

    # --- engine generations ---------------------------------------------------
    assets = REPO_ROOT / "assets" / "motions" / "soma"
    jog = load_bvh(str(assets / f"{JOG}.bvh"), skeleton)
    context_poses = wp.empty((MODEL_POSES_PER_TOKEN, skeleton.num_joints), dtype=wp.transform)
    target_poses = wp.empty((MODEL_POSES_PER_TOKEN, skeleton.num_joints), dtype=wp.transform)

    sample_window(clip, CONTEXT_END - (MODEL_POSES_PER_TOKEN - 1) / MODEL_SAMPLE_RATE, context_poses)
    sample_window(clip, TARGET_TIME_A, target_poses)

    generations = []
    with MotionBricksSOMAEngine(SOMA_N1_FP32_ONNX_PATH) as engine:
        for name in ("gen1", "gen2"):
            if name == "gen2":
                # Place a speed-adjusted jog target ahead of the character. Not a
                # behavioral recipe: the fixture only needs a valid placed target.
                sample_window(jog, TARGET_TIME_B, target_poses)
                adjust_target_pose_speed(target_poses, 1.9)
                facing = wp.vec3(1.0, 0.0, 0.0)
                ground = wp.vec3(0.75, 0.0, -0.4)
                place_target_poses(ground, facing, target_poses, target_poses)
            token_context = engine.create_input(skeleton)
            token_target = engine.create_input(skeleton)
            output = engine.create_output(skeleton)
            token_context.update_from_poses(context_poses)
            token_target.update_from_poses(target_poses)
            num_frames = engine.generate_animation(token_context, token_target, output).num_frames
            engine.synchronize()

            packed = {t: engine.device_tensor(t).numpy() for t in (
                "global_root_values", "local_root_values", "local_poses",
                "has_global_root_values", "has_local_root_values", "has_local_poses",
                "allowed_pred_num_tokens",
            )}
            pred = engine.device_tensor("pred_global_motions").numpy()
            tokens = int(engine.device_tensor("pred_num_tokens").numpy()[0])
            frames = output.poses.numpy()[:num_frames]
            contacts = output.foot_contacts.numpy()[:num_frames]

            doc = {
                "name": name,
                "context_poses": [pose_batch(context_poses[i]) for i in range(MODEL_POSES_PER_TOKEN)],
                "target_poses": [pose_batch(target_poses[i]) for i in range(MODEL_POSES_PER_TOKEN)],
                "alignment": transform7(engine._alignment.numpy()[0]),
                "packed": {
                    "global_root_values": [[fp(v) for v in frame] for frame in packed["global_root_values"][0]],
                    "local_root_values": [[fp(v) for v in frame] for frame in packed["local_root_values"][0]],
                    "local_poses": [[fp(v) for v in frame] for frame in packed["local_poses"][0]],
                    "has_global_root_values": [int(v) for v in packed["has_global_root_values"][0]],
                    "has_local_root_values": [int(v) for v in packed["has_local_root_values"][0]],
                    "has_local_poses": [int(v) for v in packed["has_local_poses"][0]],
                    "allowed_pred_num_tokens": [int(v) for v in packed["allowed_pred_num_tokens"][0]],
                },
                "pred_global_motions": [[fp(v) for v in frame] for frame in pred[0]],
                "pred_num_tokens": tokens,
                "num_frames": int(num_frames),
                "frames": [
                    [{"px": fp(frame[j, 0]), "py": fp(frame[j, 1]), "pz": fp(frame[j, 2]),
                      "qx": fp(frame[j, 3]), "qy": fp(frame[j, 4]), "qz": fp(frame[j, 5]), "qw": fp(frame[j, 6])}
                     for j in range(skeleton.num_joints)]
                    for frame in frames
                ],
                "foot_contacts": [[fp(v) for v in row] for row in contacts],
            }
            generations.append(doc)
            context_poses = wp.clone(output.poses[num_frames - MODEL_POSES_PER_TOKEN: num_frames])
            print(f"{name}: {num_frames} frames from {tokens} tokens")

    fixture = {
        "schema": 1,
        "model": "mmm_v6_soma_fp32",
        "skeleton_joint_count": skeleton.num_joints,
        "model_joint_names": MODEL_JOINT_NAMES,
        "generations": generations,
    }
    out = FIXTURES / "golden_soma.json"
    out.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
