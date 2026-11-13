"""Convert a Mixamo action from the current Blender scene into a MotionBricks SOMA .bvh.

Usage:
    Blender Text Editor -> Run Script (uses defaults), or:
        blender file.blend --python tools/blender/mixamo_to_soma_bvh.py -- --out out.bvh

The retarget maps the Mixamo rig onto the 77-joint SOMA skeleton defined in
unity/com.nvlabs.motionbricks/Runtime/Skeleton/soma_skeleton.json. That skeleton's reference
pose is a T-pose and its joint frames are bone-aligned (each frame's X axis points toward the
main child). This is exactly the convention MotionBricks' SOMA model consumes, because the
model's 6D pose features are FK world rotations of the BVH channel rotation data (see
motionbricks/model/soma/kernels.py, encode_motionbricks_inputs_kernel). The exported file is
Hips-rooted with no "Root" wrapper joint, so the model root joint "Hips" is stored at index
zero as motionbricks.model.soma.engine._bind_pose_skeleton requires.

Artifacts:
    - a SOMA-convention BVH (offsets in cm from the canonical reference pose, 3-channel
      Zrotation Yrotation Xrotation per joint, motion resampled to 120 Hz like the sample
      clips under assets/motions/soma)
    - a verification armature "SOMA_Retarget" in the scene with the retargeted action baked at
      the source frame rate for side-by-side comparison. The source rig is left untouched.

Self tests built in (the run aborts if any of them fail):
    - Euler ZYX decompose/compose round trip (including gimbal cases)
    - rest chain FK vs the canonical reference FK
    - retarget at the rest pose must reproduce the reference pose exactly
    - BVH file round trip: re-read the written file, FK every frame, compare against the
      in-memory retarget result
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import tempfile
import traceback

import bpy
from mathutils import Matrix, Quaternion, Vector

OUTPUT_FPS = 120.0
RIG_OBJECT_NAME = "SOMA_Retarget"
REPORT_TITLE = "SOMA_Conversion_Report"
SOMA_SKELETON_ENV = "SOMA_SKELETON_PATH"
SOMA_SKELETON_REL = os.path.join(
    "unity", "com.nvlabs.motionbricks", "Runtime", "Skeleton", "soma_skeleton.json"
)
MIXAMO_NAMESPACE = "mixamorig:"
HIPS_BONE = MIXAMO_NAMESPACE + "Hips"

MODEL_JOINTS_FALLBACK = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd",
    "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)

SOMA_SKELETON_FALLBACK = (
    ("Hips", -1, 0, 1.012054, 0, 0.496573, 0.503404, 0.496573, 0.503404),
    ("Spine1", 0, 0.051311, 0.000593, 0, 0, 0, -0.004111, 0.999992),
    ("Spine2", 1, 0.072061, 0, 0, 0, 0, -0.029529, 0.999564),
    ("Chest", 2, 0.07262, -0.000666, 0, 0, 0, 0.038697, 0.999251),
    ("Neck1", 3, 0.263508, 0, 0, 0, 0, 0.150518, 0.988607),
    ("Neck2", 4, 0.078721, 0, 0, 0, 0, 0, 1),
    ("Head", 5, 0.066593, 0, 0, 0, 0, -0.211553, 0.977367),
    ("HeadEnd", 6, 0.161575, 0.001599, 0, 0, 0, 0, 1),
    ("Jaw", 6, 0.001112, 0.030213, 0, 0, 0, 0.00497, 0.999988),
    ("LeftEye", 6, 0.042952, 0.08224, 0.031058, 0.033096, 0.001, 0.097265, 0.994708),
    ("RightEye", 6, 0.042952, 0.08224, -0.031058, -0.033096, -0.001, 0.097265, 0.994708),
    ("LeftShoulder", 3, 0.233884, 0.050382, 0.016136, 0.125499, -0.695881, -0.123037, 0.69632),
    ("LeftArm", 11, 0.15743, 0, 0, 0, 0, 0.175741, 0.984436),
    ("LeftForeArm", 12, 0.28916, 0, 0, 0, 0, 0, 1),
    ("LeftHand", 13, 0.270262, 0, 0, 0.084666, 0.000114, 0.000298, 0.996409),
    ("LeftHandThumb1", 14, 0.020852, 0.03313, 0.008397, 0.470529, -0.042939, 0.267821, 0.839661),
    ("LeftHandThumb2", 15, 0.047045, 0, 0, 0.181556, -0.140146, -0.232387, 0.945195),
    ("LeftHandThumb3", 16, 0.028564, 0, 0, 0, 0, 0, 1),
    ("LeftHandThumbEnd", 17, 0.031806, 0, 0, 0, 0, 0, 1),
    ("LeftHandIndex1", 14, 0.03171, 0.022789, 0.001212, -0.012302, 0.003106, 0.013654, 0.999826),
    ("LeftHandIndex2", 19, 0.063209, 0, 0, -0.072394, -0.002049, -0.014168, 0.997273),
    ("LeftHandIndex3", 20, 0.036355, 0, 0, 0, 0, 0, 1),
    ("LeftHandIndex4", 21, 0.023208, 0, 0, 0, 0, 0, 1),
    ("LeftHandIndexEnd", 22, 0.027612, 0, 0, 0, 0, 0, 1),
    ("LeftHandMiddle1", 14, 0.031009, 0.009073, -0.004079, -0.029877, -0.031692, -0.076527, 0.996116),
    ("LeftHandMiddle2", 24, 0.060545, 0, 0, -0.054568, 0.024995, 0.078635, 0.995095),
    ("LeftHandMiddle3", 25, 0.045741, 0, 0, 0, 0, 0, 1),
    ("LeftHandMiddle4", 26, 0.029249, 0, 0, 0, 0, 0, 1),
    ("LeftHandMiddleEnd", 27, 0.023414, 0, 0, 0, 0, 0, 1),
    ("LeftHandRing1", 14, 0.029041, -0.003612, 0.000799, -0.039477, -0.055117, -0.108545, 0.991777),
    ("LeftHandRing2", 29, 0.058845, 0, 0, -0.044639, 0.045627, 0.112522, 0.991597),
    ("LeftHandRing3", 30, 0.043685, 0, 0, 0, 0, 0, 1),
    ("LeftHandRing4", 31, 0.026751, 0, 0, 0, 0, 0, 1),
    ("LeftHandRingEnd", 32, 0.019416, 0, 0, 0, 0, 0, 1),
    ("LeftHandPinky1", 14, 0.028617, -0.0163, 0.005648, -0.07089, -0.13775, -0.15043, 0.976407),
    ("LeftHandPinky2", 34, 0.054877, 0, 0, -0.012057, 0.124429, 0.161253, 0.978964),
    ("LeftHandPinky3", 35, 0.030999, 0, 0, 0, 0, 0, 1),
    ("LeftHandPinky4", 36, 0.01585, 0, 0, 0, 0, 0, 1),
    ("LeftHandPinkyEnd", 37, 0.018838, 0, 0, 0, 0, 0, 1),
    ("RightShoulder", 3, 0.233885, 0.050382, -0.016167, 0.695881, 0.125499, 0.69632, 0.123037),
    ("RightArm", 39, -0.157431, 0, 0, 0, 0, 0.175741, 0.984436),
    ("RightForeArm", 40, -0.289151, 0, 0, 0, 0, 0, 1),
    ("RightHand", 41, -0.270262, 0, 0, 0.084666, 0.000114, 0.000298, 0.996409),
    ("RightHandThumb1", 42, -0.020851, -0.03313, -0.008396, 0.470542, -0.042946, 0.267809, 0.839657),
    ("RightHandThumb2", 43, -0.047047, 0, 0, 0.181544, -0.140133, -0.232382, 0.9452),
    ("RightHandThumb3", 44, -0.028564, 0, 0, 0, 0, 0, 1),
    ("RightHandThumbEnd", 45, -0.031806, 0, 0, 0, 0, 0, 1),
    ("RightHandIndex1", 42, -0.031714, -0.022786, -0.001217, -0.01228, 0.003145, 0.013672, 0.999826),
    ("RightHandIndex2", 47, -0.063207, 0, 0, -0.072416, -0.002087, -0.014189, 0.997271),
    ("RightHandIndex3", 48, -0.036355, 0, 0, 0, 0, 0, 1),
    ("RightHandIndex4", 49, -0.023208, 0, 0, 0, 0, 0, 1),
    ("RightHandIndexEnd", 50, -0.027612, 0, 0, 0, 0, 0, 1),
    ("RightHandMiddle1", 42, -0.03101, -0.009073, 0.004079, -0.029878, -0.03169, -0.076524, 0.996116),
    ("RightHandMiddle2", 52, -0.060546, 0, 0, -0.054567, 0.024992, 0.078632, 0.995095),
    ("RightHandMiddle3", 53, -0.045741, 0, 0, 0, 0, 0, 1),
    ("RightHandMiddle4", 54, -0.029248, 0, 0, 0, 0, 0, 1),
    ("RightHandMiddleEnd", 55, -0.023415, 0, 0, 0, 0, 0, 1),
    ("RightHandRing1", 42, -0.029041, 0.003612, -0.000798, -0.039477, -0.055113, -0.108543, 0.991777),
    ("RightHandRing2", 57, -0.058847, 0, 0, -0.044639, 0.045623, 0.11252, 0.991597),
    ("RightHandRing3", 58, -0.043685, 0, 0, 0, 0, 0, 1),
    ("RightHandRing4", 59, -0.026752, 0, 0, 0, 0, 0, 1),
    ("RightHandRingEnd", 60, -0.019416, 0, 0, 0, 0, 0, 1),
    ("RightHandPinky1", 42, -0.028619, 0.016301, -0.005648, -0.070891, -0.137748, -0.150424, 0.976408),
    ("RightHandPinky2", 62, -0.054877, 0, 0, -0.012056, 0.124427, 0.161247, 0.978965),
    ("RightHandPinky3", 63, -0.030998, 0, 0, 0, 0, 0, 1),
    ("RightHandPinky4", 64, -0.01585, 0, 0, 0, 0, 0, 1),
    ("RightHandPinkyEnd", 65, -0.018839, 0, 0, 0, 0, 0, 1),
    ("LeftLeg", 0, -0.083079, 0.021259, 0.100069, 0, 0, 0.999977, -0.00683),
    ("LeftShin", 67, 0.433003, 0, 0, 0, 0, 0, 1),
    ("LeftFoot", 68, 0.424241, 0, 0, 0, 0, -0.566943, 0.823757),
    ("LeftToeBase", 69, 0.141328, 0, 0, 0.000235, 0.000699, -0.04126, 0.999148),
    ("LeftToeEnd", 70, 0.067743, -0.001182, 0, 0, 0, -0.008727, 0.999962),
    ("RightLeg", 0, -0.083079, 0.021259, -0.100015, 0, 0, 0.00683, 0.999977),
    ("RightShin", 72, -0.433004, 0, 0, 0, 0, 0, 1),
    ("RightFoot", 73, -0.42424, 0, 0, 0, 0, -0.566943, 0.823757),
    ("RightToeBase", 74, -0.141328, 0, 0, 0.000235, 0.000699, -0.041258, 0.999148),
    ("RightToeEnd", 75, -0.067743, 0.001182, 0, 0, 0, -0.008727, 0.999962),
)

DST_TO_SRC = {
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Chest": "Spine2",
    "Neck1": "Neck",
    "LeftLeg": "LeftUpLeg",
    "RightLeg": "RightUpLeg",
    "LeftShin": "LeftLeg",
    "RightShin": "RightLeg",
    "LeftToeEnd": "LeftToe_End",
    "RightToeEnd": "RightToe_End",
    "HeadEnd": "HeadTop_End",
    "LeftHandThumbEnd": "LeftHandThumb4",
    "RightHandThumbEnd": "RightHandThumb4",
}

RIGID_TIP_SRC = {
    "LeftHandIndexEnd": "LeftHandIndex4",
    "RightHandIndexEnd": "RightHandIndex4",
    "LeftHandMiddleEnd": "LeftHandMiddle4",
    "RightHandMiddleEnd": "RightHandMiddle4",
    "LeftHandRingEnd": "LeftHandRing4",
    "RightHandRingEnd": "RightHandRing4",
    "LeftHandPinkyEnd": "LeftHandPinky4",
    "RightHandPinkyEnd": "RightHandPinky4",
}

MUST_JOINTS = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Head", "LeftShoulder", "RightShoulder",
    "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand", "LeftLeg",
    "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase",
)

FIT_PAIRS = (
    ("Hips", "Neck", "Hips", "Neck1"),
    ("Spine1", "Neck", "Chest", "Neck1"),
    ("LeftShoulder", "LeftArm", "LeftShoulder", "LeftArm"),
    ("RightShoulder", "RightArm", "RightShoulder", "RightArm"),
    ("LeftArm", "LeftForeArm", "LeftArm", "LeftForeArm"),
    ("RightArm", "RightForeArm", "RightArm", "RightForeArm"),
    ("LeftForeArm", "LeftHand", "LeftForeArm", "LeftHand"),
    ("RightForeArm", "RightHand", "RightForeArm", "RightHand"),
    ("LeftUpLeg", "LeftLeg", "LeftLeg", "LeftShin"),
    ("RightUpLeg", "RightLeg", "RightLeg", "RightShin"),
    ("LeftLeg", "LeftFoot", "LeftShin", "LeftFoot"),
    ("RightLeg", "RightFoot", "RightShin", "RightFoot"),
    ("LeftFoot", "LeftToeBase", "LeftFoot", "LeftToeBase"),
    ("RightFoot", "RightToeBase", "RightFoot", "RightToeBase"),
    ("LeftToeBase", "LeftToe_End", "LeftToeBase", "LeftToeEnd"),
    ("RightToeBase", "RightToe_End", "RightToeBase", "RightToeEnd"),
    ("Neck", "Head", "Neck1", "Head"),
    ("Head", "HeadTop_End", "Head", "HeadEnd"),
)

_TAIL_CHILD_PRIORITY = ("Spine", "Neck", "Head", "ForeArm", "Shin", "Arm", "Leg", "Foot", "Hand")


def _clamp(value, low, high):
    return max(low, min(high, value))


_LOG_LINES = []


def log(message):
    """Print a progress line and keep it for the in-Blender report.

    Blender builds without a system console (e.g. Steam installs) drop stdout, so every message
    is also collected and shown in a text datablock when the run finishes.
    """
    _LOG_LINES.append(str(message))
    print(message)


def write_report(text=None):
    """Show the conversion report inside Blender (console output may not be visible)."""
    body = text if text is not None else "\n".join(_LOG_LINES)
    name = REPORT_TITLE
    existing = bpy.data.texts.get(name)
    if existing is not None:
        bpy.data.texts.remove(existing)
    block = bpy.data.texts.new(name)
    block.write(body)
    try:
        for area in bpy.context.screen.areas:
            if area.type == "TEXT_EDITOR":
                area.spaces.active.text = block
                break
    except AttributeError:
        pass
    return block


def _axis_quat(axis: str, degrees: float) -> Quaternion:
    return Matrix.Rotation(math.radians(degrees), 3, axis).to_quaternion()


def quat_from_zyx_deg(z_deg, y_deg, x_deg):
    """BVH channels list rotations in file order; R = Rz @ Ry @ Rx for 'Zrotation Yrotation Xrotation'."""
    return _axis_quat("Z", z_deg) @ _axis_quat("Y", y_deg) @ _axis_quat("X", x_deg)


def quat_to_zyx_deg(q):
    """Decompose q so quat_from_zyx_deg(z, y, x) == q. Returns (z, y, x) in degrees."""
    m = q.to_matrix()
    s_beta = _clamp(-m[2][0], -1.0, 1.0)
    beta = math.asin(s_beta)
    if abs(m[2][0]) < 0.99999:
        alpha = math.atan2(m[1][0], m[0][0])
        gamma = math.atan2(m[2][1], m[2][2])
    else:
        alpha = math.atan2(m[1][2], m[1][1]) if s_beta > 0.0 else math.atan2(-m[1][2], m[1][1])
        gamma = 0.0
    return (math.degrees(alpha), math.degrees(beta), math.degrees(gamma))


def quat_geodesic_deg(qa, qb):
    d = qa.inverted() @ qb
    angle = 2.0 * math.atan2(math.sqrt(d.x * d.x + d.y * d.y + d.z * d.z), abs(d.w))
    return math.degrees(angle)


def quat_slerp(q0, q1, t):
    d = _clamp(q0.dot(q1), -1.0, 1.0)
    neg = d < 0.0
    if neg:
        d = -d
    if d > 0.9995:
        sign = -1.0 if neg else 1.0
        return Quaternion((
            q0.x + (sign * q1.x - q0.x) * t,
            q0.y + (sign * q1.y - q0.y) * t,
            q0.z + (sign * q1.z - q0.z) * t,
            q0.w + (sign * q1.w - q0.w) * t,
        )).normalized()
    theta = math.acos(d)
    s1 = math.sin((1.0 - t) * theta) / math.sin(theta)
    s2 = math.sin(t * theta) / math.sin(theta)
    if neg:
        s2 = -s2
    return Quaternion((
        s1 * q0.w + s2 * q1.w,
        s1 * q0.x + s2 * q1.x,
        s1 * q0.y + s2 * q1.y,
        s1 * q0.z + s2 * q1.z,
    )).normalized()


def fk_world(offsets_m, root_pos, rch, parents):
    """World transforms of BVH-style locals. offsets_m: meters. rch: local rotation quaternions."""
    n = len(rch)
    world_q = [None] * n
    world_p = [None] * n
    for i in range(n):
        local_pos = root_pos if i == 0 else offsets_m[i]
        local_q = rch[i]
        par = parents[i]
        if par < 0:
            world_q[i] = local_q
            world_p[i] = local_pos
        else:
            world_q[i] = (world_q[par] @ local_q).normalized()
            world_p[i] = world_p[par] + world_q[par] @ local_pos
    return world_q, world_p


def load_soma_skeleton(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        names = list(doc["joint_names"])
        parents = list(doc["parent_indices"])
        if not (len(names) == len(parents) == len(doc["reference_local"])):
            raise ValueError("soma_skeleton.json tables have mismatched lengths")
        skel = {"names": names, "parents": parents, "p_loc": [], "q_loc": []}
        for entry in doc["reference_local"]:
            skel["p_loc"].append(Vector((entry["px"], entry["py"], entry["pz"])))
            skel["q_loc"].append(
                Quaternion((entry["qw"], entry["qx"], entry["qy"], entry["qz"])).normalized())
        return skel
    names = [row[0] for row in SOMA_SKELETON_FALLBACK]
    parents = [row[1] for row in SOMA_SKELETON_FALLBACK]
    skel = {"names": names, "parents": parents, "p_loc": [], "q_loc": []}
    for row in SOMA_SKELETON_FALLBACK:
        skel["p_loc"].append(Vector((row[2], row[3], row[4])))
        skel["q_loc"].append(Quaternion((row[8], row[5], row[6], row[7])).normalized())
    log(f"[skel] using the embedded SOMA skeleton ({len(names)} joints)")
    return skel


def reference_fk(skel):
    """FK of the skeleton's reference local transforms; writes skel['P'], skel['Q'], skel['children']."""
    n = len(skel["names"])
    parents = skel["parents"]
    P = [None] * n
    Q = [None] * n
    for i in range(n):
        par = parents[i]
        if par < 0:
            P[i] = skel["p_loc"][i].copy()
            Q[i] = skel["q_loc"][i]
        else:
            P[i] = P[par] + Q[par] @ skel["p_loc"][i]
            Q[i] = (Q[par] @ skel["q_loc"][i]).normalized()
    skel["P"] = P
    skel["Q"] = Q
    skel["children"] = [[j for j in range(n) if parents[j] == i] for i in range(n)]


def load_model_joint_names():
    manifest = _find_repo_file(os.path.join(
        "motionbricks", "assets", "onnx", "soma", "soma_n1.json"))
    if manifest:
        try:
            with open(manifest, encoding="utf-8") as handle:
                return list(json.load(handle)["joint_names"])
        except (OSError, KeyError):
            pass
    return list(MODEL_JOINTS_FALLBACK)


def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return None


def _repo_roots():
    roots = []
    script_dir = _script_dir()
    if script_dir:
        roots.append(os.path.normpath(os.path.join(script_dir, "..", "..")))
        roots.append(os.path.normpath(os.path.join(script_dir, "..", "..", "..")))
    blend_path = bpy.data.filepath
    if blend_path:
        blend_dir = os.path.dirname(os.path.abspath(blend_path))
        roots.append(blend_dir)
        roots.append(os.path.normpath(os.path.join(blend_dir, "..")))
        roots.append(os.path.normpath(os.path.join(blend_dir, "..", "..")))
    roots.append(os.getcwd())
    roots.append(os.path.normpath(os.path.join(os.getcwd(), "..")))
    seen = []
    for root in roots:
        if root and root not in seen:
            seen.append(root)
    return seen


def _find_repo_file(relative_path):
    env_override = os.environ.get(SOMA_SKELETON_ENV)
    if env_override and os.path.exists(env_override):
        return env_override
    for root in _repo_roots():
        candidate = os.path.join(root, relative_path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def _find_repo_root():
    """Repository root when the script runs from a checkout, else None."""
    found = _find_repo_file(SOMA_SKELETON_REL)
    if found:
        return os.path.dirname(os.path.abspath(found)).split(os.sep + "unity" + os.sep)[0]
    return None


def default_soma_skeleton_path():
    return _find_repo_file(SOMA_SKELETON_REL)


def _strip_namespace(name):
    return name.rsplit(":", 1)[-1]


def find_source_rig(source_name=None):
    if source_name:
        obj = bpy.data.objects.get(source_name)
        if obj is None or obj.type != "ARMATURE":
            raise ValueError(f"source armature '{source_name}' not found")
        return obj
    mixamo_rigs = [o for o in bpy.data.objects if o.type == "ARMATURE" and HIPS_BONE in o.pose.bones]
    animated = [o for o in mixamo_rigs if o.animation_data and o.animation_data.action]
    if len(animated) != 1:
        found = [o.name for o in mixamo_rigs]
        raise ValueError(f"expected exactly one animated mixamorig armature, found: {found}")
    return animated[0]


def build_mapping(source_rig, soma_names):
    src_by = {}
    for bone in source_rig.data.bones:
        src_by[_strip_namespace(bone.name)] = bone.name
    if HIPS_BONE not in source_rig.pose.bones:
        raise ValueError(f"source rig '{source_rig.name}' has no '{HIPS_BONE}' bone")
    mapping = {}
    missing = []
    for som_name in soma_names:
        src = DST_TO_SRC.get(som_name)
        if src is None:
            src = som_name if som_name in src_by else RIGID_TIP_SRC.get(som_name)
        if src is not None and src in src_by:
            mapping[som_name] = src_by[src]
        else:
            missing.append(som_name)
    consumers = {}
    for som_name, src in mapping.items():
        consumers.setdefault(src, []).append(som_name)
    for src, users in consumers.items():
        if len(users) > 1 and any(u in MUST_JOINTS for u in users):
            raise ValueError(f"source bone {src} would drive several required joints: {users}")
    must_missing = [m for m in MUST_JOINTS if m not in mapping]
    if must_missing:
        raise ValueError(f"the source rig lacks required joints (SOMA: {must_missing})")
    return mapping, missing


def _bone_depth(bone):
    depth = 0
    node = bone
    while node.parent is not None:
        depth += 1
        node = node.parent
    return depth


def _index_action_fcurves(act):
    index = {}
    for fc in act.fcurves:
        path = fc.data_path
        if not path.startswith("pose.bones["):
            continue
        bone_name = path.split('["', 1)[1].split('"]', 1)[0]
        channel = path.rsplit(".", 1)[-1]
        if channel not in ("location", "rotation_quaternion", "scale"):
            continue
        per_bone = index.setdefault(bone_name, {})
        per_bone.setdefault(channel, {})[fc.array_index] = fc
    return index


def sample_source_poses(source_rig, mapping, start, end):
    """Per source frame: world quaternions of every mapped mixamo bone + hips translation.

    Computed by pure fcurve FK with the canonical Blender pose composition
    matrix_j = parent_matrix @ inv(parent_rest) @ rest_j @ basis_j, because depsgraph-driven
    pose evaluation can stall in headless contexts (pose channels receive the keyed values while
    the evaluated matrices stay frozen). matrix_local / fcurves are plain data, so this is exact
    and side effect free (no frame changes, no depsgraph).
    """
    pose_bones = source_rig.pose.bones
    for pb in pose_bones:
        if pb.rotation_mode != "QUATERNION":
            raise ValueError(f"bone '{pb.name}' uses rotation mode {pb.rotation_mode}; "
                             "only QUATERNION mixamo actions are supported")
    ordered = sorted(source_rig.data.bones, key=_bone_depth)
    rest_m = {b.name: b.matrix_local.copy() for b in ordered}
    fcs = _index_action_fcurves(source_rig.animation_data.action)
    static = {}
    for name, pb in pose_bones.items():
        q = pb.rotation_quaternion
        static[name] = {
            "location": tuple(pb.location),
            "rotation_quaternion": tuple((q.w, q.x, q.y, q.z)),
            "scale": tuple(pb.scale),
        }

    def _channel(channels, stat, name, idx):
        fc = channels.get(name, {}).get(idx)
        return float(fc.evaluate(frame)) if fc else stat[name][idx]

    per_frame = []
    for frame in range(start, end + 1):
        worlds = {}
        for bone in ordered:
            channels = fcs.get(bone.name, {})
            stat = static.get(bone.name)
            loc = tuple(_channel(channels, stat, "location", i) for i in range(3))
            rot = tuple(_channel(channels, stat, "rotation_quaternion", i) for i in range(4))
            scl = tuple(_channel(channels, stat, "scale", i) for i in range(3))
            basis = Matrix.Translation(loc) @ Quaternion(rot).normalized().to_matrix().to_4x4() \
                @ Matrix.Diagonal((scl[0], scl[1], scl[2], 1.0))
            parent = bone.parent
            if parent is None:
                worlds[bone.name] = rest_m[bone.name] @ basis
            else:
                worlds[bone.name] = (worlds[parent.name] @ rest_m[parent.name].inverted()
                                     @ rest_m[bone.name] @ basis)
        quats = {som_name: worlds[src].to_quaternion().normalized() for som_name, src in mapping.items()}
        per_frame.append({"quats": quats, "hips": worlds[HIPS_BONE].translation.copy()})
    return per_frame


def fit_yaw_alignment(source_rig, skel, mapping):
    """Yaw about the shared up axis that aligns the source's rest T-pose to the SOMA reference T-pose.

    Raises when the two rest poses disagree beyond cosmetic anatomical differences, so that the
    axis conventions cannot silently drift.
    """
    bones = source_rig.data.bones
    names = skel["names"]
    pairs = []
    num = 0.0
    den = 0.0
    for src_bone, src_child, som_joint, som_child in FIT_PAIRS:
        src_bone = MIXAMO_NAMESPACE + src_bone
        src_child = MIXAMO_NAMESPACE + src_child
        if src_bone not in bones or src_child not in bones:
            continue
        som_idx = names.index(som_joint)
        som_child_idx = names.index(som_child)
        v_src = bones[src_child].head_local - bones[src_bone].head_local
        v_soma = skel["P"][som_child_idx] - skel["P"][som_idx]
        if v_src.length < 1e-6 or v_soma.length < 1e-6:
            continue
        u = Vector((v_src.x, 0.0, v_src.z))
        v = Vector((v_soma.x, 0.0, v_soma.z))
        if min(u.length / v_src.length, v.length / v_soma.length) < 0.25:
            continue
        d_src = v_src.normalized()
        d_soma = v_soma.normalized()
        u = u.normalized()
        v = v.normalized()
        num += u.z * v.x - u.x * v.z
        den += u.x * v.x + u.z * v.z
        pairs.append((som_joint, u, v))
    if not pairs:
        raise ValueError("no comparable rest-pose chains found for the yaw fit")
    theta = math.atan2(num, den)
    yaw_q = _axis_quat("Y", math.degrees(theta))
    residuals = [
        (som_joint, math.degrees((yaw_q @ u).angle(v))) for som_joint, u, v in pairs
    ]
    return yaw_q, math.degrees(theta), residuals


def retarget_frame(sample, mapping, skel, yaw_q):
    """World quaternions W per SOMA joint.

    W_j = (yaw @ src_q) @ inv(yaw @ src_rest_q) @ Q_ref_j, i.e. the source bone's rest-aligned
    rotation delta transplanted onto the SOMA bone-aligned reference frame. Unmapped joints stay
    at the reference orientation.
    """
    names = skel["names"]
    Q = skel["Q"]
    src_rest = skel["src_rest"]
    quats = sample["quats"]
    world = []
    for i, name in enumerate(names):
        src = mapping.get(name)
        if src is None:
            world.append(Q[i])
            continue
        src_q = (yaw_q @ quats[name]).normalized()
        src_rest_q = (yaw_q @ src_rest[src]).normalized()
        delta = src_q @ src_rest_q.inverted()
        world.append((delta @ Q[i]).normalized())
    return world


def hips_world(sample, yaw_q, scale):
    return yaw_q.to_matrix() @ (sample["hips"] * scale)


def channels_from_world(world, parents):
    """BVH local rotation channels: R_ch_root = W_root; R_ch_j = inv(W_parent) @ W_j."""
    rch = []
    for i, q in enumerate(world):
        par = parents[i]
        rch.append(q if par < 0 else world[par].inverted() @ q)
    return rch


def resample_to_fps(world_frames, hips_frames, duration_s, out_fps):
    if len(world_frames) < 2:
        return [world_frames[0]], [hips_frames[0]]
    count = max(2, int(round(duration_s * out_fps)) + 1)
    span = duration_s / (len(world_frames) - 1)
    out_world = []
    out_hips = []
    for m in range(count):
        seconds = min(duration_s, m / out_fps)
        pos = seconds / span if span > 0 else 0.0
        k0 = min(int(math.floor(pos)), len(world_frames) - 2)
        frac = _clamp(pos - k0, 0.0, 1.0)
        out_world.append([
            quat_slerp(world_frames[k0][j], world_frames[k0 + 1][j], frac)
            for j in range(len(world_frames[0]))
        ])
        out_hips.append(hips_frames[k0].lerp(hips_frames[k0 + 1], frac))
    return out_world, out_hips


def make_quats_sign_continuous(rch_frames):
    """Flip quaternion signs so interpolation between frames never takes the long way around."""
    out = [[q.copy() for q in rch_frames[0]]]
    for row in rch_frames[1:]:
        prev = out[-1]
        cur = []
        for i, q in enumerate(row):
            q = q.copy()
            if q.dot(prev[i]) < 0.0:
                q.negate()
            cur.append(q)
        out.append(cur)
    return out


def write_bvh(path, names, parents, p_loc, rch_frames, hips_frames, fps):
    children = [
        [j for j in range(len(names)) if parents[j] == i] for i in range(len(names))
    ]
    lines = [
        "HIERARCHY",
        "ROOT Hips",
        "{",
        "  OFFSET 0.000000 0.000000 0.000000",
        "  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation",
    ]

    def emit(joint, depth):
        off = p_loc[joint] * 100.0
        lines.append(f"{'  ' * (depth + 1)}JOINT {names[joint]}")
        lines.append("  " * (depth + 1) + "{")
        lines.append(f"{'  ' * (depth + 2)}OFFSET {off.x:.6f} {off.y:.6f} {off.z:.6f}")
        lines.append(f"{'  ' * (depth + 2)}CHANNELS 3 Zrotation Yrotation Xrotation")
        for child in children[joint]:
            emit(child, depth + 2)
        lines.append("  " * (depth + 1) + "}")

    for child in children[0]:
        emit(child, 0)
    lines.append("}")
    lines.append("MOTION")
    lines.append(f"Frames: {len(rch_frames)}")
    lines.append(f"Frame Time: {1.0 / fps:.6f}")
    for rch, hips in zip(rch_frames, hips_frames):
        hd = quat_to_zyx_deg(rch[0])
        values = [
            f"{hips.x * 100.0:.6f} {hips.y * 100.0:.6f} {hips.z * 100.0:.6f}",
            f"{hd[0]:.6f} {hd[1]:.6f} {hd[2]:.6f}",
        ]
        for j in range(1, len(rch)):
            d = quat_to_zyx_deg(rch[j])
            values.append(f"{d[0]:.6f} {d[1]:.6f} {d[2]:.6f}")
        lines.append(" ".join(values))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def read_bvh(path):
    """Minimal BVH reader for the self tests. Mirrors motionbricks/io/bvh.py semantics."""
    names, parents, offsets, channels = [], [], [], []
    stack = []
    in_end_site = False
    in_motion = False
    samples = []
    frame_time = None
    declared = None
    with open(path, encoding="utf-8") as handle:
        raw = handle.read().splitlines()
    for line in raw:
        tokens = line.split()
        if not tokens:
            continue
        if not in_motion:
            if tokens[0] == "MOTION":
                in_motion = True
            elif tokens[0] in ("ROOT", "JOINT"):
                names.append(_strip_namespace(tokens[1]))
                parents.append(stack[-1] if stack else -1)
                stack.append(len(names) - 1)
                offsets.append([0.0, 0.0, 0.0])
                channels.append([])
            elif tokens[:2] == ["End", "Site"]:
                in_end_site = True
            elif tokens[0] == "OFFSET" and not in_end_site:
                offsets[stack[-1]] = [float(v) for v in tokens[1:4]]
            elif tokens[0] == "CHANNELS":
                channels[stack[-1]] = tokens[2:]
            elif tokens[0] == "}":
                if in_end_site:
                    in_end_site = False
                else:
                    stack.pop()
        else:
            if tokens[0] == "Frames:":
                declared = int(tokens[1])
            elif tokens[:2] == ["Frame", "Time:"]:
                frame_time = float(tokens[2])
            else:
                samples.append([float(v) for v in tokens])
    per_frame = []
    for row in samples:
        loc = []
        rot = []
        column = 0
        for ch in channels:
            pos = [0.0, 0.0, 0.0]
            entries = []
            for token in ch:
                axis = token[0].lower()
                value = row[column]
                if token.lower().endswith("position"):
                    pos["xyz".index(axis)] = value
                else:
                    entries.append((axis, value))
                column += 1
            loc.append(Vector(pos))
            rot.append(entries)
        per_frame.append((loc, rot))
    return {"names": names, "parents": parents,
            "offsets_m": [Vector(o) * 0.01 for o in offsets],
            "channels": channels, "per_frame": per_frame,
            "declared": declared, "frame_time": frame_time}


def frames_from_reader(data):
    """Per frame: (local rotation quats per joint, hips world pos in meters)."""
    frames = []
    for loc, rot in data["per_frame"]:
        rch = []
        for j in range(len(data["names"])):
            quat = None
            for axis, value in rot[j]:
                step = _axis_quat(axis.upper(), value)
                quat = step if quat is None else quat @ step
            rch.append(quat.normalized())
        frames.append((rch, loc[0] * 0.01))
    return frames


def y_up_to_blender():
    """Rotation taking SOMA Y-up world points/rotations into Blender Z-up armature space."""
    return Matrix.Rotation(math.radians(90.0), 3, "X")


def euler_self_test():
    problems = []
    probes = [
        quat_from_zyx_deg(0.0, 0.0, 0.0),
        quat_from_zyx_deg(35.0, 90.0, 15.0),
        quat_from_zyx_deg(-35.0, -90.0, -15.0),
        quat_from_zyx_deg(33.0, 47.0, 21.0),
        quat_from_zyx_deg(-120.0, 15.0, 87.0),
        quat_from_zyx_deg(10.0, -95.0, 92.0),
    ]
    worst = 0.0
    for i, q in enumerate(probes):
        z, y, x = quat_to_zyx_deg(q)
        err = quat_geodesic_deg(q, quat_from_zyx_deg(z, y, x))
        worst = max(worst, err)
        if err > 1e-3:
            problems.append(f"euler round trip probe {i} failed: {err:.8f} deg")
    log(f"[test] Euler ZYX round trip max error: {worst:.10f} deg")
    return problems


def rest_fk_test(skel):
    W, P = fk_world(skel["p_loc"], skel["P"][0], list(skel["q_loc"]), skel["parents"])
    pos_err = max((a - b).length for a, b in zip(P, skel["P"]))
    rot_err = max(quat_geodesic_deg(a, b) for a, b in zip(W, skel["Q"]))
    log(f"[test] rest chain FK vs reference FK: pos_err={pos_err:.2e} m rot_err={rot_err:.2e} deg")
    problems = []
    if pos_err > 1e-5 or rot_err > 1e-4:
        problems.append(f"rest fk test failed: pos_err={pos_err:.2e} m rot_err={rot_err:.2e} deg")
    return problems


def rest_retarget_test(skel, mapping, yaw_q, scale):
    """Retargeting a frame whose source bones sit exactly at their rest poses must reproduce the
    SOMA reference pose exactly, and the written file must FK back to the same thing."""
    hips_rest = skel["src_rest_hips"]
    sample = {
        "quats": {name: skel["src_rest"][src] for name, src in mapping.items()},
        "hips": hips_rest.copy(),
    }
    world = retarget_frame(sample, mapping, skel, yaw_q)
    hips = hips_world(sample, yaw_q, scale)
    rch = channels_from_world(world, skel["parents"])
    rot_err = max(quat_geodesic_deg(a, b) for a, b in zip(rch, skel["q_loc"]))

    path_obj = tempfile.NamedTemporaryFile(suffix=".bvh", delete=False)
    path_obj.close()
    os.remove(path_obj.name)
    write_bvh(path_obj.name, skel["names"], skel["parents"], skel["p_loc"],
              [rch], [hips], OUTPUT_FPS)
    try:
        data = read_bvh(path_obj.name)
        frames = frames_from_reader(data)
        back_rch, back_hips = frames[0]
        file_rot_err = max(quat_geodesic_deg(a, b) for a, b in zip(back_rch, skel["q_loc"]))
        file_pos_err = (back_hips - hips).length
        W_file, _ = fk_world(skel["p_loc"], back_hips, back_rch, skel["parents"])
        file_fk_rot_err = max(quat_geodesic_deg(a, b) for a, b in zip(W_file, skel["Q"]))
    finally:
        if os.path.exists(path_obj.name):
            os.remove(path_obj.name)
    log(f"[test] rest retarget in memory: rot_err={rot_err:.2e} deg")
    log(f"[test] rest retarget file round trip: rot_err={file_rot_err:.2e} deg "
          f"hips_pos_err={file_pos_err:.2e} m fk_rot_err={file_fk_rot_err:.2e} deg")
    problems = []
    if rot_err > 1e-3 or file_rot_err > 1e-3 or file_pos_err > 1e-5 or file_fk_rot_err > 1e-3:
        problems.append(f"rest retarget test failed: rot={rot_err:.2e} file_rot={file_rot_err:.2e} "
                        f"file_pos={file_pos_err:.2e} file_fk={file_fk_rot_err:.2e}")
    return problems


def roundtrip_test(path, skel, expected_rch, expected_hips, expected_world_p):
    data = read_bvh(path)
    problems = []
    if data["names"] != skel["names"]:
        problems.append("round trip: joint names differ")
    if data["parents"] != skel["parents"]:
        problems.append("round trip: parents differ")
    if data["declared"] != len(data["per_frame"]):
        problems.append(f"round trip: Frames:={data['declared']} but {len(data['per_frame'])} rows")
    for i, off in enumerate(data["offsets_m"]):
        if i == 0:
            continue
        if (off - skel["p_loc"][i]).length > 1e-6:
            problems.append(f"round trip: offset mismatch at {skel['names'][i]}")
            break
    if problems:
        return problems, 1e9, 1e9
    max_rot = 0.0
    max_pos = 0.0
    frames = frames_from_reader(data)
    if len(frames) != len(expected_rch):
        problems.append(f"round trip: {len(frames)} frames read, expected {len(expected_rch)}")
        return problems, max_rot, max_pos
    for (rch, hips_pos), exp_rch, exp_hips, exp_p in zip(
            frames, expected_rch, expected_hips, expected_world_p):
        for got, want in zip(rch, exp_rch):
            max_rot = max(max_rot, quat_geodesic_deg(got, want))
        max_pos = max(max_pos, (hips_pos - exp_hips).length)
        _, P = fk_world(skel["p_loc"], hips_pos, rch, skel["parents"])
        max_pos = max(max_pos, max((a - b).length for a, b in zip(P, exp_p)))
    log(f"[test] BVH round trip: max rot err={max_rot:.8f} deg  max pos err={max_pos:.10f} m")
    if max_rot > 1e-2 or max_pos > 1e-4:
        problems.append(f"round trip test failed: rot_err={max_rot:.2e} deg pos_err={max_pos:.2e} m")
    return problems, max_rot, max_pos


def model_joint_coverage(model_joints, soma_names):
    missing = [m for m in model_joints if m not in soma_names]
    log(f"[test] model joint coverage: {len(model_joints) - len(missing)}/{len(model_joints)}")
    if missing:
        log(f"[warn] model joints missing from output skeleton: {missing}")
    return missing


def _choose_bone_tail(joint, skel):
    children = skel["children"][joint]
    if not children:
        return None
    for pref in _TAIL_CHILD_PRIORITY:
        for kid in children:
            if pref in skel["names"][kid]:
                return kid
    return max(children, key=lambda kid: (skel["P"][kid] - skel["P"][joint]).length)


def _purge_scene_artifacts(rig_name, action_name):
    obj = bpy.data.objects.get(rig_name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)
    data = bpy.data.armatures.get(rig_name)
    if data is not None and data.users == 0:
        bpy.data.armatures.remove(data)
    stale = bpy.data.actions.get(action_name)
    if stale is not None and stale.users == 0:
        bpy.data.actions.remove(stale)


def create_scene_rig(scene, skel, frames, src_action_name, src_start):
    """Verification armature 'SOMA_Retarget' with the retargeted action baked at source frames.

    frames: per source frame, a (world_q, world_p) pair from fk_world() on the retargeted clips.
    """
    names = skel["names"]
    parents = skel["parents"]
    n = len(names)
    action_name = f"{src_action_name}_soma"
    _purge_scene_artifacts(RIG_OBJECT_NAME, action_name)

    C = y_up_to_blender()
    Cq = C.to_quaternion()
    Cq_inv = Cq.inverted()

    arm = bpy.data.armatures.new(RIG_OBJECT_NAME)
    obj = bpy.data.objects.new(RIG_OBJECT_NAME, arm)
    scene.collection.objects.link(obj)

    prev_active = bpy.context.view_layer.objects.active
    prev_selected = {o.name: o.select_get() for o in bpy.data.objects}
    for o in bpy.data.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = {}
    for j in range(n):
        eb = arm.edit_bones.new(names[j])
        head_bl = C @ skel["P"][j]
        tail_kid = _choose_bone_tail(j, skel)
        if tail_kid is None:
            par = parents[j]
            if par >= 0:
                handle = head_bl - (C @ skel["P"][par])
            else:
                handle = Vector((0.0, 0.0, 0.03))
            if handle.length < 1e-6:
                handle = Vector((0.0, 0.0, 0.03))
            eb.tail = head_bl + handle.normalized() * 0.03
        else:
            eb.tail = C @ skel["P"][tail_kid]
        if parents[j] >= 0:
            eb.parent = edit_bones[names[parents[j]]]
        edit_bones[names[j]] = eb
    for j in range(n):
        eb = edit_bones[names[j]]
        desired = ((Cq @ skel["Q"][j]) @ Cq_inv) @ _axis_quat("Z", -90.0)
        eb.matrix = Matrix.Translation(C @ skel["P"][j]) @ desired.to_matrix().to_4x4()
    bpy.ops.object.mode_set(mode="OBJECT")

    rest_4 = [arm.bones[name].matrix_local.copy() for name in names]
    rest_q = [m.to_quaternion() for m in rest_4]
    carry = []
    for j in range(n):
        frame_ref_bl = ((Cq @ skel["Q"][j]) @ Cq_inv)
        carry.append(frame_ref_bl.inverted() @ rest_q[j])

    action = bpy.data.actions.new(action_name)
    action.use_fake_user = True
    obj.animation_data_create()
    obj.animation_data.action = action

    bpy.ops.object.mode_set(mode="POSE")
    pose_bones = obj.pose.bones
    actual = [None] * n
    for k in range(len(frames)):
        world_q, world_p = frames[k]
        frame = src_start + k
        scene.frame_set(frame)
        bases = []
        for j in range(n):
            par = parents[j]
            pm = actual[par] if par >= 0 else Matrix.Identity(4)
            pb_rest = rest_4[par] if par >= 0 else Matrix.Identity(4)
            rot_des = ((Cq @ world_q[j]) @ Cq_inv) @ carry[j]
            dst = Matrix.Translation(C @ world_p[j]) @ rot_des.to_matrix().to_4x4()
            basis = rest_4[j].inverted() @ pb_rest @ pm.inverted() @ dst
            bases.append(basis)
            pb = pose_bones[names[j]]
            key_rot = Quaternion(bases[j].to_quaternion())
            prev_rot = pb.rotation_quaternion
            if key_rot.dot(prev_rot) < 0.0:
                key_rot.negate()
            pb.rotation_quaternion = key_rot
            pb.location = bases[j].translation
            pb.keyframe_insert("rotation_quaternion", frame=frame)
            pb.keyframe_insert("location", frame=frame)
            actual[j] = dst
        if k == 0 or k == len(frames) - 1:
            for j, name in enumerate(names):
                pb = pose_bones[name]
                rot_pre = quat_geodesic_deg(pb.rotation_quaternion, bases[j].to_quaternion())
                loc_err = (Vector(pb.location) - bases[j].translation).length
                if loc_err > 1e-4 or rot_pre > 1e-2:
                    raise ValueError(f"scene rig bake drifted at {name}: "
                                     f"pos={loc_err:.6f} m rot={rot_pre:.6f} deg")
    scene.frame_set(src_start)
    bpy.ops.object.mode_set(mode="OBJECT")
    for name, was_selected in prev_selected.items():
        obj2 = bpy.data.objects.get(name)
        if obj2 is not None and obj2 != obj:
            obj2.select_set(was_selected)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = prev_active or obj
    obj.location = (1.2, 0.0, 0.0)
    log(f"[scene] verification rig '{RIG_OBJECT_NAME}' created (offset +X); action "
          f"'{action_name}' baked over {len(frames)} frames")
    return obj


def main(source=None, action=None, out=None, soma_skeleton=None,
         output_fps=OUTPUT_FPS, scene_rig=True):
    scene = bpy.context.scene
    orig_frame = scene.frame_current
    try:
        model_joints = load_model_joint_names()
        source_rig = find_source_rig(source)
        if not (source_rig.animation_data and source_rig.animation_data.action):
            raise ValueError(f"armature '{source_rig.name}' has no active action to convert")
        act = source_rig.animation_data.action
        if action and action != act.name:
            raise ValueError(f"action '{action}' is not the active action '{act.name}' on "
                             f"'{source_rig.name}'; set it active first")

        skeleton_path = soma_skeleton or default_soma_skeleton_path()
        if skeleton_path and not os.path.exists(skeleton_path):
            skeleton_path = None
        if skeleton_path:
            log(f"[skel] skeleton: {skeleton_path}")
        else:
            log("[skel] soma_skeleton.json not found; using the embedded SOMA skeleton")
        skel = load_soma_skeleton(skeleton_path)
        reference_fk(skel)

        problems = euler_self_test() + rest_fk_test(skel)
        if problems:
            for problem in problems:
                log(f"[fail] {problem}")
            raise ValueError("self tests failed; see log")

        mapping, missing = build_mapping(source_rig, skel["names"])
        log(f"[map] retargeting {len(mapping)}/{len(skel['names'])} SOMA joints from "
              f"'{source_rig.name}' action '{act.name}'")
        if missing:
            log(f"[map] unmapped (kept at reference pose): {missing}")
        constant_model_joints = [m for m in missing if m in model_joints]
        if constant_model_joints:
            log(f"[warn] model joints stay constant (unmapped source): {constant_model_joints}")

        skel["src_rest"] = {
            src: source_rig.data.bones[src].matrix_local.to_quaternion().normalized()
            for src in sorted(set(mapping.values()))
        }
        skel["src_rest_hips"] = source_rig.data.bones[HIPS_BONE].head_local.copy()

        yaw_q, theta_deg, residuals = fit_yaw_alignment(source_rig, skel, mapping)
        fit_worst = max((r for _, r in residuals), default=0.0)
        log(f"[fit] rest yaw alignment {theta_deg:.4f} deg over {len(residuals)} chains; "
              f"max residual {fit_worst:.4f} deg")
        if fit_worst > 30.0:
            raise ValueError(f"rest pose mismatch too large ({fit_worst:.2f} deg); "
                             "expected both rigs resting in a T-pose")
        if fit_worst > 5.0:
            log(f"[warn] rest pose residual {fit_worst:.2f} deg > 5 deg; retarget quality may suffer")

        hips_rest_len = source_rig.data.bones[HIPS_BONE].head_local.length
        scale = skel["P"][0].length / hips_rest_len
        log(f"[fit] proportion scale {scale:.6f}")

        missing_model = model_joint_coverage(model_joints, skel["names"])
        if missing_model:
            raise ValueError(f"model joints missing from output skeleton: {missing_model}")

        start = int(round(act.frame_range[0]))
        end = int(round(act.frame_range[1]))
        if end < start:
            raise ValueError(f"action '{act.name}' has an empty frame range")

        rest_problems = rest_retarget_test(skel, mapping, yaw_q, scale)
        if rest_problems:
            for problem in rest_problems:
                log(f"[fail] {problem}")
            raise ValueError("rest retarget test failed; see log")

        src_samples = sample_source_poses(source_rig, mapping, start, end)

        world_rate = []
        hips_rate = []
        for sample in src_samples:
            world_rate.append(retarget_frame(sample, mapping, skel, yaw_q))
            hips_rate.append(hips_world(sample, yaw_q, scale))
        rch_rate = [channels_from_world(w, skel["parents"]) for w in world_rate]

        frames_src_rate = []
        for world, hips, rch in zip(world_rate, hips_rate, rch_rate):
            _, P = fk_world(skel["p_loc"], hips, rch, skel["parents"])
            frames_src_rate.append((world, P))

        duration = (end - start) / scene.render.fps
        out_world, out_hips = resample_to_fps(world_rate, hips_rate, duration, output_fps)
        out_rch = make_quats_sign_continuous(
            [channels_from_world(w, skel["parents"]) for w in out_world])
        out_world_p = []
        for rch, hips in zip(out_rch, out_hips):
            _, P = fk_world(skel["p_loc"], hips, rch, skel["parents"])
            out_world_p.append(P)

        safe_name = re.sub(r"[^0-9A-Za-z]+", "_", act.name).strip("_") or "clip"
        if out:
            out_path = out
        else:
            repo_root = _find_repo_root()
            if repo_root:
                out_dir = os.path.join(repo_root, "converted")
            else:
                out_dir = os.path.join(
                    os.path.dirname(bpy.path.abspath(bpy.data.filepath))
                    if bpy.data.filepath else os.getcwd(), "converted")
            out_path = os.path.join(out_dir, f"{safe_name}__soma.bvh")
        write_bvh(out_path, skel["names"], skel["parents"], skel["p_loc"],
                  out_rch, out_hips, output_fps)
        log(f"[bvh] wrote {out_path}: {len(out_rch)} frames @ {output_fps} Hz "
              f"(source: {end - start + 1} frames @ {scene.render.fps} fps)")

        rt_problems, rot_err, pos_err = roundtrip_test(
            out_path, skel, out_rch, out_hips, out_world_p)
        if rt_problems:
            for problem in rt_problems:
                log(f"[fail] {problem}")
            raise ValueError("BVH round trip test failed; see log")

        if scene_rig:
            create_scene_rig(scene, skel, frames_src_rate, act.name, start)
        log("[done] conversion complete")
        return {
            "bvh": out_path, "frames": len(out_rch),
            "rot_err_deg": rot_err, "pos_err_m": pos_err,
            "yaw_deg": theta_deg, "scale": scale,
        }
    finally:
        scene.frame_set(orig_frame)


if __name__ == "__main__":
    raw_args = sys.argv
    cli_args = raw_args[raw_args.index("--") + 1:] if "--" in raw_args else []
    parser = argparse.ArgumentParser(
        description="Convert the scene's Mixamo action into a MotionBricks SOMA .bvh")
    parser.add_argument("--source", default=None,
                        help="source armature name (default: the animated mixamorig rig)")
    parser.add_argument("--action", default=None,
                        help="action name (default: the rig's active action)")
    parser.add_argument("--out", default=None,
                        help="output .bvh path (default: <blend dir>/converted/<action>__soma.bvh)")
    parser.add_argument("--soma-skeleton", default=None,
                        help="path to soma_skeleton.json")
    parser.add_argument("--fps", default=OUTPUT_FPS, type=float, help="output frame rate")
    parser.add_argument("--no-scene-rig", action="store_true",
                        help="skip creating the verification armature")
    opts = parser.parse_args(cli_args)
    try:
        main(out=opts.out, source=opts.source, action=opts.action,
             soma_skeleton=opts.soma_skeleton, output_fps=opts.fps,
             scene_rig=not opts.no_scene_rig)
    except Exception:
        log(traceback.format_exc())
        write_report()
        raise
    write_report()
