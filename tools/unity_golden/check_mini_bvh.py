# SPDX-License-Identifier: Apache-2.0
"""Compare the mini BVH loaded by the reference loader against the fixture samples."""
import json
import sys
from pathlib import Path

import warp as wp

from motionbricks.io.bvh import load_bvh
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
from motionbricks.io.usd import load_skeleton_from_usd

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "unity" / "com.nvlabs.motionbricks" / "Tests" / "Fixtures"

wp.init()
wp.set_device("cuda:0")

skeleton = load_skeleton_from_usd(
    get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH)
mini = load_bvh(str(FIXTURES / "mini_walk.bvh"), skeleton)
doc = json.loads((FIXTURES / "bvh_samples.json").read_text(encoding="utf-8"))

print(f"mini clip: {mini.num_frames} frames at {mini.sample_rate} Hz")

offsets = []
for row in doc["root_offsets"]:
    offsets.append(wp.transform(wp.vec3(*row[:3]), wp.quat(*row[3:])))

pose = wp.empty(skeleton.num_joints, dtype=wp.transform)
worst_pos = 0.0
worst_rot = 0.0
worst_where = ""
for entry in doc["samples"]:
    mini.sample(entry["time"], pose, root_offset=offsets[entry["root_offset"]])
    actual = pose.numpy()
    for joint, expected in enumerate(entry["pose"]):
        dp = ((actual[joint, 0] - expected["px"]) ** 2
              + (actual[joint, 1] - expected["py"]) ** 2
              + (actual[joint, 2] - expected["pz"]) ** 2) ** 0.5
        dq = abs(actual[joint, 3] * expected["qx"] + actual[joint, 4] * expected["qy"]
                 + actual[joint, 5] * expected["qz"] + actual[joint, 6] * expected["qw"])
        dr = 2.0 * (1.0 - min(1.0, dq))
        if dp > worst_pos:
            worst_pos = dp
            worst_where = f"t={entry['time']} off={entry['root_offset']} joint={joint} ({skeleton.joint_names[joint]})"
        worst_rot = max(worst_rot, dr)

print(f"worst position error: {worst_pos * 1000:.4f} mm at {worst_where}")
print(f"worst rotation error: {worst_rot:.6f} (quat delta)")
