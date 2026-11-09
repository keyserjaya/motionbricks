# SPDX-License-Identifier: Apache-2.0
"""Print the SOMA reference pose of selected joints, for retargeter checks."""
import json
import sys
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "unity" / "com.nvlabs.motionbricks" / "Runtime" / "Skeleton" / "soma_skeleton.json"
doc = json.loads(path.read_text(encoding="utf-8"))
names = doc["joint_names"]
reference = doc["reference_local"]
wanted = {
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
}
for index, name in enumerate(names):
    if name not in wanted:
        continue
    v = reference[index]
    print("%-16s p=(%+.3f,%+.3f,%+.3f) q=(%+.3f,%+.3f,%+.3f,%+.3f)" % (
        name, v["px"], v["py"], v["pz"], v["qx"], v["qy"], v["qz"], v["qw"]))

# World positions of the rest pose.
parents = doc["parent_indices"]
import math


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qrot(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


world_pos = []
world_rot = []
for index, name in enumerate(names):
    v = reference[index]
    p = (v["px"], v["py"], v["pz"])
    q = (v["qx"], v["qy"], v["qz"], v["qw"])
    parent = parents[index]
    if parent < 0:
        world_pos.append(p)
        world_rot.append(q)
    else:
        pp = world_pos[parent]
        pq = world_rot[parent]
        rp = qrot(pq, p)
        world_pos.append((pp[0] + rp[0], pp[1] + rp[1], pp[2] + rp[2]))
        world_rot.append(qmul(pq, q))

print()
print("world positions of the reference pose:")
for index, name in enumerate(names):
    if name in wanted:
        p = world_pos[index]
        print("%-16s w=(%+.3f,%+.3f,%+.3f)" % (name, p[0], p[1], p[2]))
