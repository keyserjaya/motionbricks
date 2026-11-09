# SPDX-License-Identifier: Apache-2.0
"""Compare the fixture's BVH sample positions against the mini BVH offsets and USD reference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "unity" / "com.nvlabs.motionbricks" / "Tests" / "Fixtures"
SKELETON = ROOT / "unity" / "com.nvlabs.motionbricks" / "Runtime" / "Skeleton" / "soma_skeleton.json"

doc = json.loads((FIXTURES / "bvh_samples.json").read_text(encoding="utf-8"))
skeleton = json.loads(SKELETON.read_text(encoding="utf-8"))
names = skeleton["joint_names"]
reference = skeleton["reference_local"]

# Parse the mini BVH's offsets per joint.
mini = (FIXTURES / "mini_walk.bvh").read_text(encoding="utf-8").splitlines()
offsets = {}
current = None
for line in mini:
    tokens = line.split()
    if not tokens:
        continue
    if tokens[0] in ("ROOT", "JOINT"):
        name = tokens[1].rsplit(":", 1)[-1]
        current = name
    elif tokens[0] == "OFFSET" and current is not None:
        offsets[current] = [float(v) * 0.01 for v in tokens[1:4]]
    elif tokens[0] == "MOTION":
        break

print("sample 0 (identity offset) joint positions vs expectations:")
sample = doc["samples"][0]["pose"]
for name in ("Hips", "Spine1", "Spine2", "Chest", "Neck1", "LeftArm", "LeftLeg", "RightLeg", "LeftToeBase"):
    idx = names.index(name)
    fx = sample[idx]
    off = offsets.get(name)
    ref = reference[idx]
    print(f"{name:14s} fixture=({fx['px']:+.4f},{fx['py']:+.4f},{fx['pz']:+.4f}) "
          f"bvh_offset={off if off is None else ('(%+.4f,%+.4f,%+.4f)' % tuple(off))} "
          f"usd_ref=({ref['px']:+.4f},{ref['py']:+.4f},{ref['pz']:+.4f})")

print()
print("times:", doc["times"][:5], "...", doc["times"][-3:])
