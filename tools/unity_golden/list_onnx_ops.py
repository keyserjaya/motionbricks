# SPDX-License-Identifier: Apache-2.0
"""List the ONNX operator types used by the SOMA models."""
import sys
from collections import Counter

try:
    import onnx
except ImportError:
    print("onnx not installed")
    sys.exit(1)

for path in (
    r"motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx",
    r"motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx",
):
    model = onnx.load(path, load_external_data=False)
    ops = Counter(node.op_type for node in model.graph.node)
    print(path)
    print("  opset:", [(o.domain or "ai.onnx", o.version) for o in model.opset_import])
    for op, count in sorted(ops.items()):
        print(f"  {op}: {count}")
    print()
