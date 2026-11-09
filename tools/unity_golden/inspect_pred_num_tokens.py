# SPDX-License-Identifier: Apache-2.0
"""Inspect the ONNX subgraph that produces pred_num_tokens."""
import sys

import onnx

path = r"motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx"
model = onnx.load(path, load_external_data=False)
graph = model.graph

# Map output name -> producing node.
producer = {}
for node in graph.node:
    for output in node.output:
        producer[output] = node

target = "pred_num_tokens"
visited = set()
stack = [target]
print(f"Subgraph feeding '{target}':")
while stack:
    name = stack.pop()
    if name in visited:
        continue
    visited.add(name)
    node = producer.get(name)
    if node is None:
        continue
    attrs = ", ".join(
        f"{a.name}={onnx.helper.get_attribute_value(a)}"
        for a in node.attribute
    )
    print(f"  {node.op_type:20s} out={list(node.output)} in={list(node.input)} {attrs}")
    for inp in node.input:
        stack.append(inp)

print()
print("output defs:")
for output in graph.output:
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in output.type.tensor_type.shape.dim]
    print(f"  {output.name}: dtype={output.type.tensor_type.elem_type} shape={dims}")
