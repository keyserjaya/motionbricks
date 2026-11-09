# SPDX-License-Identifier: Apache-2.0
"""Simplify the SOMA ONNX model's token-count path for the Unity Inference Engine.

The exported graph computes pred_num_tokens through an INT64 sentinel chain and a
ScatterND / ConstantOfShape masking chain. InferenceEngine mis-executes both (the
count came out as 198 = 18 * 11 and the masked logits read back as NaN), and since
pred_num_tokens feeds back into the internal decoder, later motion frames were
corrupted too.

This script copies each model to assets/onnx/soma/unity/ with the token-count path
rewired to plain Slice / Equal / Where / ArgMax ops over the raw token-count logits
(Gemm output), and exposes those raw logits as pred_num_token_logits so C# backends
can apply the same argmax rule themselves:

    count = argmax(i in 0..10 where allowed[i]==1 ? logits[i] : -inf) + 6

Column 11 of the raw logits is an auxiliary class that the original graph forces to
-inf; the rule never selects it. The rule is validated below against the golden
fixture (TensorRT reference).

The originals are left untouched. Run from the repository root:

    uv run --with onnx --with onnxruntime python tools/unity_golden/simplify_token_output.py
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = REPO_ROOT / "motionbricks" / "assets" / "onnx" / "soma"
OUT_DIR = MODELS / "unity"
GOLDEN = REPO_ROOT / "unity" / "com.nvlabs.motionbricks" / "Tests" / "Fixtures" / "golden_soma.json"

# Raw token-count logits: a plain Gemm output (12 columns: 11 token counts 6..16
# plus one auxiliary class).
LOGITS = "/mmm_net/_proj_num_token_output_logit/Gemm_output_0"
ALLOWED = "allowed_pred_num_tokens"


def simplify(path: Path, out_path: Path) -> None:
    print(f"processing {path.name}")
    model = onnx.load(str(path))
    graph = model.graph

    # The Gemm output type follows the model's weights (float32 or float16).
    logits_dtype = TensorProto.FLOAT
    for init in graph.initializer:
        if "num_token_output_logit" in init.name:
            logits_dtype = init.data_type
    print(f"  logits dtype: {logits_dtype}")

    def const(name: str, np_dtype, values) -> None:
        # numpy_helper writes raw_data bytes, which is what the InferenceEngine
        # importer reads (helper.make_tensor's typed fields are not handled there).
        graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=np_dtype), name))

    const("mb_zero_i64", np.int64, [0])
    const("mb_eleven_i64", np.int64, [11])
    const("mb_one_i64", np.int64, [1])
    const("mb_six_i32", np.int32, [6])
    const("mb_one_f32", np.float32, [1.0])
    const("mb_neginf_f32", np.float32, [float("-inf")])

    # logits11 = Slice(logits, starts=0, ends=11, axes=1, steps=1)   -> [batch, 11]
    graph.node.append(
        helper.make_node(
            "Slice",
            [LOGITS, "mb_zero_i64", "mb_eleven_i64", "mb_one_i64", "mb_one_i64"],
            ["mb_logits11"],
            name="mb_slice_logits",
        )
    )
    # Cast to float32 so the masking works the same on the fp16 model.
    graph.node.append(
        helper.make_node("Cast", ["mb_logits11"], ["mb_logits11_f32"], name="mb_cast_logits", to=TensorProto.FLOAT)
    )
    # masked = Where(Cast(allowed, float) == 1, logits11, -inf)       -> [batch, 11]
    graph.node.append(helper.make_node("Cast", [ALLOWED], ["mb_allowed_f"], name="mb_cast_allowed", to=TensorProto.FLOAT))
    graph.node.append(helper.make_node("Equal", ["mb_allowed_f", "mb_one_f32"], ["mb_allowed_eq"], name="mb_equal"))
    graph.node.append(
        helper.make_node("Where", ["mb_allowed_eq", "mb_logits11_f32", "mb_neginf_f32"], ["mb_masked"], name="mb_where")
    )
    # pred_num_tokens = ArgMax(masked, axis=-1) + 6                   -> [batch]
    graph.node.append(
        helper.make_node("ArgMax", ["mb_masked"], ["mb_argmax"], name="mb_argmax", axis=-1, keepdims=0)
    )
    graph.node.append(
        helper.make_node("Cast", ["mb_argmax"], ["mb_argmax_i32"], name="mb_cast_argmax", to=TensorProto.INT32)
    )
    graph.node.append(helper.make_node("Add", ["mb_argmax_i32", "mb_six_i32"], ["pred_num_tokens"], name="mb_add"))

    # Rename the old sentinel producer so the clean chain owns the canonical name.
    for node in graph.node:
        if node.name == "mb_add":
            continue
        if "pred_num_tokens" in node.output:
            node.output[list(node.output).index("pred_num_tokens")] = "mb_old_pred_num_tokens"

    # Expose the raw logits for C# backends.
    graph.node.append(helper.make_node("Identity", [LOGITS], ["pred_num_token_logits"], name="mb_logits_identity"))
    graph.output.append(
        helper.make_tensor_value_info("pred_num_token_logits", logits_dtype, ["batch_size", 12])
    )

    topological_sort(graph)
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    print(f"  saved {out_path}")


def topological_sort(graph) -> None:
    """Re-orders graph nodes so every input is produced before it is consumed."""
    producers = {}
    for node in graph.node:
        for output in node.output:
            producers[output] = node

    sorted_nodes = []
    visited = set()
    in_progress = set()

    def visit(node) -> None:
        key = id(node)
        if key in visited:
            return
        if key in in_progress:
            raise RuntimeError(f"cycle at node {node.name}")
        in_progress.add(key)
        for input_name in node.input:
            producer = producers.get(input_name)
            if producer is not None and producer is not node:
                visit(producer)
        in_progress.discard(key)
        visited.add(key)
        sorted_nodes.append(node)

    for node in list(graph.node):
        visit(node)

    del graph.node[:]
    graph.node.extend(sorted_nodes)


def cs_argmax_rule(logits: np.ndarray, allowed: np.ndarray) -> int:
    """The argmax rule the C# backend applies to pred_num_token_logits."""
    best_index = 0
    best_score = float("-inf")
    for column in range(11):
        if not allowed[column]:
            continue
        score = float(logits[column])
        if score > best_score:
            best_score = score
            best_index = column
    return best_index + 6


def validate(original_path: Path, rewired_path: Path, golden: dict, strict: bool) -> None:
    import onnxruntime as ort

    sess_new = ort.InferenceSession(str(rewired_path), providers=["CPUExecutionProvider"])
    sess_old = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    names_new = [o.name for o in sess_new.get_outputs()]
    names_old = [o.name for o in sess_old.get_outputs()]
    print(f"  outputs: {names_new}")

    for gen in golden["generations"]:
        packed = gen["packed"]
        feeds = {
            "global_root_values": np.asarray(packed["global_root_values"], dtype=np.float32)[np.newaxis],
            "local_root_values": np.asarray(packed["local_root_values"], dtype=np.float32)[np.newaxis],
            "local_poses": np.asarray(packed["local_poses"], dtype=np.float32)[np.newaxis],
            "has_global_root_values": np.asarray(packed["has_global_root_values"], dtype=np.int32)[np.newaxis],
            "has_local_root_values": np.asarray(packed["has_local_root_values"], dtype=np.int32)[np.newaxis],
            "has_local_poses": np.asarray(packed["has_local_poses"], dtype=np.int32)[np.newaxis],
            "allowed_pred_num_tokens": np.asarray(packed["allowed_pred_num_tokens"], dtype=np.int32)[np.newaxis],
        }
        by_new = dict(zip(names_new, sess_new.run(None, feeds)))
        by_old = dict(zip(names_old, sess_old.run(None, feeds)))

        tokens_new = int(np.asarray(by_new["pred_num_tokens"]).reshape(-1)[0])
        tokens_old = int(np.asarray(by_old["pred_num_tokens"]).reshape(-1)[0])
        logits = np.asarray(by_new["pred_num_token_logits"][0])
        cs_rule = cs_argmax_rule(logits, np.asarray(packed["allowed_pred_num_tokens"], dtype=np.int32))
        expected_tokens = int(gen["pred_num_tokens"])
        print(f"  {gen['name']}: tokens rewired={tokens_new} original={tokens_old} "
              f"golden(TRT)={expected_tokens} cs_rule={cs_rule}")
        print(f"    logits={np.array2string(logits, precision=3)}")

        motions_new = np.asarray(by_new["pred_global_motions"])
        motions_old = np.asarray(by_old["pred_global_motions"])
        worst_same_backend = float(np.max(np.abs(motions_new - motions_old)))
        expected_motions = np.asarray(gen["pred_global_motions"], dtype=np.float32)
        worst_vs_golden = float(np.max(np.abs(motions_new[0] - expected_motions)))
        print(f"    motions rewired-vs-original (ORT CPU): {worst_same_backend:.6f}")
        print(f"    motions rewired-vs-golden (TensorRT):  {worst_vs_golden:.6f}")

        if strict:
            assert tokens_new == expected_tokens, f"{gen['name']}: token mismatch"
            assert tokens_new == tokens_old, f"{gen['name']}: rewire changed the token count"
            assert cs_rule == expected_tokens, f"{gen['name']}: C# argmax rule mismatch"
            assert worst_same_backend < 1e-4, f"{gen['name']}: rewire changed motions {worst_same_backend}"


def main() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for name, strict in (("mmm_v6_soma_fp32.onnx", True), ("mmm_v6_soma_fp16.onnx", False)):
        path = MODELS / name
        out_path = OUT_DIR / name
        simplify(path, out_path)
        validate(path, out_path, golden, strict)
    print("done")


if __name__ == "__main__":
    main()
