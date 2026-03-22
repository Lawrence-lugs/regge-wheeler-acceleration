from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference


def _tensor(name: str, array: np.ndarray, dtype: int = TensorProto.FLOAT) -> onnx.TensorProto:
    return helper.make_tensor(name=name, data_type=dtype, dims=list(array.shape), vals=array.flatten().tolist())


def build_workload(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    x_shape = [1, 128]

    w_fc = rng.standard_normal((128, 128), dtype=np.float32) * 0.05
    b_fc = np.zeros((128,), dtype=np.float32)
    bias_vec = np.full((1, 128), 0.01, dtype=np.float32)
    scale_vec = np.full((1, 128), 0.99, dtype=np.float32)
    constraint_bias = np.full((1, 128), 0.02, dtype=np.float32)
    out_scale = np.ones((1, 128), dtype=np.float32)

    initializers = [
        _tensor("W_fc", w_fc),
        _tensor("B_fc", b_fc),
        _tensor("BiasVec", bias_vec),
        _tensor("ScaleVec", scale_vec),
        _tensor("ConstraintBias", constraint_bias),
        _tensor("OutScale", out_scale),
    ]

    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape)
    output_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, x_shape)

    nodes = [
        helper.make_node("Gemm", ["X", "W_fc", "B_fc"], ["gemm_out"], name="rw_dense_gemm"),
        helper.make_node("Add", ["gemm_out", "BiasVec"], ["act_out"], name="rw_vector_add"),
        helper.make_node("Mul", ["act_out", "ScaleVec"], ["scaled_out"], name="rw_constraint_mul"),
        helper.make_node("Add", ["scaled_out", "ConstraintBias"], ["constraint_out"], name="rw_constraint_add"),
        helper.make_node("Mul", ["constraint_out", "OutScale"], ["Y"], name="rw_output_mul"),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="regge_wheeler_stream_graph",
        inputs=[input_x],
        outputs=[output_y],
        initializer=initializers,
    )

    model = helper.make_model(graph, producer_name="torch_dev_stream_sim")
    model.opset_import[0].version = 19
    inferred = shape_inference.infer_shapes(model)
    onnx.save(inferred, output_path)
    return output_path


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "inputs" / "workload" / "pinn_workload.onnx"
    path = build_workload(target)
    print("Wrote ONNX workload to", path)


if __name__ == "__main__":
    main()
