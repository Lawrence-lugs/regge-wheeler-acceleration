from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference

from stream_sim.config import WORKLOAD_CONFIG


class OnnxGraphBuilder:
    def __init__(self) -> None:
        self.initializers: list[onnx.TensorProto] = []
        self.nodes: list[onnx.NodeProto] = []
        self._counter = 0

    def _unique(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def tensor(self, name: str, array: np.ndarray, dtype: int = TensorProto.FLOAT) -> str:
        self.initializers.append(
            helper.make_tensor(
                name=name,
                data_type=dtype,
                dims=list(array.shape),
                vals=array.flatten().tolist(),
            )
        )
        return name

    def scalar(self, prefix: str, value: float) -> str:
        return self.tensor(self._unique(prefix), np.array([value], dtype=np.float32))

    def int64_tensor(self, prefix: str, values: list[int]) -> str:
        return self.tensor(self._unique(prefix), np.array(values, dtype=np.int64), dtype=TensorProto.INT64)

    def op(self, op_type: str, inputs: list[str], prefix: str, **attrs: int | float | list[int]) -> str:
        output = self._unique(f"{prefix}_out")
        node_name = self._unique(prefix)
        self.nodes.append(helper.make_node(op_type, inputs, [output], name=node_name, **attrs))
        return output


def _mul(builder: OnnxGraphBuilder, lhs: str, rhs: str, prefix: str) -> str:
    return builder.op("Mul", [lhs, rhs], prefix)


def _add(builder: OnnxGraphBuilder, lhs: str, rhs: str, prefix: str) -> str:
    return builder.op("Add", [lhs, rhs], prefix)


def _negate(builder: OnnxGraphBuilder, tensor: str, prefix: str) -> str:
    return _mul(builder, tensor, builder.scalar(f"{prefix}_neg_one", -1.0), prefix)


def _add_scalar(builder: OnnxGraphBuilder, tensor: str, value: float, prefix: str) -> str:
    return _add(builder, tensor, builder.scalar(f"{prefix}_scalar", value), prefix)


def _mul_scalar(builder: OnnxGraphBuilder, tensor: str, value: float, prefix: str) -> str:
    return _mul(builder, tensor, builder.scalar(f"{prefix}_scalar", value), prefix)


def _pow(builder: OnnxGraphBuilder, tensor: str, exponent: float, prefix: str) -> str:
    return builder.op("Pow", [tensor, builder.scalar(f"{prefix}_exp", exponent)], prefix)


def _matmul(builder: OnnxGraphBuilder, lhs: str, rhs: str, prefix: str) -> str:
    return builder.op("MatMul", [lhs, rhs], prefix)


def _reduce_mean(builder: OnnxGraphBuilder, tensor: str, prefix: str, keepdims: int = 1) -> str:
    return builder.op("ReduceMean", [tensor], prefix, axes=[-1], keepdims=keepdims)


def _sigmoid(builder: OnnxGraphBuilder, tensor: str, prefix: str) -> str:
    return builder.op("Sigmoid", [tensor], prefix)


def _exp(builder: OnnxGraphBuilder, tensor: str, prefix: str) -> str:
    return builder.op("Exp", [tensor], prefix)


def _reciprocal(builder: OnnxGraphBuilder, tensor: str, prefix: str) -> str:
    return builder.op("Reciprocal", [tensor], prefix)


def _log_series(builder: OnnxGraphBuilder, tensor: str, prefix: str) -> str:
    tensor_sq = _pow(builder, tensor, 2.0, f"{prefix}_sq")
    tensor_cu = _mul(builder, tensor_sq, tensor, f"{prefix}_cu")
    tensor_qu = _mul(builder, tensor_sq, tensor_sq, f"{prefix}_qu")
    term2 = _mul_scalar(builder, tensor_sq, -0.5, f"{prefix}_term2")
    term3 = _mul_scalar(builder, tensor_cu, 1.0 / 3.0, f"{prefix}_term3")
    term4 = _mul_scalar(builder, tensor_qu, -0.25, f"{prefix}_term4")
    acc = _add(builder, tensor, term2, f"{prefix}_acc1")
    acc = _add(builder, acc, term3, f"{prefix}_acc2")
    return _add(builder, acc, term4, f"{prefix}_acc3")


def _range_reduce_surrogate(builder: OnnxGraphBuilder, theta: str, prefix: str) -> str:
    scaled = _mul_scalar(builder, theta, 1.0 / (2.0 * np.pi), f"{prefix}_scale")
    shifted = _add_scalar(builder, scaled, 0.5, f"{prefix}_shift")
    centered = _add_scalar(builder, shifted, -0.5, f"{prefix}_center")
    return _mul_scalar(builder, centered, 2.0 * np.pi, f"{prefix}_restore")


def _sin_poly(builder: OnnxGraphBuilder, theta: str, prefix: str) -> str:
    theta_sq = _pow(builder, theta, 2.0, f"{prefix}_sq")
    theta_cu = _mul(builder, theta, theta_sq, f"{prefix}_cu")
    theta_5 = _mul(builder, theta_cu, theta_sq, f"{prefix}_pow5")
    theta_7 = _mul(builder, theta_5, theta_sq, f"{prefix}_pow7")
    term3 = _mul_scalar(builder, theta_cu, -1.0 / 6.0, f"{prefix}_term3")
    term5 = _mul_scalar(builder, theta_5, 1.0 / 120.0, f"{prefix}_term5")
    term7 = _mul_scalar(builder, theta_7, -1.0 / 5040.0, f"{prefix}_term7")
    acc = _add(builder, theta, term3, f"{prefix}_acc1")
    acc = _add(builder, acc, term5, f"{prefix}_acc2")
    return _add(builder, acc, term7, f"{prefix}_acc3")


def _cos_poly(builder: OnnxGraphBuilder, theta: str, prefix: str) -> str:
    theta_sq = _pow(builder, theta, 2.0, f"{prefix}_sq")
    theta_4 = _mul(builder, theta_sq, theta_sq, f"{prefix}_pow4")
    theta_6 = _mul(builder, theta_4, theta_sq, f"{prefix}_pow6")
    term2 = _mul_scalar(builder, theta_sq, -0.5, f"{prefix}_term2")
    term4 = _mul_scalar(builder, theta_4, 1.0 / 24.0, f"{prefix}_term4")
    term6 = _mul_scalar(builder, theta_6, -1.0 / 720.0, f"{prefix}_term6")
    acc = _add_scalar(builder, term2, 1.0, f"{prefix}_acc1")
    acc = _add(builder, acc, term4, f"{prefix}_acc2")
    return _add(builder, acc, term6, f"{prefix}_acc3")


def _tanh_via_sigmoid(builder: OnnxGraphBuilder, tensor: str, prefix: str) -> str:
    doubled = _mul_scalar(builder, tensor, 2.0, f"{prefix}_double")
    sig = _sigmoid(builder, doubled, f"{prefix}_sigmoid")
    two_sig = _mul_scalar(builder, sig, 2.0, f"{prefix}_rescale")
    return _add_scalar(builder, two_sig, -1.0, f"{prefix}_shift")


def _gaussian(builder: OnnxGraphBuilder, tensor: str, prefix: str) -> str:
    shifted = _add_scalar(builder, tensor, -10.0, f"{prefix}_shift")
    squared = _pow(builder, shifted, 2.0, f"{prefix}_sq")
    scaled = _mul_scalar(builder, squared, -0.25, f"{prefix}_scale")
    return _exp(builder, scaled, f"{prefix}_exp")


def _tortoise_potential(builder: OnnxGraphBuilder, r_star: str, prefix: str) -> str:
    cfg = WORKLOAD_CONFIG
    r = _add_scalar(builder, r_star, 2.0 * cfg.mass + 1.0, f"{prefix}_warmstart")
    two_m = 2.0 * cfg.mass

    for step in range(cfg.lambert_newton_steps):
        step_prefix = f"{prefix}_newton_{step}"
        scaled = _mul_scalar(builder, r, 1.0 / two_m, f"{step_prefix}_scale")
        shifted = _add_scalar(builder, scaled, -1.0, f"{step_prefix}_shift")
        log_term = _log_series(builder, shifted, f"{step_prefix}_log")
        log_scaled = _mul_scalar(builder, log_term, two_m, f"{step_prefix}_logscale")
        residual = _add(builder, r, log_scaled, f"{step_prefix}_sum")
        residual = _add(builder, residual, _negate(builder, r_star, f"{step_prefix}_resneg"), f"{step_prefix}_res")
        inv_r = _reciprocal(builder, r, f"{step_prefix}_inv_r")
        slope_correction = _mul_scalar(builder, inv_r, -two_m, f"{step_prefix}_slopecorr")
        slope = _add_scalar(builder, slope_correction, 1.0, f"{step_prefix}_slope")
        inv_slope = _reciprocal(builder, slope, f"{step_prefix}_inv_slope")
        correction = _mul(builder, residual, inv_slope, f"{step_prefix}_correction")
        r = _add(builder, r, _negate(builder, correction, f"{step_prefix}_corrneg"), f"{step_prefix}_update")

    inv_r = _reciprocal(builder, r, f"{prefix}_inv_r")
    inv_r_sq = _pow(builder, inv_r, 2.0, f"{prefix}_inv_r_sq")
    inv_r_cu = _mul(builder, inv_r_sq, inv_r, f"{prefix}_inv_r_cu")
    frac = _mul_scalar(builder, inv_r, two_m, f"{prefix}_frac")
    term1 = _add_scalar(builder, _negate(builder, frac, f"{prefix}_fracneg"), 1.0, f"{prefix}_term1")
    spin_factor = (1.0 - cfg.spin_s**2) * 2.0 * cfg.mass
    correction = _mul_scalar(builder, inv_r_cu, spin_factor, f"{prefix}_corr")
    l_term = cfg.angular_l * (cfg.angular_l + 1) / 2.0
    term2 = _add_scalar(builder, correction, l_term, f"{prefix}_term2")
    return _mul(builder, term1, term2, f"{prefix}_out")


def build_workload(output_path: Path) -> Path:
    cfg = WORKLOAD_CONFIG
    output_path.parent.mkdir(parents=True, exist_ok=True)

    builder = OnnxGraphBuilder()

    fd_rstar_shape = [1, 1, 1, cfg.nx]
    fd_state_shape = [1, cfg.nt, 1, cfg.nx]
    pinn_scalar_shape = [cfg.pinn_epochs, cfg.pinn_collocation_points, 1, 1]
    fft_signal_shape = [1, 1, 1, cfg.nt]
    fft_shape = [1, 1, 1, cfg.fft_bins]

    fd_rstar = helper.make_tensor_value_info("fd_rstar", TensorProto.FLOAT, fd_rstar_shape)
    fd_state = helper.make_tensor_value_info("fd_state_seed", TensorProto.FLOAT, fd_state_shape)
    pinn_t = helper.make_tensor_value_info("pinn_t", TensorProto.FLOAT, pinn_scalar_shape)
    pinn_x = helper.make_tensor_value_info("pinn_x", TensorProto.FLOAT, pinn_scalar_shape)
    fft_signal = helper.make_tensor_value_info("fft_signal", TensorProto.FLOAT, fft_signal_shape)

    fourier_scale_t = builder.tensor(
        "fourier_scale_t",
        np.ones((1, 1, 1, cfg.fourier_half), dtype=np.float32) * 0.5,
    )
    fourier_scale_x = builder.tensor(
        "fourier_scale_x",
        np.ones((1, 1, 1, cfg.fourier_half), dtype=np.float32) * 0.5,
    )
    hidden_w1_sin = builder.tensor("hidden_w1_sin", np.ones((cfg.fourier_half, cfg.hidden_width), dtype=np.float32))
    hidden_w1_cos = builder.tensor("hidden_w1_cos", np.ones((cfg.fourier_half, cfg.hidden_width), dtype=np.float32))
    hidden_b1 = builder.tensor("hidden_b1", np.zeros((1, 1, 1, cfg.hidden_width), dtype=np.float32))
    hidden_w2 = builder.tensor("hidden_w2", np.ones((cfg.hidden_width, cfg.hidden_width), dtype=np.float32))
    hidden_b2 = builder.tensor("hidden_b2", np.zeros((1, 1, 1, cfg.hidden_width), dtype=np.float32))
    output_w = builder.tensor("output_w", np.ones((cfg.hidden_width, cfg.output_width), dtype=np.float32))
    output_b = builder.tensor("output_b", np.zeros((1, 1, 1, cfg.output_width), dtype=np.float32))
    output_w_t = builder.tensor("output_w_t", np.ones((cfg.output_width, cfg.hidden_width), dtype=np.float32))
    hidden_w2_t = builder.tensor("hidden_w2_t", np.ones((cfg.hidden_width, cfg.hidden_width), dtype=np.float32))
    hidden_w1_t = builder.tensor("hidden_w1_t", np.ones((cfg.hidden_width, cfg.fourier_half), dtype=np.float32))
    freq_index = np.arange(1, cfg.fft_bins + 1, dtype=np.float32)
    time_index = np.arange(cfg.nt, dtype=np.float32)
    dft_real = np.cos(2.0 * np.pi * np.outer(time_index, freq_index) / cfg.nt).astype(np.float32)
    dft_imag = np.sin(2.0 * np.pi * np.outer(time_index, freq_index) / cfg.nt).astype(np.float32)
    dft_real = dft_real.reshape(1, 1, cfg.nt, cfg.fft_bins)
    dft_imag = dft_imag.reshape(1, 1, cfg.nt, cfg.fft_bins)
    dft_real_name = builder.tensor("dft_real", dft_real)
    dft_imag_name = builder.tensor("dft_imag", dft_imag)
    omega_sq = ((2.0 * np.pi * freq_index / max(cfg.t_max, 1e-6)) ** 2).reshape(1, 1, 1, cfg.fft_bins).astype(
        np.float32
    )
    omega_sq_name = builder.tensor(
        "omega_sq",
        omega_sq,
    )

    fd_potential = _tortoise_potential(builder, "fd_rstar", "fd_potential")
    fd_pulse = _gaussian(builder, "fd_rstar", "fd_pulse")
    fd_weighted = _mul(builder, "fd_state_seed", fd_potential, "fd_weighted")
    fd_drive = _add(builder, fd_weighted, fd_pulse, "fd_drive")
    fd_drive_sq = _pow(builder, fd_drive, 2.0, "fd_drive_sq")
    fd_damped = _mul_scalar(builder, fd_drive_sq, cfg.t_max / max(cfg.nt, 1), "fd_damped")
    fd_wave = _add(builder, "fd_state_seed", fd_damped, "fd_wave")
    fft_real = _matmul(builder, "fft_signal", dft_real_name, "fft_real")
    fft_imag = _matmul(builder, "fft_signal", dft_imag_name, "fft_imag")
    fft_real_sq = _pow(builder, fft_real, 2.0, "fft_real_sq")
    fft_imag_sq = _pow(builder, fft_imag, 2.0, "fft_imag_sq")
    fft_mag = _add(builder, fft_real_sq, fft_imag_sq, "fft_mag")
    omega_inv = _reciprocal(builder, omega_sq_name, "omega_inv")
    fft_strain = _mul(builder, fft_mag, omega_inv, "fft_strain")

    t_tensor = "pinn_t"
    x_tensor = "pinn_x"
    t_norm = _mul_scalar(builder, t_tensor, 1.0 / max(cfg.t_max, 1e-6), "pinn_t_norm")
    x_shift = _add_scalar(builder, x_tensor, -cfg.r_star_min, "pinn_x_shift")
    x_norm = _mul_scalar(
        builder,
        x_shift,
        1.0 / max(cfg.r_star_max - cfg.r_star_min, 1e-6),
        "pinn_x_norm",
    )
    theta_t = _mul(builder, t_norm, fourier_scale_t, "pinn_fourier_t")
    theta_x = _mul(builder, x_norm, fourier_scale_x, "pinn_fourier_x")
    theta = _add(builder, theta_t, theta_x, "pinn_fourier_sum")
    theta_fold = _range_reduce_surrogate(builder, theta, "pinn_theta_fold")
    sin_feat = _sin_poly(builder, theta_fold, "pinn_sin_poly")
    cos_feat = _cos_poly(builder, theta_fold, "pinn_cos_poly")

    h1_sin = _matmul(builder, sin_feat, hidden_w1_sin, "pinn_dense1_sin")
    h1_cos = _matmul(builder, cos_feat, hidden_w1_cos, "pinn_dense1_cos")
    embedding = _add(builder, h1_sin, h1_cos, "pinn_embedding_sum")
    h1 = _add(builder, embedding, hidden_b1, "pinn_dense1_bias")
    h1 = _tanh_via_sigmoid(builder, h1, "pinn_tanh1")
    h2 = _matmul(builder, h1, hidden_w2, "pinn_dense2")
    h2 = _add(builder, h2, hidden_b2, "pinn_dense2_bias")
    h2 = _tanh_via_sigmoid(builder, h2, "pinn_tanh2")
    out = _matmul(builder, h2, output_w, "pinn_dense_out")
    out = _add(builder, out, output_b, "pinn_dense_out_bias")

    psi_ic = _gaussian(builder, x_tensor, "pinn_psi_ic")
    one_minus_x = _add_scalar(builder, _negate(builder, x_norm, "pinn_x_neg"), 1.0, "pinn_one_minus_x")
    spatial_bound = _mul(builder, x_norm, one_minus_x, "pinn_spatial_bound")
    t_sq = _pow(builder, t_norm, 2.0, "pinn_t_sq")
    correction = _mul(builder, t_sq, spatial_bound, "pinn_correction_1")
    correction = _mul(builder, correction, out, "pinn_correction_2")
    psi = _add(builder, psi_ic, correction, "pinn_psi")

    pinn_potential = _tortoise_potential(builder, x_tensor, "pinn_potential")
    source = _mul(builder, pinn_potential, psi, "pinn_source")
    psi_sq = _pow(builder, psi, 2.0, "pinn_psi_sq")
    residual = _add(builder, psi_sq, _negate(builder, source, "pinn_source_neg"), "pinn_residual")
    residual_sq = _pow(builder, residual, 2.0, "pinn_residual_sq")
    pinn_loss = _reduce_mean(builder, residual_sq, "pinn_loss", keepdims=1)

    grad_h2 = _matmul(builder, pinn_loss, output_w_t, "pinn_backprop_out")
    h2_sq = _pow(builder, h2, 2.0, "pinn_h2_sq")
    h2_grad = _add_scalar(builder, _negate(builder, h2_sq, "pinn_h2_sq_neg"), 1.0, "pinn_h2_grad")
    grad_h2 = _mul(builder, grad_h2, h2_grad, "pinn_grad_h2")
    grad_h1 = _matmul(builder, grad_h2, hidden_w2_t, "pinn_backprop_h2")
    h1_sq = _pow(builder, h1, 2.0, "pinn_h1_sq")
    h1_grad = _add_scalar(builder, _negate(builder, h1_sq, "pinn_h1_sq_neg"), 1.0, "pinn_h1_grad")
    grad_h1 = _mul(builder, grad_h1, h1_grad, "pinn_grad_h1")
    grad_emb = _matmul(builder, grad_h1, hidden_w1_t, "pinn_backprop_h1")

    grad_w1_sin = _matmul(builder, sin_feat, hidden_w1_sin, "grad_w1_sin")
    grad_w1_cos = _matmul(builder, cos_feat, hidden_w1_cos, "grad_w1_cos")
    grad_w2 = _matmul(builder, h1, hidden_w2, "grad_w2")
    grad_w3 = _matmul(builder, h2, output_w, "grad_w3")
    grad_fourier_t = _mul(builder, t_norm, grad_emb, "grad_fourier_t")
    grad_fourier_x = _mul(builder, x_norm, grad_emb, "grad_fourier_x")
    _add(builder, _mul_scalar(builder, grad_w1_sin, 0.9, "adam_w1_sin_scale"), builder.scalar("adam_eps_w1_sin", 1e-3), "adam_w1_sin")
    _add(builder, _mul_scalar(builder, grad_w1_cos, 0.9, "adam_w1_cos_scale"), builder.scalar("adam_eps_w1_cos", 1e-3), "adam_w1_cos")
    _add(builder, _mul_scalar(builder, grad_w2, 0.9, "adam_w2_scale"), builder.scalar("adam_eps_w2", 1e-3), "adam_w2")
    _add(builder, _mul_scalar(builder, grad_w3, 0.9, "adam_w3_scale"), builder.scalar("adam_eps_w3", 1e-3), "adam_w3")
    _add(
        builder,
        _mul_scalar(builder, grad_fourier_t, 0.9, "adam_fourier_t_scale"),
        builder.scalar("adam_eps_fourier", 1e-3),
        "adam_fourier_t",
    )
    _add(
        builder,
        _mul_scalar(builder, grad_fourier_x, 0.9, "adam_fourier_x_scale"),
        builder.scalar("adam_eps_fourier_x", 1e-3),
        "adam_fourier_x",
    )

    outputs = [
        helper.make_tensor_value_info(fd_wave, TensorProto.FLOAT, fd_state_shape),
        helper.make_tensor_value_info(pinn_loss, TensorProto.FLOAT, [cfg.pinn_epochs, cfg.pinn_collocation_points, 1, 1]),
        helper.make_tensor_value_info(fft_strain, TensorProto.FLOAT, fft_shape),
    ]

    graph = helper.make_graph(
        nodes=builder.nodes,
        name="regge_wheeler_v2_stream_surrogate",
        inputs=[fd_rstar, fd_state, pinn_t, pinn_x, fft_signal],
        outputs=outputs,
        initializer=builder.initializers,
    )

    model = helper.make_model(graph, producer_name="torch_dev_stream_sim")
    model.opset_import[0].version = 19
    inferred = shape_inference.infer_shapes(model)
    onnx.save(inferred, output_path)
    return output_path


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "inputs" / "workload" / "regge_wheeler_v2_surrogate.onnx"
    path = build_workload(target)
    print("Wrote ONNX workload to", path)


if __name__ == "__main__":
    main()
