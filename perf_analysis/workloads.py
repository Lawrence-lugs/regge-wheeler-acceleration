from __future__ import annotations

import numpy as np

from .config import AnalysisConfig
from .primitives import PrimitiveExecutor


def _approx_tortoise_to_r(
    executor: PrimitiveExecutor,
    r_star: np.ndarray,
    *,
    section: str,
    repetitions: int = 1,
) -> np.ndarray:
    cfg = executor.config
    two_m = np.float32(2.0 * cfg.mass)
    r = np.maximum(r_star + two_m, two_m + np.float32(1e-3)).astype(np.float32)

    for iteration in range(cfg.lambert_newton_steps):
        iter_section = f"{section}.newton_{iteration}"
        ratio = executor.binary("div", r, two_m, section=iter_section, repetitions=repetitions)
        shifted = executor.binary("sub", ratio, 1.0, section=iter_section, repetitions=repetitions)
        log_term = executor.unary("log", shifted, section=iter_section, repetitions=repetitions)
        scaled_log = executor.binary("multiply", log_term, two_m, section=iter_section, repetitions=repetitions)
        f_val = executor.binary("add", r, scaled_log, section=iter_section, repetitions=repetitions)
        f_val = executor.binary("sub", f_val, r_star, section=iter_section, repetitions=repetitions)
        inv_r = executor.binary("div", two_m, r, section=iter_section, repetitions=repetitions)
        deriv_recip = executor.binary("sub", 1.0, inv_r, section=iter_section, repetitions=repetitions)
        correction = executor.binary(
            "multiply", f_val, deriv_recip, section=iter_section, repetitions=repetitions
        )
        r = executor.binary("sub", r, correction, section=iter_section, repetitions=repetitions)
    return r


def compute_regge_wheeler_potential(
    executor: PrimitiveExecutor,
    r_star: np.ndarray,
    *,
    section: str,
    repetitions: int = 1,
) -> np.ndarray:
    cfg = executor.config
    two_m = np.float32(2.0 * cfg.mass)
    l_term = np.float32(cfg.angular_l * (cfg.angular_l + 1) / 2.0)
    spin_factor = np.float32((1 - cfg.spin_s**2) * 2.0 * cfg.mass)

    r = _approx_tortoise_to_r(executor, r_star, section=f"{section}.radius", repetitions=repetitions)
    inv_r = executor.binary("div", 1.0, r, section=section, repetitions=repetitions)
    frac = executor.binary("multiply", two_m, inv_r, section=section, repetitions=repetitions)
    term1 = executor.binary("sub", 1.0, frac, section=section, repetitions=repetitions)
    inv_r2 = executor.binary("multiply", inv_r, inv_r, section=section, repetitions=repetitions)
    inv_r3 = executor.binary("multiply", inv_r2, inv_r, section=section, repetitions=repetitions)
    correction = executor.binary("multiply", spin_factor, inv_r3, section=section, repetitions=repetitions)
    term2 = executor.binary("add", l_term, correction, section=section, repetitions=repetitions)
    return executor.binary("multiply", term1, term2, section=section, repetitions=repetitions)


def _gaussian_pulse(
    executor: PrimitiveExecutor,
    x: np.ndarray,
    *,
    section: str,
    repetitions: int = 1,
) -> np.ndarray:
    shifted = executor.binary("sub", x, 10.0, section=section, repetitions=repetitions)
    squared = executor.binary("multiply", shifted, shifted, section=section, repetitions=repetitions)
    scaled = executor.binary("multiply", squared, -0.25, section=section, repetitions=repetitions)
    return executor.unary("exp", scaled, section=section, repetitions=repetitions)


def simulate_fd_solver(executor: PrimitiveExecutor) -> None:
    cfg = executor.config
    dx = np.float32((cfg.r_star_max - cfg.r_star_min) / (cfg.nx - 1))
    dt = np.float32(cfg.t_max / (cfg.nt - 1))
    x = np.linspace(cfg.r_star_min, cfg.r_star_max, cfg.nx, dtype=np.float32)
    interior = cfg.nx - 2

    potential = compute_regge_wheeler_potential(executor, x, section="fd.potential")
    psi0 = _gaussian_pulse(executor, x, section="fd.initial_condition")
    center = psi0[1:-1]
    left = psi0[:-2]
    right = psi0[2:]
    v_center = potential[1:-1]

    twice_center = executor.binary("multiply", center, 2.0, section="fd.first_step")
    laplacian = executor.binary("sub", right, twice_center, section="fd.first_step")
    laplacian = executor.binary("add", laplacian, left, section="fd.first_step")
    laplacian = executor.binary(
        "multiply", laplacian, np.float32(0.5 * (dt * dt) / (dx * dx)), section="fd.first_step"
    )
    potential_term = executor.binary("multiply", v_center, center, section="fd.first_step")
    potential_term = executor.binary(
        "multiply", potential_term, np.float32(0.5 * dt * dt), section="fd.first_step"
    )
    next_step = executor.binary("add", center, laplacian, section="fd.first_step")
    executor.binary("sub", next_step, potential_term, section="fd.first_step")

    representative_prev = center.copy()
    representative_curr = center.copy()
    rec_section = "fd.recurrent_update"
    twice_curr = executor.binary(
        "multiply", representative_curr, 2.0, section=rec_section, repetitions=cfg.nt - 2
    )
    d2psi = executor.binary("sub", right, twice_curr, section=rec_section, repetitions=cfg.nt - 2)
    d2psi = executor.binary("add", d2psi, left, section=rec_section, repetitions=cfg.nt - 2)
    d2psi = executor.binary(
        "multiply", d2psi, np.float32(1.0 / (dx * dx)), section=rec_section, repetitions=cfg.nt - 2
    )
    source = executor.binary(
        "multiply", v_center, representative_curr, section=rec_section, repetitions=cfg.nt - 2
    )
    rhs = executor.binary("sub", d2psi, source, section=rec_section, repetitions=cfg.nt - 2)
    rhs = executor.binary(
        "multiply", rhs, np.float32(dt * dt), section=rec_section, repetitions=cfg.nt - 2
    )
    next_state = executor.binary("sub", twice_curr, representative_prev, section=rec_section, repetitions=cfg.nt - 2)
    executor.binary("add", next_state, rhs, section=rec_section, repetitions=cfg.nt - 2)

    obs_delta = executor.binary("sub", x, cfg.observer_r_star, section="post.observer_index")
    obs_abs = executor.unary("abs", obs_delta, section="post.observer_index")
    executor.reduce_sum(obs_abs.reshape(1, -1), section="post.observer_index")

    if interior <= 0:
        raise ValueError("nx must be greater than 2.")


def _simulate_network_forward(
    executor: PrimitiveExecutor,
    *,
    batch_size: int,
    section: str,
    repetitions: int,
) -> None:
    cfg = executor.config
    batch = np.ones((batch_size, 1), dtype=np.float32)
    t = np.linspace(0.0, cfg.t_max, batch_size, dtype=np.float32).reshape(-1, 1)
    x = np.linspace(cfg.r_star_min, cfg.r_star_max, batch_size, dtype=np.float32).reshape(-1, 1)

    # Input normalization
    t_norm = executor.binary("div", t, np.float32(cfg.t_max), section=f"{section}.normalize", repetitions=repetitions)
    x_shift = executor.binary("sub", x, np.float32(cfg.r_star_min), section=f"{section}.normalize", repetitions=repetitions)
    x_norm = executor.binary(
        "div",
        x_shift,
        np.float32(cfg.r_star_max - cfg.r_star_min),
        section=f"{section}.normalize",
        repetitions=repetitions,
    )

    # Fourier embedding
    embed_half = cfg.embedding_features // 2
    inputs = np.concatenate([t_norm, x_norm], axis=1)
    fourier_weights = np.ones((2, embed_half), dtype=np.float32) # fourier weights 1 for now
    projection = executor.matmul(inputs, fourier_weights, section=f"{section}.embedding", repetitions=repetitions)
    projection = executor.binary(
        "multiply", projection, np.float32(2.0 * np.pi), section=f"{section}.embedding", repetitions=repetitions
    )
    sin_proj = executor.unary("sin", projection, section=f"{section}.embedding", repetitions=repetitions)
    cos_proj = executor.unary("cos", projection, section=f"{section}.embedding", repetitions=repetitions)
    embedding = np.concatenate([sin_proj, cos_proj], axis=1)

    # Execute input-to-hidden layer
    hidden_weights = np.ones((cfg.embedding_features, cfg.hidden_width), dtype=np.float32)
    hidden = executor.matmul(embedding, hidden_weights, section=f"{section}.dense1", repetitions=repetitions)
    hidden = executor.unary("tanh", hidden, section=f"{section}.dense1", repetitions=repetitions)

    # Execute hidden-to-hidden layers
    recurrent_weights = np.ones((cfg.hidden_width, cfg.hidden_width), dtype=np.float32)
    for layer_idx in range(cfg.hidden_layers - 1):
        hidden = executor.matmul(
            hidden,
            recurrent_weights,
            section=f"{section}.dense_hidden_{layer_idx + 2}",
            repetitions=repetitions,
        )
        hidden = executor.unary(
            "tanh",
            hidden,
            section=f"{section}.dense_hidden_{layer_idx + 2}",
            repetitions=repetitions,
        )

    # Execute hidden-to-output layer
    output_weights = np.ones((cfg.hidden_width, cfg.output_width), dtype=np.float32)
    nn_out = executor.matmul(hidden, output_weights, section=f"{section}.dense_out", repetitions=repetitions)

    # Construct hard-coded initial condition + boundary condition satisfying ansatz and add to network output
    psi_ic = _gaussian_pulse(executor, x, section=f"{section}.hard_ansatz", repetitions=repetitions)
    one_minus_x = executor.binary("sub", 1.0, x_norm, section=f"{section}.hard_ansatz", repetitions=repetitions)
    spatial_bound = executor.binary(
        "multiply", x_norm, one_minus_x, section=f"{section}.hard_ansatz", repetitions=repetitions
    )
    t_sq = executor.binary("multiply", t_norm, t_norm, section=f"{section}.hard_ansatz", repetitions=repetitions)
    correction = executor.binary(
        "multiply", t_sq, spatial_bound, section=f"{section}.hard_ansatz", repetitions=repetitions
    )
    correction = executor.binary(
        "multiply", correction, nn_out, section=f"{section}.hard_ansatz", repetitions=repetitions
    )
    psi = executor.binary("add", psi_ic, correction, section=f"{section}.hard_ansatz", repetitions=repetitions)

    potential = compute_regge_wheeler_potential(
        executor,
        x.squeeze(-1),
        section=f"{section}.residual_potential",
        repetitions=repetitions,
    ).reshape(batch_size, 1)
    psi_tt = np.ones_like(batch)
    psi_xx = np.ones_like(batch)
    residual = executor.binary("sub", psi_tt, psi_xx, section=f"{section}.residual", repetitions=repetitions)
    source = executor.binary("multiply", potential, psi, section=f"{section}.residual", repetitions=repetitions)
    executor.binary("add", residual, source, section=f"{section}.residual", repetitions=repetitions)


def _simulate_network_backward(
    executor: PrimitiveExecutor,
    *,
    batch_size: int,
    repetitions: int,
    section_prefix: str = "pinn",
) -> None:
    """Backpropagation pass through the PINN network.

    ``section_prefix`` lets callers distinguish v2 and v3 backward sections
    in the stats output (e.g. ``"pinn"`` vs ``"pinn_rk1"``).  All section
    names are ``{section_prefix}.backward.<layer>``.
    """
    cfg = executor.config
    embed_half = cfg.embedding_features // 2
    embedding = np.ones((batch_size, cfg.embedding_features), dtype=np.float32)
    hidden = np.ones((batch_size, cfg.hidden_width), dtype=np.float32)
    grad_out = np.ones((batch_size, cfg.output_width), dtype=np.float32)

    bp = f"{section_prefix}.backward"

    # Output gradient to obtain gradients for PDE residual loss
    out_weights_t = np.ones((cfg.output_width, cfg.hidden_width), dtype=np.float32)
    hidden_t = np.ones((cfg.hidden_width, batch_size), dtype=np.float32)
    executor.matmul(hidden_t, grad_out, section=f"{bp}.dense_out_grad", repetitions=repetitions)
    grad_hidden = executor.matmul(
        grad_out, out_weights_t, section=f"{bp}.dense_out_backprop", repetitions=repetitions
    )

    tanh_sq = executor.binary(
        "multiply", hidden, hidden, section=f"{bp}.tanh_out", repetitions=repetitions
    )
    tanh_grad = executor.binary("sub", 1.0, tanh_sq, section=f"{bp}.tanh_out", repetitions=repetitions)
    grad_hidden = executor.binary(
        "multiply", grad_hidden, tanh_grad, section=f"{bp}.tanh_out", repetitions=repetitions
    )

    recurrent_t = np.ones((cfg.hidden_width, batch_size), dtype=np.float32)
    recurrent_w_t = np.ones((cfg.hidden_width, cfg.hidden_width), dtype=np.float32)
    for layer_idx in range(cfg.hidden_layers - 1):
        prefix = f"{bp}.hidden_{layer_idx + 2}"
        executor.matmul(recurrent_t, grad_hidden, section=f"{prefix}.weight_grad", repetitions=repetitions)
        grad_hidden = executor.matmul(
            grad_hidden, recurrent_w_t, section=f"{prefix}.backprop", repetitions=repetitions
        )
        tanh_sq = executor.binary(
            "multiply", hidden, hidden, section=f"{prefix}.tanh", repetitions=repetitions
        )
        tanh_grad = executor.binary("sub", 1.0, tanh_sq, section=f"{prefix}.tanh", repetitions=repetitions)
        grad_hidden = executor.binary(
            "multiply", grad_hidden, tanh_grad, section=f"{prefix}.tanh", repetitions=repetitions
        )

    embed_t = np.ones((cfg.embedding_features, batch_size), dtype=np.float32)
    hidden_w_t = np.ones((cfg.hidden_width, cfg.embedding_features), dtype=np.float32)
    executor.matmul(embed_t, grad_hidden, section=f"{bp}.dense1_grad", repetitions=repetitions)
    grad_embedding = executor.matmul(
        grad_hidden, hidden_w_t, section=f"{bp}.dense1_backprop", repetitions=repetitions
    )

    proj_grad = np.ones((batch_size, embed_half), dtype=np.float32)
    fourier_input_t = np.ones((2, batch_size), dtype=np.float32)
    fourier_w_t = np.ones((embed_half, 2), dtype=np.float32)
    executor.matmul(
        fourier_input_t, proj_grad, section=f"{bp}.fourier_grad", repetitions=repetitions
    )
    executor.matmul(
        proj_grad, fourier_w_t, section=f"{bp}.fourier_backprop", repetitions=repetitions
    )
    executor.binary(
        "multiply", grad_embedding, np.float32(1.0), section=f"{bp}.embedding_merge", repetitions=repetitions
    )


def _simulate_adam_updates(executor: PrimitiveExecutor, *, repetitions: int) -> None:
    cfg = executor.config
    total_params = (
        (cfg.embedding_features // 2) * 2
        + cfg.embedding_features * cfg.hidden_width
        + (cfg.hidden_layers - 1) * cfg.hidden_width * cfg.hidden_width
        + cfg.hidden_width * cfg.output_width
        + cfg.hidden_layers * cfg.hidden_width
        + cfg.output_width
    )
    params = np.ones((total_params,), dtype=np.float32)
    for op_idx in range(cfg.adam_elementwise_ops):
        executor.binary(
            "multiply" if op_idx % 2 == 0 else "add",
            params,
            np.float32(0.9 if op_idx % 2 == 0 else 1e-3),
            section="pinn.optimizer",
            repetitions=repetitions,
        )


def simulate_pinn_training(executor: PrimitiveExecutor) -> None:
    """v2 autograd-heavy PINN training surrogate.

    Forward pass equivalents (``pinn_forward_pass_equivalents``) amortise the
    create_graph autograd overhead for computing PDE derivatives into the
    forward section.  Backward pass uses the standard single-pass model.
    """
    cfg = executor.config
    _simulate_network_forward(
        executor,
        batch_size=cfg.pinn_collocation_points,
        section="pinn.forward_surrogate",
        repetitions=cfg.pinn_epochs * cfg.pinn_forward_pass_equivalents,
    )
    _simulate_network_backward(
        executor,
        batch_size=cfg.pinn_collocation_points,
        repetitions=cfg.pinn_epochs * cfg.pinn_backward_pass_equivalents,
        section_prefix="pinn",
    )
    _simulate_adam_updates(executor, repetitions=cfg.pinn_epochs)


def _simulate_pinn_rk1_fd_combine(
    executor: PrimitiveExecutor,
    *,
    batch_size: int,
    repetitions: int,
) -> None:
    """Finite-difference combination step for the RK1 residual.

    After the stencil forward evaluations, combines the model outputs into
    ψ_tt and ψ_xx via central-difference formulas, builds the PDE residual,
    and computes the MSE scalar loss.  All operations are element-wise on
    (batch_size, 1) tensors — cost is negligible relative to the stencil
    forward passes and backward pass.
    """
    psi = np.ones((batch_size, 1), dtype=np.float32)

    # ψ_tt = (ψ_tp − 2·ψ_c + ψ_tm) / ∆t²  — 3 binary ops + 1 divide
    twice_c = executor.binary("multiply", psi, 2.0, section="pinn_rk1.fd_combine.psi_tt", repetitions=repetitions)
    diff_t = executor.binary("sub", psi, twice_c, section="pinn_rk1.fd_combine.psi_tt", repetitions=repetitions)
    psi_tt = executor.binary("add", diff_t, psi, section="pinn_rk1.fd_combine.psi_tt", repetitions=repetitions)
    executor.binary("multiply", psi_tt, np.float32(1.0), section="pinn_rk1.fd_combine.psi_tt", repetitions=repetitions)

    # ψ_xx = (ψ_xp − 2·ψ_c + ψ_xm) / ∆x²  — same pattern
    twice_c = executor.binary("multiply", psi, 2.0, section="pinn_rk1.fd_combine.psi_xx", repetitions=repetitions)
    diff_x = executor.binary("sub", psi, twice_c, section="pinn_rk1.fd_combine.psi_xx", repetitions=repetitions)
    psi_xx = executor.binary("add", diff_x, psi, section="pinn_rk1.fd_combine.psi_xx", repetitions=repetitions)
    executor.binary("multiply", psi_xx, np.float32(1.0), section="pinn_rk1.fd_combine.psi_xx", repetitions=repetitions)

    # residual = ψ_tt − ψ_xx + V·ψ_c
    res = executor.binary("sub", psi_tt, psi_xx, section="pinn_rk1.fd_combine.residual", repetitions=repetitions)
    v_psi = executor.binary("multiply", psi, psi, section="pinn_rk1.fd_combine.residual", repetitions=repetitions)
    residual = executor.binary("add", res, v_psi, section="pinn_rk1.fd_combine.residual", repetitions=repetitions)

    # MSE loss: mean(residual²)
    res_sq = executor.binary("multiply", residual, residual, section="pinn_rk1.fd_combine.mse", repetitions=repetitions)
    executor.reduce_sum(res_sq.reshape(1, -1), section="pinn_rk1.fd_combine.mse", repetitions=repetitions)


def simulate_pinn_rk1_training(executor: PrimitiveExecutor) -> None:
    """v3 RK1 finite-difference PINN training surrogate.

    Each training step performs one forward pass and one backward pass.
    For the forward pass, stencil points are packed into the batch dimension:
    ``effective_batch = effective_centers * pinn_rk1_fd_stencil_evals`` where
    ``effective_centers = pinn_collocation_points * pinn_rk1_collocation_multiplier``.
    There is no ``create_graph`` autograd overhead — PDE derivatives are
    assembled from the packed forward outputs in ``pinn_rk1.fd_combine.*``.
    """
    cfg = executor.config
    effective_centers = cfg.pinn_collocation_points * cfg.pinn_rk1_collocation_multiplier
    effective_batch = effective_centers * cfg.pinn_rk1_fd_stencil_evals
    backward_batch = effective_centers if cfg.pinn_rk1_backprop_centers_only else effective_batch
    _simulate_network_forward(
        executor,
        batch_size=effective_batch,
        section="pinn_rk1.stencil_forward",
        repetitions=cfg.pinn_epochs,
    )
    _simulate_pinn_rk1_fd_combine(
        executor,
        batch_size=effective_centers,
        repetitions=cfg.pinn_epochs,
    )
    _simulate_network_backward(
        executor,
        batch_size=backward_batch,
        repetitions=cfg.pinn_epochs,
        section_prefix="pinn_rk1",
    )
    _simulate_adam_updates(executor, repetitions=cfg.pinn_epochs)


def simulate_strain_postprocess(executor: PrimitiveExecutor) -> None:
    cfg = executor.config
    freqs = max((cfg.nt // 2) - 1, 1)
    dft_matrix = np.ones((freqs, cfg.nt), dtype=np.float32)
    signal = np.ones((cfg.nt, 1), dtype=np.float32)
    executor.matmul(
        dft_matrix,
        signal,
        section="post.rfft_surrogate",
        repetitions=cfg.dft_surrogate_repetitions,
    )
    omega = np.linspace(1.0, np.pi * freqs, freqs, dtype=np.float32).reshape(-1, 1)
    omega_sq = executor.binary("multiply", omega, omega, section="post.strain_scale")
    magnitude = np.ones((freqs, 1), dtype=np.float32)
    executor.binary(
        "div",
        magnitude,
        omega_sq,
        section="post.strain_scale",
        repetitions=cfg.dft_surrogate_repetitions,
    )


def simulate_full_workload(executor: PrimitiveExecutor) -> None:
    simulate_fd_solver(executor)
    if executor.config.pinn_model_version == "v2_autograd":
        simulate_pinn_training(executor)
    else:
        simulate_pinn_rk1_training(executor)
    simulate_strain_postprocess(executor)