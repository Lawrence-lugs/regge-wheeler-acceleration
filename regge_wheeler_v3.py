#%%

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

# ==========================================
# 1. Physics & Coordinates
# ==========================================
M = 1.0
l = 2
s = 2
F_factor = 1.0

# Finite-difference step sizes used for PDE residual approximation.
# These are the ∆t and ∆x of the stencil applied to the PINN output,
# not the step sizes of the FD reference solver.
DT_FD = 0.1
DX_FD = 0.15


def tortoise_to_r(r_star, M=1.0):
    z = np.exp(r_star / (2 * M) - 1.0)
    w = np.real(lambertw(z))
    return 2 * M * (w + 1.0)


def V_RW(r_star, M=1.0, l=2, s=2):
    """Regge-Wheeler potential in tortoise coordinates.  Uses r**2 (fixed
    from v2 which had the Python XOR operator ^ by accident)."""
    r = tortoise_to_r(r_star, M)
    term1 = 1.0 - (2.0 * M / r)
    term2 = (l * (l + 1) / 2) + (1.0 - s**2) * (2.0 * M / r**3)
    return term1 * term2


# ==========================================
# 2. Finite Difference Solver (reference)
# ==========================================
def solve_fd(r_star_min, r_star_max, t_max, Nx, Nt):
    dx = (r_star_max - r_star_min) / (Nx - 1)
    dt = t_max / (Nt - 1)

    x = np.linspace(r_star_min, r_star_max, Nx)
    t = np.linspace(0, t_max, Nt)

    V = V_RW(x, M, l, s)
    psi = np.zeros((Nt, Nx))

    # Gaussian pulse initial condition
    psi[0, :] = np.exp(-((x - 10) ** 2) / 4.0)
    # First time step via Taylor expansion (second-order accurate)
    psi[1, 1:-1] = (
        psi[0, 1:-1]
        + 0.5 * (dt**2 / dx**2) * (psi[0, 2:] - 2 * psi[0, 1:-1] + psi[0, :-2])
        - 0.5 * dt**2 * V[1:-1] * psi[0, 1:-1]
    )

    for n in range(1, Nt - 1):
        d2psi_dx2 = (psi[n, 2:] - 2 * psi[n, 1:-1] + psi[n, :-2]) / dx**2
        psi[n + 1, 1:-1] = (
            2 * psi[n, 1:-1]
            - psi[n - 1, 1:-1]
            + dt**2 * (d2psi_dx2 - V[1:-1] * psi[n, 1:-1])
        )

    return t, x, psi


# ==========================================
# 3. PINN Architecture  (identical to v2)
# ==========================================
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, in_features, out_features, scale=2.0):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(in_features, out_features // 2) * scale,
            requires_grad=False,
        )

    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ReggeWheelerPINN(nn.Module):
    def __init__(self, x_min, x_max, t_max):
        super().__init__()
        self.x_min, self.x_max, self.t_max = x_min, x_max, t_max
        self.embedding = FourierFeatureEmbedding(2, 64)
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, t, x):
        t_norm = t / self.t_max
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)

        emb = self.embedding(torch.cat([t_norm, x_norm], dim=1))
        nn_out = self.net(emb)

        # Hard ansatz: enforces IC at t=0 and zero BCs at x=x_min, x_max
        psi_ic = torch.exp(-((x - 10) ** 2) / 4.0)
        spatial_bound = x_norm * (1.0 - x_norm)
        return psi_ic + (t**2) * spatial_bound * nn_out


# ==========================================
# 4. RK1 Finite-Difference PDE Residual
# ==========================================
def pde_residual_fd(
    model,
    t,
    x,
    *,
    x_min: float,
    x_max: float,
    t_max_val: float,
    dt_fd: float = DT_FD,
    dx_fd: float = DX_FD,
):
    """PDE residual via a batch-enforced 5-point central stencil.

    Collocation points are sampled from the interior so that (t±dt, x) and
    (t, x±dx) always remain in-domain.  We then evaluate all stencil points in
    a *single* batched model forward and split the result back into
    {center, t+dt, t-dt, x+dx, x-dx}.  This keeps one forward pass and one
    backward pass per training step.
    """
    # Expected shape is (N, 1); residual formulas are elementwise.
    if t.ndim != 2 or x.ndim != 2 or t.shape != x.shape:
        raise ValueError("t and x must have identical shape (N, 1).")

    # Build an enforced central-difference stencil in one batched forward call.
    t_stencil = torch.cat([t, t + dt_fd, t - dt_fd, t, t], dim=0)
    x_stencil = torch.cat([x, x, x, x + dx_fd, x - dx_fd], dim=0)
    psi_stencil = model(t_stencil, x_stencil)
    psi_c, psi_tp, psi_tm, psi_xp, psi_xm = psi_stencil.chunk(5, dim=0)

    # Central differences for second derivatives.
    dt2 = dt_fd**2
    dx2 = dx_fd**2
    psi_tt = (psi_tp - 2.0 * psi_c + psi_tm) / dt2
    psi_xx = (psi_xp - 2.0 * psi_c + psi_xm) / dx2

    # --- Regge-Wheeler potential (no graph required; same physics as V_RW) ---
    r_np = tortoise_to_r(x.detach().cpu().numpy(), M)
    r_tensor = torch.tensor(r_np, dtype=torch.float32, device=x.device)
    V_tensor = (1.0 - 2.0 * M / r_tensor) * (
        l * (l + 1) / r_tensor**2 + (1.0 - s**2) * 2.0 * M / r_tensor**3
    )

    residual = psi_tt - psi_xx + V_tensor * psi_c
    return torch.mean(residual**2)


# ==========================================
# 5. Execution & Relative Plotting
# ==========================================
if __name__ == "__main__":
    r_star_min, r_star_max, t_max = 2.0, 30.0, 20.0

    # 1. Run FD Baseline
    t_fd, x_fd, psi_fd = solve_fd(r_star_min, r_star_max, t_max, Nx=200, Nt=400)
    obs_idx = np.argmin(np.abs(x_fd - 20.0))
    psi_obs_fd = psi_fd[:, obs_idx]

    # 2. Train PINN with RK1 finite-difference residual
    print("Training PINN (RK1 FD residual)...")
    model = ReggeWheelerPINN(r_star_min, r_star_max, t_max)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    N_colloc = 3000
    # Enforce interior collocation points so the 5-point stencil is always valid.
    t_colloc = DT_FD + torch.rand(N_colloc, 1) * (t_max - 2.0 * DT_FD)
    x_colloc = (r_star_min + DX_FD) + torch.rand(N_colloc, 1) * (
        (r_star_max - r_star_min) - 2.0 * DX_FD
    )

    epochs = 2000
    best_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        # No t.requires_grad_() or x.requires_grad_() needed.
        # All PDE derivatives come from forward evaluations at stencil points;
        # loss.backward() provides the single parameter-gradient backward pass.
        loss = pde_residual_fd(
            model,
            t_colloc,
            x_colloc,
            x_min=r_star_min,
            x_max=r_star_max,
            t_max_val=t_max,
            dt_fd=DT_FD,
            dx_fd=DX_FD,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

    model.load_state_dict(best_model_state)
    print(f"\nLoaded best model with loss: {best_loss:.5f}")

    # Extract PINN observer slice
    t_tensor = torch.tensor(t_fd, dtype=torch.float32).unsqueeze(1)
    x_tensor = torch.full_like(t_tensor, 20.0)
    with torch.no_grad():
        psi_obs_pinn = model(t_tensor, x_tensor).squeeze().numpy()

    # 3. Signal processing
    def compute_strain(t_array, psi_time):
        dt = t_array[1] - t_array[0]
        Psi_tilde = np.fft.rfft(psi_time)
        freqs = np.fft.rfftfreq(len(t_array), d=dt)
        valid = freqs > 0
        freqs, Psi_tilde = freqs[valid], Psi_tilde[valid]
        omega = 2 * np.pi * freqs
        h_tilde = F_factor * Psi_tilde / (omega**2)
        return freqs, np.abs(h_tilde)

    freqs_fd, h_fd = compute_strain(t_fd, psi_obs_fd)
    freqs_pinn, h_pinn = compute_strain(t_fd, psi_obs_pinn)

    # 4. Plotting
    fig_w, fig_h = 2.5, 4
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=300)

    ax1.plot(t_fd, psi_obs_fd, label="FD", color="black", linewidth=1.5)
    ax1.plot(t_fd, psi_obs_pinn, label="PINN v3", color="red", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Time $t$")
    ax1.set_ylabel("Perturbation Field\n$\\psi(t, r^*=20)$")

    ax2.plot(freqs_fd, h_fd, label="FD Strain", color="black", linewidth=1.5)
    ax2.plot(freqs_pinn, h_pinn, label="PINN v3 Strain", color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Projected Strain\n$|\\tilde{h}(\\omega)|$")
    ax2.set_yscale("log")

    for ax in [ax1, ax2]:
        ax.grid(True, linestyle="--", alpha=0.6)

    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=2)

    plt.tight_layout()
    plt.savefig("regge_wheeler_v3_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.show()
