#%%

# Physics-Informed Neural Network (PINN) demo — FIXED v2
# Solving a Regge-Wheeler-like equation in tortoise coordinates
#
# CUMULATIVE BUG LIST (all fixed here):
#  v1-fix-1: PINN physics_loss had `residual = residual * psi`
#            → trivial ψ≡0 solution minimised the loss; removed.
#  v1-fix-2: psi4_to_strain used cumsum(cumsum(ψ)) → DC drift; "fixed" with
#            mean subtraction, but that does NOT prevent the ramp (see Bug B).
#  v1-fix-3: FD time axis used linspace(0,t_max,Nt) instead of arange(Nt)*dt.
#  v1-fix-4: FD BCs were Neumann (reflective); replaced with Sommerfeld.
#
#  v2-fix-A: POTENTIAL V(r)=2/r² is monotone — no barrier, no QNMs.
#            Fix: work in tortoise coordinate r* and use the Regge-Wheeler
#            potential, which IS bell-shaped and supports QNM oscillations.
#
#  v2-fix-B: STRAIN EXTRACTION: even with mean subtraction, cumsum of a
#            single positive pulse is always ≥ 0 (monotone ramp), which is
#            exactly the "never goes negative" symptom reported.
#            Fix: Fixed-Frequency Integration (FFI) — divide by -ω² in the
#            frequency domain with a low-frequency cutoff to suppress DC.
#
#  v3-fix-C: PINN waveform mismatch vs FD (single low-frequency cycle).
#            Root cause: soft IC enforcement + tanh spectral bias over large
#            domain. Fix: hard IC ansatz plus Fourier feature inputs.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

np.random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# ============================================================
# 1. Problem Setup — Regge-Wheeler in tortoise coordinate r*
# ============================================================
# PDE: d²ψ/dt² - d²ψ/dr*² + V(r(r*)) ψ = 0
#
# r* = r + 2M ln(r/(2M) - 1)    (tortoise coordinate)
# V_RW(r) = (1 - 2M/r)[ l(l+1)/r² + (1-s²)·2M/r³ ]
#   with M=1 (BH mass), l=2 (quadrupole), s=0 (scalar, simplest case)
#
# The potential peaks near r ≈ 3M = 3 (the photon sphere) and falls to
# zero at both the horizon (r→2M, r*→-∞) and spatial infinity.
# This bell shape is what produces QNM ring-down oscillations.
#
# For this demo we work directly in r* coordinates.
# Domain in r*: from rstar_min (just outside horizon) to rstar_max.

M   = 1.0  # BH mass
l   = 2    # angular momentum number
s   = 0    # spin weight (scalar perturbation for simplicity)

def r_from_rstar(rstar, M=1.0, tol=1e-8, max_iter=50):
    """Invert r* = r + 2M ln(r/(2M) - 1) via Newton's method."""
    r = np.maximum(rstar + 2*M, 2.1*M)   # initial guess
    for _ in range(max_iter):
        f  = r + 2*M*np.log(np.maximum(r/(2*M) - 1, tol)) - rstar
        df = r / (r - 2*M)
        dr = f / df
        r  = r - dr
        r  = np.maximum(r, 2.01*M)
        if np.max(np.abs(dr)) < tol:
            break
    return r

def V_rw(rstar):
    """
    Regge-Wheeler potential as function of tortoise coordinate r*.
    Bell-shaped: V→0 at both boundaries, peak near r*≈1 (r≈3M).
    """
    r = r_from_rstar(rstar)
    f = 1.0 - 2*M / r
    return f * (l*(l+1)/r**2 + (1 - s**2)*2*M/r**3)

# Domain in r* (r=3 → r*≈1.1; r=30 → r*≈33.4; r=2.5 → r*≈-0.9)
rstar_min = -5.0     # close to horizon
rstar_max = 60.0     # far field observer region
t_min     = 0.0
t_max     = 80.0

# Initial Gaussian centred at r*=10 (r≈11, safely in the wave zone)
rstar0 = 10.0
sigma0 = 1.5

# Affine scaling to roughly [-1, 1] to avoid tanh saturation.
rs_mid   = 0.5 * (rstar_min + rstar_max)
rs_half  = 0.5 * (rstar_max - rstar_min)
t_mid    = 0.5 * (t_min + t_max)
t_half   = 0.5 * (t_max - t_min)

FOURIER_MODES = 6
FOURIER_BASE  = np.pi

# ============================================================
# 2. Neural Network (unchanged architecture)
# ============================================================
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)-2):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f"tanh_{i}", nn.Tanh())
        self.net.add_module("output", nn.Linear(layers[-2], layers[-1]))
        # Luna-style: learnable amplitude for normalized network contribution.
        self.amplitude = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(self, x):
        return self.amplitude * self.net(x)

INPUT_DIM = 4 + 6 * FOURIER_MODES  # rs_n, t_n, u_n, v_n + 3 trig pairs per Fourier mode (rs, t, u)
model = PINN([INPUT_DIM, 128, 128, 128, 1])

def gaussian_initial_profile(rs):
    return torch.exp(-(rs - rstar0)**2 / (2 * sigma0**2))

def linear_boundary_profile(rs):
    """Linear profile matching the initial-data values at both boundaries."""
    xi = (rs - rstar_min) / (rstar_max - rstar_min)
    g_l = float(np.exp(-(rstar_min - rstar0)**2 / (2 * sigma0**2)))
    g_r = float(np.exp(-(rstar_max - rstar0)**2 / (2 * sigma0**2)))
    return (1.0 - xi) * g_l + xi * g_r

def boundary_vanishing_factor(rs):
    """0 at both boundaries, O(1) in the interior."""
    xi = (rs - rstar_min) / (rstar_max - rstar_min)
    return xi * (1.0 - xi)

def make_model_input(rs, t):
    rs_n = (rs - rs_mid) / rs_half
    t_n  = (t  - t_mid)  / t_half
    feats = [rs_n, t_n]
    
    # Characteristic coordinates: retarded u = r* - t, advanced v = r* + t
    u = rs - t      # retarded: light-cone coordinate
    v = rs + t      # advanced
    u_n = (u - (-t_max + rstar_min)) / (2.0 * t_max + (rstar_max - rstar_min))
    v_n = (v - (rstar_min)) / ((rstar_max + t_max) - (rstar_min - t_max))
    
    feats.append(u_n)
    feats.append(v_n)
    
    for k in range(FOURIER_MODES):
        omega = FOURIER_BASE * (2**k)
        feats.append(torch.sin(omega * rs_n))
        feats.append(torch.cos(omega * rs_n))
        feats.append(torch.sin(omega * t_n))
        feats.append(torch.cos(omega * t_n))
        # Also modulate by retarded time to favor outgoing characteristics
        feats.append(torch.sin(omega * u_n))
        feats.append(torch.cos(omega * u_n))
    return torch.cat(feats, dim=1)

def causal_envelope(rs, t, base_width=2.0):
    """Smooth causal envelope: ~1 in future light cone (r* > t), ~0 in past (r* < t).
    Encodes that solution at (r*,t) is influenced by propagating outward from r*=0 at earlier times.
    """
    # Characteristic distance from light cone r* = t
    xi = (rs - t) / base_width
    # Smooth heaviside: 0 for xi << -1, 1 for xi >> 1
    return 0.5 * (1.0 + torch.tanh(xi))

def model_psi(model, rs, t):
    # Characteristic ansatz with hard IC+BC + causal-envelope modulation:
    # 1) psi(rs,0) = Gaussian(rs), d_t psi(rs,0) = 0
    # 2) psi(r*_min,t) = BC value, psi(r*_max,t) = BC value
    # 3) Network contribution modulated by causal envelope to suppress past light cone
    tau = t / t_max
    g0 = gaussian_initial_profile(rs)
    gbc = linear_boundary_profile(rs)
    b = boundary_vanishing_factor(rs)
    env = causal_envelope(rs, t, base_width=2.0)
    core = model(make_model_input(rs, t))
    # core is already normalized via model.amplitude
    return gbc + (1.0 - tau**2) * (g0 - gbc) + tau**2 * b * env * core

# ============================================================
# 3. Sampling
# ============================================================
def sample_interior_points(N, t_upper=t_max):
    rs = np.random.uniform(rstar_min, rstar_max, (N, 1)).astype(np.float32)
    t  = np.random.uniform(t_min,     t_upper,   (N, 1)).astype(np.float32)
    rs_t = torch.tensor(rs, dtype=torch.float32, requires_grad=True)
    t_t  = torch.tensor(t,  dtype=torch.float32, requires_grad=True)
    return rs_t, t_t

def sample_boundary_times(N, t_upper=t_max):
    t = np.random.uniform(t_min, t_upper, (N, 1)).astype(np.float32)
    return torch.tensor(t, dtype=torch.float32, requires_grad=True)

# ============================================================
# 4. Physics Loss  (v1 fix: no `residual *= psi`)
# ============================================================
def physics_loss(model, rs, t):
    psi = model_psi(model, rs, t)

    psi_rs = torch.autograd.grad(psi, rs, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    psi_t  = torch.autograd.grad(psi, t,  grad_outputs=torch.ones_like(psi), create_graph=True)[0]

    psi_rsrs = torch.autograd.grad(psi_rs, rs, grad_outputs=torch.ones_like(psi_rs), create_graph=True)[0]
    psi_tt   = torch.autograd.grad(psi_t,  t,  grad_outputs=torch.ones_like(psi_t),  create_graph=True)[0]

    rs_np = rs.detach().cpu().numpy()
    Vval = torch.tensor(V_rw(rs_np), dtype=torch.float32)

    # PDE residual: ψ_tt - ψ_{r*r*} + V(r*(r)) ψ = 0
    residual = psi_tt - psi_rsrs + Vval * psi
    return torch.mean(residual**2)

def initial_pde_loss(model, N=512):
    rs = np.random.uniform(rstar_min, rstar_max, (N, 1)).astype(np.float32)
    rs_t = torch.tensor(rs, dtype=torch.float32, requires_grad=True)
    t0 = torch.zeros_like(rs_t, requires_grad=True)

    psi = model_psi(model, rs_t, t0)
    psi_rs = torch.autograd.grad(psi, rs_t, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    psi_t  = torch.autograd.grad(psi, t0,   grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    psi_rsrs = torch.autograd.grad(psi_rs, rs_t, grad_outputs=torch.ones_like(psi_rs), create_graph=True)[0]
    psi_tt   = torch.autograd.grad(psi_t,  t0,   grad_outputs=torch.ones_like(psi_t), create_graph=True)[0]

    V0 = torch.tensor(V_rw(rs_t.detach().cpu().numpy()), dtype=torch.float32)
    res0 = psi_tt - psi_rsrs + V0 * psi
    return torch.mean(res0**2)

def sommerfeld_bc_loss(model, N_bc=64, t_upper=t_max):
    """Sparse high-weight Sommerfeld BC loss evaluated only at boundaries.
    Left (horizon): psi_t - psi_r* = 0  (ingoing)
    Right (infinity): psi_t + psi_r* = 0  (outgoing)
    """
    t_vals = np.random.uniform(t_min, t_upper, (N_bc, 1)).astype(np.float32)
    t_bc = torch.tensor(t_vals, dtype=torch.float32, requires_grad=True)

    # Left boundary (horizon r* = rstar_min)
    rs_l = torch.full_like(t_bc, rstar_min, requires_grad=True)
    psi_l = model_psi(model, rs_l, t_bc)
    psi_l_t = torch.autograd.grad(psi_l, t_bc, grad_outputs=torch.ones_like(psi_l), create_graph=True)[0]
    psi_l_rs = torch.autograd.grad(psi_l, rs_l, grad_outputs=torch.ones_like(psi_l), create_graph=True)[0]
    # Ingoing: psi_t - psi_r* = 0
    res_l = psi_l_t - psi_l_rs

    # Right boundary (far field r* = rstar_max)
    rs_r = torch.full_like(t_bc, rstar_max, requires_grad=True)
    psi_r = model_psi(model, rs_r, t_bc)
    psi_r_t = torch.autograd.grad(psi_r, t_bc, grad_outputs=torch.ones_like(psi_r), create_graph=True)[0]
    psi_r_rs = torch.autograd.grad(psi_r, rs_r, grad_outputs=torch.ones_like(psi_r), create_graph=True)[0]
    # Outgoing: psi_t + psi_r* = 0
    res_r = psi_r_t + psi_r_rs

    return torch.mean(res_l**2) + torch.mean(res_r**2)

# ============================================================
# 6. Training
# ============================================================
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = int(os.getenv("PINN_EPOCHS", "3000"))
N_PHYS = int(os.getenv("PINN_N_PHYS", "2000"))
N_BC   = int(os.getenv("PINN_N_BC", "64"))
W_PDE  = 1.0
W_BC   = float(os.getenv("PINN_W_BC", "50.0"))  # High weight on boundaries only
W_A0   = float(os.getenv("PINN_W_A0", "0.0"))

for epoch in range(EPOCHS):
    # Causal curriculum: expand the trained time window over epochs.
    frac = min(1.0, (epoch + 1) / max(EPOCHS, 1))
    t_curr = t_min + frac * (t_max - t_min)

    rs_i, t_i = sample_interior_points(N_PHYS, t_upper=t_curr)

    loss_p = physics_loss(model, rs_i, t_i)
    loss_bc = sommerfeld_bc_loss(model, N_bc=N_BC, t_upper=t_curr)
    loss_a0 = initial_pde_loss(model, N=256)
    loss   = W_PDE * loss_p + W_BC * loss_bc + W_A0 * loss_a0

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(
            f"Epoch {epoch}, t<= {t_curr:.2f}, Total: {loss.item():.6e}, "
            f"PDE: {loss_p.item():.3e}, BC: {loss_bc.item():.3e}, A0: {loss_a0.item():.3e}"
        )

# ============================================================
# 7. Extract waveform at large r*
# ============================================================
def extract_waveform(model, rs_obs=45.0, Nt=800):
    t_vals  = np.linspace(t_min, t_max, Nt)
    rs_vals = np.full_like(t_vals, rs_obs)
    rs_t = torch.tensor(rs_vals[:, None], dtype=torch.float32)
    t_t  = torch.tensor(t_vals[:, None], dtype=torch.float32)
    psi = model_psi(model, rs_t, t_t).detach().numpy().flatten()
    return t_vals, psi

# ============================================================
# 8. Fixed-Frequency Integration  [v2 fix-B]
# ============================================================
# The old approach:
#   h = cumsum(cumsum(psi4 - mean)) * dt²
# is WRONG: integrating a single positive pulse twice always gives a
# monotone non-negative result regardless of mean removal.
#
# Correct approach — Fixed-Frequency Integration (FFI):
#   In the frequency domain: Ψ̃₄(ω) = -ω² h̃(ω)
#   → h̃(ω) = -Ψ̃₄(ω) / ω²
# Apply a low-frequency cutoff ω_cut below the lowest QNM frequency
# to suppress DC/near-DC modes that would otherwise diverge.
#
# For l=2, M=1: ω_QNM ≈ 0.37 - 0.089i (Leaver 1985)
# ω_cut = 0.05 is safely below the real part while suppressing DC.

def cosine_taper_window(N, frac=0.1):
    """Symmetric cosine taper to suppress FFT edge leakage."""
    w = np.ones(N)
    m = int(frac * (N - 1))
    if m <= 0:
        return w
    n = np.arange(m)
    ramp = 0.5 * (1.0 - np.cos(np.pi * (n + 1) / (m + 1)))
    w[:m] = ramp
    w[-m:] = ramp[::-1]
    return w

def psi4_to_strain_ffi(t, psi4, omega_cut=0.05, taper_frac=0.1):
    N    = len(t)
    dt   = t[1] - t[0]
    freq = np.fft.rfftfreq(N, d=dt)                  # cycles/unit-time
    omega = 2.0 * np.pi * freq                        # angular frequency

    # Detrend + taper so endpoint mismatch does not inject low-frequency power.
    trend = np.linspace(psi4[0], psi4[-1], N)
    psi4_proc = psi4 - trend
    psi4_proc = psi4_proc - np.mean(psi4_proc)
    psi4_proc = psi4_proc * cosine_taper_window(N, frac=taper_frac)

    psi4_fft = np.fft.rfft(psi4_proc)

    # Clamp denominator: replace |ω| < ω_cut with ω_cut
    omega_eff = np.where(np.abs(omega) < omega_cut, omega_cut, np.abs(omega))

    # h̃ = -Ψ̃₄ / ω²
    h_fft = -psi4_fft / omega_eff**2
    h_fft[0] = 0.0                                    # zero DC explicitly

    return np.fft.irfft(h_fft, n=N)

# ============================================================
# 9. Detector Projection (toy)
# ============================================================
def detector_response(h, theta=0.3, phi=1.0, psi_ang=0.0):
    F_plus = (0.5*(1 + np.cos(theta)**2)*np.cos(2*phi)*np.cos(2*psi_ang)
              - np.cos(theta)*np.sin(2*phi)*np.sin(2*psi_ang))
    return F_plus * h

# ============================================================
# 10. PINN extraction
# ============================================================
t_pinn, psi4_pinn = extract_waveform(model)
h_pinn = psi4_to_strain_ffi(t_pinn, psi4_pinn)
h_det  = detector_response(h_pinn)

# ============================================================
# 11. Finite Difference Solver in r* coordinates  [v2 fix-A + v1 fixes]
# ============================================================
Nr   = 500
rstar_arr = np.linspace(rstar_min, rstar_max, Nr)
dr   = rstar_arr[1] - rstar_arr[0]
dt   = 0.5 * dr                             # CFL: λ = dt/dr = 0.5

# v1-fix-3: derive Nt from actual physics rather than linspace
Nt   = int((t_max - t_min) / dt) + 1
t_fd = np.arange(Nt) * dt

# Initial data
V_arr  = V_rw(rstar_arr)
psi_fd = np.exp(-(rstar_arr - rstar0)**2 / (2*sigma0**2))
pi_fd  = np.zeros_like(rstar_arr)              # ∂_t ψ = 0

# Kick-start leapfrog
psi_rr          = np.zeros_like(psi_fd)
psi_rr[1:-1]    = (psi_fd[2:] - 2*psi_fd[1:-1] + psi_fd[:-2]) / dr**2
psi_prev        = psi_fd.copy()
psi_curr        = psi_fd + dt*pi_fd + 0.5*dt**2*(psi_rr - V_arr*psi_fd)

# Observer at r*=45 (far field)
obs_idx = np.argmin(np.abs(rstar_arr - 45.0))
psi_obs = []

for n in range(Nt):
    psi_rr       = np.zeros_like(psi_curr)
    psi_rr[1:-1] = (psi_curr[2:] - 2*psi_curr[1:-1] + psi_curr[:-2]) / dr**2

    psi_next = 2*psi_curr - psi_prev + dt**2*(psi_rr - V_arr*psi_curr)

    # v1-fix-4: Sommerfeld (outgoing radiation) BCs
    lam = dt / dr   # = 0.5
    psi_next[-1] = psi_curr[-1] + lam*(psi_curr[-2] - psi_curr[-1])
    psi_next[0]  = psi_curr[0]  + lam*(psi_curr[1]  - psi_curr[0])

    psi_prev, psi_curr = psi_curr, psi_next
    psi_obs.append(psi_curr[obs_idx])

psi_obs = np.array(psi_obs)

# Only use up to t_max (t_fd may extend slightly beyond due to floor division)
mask  = t_fd <= t_max
t_fd  = t_fd[mask]
psi_obs = psi_obs[:len(t_fd)]

h_fd = psi4_to_strain_ffi(t_fd, psi_obs)

def dominant_omega(t, signal, t_start=None):
    if t_start is not None:
        mask = t >= t_start
        t = t[mask]
        signal = signal[mask]
    if len(t) < 4:
        return 0.0
    dt = t[1] - t[0]
    sig = signal - np.mean(signal)
    spec = np.fft.rfft(sig)
    freq = np.fft.rfftfreq(len(sig), d=dt)
    if len(spec) <= 1:
        return 0.0
    k = np.argmax(np.abs(spec[1:])) + 1
    return 2.0 * np.pi * freq[k]

omega_pinn_full = dominant_omega(t_pinn, psi4_pinn)
omega_fd_full   = dominant_omega(t_fd, psi_obs)
omega_pinn_late = dominant_omega(t_pinn, psi4_pinn, t_start=25.0)
omega_fd_late   = dominant_omega(t_fd, psi_obs, t_start=25.0)
print(
    "Dominant angular frequency at observer "
    f"(full window): PINN={omega_pinn_full:.3f}, FD={omega_fd_full:.3f}"
)
print(
    "Dominant angular frequency at observer "
    f"(t>=25 window): PINN={omega_pinn_late:.3f}, FD={omega_fd_late:.3f}"
)

# ============================================================
# 12. Plot
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# Raw ψ at observer (sanity check — should show ring-down oscillation)
axes[0].plot(t_fd, psi_obs)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("ψ (r*=45)")
axes[0].set_title("FD raw waveform at observer\n(should show oscillating ring-down)")

# Raw PINN ψ at observer for direct waveform comparison.
axes[1].plot(t_pinn, psi4_pinn)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("ψ (r*=45)")
axes[1].set_title("PINN raw waveform at observer")

# PINN strain
axes[2].plot(t_pinn, h_det)
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Strain (arb.)")
axes[2].set_title("PINN GW Strain (fixed)")

# Comparison
axes[3].plot(t_pinn, h_pinn / np.max(np.abs(h_pinn)+1e-30),  label="PINN (normalised)")
axes[3].plot(t_fd,   h_fd   / np.max(np.abs(h_fd  )+1e-30),  label="FD (normalised)")
axes[3].legend()
axes[3].set_xlabel("Time")
axes[3].set_ylabel("Normalised strain")
axes[3].set_title("PINN vs Finite Difference (fixed)")

plt.tight_layout()
plt.show()
print("Done. Check the raw ψ panel — it should show oscillations even before FFI.")

# ============================================================
# Notes
# ============================================================
# - This is a pedagogical demo, NOT a full Teukolsky solver.
# - The QNM frequencies for l=2 Schwarzschild: ω ≈ 0.3737 - 0.0890i (M=1).
#   Expected ring-down period: T ≈ 2π/0.374 ≈ 16.8 time units.
#   Decay e-fold: τ ≈ 1/0.089 ≈ 11.2 time units.
# - Real solvers use hyperboloidal slicing or compactification to handle
#   outgoing BCs at null infinity properly.
# - Full Ψ₄ extraction requires Newman-Penrose tetrads.
# - omega_cut in FFI should be below Re(ω_QNM). Here 0.05 << 0.374 — safe.


#%%

# plot the potential for sanity check
rstar_plot = np.linspace(rstar_min, rstar_max, 1000)
plt.figure(figsize=(8,4))
plt.plot(rstar_plot, V_rw(rstar_plot))
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Regge-Wheeler potential V(r*)")

# compare with the original potential in r coordinates for sanity check
def V_rw_r(r):
    f = 1.0 - 2*M / r
    return f * (l*(l+1)/r**2 + (1 - s**2)*2*M/r**3)
r_plot = np.linspace(2.01*M, 30.0, 1000)
plt.plot(r_from_rstar(rstar_plot), V_rw_r(r_plot), label="V(r) vs V(r*) check")
plt.legend()
plt.show()

# plot the bad potential v/r² for comparison
def V_bad(r):
    return 2.0 / r**2
plt.figure(figsize=(8,4))
plt.plot(r_from_rstar(rstar_plot), V_bad(r_from_rstar(rstar_plot)), label="Bad potential 2/r²")
plt.xlabel("Tortoise coordinate r*")
plt.ylabel("Potential")
plt.legend()
plt.show()