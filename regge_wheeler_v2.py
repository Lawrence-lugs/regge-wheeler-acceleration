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

def tortoise_to_r(r_star, M=1.0):
    z = np.exp(r_star / (2 * M) - 1.0)
    w = np.real(lambertw(z))
    return 2 * M * (w + 1.0)

def V_RW(r_star, M=1.0, l=2, s=2):
    r = tortoise_to_r(r_star, M)
    term1 = 1 - (2 * M / r)
    term2 = (l * (l + 1) / 2) + (1 - s**2) * (2 * M / r**3)
    return term1 * term2

# ==========================================
# 2. Finite Difference Solver
# ==========================================
def solve_fd(r_star_min, r_star_max, t_max, Nx, Nt):
    dx = (r_star_max - r_star_min) / (Nx - 1)
    dt = t_max / (Nt - 1)
    
    x = np.linspace(r_star_min, r_star_max, Nx)
    t = np.linspace(0, t_max, Nt)
    
    V = V_RW(x, M, l, s)
    psi = np.zeros((Nt, Nx))
    
    # Gaussian pulse initial condition
    psi[0, :] = np.exp(-((x - 10)**2) / 4.0)
    psi[1, 1:-1] = psi[0, 1:-1] + 0.5 * (dt**2 / dx**2) * (psi[0, 2:] - 2*psi[0, 1:-1] + psi[0, :-2]) - 0.5 * dt**2 * V[1:-1] * psi[0, 1:-1]
    
    for n in range(1, Nt - 1):
        d2psi_dx2 = (psi[n, 2:] - 2*psi[n, 1:-1] + psi[n, :-2]) / dx**2
        psi[n+1, 1:-1] = 2*psi[n, 1:-1] - psi[n-1, 1:-1] + dt**2 * (d2psi_dx2 - V[1:-1] * psi[n, 1:-1])
        
    return t, x, psi

# ==========================================
# 3. PINN Architecture & Actual Training
# ==========================================
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, in_features, out_features, scale=2.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * scale, requires_grad=False)
        
    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ReggeWheelerPINN(nn.Module):
    def __init__(self, x_min, x_max, t_max):
        super().__init__()
        self.x_min, self.x_max, self.t_max = x_min, x_max, t_max
        self.embedding = FourierFeatureEmbedding(2, 64)
        self.net = nn.Sequential(
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, t, x):
        t_norm = t / self.t_max
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        
        emb = self.embedding(torch.cat([t_norm, x_norm], dim=1))
        nn_out = self.net(emb)
        
        # Hard Ansatz
        psi_ic = torch.exp(-((x - 10)**2) / 4.0)
        spatial_bound = x_norm * (1 - x_norm)
        return psi_ic + (t**2) * spatial_bound * nn_out

def pde_residual(model, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    psi = model(t, x)
    
    # Corrected: Using torch.ones_like to seed the autograd chain rule
    psi_t = torch.autograd.grad(psi, t, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    psi_tt = torch.autograd.grad(psi_t, t, grad_outputs=torch.ones_like(psi_t), create_graph=True)[0]
    
    psi_x = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    psi_xx = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
    
    # Detach x for potential calculation to avoid differentiating through Lambert W
    r_tensor = torch.tensor(tortoise_to_r(x.detach().cpu().numpy(), M), dtype=torch.float32, device=x.device)
    V_tensor = (1 - (2 * M / r_tensor)) * ((l * (l + 1) / 2) + (1 - s**2) * (2 * M / r_tensor**3))
    
    residual = psi_tt - psi_xx + V_tensor * psi
    return torch.mean(residual**2)

# ==========================================
# 4. Execution & Relative Plotting
# ==========================================
if __name__ == "__main__":
    r_star_min, r_star_max, t_max = 2.0, 30.0, 20.0
    
    # 1. Run FD Baseline
    t_fd, x_fd, psi_fd = solve_fd(r_star_min, r_star_max, t_max, Nx=200, Nt=400)
    obs_idx = np.argmin(np.abs(x_fd - 20.0))
    psi_obs_fd = psi_fd[:, obs_idx]
    
    # 2. Train PINN
    print("Training PINN...")
    model = ReggeWheelerPINN(r_star_min, r_star_max, t_max)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    # Using a StepLR scheduler to reduce learning rate every 1000 epochs
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    
    # Generate Collocation Points
    N_colloc = 3000
    t_colloc = torch.rand(N_colloc, 1) * t_max
    x_colloc = r_star_min + torch.rand(N_colloc, 1) * (r_star_max - r_star_min)

    epochs = 2000
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = pde_residual(model, t_colloc, x_colloc)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}")
    
    # Load the best model
    model.load_state_dict(best_model_state)
    print(f"\nLoaded best model with loss: {best_loss:.5f}")

    # Extract PINN observer slice
    t_tensor = torch.tensor(t_fd, dtype=torch.float32).unsqueeze(1)
    x_tensor = torch.full_like(t_tensor, 20.0)
    with torch.no_grad():
        psi_obs_pinn = model(t_tensor, x_tensor).squeeze().numpy()

    # 3. Signal Processing (Fixing the Pole)
    def compute_strain(t_array, psi_time):
        dt = t_array[1] - t_array[0]
        Psi_tilde = np.fft.rfft(psi_time)
        freqs = np.fft.rfftfreq(len(t_array), d=dt)
        
        # FIX: Drop the DC component (w=0) to prevent the pole
        valid = freqs > 0
        freqs, Psi_tilde = freqs[valid], Psi_tilde[valid]
        omega = 2 * np.pi * freqs
        
        h_tilde = F_factor * Psi_tilde / (omega**2)
        return freqs, np.abs(h_tilde)

    freqs_fd, h_fd = compute_strain(t_fd, psi_obs_fd)
    freqs_pinn, h_pinn = compute_strain(t_fd, psi_obs_pinn)


    # 4. Plotting using relative coordinate tracking
    fig_w, fig_h = 2.5,4
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=300)
    
    # Time Domain Plot (Proves oscillation)
    ax1.plot(t_fd, psi_obs_fd, label="FD $\psi(t)$", color='black', linewidth=1.5)
    ax1.plot(t_fd, psi_obs_pinn, label="PINN $\psi(t)$", color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel("Time $t$")
    ax1.set_ylabel("Perturbation Field\n$\psi(t, r^*=20)$")
    
    # Frequency Domain Strain Plot
    ax2.plot(freqs_fd, h_fd, label="FD Strain", color='black', linewidth=1.5)
    ax2.plot(freqs_pinn, h_pinn, label="PINN Strain", color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Projected Strain\n$|\\tilde{h}(\omega)|$")
    ax2.set_yscale('log') # Log scale helps visualize the ringdown frequencies
    
    # Relative positioning variables for text
    rel_x, rel_y_top, rel_y_mid = 0.05, 0.90, 0.82
    
    for ax in [ax1, ax2]:
        # ax.text(rel_x, rel_y_top, "Observer: $r^* = 20M$", transform=ax.transAxes)
    #     # ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)

    # Legend outside the plots
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=2, labels=['FD', 'PINN'])
        
    plt.tight_layout()
    plt.savefig("regge_wheeler_comparison.pdf", dpi=300, bbox_inches='tight')

    plt.show()

