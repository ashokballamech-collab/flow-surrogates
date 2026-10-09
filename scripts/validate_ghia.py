"""
Validate the cavity solver against the benchmark of Ghia, Ghia & Shin (1982),
J. Comput. Phys. 48, 387-411, Table I: u-velocity along the vertical centerline x = 0.5.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import matplotlib.pyplot as plt
from cavity_solver import solve_cavity

y_ghia = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
                   0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
u_ghia = {
    100:  [1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581,
           -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.0],
    400:  [1.0, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477,
           -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.0],
    1000: [1.0, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.06080,
           -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.0],
}

N = 128   # fine grid for validation (N must be even so a grid line sits at x = 0.5)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
for ax, Re in zip(axes, [100, 400, 1000]):
    sol = solve_cavity(Re, N=N)
    u_center = sol["u"][:, N // 2]                       # column at x = 0.5
    u_interp = np.interp(y_ghia, sol["y"], u_center)
    err = np.max(np.abs(u_interp - np.array(u_ghia[Re])))
    ax.plot(u_center, sol["y"], "-", lw=2, label=f"This solver ({N}x{N})")
    ax.plot(u_ghia[Re], y_ghia, "o", mfc="none", ms=7, label="Ghia et al. (1982)")
    ax.set_title(f"Re = {Re}   (max error {err:.3f})")
    ax.set_xlabel("u at x = 0.5"); ax.grid(alpha=0.3)
    print(f"Re={Re}: max |u - u_Ghia| = {err:.4f},  psi_min = {sol['psi'].min():.5f}")
axes[0].set_ylabel("y"); axes[0].legend(loc="lower right")
plt.tight_layout(); plt.savefig("validation_ghia.png", dpi=150)
print("Saved validation_ghia.png")
