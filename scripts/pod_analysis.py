"""
Session 4, part 1: Proper Orthogonal Decomposition (POD) of the cylinder wake.

Usage:  python pod_analysis.py            (uses wake_Re100.npz; also compares Re = 60, 100, 180)
"""
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Build the snapshot matrix: one column per snapshot
# ------------------------------------------------------------------
d = np.load("wake_Re100.npz")
snaps = d["snapshots"]                      # (n_snap, 3, ny, nx): u, v, vorticity
t = d["t_snapshots"]
x, y = d["x"], d["y"]
n_snap, _, ny, nx = snaps.shape
X = snaps[:, :2].reshape(n_snap, -1).T      # u and v stacked: shape (2*ny*nx, n_snap)
print(f"snapshot matrix: {X.shape[0]} values per snapshot, {X.shape[1]} snapshots")

X_mean = X.mean(axis=1, keepdims=True)      # the time-averaged flow
Xf = X - X_mean                             # fluctuations around the mean

# ------------------------------------------------------------------
# 2. The SVD:  Xf = U @ diag(s) @ Vt
# ------------------------------------------------------------------
U, s, Vt = np.linalg.svd(Xf, full_matrices=False)
energy = s**2 / np.sum(s**2)                # fraction of fluctuation energy in each mode
print("energy fraction of the first 8 modes (%):", np.round(100 * energy[:8], 2))
for r in [2, 4, 6, 10]:
    print(f"  first {r:2d} modes capture {100 * energy[:r].sum():6.2f}% of the energy")

# temporal coefficients a_k(t) = s_k * V_k(t): how much of mode k is present at each time
a = (np.diag(s) @ Vt)                       # shape (n_modes, n_snap)

# ------------------------------------------------------------------
# 3. Reconstruction with r modes and its error
# ------------------------------------------------------------------
def reconstruct(r):
    return X_mean + U[:, :r] @ a[:r]

ranks = np.arange(1, 21)
rel_err = [np.linalg.norm(X - reconstruct(r)) / np.linalg.norm(Xf) for r in ranks]
for r in [2, 4, 10]:
    print(f"  reconstruction error with {r:2d} modes: {100 * rel_err[r - 1]:.2f}% of the fluctuations")

# ------------------------------------------------------------------
# 4. Plots
# ------------------------------------------------------------------
def vorticity(field_uv):
    """Vorticity dv/dx - du/dy of a (u, v) field given as one long vector."""
    u, v = field_uv.reshape(2, ny, nx)
    return np.gradient(v, x, axis=1) - np.gradient(u, y, axis=0)

def show(ax, w, title):
    lim = np.abs(w).max()
    ax.contourf(x, y, w, levels=np.linspace(-lim, lim, 31), cmap="RdBu_r")
    ax.add_patch(plt.Circle((0, 0), 0.5, color="k"))
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

# (a) energy of each mode, for three Reynolds numbers
plt.figure(figsize=(6.5, 3.6))
for Re, mk in [(60, "s"), (100, "o"), (180, "^")]:
    sn = np.load(f"wake_Re{Re}.npz")["snapshots"]
    Xr = sn[:, :2].reshape(len(sn), -1).T
    sr = np.linalg.svd(Xr - Xr.mean(axis=1, keepdims=True), compute_uv=False)
    plt.semilogy(np.arange(1, 21), (sr**2 / np.sum(sr**2))[:20], mk + "-", label=f"Re = {Re}")
plt.xlabel("mode number"); plt.ylabel("energy fraction"); plt.xticks(range(1, 21, 2))
plt.grid(alpha=0.3, which="both"); plt.legend(); plt.tight_layout(); plt.savefig("pod_energy.png", dpi=150)

# (b) the mean flow and the first six modes (shown as vorticity)
fig, axes = plt.subplots(4, 2, figsize=(9, 8.2))
show(axes[0, 0], vorticity(X_mean[:, 0]), "mean flow")
show(axes[0, 1], vorticity(Xf[:, 0]), "one fluctuation snapshot")
for k in range(6):
    ax = axes[1 + k // 2, k % 2]
    show(ax, vorticity(U[:, k]), f"mode {k + 1}  ({100 * energy[k]:.1f}% of energy)")
plt.tight_layout(); plt.savefig("pod_modes.png", dpi=150)

# (c) temporal coefficients and the phase portrait of the first pair
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
for k in range(4):
    axes[0].plot(t, a[k], label=f"$a_{k + 1}(t)$")
axes[0].set_xlabel("time t"); axes[0].set_ylabel("coefficient"); axes[0].legend(ncol=4, fontsize=8)
axes[0].grid(alpha=0.3)
axes[1].plot(a[0], a[1], color="#0E7C86"); axes[1].set_aspect("equal")
axes[1].set_xlabel("$a_1$"); axes[1].set_ylabel("$a_2$"); axes[1].set_title("mode 1 against mode 2")
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("pod_coefficients.png", dpi=150)

# (d) reconstruction of one snapshot with 2 and 10 modes, and error versus number of modes
k = 40
fig = plt.figure(figsize=(11, 5.2))
for i, (field, title) in enumerate([(X[:, k], "original snapshot"),
                                    (reconstruct(2)[:, k], "2 modes"),
                                    (reconstruct(10)[:, k], "10 modes")]):
    show(fig.add_subplot(2, 3, i + 1), vorticity(field), title)
ax = fig.add_subplot(2, 1, 2)
ax.semilogy(ranks, 100 * np.array(rel_err), "o-", color="#0E7C86")
ax.set_xlabel("number of POD modes r"); ax.set_ylabel("reconstruction error (%)")
ax.set_xticks(ranks); ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig("pod_reconstruction.png", dpi=150)

np.savez("pod_Re100.npz", U=U[:, :20], s=s, a=a[:20], X_mean=X_mean[:, 0], t=t)
print("saved pod_energy.png, pod_modes.png, pod_coefficients.png, pod_reconstruction.png, pod_Re100.npz")
