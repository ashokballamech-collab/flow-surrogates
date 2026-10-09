"""
Session 4, part 2: Dynamic Mode Decomposition (DMD) of the cylinder wake.

DMD finds the best linear rule x_(k+1) = A x_k that maps each snapshot to the next,
and splits the flow into modes that each oscillate at a single frequency.
Usage:  python dmd_analysis.py
"""
import numpy as np
import matplotlib.pyplot as plt

d = np.load("wake_Re100.npz")
snaps, t = d["snapshots"], d["t_snapshots"]
x, y, St = d["x"], d["y"], float(d["St"])
n_snap, _, ny, nx = snaps.shape
dt = t[1] - t[0]                                     # time between snapshots (0.25)
X = snaps[:, :2].reshape(n_snap, -1).T               # u and v, one column per snapshot
X_mean = X.mean(axis=1, keepdims=True)
Xf = X - X_mean                                      # work with the fluctuations

# ------------------------------------------------------------------
# 1. Exact DMD on the first n_train snapshots
# ------------------------------------------------------------------
n_train, r = 100, 10                                 # training snapshots, number of modes
X1 = Xf[:, :n_train - 1]                             # snapshots 0 ... n-2
X2 = Xf[:, 1:n_train]                                # snapshots 1 ... n-1 (one step later)

U, s, Vt = np.linalg.svd(X1, full_matrices=False)    # step 1: SVD of X1
U, s, V = U[:, :r], s[:r], Vt[:r].conj().T           # step 2: keep r modes
A_tilde = U.conj().T @ X2 @ V / s                    # step 3: small r x r version of A
lam, W = np.linalg.eig(A_tilde)                      # step 4: its eigenvalues and eigenvectors
Phi = X2 @ V / s @ W                                 # step 5: DMD modes (full size)

omega = np.log(lam) / dt                             # continuous-time growth rate + i*frequency
freq = omega.imag / (2 * np.pi)                      # frequency of each mode
b = np.linalg.lstsq(Phi, Xf[:, 0], rcond=None)[0]    # amplitude of each mode at the first snapshot

order = np.argsort(np.abs(freq))
print(f"shedding frequency from the probe (St): {St:.4f}")
print(f"{'|lambda|':>9} {'growth rate':>12} {'frequency':>10} {'freq / St':>10} {'amplitude':>10}")
for k in order:
    if freq[k] >= 0:
        print(f"{abs(lam[k]):9.5f} {omega[k].real:12.5f} {freq[k]:10.4f} {freq[k] / St:10.3f} {abs(b[k]):10.2f}")

# ------------------------------------------------------------------
# 2. Forecast: march the DMD model forward past the training data
# ------------------------------------------------------------------
k_all = np.arange(n_snap)
X_dmd = (Phi @ (b[:, None] * lam[:, None] ** k_all[None, :])).real
err = np.linalg.norm(X_dmd - Xf, axis=0) / np.linalg.norm(Xf, axis=0)
print(f"mean error on the training snapshots:  {100 * err[:n_train].mean():.2f}%")
print(f"mean error on the forecast snapshots:  {100 * err[n_train:].mean():.2f}%  "
      f"(t = {t[n_train]:.1f} to {t[-1]:.1f})")

# ------------------------------------------------------------------
# 3. Plots
# ------------------------------------------------------------------
def vorticity(field):
    u, v = field.reshape(2, ny, nx)
    return np.gradient(v, x, axis=1) - np.gradient(u, y, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
th = np.linspace(0, 2 * np.pi, 300)
axes[0].plot(np.cos(th), np.sin(th), "k--", lw=0.8, label="unit circle")
axes[0].plot(lam.real, lam.imag, "o", color="#0E7C86", ms=8, label="DMD eigenvalues")
axes[0].set_aspect("equal"); axes[0].set_xlabel("real part"); axes[0].set_ylabel("imaginary part")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
amp = np.abs(b) * np.linalg.norm(Phi, axis=0)
pos = freq >= 0
axes[1].stem(freq[pos] / St, amp[pos] / amp.max(), basefmt=" ")
axes[1].set_xlabel("frequency / shedding frequency"); axes[1].set_ylabel("relative amplitude")
axes[1].set_xticks(range(6)); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("dmd_spectrum.png", dpi=150)

positive = [j for j in order if freq[j] > 0]         # modes sorted by frequency
fig, axes = plt.subplots(1, 3, figsize=(12, 2.8))
for ax, k in zip(axes, positive[:3]):
    w = vorticity(Phi[:, k].real)
    lim = np.abs(w).max()
    ax.contourf(x, y, w, levels=np.linspace(-lim, lim, 31), cmap="RdBu_r")
    ax.add_patch(plt.Circle((0, 0), 0.5, color="k"))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"DMD mode at {freq[k] / St:.2f} x shedding frequency", fontsize=9)
plt.tight_layout(); plt.savefig("dmd_modes.png", dpi=150)

jp, ip = np.argmin(np.abs(y)), np.argmin(np.abs(x - 2.0))     # a point 2 D behind the cylinder
v_true = Xf.reshape(2, ny, nx, n_snap)[1, jp, ip]
v_dmd = X_dmd.reshape(2, ny, nx, n_snap)[1, jp, ip]
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(t, v_true, "k-", lw=2, label="simulation")
axes[0].plot(t, v_dmd, "--", color="#B7791F", lw=2, label="DMD (10 modes)")
axes[0].axvline(t[n_train - 1], color="gray"); axes[0].text(t[n_train - 1] + 0.3, 0.45, "forecast ->", color="gray")
axes[0].set_ylabel("v fluctuation at x = 2D"); axes[0].legend(loc="lower left", fontsize=8); axes[0].grid(alpha=0.3)
axes[1].semilogy(t, 100 * err, color="#0E7C86"); axes[1].axvline(t[n_train - 1], color="gray")
axes[1].set_xlabel("time t"); axes[1].set_ylabel("field error (%)"); axes[1].grid(alpha=0.3, which="both")
plt.tight_layout(); plt.savefig("dmd_forecast.png", dpi=150)
print("saved dmd_spectrum.png, dmd_modes.png, dmd_forecast.png")
