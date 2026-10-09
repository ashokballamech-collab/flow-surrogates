"""
Session 5, part 2: an inverse problem.

Forward problem : Reynolds number  ->  flow field          (the surrogate of part 1)
Inverse problem : a few noisy sensor readings  ->  Reynolds number

This is the small version of "constraining a predictive model with field data":
we search for the Re (and phase) whose predicted flow best matches the measurements.

Run pod_surrogate.py first, then:  python inverse_problem.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from surrogate_net import make_network

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ------------------------------------------------------------------
# 1. Load the trained surrogate and its POD basis
# ------------------------------------------------------------------
b = np.load("pod_basis.npz")
U, X_mean, a_std = b["U"], b["X_mean"], b["a_std"]
Re_mean, Re_std = float(b["Re_mean"]), float(b["Re_std"])
x, y, ny, nx = b["x"], b["y"], int(b["ny"]), int(b["nx"])
R_MODES = U.shape[1]

model = make_network(R_MODES)
model.load_state_dict(torch.load("pod_surrogate.pt")); model.eval()
for p in model.parameters():
    p.requires_grad_(False)                       # the network is fixed; we search only over Re and phase

Ut = torch.tensor(U, dtype=torch.float32)
mean_t = torch.tensor(X_mean, dtype=torch.float32)
a_std_t = torch.tensor(a_std, dtype=torch.float32)

def field(Re, ph):
    """Predicted flow field, differentiable with respect to Re and ph."""
    inp = torch.stack([(Re - Re_mean) / Re_std, torch.cos(ph), torch.sin(ph)]).reshape(1, 3)
    return mean_t + Ut @ (model(inp)[0] * a_std_t)

# ------------------------------------------------------------------
# 2. Sensors: a few points in the wake, each reading the sideways velocity v
# ------------------------------------------------------------------
def sensor_indices(n_sensors, seed=0):
    r = np.random.default_rng(seed)
    ix = [np.argmin(np.abs(x - xx)) for xx in r.uniform(1.0, 8.0, n_sensors)]
    iy = [np.argmin(np.abs(y - yy)) for yy in r.uniform(-1.5, 1.5, n_sensors)]
    flat = [1 * ny * nx + j * nx + i for j, i in zip(iy, ix)]     # the v component at each point
    return np.array(flat), np.array([x[i] for i in ix]), np.array([y[j] for j in iy])

# ------------------------------------------------------------------
# 3. The inverse solve: gradient descent on Re and the phases
# ------------------------------------------------------------------
def solve_inverse(meas, idx, n_starts=12, n_steps=400):
    """
    Find the Reynolds number whose predicted sensor values best match the measurements.

    meas has shape (n_times, n_sensors): each sensor records several instants.
    Unknowns: one Reynolds number (shared) and one phase per instant.
    """
    meas_t = torch.tensor(meas, dtype=torch.float32)
    idx_t = torch.tensor(idx, dtype=torch.long)
    n_times = meas_t.shape[0]
    best = (np.inf, None)
    for Re0 in np.linspace(70, 170, n_starts):            # several starting guesses (multi-start)
        Re = torch.tensor(float(Re0), requires_grad=True)
        ph = torch.tensor(rng.uniform(0, 2 * np.pi, n_times), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([{"params": [Re], "lr": 2.0}, {"params": [ph], "lr": 0.1}])
        for k in range(n_steps):
            opt.zero_grad()
            pred = torch.stack([field(Re, ph[m])[idx_t] for m in range(n_times)])
            loss = ((pred - meas_t) ** 2).mean()
            loss.backward(); opt.step()
            with torch.no_grad():
                Re.clamp_(55, 190)                        # stay inside the training range
        if loss.item() < best[0]:
            best = (loss.item(), Re.item())
    return best[1], best[0]

# ------------------------------------------------------------------
# 4. Test: can we recover Re from noisy sensors?
# ------------------------------------------------------------------
if __name__ == "__main__":
    truths = sorted(int(f[7:-4]) for f in glob.glob("wake_Re*.npz"))
    print("each 'measurement' is 3 instants from the same case")
    print(f"{'true Re':>8} {'sensors':>8} {'noise':>7} {'inferred Re':>12} {'error':>8}")
    rows = []
    for Re_true in truths:
        d = np.load(f"wake_Re{Re_true}.npz")
        n = len(d["snapshots"])
        ks = [n // 2, n // 2 + 7, n // 2 + 14]                   # three instants, ~1/3 cycle apart
        snaps = d["snapshots"][ks, :2].reshape(len(ks), -1)
        sig = snaps.std()                                        # typical fluctuation size
        for n_sens, noise in [(20, 0.0), (20, 0.05), (8, 0.05), (4, 0.05)]:
            idx, sx, sy = sensor_indices(n_sens)
            meas = snaps[:, idx] + noise * sig * rng.standard_normal((len(ks), n_sens))
            Re_hat, _ = solve_inverse(meas, idx)
            err = 100 * abs(Re_hat - Re_true) / Re_true
            rows.append((Re_true, n_sens, noise, Re_hat, err))
            print(f"{Re_true:8d} {n_sens:8d} {noise:7.2f} {Re_hat:12.1f} {err:7.1f}%")

    rows = np.array(rows)
    plt.figure(figsize=(5.6, 5.2))
    for n_sens, noise, mk, lab in [(20, 0.0, "o", "20 sensors, no noise"),
                                   (20, 0.05, "s", "20 sensors, 5% noise"),
                                   (8, 0.05, "^", "8 sensors, 5% noise"),
                                   (4, 0.05, "v", "4 sensors, 5% noise")]:
        m = (rows[:, 1] == n_sens) & (rows[:, 2] == noise)
        plt.plot(rows[m, 0], rows[m, 3], mk, ms=8, label=lab)
    lim = [55, 190]
    plt.plot(lim, lim, "k--", lw=1, label="perfect recovery")
    plt.xlabel("true Reynolds number"); plt.ylabel("inferred Reynolds number")
    plt.xlim(lim); plt.ylim(lim); plt.gca().set_aspect("equal")
    plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("inverse_results.png", dpi=150)

    idx, sx, sy = sensor_indices(8)
    d = np.load(f"wake_Re{truths[len(truths) // 2]}.npz")
    plt.figure(figsize=(7, 3))
    plt.contourf(x, y, d["snapshots"][len(d["snapshots"]) // 2, 2], levels=31, cmap="RdBu_r")
    plt.plot(sx, sy, "ko", ms=8, mfc="yellow", label="sensors")
    plt.gca().add_patch(plt.Circle((0, 0), 0.5, color="k")); plt.gca().set_aspect("equal")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("sensor_positions.png", dpi=150)
    print("saved inverse_results.png, sensor_positions.png")
