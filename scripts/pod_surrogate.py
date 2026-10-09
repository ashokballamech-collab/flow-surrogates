"""
Session 5, part 1: a POD + neural-network surrogate for the cylinder wake.

Instead of predicting every grid value, the network predicts a handful of POD coefficients,
and the flow field is rebuilt from them:

    inputs : Reynolds number, and the phase of the shedding cycle (as cos and sin)
    outputs: R_MODES POD coefficients  ->  full velocity field

Usage:  python pod_surrogate.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from surrogate_net import make_network

torch.manual_seed(0)
R_MODES = 30                                     # number of POD modes kept
TRAIN_RE = [60, 80, 100, 120, 140, 160, 180]
TEST_RE = [90, 150]                              # never used for training

# ------------------------------------------------------------------
# 1. Load every case and build one shared snapshot matrix
# ------------------------------------------------------------------
def load(Re):
    d = np.load(f"wake_Re{Re}.npz")
    return d["snapshots"][:, :2], d["t_snapshots"], float(d["St"]), d["x"], d["y"]

cases = {Re: load(Re) for Re in TRAIN_RE + TEST_RE if glob.glob(f"wake_Re{Re}.npz")}
avail_train = [Re for Re in TRAIN_RE if Re in cases]
avail_test = [Re for Re in TEST_RE if Re in cases]
_, _, _, x, y = cases[avail_train[0]]
ny, nx = cases[avail_train[0]][0].shape[2:]
print(f"training cases: {avail_train}")
print(f"test cases:     {avail_test}")

X_train = np.concatenate([cases[Re][0].reshape(len(cases[Re][0]), -1).T for Re in avail_train], axis=1)
print(f"shared snapshot matrix: {X_train.shape[0]} values x {X_train.shape[1]} snapshots")

# ------------------------------------------------------------------
# 2. One POD basis for all training Reynolds numbers
# ------------------------------------------------------------------
X_mean = X_train.mean(axis=1, keepdims=True)
U, s, _ = np.linalg.svd(X_train - X_mean, full_matrices=False)
U = U[:, :R_MODES]
energy = s**2 / np.sum(s**2)
print(f"first {R_MODES} shared modes hold {100 * energy[:R_MODES].sum():.2f}% of the fluctuation energy")

def coefficients(Re):
    """POD coefficients of every snapshot of one case: shape (n_snap, R_MODES)."""
    Xc = cases[Re][0].reshape(len(cases[Re][0]), -1).T
    return (U.T @ (Xc - X_mean)).T

def phase(Re):
    """
    Phase of the shedding cycle at each snapshot, measured directly from the data.

    Modes 1 and 2 are the travelling-wave pair (Session 4), so the point (a1, a2)
    goes once round a circle per shedding cycle: its angle IS the phase.
    """
    a = coefficients(Re)
    return np.arctan2(a[:, 1], a[:, 0])

# ------------------------------------------------------------------
# 3. Training data: (Re, cos phase, sin phase) -> coefficients
# ------------------------------------------------------------------
def dataset(Re_list):
    ins, outs = [], []
    for Re in Re_list:
        ph, a = phase(Re), coefficients(Re)
        ins.append(np.stack([np.full_like(ph, Re), np.cos(ph), np.sin(ph)], axis=1))
        outs.append(a)
    return np.concatenate(ins), np.concatenate(outs)

In_tr, Out_tr = dataset(avail_train)
Re_mean = In_tr[:, 0].mean()
Re_std = max(In_tr[:, 0].std(), 1.0)             # guard: avoids dividing by zero with a single case
a_std = Out_tr.std(axis=0)                       # scale each coefficient to order one

def to_tensor(In, Out=None):
    I = In.copy(); I[:, 0] = (I[:, 0] - Re_mean) / Re_std
    t_in = torch.tensor(I, dtype=torch.float32)
    if Out is None:
        return t_in
    return t_in, torch.tensor(Out / a_std, dtype=torch.float32)

Xt, Yt = to_tensor(In_tr, Out_tr)
print(f"training samples: {len(Xt)}")

# ------------------------------------------------------------------
# 4. A small network
# ------------------------------------------------------------------
model = make_network(R_MODES)
print(f"network parameters: {sum(p.numel() for p in model.parameters())}")
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.5)
loss_fn = nn.MSELoss()
t0 = time.time()
for epoch in range(1, 10001):
    opt.zero_grad(); loss = loss_fn(model(Xt), Yt); loss.backward(); opt.step(); sched.step()
    if epoch % 4000 == 0:
        print(f"  epoch {epoch:5d}: loss {loss.item():.2e}")
print(f"training took {time.time() - t0:.1f} s")

# ------------------------------------------------------------------
# 5. Test: predict whole flow fields, including at unseen Reynolds numbers
# ------------------------------------------------------------------
def predict_fields(Re, ph):
    In = np.stack([np.full_like(ph, Re), np.cos(ph), np.sin(ph)], axis=1)
    with torch.no_grad():
        a = model(to_tensor(In)).numpy() * a_std
    return (X_mean + U @ a.T).T

print(f"\n{'Re':>5} {'kind':>14} {'POD-limit error':>16} {'surrogate error':>16}")
results = {}
for Re in avail_train + avail_test:
    truth = cases[Re][0].reshape(len(cases[Re][0]), -1)
    proj = (X_mean + U @ coefficients(Re).T).T          # best possible with these modes
    pred = predict_fields(Re, phase(Re))
    denom = np.linalg.norm(truth - X_mean.T)            # size of the fluctuations
    e_pod = np.linalg.norm(proj - truth) / denom
    e_sur = np.linalg.norm(pred - truth) / denom
    kind = "test (unseen)" if Re in avail_test else "training"
    print(f"{Re:5d} {kind:>14} {100 * e_pod:15.2f}% {100 * e_sur:15.2f}%")
    results[Re] = (truth, pred, 100 * e_sur)

t0 = time.time()
for _ in range(1000):
    predict_fields(110.0, np.array([0.0]))
print(f"\nsurrogate: {time.time() - t0:.3f} ms per flow field (the solver needs several minutes per case)")

np.savez("pod_basis.npz", U=U, X_mean=X_mean[:, 0], a_std=a_std,
         Re_mean=Re_mean, Re_std=Re_std, x=x, y=y, ny=ny, nx=nx)
torch.save(model.state_dict(), "pod_surrogate.pt")

# ------------------------------------------------------------------
# 6. Plots
# ------------------------------------------------------------------
def vort(f):
    u, v = f.reshape(2, ny, nx)
    return np.gradient(v, x, axis=1) - np.gradient(u, y, axis=0)

def show(ax, w, title, lim=None):
    lim = lim or np.abs(w).max()
    ax.contourf(x, y, w, levels=np.linspace(-lim, lim, 31), cmap="RdBu_r")
    ax.add_patch(plt.Circle((0, 0), 0.5, color="k"))
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    return lim

if avail_test:
    fig, axes = plt.subplots(len(avail_test), 3, figsize=(12, 2.8 * len(avail_test)), squeeze=False)
    for row, Re in enumerate(avail_test):
        truth, pred, err = results[Re]
        k = len(truth) // 2
        lim = show(axes[row][0], vort(truth[k]), f"Re = {Re}: solver (unseen case)")
        show(axes[row][1], vort(pred[k]), f"POD surrogate ({err:.1f}% error)", lim)
        show(axes[row][2], vort(pred[k] - truth[k]), "difference (scale / 5)", lim / 5)
    plt.tight_layout(); plt.savefig("surrogate_fields.png", dpi=150)

plt.figure(figsize=(6.5, 3.6))
plt.plot(avail_train, [results[r][2] for r in avail_train], "o-", color="#0E7C86", label="training Re")
if avail_test:
    plt.plot(avail_test, [results[r][2] for r in avail_test], "s", ms=10, color="#B7791F", label="unseen Re")
plt.xlabel("Reynolds number"); plt.ylabel("field error (%)"); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig("surrogate_error.png", dpi=150)
print("saved surrogate_fields.png, surrogate_error.png, pod_basis.npz, pod_surrogate.pt")
