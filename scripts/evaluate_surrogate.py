"""
Part 2: evaluate the trained surrogate on the held-out test cases.
Compares it with (a) the solver's true answer and (b) simple linear interpolation.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from surrogate_model import Surrogate      # the same network class used in training
from cavity_solver import solve_cavity

# ------------------------------------------------------------------
# 1. Load data and the trained model
# ------------------------------------------------------------------
d = np.load("cavity_dataset.npz")
Re_all = d["Re"]
fields = np.stack([d["u"], d["v"]], axis=1)          # (20, 2, 65, 65)
ck = torch.load("surrogate.pt", weights_only=False)
model = Surrogate(n_out=ck["n_out"])
model.load_state_dict(ck["model_state"])
model.eval()

def predict(Re):
    """Surrogate prediction of (u, v) for one Reynolds number -> array (2, 65, 65)."""
    x = torch.tensor([[(Re - ck["Re_mean"]) / ck["Re_std"]]], dtype=torch.float32)
    with torch.no_grad():
        y = model(x).numpy().reshape(ck["field_mean"].shape)
    return y * ck["field_std"] + ck["field_mean"]     # undo the normalization

def rel_error(pred, true):
    """Relative error: size of the error divided by the size of the true field."""
    return np.linalg.norm(pred - true) / np.linalg.norm(true)

def field_at(Re):
    return fields[np.where(Re_all == Re)[0][0]]

# ------------------------------------------------------------------
# 2. Baseline: straight-line interpolation between two known cases
# ------------------------------------------------------------------
def interp_baseline(Re, Re_a, Re_b):
    wgt = (Re - Re_a) / (Re_b - Re_a)
    return (1 - wgt) * field_at(Re_a) + wgt * field_at(Re_b)

tests = {350.0: (300.0, 400.0), 750.0: (700.0, 800.0), 1200.0: (950.0, 1000.0)}
print(f"{'Re':>6} | {'surrogate error':>16} | {'interpolation error':>20}")
results = {}
for Re, (a, b) in tests.items():
    true = field_at(Re)
    nn_pred = predict(Re)
    lin_pred = interp_baseline(Re, a, b)
    results[Re] = (true, nn_pred, lin_pred)
    print(f"{Re:6.0f} | {100*rel_error(nn_pred, true):15.2f}% | {100*rel_error(lin_pred, true):19.2f}%")

# ------------------------------------------------------------------
# 3. Speed: surrogate versus solver
# ------------------------------------------------------------------
t0 = time.time()
for _ in range(1000):
    predict(750.0)
t_nn = (time.time() - t0) / 1000
t0 = time.time()
solve_cavity(750, N=64, verbose=False)            # from rest, as a fresh solve would be
t_solver = time.time() - t0
print(f"surrogate: {t_nn*1e3:.3f} ms per prediction")
print(f"solver:    {t_solver:.1f} s  ->  surrogate is about {t_solver/t_nn:,.0f} times faster")

# ------------------------------------------------------------------
# 4. Plots
# ------------------------------------------------------------------
h = ck["history"]
plt.figure(figsize=(6, 3.6))
plt.semilogy(h["epoch"], h["train"], label="training loss")
plt.semilogy(h["epoch"], h["val"], label="validation loss")
plt.xlabel("epoch"); plt.ylabel("mean squared error (normalized)")
plt.legend(); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)

y = d["y"]; mid = len(d["x"]) // 2                 # column at x = 0.5
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
for k, (Re, (true, nn_pred, lin_pred)) in enumerate(results.items()):
    top, bottom = axes[0, k], axes[1, k]
    top.plot(true[0][:, mid], y, "k-", lw=2.5, label="solver (truth)")
    top.plot(nn_pred[0][:, mid], y, "--", color="#0E7C86", lw=2, label="neural network")
    top.plot(lin_pred[0][:, mid], y, ":", color="#B7791F", lw=2, label="linear interpolation")
    kind = "extrapolation" if Re > 1000 else "interpolation"
    top.set_title(f"Re = {Re:.0f}  ({kind})"); top.set_xlabel("u at x = 0.5"); top.grid(alpha=0.3)
    bottom.plot(nn_pred[0][:, mid] - true[0][:, mid], y, "--", color="#0E7C86", lw=2, label="neural network")
    bottom.plot(lin_pred[0][:, mid] - true[0][:, mid], y, ":", color="#B7791F", lw=2, label="linear interpolation")
    bottom.axvline(0, color="k", lw=0.8)
    bottom.set_xlabel("error in u (prediction - truth)"); bottom.grid(alpha=0.3)
    bottom.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
axes[0, 0].set_ylabel("y"); axes[1, 0].set_ylabel("y")
axes[0, 0].legend(loc="lower right", fontsize=8); axes[1, 0].legend(loc="lower left", fontsize=8)
plt.tight_layout(); plt.savefig("centreline_comparison.png", dpi=150)

X, Y = np.meshgrid(d["x"], d["y"])
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, Re in zip(axes, [750.0, 1200.0]):
    true, nn_pred, _ = results[Re]
    err = np.sqrt((nn_pred[0] - true[0])**2 + (nn_pred[1] - true[1])**2)
    c = ax.contourf(X, Y, err, levels=30, cmap="magma")
    fig.colorbar(c, ax=ax, label="velocity error")
    ax.set_title(f"Surrogate error, Re = {Re:.0f}"); ax.set_aspect("equal")
plt.tight_layout(); plt.savefig("error_maps.png", dpi=150)
print("saved loss_curves.png, centreline_comparison.png, error_maps.png")
