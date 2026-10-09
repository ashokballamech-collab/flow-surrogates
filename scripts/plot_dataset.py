"""Quick look at the dataset: streamlines and vorticity for a few Reynolds numbers."""
import numpy as np
import matplotlib.pyplot as plt

d = np.load("cavity_dataset.npz")
X, Y = np.meshgrid(d["x"], d["y"])
show = [100, 400, 1000]
fig, axes = plt.subplots(1, len(show), figsize=(13, 4.3))
for ax, Re in zip(axes, show):
    k = int(np.where(d["Re"] == Re)[0][0])
    ax.contourf(X, Y, np.clip(d["w"][k], -5, 5), levels=40, cmap="RdBu_r")
    ax.streamplot(d["x"], d["y"], d["u"][k], d["v"][k], color="k", density=1.4, linewidth=0.6)
    ax.set_title(f"Re = {Re}: streamlines + vorticity"); ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig("dataset_overview.png", dpi=130)
print("Saved dataset_overview.png")
