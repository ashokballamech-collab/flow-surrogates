"""Plots for one wake case: probe signal, vorticity field and the Strouhal number."""
import sys
import numpy as np
import matplotlib.pyplot as plt

Re = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
d = np.load(f"wake_Re{Re:.0f}.npz")
print(f"Re = {Re:.0f}: St = {float(d['St']):.4f}")

# 1. probe signal: growth of the instability, then steady periodic shedding
plt.figure(figsize=(8, 3))
plt.plot(d["t"], d["v_probe"], lw=0.8)
plt.xlabel("time t (in units of D/U)"); plt.ylabel("v at probe")
plt.title(f"Transverse velocity 2 D behind the cylinder, Re = {Re:.0f}")
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"probe_Re{Re:.0f}.png", dpi=150)

# 2. vorticity in the whole domain at the final time
w = d["final_fields"][2]
X, Y = np.meshgrid(d["x_full"], d["y_full"])
plt.figure(figsize=(10, 4.2))
plt.contourf(X, Y, np.clip(w, -3, 3), levels=np.linspace(-3, 3, 41), cmap="RdBu_r")
plt.colorbar(label="vorticity")
plt.gca().add_patch(plt.Circle((0, 0), 0.5, color="k"))
plt.gca().set_aspect("equal"); plt.xlabel("x / D"); plt.ylabel("y / D")
plt.title(f"Von Karman vortex street, Re = {Re:.0f}")
plt.tight_layout(); plt.savefig(f"vorticity_Re{Re:.0f}.png", dpi=150)
print("saved", f"probe_Re{Re:.0f}.png", f"vorticity_Re{Re:.0f}.png")
