"""Compare the computed Strouhal numbers with Roshko's (1954) experimental fit."""
import glob
import numpy as np
import matplotlib.pyplot as plt

Re, St = [], []
for f in sorted(glob.glob("wake_Re*.npz")):
    d = np.load(f)
    Re.append(float(d["Re"])); St.append(float(d["St"]))
order = np.argsort(Re)
Re, St = np.array(Re)[order], np.array(St)[order]
for r, s in zip(Re, St):
    ref = 0.212 - 4.5 / r
    print(f"Re = {r:5.0f}   St = {s:.4f}   Roshko fit = {ref:.4f}   difference = {100 * (s - ref) / ref:+.1f}%")

r = np.linspace(50, 190, 200)
plt.figure(figsize=(6, 3.8))
plt.plot(r, 0.212 - 4.5 / r, "k--", label="Roshko (1954) fit to experiments")
plt.plot(Re, St, "o", color="#0E7C86", ms=8, label="this solver")
plt.xlabel("Re"); plt.ylabel("Strouhal number St")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("strouhal_vs_Re.png", dpi=150)
print("saved strouhal_vs_Re.png")
