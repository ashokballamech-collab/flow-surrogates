"""
Generate the training dataset for the surrogate model: steady cavity solutions over a range of Re.
Output: cavity_dataset.npz with arrays of shape (n_cases, N+1, N+1).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cavity_solver import solve_cavity

N = 64                                                   # all cases on the same grid
Re_list = list(range(100, 1001, 50)) + [1200]            # 350 and 750 included; held out later in Python

U, V, PSI, W, info = [], [], [], [], []
prev = None
for Re in Re_list:
    sol = solve_cavity(Re, N=N, verbose=False, init=prev)   # warm start from previous Re
    print(f"Re = {Re:5d}: {sol['info']['steps']:6d} steps, {sol['info']['seconds']:5.1f} s")
    U.append(sol["u"]); V.append(sol["v"]); PSI.append(sol["psi"]); W.append(sol["w"])
    prev = sol

np.savez_compressed("cavity_dataset.npz", Re=np.array(Re_list, dtype=float),
                    x=sol["x"], y=sol["y"], u=np.array(U), v=np.array(V),
                    psi=np.array(PSI), w=np.array(W))
print("Saved cavity_dataset.npz with", len(Re_list), "cases on a", N, "x", N, "grid")
