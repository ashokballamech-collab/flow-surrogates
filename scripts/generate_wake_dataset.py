"""
Generate the wake dataset for Sessions 4-5: one file per Reynolds number.

Usage:
    python generate_wake_dataset.py              # all Reynolds numbers below
    python generate_wake_dataset.py 100 140      # only the ones you list

Each file wake_Re<Re>.npz contains, for times t_record ... t_end:
    snapshots  (n_snap, 3, ny, nx): u, v and vorticity in a window around the wake
    t_snapshots, x, y (coordinates relative to the cylinder centre)
    t, v_probe: the probe signal at every time step, St: the Strouhal number
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import sys
import numpy as np
from cylinder_solver import run_case

TRAIN_RE = [60, 80, 100, 120, 140, 160, 180]
TEST_RE = [90, 150]                       # kept for testing the surrogate later

Re_list = [float(a) for a in sys.argv[1:]] or TRAIN_RE + TEST_RE
for Re in Re_list:
    # low Re: shedding takes longer to settle, so run longer
    t_end, t_record = (150.0, 100.0) if Re < 80 else (100.0, 60.0)
    out = run_case(Re, t_end=t_end, t_record=t_record)
    np.savez_compressed(f"wake_Re{Re:.0f}.npz", **out)
    print(f"Re = {Re:5.0f}:  St = {out['St']:.4f},  {out['snapshots'].shape[0]} snapshots "
          f"of shape {out['snapshots'].shape[1:]},  run time {out['seconds'] / 60:.1f} min")
