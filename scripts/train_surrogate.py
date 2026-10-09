"""
Part 2: train a neural-network surrogate for lid-driven cavity flow.
Input:  Reynolds number Re (one number)
Output: the full velocity field (u and v on the 65 x 65 grid)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import time
import numpy as np
import torch
import torch.nn as nn
from surrogate_model import Surrogate

torch.manual_seed(0)                      # same random start every run -> reproducible results

# ------------------------------------------------------------------
# 1. Load the dataset and split it
# ------------------------------------------------------------------
d = np.load("cavity_dataset.npz")
Re_all = d["Re"]                          # shape (20,)
fields = np.stack([d["u"], d["v"]], axis=1)   # shape (20, 2, 65, 65): u and v for every case
n_cases, n_ch, ny, nx = fields.shape

test_Re = [350.0, 750.0, 1200.0]          # never used for training
val_Re = [250.0, 850.0]                   # used only to watch for overfitting
is_test = np.isin(Re_all, test_Re)
is_val = np.isin(Re_all, val_Re)
is_train = ~is_test & ~is_val
print("training Re:  ", Re_all[is_train])
print("validation Re:", Re_all[is_val])
print("test Re:      ", Re_all[is_test])

# ------------------------------------------------------------------
# 2. Normalize inputs and outputs (statistics from TRAINING data only)
# ------------------------------------------------------------------
Re_mean, Re_std = Re_all[is_train].mean(), Re_all[is_train].std()
field_mean = fields[is_train].mean(axis=0)        # average flow field, shape (2, 65, 65)
field_std = fields[is_train].std()                # one number

def to_input(Re):
    """Re values (array) -> normalized network input of shape (n, 1)."""
    return torch.tensor((np.asarray(Re) - Re_mean) / Re_std, dtype=torch.float32).reshape(-1, 1)

def to_target(f):
    """Fields (n, 2, 65, 65) -> normalized, flattened targets of shape (n, 2*65*65)."""
    return torch.tensor((f - field_mean) / field_std, dtype=torch.float32).reshape(len(f), -1)

X_train, Y_train = to_input(Re_all[is_train]), to_target(fields[is_train])
X_val, Y_val = to_input(Re_all[is_val]), to_target(fields[is_val])

# ------------------------------------------------------------------
# 3. The neural network (defined in surrogate_model.py)
# ------------------------------------------------------------------
model = Surrogate(n_out=n_ch * ny * nx)
n_params = sum(p.numel() for p in model.parameters())
print(f"network has {n_params} trainable parameters")

# ------------------------------------------------------------------
# 4. Training loop
# ------------------------------------------------------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 5000
history = {"epoch": [], "train": [], "val": []}

t0 = time.time()
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()                 # 1. clear old gradients
    Y_pred = model(X_train)               # 2. forward pass: make predictions
    loss = loss_fn(Y_pred, Y_train)       # 3. how wrong are they?
    loss.backward()                       # 4. backpropagation: compute gradients
    optimizer.step()                      # 5. update the weights

    if epoch % 50 == 0:
        with torch.no_grad():             # no gradients needed just to evaluate
            val_loss = loss_fn(model(X_val), Y_val).item()
        history["epoch"].append(epoch)
        history["train"].append(loss.item())
        history["val"].append(val_loss)
    if epoch % 1000 == 0:
        print(f"epoch {epoch:5d}   train loss {loss.item():.2e}   val loss {val_loss:.2e}")
print(f"training took {time.time() - t0:.1f} s")

# ------------------------------------------------------------------
# 5. Save everything needed to use the model later
# ------------------------------------------------------------------
torch.save({"model_state": model.state_dict(), "n_out": n_ch * ny * nx,
            "Re_mean": Re_mean, "Re_std": Re_std,
            "field_mean": field_mean, "field_std": field_std,
            "history": history}, "surrogate.pt")
print("saved surrogate.pt")
