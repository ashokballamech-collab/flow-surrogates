"""
Lid-driven cavity flow: finite-difference solver (stream function-vorticity formulation).

Equations (non-dimensional, unit square, lid moving with u = 1 at y = 1):
    Vorticity transport:  dw/dt + u dw/dx + v dw/dy = (1/Re) (d2w/dx2 + d2w/dy2)
    Stream function:      d2psi/dx2 + d2psi/dy2 = -w
    Velocities:           u = dpsi/dy,  v = -dpsi/dx

Method:
    - Uniform grid, second-order central differences in space.
    - Explicit (forward Euler) time marching of the vorticity equation until steady state.
    - Poisson equation for psi solved with a sparse direct solver (LU factorized once, reused every step).
    - Wall vorticity from Thom's formula (first-order accurate at the wall, standard for this method).
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_poisson_solver(N, h):
    """LU-factorize the 5-point Laplacian on the (N-1) x (N-1) interior nodes (psi = 0 on all walls)."""
    n = N - 1
    I = sp.identity(n, format="csr")
    T = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="csr")
    L = (sp.kron(I, T) + sp.kron(T, I)) / h**2
    return spla.splu(L.tocsc())


def solve_cavity(Re, N=64, tol=1e-6, max_steps=500_000, verbose=True, init=None):
    """
    Solve steady lid-driven cavity flow at Reynolds number Re on an (N+1) x (N+1) node grid.
    Arrays are indexed [j, i] = [y-index, x-index].
    init: optional previous solution (dict) used as the starting guess ("warm start").
    Returns a dict with x, y, u, v, psi, w and solver info.
    """
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    if init is not None and init["N"] == N:
        psi, w = init["psi"].copy(), init["w"].copy()   # start from a nearby solution
    else:
        psi, w = np.zeros((N + 1, N + 1)), np.zeros((N + 1, N + 1))
    lu = build_poisson_solver(N, h)

    # Stable time step for explicit Euler + central differences:
    #   diffusion limit:  dt <= Re h^2 / 4
    #   convection limit (central differencing, |u| <= 1):  dt <= 2 / (Re |u|^2) -> use 80% of the limit for safety
    dt = 0.8 * min(Re * h**2 / 4.0, 2.0 / Re, h)
    t0 = time.time()

    for step in range(1, max_steps + 1):
        # --- 1. Wall vorticity (Thom's formula) ---
        w[0, :]  = -2.0 * psi[1, :] / h**2                 # bottom wall
        w[-1, :] = -2.0 * psi[-2, :] / h**2 - 2.0 / h      # moving lid (u = 1)
        w[:, 0]  = -2.0 * psi[:, 1] / h**2                 # left wall
        w[:, -1] = -2.0 * psi[:, -2] / h**2                # right wall

        # --- 2. Velocities at interior nodes ---
        u = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
        v = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)

        # --- 3. Vorticity transport, explicit Euler ---
        wc = w[1:-1, 1:-1]
        dwdx = (w[1:-1, 2:] - w[1:-1, :-2]) / (2 * h)
        dwdy = (w[2:, 1:-1] - w[:-2, 1:-1]) / (2 * h)
        lap_w = (w[1:-1, 2:] + w[1:-1, :-2] + w[2:, 1:-1] + w[:-2, 1:-1] - 4 * wc) / h**2
        rate = -(u * dwdx + v * dwdy) + lap_w / Re
        w[1:-1, 1:-1] = wc + dt * rate

        # --- 4. Stream function from Poisson equation ---
        psi[1:-1, 1:-1] = lu.solve((-w[1:-1, 1:-1]).ravel()).reshape(N - 1, N - 1)

        # --- 5. Convergence check: steady state when dw/dt is tiny everywhere ---
        res = np.max(np.abs(rate))
        if verbose and step % 5000 == 0:
            print(f"  Re={Re:5.0f}  step {step:7d}  max|dw/dt| = {res:.2e}")
        if res < tol:
            break
    else:
        print(f"  WARNING: Re={Re} did not converge in {max_steps} steps (residual {res:.2e})")

    # Final velocity field on the full grid (walls included)
    u_full = np.zeros_like(psi); v_full = np.zeros_like(psi)
    u_full[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
    v_full[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
    u_full[-1, :] = 1.0                                   # lid

    info = dict(steps=step, residual=res, dt=dt, seconds=time.time() - t0, converged=bool(res < tol))
    if verbose and info["converged"]:
        print(f"  Re={Re:5.0f}  converged in {step} steps, {info['seconds']:.1f} s (residual {res:.1e})")
    return dict(x=x, y=y, u=u_full, v=v_full, psi=psi, w=w, Re=Re, N=N, info=info)


if __name__ == "__main__":
    sol = solve_cavity(100, N=64)
    print("Min stream function (primary vortex strength):", sol["psi"].min())
