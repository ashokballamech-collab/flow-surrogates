"""
Flow past a circular cylinder: 2D incompressible Navier-Stokes, projection (fractional-step) method.

Non-dimensional: cylinder diameter D = 1, inflow speed U = 1, Re = U D / nu.
Domain: 0 <= x <= Lx, 0 <= y <= Ly, cylinder centre (xc, yc).

Method:
    - Staggered (MAC) grid: pressure at cell centres, u on vertical faces, v on horizontal faces.
    - Explicit Euler for convection (central differences) and diffusion.
    - Projection: solve a pressure Poisson equation each step so the new velocity is divergence-free.
      The Poisson matrix never changes, so it is LU-factorized once (as in the cavity solver).
    - Cylinder: "blocked cells". Cells whose centre lies inside the cylinder are solid; every
      face touching a solid cell has zero velocity (a staircase approximation of the circle).
Boundary conditions:
    - Inlet (x = 0): u = 1, v = 0.
    - Outlet (x = Lx): zero velocity gradient, pressure p = 0.
    - Top and bottom (y = 0, Ly): free slip (v = 0, du/dy = 0), mimicking an unbounded flow.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class CylinderFlow:
    def __init__(self, Re, h=1/16, Lx=20.0, Ly=8.0, xc=5.0, yc=4.0, D=1.0):
        self.Re, self.h = Re, h
        self.nx, self.ny = int(round(Lx / h)), int(round(Ly / h))
        nx, ny = self.nx, self.ny
        self.xc_cells = (np.arange(nx) + 0.5) * h          # x of cell centres
        self.yc_cells = (np.arange(ny) + 0.5) * h          # y of cell centres
        X, Y = np.meshgrid(self.xc_cells, self.yc_cells)
        self.solid = (X - xc)**2 + (Y - yc)**2 <= (D / 2)**2   # (ny, nx) True inside cylinder
        self.xcyl, self.ycyl = xc, yc

        # Velocity arrays with one layer of ghost values where needed
        #   u: (ny+2, nx+1)  rows 1..ny are real, rows 0 and ny+1 are ghosts (top/bottom)
        #   v: (ny+1, nx+2)  cols 1..nx are real, cols 0 and nx+1 are ghosts (inlet/outlet)
        self.u = np.ones((ny + 2, nx + 1))
        self.v = np.zeros((ny + 1, nx + 2))

        # Faces touching a solid cell are blocked (velocity forced to zero)
        s = self.solid
        self.u_blocked = np.zeros((ny, nx + 1), bool)
        self.u_blocked[:, 1:nx] = s[:, :-1] | s[:, 1:]
        self.v_blocked = np.zeros((ny + 1, nx), bool)
        self.v_blocked[1:ny, :] = s[:-1, :] | s[1:, :]

        self.lu = self._build_poisson()
        self.t = 0.0

    # ------------------------------------------------------------------
    def _build_poisson(self):
        """Matrix of the pressure Poisson equation on the cells, factorized once."""
        nx, ny, h = self.nx, self.ny, self.h
        idx = np.arange(nx * ny).reshape(ny, nx)
        rows, cols, vals = [], [], []
        diag = np.zeros((ny, nx))
        # open interior x-faces between cell (j, i-1) and (j, i)
        open_u = ~self.u_blocked[:, 1:nx]                    # (ny, nx-1)
        jj, ii = np.nonzero(open_u)
        a, b = idx[jj, ii], idx[jj, ii + 1]
        rows += [a, b]; cols += [b, a]; vals += [np.ones(len(a)), np.ones(len(a))]
        np.add.at(diag, (jj, ii), -1); np.add.at(diag, (jj, ii + 1), -1)
        # open interior y-faces between cell (j-1, i) and (j, i)
        open_v = ~self.v_blocked[1:ny, :]                    # (ny-1, nx)
        jj, ii = np.nonzero(open_v)
        a, b = idx[jj, ii], idx[jj + 1, ii]
        rows += [a, b]; cols += [b, a]; vals += [np.ones(len(a)), np.ones(len(a))]
        np.add.at(diag, (jj, ii), -1); np.add.at(diag, (jj + 1, ii), -1)
        # outlet: p = 0 on the outlet face (ghost value = -p), adds -2 to the last column
        diag[:, -1] -= 2
        # solid cells: simple equation p = 0 (they are not connected to the fluid)
        diag[self.solid] = 1.0
        rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag.ravel())
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nx * ny, nx * ny)) / h**2
        return spla.splu(A)

    # ------------------------------------------------------------------
    def _apply_bc(self):
        u, v = self.u, self.v
        u[:, 0] = 1.0                        # inlet
        u[:, -1] = u[:, -2]                  # outlet: zero gradient
        u[0, :] = u[1, :]                    # bottom: free slip (du/dy = 0)
        u[-1, :] = u[-2, :]                  # top: free slip
        v[0, :] = 0.0; v[-1, :] = 0.0        # bottom and top walls: no flow through
        v[:, 0] = -v[:, 1]                   # inlet: v = 0 on the inlet line
        v[:, -1] = v[:, -2]                  # outlet: zero gradient
        u[1:-1][self.u_blocked] = 0.0        # cylinder
        v[:, 1:-1][self.v_blocked] = 0.0

    def divergence(self):
        h = self.h
        return ((self.u[1:-1, 1:] - self.u[1:-1, :-1]) + (self.v[1:, 1:-1] - self.v[:-1, 1:-1])) / h

    def _project(self, dt):
        """Solve for pressure and correct the velocity so that it is divergence-free."""
        h, nx, ny = self.h, self.nx, self.ny
        rhs = self.divergence() / dt
        rhs[self.solid] = 0.0
        p = self.lu.solve(rhs.ravel()).reshape(ny, nx)
        u, v = self.u, self.v
        # interior faces: u -= dt * dp/dx  (blocked faces are fixed at zero)
        du = dt * (p[:, 1:] - p[:, :-1]) / h
        u[1:-1, 1:nx] -= np.where(self.u_blocked[:, 1:nx], 0.0, du)
        u[1:-1, nx] -= dt * (-2.0 * p[:, -1]) / h        # outlet face, p = 0 there
        dv = dt * (p[1:, :] - p[:-1, :]) / h
        v[1:ny, 1:-1] -= np.where(self.v_blocked[1:ny, :], 0.0, dv)
        self.p = p

    # ------------------------------------------------------------------
    def step(self, dt):
        h, Re = self.h, self.Re
        self._apply_bc()
        u, v = self.u, self.v
        # ---- u-momentum at interior x-faces (rows 1..ny, cols 1..nx-1)
        uc = u[1:-1, 1:-1]
        v_at_u = 0.25 * (v[:-1, 1:-2] + v[:-1, 2:-1] + v[1:, 1:-2] + v[1:, 2:-1])
        dudx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * h)
        dudy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * h)
        lap_u = (u[1:-1, 2:] + u[1:-1, :-2] + u[2:, 1:-1] + u[:-2, 1:-1] - 4 * uc) / h**2
        u_new = uc + dt * (-(uc * dudx + v_at_u * dudy) + lap_u / Re)
        # ---- v-momentum at interior y-faces (rows 1..ny-1, cols 1..nx)
        vc = v[1:-1, 1:-1]
        u_at_v = 0.25 * (u[1:-2, :-1] + u[1:-2, 1:] + u[2:-1, :-1] + u[2:-1, 1:])
        dvdx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * h)
        dvdy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * h)
        lap_v = (v[1:-1, 2:] + v[1:-1, :-2] + v[2:, 1:-1] + v[:-2, 1:-1] - 4 * vc) / h**2
        v_new = vc + dt * (-(u_at_v * dvdx + vc * dvdy) + lap_v / Re)
        u[1:-1, 1:-1] = u_new
        v[1:-1, 1:-1] = v_new
        self._apply_bc()
        self._project(dt)
        self.t += dt

    # ------------------------------------------------------------------
    def fields_at_centres(self):
        """u, v and vorticity at the cell centres, shape (ny, nx) each."""
        h = self.h
        u, v = self.u, self.v
        uc = 0.5 * (u[1:-1, :-1] + u[1:-1, 1:])
        vc = 0.5 * (v[:-1, 1:-1] + v[1:, 1:-1])
        # vorticity at cell corners, then averaged to centres
        wz = (v[:, 1:] - v[:, :-1]) / h - (u[1:, :] - u[:-1, :]) / h        # (ny+1, nx+1)
        wc = 0.25 * (wz[:-1, :-1] + wz[1:, :-1] + wz[:-1, 1:] + wz[1:, 1:])
        uc[self.solid] = 0.0; vc[self.solid] = 0.0; wc[self.solid] = 0.0
        return uc, vc, wc


def stable_dt(Re, h, u_max=1.6, safety=0.8):
    """Time step from the three limits of the cavity tutorial (diffusion, convection, CFL)."""
    return safety * min(Re * h**2 / 4.0, 2.0 / (Re * u_max**2), h / u_max)


def run_case(Re, h=1/16, Lx=30.0, Ly=16.0, xc=10.0, t_end=150.0, t_record=100.0,
             dt_snap=0.25, window=(-1.0, 11.0, -3.0, 3.0), stride=2, verbose=True):
    """
    Simulate flow past the cylinder at Reynolds number Re.

    Records the probe signal (v at 2 D behind the cylinder centre) at every step, and
    snapshots of (u, v, vorticity) every dt_snap from t_record to t_end, in a window around
    the wake (x from xc-1 to xc+11, y from yc-3 to yc+3), keeping every 'stride'-th cell.
    """
    yc = Ly / 2
    f = CylinderFlow(Re, h=h, Lx=Lx, Ly=Ly, xc=xc, yc=yc)
    # small asymmetric push behind the cylinder, so that shedding starts quickly
    X, Y = np.meshgrid(f.xc_cells, np.arange(f.ny + 1) * h)
    f.v[:, 1:-1] += 0.1 * np.exp(-((X - xc - 1.5)**2 + (Y - yc)**2))
    f._apply_bc(); f._project(1.0)

    # indices of the recording window
    ix = np.where((f.xc_cells >= xc + window[0]) & (f.xc_cells <= xc + window[1]))[0][::stride]
    iy = np.where((f.yc_cells >= yc + window[2]) & (f.yc_cells <= yc + window[3]))[0][::stride]

    dt = stable_dt(Re, h)
    n_steps = int(round(t_end / dt))
    snap_every = max(1, int(round(dt_snap / dt)))
    jp, ip = int(yc / h), int((xc + 2.0) / h)                       # probe cell
    t_hist, v_probe, snaps, t_snaps = [], [], [], []
    t0 = time.time()
    for n in range(1, n_steps + 1):
        f.step(dt)
        t_hist.append(f.t)
        v_probe.append(0.5 * (f.v[jp, ip + 1] + f.v[jp + 1, ip + 1]))
        if f.t >= t_record and n % snap_every == 0:
            uc, vc, wc = f.fields_at_centres()
            snaps.append(np.stack([uc, vc, wc])[:, iy][:, :, ix].astype(np.float32))
            t_snaps.append(f.t)
        if verbose and n % 5000 == 0:
            print(f"  Re = {Re}: t = {f.t:6.1f}  ({time.time() - t0:.0f} s)")
    t_hist, v_probe = np.array(t_hist), np.array(v_probe)
    return dict(Re=Re, h=h, dt=dt, t=t_hist, v_probe=v_probe,
                St=strouhal(t_hist, v_probe, t_record),
                snapshots=np.array(snaps), t_snapshots=np.array(t_snaps),
                x=f.xc_cells[ix] - xc, y=f.yc_cells[iy] - yc,          # coordinates relative to the cylinder
                final_fields=np.stack(f.fields_at_centres()).astype(np.float32),
                x_full=f.xc_cells - xc, y_full=f.yc_cells - yc,
                seconds=time.time() - t0)


def strouhal(t, signal, t_from):
    """
    Strouhal number St = f D / U from the probe signal, using zero crossings:
    the times at which the signal crosses its mean value going upward are one period apart.
    """
    sel = t >= t_from
    tt, s = t[sel], signal[sel] - signal[sel].mean()
    k = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]                   # upward crossings
    t_cross = tt[k] - s[k] * (tt[k + 1] - tt[k]) / (s[k + 1] - s[k])   # linear interpolation
    period = np.diff(t_cross).mean()
    return 1.0 / period


if __name__ == "__main__":
    out = run_case(100, t_end=100.0, t_record=60.0)
    print(f"Re = 100: St = {out['St']:.4f}, run time {out['seconds']:.0f} s")
