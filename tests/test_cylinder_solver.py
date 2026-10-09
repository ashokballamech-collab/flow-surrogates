"""Tests for the cylinder-wake solver (projection method on a staggered grid)."""
import numpy as np
import pytest
from cylinder_solver import CylinderFlow, stable_dt, strouhal


def small_case(Re=100):
    """A deliberately tiny domain, so the tests run in a second."""
    return CylinderFlow(Re, h=1 / 8, Lx=6.0, Ly=4.0, xc=2.0, yc=2.0)


def test_cylinder_occupies_the_expected_cells():
    f = small_case()
    assert f.solid.any(), "some cells must lie inside the cylinder"
    yy, xx = np.nonzero(f.solid)
    # every solid cell centre is within half a diameter of the cylinder centre
    r = np.hypot(f.xc_cells[xx] - f.xcyl, f.yc_cells[yy] - f.ycyl)
    assert r.max() <= 0.5 + 1e-12
    # the blocked area is close to the circle's area, pi D^2 / 4
    area = f.solid.sum() * f.h**2
    assert area == pytest.approx(np.pi * 0.25, rel=0.25)


def test_projection_makes_the_velocity_divergence_free():
    """
    The whole point of the projection step: whatever the predicted velocity is,
    the corrected one must conserve mass in every fluid cell.
    """
    f = small_case()
    rng = np.random.default_rng(0)
    f.u[1:-1, 1:-1] += 0.3 * rng.standard_normal(f.u[1:-1, 1:-1].shape)
    f.v[1:-1, 1:-1] += 0.3 * rng.standard_normal(f.v[1:-1, 1:-1].shape)
    f._apply_bc()
    fluid = ~f.solid
    assert np.abs(f.divergence()[fluid]).max() > 1.0, "the test field should start messy"
    f._project(0.01)
    assert np.abs(f.divergence()[fluid]).max() < 1e-10


def test_projecting_twice_changes_nothing():
    """A projection is idempotent: the shadow of a shadow is the same shadow."""
    f = small_case()
    rng = np.random.default_rng(1)
    f.u[1:-1, 1:-1] += 0.3 * rng.standard_normal(f.u[1:-1, 1:-1].shape)
    f._apply_bc(); f._project(0.01)
    u1, v1 = f.u.copy(), f.v.copy()
    f._project(0.01)
    np.testing.assert_allclose(f.u, u1, atol=1e-12)
    np.testing.assert_allclose(f.v, v1, atol=1e-12)


def test_no_slip_on_the_cylinder_and_uniform_inflow():
    f = small_case()
    for _ in range(20):
        f.step(stable_dt(f.Re, f.h))
    uc, vc, _ = f.fields_at_centres()
    np.testing.assert_allclose(uc[f.solid], 0.0, atol=1e-12)
    np.testing.assert_allclose(vc[f.solid], 0.0, atol=1e-12)
    np.testing.assert_allclose(f.u[:, 0], 1.0, atol=1e-12)


def test_time_steps_stay_within_all_three_stability_limits():
    for Re in (60, 100, 180):
        for h in (1 / 8, 1 / 16, 1 / 32):
            dt = stable_dt(Re, h)
            assert dt <= Re * h**2 / 4 + 1e-15, "diffusion limit"
            assert dt <= 2.0 / (Re * 1.6**2) + 1e-15, "convection limit"
            assert dt <= h / 1.6 + 1e-15, "CFL limit"
            assert dt > 0


def test_a_short_run_stays_finite():
    f = small_case()
    dt = stable_dt(f.Re, f.h)
    for _ in range(100):
        f.step(dt)
    assert np.isfinite(f.u).all() and np.isfinite(f.v).all()
    assert np.abs(f.u).max() < 5.0, "velocities should stay of order the inflow speed"


@pytest.mark.parametrize("frequency", [0.1712, 0.25, 0.4])
def test_strouhal_recovers_a_known_frequency(frequency):
    """Feed the measurement routine a pure sine of known frequency."""
    t = np.arange(0, 200, 0.01)
    signal = 0.5 * np.sin(2 * np.pi * frequency * t + 0.7) + 3.0   # offset must not matter
    assert strouhal(t, signal, t_from=50.0) == pytest.approx(frequency, rel=1e-3)
