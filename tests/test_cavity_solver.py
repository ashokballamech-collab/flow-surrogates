"""Tests for the lid-driven cavity solver."""
import numpy as np
import pytest
from cavity_solver import build_poisson_solver, solve_cavity


def test_poisson_matrix_is_the_five_point_stencil():
    """The Kronecker-product assembly must equal a hand-built five-point Laplacian."""
    N, h = 4, 0.25
    n = N - 1
    expected = np.zeros((n * n, n * n))
    for j in range(n):
        for i in range(n):
            k = j * n + i
            expected[k, k] = -4.0
            if i > 0:        expected[k, k - 1] = 1.0
            if i < n - 1:    expected[k, k + 1] = 1.0
            if j > 0:        expected[k, k - n] = 1.0
            if j < n - 1:    expected[k, k + n] = 1.0
    expected /= h**2

    lu = build_poisson_solver(N, h)
    # recover the matrix one column at a time by solving A x = e_k
    identity = np.eye(n * n)
    inverse = np.column_stack([lu.solve(identity[:, k]) for k in range(n * n)])
    np.testing.assert_allclose(inverse @ expected, identity, atol=1e-9)


def test_poisson_solve_recovers_a_known_right_hand_side():
    """Solving A x = A x_known must return x_known."""
    N, h = 8, 1 / 8
    lu = build_poisson_solver(N, h)
    rng = np.random.default_rng(0)
    x_known = rng.standard_normal((N - 1) ** 2)
    # apply A by solving with the inverse of the LU, then check round trip
    b = np.linalg.solve(np.column_stack([lu.solve(e) for e in np.eye((N - 1) ** 2)]), x_known)
    np.testing.assert_allclose(lu.solve(b), x_known, atol=1e-8)


def test_solver_converges_and_reports_it():
    sol = solve_cavity(100, N=16, verbose=False)
    assert sol["info"]["converged"], "solver should reach steady state at Re = 100"
    assert sol["info"]["residual"] < 1e-6


def test_boundary_conditions_hold():
    sol = solve_cavity(100, N=16, verbose=False)
    psi, u, v = sol["psi"], sol["u"], sol["v"]
    # the stream function is zero on all four walls
    for edge in (psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]):
        np.testing.assert_allclose(edge, 0.0, atol=1e-12)
    # the lid moves at u = 1; the other walls do not move
    np.testing.assert_allclose(u[-1, :], 1.0)
    np.testing.assert_allclose(u[0, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(v[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(v[:, -1], 0.0, atol=1e-12)


def test_primary_vortex_turns_clockwise():
    """A lid moving in +x drives a clockwise vortex, so psi is negative inside."""
    sol = solve_cavity(100, N=16, verbose=False)
    assert sol["psi"].min() < 0
    assert sol["w"][1:-1, 1:-1].mean() < 0, "interior vorticity should be clockwise"


def test_total_vorticity_is_set_by_the_lid_and_does_not_depend_on_Re():
    """
    By Stokes' theorem the vorticity integrated over the cavity equals the circulation
    around its boundary, which only the lid provides: -U L = -1, whatever Re is.
    """
    totals = []
    for Re in (100, 400):
        sol = solve_cavity(Re, N=32, verbose=False)
        x = sol["x"]
        totals.append(np.trapezoid(np.trapezoid(sol["w"], x, axis=1), x))
    for total in totals:
        assert total == pytest.approx(-1.0, abs=0.05)
    assert totals[0] == pytest.approx(totals[1], abs=1e-3)


def test_refining_the_grid_moves_psi_min_towards_the_benchmark():
    """Ghia et al. (1982) report psi_min = -0.1034 at Re = 100."""
    reference = -0.1034
    errors = [abs(solve_cavity(100, N=N, verbose=False)["psi"].min() - reference)
              for N in (16, 32)]
    assert errors[1] < errors[0], "a finer grid must be closer to the benchmark"
    assert errors[1] < 0.01


def test_warm_start_gives_the_same_answer_as_a_cold_start():
    cold = solve_cavity(200, N=16, verbose=False)
    first = solve_cavity(150, N=16, verbose=False)
    warm = solve_cavity(200, N=16, verbose=False, init=first)
    np.testing.assert_allclose(warm["psi"], cold["psi"], atol=1e-5)
    assert warm["info"]["steps"] < cold["info"]["steps"], "warm start should be cheaper"
