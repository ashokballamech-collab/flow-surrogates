"""
Tests for the reduced-order-modelling steps (POD and DMD).

These test the mathematics on small, exactly-known problems rather than on
simulation output, so they are fast and the expected answer is unambiguous.
"""
import numpy as np
import pytest


def travelling_wave(n_x=200, n_t=80, speed=1.5, whole_periods=None):
    """
    sin(x - c t) sampled on a grid: the canonical two-mode problem.

    whole_periods: if given, the time window covers exactly that many wave periods.
    Only then do the two POD modes carry exactly equal energy, which makes the
    (a1, a2) trajectory an exact circle rather than a slight ellipse.
    """
    x = np.linspace(0, 2 * np.pi, n_x, endpoint=False)   # periodic in x: do not repeat the endpoint
    if whole_periods is None:
        t = np.linspace(0, 10, n_t)
    else:
        t = np.linspace(0, whole_periods * 2 * np.pi / speed, n_t, endpoint=False)
    return np.array([np.sin(x - speed * ti) for ti in t]).T, x, t


def test_full_svd_reconstructs_the_data_exactly():
    X, _, _ = travelling_wave()
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    np.testing.assert_allclose(U @ np.diag(s) @ Vt, X, atol=1e-10)


def test_pod_modes_are_orthonormal():
    X, _, _ = travelling_wave()
    U, _, _ = np.linalg.svd(X - X.mean(axis=1, keepdims=True), full_matrices=False)
    gram = U[:, :6].T @ U[:, :6]
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-10)


def test_a_travelling_wave_needs_exactly_two_modes():
    """
    sin(x - ct) = sin(x)cos(ct) - cos(x)sin(ct): two fixed shapes, two time signals.
    So the third singular value must be numerically zero.
    """
    X, _, _ = travelling_wave()
    s = np.linalg.svd(X, compute_uv=False)
    assert s[1] / s[0] > 0.8, "the pair should carry comparable energy"
    assert s[2] / s[0] < 1e-10, "there is no third mode"


def test_truncation_error_never_increases_with_more_modes():
    X, _, _ = travelling_wave()
    mean = X.mean(axis=1, keepdims=True)
    U, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    errors = []
    for r in range(1, 8):
        approx = mean + U[:, :r] @ np.diag(s[:r]) @ Vt[:r]
        errors.append(np.linalg.norm(approx - X))
    assert all(b <= a + 1e-12 for a, b in zip(errors, errors[1:]))


def test_energy_fractions_sum_to_one():
    X, _, _ = travelling_wave()
    s = np.linalg.svd(X - X.mean(axis=1, keepdims=True), compute_uv=False)
    assert (s**2 / np.sum(s**2)).sum() == pytest.approx(1.0)


def test_phase_from_the_first_mode_pair_advances_steadily():
    """
    The angle of (a1, a2) is what the Session 5 surrogate uses as its phase input.
    For a travelling wave it must advance linearly in time.
    """
    X, _, t = travelling_wave(n_t=120, whole_periods=4)
    U, s, Vt = np.linalg.svd(X - X.mean(axis=1, keepdims=True), full_matrices=False)
    assert s[0] == pytest.approx(s[1], rel=1e-9), "a whole number of periods gives an exact pair"
    a = (np.diag(s) @ Vt).T
    phase = np.unwrap(np.arctan2(a[:, 1], a[:, 0]))
    rate = np.polyfit(t, phase, 1)[0]
    residual = phase - np.polyval(np.polyfit(t, phase, 1), t)
    assert abs(rate) == pytest.approx(1.5, rel=1e-3), "phase rate equals the wave speed"
    assert np.abs(residual).max() < 1e-8, "over whole periods the phase advances uniformly"


def dmd(X, r):
    """Exact DMD, as in scripts/dmd_analysis.py."""
    X1, X2 = X[:, :-1], X[:, 1:]
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    U, s, V = U[:, :r], s[:r], Vt[:r].conj().T
    A_tilde = U.conj().T @ X2 @ V / s
    lam, W = np.linalg.eig(A_tilde)
    return lam, X2 @ V / s @ W


def test_dmd_recovers_the_frequency_of_a_travelling_wave():
    X, _, t = travelling_wave(n_t=120, speed=1.5)
    dt = t[1] - t[0]
    lam, _ = dmd(X, r=2)
    frequencies = np.abs(np.log(lam).imag / dt) / (2 * np.pi)
    expected = 1.5 / (2 * np.pi)                      # omega = c, so f = c / 2 pi
    assert frequencies.max() == pytest.approx(expected, rel=1e-3)


def test_dmd_eigenvalues_of_a_steady_oscillation_sit_on_the_unit_circle():
    X, _, _ = travelling_wave(n_t=120)
    lam, _ = dmd(X, r=2)
    np.testing.assert_allclose(np.abs(lam), 1.0, atol=1e-6)
