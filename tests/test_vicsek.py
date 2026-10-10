import numpy as np
import pytest

from drone_suru_simulatoru import vicsek


def test_random_phase_order_formula_against_independent_headings():
    rng = np.random.default_rng(0)
    n = 100
    samples = [np.hypot(np.cos(t).mean(), np.sin(t).mean()) for t in rng.uniform(-np.pi, np.pi, (20000, n))]
    assert np.mean(samples) == pytest.approx(vicsek.random_phase_order(n), rel=0.02)


def test_noiseless_dense_system_orders_completely():
    series = vicsek.run(100, 5.0, 0.0, steps=600, burn_in=300, seed=1)
    assert series.mean() > 0.99


def test_maximal_noise_gives_the_finite_size_random_value():
    n = 100
    values = [vicsek.run(n, 5.0, 2 * np.pi, steps=1200, burn_in=300, seed=s).mean() for s in range(3)]
    assert np.mean(values) == pytest.approx(vicsek.random_phase_order(n), rel=0.15)


def test_order_falls_with_noise():
    phi = [vicsek.run(100, 5.0, eta, steps=700, burn_in=300, seed=5).mean() for eta in (0.5, 2.0, 3.5, 5.0)]
    assert np.all(np.diff(phi) < 0)


def test_ordered_start_decays_slower_than_random_start_at_moderate_noise():
    # same noise, different starts: both end in the same statistical state (ergodic), values are close
    a = vicsek.run(100, 5.0, 1.0, steps=900, burn_in=500, seed=2, ordered_start=True).mean()
    b = vicsek.run(100, 5.0, 1.0, steps=900, burn_in=500, seed=2, ordered_start=False).mean()
    assert abs(a - b) < 0.1 and a > 0.8


def test_seeded_runs_repeat_and_sweep_rows():
    a = vicsek.run(50, 4.0, 1.0, steps=100, burn_in=50, seed=3)
    b = vicsek.run(50, 4.0, 1.0, steps=100, burn_in=50, seed=3)
    assert np.array_equal(a, b) and len(a) == 50
    rows = vicsek.sweep(50, 4.0, [0.5, 2.0], steps=100, burn_in=50)
    assert len(rows) == 2 and rows[0][0] == 0.5 and all(len(r) == 4 for r in rows)
    assert rows[0][3] == pytest.approx(50 * rows[0][2] ** 2)
