import numpy as np
import pytest

from drone_suru_simulatoru import coverage
from drone_suru_simulatoru.dynamics import DroneParams


def unit_grid(cells=100):
    return coverage.make_grid(1.0, 1.0, cells, cells)


def test_grid_weights_sum_to_the_domain_mass():
    pts, w = coverage.make_grid(4.0, 2.0, 40, 20)
    assert w.sum() == pytest.approx(8.0) and pts.min() > 0 and pts[:, 0].max() < 4.0
    pts, w = coverage.make_grid(1.0, 1.0, 50, 50, lambda x, y: 2.0 * np.ones_like(x))
    assert w.sum() == pytest.approx(2.0)


def test_one_drone_cost_is_one_sixth_at_the_centre():
    pts, w = unit_grid()
    assert coverage.locational_cost(np.array([[0.5, 0.5]]), pts, w) == pytest.approx(1.0 / 6.0, rel=1e-3)
    cen, mass = coverage.centroids(np.array([[0.1, 0.9]]), pts, w)
    assert np.allclose(cen, [[0.5, 0.5]], atol=1e-9) and mass[0] == pytest.approx(1.0)


def test_two_drones_converge_to_the_closed_form_optimum():
    pts, w = unit_grid()
    pos, hist = coverage.lloyd(np.array([[0.2, 0.3], [0.7, 0.6]]), pts, w, 60)
    order = np.argsort(pos[:, 0])
    assert np.allclose(pos[order], [[0.25, 0.5], [0.75, 0.5]], atol=2e-3)
    assert hist[-1] == pytest.approx(5.0 / 48.0, rel=2e-3)


def test_lloyd_cost_never_increases_and_ends_at_centroids():
    pts, w = coverage.make_grid(1.0, 1.0, 80, 80)
    for seed in range(4):
        pos0 = np.random.default_rng(seed).uniform(0, 1, (15, 2))
        pos, hist = coverage.lloyd(pos0, pts, w, 120)
        assert np.all(np.diff(hist) <= 1e-12)
        cen, _ = coverage.centroids(pos, pts, w)
        assert np.abs(cen - pos).max() < 5e-3         # a fixed point: every drone at its cell centroid


def test_cost_respects_the_fejes_toth_hexagon_bound():
    pts, w = coverage.make_grid(1.0, 1.0, 100, 100)
    ratio = {}
    for n in (7, 19, 40):
        best = min(coverage.lloyd(np.random.default_rng(s).uniform(0, 1, (n, 2)), pts, w, 80)[1][-1]
                   for s in range(2))
        ratio[n] = best / coverage.hexagon_bound(1.0, n)
        assert 1.0 < ratio[n] < 1.2           # never below the rigorous lower bound
    assert ratio[40] < ratio[19] < ratio[7]    # and the bound tightens as n grows


def test_hexagon_bound_formula():
    # a regular hexagon of area a has mean squared distance 5a/(18 sqrt 3) to its centre
    side = 1.0
    area = 1.5 * np.sqrt(3) * side ** 2
    assert coverage.hexagon_bound(area, 1) == pytest.approx(area * 5 * area / (18 * np.sqrt(3)))
    assert coverage.hexagon_bound(1.0, 10, density=3.0) == pytest.approx(3.0 * coverage.hexagon_bound(1.0, 10))


def test_empty_cell_keeps_the_drone_in_place():
    pts, w = coverage.make_grid(1.0, 1.0, 10, 10)
    pos = np.array([[0.5, 0.5], [0.5, 0.5]])          # identical drones: ties go to the first one
    cen, mass = coverage.centroids(pos, pts, w)
    assert mass[1] == 0 and np.allclose(cen[1], pos[1])


def test_density_pulls_drones_toward_the_hot_spot():
    dens = coverage.gaussian_density([(0.2, 0.2, 0.08, 5.0)], base=0.05)
    pts, w = coverage.make_grid(1.0, 1.0, 80, 80, dens)
    pos, _ = coverage.lloyd(np.random.default_rng(0).uniform(0, 1, (12, 2)), pts, w, 100)
    near = (np.linalg.norm(pos - [0.2, 0.2], axis=1) < 0.25).sum()
    assert near >= 4          # a uniform layout would put about 12 * pi * 0.25^2 = 2.4 there


def test_controller_with_a_failure_rebalances():
    from drone_suru_simulatoru.sim import World, simulate
    params = DroneParams()
    pts, w = coverage.make_grid(100.0, 100.0, 40, 40)
    ctrl = coverage.CoverageController(pts, w, params)
    pos0 = np.random.default_rng(0).uniform(10, 90, (8, 2))
    tr = simulate(pos0, np.zeros((8, 2)), ctrl, params, World(), 120.0, 0.1, kill_at=[(40.0, [0, 1])],
                  record_every=10)
    live = tr.alive[-1]
    assert live.sum() == 6
    cost = [coverage.locational_cost(tr.pos[k][tr.alive[k]], pts, w) for k in range(len(tr.t))]
    jump = int(np.searchsorted(tr.t, 40.0))
    assert cost[jump] > cost[jump - 1]                    # losing two drones hurts
    assert cost[-1] < cost[jump]                          # and the survivors recover
    cen, _ = coverage.centroids(tr.pos[-1][live], pts, w)
    assert np.abs(cen - tr.pos[-1][live]).max() < 1.0     # near a fixed point (metres)
