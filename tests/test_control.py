import numpy as np
import pytest

from drone_suru_simulatoru import graph, metrics, scenarios
from drone_suru_simulatoru.control import Boids, Formation, GoalSeeker
from drone_suru_simulatoru.dynamics import DroneParams
from drone_suru_simulatoru.sim import World, simulate


@pytest.mark.parametrize("topology", ["halka", "yol", "tam"])
def test_formation_error_decays_at_the_rate_given_by_the_laplacian_spectrum(topology):
    r = scenarios.run_formasyon(n=8, shape="cember", topology=topology, seed=3)
    s = r.summary
    assert s["olculen_hiz"] == pytest.approx(s["ongoru_hiz"], rel=2e-3)
    assert s["son_hata_m"] < 1e-3


def test_better_connected_graphs_converge_faster():
    rate = {t: scenarios.run_formasyon(n=8, topology=t, seed=3).summary["ongoru_hiz"] for t in ("yol", "halka", "tam")}
    assert rate["yol"] < rate["halka"] < rate["tam"]


def test_formation_matches_the_target_shape():
    r = scenarios.run_formasyon(n=7, shape="v", topology="tam", seed=1)
    offsets, final = r.info["offsets"], r.traj.pos[-1]
    assert np.abs(final - offsets).max() < 1e-3          # centre at the origin, leader at the tip


def test_formation_follows_a_moving_reference():
    r = scenarios.run_formasyon(n=6, shape="cizgi", topology="tam", seed=2, v_ref=(1.0, 0.5), duration=120.0)
    t = r.traj.t[-1]
    assert np.abs(r.traj.pos[-1] - r.info["offsets"] - np.array([1.0, 0.5]) * t).max() < 1e-2


def test_pinned_formation_offset_in_wind_matches_the_linear_solution():
    params = DroneParams(drag=0.1)
    wind = np.array([3.0, -2.0])
    n, gain = 6, 0.8
    offsets = scenarios.formation_offsets("cizgi", n)
    ctrl = Formation(offsets, graph.ring_adjacency(n), params, gain=gain, leader=0)
    tr = simulate(np.random.default_rng(1).uniform(-20, 20, (n, 2)), np.zeros((n, 2)), ctrl, params,
                  World(wind=wind), 400.0, 0.05, record_every=20)
    predicted = np.linalg.solve(gain * ctrl.matrix, np.outer(np.ones(n), params.tau * params.drag * wind))
    assert np.abs(ctrl.error(tr.t[-1], tr.pos[-1]) - predicted).max() < 1e-6


def test_unpinned_formation_conserves_the_mean_offset_until_the_limits_bite():
    """Without a pinned leader sum(v_des) = 0, so the mean of x - offsets is conserved while no limit is active."""
    params = DroneParams(drag=0.0)
    n = 6
    offsets = scenarios.formation_offsets("cizgi", n)
    ctrl = Formation(offsets, graph.ring_adjacency(n), params, leader=None)
    rng = np.random.default_rng(0)
    small = offsets + rng.uniform(-1.0, 1.0, (n, 2))
    tr = simulate(small, np.zeros((n, 2)), ctrl, params, World(), 60.0, 0.05)
    assert np.allclose((tr.pos[-1] - offsets).mean(axis=0), (small - offsets).mean(axis=0), atol=1e-9)
    assert metrics.rms_series(tr.pos[-1:] - offsets - (tr.pos[-1] - offsets).mean(axis=0))[0] < 1e-3
    # a big initial error saturates the speed limit drone by drone, and the conservation breaks
    big = rng.uniform(-20, 20, (n, 2))
    tr = simulate(big, np.zeros((n, 2)), ctrl, params, World(), 60.0, 0.05)
    assert np.abs((tr.pos[-1] - offsets).mean(axis=0) - (big - offsets).mean(axis=0)).max() > 0.5


def test_goal_seeker_arrival_and_preferred_velocity_cap():
    params = DroneParams()
    ctrl = GoalSeeker([[40.0, 0.0]], params, avoid=False)
    assert np.linalg.norm(ctrl.preferred(np.zeros((1, 2)))) == pytest.approx(ctrl.cruise)
    assert np.linalg.norm(ctrl.preferred(np.array([[39.0, 0.0]]))) == pytest.approx(1.0)      # slows near the goal
    tr = simulate(np.zeros((1, 2)), np.zeros((1, 2)), ctrl, params, World(), 25.0, 0.05)
    assert np.linalg.norm(tr.pos[-1, 0] - [40, 0]) < 0.1


def test_alignment_rule_drives_a_flock_to_a_common_heading():
    params = DroneParams(drag=0.0)
    rng = np.random.default_rng(1)
    n = 20
    pos0 = scenarios._spread_points(rng, n, 0.0, 10.0, 1.5)
    vel0 = rng.normal(size=(n, 2)) * 2.0
    tr = simulate(pos0, vel0, Boids(params, goal=None), params, World(), 40.0, 0.05, record_every=4)
    assert metrics.order_parameter(tr.vel[0]) < 0.5
    assert metrics.order_parameter(tr.vel[-1]) > 0.99


def test_separation_rule_pushes_close_boids_apart():
    params = DroneParams(drag=0.0)
    ctrl = Boids(params, goal=None)
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    acc = ctrl(0.0, pos, np.zeros((2, 2)), np.ones(2, dtype=bool))
    assert acc[0, 0] < 0 < acc[1, 0] and acc[0, 0] == pytest.approx(-acc[1, 0])
    far = ctrl(0.0, np.array([[0.0, 0.0], [10.0, 0.0]]), np.zeros((2, 2)), np.ones(2, dtype=bool))
    assert np.abs(far[:, 0]).max() < np.abs(acc[:, 0]).max()


def test_obstacle_repulsion_points_away_and_vanishes_far_away():
    params = DroneParams(drag=0.0)
    ctrl = Boids(params, goal=None, obstacles=[[0.0, 0.0, 2.0]])
    near = ctrl(0.0, np.array([[4.0, 0.0]]), np.zeros((1, 2)), np.ones(1, dtype=bool))
    far = ctrl(0.0, np.array([[30.0, 0.0]]), np.zeros((1, 2)), np.ones(1, dtype=bool))
    assert near[0, 0] > 0 and np.allclose(far, 0)


def test_surusu_flock_clears_obstacles_and_reaches_the_goal():
    s = scenarios.run_surusu(n=30, seed=0).summary
    assert s["en_yakin_engel_m"] > 0 and s["hedefe_yakin"] == 30
    assert s["ort_duzen"] > 0.8 and s["carpisan_cift"] == 0
