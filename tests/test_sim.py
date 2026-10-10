import numpy as np
import pytest

from drone_suru_simulatoru.control import GoalSeeker
from drone_suru_simulatoru.dynamics import DroneParams, track_velocity
from drone_suru_simulatoru.sim import World, simulate

P = DroneParams()


def cruise_to(goals, speed=3.0):
    goals = np.asarray(goals, dtype=float)

    def ctrl(t, pos, vel, alive):
        d = goals - pos
        n = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        return track_velocity(d / n * speed, vel, P)
    return ctrl


def test_recording_shapes_and_times():
    tr = simulate(np.zeros((3, 2)), np.zeros((3, 2)), lambda t, p, v, a: np.zeros((3, 2)), P, World(), 2.0, 0.1,
                  record_every=4)
    assert tr.pos.shape == (6, 3, 2) and len(tr.t) == 6
    assert np.allclose(tr.t, np.arange(6) * 0.4) and len(tr.min_sep) == 21 and tr.n == 3


def test_head_on_drones_without_avoidance_collide_and_min_sep_is_measured():
    pos0 = np.array([[0.0, 0.0], [20.0, 0.0]])
    tr = simulate(pos0, np.zeros((2, 2)), cruise_to([[20.0, 0.0], [0.0, 0.0]]), P, World(), 12.0, 0.01)
    assert tr.collided == {(0, 1)}
    recomputed = np.linalg.norm(tr.pos[:, 0] - tr.pos[:, 1], axis=1).min()
    assert tr.min_sep.min() == pytest.approx(recomputed, abs=1e-12)
    assert tr.min_sep.min() < 2 * P.radius


def test_goal_seeker_with_avoidance_passes_head_on_drones_safely():
    pos0 = np.array([[0.0, 0.0], [20.0, 0.0]])
    goals = np.array([[20.0, 0.0], [0.0, 0.0]])
    tr = simulate(pos0, np.zeros((2, 2)), GoalSeeker(goals, P), P, World(), 20.0, 0.05)
    assert not tr.collided and tr.min_sep.min() > 2 * P.radius
    assert np.linalg.norm(tr.pos[-1] - goals, axis=1).max() < 0.5


def test_killed_drones_stop_and_are_excluded_from_collisions():
    pos0 = np.array([[0.0, 0.0], [20.0, 0.0]])
    ctrl = cruise_to([[20.0, 0.0], [0.0, 0.0]])
    tr = simulate(pos0, np.zeros((2, 2)), ctrl, P, World(), 12.0, 0.01, kill_at=[(1.0, [1])])
    assert tr.events == [(1.0, 1)] and not tr.alive[-1, 1] and tr.alive[-1, 0]
    assert np.allclose(tr.vel[-1, 1], 0) and np.allclose(tr.pos[-1, 1], tr.pos[200, 1])    # frozen where it fell
    assert not tr.collided                      # the dead drone is not an obstacle for the collision count
    assert tr.pos[-1, 0, 0] > 15                # the survivor flew through it


def test_obstacle_clearance_is_measured():
    world = World(obstacles=[[10.0, 0.0, 2.0]])
    tr = simulate(np.zeros((1, 2)), np.zeros((1, 2)), cruise_to([[20.0, 0.0]]), P, world, 8.0, 0.01)
    assert tr.min_clearance < 0           # flew straight into it
    dist = np.linalg.norm(tr.pos[:, 0] - [10.0, 0.0], axis=1) - 2.0 - P.radius
    assert tr.min_clearance == pytest.approx(dist.min(), abs=1e-9)


def test_world_wind_blows_a_passive_drone():
    """Passive drone: v = w (1 - exp(-k t)), x = w (t - (1 - exp(-k t)) / k)."""
    total, k = 60.0, P.drag
    tr = simulate(np.zeros((1, 2)), np.zeros((1, 2)), lambda t, p, v, a: np.zeros((1, 2)), P,
                  World(wind=[2.0, 0.0]), total, 0.1)
    decay = np.exp(-k * total)
    assert tr.vel[-1, 0, 0] == pytest.approx(2.0 * (1 - decay), abs=1e-9)
    assert tr.pos[-1, 0, 0] == pytest.approx(2.0 * (total - (1 - decay) / k), abs=1e-7)


def test_determinism_and_inputs_not_mutated():
    pos0 = np.array([[0.0, 0.0], [20.0, 0.0]])
    keep = pos0.copy()
    goals = np.array([[20.0, 0.0], [0.0, 0.0]])
    a = simulate(pos0, np.zeros((2, 2)), GoalSeeker(goals, P), P, World(), 5.0, 0.05)
    b = simulate(pos0, np.zeros((2, 2)), GoalSeeker(goals, P), P, World(), 5.0, 0.05)
    assert np.array_equal(a.pos, b.pos) and np.array_equal(pos0, keep)
