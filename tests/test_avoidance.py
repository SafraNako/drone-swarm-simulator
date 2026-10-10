import numpy as np
import pytest

from drone_suru_simulatoru import avoidance
from drone_suru_simulatoru.dynamics import DroneParams


def test_head_on_time_to_collision_closed_form():
    # separation 10 m closing at 2 m/s, contact radius 1 m: |10 - 2t| = 1 -> t = 4.5 s
    assert avoidance.time_to_collision([10.0, 0.0], [2.0, 0.0], 1.0) == pytest.approx(4.5)


def test_time_to_collision_special_cases():
    ttc = avoidance.time_to_collision
    assert ttc([10.0, 0.0], [-2.0, 0.0], 1.0) == np.inf          # moving apart
    assert ttc([10.0, 0.0], [0.0, 0.0], 1.0) == np.inf           # no relative motion
    assert ttc([10.0, 5.0], [2.0, 0.0], 1.0) == np.inf           # passes 5 m to the side
    assert ttc([0.5, 0.0], [-1.0, 0.0], 1.0) == 0.0              # already overlapping
    assert ttc([10.0, 1.0], [2.0, 0.0], 1.0) == pytest.approx(5.0)   # grazing: touches at x = 0
    assert ttc([10.0, 1.01], [2.0, 0.0], 1.0) == np.inf


def test_time_to_collision_matches_time_stepping():
    rng = np.random.default_rng(3)
    dt = 1e-3
    t = np.arange(0, 12, dt)
    checked = 0
    for _ in range(150):
        p, u, r = rng.uniform(-15, 15, 2), rng.uniform(-6, 6, 2), rng.uniform(0.5, 3)
        exact = float(avoidance.time_to_collision(p, u, r))
        inside = np.linalg.norm(p[None] - t[:, None] * u[None], axis=1) <= r
        if exact < 12 - dt:
            assert inside.any() and abs(t[np.argmax(inside)] - exact) <= dt + 1e-12
            checked += 1
        elif exact > 12 + dt:
            assert not inside.any()
    assert checked >= 5


def test_time_to_collision_broadcasts():
    p = np.array([[10.0, 0.0], [10.0, 5.0]])
    assert np.allclose(avoidance.time_to_collision(p, [2.0, 0.0], 1.0), [4.5, np.inf])


def _choose(pos, vel, pref, **kw):
    pos, vel, pref = (np.array(a, dtype=float) for a in (pos, vel, pref))
    return avoidance.choose_velocities(pos, vel, pref, np.ones(len(pos), dtype=bool), DroneParams(), **kw)


def test_no_neighbours_keeps_the_preferred_velocity():
    chosen = _choose([[0, 0]], [[0, 0]], [[3, 0]])
    assert np.allclose(chosen, [[3, 0]])


def test_head_on_pair_deviates_sideways_and_the_chosen_velocity_is_safe():
    pos = [[0.0, 0.0], [12.0, 0.0]]
    vel = [[3.0, 0.0], [-3.0, 0.0]]
    chosen = _choose(pos, vel, vel)
    assert abs(chosen[0, 1]) > 0.1 or np.linalg.norm(chosen[0]) < 2.9    # it changes course or slows down
    rel_p = np.array(pos[1]) - np.array(pos[0])
    ttc = avoidance.time_to_collision(rel_p, chosen[0] - np.array(vel[1]), 2 * 0.3 + 0.6)
    assert ttc > 6.0                                                    # no collision within the horizon


def test_dead_drones_are_ignored_and_keep_their_preference():
    pos = np.array([[0.0, 0.0], [6.0, 0.0]])
    vel = np.array([[3.0, 0.0], [-3.0, 0.0]])
    alive = np.array([True, False])
    chosen = avoidance.choose_velocities(pos, vel, vel, alive, DroneParams())
    assert np.allclose(chosen[0], vel[0]) and np.allclose(chosen[1], vel[1])


def test_static_obstacle_in_the_way_forces_a_detour():
    chosen = _choose([[0.0, 0.0]], [[3.0, 0.0]], [[3.0, 0.0]], obstacles=[[10.0, 0.0, 2.0]])
    assert np.linalg.norm(chosen[0] - [3.0, 0.0]) > 0.5
    ttc = avoidance.time_to_collision([10.0, 0.0], chosen[0], 2.0 + 0.3 + 0.6)
    assert ttc > 6.0


def test_reciprocal_mode_changes_the_choice():
    pos = [[0.0, 0.0], [12.0, 0.5]]
    vel = [[3.0, 0.0], [-3.0, 0.0]]
    a = _choose(pos, vel, vel, reciprocal=False)
    b = _choose(pos, vel, vel, reciprocal=True)
    assert not np.allclose(a, b)
