import numpy as np
import pytest

from drone_suru_simulatoru import dynamics
from drone_suru_simulatoru.dynamics import DroneParams
from drone_suru_simulatoru.sim import World, simulate


def rk4(pos, vel, acc, drag, wind, total, n):
    h = total / n
    f = lambda v: (v, acc - drag * (v - wind))
    for _ in range(n):
        k1 = f(vel)
        k2 = f(vel + 0.5 * h * k1[1])
        k3 = f(vel + 0.5 * h * k2[1])
        k4 = f(vel + h * k3[1])
        pos = pos + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        vel = vel + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return pos, vel


def state(seed=0, n=4):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2)), rng.normal(size=(n, 2)), rng.normal(size=(n, 2))


def test_zoh_step_matches_rk4_with_fourth_order_convergence():
    pos, vel, acc = state()
    wind = np.array([2.0, -1.0])
    exact_p, exact_v = dynamics.zoh_step(pos, vel, acc, 0.3, wind, 2.0)
    errors = []
    for n in (4, 8, 16, 32):
        p, v = rk4(pos, vel, acc, 0.3, wind, 2.0, n)
        errors.append(max(np.abs(p - exact_p).max(), np.abs(v - exact_v).max()))
    assert errors[-1] < 1e-8
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(3)]
    assert all(3.7 < o < 4.4 for o in orders)       # RK4 converges to the closed form at order four


def test_zoh_step_is_a_semigroup():
    pos, vel, acc = state(1)
    wind = np.array([0.5, 0.2])
    one = dynamics.zoh_step(pos, vel, acc, 0.4, wind, 3.0)
    p, v = pos, vel
    for _ in range(60):
        p, v = dynamics.zoh_step(p, v, acc, 0.4, wind, 0.05)
    assert np.allclose(p, one[0], atol=1e-12) and np.allclose(v, one[1], atol=1e-12)


def test_zero_drag_branch_is_the_limit_of_the_drag_branch():
    pos, vel, acc = state(2)
    wind = np.array([1.0, 1.0])
    a = dynamics.zoh_step(pos, vel, acc, 0.0, wind, 1.5)
    b = dynamics.zoh_step(pos, vel, acc, 1e-6, wind, 1.5)
    assert np.allclose(a[0], b[0], atol=1e-5) and np.allclose(a[1], b[1], atol=1e-5)
    # ballistic formulas
    assert np.allclose(a[1], vel + acc * 1.5)
    assert np.allclose(a[0], pos + vel * 1.5 + 0.5 * acc * 1.5 ** 2)


def test_drag_pulls_velocity_toward_the_wind():
    wind = np.array([3.0, 0.0])
    _, v = dynamics.zoh_step(np.zeros(2), np.zeros(2), np.zeros(2), 0.5, wind, 50.0)
    assert np.allclose(v, wind, atol=1e-9)


def test_velocity_loop_is_a_first_order_lag_up_to_the_sampling_error():
    """Continuous limit v(t) = v_des (1 - exp(-t/tau)); the held command makes the error first order in dt."""
    params = DroneParams(tau=0.5, drag=0.2, a_max=100.0, v_max=100.0)
    target = np.array([[3.0, -1.0]])
    ctrl = lambda t, pos, vel, alive: dynamics.track_velocity(target, vel, params)
    errors = []
    for dt in (0.04, 0.02, 0.01):
        tr = simulate(np.zeros((1, 2)), np.zeros((1, 2)), ctrl, params, World(), 4.0, dt)
        expected = target[0] * (1.0 - np.exp(-tr.t / params.tau))[:, None]
        errors.append(np.abs(tr.vel[:, 0] - expected).max())
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.1)
    assert errors[1] / errors[2] == pytest.approx(2.0, rel=0.1)
    assert errors[2] < 0.011                    # about v_des * dt / (2 tau) = 0.0095


def test_limits_are_enforced():
    params = DroneParams(v_max=5.0, a_max=3.0)
    tr = simulate(np.zeros((3, 2)), np.zeros((3, 2)), lambda t, p, v, a: np.tile([100.0, -80.0], (3, 1)),
                  params, World(), 10.0, 0.05)
    assert np.linalg.norm(tr.acc, axis=2).max() <= 3.0 + 1e-9
    assert np.linalg.norm(tr.vel, axis=2).max() <= 5.0 + 1e-9
    assert np.linalg.norm(tr.vel[-1], axis=1) == pytest.approx(5.0)       # it does reach the limit


def test_hovering_in_wind_settles_at_tau_drag_wind():
    params = DroneParams()
    wind = np.array([3.0, -2.0])
    hold = lambda t, p, v, a: dynamics.track_velocity(np.zeros_like(p), v, params)
    tr = simulate(np.zeros((1, 2)), np.zeros((1, 2)), hold, params, World(wind=wind), 40.0, 0.05)
    assert np.allclose(tr.vel[-1, 0], params.tau * params.drag * wind, atol=1e-9)


def test_clip_norm_scales_only_long_rows():
    out = dynamics.clip_norm(np.array([[3.0, 4.0], [0.3, 0.4]]), 1.0)
    assert np.allclose(out, [[0.6, 0.8], [0.3, 0.4]])


def test_stopping_distance():
    params = DroneParams(a_max=4.0)
    assert dynamics.stopping_distance(8.0, params) == pytest.approx(8.0)


@pytest.mark.parametrize("field", ["v_max", "a_max", "tau", "radius"])
def test_params_must_be_positive(field):
    with pytest.raises(ValueError):
        DroneParams(**{field: 0.0})
    with pytest.raises(ValueError):
        DroneParams(drag=-0.1)
