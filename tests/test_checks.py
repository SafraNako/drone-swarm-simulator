import pytest

from drone_suru_simulatoru import checks


@pytest.mark.parametrize("check", checks.ALL, ids=lambda c: c.__name__)
def test_every_model_check_passes(check):
    result = check()
    assert result.ok, result.detail
    assert result.name and result.detail


def test_the_checks_can_fail():
    """A check is only worth something if it fails when the model is wrong."""
    import numpy as np
    from drone_suru_simulatoru import dynamics

    real = dynamics.zoh_step

    def sloppy(pos, vel, acc, drag, wind, dt):      # explicit Euler instead of the closed form
        return pos + vel * dt, vel + (acc - drag * (vel - wind)) * dt

    dynamics.zoh_step = sloppy
    try:
        assert not checks.integrator_vs_rk4().ok
    finally:
        dynamics.zoh_step = real
    assert np.isfinite(checks.integrator_vs_rk4().ok)
