"""Point-mass drone dynamics with exact integration of a held command.

Each drone is a point mass with linear drag toward the air velocity ``wind``::

    dv/dt = a_cmd - drag * (v - wind)         (a_cmd held constant over a step)

For a constant ``a_cmd`` this linear ODE has a closed-form solution, so a step of any length is
exact (no integration error) and ``n`` small steps equal one large step. Saturation of the
command and of the speed is applied around the exact step, not inside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DroneParams:
    v_max: float = 8.0     # m/s, ground-speed limit
    a_max: float = 6.0     # m/s^2, commanded-acceleration limit
    tau: float = 0.5       # s, time constant of the velocity loop
    drag: float = 0.1      # 1/s, linear drag toward the wind velocity
    radius: float = 0.3    # m, body radius: two drones collide when centres are closer than 2*radius

    def __post_init__(self):
        for name in ("v_max", "a_max", "tau", "radius"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.drag < 0:
            raise ValueError("drag must be non-negative")


def zoh_step(pos, vel, acc, drag, wind, dt):
    """Exact solution of ``dv/dt = acc - drag (v - wind)`` over ``dt`` for constant ``acc``."""
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    acc = np.asarray(acc, dtype=float)
    wind = np.asarray(wind, dtype=float)
    if drag < 1e-9:
        # limit drag -> 0 (the drag term only pulls toward the wind, which is then ignored)
        new_vel = vel + acc * dt
        new_pos = pos + vel * dt + 0.5 * acc * dt * dt
        return new_pos, new_vel
    v_inf = wind + acc / drag                    # terminal velocity for this command
    decay = np.exp(-drag * dt)
    new_vel = v_inf + (vel - v_inf) * decay
    new_pos = pos + v_inf * dt + (vel - v_inf) * (1.0 - decay) / drag
    return new_pos, new_vel


def clip_norm(vectors, limit):
    """Scale rows of ``vectors`` so that no row is longer than ``limit``."""
    vectors = np.asarray(vectors, dtype=float)
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(1.0, limit / np.maximum(norm, 1e-12))
    return vectors * scale


def track_velocity(v_des, vel, params: DroneParams):
    """Acceleration command of the inner velocity loop.

    ``a = (v_des - v) / tau + drag * v``: the proportional term drives the velocity to ``v_des`` and
    the feed-forward cancels the drag, assuming no wind (the controller cannot measure it). In a
    steady wind ``w`` the velocity therefore settles at ``v_des + tau * drag * w``.
    """
    return (np.asarray(v_des, dtype=float) - vel) / params.tau + params.drag * vel


def limit_command(acc, params: DroneParams):
    return clip_norm(acc, params.a_max)


def limit_speed(vel, params: DroneParams):
    return clip_norm(vel, params.v_max)


def stopping_distance(speed, params: DroneParams):
    """Distance to stop from ``speed`` at full braking, ignoring drag."""
    return speed * speed / (2.0 * params.a_max)
