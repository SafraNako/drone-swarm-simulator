"""Controllers. Each is a callable ``(t, pos, vel, alive) -> commanded acceleration (N, 2)``."""
from __future__ import annotations

import numpy as np

from .avoidance import choose_velocities
from .dynamics import DroneParams, clip_norm, track_velocity
from .graph import laplacian, pair_distances


class GoalSeeker:
    """Fly each drone to its own goal; optionally with velocity-obstacle avoidance of drones and obstacles."""

    def __init__(self, goals, params: DroneParams, cruise=None, avoid=True, obstacles=None, k_goal=1.0,
                 **avoid_kwargs):
        self.goals = np.asarray(goals, dtype=float)
        self.params = params
        self.cruise = float(cruise if cruise is not None else 0.5 * params.v_max)
        self.avoid = avoid
        self.obstacles = obstacles
        self.k_goal = k_goal
        self.avoid_kwargs = avoid_kwargs

    def preferred(self, pos):
        return clip_norm(self.k_goal * (self.goals - pos), self.cruise)

    def __call__(self, t, pos, vel, alive):
        v_pref = self.preferred(pos)
        if self.avoid:
            v_des = choose_velocities(pos, vel, v_pref, alive, self.params, obstacles=self.obstacles,
                                      **self.avoid_kwargs)
        else:
            v_des = v_pref
        return track_velocity(v_des, vel, self.params)


class Formation:
    """Consensus formation control with a leader pinned to a moving reference point.

    The error is ``e = x - offsets - c(t)`` with ``c(t) = centre0 + v_ref t``. The command is
    ``v_des = v_ref - gain (L + B) e`` through the velocity loop, so ``e`` obeys a linear system
    whose rate follows from the spectrum of ``L + B``.
    """

    def __init__(self, offsets, adj, params: DroneParams, gain=0.8, leader=0, pin_gain=1.0,
                 centre0=(0.0, 0.0), v_ref=(0.0, 0.0), avoid=False, obstacles=None):
        self.avoid, self.obstacles = avoid, obstacles
        self.offsets = np.asarray(offsets, dtype=float)
        self.adj = np.asarray(adj, dtype=bool)
        self.params = params
        self.gain = gain
        self.centre0 = np.asarray(centre0, dtype=float)
        self.v_ref = np.asarray(v_ref, dtype=float)
        b = np.zeros(len(self.offsets))
        if leader is not None:
            b[leader] = pin_gain
        self.matrix = laplacian(self.adj) + np.diag(b)

    def error(self, t, pos):
        return pos - self.offsets - (self.centre0 + self.v_ref * t)

    def __call__(self, t, pos, vel, alive):
        e = self.error(t, pos)
        v_des = self.v_ref - self.gain * (self.matrix @ e)
        if self.avoid:    # the formation command becomes the preferred velocity of the avoidance layer
            v_des = choose_velocities(pos, vel, clip_norm(v_des, 0.5 * self.params.v_max), alive,
                                      self.params, obstacles=self.obstacles)
        return track_velocity(v_des, vel, self.params)


class Boids:
    """Reynolds flocking with a goal and circular obstacles, written as accelerations (m/s^2)."""

    def __init__(self, params: DroneParams, goal=None, cruise=None, obstacles=None, r_sense=12.0,
                 r_sep=3.0, k_sep=12.0, k_ali=1.5, k_coh=0.15, k_goal=1.2, k_obs=25.0, d_obs=6.0):
        self.params = params
        self.goal = None if goal is None else np.asarray(goal, dtype=float)
        self.cruise = float(cruise if cruise is not None else 0.6 * params.v_max)
        self.obstacles = np.zeros((0, 3)) if obstacles is None else np.asarray(obstacles, dtype=float)
        self.r_sense, self.r_sep = r_sense, r_sep
        self.k_sep, self.k_ali, self.k_coh, self.k_goal = k_sep, k_ali, k_coh, k_goal
        self.k_obs, self.d_obs = k_obs, d_obs

    def __call__(self, t, pos, vel, alive):
        n = len(pos)
        dist = pair_distances(pos)
        mask = (dist < self.r_sense) & ~np.eye(n, dtype=bool) & alive[None, :]
        count = mask.sum(axis=1)
        have = count > 0
        safe = np.maximum(count, 1)[:, None]
        mean_vel = (mask @ vel) / safe
        centroid = (mask @ pos) / safe
        acc = np.where(have[:, None], self.k_ali * (mean_vel - vel) + self.k_coh * (centroid - pos), 0.0)

        close = mask & (dist < self.r_sep)
        d = np.maximum(dist, 1e-6)
        mag = np.where(close, 1.0 / d - 1.0 / self.r_sep, 0.0) / d            # (N, N)
        away = (mag[:, :, None] * (pos[:, None, :] - pos[None, :, :])).sum(axis=1)
        acc = acc + self.k_sep * away

        if self.goal is not None:
            to_goal = self.goal - pos
            dg = np.linalg.norm(to_goal, axis=1, keepdims=True)
            v_goal = to_goal / np.maximum(dg, 1e-9) * np.minimum(self.cruise, dg)
            acc = acc + self.k_goal * (v_goal - vel)

        for ox, oy, orad in self.obstacles:
            rel = pos - np.array([ox, oy])
            centre = np.maximum(np.linalg.norm(rel, axis=1), 1e-9)
            gap = np.maximum(centre - orad - self.params.radius, 0.05)
            push = np.where(gap < self.d_obs, 1.0 / gap - 1.0 / self.d_obs, 0.0)
            acc = acc + self.k_obs * (push / centre)[:, None] * rel
        return acc + self.params.drag * vel
