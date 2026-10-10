"""Collision avoidance by sampled (reciprocal) velocity obstacles.

Every drone looks at a set of candidate velocities and picks the one closest to its preferred
velocity among those that do not lead to a collision within ``horizon`` seconds, with a soft
penalty ``weight / time_to_collision`` for those that do. The time to collision is exact (a
quadratic), so it can be checked against brute-force time stepping.
"""
from __future__ import annotations

import numpy as np


def time_to_collision(p, u, radius):
    """Earliest ``t >= 0`` with ``|p - u t| <= radius``; ``inf`` if never, ``0`` if already inside.

    ``p`` is the other body's position relative to me and ``u`` my velocity relative to it, so the
    separation vector at time ``t`` is ``p - u t``. Broadcasts over leading dimensions.
    """
    p = np.asarray(p, dtype=float)
    u = np.asarray(u, dtype=float)
    a = (u * u).sum(-1)
    b = (p * u).sum(-1)
    c = (p * p).sum(-1) - np.asarray(radius, dtype=float) ** 2
    disc = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (b - np.sqrt(np.maximum(disc, 0.0))) / a
    never = (a < 1e-12) | (b <= 0.0) | (disc < 0.0)
    t = np.where(never, np.inf, t)
    return np.where(c <= 0.0, 0.0, t)


def _candidates(v_pref, vel, v_max, n_angles, speed_fractions):
    """Candidate velocities per drone: a polar grid plus the preferred and current velocity."""
    n = len(v_pref)
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)               # (A, 2)
    grid = (np.asarray(speed_fractions)[:, None, None] * v_max * unit[None]).reshape(-1, 2)   # (F*A, 2)
    grid = np.broadcast_to(grid, (n,) + grid.shape)
    extra = np.stack([v_pref, vel, np.zeros_like(vel)], axis=1)              # (N, 3, 2)
    return np.concatenate([extra, grid], axis=1)                             # (N, S, 2)


def choose_velocities(pos, vel, v_pref, alive, params, obstacles=None, horizon=6.0, margin=0.6,
                      weight=100.0, sense_range=25.0, reciprocal=False, n_angles=24,
                      speed_fractions=(0.25, 0.5, 0.75, 1.0)):
    """Pick a velocity for every drone; dead drones keep ``v_pref``."""
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    v_pref = np.asarray(v_pref, dtype=float)
    n = len(pos)
    cand = _candidates(v_pref, vel, params.v_max, n_angles, speed_fractions)          # (N, S, 2)
    cost = np.linalg.norm(cand - v_pref[:, None, :], axis=-1)                         # (N, S)

    inv_ttc = np.zeros(cand.shape[:2])
    # other drones
    rel_pos = pos[None, :, :] - pos[:, None, :]                                       # (N, N, 2): j relative to i
    dist = np.linalg.norm(rel_pos, axis=-1)
    near = (dist < sense_range) & ~np.eye(n, dtype=bool) & alive[None, :] & alive[:, None]
    if near.any():
        if reciprocal:    # each drone takes half of the avoidance: relative velocity 2c - v_i - v_j
            u = 2.0 * cand[:, :, None, :] - vel[:, None, None, :] - vel[None, None, :, :]
        else:
            u = cand[:, :, None, :] - vel[None, None, :, :]
        ttc = time_to_collision(rel_pos[:, None, :, :], u, 2.0 * params.radius + margin)   # (N, S, N)
        ttc = np.where(near[:, None, :], ttc, np.inf)
        worst = ttc.min(axis=-1)
        inv_ttc = np.maximum(inv_ttc, np.where(worst < horizon, 1.0 / np.maximum(worst, 0.05), 0.0))
    # static circular obstacles (x, y, r)
    if obstacles is not None and len(obstacles):
        obstacles = np.asarray(obstacles, dtype=float)
        rel = obstacles[None, :, :2] - pos[:, None, :]                                # (N, K, 2)
        ttc = time_to_collision(rel[:, None, :, :], cand[:, :, None, :],
                                obstacles[None, None, :, 2] + params.radius + margin)  # (N, S, K)
        worst = ttc.min(axis=-1)
        inv_ttc = np.maximum(inv_ttc, np.where(worst < horizon, 1.0 / np.maximum(worst, 0.05), 0.0))

    best = np.argmin(cost + weight * inv_ttc, axis=1)
    chosen = cand[np.arange(n), best]
    return np.where(alive[:, None], chosen, v_pref)
