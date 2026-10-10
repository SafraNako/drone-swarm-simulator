"""Quantities measured on a trajectory."""
from __future__ import annotations

import numpy as np

from .graph import n_components, range_adjacency


def order_parameter(vel, alive=None):
    """Polarisation ``|sum v| / sum |v|``: 1 when every drone moves the same way, ~0 when directions are random."""
    vel = np.asarray(vel, dtype=float)
    if alive is not None:
        vel = vel[np.asarray(alive, dtype=bool)]
    speed = np.linalg.norm(vel, axis=1).sum()
    return float(np.linalg.norm(vel.sum(axis=0)) / speed) if speed > 1e-12 else 0.0


def cohesion_radius(pos, alive=None):
    """Root-mean-square distance from the centroid."""
    pos = np.asarray(pos, dtype=float)
    if alive is not None:
        pos = pos[np.asarray(alive, dtype=bool)]
    return float(np.sqrt(((pos - pos.mean(axis=0)) ** 2).sum(axis=1).mean()))


def path_lengths(traj):
    step = np.linalg.norm(np.diff(traj.pos, axis=0), axis=2)               # (T-1, N)
    return step.sum(axis=0)


def control_effort(traj):
    """Mean over drones of the time integral of ``|a|^2`` (m^2/s^3) from the recorded accelerations."""
    dt = traj.dt * traj.record_every
    return float((np.linalg.norm(traj.acc, axis=2) ** 2).sum(axis=0).mean() * dt)


def connected_fraction(traj, comm_range):
    """Fraction of recorded frames in which the living drones form one connected communication graph."""
    ok = 0
    for r in range(len(traj.t)):
        live = traj.alive[r]
        if live.sum() <= 1:
            ok += 1
            continue
        adj = range_adjacency(traj.pos[r][live], comm_range)
        ok += n_components(adj) == 1
    return ok / len(traj.t)


def arrival_times(traj, goals, tol=0.5):
    """First time each drone is within ``tol`` of its goal and stays there to the end (nan if never)."""
    dist = np.linalg.norm(traj.pos - np.asarray(goals)[None], axis=2)      # (T, N)
    inside = dist <= tol
    times = np.full(traj.n, np.nan)
    for i in range(traj.n):
        outside = np.flatnonzero(~inside[:, i])
        if len(outside) == 0:
            times[i] = traj.t[0]
        elif outside[-1] < len(traj.t) - 1:
            times[i] = traj.t[outside[-1] + 1]
    return times


def rms_series(error):
    """Root-mean-square over drones of a ``(T, N, 2)`` error array."""
    return np.sqrt((np.asarray(error) ** 2).sum(axis=2).mean(axis=1))


def fit_decay_rate(t, values, lo, hi):
    """Least-squares slope of ``-log(values)`` against ``t`` over the samples with ``lo < values < hi``."""
    t = np.asarray(t)
    values = np.asarray(values)
    pick = (values > lo) & (values < hi)
    if pick.sum() < 5:
        raise ValueError("not enough samples in the fitting window")
    slope = np.polyfit(t[pick], np.log(values[pick]), 1)[0]
    return float(-slope)
