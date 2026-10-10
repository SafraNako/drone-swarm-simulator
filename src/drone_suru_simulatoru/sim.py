"""Simulation loop: a fleet of point-mass drones, a controller, obstacles, wind and failures."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import DroneParams, limit_command, limit_speed, zoh_step


@dataclass
class World:
    obstacles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # rows: x, y, r
    wind: np.ndarray = field(default_factory=lambda: np.zeros(2))             # m/s
    bounds: tuple = None                                                      # (xmin, xmax, ymin, ymax) for plots only

    def __post_init__(self):
        self.obstacles = np.asarray(self.obstacles, dtype=float).reshape(-1, 3)
        self.wind = np.asarray(self.wind, dtype=float)


@dataclass
class Trajectory:
    dt: float
    record_every: int
    t: np.ndarray            # (T,)
    pos: np.ndarray          # (T, N, 2)
    vel: np.ndarray          # (T, N, 2)
    acc: np.ndarray          # (T, N, 2) applied (limited) command
    alive: np.ndarray        # (T, N)
    min_sep: np.ndarray      # (steps + 1,) smallest centre distance between living drones, every step
    collided: set            # pairs (i, j), i < j, whose centres were ever closer than 2 * radius
    min_clearance: float     # smallest gap between a drone's body and an obstacle (m)
    params: DroneParams
    world: World
    events: list = field(default_factory=list)

    @property
    def n(self):
        return self.pos.shape[1]


def simulate(pos0, vel0, controller, params: DroneParams, world: World, duration, dt=0.05,
             kill_at=(), record_every=1):
    """Integrate ``duration`` seconds. ``controller(t, pos, vel, alive)`` returns commanded accelerations.

    ``kill_at`` is a sequence of ``(time, indices)``; killed drones stop dead (a crash or landing),
    stay where they are, and no longer take part in collision or connectivity bookkeeping.
    """
    pos = np.array(pos0, dtype=float)
    vel = np.array(vel0, dtype=float)
    n = len(pos)
    steps = int(round(duration / dt))
    alive = np.ones(n, dtype=bool)
    kills = sorted(((float(t), list(idx)) for t, idx in kill_at), key=lambda item: item[0])
    events = []

    n_rec = steps // record_every + 1
    rec = {k: np.zeros((n_rec, n, 2)) for k in ("pos", "vel", "acc")}
    rec_alive = np.zeros((n_rec, n), dtype=bool)
    rec_t = np.zeros(n_rec)
    min_sep = np.full(steps + 1, np.inf)
    collided = set()
    min_clear = np.inf
    pair_i, pair_j = np.triu_indices(n, 1)
    contact = 2.0 * params.radius

    def bookkeeping(k):
        nonlocal min_clear
        live = alive[pair_i] & alive[pair_j]
        if live.any():
            d = np.linalg.norm(pos[pair_i] - pos[pair_j], axis=1)
            d = np.where(live, d, np.inf)
            min_sep[k] = d.min()
            hit = np.flatnonzero(d < contact)
            for h in hit:
                collided.add((int(pair_i[h]), int(pair_j[h])))
        if len(world.obstacles) and alive.any():
            gap = (np.linalg.norm(pos[alive][:, None, :] - world.obstacles[None, :, :2], axis=-1)
                   - world.obstacles[None, :, 2] - params.radius)
            min_clear = min(min_clear, float(gap.min()))

    for k in range(steps + 1):
        t = k * dt
        while kills and kills[0][0] <= t + 1e-12:
            _, idx = kills.pop(0)
            for i in idx:
                if alive[i]:
                    alive[i] = False
                    vel[i] = 0.0
                    events.append((t, int(i)))
        bookkeeping(k)
        if k < steps:
            acc = limit_command(np.asarray(controller(t, pos, vel, alive), dtype=float), params)
            acc[~alive] = 0.0
        else:
            acc = np.zeros((n, 2))
        if k % record_every == 0:
            r = k // record_every
            rec["pos"][r], rec["vel"][r], rec["acc"][r] = pos, vel, acc
            rec_alive[r] = alive
            rec_t[r] = t
        if k == steps:
            break
        new_pos, new_vel = zoh_step(pos, vel, acc, params.drag, world.wind, dt)
        new_vel = limit_speed(new_vel, params)
        pos = np.where(alive[:, None], new_pos, pos)
        vel = np.where(alive[:, None], new_vel, 0.0)

    return Trajectory(dt=dt, record_every=record_every, t=rec_t, pos=rec["pos"], vel=rec["vel"],
                      acc=rec["acc"], alive=rec_alive, min_sep=min_sep, collided=collided,
                      min_clearance=float(min_clear), params=params, world=world, events=events)
