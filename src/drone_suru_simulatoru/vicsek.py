"""The Vicsek model of collective motion (Vicsek et al., Phys. Rev. Lett. 75, 1226, 1995).

``N`` particles with constant speed ``v0`` in a periodic square box of side ``L``. At each step a
particle takes the mean heading of all particles within ``r0`` (itself included) plus uniform noise
in ``[-eta/2, eta/2]``. The order parameter ``phi = |mean velocity| / v0`` is close to 1 for small
noise and falls to the finite-size value ``sqrt(pi) / (2 sqrt(N))`` for ``eta = 2 pi``, where every
heading is redrawn independently at every step.
"""
from __future__ import annotations

import numpy as np


def random_phase_order(n):
    """Mean order parameter of ``n`` independent uniform headings: ``sqrt(pi) / (2 sqrt(n))`` (large n)."""
    return float(np.sqrt(np.pi) / (2.0 * np.sqrt(n)))


def run(n, box, eta, steps=2000, burn_in=1000, v0=0.03, r0=1.0, seed=0, ordered_start=False):
    """Simulate and return the order parameter at every step after ``burn_in``."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, box, size=(n, 2))
    theta = np.zeros(n) if ordered_start else rng.uniform(-np.pi, np.pi, n)
    series = []
    for step in range(steps):
        diff = pos[:, None, :] - pos[None, :, :]
        diff -= box * np.round(diff / box)                       # minimum-image convention
        near = (diff ** 2).sum(-1) < r0 * r0                     # includes the particle itself
        mean_theta = np.arctan2(near @ np.sin(theta), near @ np.cos(theta))
        theta = mean_theta + eta * (rng.random(n) - 0.5)
        pos = (pos + v0 * np.stack([np.cos(theta), np.sin(theta)], axis=1)) % box
        if step >= burn_in:
            series.append(float(np.hypot(np.cos(theta).mean(), np.sin(theta).mean())))
    return np.array(series)


def sweep(n, box, etas, steps=2000, burn_in=1000, seed=0, **kw):
    """Order parameter statistics over a list of noise amplitudes.

    Returns rows ``(eta, mean phi, std phi, susceptibility)`` with susceptibility ``n * var(phi)``.
    """
    rows = []
    for k, eta in enumerate(etas):
        s = run(n, box, eta, steps=steps, burn_in=burn_in, seed=seed + k, **kw)
        rows.append((float(eta), float(s.mean()), float(s.std()), float(n * s.var())))
    return rows
