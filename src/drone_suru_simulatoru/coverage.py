"""Area coverage by Lloyd's algorithm on a grid (Voronoi / centroidal partition).

The locational cost ``H(p) = integral of density(q) * min_i |q - p_i|^2 dq`` is minimised when every
drone sits at the centroid of its own Voronoi cell. Each Lloyd step (move to the centroid) cannot
increase ``H``; the cost is evaluated on a cell-centred grid of weights.
"""
from __future__ import annotations

import numpy as np

from .avoidance import choose_velocities
from .dynamics import DroneParams, clip_norm, track_velocity


def make_grid(width, height, nx, ny, density=None):
    """Cell-centred grid over ``[0, width] x [0, height]``; returns ``(points, weights)``.

    ``density(x, y)`` defaults to 1. Weights are ``density * cell area`` (so they sum to the mass).
    """
    xs = (np.arange(nx) + 0.5) * width / nx
    ys = (np.arange(ny) + 0.5) * height / ny
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    cell = (width / nx) * (height / ny)
    w = np.full(len(pts), cell) if density is None else density(pts[:, 0], pts[:, 1]) * cell
    return pts, w


def gaussian_density(spots, base=0.05):
    """Mixture of isotropic Gaussians ``(cx, cy, sigma, amplitude)`` on a floor ``base``."""
    def density(x, y):
        out = np.full_like(x, base, dtype=float)
        for cx, cy, sigma, amp in spots:
            out += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
        return out
    return density


def assign_cells(pos, pts):
    """Index of the nearest drone for every grid point, and the squared distance to it."""
    d2 = ((pts[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    idx = d2.argmin(axis=1)
    return idx, d2[np.arange(len(pts)), idx]


def locational_cost(pos, pts, w):
    return float((w * assign_cells(pos, pts)[1]).sum())


def centroids(pos, pts, w):
    """Mass centroid of every drone's cell (a drone with an empty cell keeps its own position)."""
    idx, _ = assign_cells(pos, pts)
    n = len(pos)
    mass = np.bincount(idx, weights=w, minlength=n)
    cx = np.bincount(idx, weights=w * pts[:, 0], minlength=n)
    cy = np.bincount(idx, weights=w * pts[:, 1], minlength=n)
    cen = np.where(mass[:, None] > 0, np.stack([cx, cy], 1) / np.maximum(mass, 1e-300)[:, None], pos)
    return cen, mass


def lloyd(pos, pts, w, iterations=60):
    """Plain Lloyd iteration; returns the positions and the cost history (``iterations + 1`` values)."""
    pos = np.array(pos, dtype=float)
    history = [locational_cost(pos, pts, w)]
    for _ in range(iterations):
        pos, _ = centroids(pos, pts, w)
        history.append(locational_cost(pos, pts, w))
    return pos, np.array(history)


def hexagon_bound(area, n, density=1.0):
    """Fejes Toth's lower bound ``5 A^2 / (18 sqrt(3) N)`` of ``H`` for unit density on a convex
    polygon with at most six sides, attained asymptotically by the hexagonal lattice; scaled by ``density``."""
    return density * 5.0 * area ** 2 / (18.0 * np.sqrt(3.0) * n)


class CoverageController:
    """Each drone flies toward the centroid of its cell (Cortes et al.): ``v = gain (c_i - p_i)``."""

    def __init__(self, pts, w, params: DroneParams, gain=0.6, cruise=None, avoid=False):
        self.pts, self.w, self.params, self.gain, self.avoid = pts, w, params, gain, avoid
        self.cruise = float(cruise if cruise is not None else 0.5 * params.v_max)

    def __call__(self, t, pos, vel, alive):
        v_des = np.zeros_like(pos)
        if alive.any():
            cen, _ = centroids(pos[alive], self.pts, self.w)
            v_des[alive] = clip_norm(self.gain * (cen - pos[alive]), self.cruise)
        if self.avoid:
            v_des = choose_velocities(pos, vel, v_des, alive, self.params)
        return track_velocity(v_des, vel, self.params)
