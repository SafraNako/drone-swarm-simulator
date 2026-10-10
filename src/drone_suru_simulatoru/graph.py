"""Communication graphs between drones and their Laplacian spectra."""
from __future__ import annotations

import numpy as np


def pair_distances(pos):
    pos = np.asarray(pos, dtype=float)
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def range_adjacency(pos, radius, alive=None):
    """Undirected graph: drones closer than ``radius`` are connected (dead drones are isolated)."""
    dist = pair_distances(pos)
    adj = dist < radius
    np.fill_diagonal(adj, False)
    if alive is not None:
        alive = np.asarray(alive, dtype=bool)
        adj &= alive[:, None] & alive[None, :]
    return adj


def knn_adjacency(pos, k):
    """Symmetrised k-nearest-neighbour graph."""
    dist = pair_distances(pos)
    n = len(dist)
    k = min(k, n - 1)
    adj = np.zeros((n, n), dtype=bool)
    if k > 0:
        np.fill_diagonal(dist, np.inf)
        nearest = np.argsort(dist, axis=1)[:, :k]
        rows = np.repeat(np.arange(n), k)
        adj[rows, nearest.ravel()] = True
    return adj | adj.T


def ring_adjacency(n):
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = True
    if n < 3:
        np.fill_diagonal(adj, False)
    return adj


def path_adjacency(n):
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = True
    return adj


def complete_adjacency(n):
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    return adj


TOPOLOGIES = {
    "halka": ring_adjacency,
    "yol": path_adjacency,
    "tam": complete_adjacency,
}


def laplacian(adj):
    a = np.asarray(adj, dtype=float)
    return np.diag(a.sum(axis=1)) - a


def components(adj):
    """Connected components by breadth-first search; returns a label per node."""
    adj = np.asarray(adj, dtype=bool)
    n = len(adj)
    label = -np.ones(n, dtype=int)
    count = 0
    for start in range(n):
        if label[start] >= 0:
            continue
        label[start] = count
        queue = [start]
        while queue:
            node = queue.pop()
            for nxt in np.flatnonzero(adj[node]):
                if label[nxt] < 0:
                    label[nxt] = count
                    queue.append(nxt)
        count += 1
    return label


def n_components(adj):
    return int(components(adj).max() + 1) if len(adj) else 0


def laplacian_eigenvalues(adj):
    return np.linalg.eigvalsh(laplacian(adj))


def algebraic_connectivity(adj):
    """Second-smallest Laplacian eigenvalue (the Fiedler value); zero iff the graph is disconnected."""
    values = laplacian_eigenvalues(adj)
    return float(values[1]) if len(values) > 1 else 0.0


def nullity(adj, tol=1e-9):
    """Number of (numerically) zero Laplacian eigenvalues: equals the number of components."""
    return int((laplacian_eigenvalues(adj) < tol).sum())


def pinned_laplacian(adj, pinned, pin_gain=1.0):
    """``L + B`` with ``B`` diagonal: ``pin_gain`` on the pinned (leader) nodes, zero elsewhere."""
    b = np.zeros(len(adj))
    b[list(np.atleast_1d(pinned))] = pin_gain
    return laplacian(adj) + np.diag(b)


def first_order_rate(matrix, gain):
    """Decay rate of ``de/dt = -gain M e``: ``gain`` times the smallest nonzero eigenvalue of ``M``."""
    values = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    values = values[values > 1e-9]
    return float(gain * values.min()) if len(values) else 0.0


def second_order_slowest_rate(matrix, gain, tau):
    """Decay rate of the slowest mode of ``tau dv/dt = u - v``, ``u = -gain M e``, ``de/dt = v``.

    Each nonzero eigenvalue ``lam`` of ``M`` gives ``tau s^2 + s + gain lam = 0``. Returns
    ``-max Re(s)`` (positive when the formation error decays).
    """
    values = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    values = values[values > 1e-9]
    if len(values) == 0:
        return 0.0
    disc = 1.0 - 4.0 * tau * gain * values
    real = np.where(disc >= 0, (-1.0 + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * tau), -1.0 / (2.0 * tau))
    return float(-real.max())
