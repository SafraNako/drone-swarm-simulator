"""Task allocation: assign drones to targets at minimum total cost.

* ``auction``  - Bertsekas' auction algorithm with epsilon scaling. Distributed in spirit (each
  drone bids for its best target), and guaranteed to finish within ``n * eps`` of the optimum.
* ``greedy``   - repeatedly take the cheapest remaining drone-target pair.
* ``exact``    - dynamic programming over subsets of targets (optimal, ``O(2^n n)``, n <= 14).
* ``brute_force`` - every permutation (optimal, n <= 8); the independent check of ``exact``.

Cost matrices are ``(drones, targets)`` with at least as many drones as targets; the missing
targets are padded with zero-cost dummies, which is the same as leaving the surplus drones idle.
Every function returns ``target_of_drone``: the target index of each drone, or -1 for an idle drone.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np


def distance_cost(drones, targets, power=1.0):
    d = np.linalg.norm(np.asarray(drones)[:, None, :] - np.asarray(targets)[None, :, :], axis=2)
    return d ** power


def _square(cost):
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    if m > n:
        raise ValueError("more targets than drones")
    if m < n:
        cost = np.hstack([cost, np.zeros((n, n - m))])
    return cost, m


def _real(col, m):
    return np.where(col < m, col, -1)


def total_cost(cost, target_of_drone):
    cost = np.asarray(cost, dtype=float)
    chosen = np.asarray(target_of_drone)
    rows = np.flatnonzero(chosen >= 0)
    if len(set(chosen[rows].tolist())) != len(rows):
        raise ValueError("a target is assigned twice")
    return float(cost[rows, chosen[rows]].sum())


def auction(cost, eps_final=1e-6, factor=5.0):
    """Returns ``(target_of_drone, bids)``; the cost is within ``n * eps_final`` of the optimum."""
    square, m = _square(cost)
    n = square.shape[0]
    if n == 1:
        return _real(np.array([0]), m), 0
    benefit = -square
    prices = np.zeros(n)
    spread = float(benefit.max() - benefit.min())
    eps = max(spread / 2.0, eps_final)
    bids = 0
    while True:
        owner = -np.ones(n, dtype=int)     # owner[j]: drone currently holding target j
        held = -np.ones(n, dtype=int)      # held[i]: target currently held by drone i
        free = list(range(n))
        while free:
            i = free.pop()
            values = benefit[i] - prices
            j = int(np.argmax(values))
            best = values[j]
            values[j] = -np.inf
            second = values.max()
            prices[j] += best - second + eps
            previous = owner[j]
            if previous >= 0:
                held[previous] = -1
                free.append(previous)
            owner[j], held[i] = i, j
            bids += 1
        if eps <= eps_final:
            break
        eps = max(eps / factor, eps_final)
    return _real(held, m), bids


def greedy(cost):
    square, m = _square(cost)
    n = square.shape[0]
    order = np.argsort(square[:, :m].ravel(), kind="stable") if m else []
    held = -np.ones(n, dtype=int)
    taken = np.zeros(m, dtype=bool)
    for flat in order:
        i, j = divmod(int(flat), m)
        if held[i] < 0 and not taken[j]:
            held[i], taken[j] = j, True
    return held


def exact(cost):
    """Optimal assignment by dynamic programming over subsets (n <= 14)."""
    square, m = _square(cost)
    n = square.shape[0]
    if n > 14:
        raise ValueError("exact solver limited to 14 drones")
    full = 1 << n
    dp = np.full(full, np.inf)
    pick = -np.ones(full, dtype=int)
    dp[0] = 0.0
    for mask in range(full):
        if not np.isfinite(dp[mask]):
            continue
        row = bin(mask).count("1")
        if row == n:
            continue
        for j in range(n):
            if not mask >> j & 1:
                new = mask | 1 << j
                value = dp[mask] + square[row, j]
                if value < dp[new]:
                    dp[new], pick[new] = value, j
    held = -np.ones(n, dtype=int)
    mask = full - 1
    for row in range(n - 1, -1, -1):
        j = pick[mask]
        held[row] = j
        mask ^= 1 << j
    return _real(held, m)


def brute_force(cost):
    square, m = _square(cost)
    n = square.shape[0]
    if n > 8:
        raise ValueError("brute force limited to 8 drones")
    best, best_perm = np.inf, None
    for perm in permutations(range(n)):
        value = sum(square[i, j] for i, j in enumerate(perm))
        if value < best:
            best, best_perm = value, perm
    return _real(np.array(best_perm), m)
