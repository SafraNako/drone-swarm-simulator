"""Ready-made scenarios. Each returns a :class:`Result` with the trajectory and a summary dict."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import assign, coverage, graph, metrics
from .control import Boids, Formation, GoalSeeker
from .dynamics import DroneParams
from .sim import Trajectory, World, simulate

SHAPES = ("cember", "cizgi", "v", "izgara")
DENSITIES = ("duzgun", "gauss")
ASSIGNERS = ("ihale", "acgozlu", "tam")
SCENARIOS = {
    "takas": "çember üzerindeki dronelar karşı noktaya geçer (hız engeli ile çarpışma kaçınma)",
    "surusu": "Reynolds sürüsü (ayrılma, hizalanma, bağlanma): engeller arasından hedefe",
    "formasyon": "uzlaşma tabanlı formasyon kontrolü; hata bozunması ile Laplace tayfı karşılaştırması",
    "kapsama": "Lloyd algoritmasıyla alan kapsama, isteğe bağlı drone arızası",
    "gorev": "dronelara hedef ataması (ihale algoritması) ve kaçınmalı uçuş",
}


@dataclass
class Result:
    name: str
    traj: Trajectory
    summary: dict
    info: dict = field(default_factory=dict)


def _spread_points(rng, n, low, high, min_dist, tries=20000):
    """``n`` uniform random points in ``[low, high]^2`` no closer than ``min_dist`` (rejection sampling)."""
    pts = []
    for _ in range(tries):
        cand = rng.uniform(low, high, 2)
        if all(np.linalg.norm(cand - q) >= min_dist for q in pts):
            pts.append(cand)
            if len(pts) == n:
                return np.array(pts)
    raise ValueError(f"cannot place {n} points {min_dist} m apart in the given area")


def _moving_order(traj: Trajectory, min_speed=1.0):
    """Mean polarisation over the frames in which the swarm is actually moving (mean speed above ``min_speed``)."""
    speeds = np.linalg.norm(traj.vel, axis=2).mean(axis=1)
    frames = np.flatnonzero(speeds > min_speed)
    if len(frames) == 0:
        return float("nan")
    return float(np.mean([metrics.order_parameter(traj.vel[k], traj.alive[k]) for k in frames]))


def _collision_info(traj: Trajectory):
    return {"carpisan_cift": len(traj.collided), "en_kucuk_aralik_m": float(traj.min_sep.min())}


# ----------------------------------------------------------------------------------------------
def run_takas(n=12, radius=20.0, duration=None, avoid=True, reciprocal=False, cruise=None, wind=(0.0, 0.0),
              params=None, dt=0.05, seed=0, jitter=0.0):
    """Antipodal swap: ``n`` drones on a circle each fly to the opposite point; all meet in the middle."""
    params = params or DroneParams()
    rng = np.random.default_rng(seed)
    ang = 2.0 * np.pi * np.arange(n) / n
    pos0 = radius * np.stack([np.cos(ang), np.sin(ang)], axis=1) + jitter * rng.standard_normal((n, 2))
    goals = -radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    cruise = cruise if cruise is not None else 0.5 * params.v_max
    duration = duration or (2.0 * radius / cruise * 2.5 + 10.0)
    ctrl = GoalSeeker(goals, params, cruise=cruise, avoid=avoid, reciprocal=reciprocal)
    traj = simulate(pos0, np.zeros((n, 2)), ctrl, params, World(wind=wind), duration, dt, record_every=2)
    arrive = metrics.arrival_times(traj, goals, tol=0.5)
    straight = float(np.linalg.norm(goals - pos0, axis=1).sum())
    summary = {
        "n": n, "avoid": avoid, "reciprocal": reciprocal, "cruise": cruise, "sure_s": duration,
        **_collision_info(traj),
        "varan": int(np.isfinite(arrive).sum()),
        "varis_son_s": float(np.nanmax(arrive)) if np.isfinite(arrive).any() else float("nan"),
        "yol_verimi": straight / float(metrics.path_lengths(traj).sum()),
        "kontrol_eforu": metrics.control_effort(traj),
    }
    return Result("takas", traj, summary, {"goals": goals})


# ----------------------------------------------------------------------------------------------
def run_surusu(n=30, n_obstacles=2, duration=60.0, seed=0, params=None, dt=0.05, wind=(0.0, 0.0)):
    """Boids flock from the left edge to a goal on the right with circular obstacles in between."""
    params = params or DroneParams()
    rng = np.random.default_rng(seed)
    pos0 = _spread_points(rng, n, 0.0, 16.0, 2.0) - np.array([0.0, 8.0])
    vel0 = np.zeros((n, 2))
    goal = np.array([90.0, 0.0])
    slots = [(40.0, 4.0, 6.0), (62.0, -5.0, 7.0), (25.0, -12.0, 4.0), (75.0, 12.0, 5.0)]
    obstacles = np.array(slots[:max(0, min(n_obstacles, len(slots)))]).reshape(-1, 3)
    ctrl = Boids(params, goal=goal, obstacles=obstacles)
    traj = simulate(pos0, vel0, ctrl, params, World(obstacles=obstacles, wind=wind), duration, dt,
                    record_every=2)
    final = traj.pos[-1]
    summary = {
        "n": n, "engel": len(obstacles), "sure_s": duration, **_collision_info(traj),
        "en_yakin_engel_m": traj.min_clearance if len(obstacles) else float("nan"),
        "hedefe_yakin": int((np.linalg.norm(final - goal, axis=1) < 15.0).sum()),
        "ort_duzen": _moving_order(traj),
        "son_kume_yaricapi_m": metrics.cohesion_radius(final),
    }
    return Result("surusu", traj, summary, {"goals": goal[None]})


# ----------------------------------------------------------------------------------------------
def formation_offsets(shape, n, spacing=4.0):
    """Offsets of ``n`` drones in the formation; drone 0 (the leader) is at the tip / start."""
    if shape == "cember":
        r = max(spacing * n / (2.0 * np.pi), spacing)
        a = 2.0 * np.pi * np.arange(n) / n
        return r * np.stack([np.cos(a), np.sin(a)], axis=1)
    if shape == "cizgi":
        return np.stack([(np.arange(n) - (n - 1) / 2.0) * spacing, np.zeros(n)], axis=1)
    if shape == "v":
        k = (np.arange(n) + 1) // 2
        side = np.where(np.arange(n) % 2 == 1, 1.0, -1.0) * (np.arange(n) > 0)
        return np.stack([-0.8 * spacing * k, 0.6 * spacing * k * side], axis=1)
    if shape == "izgara":
        cols = int(np.ceil(np.sqrt(n)))
        idx = np.arange(n)
        return np.stack([(idx % cols) * spacing, (idx // cols) * spacing], axis=1)
    raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")


def run_formasyon(n=8, shape="cember", topology="halka", gain=0.8, duration=None, seed=0, params=None,
                  dt=0.05, v_ref=(0.0, 0.0), wind=(0.0, 0.0), avoid=False, spacing=4.0):
    """Drones start at random places and converge to ``shape``; the leader (0) is pinned to the reference."""
    params = params or DroneParams(drag=0.0)
    rng = np.random.default_rng(seed)
    if topology == "knn":
        adj_of = lambda p: graph.knn_adjacency(p, 3)
    elif topology in graph.TOPOLOGIES:
        adj_of = lambda p: graph.TOPOLOGIES[topology](n)
    else:
        raise ValueError(f"unknown topology {topology!r}; choose from {tuple(graph.TOPOLOGIES) + ('knn',)}")
    offsets = formation_offsets(shape, n, spacing)
    pos0 = rng.uniform(-40.0, 40.0, size=(n, 2))
    adj = adj_of(offsets)     # topology fixed by the target shape, so the spectrum is known in advance
    ctrl = Formation(offsets, adj, params, gain=gain, leader=0, v_ref=v_ref, avoid=avoid)
    predicted = graph.second_order_slowest_rate(ctrl.matrix, gain, params.tau)
    if duration is None:
        e0 = float(metrics.rms_series((pos0 - offsets)[None])[0])
        duration = float(min(900.0, 1.1 * np.log(max(e0, 1.0) / 1e-4) / max(predicted, 1e-6) + 20.0))
    rec = 4
    traj = simulate(pos0, np.zeros((n, 2)), ctrl, params, World(wind=wind), duration, dt, record_every=rec)
    err = metrics.rms_series(np.stack([ctrl.error(t, traj.pos[k]) for k, t in enumerate(traj.t)]))
    try:
        measured = metrics.fit_decay_rate(traj.t, err, max(err[-1] * 1.2, 1e-12), 1e-2)
    except ValueError:
        measured = float("nan")
    summary = {
        "n": n, "sekil": shape, "topoloji": topology, "kazanc": gain, "sure_s": duration,
        "cebirsel_baglanti": graph.algebraic_connectivity(adj),
        "ongoru_hiz": predicted, "olculen_hiz": measured,
        "baslangic_hata_m": float(err[0]), "son_hata_m": float(err[-1]), **_collision_info(traj),
    }
    return Result("formasyon", traj, summary, {"offsets": offsets, "error": err, "controller": ctrl})


# ----------------------------------------------------------------------------------------------
def coverage_setup(density="gauss", size=100.0, cells=60):
    if density == "duzgun":
        return coverage.make_grid(size, size, cells, cells)
    if density == "gauss":
        spots = [(0.25 * size, 0.70 * size, 0.10 * size, 1.0), (0.75 * size, 0.30 * size, 0.14 * size, 1.2)]
        return coverage.make_grid(size, size, cells, cells, coverage.gaussian_density(spots))
    raise ValueError(f"unknown density {density!r}; choose from {DENSITIES}")


def run_kapsama(n=20, density="gauss", duration=150.0, seed=0, params=None, dt=0.1, size=100.0, cells=60,
                kill=None, wind=(0.0, 0.0), avoid=True):
    """Drones deployed from the lower-left corner spread out to minimise the locational cost.

    ``kill = (time, count)`` removes ``count`` random drones at ``time``; the rest re-balance.
    """
    params = params or DroneParams()
    rng = np.random.default_rng(seed)
    pts, w = coverage_setup(density, size, cells)
    pos0 = _spread_points(rng, n, 2.0, 0.2 * size, 2.0)     # deployed from a corner base
    kill_at, killed = [], []
    if kill:
        killed = sorted(rng.choice(n, size=min(kill[1], n - 1), replace=False).tolist())
        kill_at = [(kill[0], killed)]
    ctrl = coverage.CoverageController(pts, w, params, avoid=avoid)
    rec = 5
    traj = simulate(pos0, np.zeros((n, 2)), ctrl, params, World(wind=wind, bounds=(0, size, 0, size)),
                    duration, dt, kill_at=kill_at, record_every=rec)
    cost = np.array([coverage.locational_cost(traj.pos[k][traj.alive[k]], pts, w) for k in range(len(traj.t))])
    # reference: the converged plain Lloyd iteration from the same start
    ref_pos, ref_hist = coverage.lloyd(pos0 if not killed else np.delete(pos0, killed, axis=0), pts, w, 200)
    best_restart = min(coverage.lloyd(rng.uniform(0, size, (n - len(killed), 2)), pts, w, 100)[1][-1]
                       for _ in range(6))
    summary = {
        "n": n, "yogunluk": density, "sure_s": duration, "yok_edilen": len(killed),
        "baslangic_maliyet": float(cost[0]), "son_maliyet": float(cost[-1]),
        "lloyd_sinir_maliyet": float(ref_hist[-1]), "en_iyi_yeniden_baslatma": float(best_restart),
        **_collision_info(traj),
    }
    if density == "duzgun":
        summary["altigen_alt_sinir"] = coverage.hexagon_bound(size * size, n - len(killed))
    return Result("kapsama", traj, summary, {"pts": pts, "w": w, "cost": cost, "size": size, "killed": killed})


# ----------------------------------------------------------------------------------------------
def run_gorev(n=10, n_targets=None, method="ihale", duration=None, seed=0, params=None, dt=0.05, size=80.0):
    """Assign drones to targets (``ihale`` auction, ``acgozlu`` greedy, ``tam`` exact), then fly there."""
    params = params or DroneParams()
    m = n_targets or n
    rng = np.random.default_rng(seed)
    pos0 = _spread_points(rng, n, 0, size, 3.0)
    targets = _spread_points(rng, m, 0, size, 3.0)    # closer targets would be unreachable under avoidance
    cost = assign.distance_cost(pos0, targets)
    if method == "ihale":
        chosen, bids = assign.auction(cost)
    elif method == "acgozlu":
        chosen, bids = assign.greedy(cost), 0
    elif method == "tam":
        chosen, bids = assign.exact(cost), 0
    else:
        raise ValueError(f"unknown method {method!r}; choose from {ASSIGNERS}")
    goals = np.where((chosen >= 0)[:, None], targets[np.maximum(chosen, 0)], pos0)
    cruise = 0.5 * params.v_max
    duration = duration or float(np.linalg.norm(goals - pos0, axis=1).max() / cruise * 1.6 + 10.0)
    ctrl = GoalSeeker(goals, params, cruise=cruise)
    traj = simulate(pos0, np.zeros((n, 2)), ctrl, params, World(), duration, dt, record_every=2)
    arrive = metrics.arrival_times(traj, goals, tol=0.5)
    summary = {
        "n": n, "hedef": m, "yontem": method, "ihale_sayisi": bids,
        "atama_maliyeti_m": assign.total_cost(cost, chosen), "sure_s": duration,
        "varis_son_s": float(np.nanmax(arrive)) if np.isfinite(arrive).any() else float("nan"),
        "varan": int((np.isfinite(arrive) | (chosen < 0)).sum()), **_collision_info(traj),
    }
    return Result("gorev", traj, summary, {"goals": goals, "targets": targets, "assignment": chosen})
