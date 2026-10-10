"""Model checks: each simulation component against an analytic or independent result.

``run_all()`` returns a list of :class:`Check`; the CLI prints them (``dogrulama``) and the tests
assert them. Everything is seeded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import assign, avoidance, coverage, dynamics, graph, metrics, scenarios, vicsek
from .control import Boids, Formation
from .dynamics import DroneParams
from .sim import World, simulate


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


# ----------------------------------------------------------------------------------------------
def _rk4(pos, vel, acc, drag, wind, total, n):
    h = total / n

    def f(v):
        return v, acc - drag * (v - wind)

    for _ in range(n):
        k1 = f(vel)
        k2 = f(vel + 0.5 * h * k1[1])
        k3 = f(vel + 0.5 * h * k2[1])
        k4 = f(vel + h * k3[1])
        pos = pos + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        vel = vel + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return pos, vel


def integrator_vs_rk4(seed=0):
    """The closed-form step against RK4 with ever smaller steps (error should fall ~16x per halving)."""
    rng = np.random.default_rng(seed)
    pos, vel, acc = (rng.normal(size=(6, 2)) for _ in range(3))
    wind, drag = np.array([2.0, -1.0]), 0.3
    exact_p, exact_v = dynamics.zoh_step(pos, vel, acc, drag, wind, 2.0)
    errors = []
    for n in (4, 8, 16, 32, 64):
        p, v = _rk4(pos, vel, acc, drag, wind, 2.0, n)
        errors.append(float(max(np.abs(p - exact_p).max(), np.abs(v - exact_v).max())))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    pp, vv = pos, vel
    for _ in range(40):
        pp, vv = dynamics.zoh_step(pp, vv, acc, drag, wind, 0.05)
    semigroup = float(max(np.abs(pp - exact_p).max(), np.abs(vv - exact_v).max()))
    ok = errors[-1] < 1e-8 and min(orders) > 3.5 and semigroup < 1e-12
    return Check("integrator vs RK4", ok,
                 f"RK4 error {errors[0]:.1e} -> {errors[-1]:.1e} (order {np.mean(orders):.2f}); "
                 f"40 steps of 0.05 s vs one step of 2 s: {semigroup:.1e}")


def limits_and_wind():
    """Acceleration and speed limits hold; in a steady wind the velocity settles at ``tau * drag * w``."""
    params = DroneParams(v_max=5.0, a_max=3.0)

    def reckless(t, pos, vel, alive):
        return np.tile([100.0, -80.0], (len(pos), 1))

    tr = simulate(np.zeros((3, 2)), np.zeros((3, 2)), reckless, params, World(), 10.0, 0.05)
    amax = float(np.linalg.norm(tr.acc, axis=2).max())
    vmax = float(np.linalg.norm(tr.vel, axis=2).max())
    wind = np.array([3.0, -2.0])
    params = DroneParams()
    hold = lambda t, pos, vel, alive: dynamics.track_velocity(np.zeros_like(pos), vel, params)
    tr2 = simulate(np.zeros((2, 2)), np.zeros((2, 2)), hold, params, World(wind=wind), 40.0, 0.05)
    settled = tr2.vel[-1, 0]
    predicted = params.tau * params.drag * wind
    ok = amax <= 3.0 + 1e-9 and vmax <= 5.0 + 1e-9 and np.allclose(settled, predicted, atol=1e-6)
    return Check("limits and wind offset", ok,
                 f"max |a| {amax:.3f} <= 3, max |v| {vmax:.3f} <= 5; hover velocity in wind "
                 f"{np.round(settled, 4).tolist()} vs tau*drag*w {np.round(predicted, 4).tolist()}")


def graph_spectra(seed=0, cases=200):
    """BFS components equal the Laplacian nullity; lambda_2 > 0 iff connected; closed forms for ring/path/complete."""
    rng = np.random.default_rng(seed)
    mismatches = 0
    for _ in range(cases):
        n = int(rng.integers(3, 14))
        adj = graph.range_adjacency(rng.uniform(0, 10, (n, 2)), rng.uniform(1.5, 6.0))
        comps = graph.n_components(adj)
        mismatches += comps != graph.nullity(adj)
        mismatches += (graph.algebraic_connectivity(adj) > 1e-9) != (comps == 1)
    n = 9
    forms = {
        "ring": (graph.algebraic_connectivity(graph.ring_adjacency(n)), 2 - 2 * np.cos(2 * np.pi / n)),
        "path": (graph.algebraic_connectivity(graph.path_adjacency(n)), 2 - 2 * np.cos(np.pi / n)),
        "complete": (graph.algebraic_connectivity(graph.complete_adjacency(n)), float(n)),
    }
    worst = max(abs(a - b) for a, b in forms.values())
    return Check("graph Laplacian spectra", mismatches == 0 and worst < 1e-10,
                 f"{cases} random graphs: components vs nullity mismatches {mismatches}; "
                 f"ring/path/complete lambda_2 vs closed form, worst difference {worst:.1e}")


def consensus_rates(n=8, gain=0.8):
    """Simulated formation-error decay against the slowest mode of the Laplacian spectrum."""
    rows, worst = [], 0.0
    for topology in ("halka", "yol", "tam"):
        r = scenarios.run_formasyon(n=n, shape="cember", topology=topology, gain=gain, seed=3)
        s = r.summary
        rel = abs(s["olculen_hiz"] - s["ongoru_hiz"]) / s["ongoru_hiz"]
        worst = max(worst, rel)
        rows.append(f"{topology} measured {s['olculen_hiz']:.5f} vs predicted {s['ongoru_hiz']:.5f}")
    return Check("consensus rate vs spectrum", worst < 2e-3,
                 "; ".join(rows) + f" (worst relative difference {worst:.1e})")


def wind_offset(n=6, gain=0.8):
    """Steady offset of a pinned formation in wind, from ``gain M e = tau drag w``."""
    params = DroneParams(drag=0.1)
    wind = np.array([3.0, -2.0])
    offsets = scenarios.formation_offsets("cizgi", n)
    adj = graph.ring_adjacency(n)
    ctrl = Formation(offsets, adj, params, gain=gain, leader=0)
    rng = np.random.default_rng(1)
    tr = simulate(rng.uniform(-20, 20, (n, 2)), np.zeros((n, 2)), ctrl, params, World(wind=wind), 400.0, 0.05,
                  record_every=20)
    measured = ctrl.error(tr.t[-1], tr.pos[-1])
    predicted = np.linalg.solve(gain * ctrl.matrix, np.outer(np.ones(n), params.tau * params.drag * wind))
    diff = float(np.abs(measured - predicted).max())
    return Check("formation offset in wind", diff < 1e-6,
                 f"largest error component {np.abs(measured).max():.4f} m; simulation vs linear solve differ by {diff:.1e} m")


def time_to_collision_vs_stepping(seed=0, cases=400):
    """The quadratic time to collision against brute-force time stepping."""
    rng = np.random.default_rng(seed)
    dt, horizon = 1e-3, 12.0
    t = np.arange(0.0, horizon, dt)
    worst, hits, mismatches = 0.0, 0, 0
    for _ in range(cases):
        p = rng.uniform(-15, 15, 2)
        u = rng.uniform(-6, 6, 2)
        radius = rng.uniform(0.5, 3.0)
        exact = float(avoidance.time_to_collision(p, u, radius))
        inside = np.linalg.norm(p[None] - t[:, None] * u[None], axis=1) <= radius
        brute = float(t[np.argmax(inside)]) if inside.any() else np.inf
        if exact < horizon - dt:
            hits += 1
            if not np.isfinite(brute):
                mismatches += 1
            else:
                worst = max(worst, abs(exact - brute))
        elif exact > horizon + dt:
            mismatches += np.isfinite(brute)
    return Check("time to collision vs time stepping", mismatches == 0 and worst <= dt + 1e-12,
                 f"{cases} random pairs, {hits} colliding within {horizon:g} s: largest time difference {worst:.1e} s "
                 f"(step {dt:g} s), disagreements about whether a collision happens: {mismatches}")


def swap_avoidance(sizes=(2, 4, 8, 12, 16)):
    """Antipodal swap with and without avoidance, 20 m circle."""
    with_a, without_a, sep = [], [], []
    for n in sizes:
        a = scenarios.run_takas(n=n, avoid=True)
        b = scenarios.run_takas(n=n, avoid=False)
        with_a.append(a.summary["carpisan_cift"])
        without_a.append(b.summary["carpisan_cift"])
        sep.append(a.summary["en_kucuk_aralik_m"])
        if a.summary["varan"] != n:
            with_a[-1] += 1000
    ok = all(c == 0 for c in with_a) and all(c > 0 for c in without_a)
    return Check("velocity-obstacle swap", ok,
                 f"n={list(sizes)}: colliding pairs with avoidance {with_a}, without {without_a}; "
                 f"smallest separation with avoidance {min(sep):.2f} m (contact 0.60 m)")


def auction_vs_exact(seed=0, cases=250):
    """Auction within ``n eps`` of two exact solvers; the optimal Euclidean matching has no crossing edges."""
    eps = 1e-6
    worst_gap, greedy_excess, crossings, dp_bf = 0.0, [], 0, 0.0
    for k in range(cases):
        rng = np.random.default_rng(seed + k)
        n = int(rng.integers(2, 9))
        m = int(rng.integers(1, n + 1))
        drones, targets = rng.uniform(0, 10, (n, 2)), rng.uniform(0, 10, (m, 2))
        cost = assign.distance_cost(drones, targets)
        brute = assign.brute_force(cost)
        optimum = assign.total_cost(cost, brute)
        dp_bf = max(dp_bf, abs(optimum - assign.total_cost(cost, assign.exact(cost))))
        auc, _ = assign.auction(cost, eps_final=eps)
        worst_gap = max(worst_gap, (assign.total_cost(cost, auc) - optimum) / (n * eps))
        greedy_excess.append(assign.total_cost(cost, assign.greedy(cost)) / optimum - 1.0)
        crossings += _count_crossings(drones, targets, assign.exact(cost))
    return Check("auction vs exact assignment", worst_gap <= 1.0 + 1e-9 and dp_bf < 1e-9 and crossings == 0,
                 f"{cases} instances: auction gap <= {worst_gap:.2f} n*eps; subset DP vs brute force {dp_bf:.1e}; "
                 f"crossing edges in optimal matchings {crossings}; greedy excess mean "
                 f"{100 * np.mean(greedy_excess):.1f}% max {100 * np.max(greedy_excess):.0f}%")


def _count_crossings(drones, targets, chosen):
    segs = [(drones[i], targets[j]) for i, j in enumerate(chosen) if j >= 0]

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    count = 0
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            (a, b), (c, d) = segs[i], segs[j]
            if orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0:
                count += 1
    return count


def lloyd_checks(seed=0):
    """Lloyd: closed forms for N=1 and N=2, monotone cost, Fejes Toth hexagon bound, restarts."""
    pts, w = coverage.make_grid(1.0, 1.0, 100, 100)
    _, h1 = coverage.lloyd(np.array([[0.1, 0.9]]), pts, w, 5)
    _, h2 = coverage.lloyd(np.array([[0.2, 0.3], [0.7, 0.6]]), pts, w, 60)
    closed1, closed2 = 1.0 / 6.0, 5.0 / 48.0
    pts2, w2 = coverage.make_grid(1.0, 1.0, 100, 100)
    monotone, ratios = True, []
    for k in range(3):
        rng = np.random.default_rng(seed + k)
        _, hist = coverage.lloyd(rng.uniform(0, 1, (40, 2)), pts2, w2, 80)
        monotone &= bool(np.all(np.diff(hist) <= 1e-12))
        ratios.append(hist[-1] / coverage.hexagon_bound(1.0, 40))
    rel1, rel2 = abs(h1[-1] - closed1) / closed1, abs(h2[-1] - closed2) / closed2
    ok = rel1 < 2e-3 and rel2 < 2e-3 and monotone and min(ratios) > 0.995 and max(ratios) < 1.10
    return Check("Lloyd coverage", ok,
                 f"N=1 {h1[-1]:.5f} vs 1/6 ({rel1:.1e}); N=2 {h2[-1]:.5f} vs 5/48 ({rel2:.1e}); cost never rose; "
                 f"N=40 cost / hexagon lower bound {min(ratios):.3f}-{max(ratios):.3f}")


def coverage_flow(n=12, seed=0):
    """The drone flow (with dynamics and avoidance) reaches a Lloyd fixed point; the cost does not rise by more than noise."""
    r = scenarios.run_kapsama(n=n, density="gauss", duration=120.0, seed=seed, cells=40)
    cost = r.info["cost"]
    rises = float(np.max(np.diff(cost) / cost[:-1]))
    gap = abs(cost[-1] - r.summary["lloyd_sinir_maliyet"]) / r.summary["lloyd_sinir_maliyet"]
    return Check("coverage flow", rises < 0.02 and gap < 0.05 and r.summary["carpisan_cift"] == 0,
                 f"cost {cost[0]:.3g} -> {cost[-1]:.3g}; largest relative rise per frame {rises:.1e}; "
                 f"within {100 * gap:.1f}% of plain Lloyd from the same start; collisions {r.summary['carpisan_cift']}")


def vicsek_checks(n=100, box=5.0, seed=0):
    ordered = vicsek.run(n, box, 0.0, steps=800, burn_in=400, seed=seed).mean()
    random_phi = np.mean([vicsek.run(n, box, 2 * np.pi, steps=1500, burn_in=500, seed=seed + k).mean()
                          for k in range(3)])
    expected = vicsek.random_phase_order(n)
    sweep = [vicsek.run(n, box, eta, steps=900, burn_in=400, seed=seed + 10 + k).mean()
             for k, eta in enumerate((0.5, 1.5, 2.5, 3.5, 4.5))]
    monotone = bool(np.all(np.diff(sweep) < 0))
    ok = ordered > 0.99 and abs(random_phi - expected) / expected < 0.2 and monotone
    return Check("Vicsek order parameter", ok,
                 f"phi(eta=0) {ordered:.3f}; phi(eta=2pi) {random_phi:.4f} vs sqrt(pi)/(2 sqrt(N)) {expected:.4f}; "
                 f"phi at eta=0.5..4.5 {np.round(sweep, 2).tolist()}")


def flocking_checks(seed=0):
    """Alignment drives a connected flock to a common heading; the default flock clears the obstacles."""
    params = DroneParams(drag=0.0)
    rng = np.random.default_rng(seed)
    n = 25
    pos0 = scenarios._spread_points(rng, n, 0.0, 12.0, 1.5)
    vel0 = rng.normal(size=(n, 2)) * 2.0
    tr = simulate(pos0, vel0, Boids(params, goal=None), params, World(), 40.0, 0.05, record_every=4)
    before, after = metrics.order_parameter(tr.vel[0]), metrics.order_parameter(tr.vel[-1])
    r = scenarios.run_surusu(n=30, seed=seed)
    s = r.summary
    ok = after > 0.99 and before < 0.5 and s["en_yakin_engel_m"] > 0 and s["hedefe_yakin"] == s["n"]
    return Check("flocking", ok,
                 f"order parameter {before:.2f} -> {after:.3f} with alignment only; default flock: obstacle clearance "
                 f"{s['en_yakin_engel_m']:.2f} m, {s['hedefe_yakin']}/{s['n']} near the goal, "
                 f"colliding pairs {s['carpisan_cift']}")


ALL = (integrator_vs_rk4, limits_and_wind, graph_spectra, consensus_rates, wind_offset,
       time_to_collision_vs_stepping, swap_avoidance, auction_vs_exact, lloyd_checks, coverage_flow,
       vicsek_checks, flocking_checks)


def run_all():
    return [check() for check in ALL]
