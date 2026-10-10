"""Plots and animations (matplotlib, Agg backend)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from . import coverage  # noqa: E402

TITLES = {"takas": "Takas", "surusu": "Sürü", "formasyon": "Formasyon", "kapsama": "Kapsama", "gorev": "Görev atama"}


def _extent(result):
    pos = result.traj.pos.reshape(-1, 2)
    pts = [pos]
    if len(result.traj.world.obstacles):
        ob = result.traj.world.obstacles
        pts += [ob[:, :2] + ob[:, 2:3], ob[:, :2] - ob[:, 2:3]]
    for key in ("goals", "targets"):
        if key in result.info:
            pts.append(np.asarray(result.info[key]).reshape(-1, 2))
    if result.name == "kapsama":
        size = result.info["size"]
        pts.append(np.array([[0, 0], [size, size]]))
    allp = np.concatenate(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    pad = 0.06 * max(hi - lo)
    return lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad


def _static(ax, result):
    for ox, oy, r in result.traj.world.obstacles:
        ax.add_patch(Circle((ox, oy), r, color="0.55", zorder=1))
    for key, marker in (("goals", "x"), ("targets", "x")):
        if key in result.info:
            g = np.asarray(result.info[key]).reshape(-1, 2)
            ax.plot(g[:, 0], g[:, 1], marker, color="crimson", ms=6, zorder=2, ls="none")
    x0, x1, y0, y1 = _extent(result)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _cells(ax, result, k):
    info = result.info
    pts, size = info["pts"], info["size"]
    live = result.traj.alive[k]
    idx, _ = coverage.assign_cells(result.traj.pos[k][live], pts)
    cells = int(round(np.sqrt(len(pts))))
    ax.imshow(idx.reshape(cells, cells).T, origin="lower", extent=(0, size, 0, size), cmap="tab20",
              alpha=0.30, zorder=0, interpolation="nearest")
    ax.contour(np.linspace(0, size, cells), np.linspace(0, size, cells), info["w"].reshape(cells, cells).T,
               levels=5, colors="k", linewidths=0.4, alpha=0.5)


def plot_result(result, path, dpi=110):
    """Trajectories on the left, the scenario's key time series on the right."""
    traj = result.traj
    series = _series(result)
    fig, axes = plt.subplots(1, 2 if series else 1, figsize=(11 if series else 6, 5.2), squeeze=False)
    ax = axes[0, 0]
    if result.name == "kapsama":
        _cells(ax, result, len(traj.t) - 1)
    _static(ax, result)
    cmap = plt.get_cmap("turbo")
    for i in range(traj.n):
        colour = cmap(i / max(traj.n - 1, 1))
        ax.plot(traj.pos[:, i, 0], traj.pos[:, i, 1], color=colour, lw=0.9, alpha=0.8, zorder=3)
        ax.plot(traj.pos[0, i, 0], traj.pos[0, i, 1], "o", color=colour, ms=3.5, mfc="none", zorder=4)
        if traj.alive[-1, i]:
            ax.plot(traj.pos[-1, i, 0], traj.pos[-1, i, 1], "o", color=colour, ms=4.5, zorder=4)
        else:
            ax.plot(traj.pos[-1, i, 0], traj.pos[-1, i, 1], "x", color="k", ms=7, mew=1.6, zorder=5)
    ax.set_title(f"{TITLES.get(result.name, result.name)}: {traj.n} drone, "
                 f"çarpışan çift: {len(traj.collided)}")
    if series:
        series(axes[0, 1])
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _series(result):
    traj, info, summary = result.traj, result.info, result.summary
    if result.name == "formasyon":
        def draw(ax):
            err = info["error"]
            ax.semilogy(traj.t, err, color="tab:blue", label="ölçülen hata")
            p = summary["ongoru_hiz"]
            if np.isfinite(p) and err[0] > 0:
                ref = err[len(err) // 2] * np.exp(-p * (traj.t - traj.t[len(err) // 2]))
                ax.semilogy(traj.t[len(err) // 3:], ref[len(err) // 3:], "--", color="crimson",
                            label=f"Laplace tayfından öngörü: e^(-{p:.4f} t)")
            ax.set_xlabel("t (s)")
            ax.set_ylabel("formasyon hatası, rms (m)")
            ax.set_title(f"{summary['topoloji']} grafı, λ₂ = {summary['cebirsel_baglanti']:.3f}")
            ax.legend()
            ax.grid(alpha=0.3, which="both")
        return draw
    if result.name == "kapsama":
        def draw(ax):
            ax.semilogy(traj.t, info["cost"], color="tab:blue", label="yerleşim maliyeti H")
            ax.axhline(summary["lloyd_sinir_maliyet"], color="crimson", ls="--", label="Lloyd yakınsama değeri")
            if traj.events:
                ax.axvline(traj.events[0][0], color="k", ls=":", label="arıza")
            ax.set_xlabel("t (s)")
            ax.set_ylabel("H")
            ax.legend()
            ax.grid(alpha=0.3, which="both")
        return draw
    if result.name in ("takas", "gorev", "surusu"):
        def draw(ax):
            t = np.arange(len(traj.min_sep)) * traj.dt
            ax.plot(t, traj.min_sep, color="tab:blue", label="en küçük aralık")
            ax.axhline(2 * traj.params.radius, color="crimson", ls="--", label="temas (2·yarıçap)")
            ax.set_xlabel("t (s)")
            ax.set_ylabel("en küçük merkez aralığı (m)")
            ax.set_ylim(bottom=0)
            ax.legend()
            ax.grid(alpha=0.3)
        return draw
    return None


def animate(result, path, fps=20, max_frames=140, tail=12, dpi=70):
    """Animated GIF of the run."""
    traj = result.traj
    frames = np.unique(np.linspace(0, len(traj.t) - 1, min(max_frames, len(traj.t))).astype(int))
    x0, x1, y0, y1 = _extent(result)
    width = 7.0
    height = float(np.clip(width * (y1 - y0) / (x1 - x0), 3.2, 7.0)) + 0.7   # room for title and labels
    fig, ax = plt.subplots(figsize=(width, height))
    writer = PillowWriter(fps=fps)
    cmap = plt.get_cmap("turbo")
    colours = [cmap(i / max(traj.n - 1, 1)) for i in range(traj.n)]
    with writer.saving(fig, str(path), dpi):
        for k in frames:
            ax.clear()
            if result.name == "kapsama":
                _cells(ax, result, k)
            _static(ax, result)
            lo = max(0, k - tail)
            live = traj.alive[k]
            for i in range(traj.n):
                ax.plot(traj.pos[lo:k + 1, i, 0], traj.pos[lo:k + 1, i, 1], color=colours[i], lw=1.0, alpha=0.6)
            ax.scatter(traj.pos[k][live, 0], traj.pos[k][live, 1], c=[colours[i] for i in np.flatnonzero(live)],
                       s=26, zorder=5, edgecolors="k", linewidths=0.4)
            if (~live).any():
                ax.scatter(traj.pos[k][~live, 0], traj.pos[k][~live, 1], marker="x", c="k", s=30, zorder=5)
            ax.set_title(f"{TITLES.get(result.name, result.name)}  t = {traj.t[k]:.1f} s")
            writer.grab_frame()
    plt.close(fig)


def plot_vicsek(rows, n, box, path, dpi=110):
    from .vicsek import random_phase_order
    eta = np.array([r[0] for r in rows])
    phi = np.array([r[1] for r in rows])
    err = np.array([r[2] for r in rows])
    chi = np.array([r[3] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].errorbar(eta, phi, err, fmt="o-", capsize=2)
    ax[0].axhline(random_phase_order(n), color="crimson", ls="--", label="rastgele yönler: √π / (2√N)")
    ax[0].set_xlabel("gürültü η")
    ax[0].set_ylabel("düzen parametresi φ")
    ax[0].set_title(f"Vicsek modeli, N = {n}, L = {box:g}")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(eta, chi, "s-", color="tab:green")
    ax[1].set_xlabel("gürültü η")
    ax[1].set_ylabel("N var(φ)")
    ax[1].set_title("duyarlılık (sonlu boyut: geçişte tepe)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
