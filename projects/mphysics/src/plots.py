"""Figures for the mphysics notebooks; each returns the figure (or animation)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from core import viz_style as vs

BODIES = [vs.SERIES["stratiform"], vs.SERIES["convective"], vs.SERIES["frontal"]]


def orbits(*trajectories):
    """x-y paths of every body, one panel per trajectory."""
    fig, axes = plt.subplots(1, len(trajectories), figsize=(6.5 * len(trajectories), 5.5),
                             squeeze=False)
    for ax, traj in zip(axes[0], trajectories):
        for b in range(traj.pos.shape[1]):
            c = BODIES[b % len(BODIES)]
            ax.plot(traj.pos[:, b, 0], traj.pos[:, b, 1], color=c, lw=0.8,
                    label=f"body {b} (m={traj.masses[b]:g})")
            ax.plot(*traj.pos[0, b, :2], "o", color=c, ms=7, mec=vs.SURFACE, mew=1.5)
        ax.set(aspect="equal", xlabel="x", ylabel="y", title=traj.label)
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def two_body_measurements(traj, m: dict, r: np.ndarray, a: np.ndarray):
    """Distance, speed, the 1/r^2 relation, energy and angular momentum."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    ax[0, 0].plot(m["t"], r, color=BODIES[0])
    ax[0, 0].set(title="distance r", xlabel="t")
    ax[0, 1].plot(m["t"], m["speed"][:, 1], color=BODIES[0])
    ax[0, 1].set(title="speed of the planet", xlabel="t")
    x = 1 / r ** 2
    slope, icept = np.polyfit(x, a, 1)
    ax[0, 2].scatter(x, a, s=3, color=BODIES[0], alpha=0.4)
    ax[0, 2].plot(x[[x.argmin(), x.argmax()]], slope * x[[x.argmin(), x.argmax()]] + icept,
                  color=vs.INK_PRIMARY, label=f"fit: |a| = {slope:.3f}/r² {icept:+.1e}")
    ax[0, 2].set(title="|a| against 1/r²", xlabel="1/r²")
    ax[0, 2].legend()
    for key, c in (("kinetic", BODIES[1]), ("potential", BODIES[2]), ("energy", vs.INK_PRIMARY)):
        ax[1, 0].plot(m["t"], m[key], color=c, label=key)
    ax[1, 0].set(title=f"energy (drift {m['energy_drift']:.1e})", xlabel="t")
    ax[1, 0].legend()
    ax[1, 1].plot(m["t"], m["angular_momentum"], color=BODIES[0])
    ax[1, 1].set(title=f"angular momentum (drift {m['angular_momentum_drift']:.1e})", xlabel="t")
    ax[1, 2].axis("off")
    fig.tight_layout()
    return fig


def conservation(m: dict, title: str = ""):
    """Pairwise distances, energy and angular momentum of an n-body run."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for k, (i, j) in enumerate(m["pairs"]):
        ax[0].plot(m["t"], m["dist"][:, k], color=BODIES[k % 3], lw=1, label=f"d({i},{j})")
    ax[0].set(title="pairwise distance", xlabel="t")
    ax[0].legend()
    for key, c in (("kinetic", BODIES[1]), ("potential", BODIES[2]), ("energy", vs.INK_PRIMARY)):
        ax[1].plot(m["t"], m[key], color=c, label=key)
    ax[1].set(title=f"energy (drift {m['energy_drift']:.1e})", xlabel="t")
    ax[1].legend()
    ax[2].plot(m["t"], m["angular_momentum"], color=BODIES[0])
    ax[2].set(title="total angular momentum", xlabel="t")
    if title:
        fig.suptitle(title, x=0.05, ha="left")
    fig.tight_layout()
    return fig


def force_law(r: np.ndarray, a: np.ndarray, model, gm: float = 1.0):
    """Data, the true GM/r^2, and PySR's best equation."""
    fig, ax = plt.subplots(figsize=(7, 4))
    grid = np.linspace(r.min(), r.max(), 200)
    ax.scatter(r, a, s=4, color=vs.INK_MUTED, alpha=0.4, label="measured")
    ax.plot(grid, gm / grid ** 2, color=vs.INK_PRIMARY, lw=2, label=f"true: {gm:g}/r²")
    ax.plot(grid, model.predict(grid.reshape(-1, 1)), color=BODIES[1], ls="--", lw=2,
            label=f"PySR: {model.sympy()}")
    ax.set(xlabel="distance r", ylabel="|a|")
    ax.legend()
    return fig


def sindy_rollouts(fits: dict):
    """Each SINDy model integrated from the first state, against the data.

    ``fits`` maps a label to ``(model, x, t)`` from ``nbody.discover_dynamics``.
    """
    fig, axes = plt.subplots(1, len(fits), figsize=(6 * len(fits), 5), squeeze=False)
    for ax, (label, (model, x, t)) in zip(axes[0], fits.items()):
        ax.plot(x[:, 0], x[:, 1], color=vs.INK_PRIMARY, lw=2, label="data")
        from nbody import rollout

        sim = rollout(model, x[0], t)
        done = np.isfinite(sim[:, 0])
        ax.plot(sim[done, 0], sim[done, 1], color=BODIES[1], ls="--", lw=1.5,
                label="SINDy rollout" + ("" if done.all() else
                                         f" (diverged at t = {t[done][-1]:.1f})"))
        span = np.abs(x[:, :2]).max() * 1.4
        ax.set(aspect="equal", xlim=(-span, span), ylim=(-span, span), title=label,
               xlabel="x", ylabel="y")
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def pinn_fits(throw, runs: dict, title: str = ""):
    """Noisy data, the true trajectory, and each network's fit."""
    import torch

    t_data, h_data, _ = throw.tensors()
    grid = np.linspace(0, throw.t_max, 200, dtype=np.float32).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.scatter(t_data, h_data, color=vs.INK_MUTED, s=18, label="noisy data", zorder=3)
    ax.plot(grid, throw.height(grid), color=vs.INK_PRIMARY, lw=2.5, label="truth")
    colors = [vs.STATUS_CRITICAL] + BODIES + [vs.INK_SECONDARY]
    for (lam, r), c in zip(runs.items(), colors):
        with torch.no_grad():
            pred = r["model"](torch.from_numpy(grid)).numpy()
        ax.plot(grid, pred, color=c, lw=1.4,
                label="plain network" if lam == 0 else f"λ_ode = {lam:g}")
    ax.set(xlabel="t", ylabel="height", title=title)
    ax.legend(ncol=2, fontsize=8)
    return fig


def pinn_error_vs_weight(*tables):
    """Error against the true trajectory as a function of the physics weight."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for table, c in zip(tables, BODIES):
        t = table[table.lambda_ode > 0]
        ax.plot(t.lambda_ode, t.rmse_vs_truth, "o-", color=c, label=f"{t.residual.iloc[0]} residual")
    plain = tables[0][tables[0].lambda_ode == 0].rmse_vs_truth
    if len(plain):
        ax.axhline(float(plain.iloc[0]), color=vs.STATUS_CRITICAL, ls="--", lw=1,
                   label="plain network")
    ax.set(xscale="log", yscale="log", xlabel="physics weight λ_ode",
           ylabel="RMSE vs true trajectory")
    ax.legend()
    return fig


def animate(*trajectories, frame_skip: int = 5, trail: int = 200):
    """An HTML animation of the trajectories (heavy: not used in the notebooks)."""
    from matplotlib.animation import FuncAnimation

    fig, axes = plt.subplots(1, len(trajectories), figsize=(6 * len(trajectories), 5),
                             squeeze=False)
    idx = np.arange(0, min(len(t.t) for t in trajectories), frame_skip)
    artists = []
    for ax, traj in zip(axes[0], trajectories):
        lim = np.abs(traj.pos[..., :2]).max() * 1.1
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", title=traj.label)
        artists.append([(ax.plot([], [], "o", color=BODIES[b % 3])[0],
                         ax.plot([], [], "-", color=BODIES[b % 3], lw=0.6)[0])
                        for b in range(traj.pos.shape[1])])

    def step(k):
        i = idx[k]
        out = []
        for traj, arts in zip(trajectories, artists):
            for b, (dot, tail) in enumerate(arts):
                dot.set_data([traj.pos[i, b, 0]], [traj.pos[i, b, 1]])
                tail.set_data(traj.pos[max(0, i - trail):i + 1, b, 0],
                              traj.pos[max(0, i - trail):i + 1, b, 1])
                out += [dot, tail]
        return out

    anim = FuncAnimation(fig, step, frames=len(idx), interval=30, blit=True)
    plt.close(fig)
    return anim
