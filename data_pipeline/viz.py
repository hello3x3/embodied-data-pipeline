"""Matplotlib renderers: one plan overview + one card per action chunk.

Everything is drawn in the **native head image** (640x480). The point of these
figures is to make the new contract visible:

* trajectory points are every 4 action steps (6.25 Hz), not every step;
* the near window is 7 points (steps 0..24) while the action chunk is 25 steps;
* only ``executed`` (12) action steps are meant to run before replanning.

Labels are ASCII on purpose (matplotlib default fonts have no CJK coverage).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib import gridspec          # noqa: E402

from .chunks import Chunk                # noqa: E402
from .config import (                    # noqa: E402
    ACTION_HORIZON,
    EXEC_STEPS,
    FAR_POINTS,
    NEAR_POINTS,
)
from .io import Episode                  # noqa: E402
from .tracks import PlanTracks           # noqa: E402

CH_COLORS = {"anchor_left": "#2b6cb0", "anchor_right": "#38a169", "object": "#dd6b20"}
PRIMARY = {"L": "anchor_left", "R": "anchor_right"}


def _rgb(ep: Episode, frame: int) -> np.ndarray:
    return cv2.cvtColor(ep.read_color(frame), cv2.COLOR_BGR2RGB)


def _draw_path(ax, uv, color, lw=1.4, ls="-", alpha=0.9) -> None:
    ok = np.isfinite(uv).all(axis=1)
    if ok.sum() < 2:
        return
    pts = uv[ok]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, alpha=alpha, zorder=3)


def render_plan_overview(
    ep: Episode,
    pt: PlanTracks,
    chunks: list[Chunk],
    out_path: Path,
    task: str,
    active_arm: str = "R",
) -> None:
    fig = plt.figure(figsize=(13.5, 7.2), dpi=110)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.75, 1.0], figure=fig)
    ax = fig.add_subplot(gs[0])
    ax.imshow(_rgb(ep, pt.t_start))
    ax.set_xlim(0, ep.hw["cam_head"][1])
    ax.set_ylim(ep.hw["cam_head"][0], 0)
    ax.set_axis_off()
    ax.set_title(
        f"{task}  plan {pt.phase}  frames {pt.t_start}-{pt.t_end}  "
        f"{pt.n_points} points @ stride {pt.stride}  ({pt.duration_steps} steps)",
        fontsize=10, loc="left")

    for ch in pt.channels:
        color = CH_COLORS.get(ch.name, "gray")
        _draw_path(ax, ch.uv, color, lw=1.2, ls="--" if ch.kind == "object" else "-")
        ok = np.isfinite(ch.uv).all(axis=1) & ch.valid
        ax.scatter(ch.uv[ok, 0], ch.uv[ok, 1], s=10, color=color, zorder=4)
        ax.scatter(ch.uv[~ch.valid & np.isfinite(ch.uv).all(axis=1), 0],
                   ch.uv[~ch.valid & np.isfinite(ch.uv).all(axis=1), 1],
                   s=14, facecolors="none", edgecolors=color, lw=0.8, zorder=4)

    prim = next((c for c in pt.channels if c.name == PRIMARY[active_arm]), pt.channels[0])
    # label every point index on the primary anchor (readable at ~20 points/plan)
    for k in range(len(prim)):
        if not np.isfinite(prim.uv[k]).all():
            continue
        if len(prim) <= 24 or k % 2 == 0:
            ax.text(prim.uv[k, 0] + 4, prim.uv[k, 1] - 4, str(k),
                    fontsize=7, color="#1a202c", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))

    # chunk windows
    for c in chunks:
        seg = prim.uv[c.cursor: c.cursor + (c.executed_steps // pt.stride) + 1]
        if len(seg) >= 2:
            ax.plot(seg[:, 0], seg[:, 1], color="#e53e3e", lw=3, alpha=0.55, zorder=5)
        p = prim.uv[c.cursor]
        if np.isfinite(p).all():
            ax.scatter([p[0]], [p[1]], s=110, marker="*", color="#e53e3e",
                       edgecolors="white", lw=0.6, zorder=7)
            ax.text(p[0] - 10, p[1] + 16, f"c{c.index}", fontsize=8, color="#c53030",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#e53e3e", lw=0.6))

    handles = [plt.Line2D([], [], color=CH_COLORS["anchor_left"], label="anchor_left"),
               plt.Line2D([], [], color=CH_COLORS["anchor_right"], label="anchor_right"),
               plt.Line2D([], [], color=CH_COLORS["object"], ls="--", label="object"),
               plt.Line2D([], [], color="#e53e3e", lw=3, label="executed / chunk"),
               plt.Line2D([], [], marker="*", ls="", color="#e53e3e", label="chunk cursor")]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)

    # text panel
    ax2 = fig.add_subplot(gs[1])
    ax2.set_axis_off()
    obj = next((c for c in pt.channels if c.kind == "object"), None)
    lines = [
        f"episode   {ep.path.name}",
        f"task      {task}   T={ep.T}  fps={ep.fps}",
        f"phase     {pt.phase}",
        f"frames    {pt.t_start} .. {pt.t_end}",
        f"points    {pt.n_points}  (stride {pt.stride} = {ep.fps / pt.stride:.2f} Hz)",
        f"steps     {pt.duration_steps}  (= stride*(L-1))",
        f"object    valid {int(obj.valid.sum())}/{len(obj) if obj else 0}"
        if obj else "object    (none)",
        "",
        "chunks:  cursor  step   remaining  exec",
    ]
    for c in chunks:
        lines.append(
            f"  c{c.index:<3d} {c.cursor:>5d}  {c.step_cursor:>4d}   "
            f"{c.remaining_steps:>5d}     {c.executed_steps:>3d}")
    lines += ["", "contract:",
              f"  near {NEAR_POINTS} pts = steps 0..{ACTION_HORIZON - 1} (every {pt.stride})",
              f"  far  {FAR_POINTS} pts = arc-length over the rest",
              f"  executed = min({EXEC_STEPS}, remaining_steps)"]
    ax2.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
             fontsize=8.5, color="#1a202c")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
