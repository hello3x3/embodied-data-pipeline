"""Chunking: turn one plan's trajectory-rate tracks into fixed action chunks.

The invariants (see ``docs/action_head.md`` §6 and the data contract):

* the trajectory is a polyline at ``stride`` action steps per point;
* a chunk always predicts ``ACTION_HORIZON`` steps, but only
  ``executed = min(EXEC_STEPS, remaining_steps)`` of them are meant to run;
* ``cursor`` is a **point index** and advances by ``executed // stride`` points,
  so it always lands on a trajectory point;
* ``action_mask`` supervises only the steps that still belong to this plan.

This module is the reference implementation the training packer must match.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import (
    ACTION_HORIZON,
    EXEC_STEPS,
    FAR_POINTS,
    NEAR_POINTS,
    POINT_STRIDE,
)
from .tracks import PlanTracks

__all__ = ["Chunk", "iter_chunks", "near_window", "far_sample"]


def near_window(uv: np.ndarray, valid: np.ndarray, cursor: int, n: int) -> tuple:
    """``n`` points from ``cursor``; pad with the last one, ``valid=False``."""
    L = len(uv)
    idx = np.arange(cursor, cursor + n)
    pts = np.zeros((n, 2), np.float64)
    val = np.zeros(n, bool)
    inside = idx < L
    if inside.any():
        pts[inside] = uv[idx[inside]]
        val[inside] = valid[idx[inside]]
    fill = uv[min(L - 1, max(cursor - 1, 0))] if L else np.zeros(2)
    for i in np.where(~inside)[0]:
        pts[i] = fill
    return pts, val


def far_sample(uv: np.ndarray, valid: np.ndarray, cursor: int, m: int) -> tuple:
    """Arc-length equidistant samples over the remaining polyline (including end)."""
    rest, rv = uv[cursor:], valid[cursor:]
    if rest.shape[0] == 0:
        return np.zeros((m, 2)), np.zeros(m, bool), np.linspace(0, 1, m)
    if rest.shape[0] == 1:
        return np.repeat(rest, m, axis=0), np.repeat(rv, m), np.linspace(0, 1, m)
    d = np.linalg.norm(np.diff(rest, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cum[-1])
    if total <= 1e-9:                       # static path: repeat the endpoint
        return (np.repeat(rest[-1:], m, axis=0), np.repeat(rv[-1:], m),
                np.linspace(0.0, 1.0, m))
    targets = np.linspace(0.0, total, m)
    idx = np.searchsorted(cum, targets, side="left").clip(0, rest.shape[0] - 1)
    return rest[idx], rv[idx], targets / total


@dataclass
class Chunk:
    index: int
    cursor: int                       # point index
    step_cursor: int                  # action step index
    t_cursor: int                     # frame
    near_steps: np.ndarray            # (NEAR_POINTS,) = 0,4,...,24
    near_points: int                  # valid points in the near window
    remaining_steps: int
    executed_steps: int
    action: np.ndarray                # (ACTION_HORIZON, 14) commanded targets
    action_mask: np.ndarray           # (ACTION_HORIZON,) bool
    channels: dict = field(default_factory=dict)   # name -> dict(uv, valid, far, far_arc)

    def near_table(self, name: str) -> list[dict]:
        """JSON-safe rows: non-finite coordinates become ``None`` (not NaN)."""
        ch = self.channels[name]
        out = []
        for s, p, v in zip(self.near_steps, ch["near"], ch["near_valid"]):
            finite = bool(np.isfinite(p).all())
            out.append({"step": int(s),
                        "u": round(float(p[0]), 2) if finite else None,
                        "v": round(float(p[1]), 2) if finite else None,
                        "valid": bool(v)})
        return out


def iter_chunks(
    pt: PlanTracks,
    actions: np.ndarray,
    horizon: int = ACTION_HORIZON,
    exec_steps: int = EXEC_STEPS,
    near_points: int = NEAR_POINTS,
    far_points: int = FAR_POINTS,
) -> list[Chunk]:
    """Roll a receding-horizon cursor over one plan's tracks."""
    if exec_steps % pt.stride:
        raise ValueError(
            f"exec_steps={exec_steps} 不是 stride={pt.stride} 的倍数：游标会落在点之间")
    L = pt.n_points
    if L < 2:
        return []
    steps_per_exec = exec_steps // pt.stride
    chunks: list[Chunk] = []
    cursor = 0
    index = 0
    #: step index where the phase ends (inclusive); may be 0–3 steps past the
    #: last trajectory point, covered by the padded last point.
    phase_steps = pt.t_end - pt.t_start + 1
    while True:
        t_cursor = pt.t_start + cursor * pt.stride
        remaining_steps = phase_steps - (t_cursor - pt.t_start)
        if remaining_steps <= 0:
            break
        executed = min(exec_steps, remaining_steps)
        mask = np.zeros(horizon, bool)
        mask[: min(horizon, remaining_steps)] = True

        chans = {}
        for ch in pt.channels:
            pts, val = near_window(ch.uv, ch.valid, cursor, near_points)
            far_uv, far_val, far_arc = far_sample(ch.uv, ch.valid, cursor, far_points)
            chans[ch.name] = {
                "near": pts, "near_valid": val,
                "far": far_uv, "far_valid": far_val, "far_arc": far_arc,
                "kind": ch.kind,
            }

        act = actions[t_cursor: t_cursor + horizon]
        if len(act) < horizon:
            act = np.pad(act, ((0, horizon - len(act)), (0, 0)), mode="edge")

        chunks.append(Chunk(
            index=index, cursor=cursor, step_cursor=cursor * pt.stride,
            t_cursor=int(t_cursor),
            near_steps=np.arange(near_points) * pt.stride,
            near_points=int(min(near_points, L - cursor)),
            remaining_steps=int(remaining_steps), executed_steps=int(executed),
            action=np.asarray(act, np.float64), action_mask=mask, channels=chans,
        ))
        if executed >= remaining_steps:      # last chunk of the phase
            break
        cursor += steps_per_exec
        index += 1
    return chunks
