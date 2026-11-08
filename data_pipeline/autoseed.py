"""Automatic object seed at a grasp event (no VLM, no human).

For one subtask (one grasp cycle of one arm):

1. take the frame ``t_close - 3`` (object visible, gripper close),
2. project the **two-finger midpoint** of that arm into the image,
3. collect Shi-Tomasi corners in a ring around it (14–70 px; excludes the gripper),
4. track all candidates at once with CoTracker,
5. score each candidate after the grasp:
      rigidity  = std( distance(point, gripper) )   -> small
      movement  = max displacement                   -> large
      visibility                                     -> high
   keep the best one; if none passes, return ``None`` (subtask gets anchors only).

Why "distance to the gripper": after a rigid grasp the object point keeps a
roughly constant offset to the gripper midpoint (2D, static head camera).
That is also the QC signal the data pipeline can gate on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .cotracker_track import track_points_cotracker
from .geometry import gripper_midpoint_3d, project_point_depth
from .io import Episode
from .task_timeline import Subtask

__all__ = ["SeedResult", "auto_seed_track"]


@dataclass
class SeedResult:
    seed_uv: tuple[float, float]
    t_seed: int
    uv: dict[int, np.ndarray]        # frame -> (2,)
    qc: dict                          # rigidity / movement / visibility
    mask: np.ndarray | None = None    # optional SAM mask at t_seed
    info: dict | None = None          # optional mask diagnostics
    corner_uv: np.ndarray | None = None   # (T,N,2) all CoTracker corner tracks
    corner_vis: np.ndarray | None = None  # (T,N) visibility
    corner_t0: int | None = None          # frame of corner_uv[0]


def _gripper_uv(ep: Episode, arm: str, frames: list[int]) -> np.ndarray:
    poses = ep.L if arm == "L" else ep.R
    K, E = ep.K["cam_head"], ep.E["cam_head"]
    out = []
    for f in frames:
        p, z = project_point_depth(gripper_midpoint_3d(poses[f]), K, E[f])
        out.append(p if z > 1e-6 else np.array([np.nan, np.nan]))
    return np.asarray(out, np.float64)


def _candidates(gray: np.ndarray, center, r_in: float = 14.0, r_out: float = 70.0,
                max_corners: int = 24) -> np.ndarray:
    h, w = gray.shape
    u, v = float(center[0]), float(center[1])
    if not (0 <= u < w and 0 <= v < h):
        return np.zeros((0, 2))
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(u), int(v)), int(r_out), 255, -1)
    cv2.circle(mask, (int(u), int(v)), int(r_in), 0, -1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01,
                                  minDistance=5, mask=mask)
    out = [p.ravel().astype(np.float64) for p in pts] if pts is not None else []
    if len(out) < 4:                                  # textureless object: ring fallback
        for r in (r_in + 12, (r_in + r_out) / 2, r_out - 8):
            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                out.append(np.array([u + r * np.cos(a), v + r * np.sin(a)]))
    return np.asarray(out, np.float64)


def auto_seed_track(ep: Episode, subtask: Subtask, device: str = "cuda",
                    before: int = 3) -> Optional[SeedResult]:
    t_start = subtask.phases[0][1]
    t_end = min(ep.T - 1, subtask.phases[-1][2])
    t_seed = int(np.clip(subtask.t_close - before, t_start, t_end))
    frames = list(range(t_seed, t_end + 1))
    grip = _gripper_uv(ep, subtask.arm, frames)
    if not np.isfinite(grip[0]).all():
        return None

    gray = cv2.cvtColor(ep.read_color(t_seed), cv2.COLOR_BGR2GRAY)
    cand = _candidates(gray, grip[0])
    if len(cand) == 0:
        return None
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in ep.read_colors(frames)]
    uv, vis = track_points_cotracker(rgb, cand, device=device)   # (T,N,2),(T,N)

    i_close = max(0, subtask.t_close - t_seed)
    d = np.linalg.norm(uv - grip[:, None, :], axis=2)            # (T,N)
    post = d[i_close:]
    rigidity = post.std(axis=0)
    mean_d = post.mean(axis=0)
    movement = np.linalg.norm(uv - uv[0:1], axis=2).max(axis=0)
    vis_mean = vis.mean(axis=0)

    ok = ((vis_mean > 0.6) & (movement > 15) & (rigidity < 12)
          & (mean_d > 15) & (mean_d < 120))
    if not ok.any():
        return None
    idx = np.where(ok)[0]
    # prefer rigid, well-visible and moving; slight bonus for more movement
    score = rigidity[idx] + 0.15 * np.maximum(0, 40 - movement[idx]) - 5 * vis_mean[idx]
    n = int(idx[int(np.argmin(score))])

    h, w = ep.hw["cam_head"]
    out: dict[int, np.ndarray] = {}
    for i, f in enumerate(frames):
        p = uv[i, n]
        if vis[i, n] > 0.5 and np.isfinite(p).all() and 0 <= p[0] < w and 0 <= p[1] < h:
            out[f] = p.astype(np.float64)
    if not out:
        return None
    # reach prefix: object static before the query frame
    for f in range(t_start, t_seed):
        out[f] = cand[n].astype(np.float64)
    return SeedResult(
        seed_uv=(float(cand[n][0]), float(cand[n][1])), t_seed=t_seed, uv=out,
        qc={"rigidity": float(rigidity[n]), "movement": float(movement[n]),
            "mean_dist": float(mean_d[n]), "visibility": float(vis_mean[n]),
            "n_candidates": int(len(cand)), "n_ok": int(ok.sum())},
    )
