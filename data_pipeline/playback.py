"""Frame-by-frame playback renderer for one action chunk (OpenCV, no matplotlib).

Each chunk produces ``executed`` JPEG frames: the real RoboDojo RGB frame at
``t_cursor + k`` with

* the active arm's **7 near waypoints** (s0,s4,...,s24) as numbered dots,
* the **commanded EE trail** for the next 25 action steps (from ``action/*_ee_poses``),
* the current step highlighted, and a 25-cell step ruler at the bottom.

Why the frames rather than a static card: the point of the review is to see the
action chunk *execute*, and to see that near (7 points) and action (25 steps)
are two different clocks.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .chunks import Chunk
from .geometry import gripper_midpoint_3d, project_point
from .io import Episode
from .tracks import PlanTracks

__all__ = ["render_chunk_frames"]

# BGR
_C_WP = (60, 60, 235)        # near waypoint (red)
_C_WP_DIM = (120, 120, 120)  # waypoint already passed
_C_TRAIL = (90, 190, 90)     # commanded EE trail (green)
_C_TRAIL_FUT = (150, 150, 150)
_C_OBJ = (0, 150, 255)       # object keypoint (orange)
_C_OTHER = (190, 130, 70)    # the other arm's anchor (steel blue)
_C_OTHER_FUT = (120, 120, 120)
_C_CUR = (0, 215, 255)       # current step (yellow)
_C_TEXT = (245, 245, 245)
_C_SHADE = (40, 40, 40)


def _midpoint3d(pose: np.ndarray) -> np.ndarray:
    return gripper_midpoint_3d(pose)


def _project(points3d: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    return np.stack([project_point(p, K, E) for p in points3d])


def _put(img, text, org, color=_C_TEXT, scale=0.5, thick=1, outline=True) -> None:
    """Text helper.

    ``outline=True`` draws a black halo — only for **short** labels: OpenCV's
    thick glyphs advance slightly wider than thin ones, so a long outlined
    string renders a duplicate tail (found the hard way).
    """
    if not outline:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        return
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_ruler(img, step: int, executed: int, horizon: int, stride: int) -> None:
    H, W = img.shape[:2]
    x0, x1, y = 14, W - 14, H - 26
    cell = (x1 - x0) / horizon
    cv2.rectangle(img, (x0 - 4, y - 14), (x1 + 4, y + 10), _C_SHADE, -1)
    for s in range(horizon):
        a = int(x0 + s * cell)
        b = int(x0 + (s + 1) * cell) - 1
        if s < executed:
            cv2.rectangle(img, (a, y - 6), (b, y + 6), (60, 120, 60), -1)
        if s % stride == 0:
            cv2.line(img, (a, y - 9), (a, y + 9), _C_WP, 2)
        cv2.rectangle(img, (a, y - 6), (b, y + 6), (90, 90, 90), 1)
    cur = int(x0 + step * cell)
    cv2.rectangle(img, (cur, y - 9), (int(cur + cell), y + 9), _C_CUR, 2)
    txt = (f"step {step}/{horizon - 1}   executed 0..{executed - 1}   "
           f"near waypoints every {stride} steps")
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(img, (x0 - 4, y - 32), (x0 + tw + 4, y - 14), _C_SHADE, -1)
    _put(img, txt, (x0, y - 18), _C_TEXT, 0.42, outline=False)


def _draw_overlay(canvas, wp, tr, steps, k, stride, to_xy, scale=1.0, font=0.42,
                  label_all=True):
    """Draw trail + waypoints with a coordinate transform (main image or zoom)."""
    def pt(p):
        q = to_xy(p)
        return (int(round(q[0])), int(round(q[1])))

    for i in range(len(tr) - 1):
        if not (np.isfinite(tr[i]).all() and np.isfinite(tr[i + 1]).all()):
            continue
        color = _C_TRAIL if i <= k else _C_TRAIL_FUT
        cv2.line(canvas, pt(tr[i]), pt(tr[i + 1]), color,
                 max(1, int(round(2 * scale))), cv2.LINE_AA)
    for i, p in enumerate(tr):
        if not np.isfinite(p).all():
            continue
        if i <= k:
            cv2.circle(canvas, pt(p), max(1, int(round(3 * scale))), _C_TRAIL, -1, cv2.LINE_AA)
        elif i % stride == 0:
            cv2.drawMarker(canvas, pt(p), _C_TRAIL_FUT, cv2.MARKER_TILTED_CROSS,
                           max(5, int(round(7 * scale))), 1)

    for j in range(len(wp) - 1):
        if np.isfinite(wp[j]).all() and np.isfinite(wp[j + 1]).all():
            cv2.line(canvas, pt(wp[j]), pt(wp[j + 1]), _C_WP,
                     max(1, int(round(2 * scale))), cv2.LINE_AA)
    for j, p in enumerate(wp):
        if not np.isfinite(p).all():
            continue
        color = _C_WP_DIM if j * stride < k else _C_WP
        q = pt(p)
        cv2.circle(canvas, q, max(2, int(round(5 * scale))), color, -1, cv2.LINE_AA)
        cv2.circle(canvas, q, max(3, int(round(7 * scale))), (255, 255, 255), 1, cv2.LINE_AA)
        if not label_all and j not in (0, len(wp) - 1):
            continue
        dx = int(10 * scale) if j % 2 == 0 else -int(34 * scale)
        dy = -int(9 * scale) if j % 2 == 0 else int(14 * scale)
        _put(canvas, f"s{steps[j]}", (q[0] + dx, q[1] + dy), color, font)

    p = tr[k]
    if np.isfinite(p).all():
        q = pt(p)
        cv2.circle(canvas, q, max(3, int(round(9 * scale))), _C_CUR, 2, cv2.LINE_AA)
        cv2.circle(canvas, q, max(1, int(round(3 * scale))), _C_CUR, -1, cv2.LINE_AA)


def _draw_side_arm(img, wp, tr, k: int, label: str) -> None:
    """The non-active arm: thin trail + hollow near points + a small label."""
    for i in range(len(tr) - 1):
        if np.isfinite(tr[i]).all() and np.isfinite(tr[i + 1]).all():
            color = _C_OTHER if i <= k else _C_OTHER_FUT
            cv2.line(img, tuple(tr[i].astype(int)), tuple(tr[i + 1].astype(int)),
                     color, 1, cv2.LINE_AA)
    for p in wp:
        if np.isfinite(p).all():
            cv2.circle(img, tuple(p.astype(int)), 4, _C_OTHER, 1, cv2.LINE_AA)
    p = tr[k]
    if np.isfinite(p).all():
        q = tuple(p.astype(int))
        cv2.circle(img, q, 5, _C_OTHER, -1, cv2.LINE_AA)
        cv2.circle(img, q, 7, (255, 255, 255), 1, cv2.LINE_AA)
        _put(img, label, (q[0] + 8, q[1] + 4), _C_OTHER, 0.42)


def _draw_object(img, ch, k: int, stride: int) -> None:
    """Object keypoint: remaining path + the same 7 near waypoints (orange)."""
    if ch is None:
        return
    far, far_valid = ch["far"], ch["far_valid"]
    for i in range(len(far) - 1):
        if np.isfinite(far[i]).all() and np.isfinite(far[i + 1]).all():
            a, b = tuple(far[i].astype(int)), tuple(far[i + 1].astype(int))
            cv2.line(img, a, b, _C_OBJ, 1, cv2.LINE_AA)
    for i, p in enumerate(far):
        if np.isfinite(p).all():
            cv2.drawMarker(img, tuple(p.astype(int)), _C_OBJ, cv2.MARKER_TILTED_CROSS, 7, 1)
    near, nv = ch["near"], ch["near_valid"]
    for j, p in enumerate(near):
        if not np.isfinite(p).all():
            continue
        q = tuple(p.astype(int))
        if nv[j]:
            cv2.circle(img, q, 5, _C_OBJ, -1, cv2.LINE_AA)
            cv2.circle(img, q, 7, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, q, 6, _C_OBJ, 1, cv2.LINE_AA)
    ok = np.isfinite(near).all(axis=1)
    if ok.any() and nv.any():
        j = int(np.where(nv)[0][0])
        p = near[j]
        _put(img, "object", (int(p[0]) + 9, int(p[1]) + 22), _C_OBJ, 0.4)


def render_chunk_frames(
    ep: Episode,
    pt: PlanTracks,
    chunk: Chunk,
    out_dir: Path,
    task: str,
    active_arm: str = "R",
    quality: int = 82,
    tag: str | None = None,
) -> list[str]:
    arm = "left" if active_arm == "L" else "right"
    ee_state = ep.L if arm == "left" else ep.R
    ee_action = ep.ee_action[arm]
    K = ep.K["cam_head"]
    E = ep.E["cam_head"]
    H, W = ep.hw["cam_head"]
    stride = pt.stride
    n_near = len(chunk.near_steps)
    horizon = len(chunk.action)

    # 3D positions: near waypoints from measured poses, trail from commanded poses
    wp_frames = [min(ep.T - 1, chunk.t_cursor + j * stride) for j in range(n_near)]
    wp3d = np.stack([_midpoint3d(ee_state[f]) for f in wp_frames])
    tr_frames = [min(ep.T - 1, chunk.t_cursor + s) for s in range(horizon)]
    tr3d = np.stack([_midpoint3d(ee_action[f]) for f in tr_frames])

    # the other arm (always drawn too: both anchors are model inputs)
    other_arm = "left" if arm == "right" else "right"
    ee_other_state = ep.L if other_arm == "left" else ep.R
    ee_other_action = ep.ee_action[other_arm]
    wp3d_o = np.stack([_midpoint3d(ee_other_state[f]) for f in wp_frames])
    tr3d_o = np.stack([_midpoint3d(ee_other_action[f]) for f in tr_frames])
    label_o = "L" if other_arm == "left" else "R"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for k in range(chunk.executed_steps):
        frame = chunk.t_cursor + k
        raw = ep.read_color(frame)
        E_k = E[frame]
        wp = _project(wp3d, K, E_k)
        tr = _project(tr3d, K, E_k)
        wp_o = _project(wp3d_o, K, E_k)
        tr_o = _project(tr3d_o, K, E_k)
        img = raw.copy()

        # anchor (active arm): red 7 near waypoints + green 25-step EE trail + yellow cur
        _draw_overlay(img, wp, tr, chunk.near_steps, k, stride, lambda p: p, 1.0, 0.42,
                      label_all=False)
        # the other arm's anchor (both hands are conditions of the model)
        _draw_side_arm(img, wp_o, tr_o, k, label_o)
        # object keypoint: orange 7 near waypoints + path
        _draw_object(img, chunk.channels.get("object"), k, stride)

        # banner
        cv2.rectangle(img, (0, 0), (W, 24), _C_SHADE, -1)
        j = k // stride
        where = f"at s{chunk.near_steps[min(j, n_near - 1)]}" if k % stride == 0 else \
            f"between s{chunk.near_steps[min(j, n_near - 1)]} and " \
            f"s{chunk.near_steps[min(j + 1, n_near - 1)]}"
        obj = chunk.channels.get("object")
        obj_txt = f"  obj {int(obj['near_valid'].sum())}/7" if obj is not None else ""
        _put(img, f"{task} {pt.phase} chunk{chunk.index}  frame {frame}  "
                  f"step {k} (exec {k + 1}/{chunk.executed_steps})  {where}{obj_txt}",
             (8, 16), _C_TEXT, 0.44, outline=False)

        _draw_ruler(img, k, chunk.executed_steps, horizon, stride)

        stem = tag or pt.phase
        path = out_dir / f"{task}_{stem}_c{chunk.index}_f{k:02d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        paths.append(str(path))
    return paths

def render_phase_frames(
    ep: Episode,
    pt: PlanTracks,
    out_dir: Path,
    task: str,
    active_arm: str = "R",
    quality: int = 82,
    tag: str | None = None,
    corners: tuple[np.ndarray, np.ndarray, int] | None = None,
) -> list[str]:
    """One **phase** = one variable-length point trajectory; render every frame.

    No chunking here: the phase trajectory is stored as a whole and the
    train-time loader slides its own 7-point window. The video just shows the
    phase end to end, with both arms' anchors and the object point.
    """
    arm = "left" if active_arm == "L" else "right"
    other = "right" if arm == "left" else "left"
    K, E = ep.K["cam_head"], ep.E["cam_head"]
    h, w = ep.hw["cam_head"]
    frames = list(range(pt.t_start, pt.t_end + 1))
    n = len(frames)

    def mid3(poses, f):
        return gripper_midpoint_3d(poses[f])

    L3 = np.stack([mid3(ep.L, f) for f in frames])
    R3 = np.stack([mid3(ep.R, f) for f in frames])
    act3 = L3 if arm == "left" else R3
    oth3 = R3 if arm == "left" else L3
    Luv_px = np.stack([project_point(p, K, E[f]) for p, f in zip(L3, frames)])
    Ruv_px = np.stack([project_point(p, K, E[f]) for p, f in zip(R3, frames)])
    act_uv = Luv_px if arm == "left" else Ruv_px
    oth_uv = Ruv_px if arm == "left" else Luv_px
    act_col = (60, 60, 235) if arm == "left" else (105, 161, 56)
    oth_col = (190, 130, 70)

    # 6.25 Hz phase paths (pixel space; head camera is static)
    ch = {c.name: c for c in pt.channels}
    objs = [c for c in pt.channels if c.kind == "object"]
    obj_colors = [(0, 150, 255), (0, 0, 255), (0, 255, 255)]
    stride = pt.stride

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, t in enumerate(frames):
        img = ep.read_color(t).copy()
        E_t = E[t]
        # local phase path (last ~40 points); the full path is in the overview
        jcur = int(np.clip((t - pt.t_start) // stride, 0, max(0, pt.n_points - 1)))
        lo = max(0, jcur - 39)
        for c, color in ((ch.get("anchor_left"), (176, 108, 43)),
                         (ch.get("anchor_right"), (105, 161, 56))):
            if c is None:
                continue
            seg = c.uv[lo:jcur + 1]
            ok = np.isfinite(seg).all(axis=1)
            pts = seg[ok]
            for j in range(len(pts) - 1):
                cv2.line(img, tuple(pts[j].astype(int)), tuple(pts[j + 1].astype(int)),
                         color, 1, cv2.LINE_AA)
        for k, oc in enumerate(objs):
            color = obj_colors[k % 3]
            seg = oc.uv[lo:jcur + 1]
            ok = np.isfinite(seg).all(axis=1)
            pts = seg[ok]
            for j in range(len(pts) - 1):
                cv2.line(img, tuple(pts[j].astype(int)), tuple(pts[j + 1].astype(int)),
                         color, 1, cv2.LINE_AA)
            for j, p in enumerate(seg):
                if oc.valid[lo + j] and np.isfinite(p).all():
                    cv2.circle(img, tuple(p.astype(int)), 1, color, -1, cv2.LINE_AA)

        # trails (last 50 frames) + current anchors
        for uv_all, color, lab in ((act_uv, act_col, "R" if arm == "right" else "L"),
                                   (oth_uv, oth_col, "L" if other == "left" else "R")):
            uv = uv_all[max(0, i - 50):i + 1]
            for j in range(len(uv) - 1):
                if np.isfinite(uv[j]).all() and np.isfinite(uv[j + 1]).all():
                    cv2.line(img, tuple(uv[j].astype(int)), tuple(uv[j + 1].astype(int)),
                             color, 2, cv2.LINE_AA)
            p = uv_all[i]
            if np.isfinite(p).all():
                q = tuple(p.astype(int))
                cv2.circle(img, q, 6, color, -1, cv2.LINE_AA)
                cv2.circle(img, q, 8, (255, 255, 255), 1, cv2.LINE_AA)
                _put(img, lab, (q[0] + 9, q[1] + 4), color, 0.42)

        # all CoTracker corner tracks (small dots): this is the raw object
        # tracking, before the fused/median point used by the model.
        if corners is not None:
            cuv, cvis, ct0 = corners
            ci = t - int(ct0)
            if 0 <= ci < len(cuv):
                for n in range(cuv.shape[1]):
                    # short trail (last 5 frames) so the motion is visible,
                    # then a 1 px dot; low visibility becomes a hollow ring.
                    for k in range(max(0, ci - 5), ci):
                        a, b = cuv[k, n], cuv[k + 1, n]
                        if np.isfinite(a).all() and np.isfinite(b).all():
                            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                                     (200, 0, 200), 1, cv2.LINE_AA)
                    if np.isfinite(cuv[ci, n]).all():
                        p = tuple(cuv[ci, n].astype(int))
                        if cvis[ci, n] > 0.3:
                            cv2.circle(img, p, 1, (255, 0, 255), -1, cv2.LINE_AA)
                        else:
                            cv2.circle(img, p, 2, (255, 0, 255), 1, cv2.LINE_AA)

        # current object points (up to 3 slots)
        for k, oc in enumerate(objs):
            j = int(np.clip((t - pt.t_start) // stride, 0, max(0, len(oc) - 1)))
            if oc.valid[j] and np.isfinite(oc.uv[j]).all():
                p = tuple(oc.uv[j].astype(int))
                color = obj_colors[k % 3]
                cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)
                cv2.circle(img, p, 5, (255, 255, 255), 1, cv2.LINE_AA)
                _put(img, f"o{k}", (p[0] + 7, p[1] + 3), color, 0.38)

        # banner + phase progress
        cv2.rectangle(img, (0, 0), (w, 24), _C_SHADE, -1)
        _put(img, f"{task} {pt.phase} [{pt.t_start},{pt.t_end}]  frame {t}/{ep.T - 1}  "
                  f"phase {i + 1}/{n}  ({pt.n_points} pts @ {stride} steps)",
             (8, 16), _C_TEXT, 0.44, outline=False)
        y = h - 12
        x0, x1 = 12, w - 12
        cv2.rectangle(img, (x0, y - 4), (x1, y + 4), (70, 70, 70), 1)
        cx = int(x0 + (x1 - x0) * (i + 1) / max(1, n))
        cv2.rectangle(img, (x0, y - 4), (cx, y + 4), (60, 120, 60), -1)

        stem = tag or pt.phase
        path = out_dir / f"{task}_{stem}_f{i:04d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        paths.append(str(path))
    return paths
