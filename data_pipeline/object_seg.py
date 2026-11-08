"""SAM2 object mask → corner seeds → CoTracker.

Why: the old seed was a single manual pixel per task (or corners in a ring
around the gripper), which is fragile for thin / unknown objects. Here we

1. prompt **SAM2** at the grasp frame with the two-finger midpoint as a
   positive point and the gripper body (10 cm behind the TCP) as a negative
   point → object mask (robot arm masks score badly and are rejected),
2. take Shi-Tomasi corners **inside the mask** (grid fallback for textureless
   objects),
3. track all candidates with CoTracker, score them by rigidity to the gripper
   after the grasp + movement + visibility,
4. output the **median** of the top-k tracks (robust to one drifting point).

The package is loaded from the shared checkout (``config.SAM2_CODE``); the
857 MB hiera-large checkpoint is local (``config.SAM2_CKPT``).
"""
from __future__ import annotations

import sys
from typing import Optional

import cv2
import numpy as np

from .autoseed import SeedResult
from .config import SAM2_CKPT, SAM2_CODE, SAM2_CONFIG, TCP_OFFSET_M
from .cotracker_track import track_points_cotracker
from .geometry import gripper_midpoint_3d, project_point_depth, quat_to_R
from .io import Episode
from .task_timeline import Subtask

__all__ = ["segment_and_track", "sam2_available", "segment_object"]

_PRED = None


def sam2_available() -> bool:
    try:
        _ensure_path()
        import sam2  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _ensure_path() -> None:
    if SAM2_CODE not in sys.path:
        sys.path.insert(0, SAM2_CODE)


def _predictor(device: str = "cuda"):
    global _PRED
    if _PRED is None:
        _ensure_path()
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)
        _PRED = SAM2ImagePredictor(model)
    return _PRED


def _project(pw, K, E):
    uv, z = project_point_depth(np.asarray(pw, np.float64), K, E)
    return uv, z


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes (e.g. the dark opening of a pen holder) before picking points."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (mask | (ff == 0)).astype(bool)


def _mask_candidates(mask: np.ndarray, gray: np.ndarray, max_corners: int = 28) -> np.ndarray:
    m = _fill_holes(mask)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # interior points first: a corner on the rim is the first thing the gripper
    # occludes, and CoTracker then slides onto the finger.
    er = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=1)
    if er.sum() < 20:
        er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01,
                                  minDistance=6, mask=er)
    corners = pts.reshape(-1, 2) if pts is not None else np.zeros((0, 2), np.float64)
    # distance-transform peaks = well inside the object body
    dt = cv2.distanceTransform(er, cv2.DIST_L2, 5)
    peaks = np.zeros_like(er)
    peaks[(dt == cv2.dilate(dt, np.ones((7, 7), np.uint8))) & (dt > 1.5)] = 255
    ys, xs = np.nonzero(peaks)
    interior = np.stack([xs, ys], axis=1).astype(np.float64) if len(xs) else np.zeros((0, 2))
    out = np.concatenate([interior, corners], axis=0) if len(corners) else interior
    if len(out) < 6:                       # textureless / tiny: grid inside the mask
        ys, xs = np.nonzero(er)
        if len(xs):
            take = np.linspace(0, len(xs) - 1, min(max_corners, len(xs))).astype(int)
            out = np.concatenate([out, np.stack([xs[take], ys[take]], axis=1)], axis=0)
    # deduplicate, keep interior points first
    keep = []
    for p in out:
        if all(np.linalg.norm(p - q) > 3 for q in keep):
            keep.append(p)
        if len(keep) >= max_corners:
            break
    return np.asarray(keep, np.float64)


def segment_object(ep: Episode, frame: int, arm: str, device: str = "cuda",
                   predictor=None) -> Optional[dict]:
    """Return the best object mask at ``frame`` (or None).

    Positive prompt: TCP (between the fingers). Negative prompt: gripper body
    (and wrist when visible). Masks overlapping the gripper body are rejected.
    """
    pred = predictor or _predictor(device)
    rgb = cv2.cvtColor(ep.read_color(frame), cv2.COLOR_BGR2RGB)
    poses = ep.L if arm == "L" else ep.R
    pose = poses[frame]
    Rm = quat_to_R(pose[3:])
    tcp = pose[:3] + Rm @ np.asarray(TCP_OFFSET_M, np.float64)
    body = tcp - 0.10 * Rm[:, 0]
    K, E = ep.K["cam_head"], ep.E["cam_head"][frame]
    uv_tcp, z_tcp = _project(tcp, K, E)
    uv_body, z_body = _project(body, K, E)
    if z_tcp <= 1e-6 or not np.isfinite(uv_tcp).all():
        return None
    h, w = rgb.shape[:2]
    pts = [uv_tcp]
    labels = [1]
    for p in (uv_body,):
        if np.isfinite(p).all() and 0 <= p[0] < w and 0 <= p[1] < h:
            pts.append(p)
            labels.append(0)
    pts = np.asarray(pts, np.float64)

    pred.set_image(rgb)
    masks, scores, _logits = pred.predict(point_coords=pts,
                                          point_labels=np.asarray(labels, np.int32),
                                          multimask_output=True)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float64)
    u0, v0 = float(uv_tcp[0]), float(uv_tcp[1])
    body_disk = np.zeros((h, w), np.uint8)
    if labels[-1] == 0:
        bx, by = int(uv_body[0]), int(uv_body[1])
        if 0 <= bx < w and 0 <= by < h:
            cv2.circle(body_disk, (bx, by), 25, 1, -1)
    body_disk = body_disk.astype(bool)

    best, best_score, best_info = None, -1e9, {}
    for m, s in zip(masks, scores):
        m = m.astype(bool)
        area = float(m.mean())
        if area < 3e-4 or area > 0.15:
            continue
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        dist = float(np.hypot(cx - u0, cy - v0))
        if dist > 100:
            continue
        body_ov = float((m & body_disk).sum() / max(1, body_disk.sum()))
        if body_ov > 0.6:
            continue
        in_tcp = bool(m[int(np.clip(v0, 0, h - 1)), int(np.clip(u0, 0, w - 1))])
        size_pen = abs(np.log(max(area, 1e-8) / 0.008))
        score = (3.0 * float(s) - 0.02 * dist + 0.003 * float(sat[m].mean())
                 - 3.0 * body_ov + 0.3 * float(in_tcp) - 0.3 * size_pen)
        if score > best_score:
            best, best_score = m, score
            best_info = {"sam_score": float(s), "mask_area": area,
                         "mask_dist": dist, "mask_sat": float(sat[m].mean()),
                         "body_overlap": body_ov, "in_tcp": in_tcp,
                         "tcp_uv": [round(u0, 2), round(v0, 2)],
                         "body_uv": ([round(float(uv_body[0]), 2),
                                      round(float(uv_body[1]), 2)]
                                     if np.isfinite(uv_body).all() else None)}
    if best is None:
        return None
    # keep the component containing / closest to the TCP
    num, labels_img, stats, cents = cv2.connectedComponentsWithStats(best.astype(np.uint8), 8)
    if num > 1:
        scores_c = []
        for k in range(1, num):
            comp = labels_img == k
            d = np.hypot(cents[k][0] - u0, cents[k][1] - v0)
            scores_c.append((int(comp[int(np.clip(v0, 0, h - 1)), int(np.clip(u0, 0, w - 1))]), -d, k))
        k = max(scores_c)[2]
        best = labels_img == k
    best_info["t_seed"] = int(frame)
    best_info["score"] = float(best_score)
    return {"mask": best, "info": best_info}


def fuse_object_track(uv: np.ndarray, vis: np.ndarray, grip: np.ndarray,
                      idx: np.ndarray, i_close: int, i_open: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-aware fusion of CoTracker tracks.

    * before grasp: object is static → CoTracker if visible, else the seed;
    * grasped (close..open): CoTracker if visible **or the 5 points agree**
      (spread < 18 px); otherwise the rigid offset prior. A pen rotating in the
      fingers makes the 2D offset change by tens of px, so the prior is only a
      short-term occlusion fallback, not the primary signal;
    * after release: **freeze** at the release position — the object stays where
      it was placed while the gripper retracts.

    Returns ``(fused (T,2), trust (T,), prior (T,2))``.
    """
    T = len(uv)
    K = len(idx)
    pts = uv[:, idx]                                   # (T,K,2)
    pred = grip[:, None, :] + (uv[0, idx][None, :, :] - grip[0][None, None, :])
    prior = np.nanmedian(pred, axis=1)
    spread = np.zeros(T, np.float64)
    for i in range(T):
        P = pts[i]
        if np.isfinite(P).all():
            dd = np.linalg.norm(P[:, None] - P[None, :], axis=2)
            spread[i] = float(dd.max())
    # coherent but wrong tracks (e.g. sliding onto the gripper) show a large
    # jump away from the rigid prior; 80 px accepts pen rotation (40–60) but
    # rejects the cup's 200 px jump at the end.
    res_med = np.nanmedian(np.linalg.norm(pts - pred, axis=2), axis=1)
    trust = (res_med < 80) & ((vis[:, idx] > 0.2).any(axis=1) | (spread < 18))
    fused = np.empty((T, 2), np.float64)
    for i in range(T):
        if i < i_close:
            fused[i] = (np.nanmedian(pts[i], axis=0)
                        if (vis[i, idx] > 0.2).any() or spread[i] < 18
                        else uv[0, idx].mean(axis=0))
        elif i <= i_open:
            fused[i] = np.nanmedian(pts[i], axis=0) if trust[i] else prior[i]
        else:
            fused[i] = fused[i_open]
    return fused, trust, prior


def segment_and_track(ep: Episode, sub: Subtask, device: str = "cuda",
                      max_candidates: int = 28, top_k: int = 5) -> Optional[SeedResult]:
    """SAM2 mask + CoTracker corners for one subtask. None → caller falls back."""
    t0 = sub.phases[0][1]
    t1 = sub.phases[-1][2]
    base = int(np.clip(sub.t_close - 3, t0, t1))
    tries = [base, base - 3, base - 6, base + 2]
    tries = [t for i, t in enumerate(tries) if t0 <= t <= t1 and t not in tries[:i]]

    pred = _predictor(device)
    chosen = None
    for t in tries:
        r = segment_object(ep, t, sub.arm, device=device, predictor=pred)
        if r is not None:
            chosen = r
            break
    if chosen is None:
        return None

    mask, info = chosen["mask"], chosen["info"]
    t_seed = int(info["t_seed"])
    rgb = cv2.cvtColor(ep.read_color(t_seed), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cand = _mask_candidates(mask, gray, max_candidates)
    if len(cand) < 4:
        return None
    info["candidates"] = cand

    frames = list(range(t_seed, t1 + 1))
    rgb_frames = [cv2.cvtColor(ep.read_color(f), cv2.COLOR_BGR2RGB) for f in frames]
    uv, vis = track_points_cotracker(rgb_frames, cand, device=device)
    if uv.shape[0] == 0:
        return None

    poses = ep.L if sub.arm == "L" else ep.R
    K, E = ep.K["cam_head"], ep.E["cam_head"]
    grip = np.stack([_project(gripper_midpoint_3d(poses[f]), K, E[f])[0] for f in frames])
    i_close = max(0, sub.t_close - t_seed)
    d = np.linalg.norm(uv - grip[:, None, :], axis=2)
    post = d[i_close:]
    rigidity = post.std(axis=0)
    mean_d = post.mean(axis=0)
    movement = np.linalg.norm(uv - uv[0:1], axis=2).max(axis=0)
    vis_mean = vis.mean(axis=0)

    # Thin objects (pens/tubes) rotate slightly in the fingers and CoTracker
    # visibility drops when they are occluded; 15 px / 0.5 was too strict.
    ok = ((vis_mean > 0.2) & (movement > 8) & (rigidity < 30)
          & (mean_d > 5) & (mean_d < 200))
    if not ok.any():
        return None
    idx = np.where(ok)[0]
    idx = idx[np.argsort(rigidity[idx])][:top_k]
    # phase-aware fusion (see fuse_object_track)
    i_open = min(max(0, sub.t_open - t_seed), len(frames) - 1)
    med, trust, prior = fuse_object_track(uv, vis, grip, idx, i_close, i_open)
    med_vis = np.nanmedian(vis[:, idx], axis=1)
    if not np.isfinite(med).all():
        return None
    med_vis = np.nanmedian(vis[:, idx], axis=1)
    if not np.isfinite(med).all():
        return None
    med_move = float(np.linalg.norm(med - med[0:1], axis=1).max())
    med_rig = float(np.linalg.norm(med[i_close:i_open + 1] - grip[i_close:i_open + 1],
                                   axis=1).std())
    if med_move < 8:
        return None

    h, w = ep.hw["cam_head"]
    out: dict[int, np.ndarray] = {}
    for i, f in enumerate(frames):
        p = med[i]
        if np.isfinite(p).all() and 0 <= p[0] < w and 0 <= p[1] < h:
            out[f] = p.astype(np.float64)
    for f in range(t0, t_seed):
        out[f] = med[0].astype(np.float64)

    info_qc = {k: v for k, v in info.items() if k != "candidates"}
    qc = {**info_qc,
          "rigidity": med_rig, "movement": med_move,
          "visibility": float(np.nanmean(med_vis)),
          "n_candidates": int(len(cand)), "n_ok": int(len(idx)),
          "track_fraction": float(trust.mean()),
          "method": "sam2+cotracker+rigid"}
    return SeedResult(seed_uv=(float(med[0][0]), float(med[0][1])), t_seed=t_seed,
                      uv=out, qc=qc, mask=mask, info=info,
                      corner_uv=uv, corner_vis=vis, corner_t0=t_seed)
