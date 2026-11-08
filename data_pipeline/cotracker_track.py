"""CoTracker3 wrapper for one seeded object point.

CoTracker does **not** detect objects: it propagates a query ``(frame, u, v)``
through a clip. This module only wraps the model; seeding stays in the caller
(currently the per-task manual ``SEEDS``, later an automatic grasp-time seed).

Usage::

    uv, vis = track_point_cotracker(rgb_frames, seed_uv, thr=0.5)
    # uv: (T, 2) pixel coords, vis: (T,) in [0, 1]

The clip should start at the query frame (backward tracking is not used; the
reach prefix is filled with the static seed by the caller).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["DEFAULT_CKPT", "cotracker_available", "track_point_cotracker",
           "track_points_cotracker"]

DEFAULT_CKPT = Path(os.environ.get(
    "COTRACKER_CKPT", "/mnt/nas/share/weights/cotracker/scaled_offline.pth"))
_MODEL = None
_DEVICE = None


def cotracker_available() -> bool:
    try:
        import cotracker  # noqa: F401
    except ImportError:
        return False
    return DEFAULT_CKPT.is_file()


def _load(device: str = "cuda", window_len: int = 60):
    global _MODEL, _DEVICE
    if _MODEL is not None and _DEVICE == device:
        return _MODEL
    import torch
    from cotracker.predictor import CoTrackerPredictor

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = CoTrackerPredictor(checkpoint=str(DEFAULT_CKPT), offline=True,
                               window_len=window_len).to(device).eval()
    _MODEL, _DEVICE = model, device
    return model


def _track_window(frames_rgb, seeds_uv, query_index, device, window_len):
    import torch

    video = torch.from_numpy(np.stack([np.asarray(f, np.uint8) for f in frames_rgb]))
    video = video.permute(0, 3, 1, 2)[None].float()
    q = [[float(query_index), float(u), float(v)] for u, v in seeds_uv]
    queries = torch.tensor([q], dtype=torch.float32)                 # (1,N,3)
    model = _load(device, window_len=window_len)
    with torch.no_grad():
        tracks, vis = model(video.to(device), queries=queries.to(device))
    return (tracks[0].detach().cpu().numpy().astype(np.float32),
            vis[0].detach().cpu().numpy().astype(np.float32))


def track_points_cotracker(
    frames_rgb: Sequence[np.ndarray],
    seeds_uv: Sequence[Sequence[float]],
    query_index: int = 0,
    device: str = "cuda",
    window_len: int = 60,
    max_window: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Track several points at once: ``(uv (T,N,2), visibility (T,N))``.

    Long clips are tracked in windows of ``max_window`` frames, re-querying at
    the previous window's last position. This bounds CoTracker's memory (an
    854-frame clip OOMs on a 24 GB card).
    """
    if len(frames_rgb) == 0:
        raise ValueError("frames_rgb 为空")
    if len(seeds_uv) == 0:
        raise ValueError("seeds_uv 为空")
    T, N = len(frames_rgb), len(seeds_uv)
    if T <= max_window:
        return _track_window(frames_rgb, seeds_uv, query_index, device, window_len)

    uv_out = np.zeros((T, N, 2), np.float32)
    vis_out = np.zeros((T, N), np.float32)
    cur, q = 0, [tuple(map(float, p)) for p in seeds_uv]
    while cur < T:
        end = min(T, cur + max_window)
        uv, vis = _track_window(frames_rgb[cur:end], q, 0, device, window_len)
        uv_out[cur:end] = uv
        vis_out[cur:end] = vis
        q = [tuple(p) for p in uv[-1]]
        cur = end
    return uv_out, vis_out


def track_point_cotracker(
    frames_rgb: Sequence[np.ndarray],
    seed_uv: Sequence[float],
    query_index: int = 0,
    device: str = "cuda",
    window_len: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Track one point through ``frames_rgb`` (query frame = ``query_index``).

    Returns ``(uv (T,2) float32, visibility (T,) float32)`` in pixel coords.
    Long clips are windowed by :func:`track_points_cotracker`.
    """
    uv, vis = track_points_cotracker(frames_rgb, [seed_uv], query_index=query_index,
                                     device=device, window_len=window_len)
    return uv[:, 0], vis[:, 0]
