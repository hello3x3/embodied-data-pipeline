"""One-off diagnostic: show every stage of the object-seed pipeline.

Stages:
  1 raw frame at t_seed
  2 SAM2 three masks + scores
  3 chosen mask + interior/corner candidates + TCP/body prompts
  4 CoTracker tracks at t_seed
  5 CoTracker tracks at t_close
  6 CoTracker tracks at the end
  7 fused output vs rigid prior
  8 per-frame visibility / residual / source (CoTracker or prior)

Usage:
  python -m data_pipeline.diag_object --task fill_pen_holder --sub 0 \
      --out outputs/diag/obj_fill_s0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from .config import ROOT                 # noqa: E402
from .cotracker_track import track_points_cotracker  # noqa: E402
from .geometry import gripper_midpoint_3d, project_point_depth, quat_to_R  # noqa: E402
from .io import load_episode             # noqa: E402
from .object_seg import (_mask_candidates, _predictor,  # noqa: E402
                         fuse_object_track, segment_object)
from .task_timeline import segment_episode  # noqa: E402


def _blend(img, mask, color, alpha=0.35, thick=2):
    out = img.copy()
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(color, np.float64) * alpha).astype(np.uint8)
    cnt = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(out, cnt, -1, color, thick)
    return out


def _draw_tracks(img, uv, upto, color_all=(110, 110, 110), sel=None,
                 fused=None, colors=((0, 0, 255), (0, 255, 0), (255, 0, 0),
                                     (0, 200, 255), (255, 0, 255))):
    out = img.copy()
    palette = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 180, 255),
               (200, 0, 200), (0, 200, 200), (120, 80, 255), (80, 160, 255)]
    for n in range(uv.shape[1]):
        col = palette[n % len(palette)]
        pts = uv[:upto + 1, n]
        for j in range(len(pts) - 1):
            if np.isfinite(pts[j]).all() and np.isfinite(pts[j + 1]).all():
                cv2.line(out, tuple(pts[j].astype(int)), tuple(pts[j + 1].astype(int)),
                         col, 1, cv2.LINE_AA)
        p = uv[upto, n]
        if np.isfinite(p).all():
            cv2.circle(out, tuple(p.astype(int)), 1, col, -1, cv2.LINE_AA)
    if sel is not None:
        for n in sel:
            p = uv[upto, n]
            if np.isfinite(p).all():
                cv2.circle(out, tuple(p.astype(int)), 2, (0, 0, 0), -1)
    if fused is not None:
        p = fused[upto]
        if np.isfinite(p).all():
            q = tuple(p.astype(int))
            cv2.circle(out, q, 4, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fill_pen_holder")
    ap.add_argument("--sub", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/diag/obj"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ep = load_episode(ROOT / f"{args.task}/arx_x5/data/episode_0000000.hdf5",
                      task=args.task, load_gray=False)
    sub = segment_episode(ep)[args.sub]
    t0, t1 = sub.phases[0][1], sub.phases[-1][2]
    t_seed = int(np.clip(sub.t_close - 3, t0, t1))
    print(f"task={args.task} sub=#{sub.index+1}{sub.arm} close={sub.t_close} "
          f"open={sub.t_open} t_seed={t_seed}")

    pred = _predictor(args.device)
    rgb = cv2.cvtColor(ep.read_color(t_seed), cv2.COLOR_BGR2RGB)
    poses = ep.L if sub.arm == "L" else ep.R
    pose = poses[t_seed]
    Rm = quat_to_R(pose[3:])
    tcp = pose[:3] + Rm @ np.array([0.145, 0.0, 0.0])
    body = tcp - 0.10 * Rm[:, 0]
    K, E = ep.K["cam_head"], ep.E["cam_head"][t_seed]
    uv_tcp, zt = project_point_depth(tcp, K, E)
    uv_body, zb = project_point_depth(body, K, E)
    pts = [uv_tcp]
    labels = [1]
    if np.isfinite(uv_body).all() and zb > 1e-6:
        pts.append(uv_body); labels.append(0)
    pred.set_image(rgb)
    masks, scores, _ = pred.predict(point_coords=np.asarray(pts, np.float64),
                                    point_labels=np.asarray(labels, np.int32),
                                    multimask_output=True)

    chosen = segment_object(ep, t_seed, sub.arm, device=args.device, predictor=pred)
    if chosen is None:
        print("SAM2 returned no mask")
        return 1
    mask, info = chosen["mask"], chosen["info"]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cand = _mask_candidates(mask, gray, 28)
    print("tcp", np.round(uv_tcp, 1), "body", np.round(uv_body, 1))
    print("sam scores", np.round(scores, 3), "chosen", info)
    print("candidates", len(cand))

    frames = list(range(t_seed, t1 + 1))
    rgbs = [cv2.cvtColor(ep.read_color(f), cv2.COLOR_BGR2RGB) for f in frames]
    uv, vis = track_points_cotracker(rgbs, cand, device=args.device)
    grip = np.stack([project_point_depth(gripper_midpoint_3d(poses[f]), K, E)[0]
                     for f in frames]) if False else np.stack(
        [project_point_depth(gripper_midpoint_3d(poses[f]), K, ep.E["cam_head"][f])[0]
         for f in frames])
    i_close = max(0, sub.t_close - t_seed)
    d = np.linalg.norm(uv - grip[:, None, :], axis=2)
    post = d[i_close:]
    rigidity = post.std(axis=0)
    mean_d = post.mean(axis=0)
    movement = np.linalg.norm(uv - uv[0:1], axis=2).max(axis=0)
    vis_mean = vis.mean(axis=0)
    ok = ((vis_mean > 0.2) & (movement > 8) & (rigidity < 30)
          & (mean_d > 5) & (mean_d < 200))
    idx = np.where(ok)[0]
    idx = idx[np.argsort(rigidity[idx])][:5]
    i_open = min(max(0, sub.t_open - t_seed), len(frames) - 1)
    fused, src, prior = fuse_object_track(uv, vis, grip, idx, i_close, i_open)
    pred_rigid = grip[:, None, :] + (uv[0, idx][None] - grip[0][None])
    res = np.linalg.norm(uv[:, idx] - pred_rigid, axis=2)
    print(f"ok {len(idx)}/{len(cand)} track_fraction={src.mean():.2f} "
          f"rigidity={np.linalg.norm(fused[i_close:i_open+1] - grip[i_close:i_open+1], axis=1).std():.1f}")

    raw = ep.read_color(t_seed)
    # 2: three masks
    m3 = raw.copy()
    for k, (m, s) in enumerate(zip(masks, scores)):
        m3 = _blend(m3, m, [(0, 0, 255), (0, 255, 0), (255, 0, 0)][k], 0.28, 1)
        ys, xs = np.nonzero(m)
        if len(xs):
            cv2.putText(m3, f"m{k} {s:.2f}", (int(xs.mean()), int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    # 3: chosen mask + candidates
    m1 = _blend(raw, mask, (0, 255, 0), 0.3, 2)
    for p in cand:
        cv2.circle(m1, tuple(np.round(p).astype(int)), 1, (0, 0, 255), -1)
    cv2.circle(m1, tuple(np.round(uv_tcp).astype(int)), 4, (0, 255, 255), -1)
    if np.isfinite(uv_body).all():
        cv2.circle(m1, tuple(np.round(uv_body).astype(int)), 3, (255, 0, 255), -1)

    i_mid = len(frames) // 2
    panels = [
        (raw, "1 raw @ t_seed"),
        (m3, "2 SAM2 masks (3 candidates)"),
        (m1, "3 chosen mask + candidates (red) / TCP (yellow) / body (magenta)"),
        (_draw_tracks(raw, uv, 0, sel=idx, fused=fused), f"4 tracks @ t_seed"),
        (_draw_tracks(ep.read_color(frames[i_close]), uv, i_close, sel=idx,
                      fused=fused), f"5 tracks @ close ({sub.t_close})"),
        (_draw_tracks(ep.read_color(frames[i_mid]), uv, i_mid, sel=idx,
                      fused=fused), f"6 tracks @ mid ({frames[i_mid]})"),
        (_draw_tracks(ep.read_color(frames[-1]), uv, len(frames) - 1, sel=idx,
                      fused=fused), f"7 tracks @ end ({frames[-1]})"),
    ]

    fig = plt.figure(figsize=(19, 12), dpi=100)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05])
    for k, (img, title) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    # 8: fused vs CoTracker vs rigid prior at the end frame
    img8 = ep.read_color(frames[-1]).copy()
    trk_med = np.empty((len(frames), 2))
    for i in range(len(frames)):
        trk_med[i] = np.nanmedian(uv[i, idx], axis=0)
    for i in range(len(frames) - 1):
        a, b = fused[i], fused[i + 1]
        if np.isfinite(a).all() and np.isfinite(b).all():
            cv2.line(img8, tuple(a.astype(int)), tuple(b.astype(int)), (255, 255, 0), 2)
        a, b = pred_rigid[i].mean(0), pred_rigid[i + 1].mean(0)
        cv2.line(img8, tuple(a.astype(int)), tuple(b.astype(int)), (255, 0, 255), 1)
        if np.isfinite(trk_med[i]).all() and np.isfinite(trk_med[i + 1]).all():
            cv2.line(img8, tuple(trk_med[i].astype(int)), tuple(trk_med[i + 1].astype(int)),
                     (0, 255, 255), 1)
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(cv2.cvtColor(img8, cv2.COLOR_BGR2RGB))
    ax.set_title("8 yellow=fused  cyan=CoTracker  magenta=rigid prior", fontsize=9)
    ax.axis("off")
    # 9: diagnostics plot
    ax = fig.add_subplot(gs[2, 1:])
    x = np.arange(len(frames))
    ax.plot(x, np.nanmedian(vis[:, idx], axis=1), color="#1f77b4", lw=1, label="visibility (median)")
    ax.plot(x, np.nanmedian(res, axis=1), color="#d62728", lw=1, label="|CoTracker - rigid prior| px")
    ax.axhline(30, color="#d62728", ls=":", lw=0.8)
    ax.fill_between(x, 0, 1.05, where=src, color="#2ca02c", alpha=0.12, label="CoTracker accepted")
    ax.axvline(i_close, color="k", ls="--", lw=0.8)
    ax.set_ylim(0, 1.05 * max(1.0, np.nanmax(res)))
    ax.set_xlabel("frame in subtask")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"fused rigidity {np.linalg.norm(fused[i_close:i_open+1] - grip[i_close:i_open+1], axis=1).std():.1f}px · "
                 f"CoTracker/agreement {src.mean():.0%} of frames", fontsize=9)
    fig.tight_layout()
    p = out / f"{args.task}_s{sub.index}.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print("saved", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
