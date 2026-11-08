"""Render 10 sample subtasks: all corner tracks + fused point.

For each subtask, pick the frame with the largest median corner motion, draw
all CoTracker corner tracks (thin, palette), their current dots, the fused
object path (orange) and point, plus QC.  Also compose a 2x5 contact sheet.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import ROOT
from .io import load_episode
from .object_seg import segment_and_track
from .task_timeline import segment_episode

PALETTE = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 180, 255),
           (200, 0, 200), (0, 200, 200), (120, 80, 255), (80, 160, 255)]


def _label(img, text, org, scale=0.5, color=(240, 240, 240)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_subtask(ep, sub, res, out_path: Path, scale: int = 2) -> dict:
    t0 = res.t_seed
    cuv, cvis = res.corner_uv, res.corner_vis
    T = len(cuv)
    # frame with the largest median corner motion (after the grasp)
    sp = np.nanmedian(np.linalg.norm(np.diff(cuv, axis=0), axis=2), axis=1)
    lo = max(0, sub.t_close - t0)
    hi = T - 1
    if hi > lo:
        k = int(lo + np.argmax(sp[lo:hi]))
    else:
        k = T - 1
    frame = t0 + k

    base = ep.read_color(frame)
    h, w = base.shape[:2]
    img = cv2.resize(base, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    # all corner paths up to the chosen frame
    for n in range(cuv.shape[1]):
        col = PALETTE[n % len(PALETTE)]
        pts = cuv[:k + 1, n] * scale
        for j in range(len(pts) - 1):
            if np.isfinite(pts[j]).all() and np.isfinite(pts[j + 1]).all():
                cv2.line(img, tuple(pts[j].astype(int)), tuple(pts[j + 1].astype(int)),
                         col, 1, cv2.LINE_AA)
        p = pts[k]
        if np.isfinite(p).all():
            if cvis[k, n] > 0.3:
                cv2.circle(img, tuple(p.astype(int)), 2, col, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, tuple(p.astype(int)), 3, col, 1, cv2.LINE_AA)
    # fused path + point
    fu = np.stack([res.uv[f] for f in range(t0, frame + 1)]) * scale
    for j in range(len(fu) - 1):
        if np.isfinite(fu[j]).all() and np.isfinite(fu[j + 1]).all():
            cv2.line(img, tuple(fu[j].astype(int)), tuple(fu[j + 1].astype(int)),
                     (0, 150, 255), 2, cv2.LINE_AA)
    if len(fu):
        p = tuple(fu[-1].astype(int))
        cv2.circle(img, p, 5, (0, 150, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, 7, (255, 255, 255), 1, cv2.LINE_AA)

    q = res.qc
    title = (f"{ep.task} #{sub.index + 1}{sub.arm}  close/open {sub.t_close}/{sub.t_open}  "
             f"frame {frame}  corners {cuv.shape[1]}  "
             f"track {q.get('track_fraction', 0):.2f}  rigid {q.get('rigidity', 0):.1f}  "
             f"move {q.get('movement', 0):.0f}")
    canvas = np.zeros((img.shape[0] + 30, img.shape[1], 3), np.uint8)
    canvas[:30] = (35, 35, 35)
    canvas[30:] = img
    _label(canvas, title, (8, 21), 0.55)
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"task": ep.task, "index": sub.index, "arm": sub.arm,
            "frame": int(frame), "corners": int(cuv.shape[1]), "path": out_path,
            "qc": q}


def main() -> int:
    out = Path("outputs/diag/samples")
    out.mkdir(parents=True, exist_ok=True)
    tasks = ["fill_pen_holder", "stack_bowls", "insert_tubes"]
    results = []
    for task in tasks:
        ep = load_episode(ROOT / f"{task}/arx_x5/data/episode_0000000.hdf5",
                          task=task, load_gray=False)
        for sub in segment_episode(ep):
            res = segment_and_track(ep, sub, device="cuda")
            if res is None or res.corner_uv is None:
                print(f"[!] {task} #{sub.index+1}: no tracks")
                continue
            p = out / f"{task}_s{sub.index}.jpg"
            r = render_subtask(ep, sub, res, p)
            results.append(r)
            print(f"[ok] {r['task']} #{r['index']+1}{r['arm']}  frame={r['frame']}  "
                  f"corners={r['corners']}  track={r['qc'].get('track_fraction', 0):.2f}")

    # 2 x 5 contact sheet
    tiles = []
    for r in results:
        im = cv2.imread(str(r["path"]))
        tiles.append(cv2.resize(im, (640, 512), interpolation=cv2.INTER_AREA))
    while len(tiles) < 10:
        tiles.append(np.zeros((512, 640, 3), np.uint8))
    rows = [np.hstack(tiles[i * 5:(i + 1) * 5]) for i in range(2)]
    sheet = np.vstack(rows)
    sp = out / "samples_sheet.jpg"
    cv2.imwrite(str(sp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("sheet", sp, sheet.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
