"""One representative tracking panel per **task** (10 different tasks).

For each task: pick the subtask with the longest move phase, run SAM2+CoTracker,
and render all corner tracks + the fused object point at the frame of maximum
corner motion.  Also compose a 2x5 contact sheet.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import ROOT
from .diag_samples import _label, render_subtask
from .io import load_episode
from .object_seg import segment_and_track
from .task_timeline import segment_episode

TASKS = [
    "fill_egg_holder",
    "pour_balls_into_vase",
    "fold_clothes",
    "play_Xylophone",
    "sweep_blocks",
    "store_laptop_and_headphones",
    "plug_in_charger",
    "make_toast",
    "play_stacking_toy",
    "arrange_largest_number",
]


def _pick_subtask(subs):
    best, best_len = None, -1
    for s in subs:
        if s.arm not in ("L", "R"):
            continue
        mv = max((b - a for name, a, b in s.phases if name == "move"), default=-1)
        if mv > best_len:
            best, best_len = s, mv
    return best


def main() -> int:
    out = Path("outputs/diag/tasks")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in TASKS:
        p = ROOT / f"{task}/arx_x5/data/episode_0000000.hdf5"
        if not p.is_file():
            print(f"[!] {task}: missing episode")
            continue
        ep = load_episode(p, task=task, load_gray=False)
        cand = [x for x in segment_episode(ep) if x.arm in ("L", "R")]
        best = None
        for s in cand:
            r = segment_and_track(ep, s, device="cuda")
            if r is not None and r.corner_uv is not None:
                if best is None or r.qc.get("movement", 0) > best[1].qc.get("movement", 0):
                    best = (s, r)
        if best is None:
            sub = _pick_subtask(segment_episode(ep))
            res = None
        else:
            sub, res = best
        if res is None or res.corner_uv is None:
            # still show the chosen frame, marked as failure
            f = int(np.clip(sub.t_close, 0, ep.T - 1))
            img = ep.read_color(f)
            img = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((990, 1280, 3), np.uint8)
            canvas[:30] = (35, 35, 35); canvas[30:] = img
            _label(canvas, f"{task} #{sub.index+1}{sub.arm} close {sub.t_close}  "
                           f"SAM2/COTRACKER FAILED", (8, 21), 0.55)
            op = out / f"{task}.jpg"
            cv2.imwrite(str(op), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
            rows.append({"task": task, "sub": sub, "res": None, "path": op})
            print(f"[!] {task} #{sub.index+1}{sub.arm}: tracking failed")
            continue
        op = out / f"{task}.jpg"
        r = render_subtask(ep, sub, res, op)
        r["task"] = task
        rows.append(r)
        print(f"[ok] {task} #{sub.index+1}{sub.arm} T={ep.T} frame={r['frame']} "
              f"corners={r['corners']} track={r['qc'].get('track_fraction', 0):.2f} "
              f"rigid={r['qc'].get('rigidity', 0):.1f}")

    tiles = []
    for r in rows:
        im = cv2.imread(str(r["path"]))
        tiles.append(cv2.resize(im, (640, 512), interpolation=cv2.INTER_AREA))
    while len(tiles) < 10:
        tiles.append(np.zeros((512, 640, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i * 5:(i + 1) * 5]) for i in range(2)])
    sp = out / "tasks_sheet.jpg"
    cv2.imwrite(str(sp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("sheet", sp, sheet.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
