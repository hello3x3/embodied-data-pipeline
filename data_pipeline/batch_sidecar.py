"""Batch-build sidecars for every episode, **without object tracking / frames**.

This is the cheap full-dataset path:

    load_episode -> segment_episode -> anchor tracks -> JSON

No SAM2, no CoTracker, no JPEG rendering.  It is meant for cleaning thousands
of episodes (e.g. 35 tasks x 100 episodes) into ``sidecar/*.json``.

Usage:
    python -m data_pipeline.batch_sidecar --workers 8 \
        --out outputs/robodojo_sidecars
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .build_plan_queue import _sidecar_phase
from .config import POINT_STRIDE, QUAT_ORDER, ROOT, TCP_OFFSET_M
from .io import load_episode
from .plans import PlanRange
from .task_timeline import segment_episode
from .tracks import build_plan_tracks, project_anchors


def _episodes(root: Path, task: str) -> list[Path]:
    return sorted((root / task / "arx_x5" / "data").glob("episode_*.hdf5"))


def _build_one(job):
    task, path_s, out_root_s, overwrite = job
    path, out_root = Path(path_s), Path(out_root_s)
    out = out_root / task / f"{task}__{path.stem}.json"
    if out.exists() and not overwrite:
        return ("skip", task, path.stem, 0, 0, out.stat().st_size)
    try:
        ep = load_episode(path, task=task, load_gray=False)
        subs = [s for s in segment_episode(ep) if s.arm in ("L", "R")]
        side = []
        if subs:
            Luv, Lv, Ruv, Rv = project_anchors(ep)
            for s in subs:
                phases = []
                for ph, a, b in s.phases:
                    pt = build_plan_tracks(
                        ep, PlanRange(ph, a, b), Luv, Ruv, {}, s.arm,
                        POINT_STRIDE, Lvalid=Lv, Rvalid=Rv)
                    phases.append(_sidecar_phase(
                        pt, f"{task}/{path.stem}/s{s.index}/{ph}"))
                side.append({
                    "index": s.index, "arm": s.arm,
                    "t_close": int(s.t_close), "t_open": int(s.t_open),
                    "object_qc": {}, "mask_image": None, "object_corners": None,
                    "phases": phases,
                })
        h, w = ep.hw["cam_head"]
        payload = {
            "episode": str(path.relative_to(ROOT)),
            "task": task, "fps": int(ep.fps), "stride": int(POINT_STRIDE),
            "image_size": [int(h), int(w)],
            "tcp_offset": [float(x) for x in TCP_OFFSET_M],
            "quat_order": QUAT_ORDER,
            "status": "ok" if side else "no_grasp",
            "subtasks": side,
        }
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return ("ok" if side else "no_grasp", task, path.stem, ep.T,
                len(side), len(text))
    except Exception as exc:  # noqa: BLE001
        return ("error", task, path.stem, str(exc), traceback.format_exc(limit=2), 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=Path("outputs/robodojo_sidecars"))
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="max episodes per task (0=all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.tasks:
        tasks = args.tasks
    else:
        tasks = sorted(p.name for p in args.root.iterdir()
                       if (p / "arx_x5" / "data").is_dir())
    jobs = []
    for task in tasks:
        eps = _episodes(args.root, task)
        if args.limit:
            eps = eps[:args.limit]
        jobs.extend((task, str(p), str(args.out), args.overwrite) for p in eps)

    print(f"[i] tasks={len(tasks)} jobs={len(jobs)} workers={args.workers} out={args.out}")
    t0 = time.time()
    counts = {"ok": 0, "no_grasp": 0, "skip": 0, "error": 0}
    total_bytes = 0
    errors = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_build_one, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            status = r[0]
            counts[status] = counts.get(status, 0) + 1
            if status == "error":
                errors.append({"task": r[1], "episode": r[2], "error": r[3],
                               "trace": r[4]})
            else:
                total_bytes += int(r[5] or 0)
            done += 1
            if done % 200 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  {counts}  "
                      f"{time.time()-t0:.0f}s  {total_bytes/1e6:.1f}MB")

    if errors:
        (args.out / "errors.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in errors) + "\n",
            encoding="utf-8")
    print(f"[done] {counts}  {time.time()-t0:.1f}s  "
          f"sidecar={total_bytes/1e6:.1f}MB  errors={len(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
