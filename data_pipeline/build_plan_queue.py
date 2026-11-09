"""Build the phase-level sidecar + review page: task → subtask → phase → frames.

No chunking here. One **phase** (reach / pick / move / place) is stored as one
variable-length dense point trajectory (6.25 Hz, one point every 4 action steps)
plus its anchors/object tracks. Chunking (7-point window + 25-step action) is a
**training-time** operation done by sliding a cursor over that trajectory.

Outputs::

    <out>/index.html
    <out>/plan_queue_data.json
    <out>/assets/<task>/s<i>_<phase>_overview.png
    <out>/assets/<task>/<task>_s<i>_<phase>_f0000.jpg   (one per frame)
    <out>/sidecar/<task>__<episode>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .autoseed import auto_seed_track
from .object_seg import segment_and_track
from .config import (
    EXEC_STEPS,
    FAR_POINTS,
    NEAR_POINTS,
    POINT_STRIDE,
    QUAT_ORDER,
    ROOT,
    SEEDS,
    TASKS,
    TCP_OFFSET_M,
)
from .io import Episode, load_episode
from .playback import render_phase_frames
from .plans import PlanRange
from .task_timeline import Subtask, segment_episode
from .tracks import build_plan_tracks, project_anchors
from .tracking import track_point
from .web import write_index

ACTION_HORIZON = 25       # only for the contract text in meta; chunking is train-time


def _uv_list(arr: np.ndarray) -> list:
    out = []
    for p in np.asarray(arr, np.float64):
        out.append(None if not np.isfinite(p).all()
                   else [round(float(p[0]), 2), round(float(p[1]), 2)])
    return out


def _sidecar_phase(pt, plan_id: str) -> dict:
    tracks = {}
    for ch in pt.channels:
        entry = {"uv": _uv_list(ch.uv), "valid": [bool(v) for v in ch.valid]}
        if ch.active is not None:
            entry["active"] = [bool(a) for a in ch.active]
        tracks[ch.name] = entry
    return {"plan_id": plan_id, "phase": pt.phase, "t_start": pt.t_start,
            "t_end": pt.t_end, "t_last": pt.t_last, "n_points": pt.n_points,
            "stride": pt.stride, "duration_steps": pt.duration_steps,
            "object_id": 1, "tracks": tracks}


def _track_subtask(ep: Episode, sub: Subtask, task: str, tracker: str,
                   device: str, object_method: str = "sam2"):
    """Returns (obj dict, qc, SeedResult | None)."""
    if object_method == "none":
        return {}, {}, None
    if tracker == "cotracker":
        res = (segment_and_track(ep, sub, device=device)
               if object_method == "sam2" else auto_seed_track(ep, sub, device=device))
        if res is None and object_method == "sam2":
            res = auto_seed_track(ep, sub, device=device)      # fallback
        if res is None:
            return {}, {}, None
        return (res.uv,
                {**res.qc, "seed_uv": [round(x, 1) for x in res.seed_uv],
                 "t_seed": res.t_seed, "n_frames": len(res.uv)},
                res)
    if tracker == "lk":
        if sub.index != 0 or task not in SEEDS or not ep.grays:
            return {}, {}, None
        t0 = int(np.clip(sub.t_close + 3, sub.phases[0][1], sub.phases[-1][2]))
        obj, _ = track_point(ep.grays, SEEDS[task], t0,
                             start=sub.phases[0][1], end=sub.phases[-1][2])
        return obj, {"seed_uv": list(SEEDS[task]), "t_seed": t0, "n_frames": len(obj)}, None
    raise ValueError(f"未知 tracker {tracker!r}")


def build_task(task: str, root: Path, out_dir: Path, episode_name: str, stride: int,
               tracker: str = "cotracker", device: str = "cuda", max_frames: int = 0,
               object_method: str = "sam2"):
    path = root / task / "arx_x5" / "data" / episode_name
    if not path.is_file():
        raise FileNotFoundError(f"找不到 episode：{path}")
    ep = load_episode(path, task=task, load_gray=(tracker == "lk"))
    # (SAM2 model is loaded lazily inside object_seg on first use)
    subtasks = segment_episode(ep)
    Luv, Lvalid, Ruv, Rvalid = project_anchors(ep)
    print(f"[{task}] T={ep.T} subtasks={len(subtasks)}")

    assets = out_dir / "assets" / task
    web_subs, side_subs = [], []
    for sub in subtasks:
        obj, qc, seed_res = _track_subtask(ep, sub, task, tracker, device, object_method)
        web_phases, side_phases = [], []
        for pi, (phase, a, b) in enumerate(sub.phases):
            pt = build_plan_tracks(ep, PlanRange(phase, a, b), Luv, Ruv, obj, sub.arm,
                                   stride, Lvalid=Lvalid, Rvalid=Rvalid)
            tag = f"s{sub.index}_p{pi}_{phase}"
            overview = assets / f"{tag}_overview.png"
            from .viz import render_plan_overview     # imported lazily (matplotlib)
            render_plan_overview(ep, pt, [], overview, task=task, active_arm=sub.arm)

            corners = None
            if seed_res is not None and seed_res.corner_uv is not None:
                corners = (seed_res.corner_uv, seed_res.corner_vis, seed_res.corner_t0)
            frames = render_phase_frames(ep, pt, assets, task, active_arm=sub.arm,
                                         tag=tag, corners=corners)
            if max_frames:
                frames = frames[:max_frames]
            rel = [str(Path(f).relative_to(out_dir)).replace("\\", "/") for f in frames]
            obj_ch = next((c for c in pt.channels if c.kind == "object"), None)
            web_phases.append({
                "phase": pt.phase, "t_start": pt.t_start, "t_end": pt.t_end,
                "n_points": pt.n_points, "duration_steps": pt.duration_steps,
                "object_valid": int(obj_ch.valid.sum()) if obj_ch is not None else 0,
                "overview": str(overview.relative_to(out_dir)).replace("\\", "/"),
                "frames": rel, "n_frames": len(rel),
            })
            side_phases.append(_sidecar_phase(
                pt, f"{task}/{path.stem}/s{sub.index}/p{pi}_{phase}"))
            print(f"  #{sub.index + 1} {sub.arm} p{pi} {phase:5s} [{a},{b}]  "
                  f"{pt.n_points}pts {pt.duration_steps}steps -> {len(rel)} frames")
        mask_image = None
        if seed_res is not None and seed_res.mask is not None:
            import cv2
            from pathlib import Path as _P
            mdir = assets / "masks"
            mdir.mkdir(parents=True, exist_ok=True)
            f = int(seed_res.info.get("t_seed", sub.t_close))
            im = ep.read_color(f).copy()
            cnt = cv2.findContours(seed_res.mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(im, cnt, -1, (0, 255, 0), 2)
            for p in seed_res.info.get("candidates", []):
                cv2.circle(im, tuple(np.round(p).astype(int)), 2, (0, 0, 255), -1)
            tcp = seed_res.info.get("tcp_uv")
            if tcp:
                cv2.circle(im, (int(tcp[0]), int(tcp[1])), 4, (0, 255, 255), -1)
            mp = mdir / f"s{sub.index}_mask.jpg"
            cv2.imwrite(str(mp), im)
            mask_image = str(mp.relative_to(out_dir)).replace("\\", "/")
        web_subs.append({"index": sub.index, "arm": sub.arm, "t_close": sub.t_close,
                         "t_open": sub.t_open, "object_qc": qc,
                         "mask_image": mask_image, "phases": web_phases})
        corner_data = None
        if seed_res is not None and seed_res.corner_uv is not None:
            cuv, cvis, ct0 = seed_res.corner_uv, seed_res.corner_vis, seed_res.corner_t0
            take = list(range(0, len(cuv), stride))
            corner_data = {
                "t0": int(ct0), "stride": int(stride),
                "tracks": [{"uv": _uv_list(cuv[take, n]),
                            "valid": [bool(v > 0.5) for v in cvis[take, n]]}
                           for n in range(cuv.shape[1])],
            }
        side_subs.append({"index": sub.index, "arm": sub.arm, "t_close": sub.t_close,
                          "t_open": sub.t_open, "object_qc": qc,
                          "mask_image": mask_image, "object_corners": corner_data,
                          "phases": side_phases})

    h, w = ep.hw["cam_head"]
    rel_path = str(path.relative_to(root))
    sidecar = {
        "episode": rel_path, "task": task, "fps": int(ep.fps), "stride": int(stride),
        "image_size": [int(h), int(w)], "tcp_offset": [float(x) for x in TCP_OFFSET_M],
        "quat_order": QUAT_ORDER, "subtasks": side_subs,
    }
    web_task = {"task": task, "episode": path.name, "rel_path": rel_path,
                "T": int(ep.T), "fps": int(ep.fps), "image_size": [int(h), int(w)],
                "subtasks": web_subs}
    return web_task, sidecar


def main() -> int:
    ap = argparse.ArgumentParser(description="RoboDojo → task/subtask/phase + review page")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--episode", default="episode_0000000.hdf5")
    ap.add_argument("--out", type=Path, default=Path("outputs/task_review"))
    ap.add_argument("--stride", type=int, default=POINT_STRIDE)
    ap.add_argument("--tracker", default="cotracker", choices=["cotracker", "lk"])
    ap.add_argument("--object", default="sam2", choices=["sam2", "auto", "none"],
                    help="object seed: SAM2+CoTracker, ring fallback, or none (anchors only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0, help="调试：每阶段最多渲染多少帧")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "sidecar").mkdir(exist_ok=True)
    print(f"[i] root={args.root} out={out} tracker={args.tracker} stride={args.stride}")

    web_tasks, side_files, errors = [], [], []
    for task in args.tasks:
        try:
            web_task, sidecar = build_task(task, args.root, out, args.episode, args.stride,
                                           args.tracker, args.device, args.max_frames,
                                           args.object)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{task}: {exc}")
            print(f"[!] {task} 失败：{exc}")
            continue
        web_tasks.append(web_task)
        sp = out / "sidecar" / f"{task}__{Path(args.episode).stem}.json"
        sp.write_text(json.dumps(sidecar, indent=1, ensure_ascii=False, allow_nan=False),
                      encoding="utf-8")
        side_files.append(str(sp.relative_to(out)))

    if not web_tasks:
        return 2
    data = {
        "meta": {
            "stride": int(args.stride), "traj_hz": round(25 / args.stride, 3),
            "near_points": NEAR_POINTS, "chunk": ACTION_HORIZON,
            "exec_steps": EXEC_STEPS, "far_points": FAR_POINTS,
            "image_size": web_tasks[0]["image_size"], "tracker": args.tracker,
            "root": str(args.root), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "errors": errors,
        },
        "tasks": web_tasks, "sidecar": side_files,
    }
    (out / "plan_queue_data.json").write_text(
        json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    page = write_index(out, data)
    n_frames = sum(ph["n_frames"] for t in web_tasks for s in t["subtasks"] for ph in s["phases"])
    print(f"[ok] {len(web_tasks)} tasks / "
          f"{sum(len(t['subtasks']) for t in web_tasks)} subtasks / "
          f"{sum(len(s['phases']) for t in web_tasks for s in t['subtasks'])} phases / "
          f"{n_frames} frames")
    print(f"[ok] page: {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
