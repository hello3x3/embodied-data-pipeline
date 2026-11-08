"""Re-wash the 13-task 8772 dataset to the 2-anchor + 3-object-point convention.

* keeps the existing sidecar anchors;
* picks 3 random corner tracks per subtask (fixed for the page; training may
  resample per epoch from ``object_corners``);
* adds ``object_points`` and per-phase ``object_0..2`` tracks to the sidecar;
* re-renders every phase frame with the 3 object points;
* updates ``object_valid`` and rewrites index.html.

No SAM2/CoTracker is re-run: the corner tracks already exist in the sidecar.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from .io import load_episode
from .playback import render_phase_frames
from .tracks import Channel, PlanTracks
from .web import write_index

BASE = Path("outputs/data_pipeline_web_cotrack")
SIDE = BASE / "sidecar"


def _uv_list(a):
    return [None if not np.isfinite(p).all() else [round(float(p[0]), 2), round(float(p[1]), 2)]
            for p in a]


def _corner_to_phase(track, frames, t0c, stride):
    idx = np.round((np.asarray(frames) - t0c) / stride).astype(int)
    ok = (idx >= 0) & (idx < len(track["uv"]))
    uv = np.full((len(frames), 2), np.nan)
    val = np.zeros(len(frames), bool)
    for i, j in enumerate(idx):
        if ok[i]:
            p = track["uv"][j]
            if p is not None:
                uv[i] = p
                val[i] = bool(track["valid"][j])
    return uv, val


def main() -> int:
    data = json.loads((BASE / "plan_queue_data.json").read_text(encoding="utf-8"))
    for ti, wt in enumerate(data["tasks"]):
        task = wt["task"]
        from .config import ROOT
        ep = load_episode(ROOT / wt["rel_path"], task=task, load_gray=False)
        sp = SIDE / f"{task}__{Path(wt['episode']).stem}.json"
        side = json.loads(sp.read_text(encoding="utf-8"))
        print(f"[{ti+1}/{len(data['tasks'])}] {task}", flush=True)
        for si, sub in enumerate(side["subtasks"]):
            oc = sub.get("object_corners")
            sel = []
            if oc and len(oc["tracks"]) > 0:
                n = min(3, len(oc["tracks"]))
                rng = np.random.default_rng(zlib.crc32(f"{task}/{sub['index']}".encode()))
                sel = rng.choice(len(oc["tracks"]), size=n, replace=False).tolist()
                sub["object_points"] = {
                    "t0": oc["t0"], "stride": oc["stride"],
                    "tracks": [oc["tracks"][i] for i in sel],
                }
            for pid, ph in enumerate(sub["phases"]):
                frames = list(range(ph["t_start"], ph["t_end"] + 1, 4))
                anchors = []
                for nm in ("anchor_left", "anchor_right"):
                    tr = ph["tracks"][nm]
                    uv = np.array([np.nan if p is None else p for p in tr["uv"]], np.float64)
                    va = np.asarray(tr["valid"], bool)
                    anchors.append(Channel(name=nm, kind="anchor", uv=uv, valid=va))
                obj_ch = []
                old_obj = ph["tracks"].get("object")
                for k, i in enumerate(sel):
                    tr = sub["object_points"]["tracks"][k]
                    uv, va = _corner_to_phase(tr, frames, sub["object_points"]["t0"],
                                              sub["object_points"]["stride"])
                    active = (np.asarray(old_obj["active"], bool)
                              if old_obj is not None and len(old_obj["active"]) == len(frames)
                              else np.ones(len(frames), bool))
                    obj_ch.append(Channel(name=f"object_{k}", kind="object", uv=uv,
                                          valid=va, active=active))
                pt = PlanTracks(phase=ph["phase"], t_start=ph["t_start"],
                                t_end=ph["t_end"], stride=4, channels=anchors + obj_ch)
                tag = f"s{sub['index']}_p{pid}_{ph['phase']}"
                render_phase_frames(ep, pt, BASE / "assets" / task, task,
                                    active_arm=sub["arm"], tag=tag)
                ph["object_valid"] = int(sum(int(c.valid.sum()) for c in obj_ch))
                for k, c in enumerate(obj_ch):
                    ph["tracks"][f"object_{k}"] = {
                        "uv": _uv_list(c.uv),
                        "valid": [bool(v) for v in c.valid],
                        "active": [bool(v) for v in c.active],
                    }
                # update web JSON phase
                wph = wt["subtasks"][si]["phases"][pid]
                wph["object_valid"] = ph["object_valid"]
        sp.write_text(json.dumps(side, indent=1, ensure_ascii=False, allow_nan=False),
                      encoding="utf-8")
    (BASE / "plan_queue_data.json").write_text(
        json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_index(BASE, data)
    print("done; index rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
