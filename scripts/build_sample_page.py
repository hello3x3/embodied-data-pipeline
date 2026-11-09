"""Pick one episode per task from the full sidecar set and build one page.

Frames are generated with ``--object none`` (the full data has no object
tracks); each task contributes one random valid episode.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.web import write_index  # noqa: E402


def pick(sidecar_dir: Path, task: str, rng: random.Random):
    files = sorted((sidecar_dir / task).glob("*.json"))
    valid = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if d.get("status") == "ok" and d.get("subtasks"):
            valid.append((f, d))
    if not valid:
        return None
    rich = [x for x in valid if len(x[1]["subtasks"]) >= 2]
    pool = rich or valid
    f, d = rng.choice(pool)
    return Path(d["episode"]).name, f


def build_one(task: str, episode: str, tmp: Path, py: str, root: Path):
    out = tmp / task
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
    cmd = [py, "-m", "data_pipeline.build_plan_queue",
           "--tasks", task, "--episode", episode,
           "--object", "none", "--out", str(out)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        return task, False, p.stderr[-800:]
    return task, True, f"{time.time()-t0:.0f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecars", type=Path, default=Path("outputs/robodojo_sidecars"))
    ap.add_argument("--out", type=Path, default=Path("outputs/robodojo_review_sample"))
    ap.add_argument("--tmp", type=Path, default=Path("outputs/_sample_parts"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = ROOT
    rng = random.Random(args.seed)

    tasks = sorted(p.name for p in args.sidecars.iterdir() if p.is_dir())
    chosen = {}
    for task in tasks:
        r = pick(args.sidecars, task, rng)
        if r is not None:
            chosen[task] = r[0]
    print(f"[i] tasks_with_sidecar={len(tasks)} chosen={len(chosen)}")

    shutil.rmtree(args.tmp, ignore_errors=True)
    args.tmp.mkdir(parents=True, exist_ok=True)
    ok, fail = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_one, t, e, args.tmp, sys.executable, root)
                for t, e in chosen.items()]
        for fut in as_completed(futs):
            task, good, msg = fut.result()
            (ok if good else fail).append(task)
            print(f"  {'ok ' if good else 'ERR'} {task:30s} {msg}")
    print(f"[i] built {len(ok)}/{len(chosen)}  failed={fail}")

    shutil.rmtree(args.out, ignore_errors=True)
    (args.out / "assets").mkdir(parents=True, exist_ok=True)
    (args.out / "sidecar").mkdir(parents=True, exist_ok=True)
    data = {"meta": {}, "tasks": [], "sidecar": []}
    for task in ok:
        part = args.tmp / task
        d = json.loads((part / "plan_queue_data.json").read_text(encoding="utf-8"))
        if not data["meta"]:
            data["meta"] = d.get("meta", {})
        data["tasks"].append(d["tasks"][0])
        shutil.copytree(part / "assets" / task, args.out / "assets" / task)
        for f in (part / "sidecar").glob("*.json"):
            shutil.copy2(f, args.out / "sidecar" / f.name)
            data["sidecar"].append(str(Path("sidecar") / f.name))
    data["meta"]["sample"] = {"seed": args.seed, "tasks": len(data["tasks"])}
    (args.out / "plan_queue_data.json").write_text(
        json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    write_index(args.out, data)
    (args.out / "sample_manifest.json").write_text(
        json.dumps(chosen, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(args.tmp, ignore_errors=True)
    print(f"[ok] page={args.out/'index.html'} tasks={len(data['tasks'])}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
