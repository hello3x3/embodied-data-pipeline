# Embodied data pipeline

Convert robot manipulation episodes into:

1. **subtask / phase dense trajectories** (training sidecar), and
2. a **static review page** with frame playback (task → subtask → phase → frame).

The current adapter is for **RoboDojo depth HDF5**; other datasets can be
added by implementing the same `Episode` loader contract (see
[Datasets](#datasets)). The data side does **not** cut chunks. One phase is one variable-length 6.25 Hz
trajectory; the training loader slides its own 7-point window over it.

```text
HDF5
 └─ load episode (state/action, EE poses, K/E, RGB)
     └─ segment both arms into grasp cycles
         └─ reach / pick / move / place / stop (+ final reach)
             └─ anchors: TCP midpoint, 25 Hz -> 6.25 Hz
             └─ object: SAM2 mask -> corners -> CoTracker -> fused point
                 └─ sidecar JSON + JPEG frames + index.html
```

## Contract

| item | value |
|---|---|
| action rate | 25 Hz |
| trajectory rate | **6.25 Hz** (one point every 4 action steps) |
| action chunk | 25 steps |
| near window | 7 points (steps 0,4,...,24) |
| executed | 12 steps = 3 points |
| far | 8 arc-length points |
| anchor | two-finger midpoint = `ee_pose[:3] + R(quat) @ (0.145, 0, 0)` |
| quaternion | `wxyz` |
| stop | only if ≥ 25 frames (1 s) |
| pick / place | padded to ≥ 25 frames; `place` ends at fully-open |

Phase semantics:

- `reach`: gripper fully open (does not include closing);
- `pick`: open → fully closed;
- `move`: fully closed;
- `place`: closed → fully open;
- `stop`: active arm holds still (waiting/holding);
- after `place`, post-place retract is `reach` (the last subtask reaches to `T-1`).

## Requirements

- Python 3.10+
- numpy, opencv-python, h5py, matplotlib, torch
- **SAM2 source checkout** + checkpoint (not pip-installed here)
- **CoTracker** package + checkpoint
- `hydra-core`, `iopath`, `omegaconf` for SAM2

Use the Python environment that already has torch/opencv/h5py (on this machine:
`/home/shujiuhe/apps/miniconda3/envs/omniembodied/bin/python`).  The Makefile
uses `PY ?= python3`; pass `PY=...` or activate the env first.

```bash
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/co-tracker.git"
# SAM2: clone facebookresearch/sam2 and install its deps; no pip install of sam2
pip install hydra-core iopath omegaconf
```

Model paths can be overridden by environment variables:

```bash
export ROBODOJO_DEPTH_ROOT=/mnt/nas/datasets2/RoboDojo_depth
export SAM2_CODE=/path/to/sam2
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
export SAM2_CKPT=/path/to/sam2.1_hiera_large.pt
export COTRACKER_CKPT=/path/to/scaled_offline.pth
```

Defaults match this machine's shared weights; no network is needed at runtime.

## Datasets

`data_pipeline/io.py` currently implements the RoboDojo depth HDF5 layout:
`vision/cam_*/colors`, `state|action/*_ee_poses`, `state|action/*_joint_states`,
`vision/cam_*/intrinsic_matrix|extrinsic_matrix`.

To add another dataset, provide a loader that returns an `Episode` with the
same fields (`T`, `L/R`, `gL/gR`, `actions`, `proprio`, `ee_action`, `K/E/hw`,
`read_color`). The rest of the pipeline (segmentation, tracks, sidecar, page)
is dataset-agnostic. Planned layout: `data_pipeline/adapters/<dataset>.py`.

## Build

```bash
cd ~/projects/embodied-data-pipeline
# use the conda env's python (or activate it first)
make build PY=$HOME/apps/miniconda3/envs/omniembodied/bin/python \
           TASKS="fill_pen_holder stack_bowls insert_tubes" \
           EPISODE=episode_0000000.hdf5 OUT=outputs/data_pipeline_web
```

or directly:

```bash
python -m data_pipeline.build_plan_queue \
  --tasks fill_pen_holder stack_bowls insert_tubes \
  --episode episode_0000000.hdf5 \
  --object sam2 \
  --out outputs/data_pipeline_web
```

Notes:

- One build applies **one episode** to all `--tasks`.
- To mix episodes (e.g. one task from `episode_0000012.hdf5`, others from
  `episode_0000000.hdf5`), build separately and merge `assets/`, `sidecar/`,
  and `plan_queue_data.json`, then regenerate `index.html` with
  `data_pipeline.web.write_index`.
- `--object auto` uses the old ring-corner fallback instead of SAM2.
- Episode files: `<task>/arx_x5/data/episode_*.hdf5` (100 per task).

## Outputs

```text
outputs/data_pipeline_web/
  index.html                       # static page, DATA inlined
  plan_queue_data.json             # tasks/subtasks/phases/frames
  assets/<task>/
    <task>_s{i}_p{pi}_{phase}_f0000.jpg   # overlay frames
    <task>_s{i}_p{pi}_{phase}_overview.png
    s{i}_mask.jpg                         # SAM mask + candidates + TCP
  sidecar/<task>__<episode>.json          # training data
```

Sidecar:

```json
{
  "episode": "...", "task": "...", "fps": 25, "stride": 4,
  "image_size": [480, 640], "tcp_offset": [0.145, 0, 0], "quat_order": "wxyz",
  "subtasks": [{
    "index": 0, "arm": "R", "t_close": 169, "t_open": 225,
    "object_qc": {...}, "mask_image": "...",
    "object_corners": {"t0": 166, "stride": 4,
                       "tracks": [{"uv": [[u,v], ...], "valid": [true, ...]}]},
    "phases": [{
      "phase": "reach", "t_start": 0, "t_end": 163,
      "t_last": 163, "n_points": 41, "stride": 4, "duration_steps": 164,
      "tracks": {
        "anchor_left":  {"uv": [[u,v], ...], "valid": [true, ...]},
        "anchor_right": {"uv": [[u,v], ...], "valid": [true, ...]},
        "object":       {"uv": [[u,v], ...], "valid": [true, ...],
                         "active": [true, ...]}
      }
    }]
  }]
}
```

`object_corners` holds **all** CoTracker corner tracks at 6.25 Hz; the `object`
channel is their fused representative point used by the model.

## Review page

```bash
make page PY=$HOME/apps/miniconda3/envs/omniembodied/bin/python
# http://127.0.0.1:8772
# or
python -m http.server 8772 --bind 127.0.0.1 --directory outputs/data_pipeline_web
```

- Task → subtask (left/right hand) → phase → frame.
- Keyboard: `↓/↑` phase, `←/→` frame, space play.
- Frames bake the overlays: both arm anchors, **all object corners** (small
  magenta dots / hollow for low visibility), fused object point (orange),
  25-step commanded trail, mask card.
- Player preloads 24 frames and waits for `onload`; no blind `src` swap.

## Diagnostics

```bash
make check PY=$HOME/apps/miniconda3/envs/omniembodied/bin/python
make diag-object TASK=fill_egg_holder SUB=0   # 8-panel step-by-step
make diag-samples                              # 10 subtask samples
make diag-tasks                                # 10 task samples
```

## Known limitations

- SAM2 masks only the seed frame; no per-frame mask propagation.
- CoTracker + rigid fusion can still drift on textureless/occluded objects.
- Some subtasks have no object track (SAM2/QC failure).
- Two subtasks can have `pick < 25` frames when `place` must end at fully-open
  and the cycle is too short to give both 25.
- The page covers one episode per task; 100 episodes exist per task.
- Training loader (in the training repo) still uses the old 25-point/step
  cursor; `data_pipeline/chunks.py` is the training-time reference, not used
  by this data build.
