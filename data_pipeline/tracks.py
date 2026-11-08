"""Per-plan dense tracks at the **trajectory rate** (6.25 Hz by default).

A track is a polyline of pixel points in the native head image:
``uv (L,2)`` + ``valid (L,)`` (+ ``active (L,)`` for the object channel).
Points are sampled every ``stride`` action steps starting at the plan's first
frame, so point ``k`` sits at action step ``k*stride``.

The object channel is the only one that can be invalid (LK lost / occluded);
anchors always have a pose, so their ``valid`` is all True.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import POINT_STRIDE
from .geometry import gripper_midpoint_3d, project_point_depth
from .io import Episode
from .plans import PlanRange

__all__ = ["Channel", "PlanTracks", "plan_point_frames", "build_plan_tracks"]


@dataclass
class Channel:
    name: str
    kind: str                  # "anchor" | "object"
    uv: np.ndarray             # (L, 2) native pixels
    valid: np.ndarray          # (L,) bool
    active: np.ndarray | None = None   # (L,) bool, object only

    def __len__(self) -> int:
        return int(self.uv.shape[0])


@dataclass
class PlanTracks:
    phase: str
    t_start: int
    t_end: int            # plan 的语义终点（接触/释放帧）
    stride: int
    channels: list[Channel] = field(default_factory=list)

    @property
    def t_last(self) -> int:
        """最后一个轨迹点的帧号（≤ t_end；差 0..stride-1 帧）。"""
        return self.t_start + self.stride * max(0, self.n_points - 1)

    @property
    def n_points(self) -> int:
        return len(self.channels[0]) if self.channels else 0

    @property
    def duration_steps(self) -> int:
        """Executable steps covered by this phase, **inclusive of t_end**.

        Not ``stride*(L-1)``: the phase boundary rarely lands on the 4-step
        grid, so the last 0–3 frames would otherwise be unreachable (and the
        review page would stop before the last frame). The tail is guided by a
        padded last point.
        """
        return max(0, self.t_end - self.t_start + 1)

    def channel_names(self) -> list[str]:
        return [c.name for c in self.channels]

    def by_kind(self, kind: str) -> list[Channel]:
        return [c for c in self.channels if c.kind == kind]


def plan_point_frames(t_start: int, t_end: int, stride: int = POINT_STRIDE) -> list[int]:
    """Frames where trajectory points live: ``t_start + k*stride`` up to ``t_end``."""
    if t_end < t_start:
        raise ValueError(f"plan 区间非法：{t_start}..{t_end}")
    return list(range(int(t_start), int(t_end) + 1, int(stride)))


def _subsample(values: list, frames: list[int], fill=np.nan) -> np.ndarray:
    return np.asarray([values[f] if f < len(values) else fill for f in frames], np.float64)


def build_plan_tracks(
    ep: Episode,
    plan: PlanRange,
    Luv: np.ndarray,
    Ruv: np.ndarray,
    obj: dict[int, np.ndarray],
    active_arm: str,
    stride: int = POINT_STRIDE,
    Lvalid: np.ndarray | None = None,
    Rvalid: np.ndarray | None = None,
) -> PlanTracks:
    """Cut one plan's 25 Hz anchor/object tracks down to the trajectory rate."""
    frames = plan_point_frames(plan.start, plan.end, stride)
    g = ep.gL if active_arm == "L" else ep.gR
    H, W = ep.hw["cam_head"]

    out: list[Channel] = []
    for name, table, ok in (("anchor_left", Luv, Lvalid), ("anchor_right", Ruv, Rvalid)):
        uv = _subsample(list(table), frames)
        valid = (np.isfinite(uv).all(axis=1) if ok is None
                 else np.asarray([bool(ok[f]) for f in frames], bool))
        out.append(Channel(name=name, kind="anchor", uv=uv, valid=valid))

    uv_o = np.stack([
        obj[f] if f in obj else np.array([np.nan, np.nan]) for f in frames
    ]).astype(np.float64)
    valid_o = (np.isfinite(uv_o).all(axis=1)
               & (uv_o[:, 0] >= 0) & (uv_o[:, 0] < W)
               & (uv_o[:, 1] >= 0) & (uv_o[:, 1] < H))
    # ``active`` 是**意图**（夹爪是否夹住），与 ``valid``（是否测到）独立：
    # reach 阶段夹爪张开 → 0；抓起来 → 1；放下 → 0。
    # 手挡住但仍在夹持时应当 valid=0 / active=1，所以这里**不能**和 valid 取与。
    active_o = (g[frames] < 0.35)
    out.append(Channel(name="object", kind="object", uv=uv_o, valid=valid_o, active=active_o))

    return PlanTracks(phase=plan.phase, t_start=int(plan.start), t_end=int(plan.end),
                      stride=int(stride), channels=out)


def project_anchors(ep: Episode, cam: str = "cam_head") -> tuple:
    """25 Hz **两指中点** tracks for both arms: ``(Luv, Lvalid, Ruv, Rvalid)``.

    ``valid`` 要求点在相机前方且落在图像内 —— 直接用 ee_pose 会得到法兰位置，
    不是 anchor；这里先加 TCP 偏移再投影。
    """
    K, E = ep.K[cam], ep.E[cam]
    H, W = ep.hw[cam]
    out = []
    for poses in (ep.L, ep.R):
        uv, ok = [], []
        for t in range(ep.T):
            p, z = project_point_depth(gripper_midpoint_3d(poses[t]), K, E[t])
            uv.append(p)
            ok.append(bool(np.isfinite(p).all() and z > 1e-6
                           and 0 <= p[0] < W and 0 <= p[1] < H))
        out += [np.asarray(uv, np.float64), np.asarray(ok, bool)]
    return tuple(out)
