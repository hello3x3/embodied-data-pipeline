"""Episode loading from RoboDojo depth HDF5.

Two label sources, deliberately kept apart:

* ``state/``  -> measured joint positions + gripper  -> **proprio**
* ``action/`` -> commanded joint targets + gripper  -> **BC target**

The old loader preferred ``state/`` for both, which silently trained the policy
on "next measured state" instead of the command it is supposed to emit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np

from .config import CAMS


def _decode_gray(raw: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)


def _decode_rgb(raw: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


@dataclass
class Episode:
    path: Path
    task: str
    T: int
    fps: int
    L: np.ndarray            # (T, 7) left EE pose (link6)
    R: np.ndarray            # (T, 7) right EE pose
    gL: np.ndarray           # (T,) left gripper state
    gR: np.ndarray           # (T,) right gripper state
    actions: np.ndarray      # (T, 14) commanded: L6, Lg, R6, Rg
    proprio: np.ndarray      # (T, 14) measured:  L6, Lg, R6, Rg
    ee_action: dict[str, np.ndarray]   # "left"/"right" -> (T, 7) commanded EE pose
    K: dict[str, np.ndarray]     # cam -> (3, 3) native intrinsics
    E: dict[str, np.ndarray]     # cam -> (T, 4, 4) camera-to-world (GL)
    hw: dict[str, tuple[int, int]]   # cam -> (H, W) native
    instruction: str
    grays: list[np.ndarray] | None = None

    # -- convenience (head camera, what the anchors are projected with) ------
    @property
    def K_head(self) -> np.ndarray:
        return self.K["cam_head"]

    @property
    def E_head(self) -> np.ndarray:
        return self.E["cam_head"]

    def read_color(self, frame: int, cam: str = "cam_head") -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            b = f[f"vision/{cam}/colors"][frame]
        return _decode_rgb(b)

    def read_all_gray(self, cam: str = "cam_head") -> list[np.ndarray]:
        with h5py.File(self.path, "r") as f:
            return [_decode_gray(b) for b in f[f"vision/{cam}/colors"][:]]

    def read_colors(self, frames, cam: str = "cam_head") -> list[np.ndarray]:
        with h5py.File(self.path, "r") as f:
            ds = f[f"vision/{cam}/colors"]
            return [_decode_rgb(ds[int(t)]) for t in frames]


def _read_joint(f, group: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """(T,6) arm joints + (T,1) gripper, from ``group`` in {state, action}."""
    arm_j = f[f"{group}/{arm}_arm_joint_states"][:].astype(np.float64)
    grip = f[f"{group}/{arm}_ee_joint_states"][:].reshape(len(arm_j), -1)[:, -1:]
    return arm_j, grip


def load_episode(path: str | Path, task: str, load_gray: bool = True) -> Episode:
    path = Path(path)
    with h5py.File(path, "r") as f:
        T = len(f["state/left_ee_poses"])
        L = f["state/left_ee_poses"][:]
        R = f["state/right_ee_poses"][:]

        pieces_s, pieces_a = [], []
        for arm in ("left", "right"):
            js, gs = _read_joint(f, "state", arm)
            ja, ga = _read_joint(f, "action", arm)
            pieces_s += [js, gs]
            pieces_a += [ja, ga]
        proprio = np.concatenate(pieces_s, axis=1)
        actions = np.concatenate(pieces_a, axis=1)
        ee_action = {
            arm: np.asarray(f[f"action/{arm}_ee_poses"][:], np.float64)
            for arm in ("left", "right")
        }

        K, E, hw = {}, {}, {}
        for cam in CAMS:
            K[cam] = np.asarray(f[f"vision/{cam}/intrinsic_matrix"][()], np.float64)
            E[cam] = np.asarray(f[f"vision/{cam}/extrinsic_matrix"][:], np.float64)
            h, w = (int(x) for x in f[f"vision/{cam}/shape"][()][:2])
            hw[cam] = (h, w)

        fps = int(f["additional_info/frequency"][()])
        raw_instr = f["instruction"][()]
        instruction = raw_instr.decode() if isinstance(raw_instr, bytes) else str(raw_instr)

    ep = Episode(
        path=path, task=task, T=T, fps=fps, L=L, R=R,
        gL=proprio[:, 6], gR=proprio[:, 13],
        actions=actions, proprio=proprio, ee_action=ee_action, K=K, E=E, hw=hw,
        instruction=instruction,
    )
    if load_gray:
        ep.grays = ep.read_all_gray()
    return ep
