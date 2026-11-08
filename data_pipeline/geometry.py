"""Camera/robot geometry helpers."""
from __future__ import annotations

import numpy as np

from .config import TCP_OFFSET_M


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def project_point(pw: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """World point → pixel (u, v).

    RoboDojo E is Isaac/OpenGL camera-to-world.  Convert the camera-space
    point to OpenCV by flipping y/z before applying K.
    """
    R, t = E[:3, :3], E[:3, 3]
    pg = R.T @ (np.asarray(pw) - t)
    pc = np.array([pg[0], -pg[1], -pg[2]])
    uv = K @ pc
    return uv[:2] / uv[2]


def gripper_midpoint_3d(ee_pose: np.ndarray, offset=TCP_OFFSET_M) -> np.ndarray:
    """EE pose (x,y,z,qw,qx,qy,qz) -> two-finger midpoint in world (m).

    ⚠️ ``ee_poses`` 的原点是**腕部法兰**，不是指尖；anchor 必须是两指中点，
    所以要加一个固定偏移（EE 系）。偏移用 ``TCP_OFFSET_M``，必须与训练侧一致。
    """
    R = quat_to_R(np.asarray(ee_pose, np.float64)[3:])       # (w,x,y,z) -> R
    return np.asarray(ee_pose, np.float64)[:3] + R @ np.asarray(offset, np.float64)


def gripper_midpoint(ee_pose: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Two-finger midpoint projected into the image."""
    return project_point(gripper_midpoint_3d(ee_pose), K, E)


def project_point_depth(pw: np.ndarray, K: np.ndarray, E: np.ndarray) -> tuple:
    """World point -> (pixel uv, camera z in metres). ``z <= 0`` means behind."""
    R, t = E[:3, :3], E[:3, 3]
    pg = R.T @ (np.asarray(pw, np.float64) - t)
    pc = np.array([pg[0], -pg[1], -pg[2]])
    z = float(pc[2])
    if z <= 1e-6:
        return np.array([np.nan, np.nan]), z
    uv = (K @ pc)[:2] / pc[2]
    return uv, z


def to_normalized(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel → [0,1] normalized coordinates with pixel-center convention."""
    uv = np.asarray(uv, dtype=np.float64)
    return np.stack([(uv[..., 0] + 0.5) / width, (uv[..., 1] + 0.5) / height], axis=-1)
