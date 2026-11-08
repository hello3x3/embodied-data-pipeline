"""Single-point object tracking (manual seed + pyramidal LK)."""
from __future__ import annotations

import cv2
import numpy as np

from .config import FB_MAX_PX, MAX_OBJ_JUMP_PX


def track_point(
    grays: list[np.ndarray],
    seed: tuple[float, float],
    t0: int,
    start: int,
    end: int,
    fb_max: float = FB_MAX_PX,
    step_max: float = MAX_OBJ_JUMP_PX,
) -> tuple[dict[int, np.ndarray], dict[int, bool]]:
    """Track one object point forward/backward from a seed at t0.

    The tracker is deliberately guarded:
      - forward-backward consistency < fb_max
      - per-frame displacement < step_max
      - stops at first failure instead of jumping to another region
    """
    out: dict[int, np.ndarray] = {t0: np.array(seed, dtype=np.float32)}
    valid: dict[int, bool] = {t0: True}

    def run(direction: int) -> None:
        cur = np.array([[seed]], dtype=np.float32)
        rng = range(t0 + 1, end + 1) if direction > 0 else range(t0 - 1, start - 1, -1)
        for fr in rng:
            prev = fr - direction
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                grays[prev], grays[fr], cur, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            back, st2, _ = cv2.calcOpticalFlowPyrLK(
                grays[fr], grays[prev], nxt, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            fb = float(np.linalg.norm(back.reshape(2) - cur.reshape(2)))
            step = float(np.linalg.norm(nxt.reshape(2) - cur.reshape(2)))
            if (
                st.reshape(-1)[0] != 1
                or st2.reshape(-1)[0] != 1
                or fb > fb_max
                or step > step_max
            ):
                break
            cur = nxt
            out[fr] = cur.reshape(2)
            valid[fr] = True

    run(1)
    run(-1)
    return out, valid


def fill_static_prefix(
    out: dict[int, np.ndarray],
    valid: dict[int, bool],
    seed: tuple[float, float],
    start: int,
    end: int,
) -> None:
    """During reach the object is static; use the seed if LK failed backwards."""
    for fr in range(start, end):
        if fr not in out:
            out[fr] = np.array(seed, dtype=np.float32)
            valid[fr] = True
