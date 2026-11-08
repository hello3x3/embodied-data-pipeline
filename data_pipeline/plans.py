"""Short primitive plans: reach / pick / move / place, one object."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PHASES
from .io import Episode


@dataclass
class PlanRange:
    phase: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def detect_contact_release(ep: Episode, arm: str | None = None) -> tuple[str, int, int]:
    """Pick the active arm (or use override), first close, first open after it."""
    if arm is None:
        mL = float(np.ptp(ep.L, axis=0).sum())
        mR = float(np.ptp(ep.R, axis=0).sum())
        candidates = []
        for a, g, motion in [("L", ep.gL, mL), ("R", ep.gR, mR)]:
            closed = np.where(g < 0.35)[0]
            if len(closed):
                candidates.append((motion, a, int(closed[0])))
        if not candidates:
            raise ValueError("no gripper close event")
        _, arm, c = max(candidates)
    else:
        g = ep.gL if arm == "L" else ep.gR
        closed = np.where(g < 0.35)[0]
        if not len(closed):
            raise ValueError(f"{arm} has no gripper close event")
        c = int(closed[0])
    g = ep.gL if arm == "L" else ep.gR
    rel = None
    for i in range(c + 5, ep.T):
        if g[i] > 0.8:
            rel = i
            break
    if rel is None:
        rel = min(ep.T - 1, c + 50)
    return arm, c, rel


def segment_plans(ep: Episode, c: int, rel: int, reach_back: int = 30) -> list[PlanRange]:
    """Cut one object's episode into short primitive plans.

    Each returned plan is one primitive and one object; the next object
    (if any) starts a new plan queue outside this function.
    """
    reach_start = max(0, c - reach_back)
    pick_end = min(c + 10, rel - 1)
    move_end = max(pick_end + 1, rel - 8)
    place_end = min(ep.T - 1, rel + 10)
    bounds = [(reach_start, c), (c, pick_end), (pick_end, move_end), (move_end, place_end)]
    plans = []
    for phase, (a, b) in zip(PHASES, bounds):
        if b > a:
            plans.append(PlanRange(phase=phase, start=int(a), end=int(b)))
    return plans
