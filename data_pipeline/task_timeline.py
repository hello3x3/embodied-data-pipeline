"""Task-level view: whole episode → subtasks (one grasp cycle each) → 4 phases.

The plan-level pipeline (`build_plan_queue.py`) only handles the **first**
close/release, so it cannot show a full task such as `fill_pen_holder`
(4 pens). This module segments **every** grasp cycle of **both** arms and
renders a continuous episode video + timeline page:

    episode
      ├─ subtask 1: arm R, close 186 → open 306
      │    reach [156,186] · pick [186,196] · move [196,298] · place [298,316]
      ├─ subtask 2: arm R, close 385 → open 496  ...
      └─ gaps: idle / approach (no guidance)

Outputs (one directory per task)::

    <out>/<task>/episode.mp4      # continuous RGB + anchor trails + phase banner
    <out>/<task>/timeline.png     # gripper signals + subtask/phase blocks
    <out>/<task>/index.html       # video + clickable subtask/phase table
    <out>/<task>/task.json        # segmentation for scripting
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from .config import PHASES, ROOT, TASKS  # noqa: E402
from .io import Episode, load_episode    # noqa: E402
from .tracks import project_anchors      # noqa: E402

__all__ = ["Subtask", "segment_episode", "state_at", "build_task"]

CLOSE_THR = 0.35
REACH_BACK = 30
PICK_FWD = 10
MOVE_BACK = 8
PLACE_FWD = 10
MIN_CYCLE = 5        # frames; shorter close→open blips are gripper jitter
MIN_STOP_FRAMES = 25  # 1 s at 25 Hz; shorter stops are not a state
MIN_SUBTASK = 10     # frames per split part

PHASE_COLOR = {  # BGR for cv2, hex for the page
    "reach": ((128, 128, 128), "#6b7280"),
    "pick": ((11, 158, 245), "#f59e0b"),
    "move": ((129, 185, 16), "#10b981"),
    "place": ((246, 92, 139), "#8b5cf6"),
    "idle": ((60, 60, 60), "#374151"),
    "stop": ((90, 90, 90), "#6b7280"),
}


@dataclass
class Subtask:
    index: int
    arm: str                       # "L" / "R"
    t_close: int
    t_open: int
    phases: list[tuple[str, int, int]] = field(default_factory=list)

    def phase_at(self, t: int) -> str | None:
        for name, a, b in self.phases:
            if a <= t <= b:
                return name
        return None


def find_cycles(g: np.ndarray, thr: float | None = None) -> list[tuple[int, int]]:
    """Complete close→open cycles: (close_frame, open_frame).

    The threshold is **adaptive per arm**: the two grippers of this robot do not
    share a closed value (right closes to ~0.02–0.30, left to ~0.60). A fixed
    0.35 silently drops the left grasp. Use the midpoint of the observed range.
    """
    lo, hi = float(np.min(g)), float(np.max(g))
    if hi - lo < 0.1:                       # never actuated
        return []
    if thr is None:
        thr = 0.5 * (lo + hi)
    closed = (g < thr).astype(int)
    d = np.diff(closed)
    closes = np.where(d == 1)[0] + 1
    opens = np.where(d == -1)[0] + 1
    out = []
    for c in closes:
        later = opens[opens > c]
        if len(later) and int(later[0]) - int(c) >= MIN_CYCLE:
            out.append((int(c), int(later[0])))
    return out


def _clean_mask(mask: np.ndarray, min_true: int, min_false: int) -> np.ndarray:
    """Flip runs shorter than the minimum length until the mask is clean."""
    mask = mask.astype(bool).copy()
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(mask):
            j = i
            while j < len(mask) and mask[j] == mask[i]:
                j += 1
            need = min_true if mask[i] else min_false
            if j - i < need:
                mask[i:j] = not mask[i]
                changed = True
                break
            i = j
    return mask


def _static_mask(uv: np.ndarray, thr: float = 0.5, smooth: int = 5,
                 min_stop: int = MIN_STOP_FRAMES, min_move: int = 5) -> np.ndarray:
    """Per-frame "this arm is not moving" from its 25 Hz anchor track.

    0.5 px/frame ≈ 12.5 px/s at 25 Hz. Runs < 0.4 s (stop) or < 0.2 s (move)
    are flipped so the labels do not flicker.
    """
    sp = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    sp = np.concatenate([sp, sp[-1:]])
    k = np.ones(max(1, int(smooth))) / max(1, int(smooth))
    sp = np.convolve(np.nan_to_num(sp, nan=1e3), k, mode="same")
    return _clean_mask(sp < thr, min_stop, min_move)


def _merge_short_stops(phases: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """A stop shorter than 1 s is not a state: merge it into a neighbour."""
    ph = [list(p) for p in phases]
    i = 0
    while i < len(ph):
        name, a, b = ph[i]
        if name == "stop" and (b - a + 1) < MIN_STOP_FRAMES and len(ph) > 1:
            if i > 0:
                ph[i - 1][2] = b
            else:
                ph[i + 1][1] = a
            ph.pop(i)
            i = max(0, i - 1)
        else:
            i += 1
    return [tuple(p) for p in ph]


def _split_phase(name: str, a: int, b: int, static: np.ndarray):
    """Split reach/move into motion runs and ``stop`` runs; keep pick/place."""
    if name not in ("reach", "move"):
        return [(name, a, b)]
    out = []
    t = a
    while t <= b:
        v = bool(static[t])
        s = t
        while t <= b and bool(static[t]) == v:
            t += 1
        out.append(("stop" if v else name, s, t - 1))
    return out


def _place_minima(z: np.ndarray, c: int, o: int) -> list[int]:
    """Indices of placement minima after the initial pick lift (relative to c)."""
    seg = z[c:o + 1]
    rise = None
    for i in range(1, len(seg)):
        if seg[i] - seg[:i + 1].min() > 0.03:
            rise = i
            break
    if rise is None:
        return []
    out = []
    for i in range(rise + 1, len(seg) - 1):
        if seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1]:
            lo, hi = max(0, i - 15), min(len(seg), i + 16)
            prom = min(seg[lo:i + 1].max() - seg[i], seg[i:hi].max() - seg[i])
            if prom > 0.015:
                out.append(i)
    # dedupe minima closer than 8 frames
    ded = []
    for i in out:
        if ded and i - ded[-1] < 8:
            if seg[i] < seg[ded[-1]]:
                ded[-1] = i
        else:
            ded.append(i)
    return ded


def _split_cycle(ep: Episode, arm: str, c: int, o: int,
                 g: np.ndarray) -> list[tuple[int, int, bool]]:
    """Split one close→open cycle into object transfers.

    Two cases:
    * the gripper **partially reopens** between transfers (local max above
      closed + 25% of the range);
    * one grasp carries several objects and the arm makes **several placement
      dips** (EE z minima) without reopening.

    Returns ``(close, open, continued)``; ``continued=True`` means the gripper
    was already closed at the split, so the second part has no pick phase.
    """
    lo, hi = float(g[c:o + 1].min()), float(g[o])
    release = lo + 0.25 * max(1e-6, hi - lo)
    peaks = [f for f in range(c + 5, o - 4)
             if g[f] >= g[f - 1] and g[f] >= g[f + 1] and g[f] > release]
    if peaks:
        p = peaks[0]
        if p - c >= MIN_SUBTASK and o - p >= MIN_SUBTASK:
            return [(c, p, False), (p, o, False)]

    z = ep.L[:, 2] if arm == "L" else ep.R[:, 2]
    mins = _place_minima(z, c, o)
    if len(mins) >= 2:
        a, b = mins[0], mins[1]
        s = c + int(a + np.argmax(z[c + a:c + b + 1]))
        if s - c >= MIN_SUBTASK and o - s >= MIN_SUBTASK:
            return [(c, s, False), (s, o, True)]
    return [(c, o, False)]


def _grip_transitions(g: np.ndarray, c: int, o: int,
                      eps: float = 0.10) -> tuple[int, int, int, int]:
    """Find the real open→closed→open transition frames around a cycle.

    ``find_cycles`` returns the threshold crossing ``c``/``o``; the gripper
    starts closing a few frames earlier and starts opening before ``o``.
    Returns ``(close_start, fully_closed, open_start, fully_open)`` where

      reach ends at ``close_start`` (still fully open),
      pick  = close_start+1 .. fully_closed,
      move  = fully_closed+1 .. open_start-1,
      place = open_start .. fully_open.
    """
    pre = g[max(0, c - 30):c]
    open_level = float(np.percentile(pre, 90)) if len(pre) else float(g[c])
    closed_level = float(np.percentile(g[c:o + 1], 10))
    span = open_level - closed_level
    if span < 0.05:
        return max(0, c - 1), c, o, o
    hi = open_level - 0.03 * span   # reach must be fully open
    lo = closed_level + 0.10 * span  # closed plateau tolerance

    close_start = c
    while close_start > 0 and g[close_start - 1] < hi:
        close_start -= 1

    fully_closed = c
    while fully_closed < o and g[fully_closed] > lo:
        fully_closed += 1
    if fully_closed >= o:
        fully_closed = o - 1

    open_start = fully_closed
    while open_start < len(g) - 1 and g[open_start] <= lo:
        open_start += 1

    fully_open = o
    while fully_open < len(g) - 1 and g[fully_open] < hi:
        fully_open += 1
    return int(close_start), int(fully_closed), int(open_start), int(fully_open)


def _ensure_min_transitions(phases: list[tuple[str, int, int]],
                            min_len: int = 25) -> list[tuple[str, int, int]]:
    """Pad pick/place to at least ``min_len`` frames, borrowing from **both**
    sides (half from reach/previous, half from move/next; shortfall from the
    side with room). This keeps the open/close transition centred instead of
    stretching only one way.
    """
    ph = [list(p) for p in phases]
    for _ in range(50):
        ph = [p for p in ph if p[2] >= p[1]]
        changed = False
        for i, (name, a, b) in enumerate(ph):
            if name not in ("pick", "place") or (b - a + 1) >= min_len:
                continue
            need = min_len - (b - a + 1)
            left = i - 1 if i > 0 else None
            right = i + 1 if i + 1 < len(ph) else None
            avail_l = (a - ph[left][1]) if left is not None else 0
            avail_r = (ph[right][2] - b) if right is not None else 0

            def _take(sides, need):
                """balanced: half of the original need on each side, then fill"""
                out = {"l": 0, "r": 0}
                half = need // 2
                for side in sides:
                    take = min(half, avail_l if side == "l" else avail_r)
                    out[side] = take; need -= take
                for side in sides:
                    avail = (avail_l if side == "l" else avail_r) - out[side]
                    add = min(need, avail)
                    out[side] += add; need -= add
                return out

            if name == "place":
                # place must end at fully_open: borrow only from the left (move),
                # never from the post-place reach on the right
                non = ["l"] if left is not None and ph[left][0] not in ("pick", "place") else []
                tr = ["l"] if left is not None and "l" not in non else []
            else:
                non = []
                if left is not None and ph[left][0] not in ("pick", "place"):
                    non.append("l")
                if right is not None and ph[right][0] not in ("pick", "place"):
                    non.append("r")
                tr = [x for x in ("l", "r") if x not in non
                      and ((x == "l" and left is not None) or (x == "r" and right is not None))]
            t_l = _take(non, need)
            need2 = need - t_l["l"] - t_l["r"]
            t_r = _take(tr, need2) if need2 > 0 else {"l": 0, "r": 0}
            take_l = t_l["l"] + t_r["l"]; take_r = t_l["r"] + t_r["r"]
            if take_l or take_r:
                a -= take_l; b += take_r
                if left is not None:
                    ph[left][2] = a - 1
                if right is not None:
                    ph[right][1] = b + 1
                changed = True
            ph[i] = [name, a, b]
        if not changed:
            break
    ph = [p for p in ph if p[2] >= p[1]]
    for i in range(len(ph) - 1):
        if ph[i][0] == "pick" and ph[i + 1][0] == "place":
            a, b = ph[i][1], ph[i + 1][2]
            if b - a + 1 >= 2 * min_len:
                ph[i] = ["pick", a, a + min_len - 1]
                ph[i + 1] = ["place", a + min_len, b]
    return [(name, a, b) for name, a, b in ph if b >= a]


def _arm_subtasks(ep: Episode, arm: str, g: np.ndarray,
                  static: np.ndarray | None = None) -> list[Subtask]:
    """One arm's grasp cycles; its phases cover [0, T-1] for this arm.

    Phase semantics:
      reach = gripper fully open (it does *not* start closing),
      pick  = open -> closed,
      move  = fully closed,
      place = closed -> open.
    """
    cycles = find_cycles(g)
    trans = [_grip_transitions(g, c, o) for c, o in cycles]
    subs: list[Subtask] = []
    prev_end = -1
    for i, (c, o) in enumerate(cycles):
        cs, fc, os, fo = trans[i]
        next_cs = trans[i + 1][0] if i + 1 < len(cycles) else ep.T
        cs = max(0, cs)
        fc = max(cs + 1, min(fc, o))
        os = max(fc + 1, os)
        fo = max(os, fo)

        reach_start = 0 if prev_end < 0 else prev_end + 1
        reach_start = min(reach_start, max(cs, 0))
        phases: list[tuple[str, int, int]] = []
        if cs - 1 >= reach_start:
            phases.append(("reach", reach_start, cs - 1))
        if fc >= cs:
            phases.append(("pick", cs, fc))
        if os - 1 >= fc + 1:
            phases.append(("move", fc + 1, os - 1))
        place_start = max(os, fc + 1)
        # place ends when the gripper is fully open; the retract is a reach
        place_end = min(fo, next_cs - 1, ep.T - 1)
        if place_end >= place_start:
            phases.append(("place", place_start, place_end))
        elif phases:
            phases[-1] = (phases[-1][0], phases[-1][1], next_cs - 1)
        phases = _ensure_min_transitions(phases, 25)
        if static is not None:
            split = []
            for name, a, b in phases:
                split.extend(_split_phase(name, a, b, static))
            phases = _merge_short_stops(split)
        subs.append(Subtask(index=0, arm=arm, t_close=int(c), t_open=int(o),
                            phases=phases))
        prev_end = phases[-1][2]
    if subs and prev_end < ep.T - 1:
        # post-place retract / return: a reach with the gripper fully open
        last = subs[-1]
        phases = list(last.phases) + [("reach", prev_end + 1, ep.T - 1)]
        if static is not None:
            split = []
            for name, a, b in phases:
                split.extend(_split_phase(name, a, b, static))
            phases = _merge_short_stops(split)
        last.phases = phases
    return subs


def segment_episode(ep: Episode) -> list[Subtask]:
    """Both arms' grasp cycles → subtasks, ordered by close time.

    Each arm is segmented **independently** (its own phases cover [0, T-1] for
    that arm), then the two lists are merged. Subtasks of different arms can
    overlap in time — that is the real two-handed structure (e.g. the left hand
    holds the pen holder from frame 81 to 881 while the right hand picks pens).
    """
    Luv, _Lv, Ruv, _Rv = project_anchors(ep)
    static_L = _static_mask(Luv)
    static_R = _static_mask(Ruv)
    subs = (_arm_subtasks(ep, "L", ep.gL, static_L)
            + _arm_subtasks(ep, "R", ep.gR, static_R))
    # group by hand: left subtasks first (by close), then right
    subs.sort(key=lambda s: (0 if s.arm == "L" else 1, s.t_close))
    for i, s in enumerate(subs):
        s.index = i
    if subs:
        # only the globally last-ending subtask absorbs the tail → full coverage
        tail = max(subs, key=lambda s: s.phases[-1][2])
        name, a, _b = tail.phases[-1]
        tail.phases[-1] = (name, a, ep.T - 1)
        # tail may have made a short pick/place long enough to re-pad
        tail.phases = _ensure_min_transitions(tail.phases, 25)
    if not subs:
        return [Subtask(index=0, arm="-", t_close=0, t_open=ep.T - 1,
                        phases=[("reach", 0, ep.T - 1)])]
    return subs


def state_at(subtasks: list[Subtask], t: int) -> tuple[Subtask | None, str]:
    for st in subtasks:
        ph = st.phase_at(t)
        if ph is not None:
            return st, ph
    return None, "idle"


# ---------------------------------------------------------------- timeline


def render_timeline(ep: Episode, subtasks: list[Subtask], out_png: Path, task: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=120)
    fr = np.arange(ep.T)
    ax.plot(fr, ep.gL, color="#2b6cb0", lw=1.0, label="left gripper")
    ax.plot(fr, ep.gR, color="#38a169", lw=1.0, label="right gripper")
    ax.axhline(CLOSE_THR, color="#999", lw=0.8, ls=":")
    ax.set_xlim(0, ep.T - 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("frame")
    ax.set_ylabel("gripper (open 1 → closed 0)")

    y0, h = 1.12, 0.16
    for st in subtasks:
        x0 = st.phases[0][1]
        x1 = st.phases[-1][2]
        ax.axvspan(x0, x1, ymin=0, ymax=0.06, color="#1f2937", alpha=0.15)
        for name, a, b in st.phases:
            ax.axvspan(a, b, ymin=0.0, ymax=0.06, color=PHASE_COLOR[name][1], alpha=0.9)
            ax.text((a + b) / 2, 1.30, name, ha="center", fontsize=6.5, color="#111",
                    rotation=0 if b - a > 25 else 90)
        ax.text(x0, 1.46, f"#{st.index} {st.arm}", fontsize=8, color="#111")
        ax.axvline(st.t_close, color="#e53e3e", lw=1.0, ls="--")
        ax.axvline(st.t_open, color="#e53e3e", lw=1.0, ls=":")
    ax.set_title(f"{task}  task timeline: {len(subtasks)} subtasks  "
                 f"(dashed=close, dotted=open; bar = reach/pick/move/place)", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- video


def _label(img, text, org, color=(240, 240, 240), scale=0.5, outline=True):
    if outline:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_video(ep: Episode, subtasks: list[Subtask], out_mp4: Path, task: str,
                 fps: int = 25, tail: int = 50, size: tuple[int, int] | None = None) -> None:
    Luv, _, Ruv, _ = project_anchors(ep)
    h, w = ep.hw["cam_head"]
    if size is not None:
        w, h = size
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法打开视频写入器：{out_mp4}")

    for t in range(ep.T):
        img = ep.read_color(t)
        if (w, h) != (ep.hw["cam_head"][1], ep.hw["cam_head"][0]):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        sx, sy = w / ep.hw["cam_head"][1], h / ep.hw["cam_head"][0]

        # anchor trails (last `tail` frames), both arms
        for uv, color in ((Luv, (176, 108, 43)), (Ruv, (105, 161, 56))):
            seg = uv[max(0, t - tail):t + 1]
            for i in range(len(seg) - 1):
                a = (int(seg[i, 0] * sx), int(seg[i, 1] * sy))
                b = (int(seg[i + 1, 0] * sx), int(seg[i + 1, 1] * sy))
                cv2.line(img, a, b, color, 2, cv2.LINE_AA)
            if np.isfinite(uv[t]).all():
                p = (int(uv[t, 0] * sx), int(uv[t, 1] * sy))
                cv2.circle(img, p, 6, color, -1, cv2.LINE_AA)
                cv2.circle(img, p, 8, (255, 255, 255), 1, cv2.LINE_AA)

        st, phase = state_at(subtasks, t)
        # banner
        cv2.rectangle(img, (0, 0), (w, 24), (40, 40, 40), -1)
        if st is None:
            txt = f"{task}  frame {t}/{ep.T - 1}  idle / approach"
            color = (200, 200, 200)
        else:
            txt = (f"{task}  frame {t}/{ep.T - 1}  subtask {st.index + 1}/{len(subtasks)} "
                   f"({st.arm})  {phase}  t={t / fps:.2f}s")
            color = PHASE_COLOR[phase][0]
        _label(img, txt, (8, 16), color, 0.46, outline=False)

        # progress bar: subtask phases
        y = h - 12
        x0, x1 = 10, w - 10
        for st2 in subtasks:
            for name, a, b in st2.phases:
                ax0 = int(x0 + (x1 - x0) * a / max(1, ep.T - 1))
                ax1 = int(x0 + (x1 - x0) * b / max(1, ep.T - 1))
                cv2.rectangle(img, (ax0, y - 4), (max(ax0 + 1, ax1), y + 4),
                              PHASE_COLOR[name][0], -1)
        cx = int(x0 + (x1 - x0) * t / max(1, ep.T - 1))
        cv2.line(img, (cx, y - 8), (cx, y + 8), (0, 215, 255), 2)
        _label(img, f"subtask {st.index + 1 if st else 0}/{len(subtasks)}", (x0, y - 10),
               (240, 240, 240), 0.4, outline=False)

        writer.write(img)
    writer.release()


# ---------------------------------------------------------------- page


def write_page(out_dir: Path, task: str, ep: Episode, subtasks: list[Subtask],
               fps: int, video_name: str = "episode.mp4") -> Path:
    rows = []
    for st in subtasks:
        ph = " · ".join(f"{n} [{a},{b}]" for n, a, b in st.phases)
        rows.append((st.index, st.arm, st.t_close, st.t_open, ph,
                     st.phases[0][1], st.phases[-1][2]))
    data = json.dumps([{"index": r[0], "arm": r[1], "close": r[2], "open": r[3],
                        "phases": [[n, a, b] for n, a, b in st.phases],
                        "start": r[5], "end": r[6]}
                       for st, r in zip(subtasks, rows)], ensure_ascii=False)
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{task} · 完整任务时间线</title>
<style>
 body{{margin:0;background:#0b0f14;color:#d7e0ea;font:13px/1.5 -apple-system,"Segoe UI",sans-serif}}
 header{{padding:10px 16px;border-bottom:1px solid #22303d}}
 h1{{font-size:15px;margin:0}} .meta{{color:#8aa0b4;font-size:12px}}
 main{{padding:12px 16px}} video{{width:100%;max-width:960px;border:1px solid #22303d;border-radius:6px}}
 img{{width:100%;max-width:1200px;border:1px solid #22303d;border-radius:6px;margin-top:12px;background:#fff}}
 table{{border-collapse:collapse;margin-top:12px;font:12px ui-monospace,Menlo,monospace}}
 td,th{{padding:4px 8px;border-bottom:1px solid #1c2733;text-align:left}} th{{color:#8aa0b4}}
 tr:hover{{background:#131a22;cursor:pointer}} .pill{{padding:1px 6px;border-radius:999px;
 border:1px solid #22303d;font-size:11px;color:#8aa0b4}}
 #now{{margin-left:12px;color:#f6c90e}}
</style></head><body>
<header><h1>{task} · 完整任务（{len(subtasks)} 个子任务）</h1>
<div class="meta">frames 0–{ep.T - 1} · {fps} Hz · 灰色=idle/approach，彩色=reach/pick/move/place
<span id="now"></span></div></header>
<main>
<video id="v" src="{video_name}" controls autoplay loop muted></video>
<img src="timeline.png" alt="timeline">
<table><thead><tr><th>#</th><th>arm</th><th>close</th><th>open</th><th>phases (frames)</th></tr></thead>
<tbody id="tb"></tbody></table>
</main>
<script>
const SUBS = {data};
const V = document.getElementById("v"), TB = document.getElementById("tb"), NOW = document.getElementById("now");
for (const s of SUBS) {{
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${{s.index + 1}}</td><td>${{s.arm}}</td><td>${{s.close}}</td><td>${{s.open}}</td>` +
    `<td>${{s.phases.map(([n,a,b]) => `<span class="pill" data-f="${{a}}">${{n}} [${{a}},${{b}}]</span>`).join(" ")}}</td>`;
  tr.onclick = () => {{ V.currentTime = Math.max(0, (s.start - 1) / {fps}); V.play(); }};
  tr.querySelectorAll(".pill").forEach(p => p.onclick = (e) => {{
    e.stopPropagation(); V.currentTime = Math.max(0, (parseInt(p.dataset.f) - 1) / {fps}); V.play();
  }});
  TB.appendChild(tr);
}}
V.addEventListener("timeupdate", () => {{
  const f = Math.round(V.currentTime * {fps});
  const s = SUBS.find(s => f >= s.start && f <= s.end);
  NOW.textContent = " · frame " + f + (s ? ` · subtask ${{s.index + 1}}` : " · idle");
}});
</script></body></html>"""
    p = out_dir / "index.html"
    p.write_text(html, encoding="utf-8")
    return p


def build_task(task: str, root: Path, out_root: Path, fps: int = 25,
               episode_name: str = "episode_0000000.hdf5",
               make_video: bool = True) -> dict:
    path = root / task / "arx_x5" / "data" / episode_name
    ep = load_episode(path, task=task, load_gray=False)
    subtasks = segment_episode(ep)
    out = out_root / task
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "task": task, "episode": episode_name, "T": ep.T, "fps": ep.fps,
        "subtasks": [{"index": s.index, "arm": s.arm, "t_close": s.t_close,
                      "t_open": s.t_open,
                      "phases": [{"name": n, "start": a, "end": b} for n, a, b in s.phases]}
                     for s in subtasks],
    }
    (out / "task.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    render_timeline(ep, subtasks, out / "timeline.png", task)
    if make_video:
        render_video(ep, subtasks, out / "episode.mp4", task, fps=fps)
    write_page(out, task, ep, subtasks, fps)
    print(f"[{task}] T={ep.T} subtasks={len(subtasks)} -> {out}")
    for s in subtasks:
        print(f"   #{s.index + 1} {s.arm}  close {s.t_close} open {s.t_open}  "
              + "  ".join(f"{n}[{a},{b}]" for n, a, b in s.phases))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="完整任务 → 子任务 → 四阶段 时间线/视频")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--episode", default="episode_0000000.hdf5")
    ap.add_argument("--out", type=Path, default=Path("outputs/task_timeline"))
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    built = []
    for task in args.tasks:
        try:
            payload = build_task(task, args.root, args.out, fps=args.fps,
                                 episode_name=args.episode, make_video=not args.no_video)
            built.append(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[!] {task} 失败：{exc}")
    if built:
        links = "".join(
            f'<li><a href="{p["task"]}/index.html">{p["task"]}</a> '
            f'<span class="meta">T={p["T"]} · {len(p["subtasks"])} subtasks</span></li>'
            for p in built)
        (args.out / "index.html").write_text(
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<title>完整任务时间线</title><style>'
            'body{background:#0b0f14;color:#d7e0ea;font:14px/1.7 -apple-system,sans-serif;padding:24px}'
            'a{color:#f6c90e}li{margin:6px 0}.meta{color:#8aa0b4;font-size:12px}'
            '</style></head><body><h1>完整任务时间线</h1><ul>' + links + '</ul></body></html>',
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
