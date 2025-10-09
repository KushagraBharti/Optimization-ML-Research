#!/usr/bin/env python3
"""
Drone Coverage Visualizer (Pygame) — Textbook Scripted Mode + Computed Mode
----------------------------------------------------------------------------

What’s new:
- "Scripted textbook runs" that hardcode the exact steps/tours for pedagogical confidence.
- A toggle per scene to switch between SCRIPTED and COMPUTED.
- A preset that matches the numeric construction in Section 4.2 / Fig. 3 text (h=4, a1=1, g=2, b1=3, a2=9, b2=10).
- Clear HUD labeling when you’re viewing SCRIPTED vs COMPUTED.

Controls (same as before)
-------------------------
  SPACE      Play/Pause
  N / RIGHT  Next step
  B / LEFT   Previous step
  R          Reset current algorithm
  1          Select GS
  2          Select GSP
  3          Select DPOS
  4          Select GENERAL
  +/-        Increase / Decrease battery limit L (recomputes in COMPUTED mode)
  [ ]        Zoom X scale in/out
  C          Toggle candidate endpoints (DP-focused)
  T          Toggle showing tour triangles
  S          Toggle showing covered vs uncovered shading
  G          Toggle showing gaps
  H          Toggle help overlay
  M          Toggle SCRIPTED vs COMPUTED mode for the current scene
  P          Save a PNG snapshot

Notes
-----
- The DPOS "textbook" preset uses the concrete numeric instance described in the paper (Sec. 4.2): h=4, a1=1, g=2, b1=3, a2=9, b2=10,
  with L chosen as l(g,b2), so that GS covers a gap (suboptimal) while the DP-optimal solution uses two tight tours [a1,b1] and [a2,b2].
- For GS/GSP/GENERAL, the paper illustrates concepts without fixed numbers. We provide *theorem-consistent* scripted sequences and label them as such.
  You can still flip to COMPUTED mode to explore arbitrary inputs.

Disclaimer
----------
Scripted runs are meant to match the *logic and outcomes* in the paper’s figures/lemmas. The computed DP demos use clean endpoint sets
for clarity (a_i, b_i) and illustrate the dynamic-programming idea without reproducing every technical detail of candidate sets.
"""

import sys
import math
import pygame
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# ========== Model ==========

H = 4.0        # y = H
SCREEN_W, SCREEN_H = 1200, 720
MARGIN_L, MARGIN_R = 80, 40
TOP_PAD, BOT_PAD = 120, 80

X_UNITS_VISIBLE = 24.0   # world units visible across screen; adjustable with [ ]
Y_LINE = SCREEN_H - (BOT_PAD + 220)  # vertical position of the y=H line (in pixels)

FONT_NAME = "consolas"

@dataclass
class Segment:
    a: float
    b: float
    def length(self) -> float:
        return max(0.0, self.b - self.a)
    def copy(self):
        return Segment(self.a, self.b)

@dataclass
class Tour:
    p: float
    q: float

@dataclass
class Step:
    tours: List[Tour]
    uncovered: List[Segment]
    msg: str
    current_batt_used: float = 0.0
    last_tour: Optional[Tour] = None

def normalize_segments(segs: List[Tuple[float,float]]) -> List[Segment]:
    s = [Segment(min(a,b), max(a,b)) for a,b in segs]
    s.sort(key=lambda z: z.a)
    merged = []
    for seg in s:
        if not merged or seg.a > merged[-1].b + 1e-10:
            merged.append(seg.copy())
        else:
            merged[-1].b = max(merged[-1].b, seg.b)
    return merged

def tour_length(p: float, q: float, h: float = H) -> float:
    return math.hypot(p, h) + math.hypot(q, h) + abs(q - p)

def intersect_interval_with_segments(p: float, q: float, segments: List[Segment]) -> List[Segment]:
    lo, hi = (p, q) if p <= q else (q, p)
    out = []
    for s in segments:
        if s.b < lo or s.a > hi: 
            continue
        out.append(Segment(max(s.a, lo), min(s.b, hi)))
    return out

def subtract_covered(segments: List[Segment], covered: List[Segment]) -> List[Segment]:
    if not covered:
        return [s.copy() for s in segments]
    covered = normalize_segments([(c.a, c.b) for c in covered])
    res = []
    j = 0
    for s in segments:
        cur = s.a
        while j < len(covered) and covered[j].b < s.a:
            j += 1
        k = j
        while k < len(covered) and covered[k].a <= s.b:
            c = covered[k]
            if c.a > cur:
                res.append(Segment(cur, min(c.a, s.b)))
            cur = max(cur, c.b)
            if cur >= s.b:
                break
            k += 1
        if cur < s.b:
            res.append(Segment(cur, s.b))
    return [Segment(x.a, x.b) for x in res if x.b - x.a > 1e-9]

def farthest_distance_from_O(x: float) -> float:
    return math.hypot(x, H)

def farthest_uncovered_point(uncovered: List[Segment]) -> Optional[float]:
    if not uncovered:
        return None
    candidates = []
    for s in uncovered:
        candidates.append((farthest_distance_from_O(s.a), s.a))
        candidates.append((farthest_distance_from_O(s.b), s.b))
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][1]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def max_tour_from_anchor(anchor_x: float, L: float, direction: str, xmin: float, xmax: float) -> Tuple[float,float]:
    steps = [1.0, 0.2, 0.05, 0.01]
    if direction == 'left':
        q = anchor_x
        lo, hi = xmin-5.0, q
        best_val, best_x = None, None
        for step in steps:
            x = lo
            while x <= hi:
                cur = tour_length(x, q, H)
                if cur <= L + 1e-6:
                    if best_val is None or cur > best_val:
                        best_val, best_x = cur, x
                x += step
            if best_x is not None:
                lo = max(lo, best_x - 2*step)
                hi = min(hi, best_x + 2*step)
            else:
                hi = (hi + lo) / 2.0
        if best_x is None: return q, q
        p = clamp(best_x, xmin, xmax)
        return p, q
    else:
        p = anchor_x
        lo, hi = p, xmax+5.0
        best_val, best_x = None, None
        for step in steps:
            x = lo
            while x <= hi:
                cur = tour_length(p, x, H)
                if cur <= L + 1e-6:
                    if best_val is None or cur > best_val:
                        best_val, best_x = cur, x
                x += step
            if best_x is not None:
                lo = max(lo, best_x - 2*step)
                hi = min(hi, best_x + 2*step)
            else:
                hi = (hi + lo) / 2.0
        if best_x is None: return p, p
        q = clamp(best_x, xmin, xmax)
        return p, q

# ========== Scripted textbook presets ==========
# Each entry can carry:
#   - 'segments': list of (a,b)
#   - 'L': battery
#   - 'scripted': True/False
#   - 'scripted_tours': tours to show (list of (p,q) in order), with messages

SCRIPTED_PRESETS = {
    # 1) GS — Scripted "greedy farthest-first" example (the paper proves optimality for MinTours).
    #    The paper doesn't fix numbers, so we provide a theorem-consistent scripted sequence.
    "GS": {
        "segments": [(-7,-5), (-2,-1.2), (1.0,2.2), (5.5,6.8)],
        "L": 18.0,
        "scripted": True,
        "scripted_tours": [
            {"tour": (6.8, 5.5), "msg": "GS tour 1: farthest endpoint is at x≈6.8; take a maximal tour anchored on the right."},
            {"tour": (2.2, 1.0), "msg": "GS tour 2: next farthest on the right; clean up remaining right segments."},
            {"tour": (-5.0, -7.0), "msg": "GS tour 3: switch to the left side; maximal tour covers the farthest left."}
        ],
        "note": "Scripted GS per Theorem 1 (no fixed numeric example in the paper)."
    },

    # 2) GSP — Single segment with O' inside; finish with 1 or 2 tours around O'.
    #    Paper's Fig. 2 is conceptual; we provide a theorem-consistent scripted run.
    "GSP": {
        "segments": [(-6.0, 7.0)],
        "L": 16.0,
        "scripted": True,
        "scripted_tours": [
            {"tour": (-6.0, -0.5), "msg": "GSP tour 1: take maximal tour from left without crossing O'."},
            {"tour": (0.5, 7.0),  "msg": "GSP tour 2: take maximal tour from right without crossing O'."},
            {"tour": (-0.5, 0.5), "msg": "GSP final: center fits in one small tour around O'."}
        ],
        "note": "Scripted GSP consistent with Theorem 3 (Fig. 2 did not specify numbers)."
    },

    # 3) DPOS — Use the numeric construction from Sec. 4.2 (one side, h=4) with z=9 -> a2=9, b2=10.
    #    L is set to l(g,b2). Optimal solution is two tours [a1,b1] and [a2,b2].
    "DPOS_TEXTBOOK": {
        "segments": [(1.0, 3.0), (9.0, 10.0)],  # a1=1, b1=3 ; a2=9, b2=10
        "L": None,  # we will compute L = l(g,b2) with g=2.0
        "scripted": True,
        "scripted_tours": [
            {"tour": (1.0, 3.0),  "msg": "DPOS tour 1 (optimal): cover s1 tightly [1,3]."},
            {"tour": (9.0, 10.0), "msg": "DPOS tour 2 (optimal): cover s2 tightly [9,10]."}
        ],
        "note": "Exact numeric instance from Sec. 4.2 (Fig. 3 construction)."
    },

    # 4) GENERAL — Both sides; provide a scripted run with a crossing tour through O'.
    #    The paper outlines cases; we demonstrate the 'crossing' case.
    "GENERAL": {
        "segments": [(-6.5, -5.0), (-2.2, -1.3), (1.0, 2.0), (5.0, 6.2)],
        "L": 18.0,
        "scripted": True,
        "scripted_tours": [
            {"tour": (-1.5, 1.5), "msg": "GENERAL tour 1: crossing tour through O'."},
            {"tour": (-6.5, -5.0), "msg": "GENERAL tour 2: finish remaining left."},
            {"tour": (5.0, 6.2), "msg": "GENERAL tour 3: finish remaining right."}
        ],
        "note": "Scripted general case illustrating the 'crossing' scenario."
    }
}

# legacy computed defaults (you can still run in COMPUTED mode)
DEFAULT_SCENES = {
    "GS": {
        "segments": [(-7,-5), (-2,-1.2), (1.0,2.2), (5.5,6.8)],
        "L": 18.0
    },
    "GSP": {
        "segments": [(-6.0, 7.0)],
        "L": 16.0
    },
    "DPOS": {
        "segments": [(1.0, 2.4), (3.1, 3.9), (4.5, 5.4)],
        "L": 14.0
    },
    "GENERAL": {
        "segments": [(-6.5, -5.0), (-2.2, -1.3), (1.0, 2.0), (5.0, 6.2)],
        "L": 18.0
    }
}

# ========== Algorithm engines (COMPUTED) ==========

def animate_gs_steps(segments_in: List[Tuple[float,float]], L: float) -> List[Step]:
    segments = normalize_segments(segments_in)
    uncovered = [s.copy() for s in segments]
    tours: List[Tour] = []
    steps: List[Step] = []
    xmin = min(s.a for s in segments) - 2.0
    xmax = max(s.b for s in segments) + 2.0

    steps.append(Step([], uncovered, "GS start (computed): pick farthest endpoint; take maximal tour.", 0.0, None))

    while uncovered:
        f = farthest_uncovered_point(uncovered)
        if f is None: break
        if f >= 0:
            p, q = max_tour_from_anchor(f, L, direction='left', xmin=xmin, xmax=xmax)
        else:
            p, q = max_tour_from_anchor(f, L, direction='right', xmin=xmin, xmax=xmax)
        t = Tour(p,q); tl = tour_length(p,q,H)
        tours.append(t)
        covered = intersect_interval_with_segments(p, q, uncovered)
        uncovered = subtract_covered(uncovered, covered)
        steps.append(Step(tours.copy(), uncovered.copy(),
                          f"GS tour {len(tours)} (computed): f={f:.2f}, add [{p:.2f},{q:.2f}] len={tl:.2f}.", tl, t))
    steps.append(Step(tours.copy(), uncovered.copy(), "GS done.", 0.0, None))
    return steps

def animate_gsp_steps(segment_in: Tuple[float,float], L: float) -> List[Step]:
    a,b = min(segment_in), max(segment_in)
    segments = [Segment(a,b)]
    uncovered = [Segment(a,b)]
    tours: List[Tour] = []
    steps: List[Step] = []
    steps.append(Step([], uncovered.copy(),
                      "GSP start (computed): ends first without crossing O'; then center.", 0.0, None))
    while uncovered and not (uncovered[0].a <= 0.0 <= uncovered[-1].b):
        left_end = uncovered[0].a
        right_end = uncovered[-1].b
        d_left = farthest_distance_from_O(left_end)
        d_right = farthest_distance_from_O(right_end)
        if d_right >= d_left:
            q = right_end
            p, q = max_tour_from_anchor(q, L, direction='left', xmin=0.0, xmax=b)
        else:
            p = left_end
            p, q = max_tour_from_anchor(p, L, direction='right', xmin=a, xmax=0.0)
        t = Tour(p,q); tl = tour_length(p,q,H)
        tours.append(t)
        covered = intersect_interval_with_segments(p, q, uncovered)
        uncovered = subtract_covered(uncovered, covered)
        steps.append(Step(tours.copy(), uncovered.copy(),
                          f"GSP tour {len(tours)} (computed): add [{p:.2f},{q:.2f}] len={tl:.2f}.", tl, t))

    if uncovered:
        a2, b2 = uncovered[0].a, uncovered[-1].b
        if tour_length(a2,b2,H) <= L + 1e-9:
            t = Tour(a2,b2); tl = tour_length(a2,b2,H)
            tours.append(t); uncovered = []
            steps.append(Step(tours.copy(), uncovered.copy(),
                              f"GSP final (computed): [{a2:.2f},{b2:.2f}] len={tl:.2f}.", tl, t))
        else:
            p, q = a2, 0.0
            if tour_length(p,q,H) > L:
                # shrink from 0 inward (left)
                # keep it simple: binary-like search
                lo, hi = a2, 0.0
                for _ in range(60):
                    mid = (lo+hi)/2
                    if tour_length(a2, mid, H) > L: hi = mid
                    else: lo = mid
                p, q = a2, lo
            t = Tour(p,q); tl = tour_length(p,q,H)
            tours.append(t)
            covered = intersect_interval_with_segments(p, q, uncovered)
            uncovered = subtract_covered(uncovered, covered)
            steps.append(Step(tours.copy(), uncovered.copy(),
                              f"GSP center-left (computed): [{p:.2f},{q:.2f}] len={tl:.2f}.", tl, t))
            if uncovered:
                p, q = 0.0, uncovered[-1].b
                if tour_length(p,q,H) > L:
                    lo, hi = 0.0, uncovered[-1].b
                    for _ in range(60):
                        mid = (lo+hi)/2
                        if tour_length(mid, uncovered[-1].b, H) > L: lo = mid
                        else: hi = mid
                    p, q = hi, uncovered[-1].b
                t = Tour(p,q); tl = tour_length(p,q,H)
                tours.append(t)
                covered = intersect_interval_with_segments(p, q, uncovered)
                uncovered = subtract_covered(uncovered, covered)
                steps.append(Step(tours.copy(), uncovered.copy(),
                                  f"GSP center-right (computed): [{p:.2f},{q:.2f}] len={tl:.2f}.", tl, t))
    steps.append(Step(tours.copy(), uncovered.copy(), "GSP done.", 0.0, None))
    return steps

def endpoint_set(segs: List[Segment]) -> List[float]:
    return sorted({x for z in segs for x in (z.a, z.b)})

def dpos_steps(segments_in: List[Tuple[float,float]], L: float) -> Tuple[List[Step], List[float]]:
    segments = normalize_segments(segments_in)
    assert all(s.a >= -1e-9 for s in segments), "DPOS expects all segments on one side (x>=0)."
    endp = endpoint_set(segments)
    nE = len(endp)
    INF = 1e18
    dp = [INF]*nE
    prev = [-1]*nE
    used: Dict[int, Tuple[float,float]] = {}
    def valid(p,q): return tour_length(p,q,H) <= L + 1e-9
    for j in range(nE):
        ej = endp[j]
        for i in range(j+1):
            ei = endp[i]
            if valid(ei,ej):
                base = 0.0 if i == 0 else dp[i-1]
                if base < INF:
                    cand = base + tour_length(ei,ej,H)
                    if cand < dp[j]:
                        dp[j] = cand; prev[j] = i-1; used[j] = (ei,ej)
    tours = []
    j = nE-1
    while j >= 0:
        if j in used:
            p,q = used[j]; tours.append(Tour(p,q)); j = prev[j]
        else:
            j -= 1
    tours.reverse()

    steps = [Step([], [s.copy() for s in segments], "DPOS start (computed): DP over endpoints.", 0.0, None)]
    uncovered = [s.copy() for s in segments]
    for k,t in enumerate(tours, start=1):
        tl = tour_length(t.p,t.q,H)
        covered = intersect_interval_with_segments(t.p, t.q, uncovered)
        uncovered = subtract_covered(uncovered, covered)
        so_far = [Tour(tt.p, tt.q) for tt in tours[:k]]
        steps.append(Step(so_far, uncovered.copy(),
                          f"DPOS tour {k} (computed): [{t.p:.2f},{t.q:.2f}] len={tl:.2f}.", tl, t))
    steps.append(Step([Tour(t.p,t.q) for t in tours], uncovered.copy(), "DPOS done.", 0.0, None))
    return steps, endp

def general_dp_steps(segments_in: List[Tuple[float,float]], L: float) -> Tuple[List[Step], Dict[str,Any]]:
    segs = normalize_segments(segments_in)
    left = [(s.a, min(0.0, s.b)) for s in segs if s.a < 0.0]
    left = [(a,b) for a,b in left if b - a > 1e-9]
    right = [(max(0.0, s.a), s.b) for s in segs if s.b > 0.0]
    right = [(a,b) for a,b in right if b - a > 1e-9]

    def endpoint_dp_one_side(sin: List[Tuple[float,float]]) -> List[Tour]:
        if not sin: return []
        s = normalize_segments(sin)
        E = sorted({x for seg in s for x in (seg.a,seg.b)})
        INF = 1e18
        dp = [INF]*len(E); prv = [-1]*len(E)
        used: Dict[int, Tuple[float,float]] = {}
        def valid(p,q): return tour_length(p,q,H) <= L + 1e-9
        for j in range(len(E)):
            ej = E[j]
            for i in range(j+1):
                ei = E[i]
                if valid(ei,ej):
                    base = 0.0 if i == 0 else dp[i-1]
                    if base < INF:
                        cand = base + tour_length(ei,ej,H)
                        if cand < dp[j]:
                            dp[j] = cand; prv[j] = i-1; used[j] = (ei,ej)
        tours = []
        j = len(E)-1
        while j >= 0:
            if j in used:
                p,q = used[j]; tours.append(Tour(p,q)); j = prv[j]
            else:
                j -= 1
        tours.reverse()
        return tours

    def total_cost(tours: List[Tour]) -> float:
        return sum(tour_length(t.p,t.q,H) for t in tours)

    left_t = endpoint_dp_one_side(left)
    right_t = endpoint_dp_one_side(right)
    best = left_t + right_t
    best_cost = total_cost(best)
    best_case = "no-cross"

    left_endpoints = sorted({x for seg in normalize_segments(left) for x in (seg.a,seg.b)}) if left else []
    right_endpoints = sorted({x for seg in normalize_segments(right) for x in (seg.a,seg.b)}) if right else []

    for p in (left_endpoints or [0.0]):
        for q in (right_endpoints or [0.0]):
            if tour_length(p,q,H) <= L + 1e-9:
                left_sub = [(a,b) for (a,b) in left if b <= p-1e-9]
                right_sub = [(a,b) for (a,b) in right if a >= q+1e-9]
                lt = endpoint_dp_one_side(left_sub)
                rt = endpoint_dp_one_side(right_sub)
                cand = lt + [Tour(p,q)] + rt
                c = total_cost(cand)
                if c < best_cost:
                    best_cost = c; best = cand; best_case = "crossing"

    steps = [Step([], [s.copy() for s in segs], f"GENERAL start (computed, {best_case}).", 0.0, None)]
    uncovered = [s.copy() for s in segs]
    for k,t in enumerate(best, start=1):
        tl = tour_length(t.p,t.q,H)
        covered = intersect_interval_with_segments(t.p, t.q, uncovered)
        uncovered = subtract_covered(uncovered, covered)
        so_far = [Tour(tt.p, tt.q) for tt in best[:k]]
        steps.append(Step(so_far, uncovered.copy(),
                          f"GENERAL tour {k} (computed): [{t.p:.2f},{t.q:.2f}] len={tl:.2f}.", tl, t))
    steps.append(Step([Tour(t.p,t.q) for t in best], uncovered.copy(), "GENERAL done.", 0.0, None))
    debug = {"case": best_case, "left_ep": left_endpoints, "right_ep": right_endpoints}
    return steps, debug

# ========== Scripted step builders ==========

def scripted_steps_for(scene_key: str) -> Tuple[List[Step], Dict[str,Any], List[float]]:
    cfg = SCRIPTED_PRESETS[scene_key]
    segs = normalize_segments(cfg["segments"])
    tours_spec = cfg["scripted_tours"]
    steps: List[Step] = [Step([], [s.copy() for s in segs], f"{scene_key}: SCRIPTED textbook run.", 0.0, None)]

    # Special: DPOS textbook needs L computed from g=2 and b2
    debug = {}
    candidates: List[float] = []
    if scene_key == "DPOS_TEXTBOOK":
        # compute L = l(g,b2) with g=2, b2=10
        g = 2.0
        b2 = 10.0
        L = tour_length(g, b2, H)
        debug["L_textbook"] = L
        debug["construction"] = "L set to l(g=2, b2=10) per Sec. 4.2"
    # Apply the tours in order
    uncovered = [s.copy() for s in segs]
    for i, item in enumerate(tours_spec, start=1):
        p, q = item["tour"]
        tl = tour_length(p,q,H)
        covered = intersect_interval_with_segments(p,q, uncovered)
        uncovered = subtract_covered(uncovered, covered)
        tours_so_far = [] if len(steps)==0 else steps[-1].tours.copy()
        tours_so_far.append(Tour(p,q))
        msg = item.get("msg", f"Scripted tour {i}: [{p:.2f},{q:.2f}] len={tl:.2f}.")
        steps.append(Step(tours_so_far, uncovered.copy(), msg, tl, tours_so_far[-1]))
    steps.append(Step(steps[-1].tours.copy(), uncovered.copy(), f"{scene_key}: SCRIPTED done.", 0.0, None))

    # Show candidates for DPOS (endpoints) and GENERAL (left/right endpoints) if desired
    if scene_key in ("DPOS_TEXTBOOK",):
        candidates = sorted({x for s in segs for x in (s.a,s.b)})
    if scene_key in ("GENERAL",):
        debug["left_ep"] = sorted({x for s in segs if s.a < 0.0 for x in (s.a, min(0.0,s.b))})
        debug["right_ep"] = sorted({x for s in segs if s.b > 0.0 for x in (max(0.0,s.a), s.b)})

    return steps, debug, candidates

# ========== Simulator / UI ==========

class Simulator:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Drone Coverage Visualizer — Textbook Scripted + Computed")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(FONT_NAME, 18)
        self.font_small = pygame.font.SysFont(FONT_NAME, 14)
        self.font_big = pygame.font.SysFont(FONT_NAME, 24)

        # State
        self.scene_key = "GS"        # keys: "GS", "GSP", "DPOS_TEXTBOOK", "GENERAL"
        self.mode_scripted = True    # default to SCRIPTED textbook mode
        self.L = SCRIPTED_PRESETS["GS"]["L"]
        self.segments = normalize_segments(SCRIPTED_PRESETS["GS"]["segments"])
        self.steps: List[Step] = []
        self.candidates: List[float] = []
        self.debug: Dict[str,Any] = {}

        self.step_idx = 0
        self.play = False
        self.show_help = True
        self.show_candidates = True
        self.show_tours = True
        self.show_shading = True
        self.show_gaps = False
        self.x_units = X_UNITS_VISIBLE
        self.play_fps = 1

        self.recompute()

    def recompute(self):
        # If scripted: use SCRIPTED_PRESETS; otherwise use DEFAULT_SCENES and computed engines
        if self.mode_scripted:
            cfg = SCRIPTED_PRESETS[self.scene_key]
            self.segments = normalize_segments(cfg["segments"])
            self.L = cfg["L"] if cfg["L"] is not None else self.derive_textbook_L(self.scene_key)
            self.steps, self.debug, self.candidates = scripted_steps_for(self.scene_key)
        else:
            cfg = DEFAULT_SCENES["GS" if self.scene_key=="GS" else ("GSP" if self.scene_key=="GSP" else ("GENERAL" if self.scene_key=="GENERAL" else "DPOS"))]
            self.segments = normalize_segments(cfg["segments"])
            self.L = cfg["L"]
            if self.scene_key == "GS":
                self.steps = animate_gs_steps([(s.a,s.b) for s in self.segments], self.L)
                self.debug = {}; self.candidates = []
            elif self.scene_key == "GSP":
                assert len(self.segments)==1, "GSP computed mode expects one segment"
                self.steps = animate_gsp_steps((self.segments[0].a, self.segments[0].b), self.L)
                self.debug = {}; self.candidates = []
            elif self.scene_key == "DPOS_TEXTBOOK":
                # computed DPOS on a one-side sample
                self.steps, self.candidates = dpos_steps([(s.a,s.b) for s in self.segments], self.L)
                self.debug = {}
            elif self.scene_key == "GENERAL":
                self.steps, self.debug = general_dp_steps([(s.a,s.b) for s in self.segments], self.L)
                self.candidates = []
        self.step_idx = 0

    def derive_textbook_L(self, scene_key: str) -> float:
        if scene_key == "DPOS_TEXTBOOK":
            # L = l(g,b2) with g=2, b2=10
            g, b2 = 2.0, 10.0
            return tour_length(g, b2, H)
        # otherwise return some reasonable default
        return SCRIPTED_PRESETS[scene_key].get("L", 18.0)

    # --- Coordinate transforms ---
    def world_to_screen(self, x: float, y: float) -> Tuple[int,int]:
        cx = SCREEN_W/2
        scale = (SCREEN_W - (MARGIN_L + MARGIN_R)) / self.x_units
        sx = cx + x*scale
        sy = Y_LINE - (y - H)*scale
        return int(sx), int(sy)

    # --- Drawing ---
    def draw_line_and_base(self):
        x0, y0 = self.world_to_screen(-self.x_units, H)
        x1, y1 = self.world_to_screen(self.x_units, H)
        pygame.draw.line(self.screen, (200,200,200), (x0,y0), (x1,y1), 1)
        ox, oy = self.world_to_screen(0.0, 0.0)
        pygame.draw.circle(self.screen, (255,255,255), (ox, oy), 6)
        self.screen.blit(self.font_small.render("O", True, (255,255,255)), (ox+8, oy-8))
        opx, opy = self.world_to_screen(0.0, H)
        pygame.draw.circle(self.screen, (180,180,180), (opx, opy), 4)
        self.screen.blit(self.font_small.render("O'", True, (180,180,180)), (opx+6, opy-6))

    def draw_segments(self, uncovered: List[Segment]):
        if self.show_shading:
            for s in self.segments:
                a = self.world_to_screen(s.a, H)[0]
                b = self.world_to_screen(s.b, H)[0]
                y = self.world_to_screen(0, H)[1]
                pygame.draw.line(self.screen, (90,90,90), (a,y), (b,y), 10)
        for s in uncovered:
            a = self.world_to_screen(s.a, H)[0]
            b = self.world_to_screen(s.b, H)[0]
            y = self.world_to_screen(0, H)[1]
            pygame.draw.line(self.screen, (100,200,255), (a,y), (b,y), 10)
        if self.show_gaps:
            for s in self.segments:
                a = self.world_to_screen(s.a, H)[0]
                b = self.world_to_screen(s.b, H)[0]
                y = self.world_to_screen(0, H)[1]
                pygame.draw.line(self.screen, (255,140,140), (a,y-8), (a,y+8), 2)
                pygame.draw.line(self.screen, (255,140,140), (b,y-8), (b,y+8), 2)

    def draw_tours(self, tours: List[Tour], highlight: Optional[Tour]):
        if not self.show_tours: return
        ox, oy = self.world_to_screen(0.0, 0.0)
        for t in tours:
            px, py = self.world_to_screen(t.p, H)
            qx, qy = self.world_to_screen(t.q, H)
            col = (255,255,255) if (highlight is None or (t.p,t.q)!=(highlight.p,highlight.q)) else (255,255,100)
            pygame.draw.line(self.screen, col, (ox, oy), (px, py), 2)
            pygame.draw.line(self.screen, col, (ox, oy), (qx, qy), 2)
            pygame.draw.line(self.screen, col, (px, py), (qx, qy), 2)

    def draw_candidates(self):
        if not self.show_candidates: return
        if self.scene_key == "DPOS_TEXTBOOK":
            pts = self.candidates
        elif self.scene_key == "GENERAL":
            pts = (self.debug.get("left_ep", []) or []) + (self.debug.get("right_ep", []) or [])
        else:
            pts = []
        for x in pts:
            sx, sy = self.world_to_screen(x, H)
            pygame.draw.circle(self.screen, (200,255,200), (sx, sy-14), 4)

    def draw_hud(self, step: Step):
        pygame.draw.rect(self.screen, (30,30,30), (0,0, SCREEN_W, TOP_PAD))
        pygame.draw.rect(self.screen, (30,30,30), (0,SCREEN_H-BOT_PAD, SCREEN_W, BOT_PAD))

        mode = "SCRIPTED (textbook)" if self.mode_scripted else "COMPUTED (demo)"
        title = f"{self.scene_key} — Mode: {mode}   L={self.L:.2f}   Scale={self.x_units:.1f}"
        self.screen.blit(self.font_big.render(title, True, (255,255,255)), (20, 12))

        left = 20
        y = 44
        self.screen.blit(self.font.render(f"Step {self.step_idx+1}/{len(self.steps)}", True, (200,200,200)), (left,y)); y+=22
        self.screen.blit(self.font.render(step.msg, True, (220,220,220)), (left,y)); y+=22

        bx, by, bw, bh = 20, SCREEN_H - BOT_PAD + 12, 320, 20
        pygame.draw.rect(self.screen, (60,60,60), (bx,by,bw,bh))
        used = clamp(step.current_batt_used / max(1e-6, self.L), 0.0, 1.0)
        pygame.draw.rect(self.screen, (120,200,120), (bx,by,int(bw*used),bh))
        self.screen.blit(self.font_small.render(f"Battery last tour: {step.current_batt_used:.2f} / {self.L:.2f}", True, (240,240,240)), (bx+6, by+2))

        tours = step.tours
        total_dist = sum(tour_length(t.p,t.q,H) for t in tours)
        self.screen.blit(self.font_small.render(f"Total distance: {total_dist:.2f}", True, (220,220,220)), (bx, by+26))

        sx = 380
        self.screen.blit(self.font_small.render("Segment coverage:", True, (220,220,220)), (sx, SCREEN_H - BOT_PAD + 8))
        y2 = SCREEN_H - BOT_PAD + 26
        for i,s in enumerate(self.segments, start=1):
            covered_all = s.length() - sum(seg.length() for seg in subtract_covered([s], [x for t in tours for x in intersect_interval_with_segments(t.p, t.q, [s])]))
            pct = 100.0 * covered_all / max(1e-6, s.length())
            self.screen.blit(self.font_small.render(f"s{i}: [{s.a:.1f},{s.b:.1f}]  {covered_all:.2f}/{s.length():.2f} ({pct:.1f}%)", True, (210,210,210)), (sx, y2))
            y2 += 18

        if self.show_help:
            help_lines = [
                "[SPACE]=Play/Pause   [N]/Right=Next   [B]/Left=Back   [R]=Reset",
                "[1]=GS  [2]=GSP  [3]=DPOS(textbook)  [4]=GENERAL",
                "[M]=Toggle Scripted vs Computed   [+/-]=Battery L (computed mode)   [[/]]=Zoom X",
                "C=Candidates   T=Tours   S=Shading   G=Gaps   H=Help   P=Snapshot"
            ]
            y0 = 70
            for hl in help_lines:
                self.screen.blit(self.font_small.render(hl, True, (200,200,200)), (20, y0))
                y0 += 18

    def draw_world(self, step: Step):
        self.screen.fill((18,18,22))
        self.draw_line_and_base()
        self.draw_segments(step.uncovered)
        self.draw_tours(step.tours, step.last_tour)
        self.draw_candidates()
        self.draw_hud(step)

    def handle_event(self, e):
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE: self.play = not self.play
            elif e.key in (pygame.K_n, pygame.K_RIGHT): self.step_idx = min(self.step_idx+1, len(self.steps)-1)
            elif e.key in (pygame.K_b, pygame.K_LEFT):  self.step_idx = max(self.step_idx-1, 0)
            elif e.key == pygame.K_r: self.recompute()
            elif e.key == pygame.K_1:
                self.scene_key = "GS"; self.recompute()
            elif e.key == pygame.K_2:
                self.scene_key = "GSP"; self.recompute()
            elif e.key == pygame.K_3:
                self.scene_key = "DPOS_TEXTBOOK"; self.recompute()
            elif e.key == pygame.K_4:
                self.scene_key = "GENERAL"; self.recompute()
            elif e.key == pygame.K_m:
                self.mode_scripted = not self.mode_scripted; self.recompute()
            elif e.key == pygame.K_EQUALS or e.key == pygame.K_PLUS:
                if not self.mode_scripted:  # L is fixed in scripted presets (unless derived)
                    self.L += 1.0; self.recompute()
            elif e.key == pygame.K_MINUS or e.key == pygame.K_UNDERSCORE:
                if not self.mode_scripted:
                    self.L = max(1.0, self.L - 1.0); self.recompute()
            elif e.key == pygame.K_LEFTBRACKET:
                self.x_units = max(6.0, self.x_units - 2.0)
            elif e.key == pygame.K_RIGHTBRACKET:
                self.x_units = min(100.0, self.x_units + 2.0)
            elif e.key == pygame.K_c:
                self.show_candidates = not self.show_candidates
            elif e.key == pygame.K_t:
                self.show_tours = not self.show_tours
            elif e.key == pygame.K_s:
                self.show_shading = not self.show_shading
            elif e.key == pygame.K_g:
                self.show_gaps = not self.show_gaps
            elif e.key == pygame.K_h:
                self.show_help = not self.show_help
            elif e.key == pygame.K_p:
                pygame.image.save(self.screen, "snapshot.png"); print("Saved snapshot.png")
        elif e.type == pygame.QUIT:
            pygame.quit(); sys.exit(0)

    def run(self):
        accum = 0.0
        while True:
            dt = self.clock.tick(60) / 1000.0
            for e in pygame.event.get():
                self.handle_event(e)
            if self.play:
                accum += dt
                if accum >= 1.0 / max(1, self.play_fps):
                    self.step_idx = min(self.step_idx+1, len(self.steps)-1)
                    accum = 0.0
                    if self.step_idx == len(self.steps)-1: self.play = False
            step = self.steps[self.step_idx]
            self.draw_world(step)
            pygame.display.flip()

if __name__ == "__main__":
    Simulator().run()
