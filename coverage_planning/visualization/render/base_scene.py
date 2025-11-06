"""Utility primitives for pygame visualization scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pygame
except ImportError as exc:  # pragma: no cover - pygame should be installed by demos
    raise ImportError("pygame is required for the visualization renderer") from exc

Segment = Tuple[float, float]

BACKGROUND_COLOR = (18, 18, 24)
AXIS_COLOR = (120, 120, 140)
HEIGHT_COLOR = (80, 150, 200)
SEGMENT_UNCOVERED_COLOR = (200, 70, 70)
SEGMENT_COVERED_COLOR = (70, 200, 110)
DP_CANDIDATE_COLOR = (200, 200, 80)
DP_TRY_COLOR = (200, 120, 40)
DP_PICK_COLOR = (90, 200, 255)
BRIDGE_COLOR = (255, 215, 0)
DRONE_COLOR = (255, 255, 255)
DRONE_PATH_COLOR = (120, 200, 255)

GROUND_RATIO = 0.75
VERTICAL_FRACTION = 0.6
HORIZONTAL_MARGIN_RATIO = 0.05


@dataclass
class SegmentState:
    seg_id: str
    x1: float
    x2: float


@dataclass
class CandidateMarker:
    x: float
    color: Tuple[int, int, int]
    size: int = 6


@dataclass
class TransitionHighlight:
    start_x: float
    end_x: float
    color: Tuple[int, int, int]
    width: int


def _merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    sorted_intervals = sorted((min(a, b), max(a, b)) for a, b in intervals)
    merged: List[Tuple[float, float]] = [sorted_intervals[0]]
    for left, right in sorted_intervals[1:]:
        prev_left, prev_right = merged[-1]
        if left <= prev_right:
            merged[-1] = (prev_left, max(prev_right, right))
        else:
            merged.append((left, right))
    return merged


class BaseScene:
    """Coordinate transforms and basic draw helpers for the pygame renderer."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.h = 1.0
        self.L = 1.0
        self.x_min = -10.0
        self.x_max = 10.0
        self.margin_x = int(self.width * HORIZONTAL_MARGIN_RATIO)
        self.scene_width = self.width - 2 * self.margin_x
        self.floor_y = int(self.height * GROUND_RATIO)
        self.h_scale = 1.0
        self.x_scale = 1.0
        self.segments: Dict[str, SegmentState] = {}
        self.coverage_intervals: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------ transforms
    def set_scene(self, *, h: float, L: float, x_min: float, x_max: float) -> None:
        self.h = max(h, 1e-6)
        self.L = L
        if x_min >= x_max:
            x_max = x_min + 1.0
        self.x_min = x_min
        self.x_max = x_max
        self.scene_width = self.width - 2 * self.margin_x
        self.x_scale = self.scene_width / (self.x_max - self.x_min)
        vertical_span = self.height * VERTICAL_FRACTION
        self.h_scale = vertical_span / self.h

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        px = self.margin_x + int((x - self.x_min) * self.x_scale)
        py = self.floor_y - int(y * self.h_scale)
        px = max(0, min(self.width - 1, px))
        py = max(0, min(self.height - 1, py))
        return px, py

    def screen_y_for_ground(self) -> int:
        return self.floor_y

    def screen_y_for_height(self) -> int:
        return self.world_to_screen(0.0, self.h)[1]

    # ------------------------------------------------------------------ scene content
    def add_segment(self, seg_id: str, x1: float, x2: float) -> None:
        self.segments[seg_id] = SegmentState(seg_id=seg_id, x1=float(x1), x2=float(x2))

    def mark_covered(self, x1: float, x2: float) -> None:
        self.coverage_intervals.append((float(x1), float(x2)))
        self.coverage_intervals = _merge_intervals(self.coverage_intervals)

    def reset(self) -> None:
        self.segments.clear()
        self.coverage_intervals.clear()

    # ------------------------------------------------------------------ drawing helpers
    def draw_background(self, surface: "pygame.Surface") -> None:
        surface.fill(BACKGROUND_COLOR)

    def draw_axes(self, surface: "pygame.Surface") -> None:
        ground_y = self.screen_y_for_ground()
        height_y = self.screen_y_for_height()
        pygame.draw.line(surface, AXIS_COLOR, (0, ground_y), (self.width, ground_y), 2)
        pygame.draw.line(surface, HEIGHT_COLOR, (0, height_y), (self.width, height_y), 1)

    def draw_segments(self, surface: "pygame.Surface") -> None:
        ground_y = self.screen_y_for_ground()
        for seg in self.segments.values():
            x1, x2 = seg.x1, seg.x2
            sx1, sy1 = self.world_to_screen(x1, 0.0)
            sx2, sy2 = self.world_to_screen(x2, 0.0)
            pygame.draw.line(surface, SEGMENT_UNCOVERED_COLOR, (sx1, sy1), (sx2, sy2), 6)

        for covered_left, covered_right in self.coverage_intervals:
            sx1, sy1 = self.world_to_screen(covered_left, 0.0)
            sx2, sy2 = self.world_to_screen(covered_right, 0.0)
            pygame.draw.line(surface, SEGMENT_COVERED_COLOR, (sx1, sy1), (sx2, sy2), 6)

    def draw_candidates(
        self,
        surface: "pygame.Surface",
        markers: Iterable[CandidateMarker],
    ) -> None:
        for marker in markers:
            sx, sy = self.world_to_screen(marker.x, 0.0)
            pygame.draw.circle(surface, marker.color, (sx, sy), marker.size // 2)

    def draw_highlights(
        self,
        surface: "pygame.Surface",
        highlights: Iterable[TransitionHighlight],
        y_offset: float = 0.25,
    ) -> None:
        """
        Draw transient highlight lines slightly above ground level.

        Parameters
        ----------
        y_offset:
            Vertical offset as a fraction of h to avoid overlapping with the base axis.
        """
        highlight_y = max(0.0, min(self.h, self.h * y_offset))
        screen_y = self.world_to_screen(0.0, highlight_y)[1]
        for highlight in highlights:
            sx1, _ = self.world_to_screen(highlight.start_x, highlight_y)
            sx2, _ = self.world_to_screen(highlight.end_x, highlight_y)
            pygame.draw.line(
                surface,
                highlight.color,
                (sx1, screen_y),
                (sx2, screen_y),
                highlight.width,
            )

    def draw_bridge(
        self,
        surface: "pygame.Surface",
        P: float,
        q: float,
        color: Tuple[int, int, int] = BRIDGE_COLOR,
        width: int = 2,
    ) -> None:
        sx1, sy1 = self.world_to_screen(P, self.h)
        sx2, sy2 = self.world_to_screen(q, self.h)
        pygame.draw.line(surface, color, (sx1, sy1), (sx2, sy2), width)

    def draw_drone(
        self,
        surface: "pygame.Surface",
        position: Optional[Tuple[float, float]],
    ) -> None:
        if position is None:
            return
        sx, sy = self.world_to_screen(position[0], position[1])
        pygame.draw.circle(surface, DRONE_COLOR, (sx, sy), 8)

    def draw_path_preview(
        self,
        surface: "pygame.Surface",
        p: float,
        q: float,
    ) -> None:
        points = [
            self.world_to_screen(0.0, 0.0),
            self.world_to_screen(p, self.h),
            self.world_to_screen(q, self.h),
            self.world_to_screen(0.0, 0.0),
        ]
        pygame.draw.lines(surface, DRONE_PATH_COLOR, False, points, 1)

    # ------------------------------------------------------------------ geometry helpers
    def interpolate_path(
        self,
        p: float,
        q: float,
        phase: str,
        progress: float,
    ) -> Tuple[float, float]:
        progress = max(0.0, min(1.0, progress))
        if phase == "up":
            x = 0.0 + (p - 0.0) * progress
            y = 0.0 + self.h * progress
        elif phase == "across":
            x = p + (q - p) * progress
            y = self.h
        elif phase == "down":
            x = q + (0.0 - q) * progress
            y = self.h * (1.0 - progress)
        else:
            x = 0.0
            y = 0.0
        return float(x), float(y)


__all__ = [
    "BaseScene",
    "SegmentState",
    "CandidateMarker",
    "TransitionHighlight",
    "BACKGROUND_COLOR",
    "AXIS_COLOR",
    "HEIGHT_COLOR",
    "SEGMENT_UNCOVERED_COLOR",
    "SEGMENT_COVERED_COLOR",
    "DP_CANDIDATE_COLOR",
    "DP_TRY_COLOR",
    "DP_PICK_COLOR",
    "BRIDGE_COLOR",
    "DRONE_COLOR",
    "DRONE_PATH_COLOR",
]

