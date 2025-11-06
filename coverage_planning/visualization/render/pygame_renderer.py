"""Pygame renderer that consumes visualization events."""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import pygame
except ImportError as exc:  # pragma: no cover - ensure pygame is available
    raise ImportError("pygame is required for the visualization renderer") from exc

from coverage_planning.visualization.render.base_scene import (
    BRIDGE_COLOR,
    CandidateMarker,
    DP_CANDIDATE_COLOR,
    DP_PICK_COLOR,
    DP_TRY_COLOR,
    TransitionHighlight,
    BaseScene,
)


class PygameRenderer:
    """Render coverage-planning event streams."""

    SPEED_LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0]
    EVENT_INTERVAL = 1.0  # seconds per event at speed multiplier 1.0 (scaled by FPS)

    def __init__(self, width: int = 1200, height: int = 500, fps: int = 60) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.scene = BaseScene(width, height)
        self.events: List[Dict[str, object]] = []
        self.cursor = 0
        self.total_events = 0
        self.autoplay = True
        self.speed_index = 1
        self.autoplay_accumulator = 0.0
        self.clock: Optional["pygame.time.Clock"] = None
        self.screen: Optional["pygame.Surface"] = None
        self.running = False
        self.algorithm_name = "unknown"
        self.algorithm_case: Optional[str] = None
        self.h_value = 0.0
        self.L_value = 0.0
        self.current_tour_id: Optional[int] = None
        self.current_p: float = 0.0
        self.current_q: float = 0.0
        self.current_phase: Optional[str] = None
        self.current_progress: float = 0.0
        self.drone_position: Optional[Tuple[float, float]] = None
        self.dp_context_key: Optional[str] = None
        self.dp_contexts: Dict[str, Dict[str, object]] = {}
        self.dp_highlights: List[Dict[str, object]] = []
        self.bridge_highlights: List[Dict[str, object]] = []
        self.scene_initialized = False
        self.completed = False
        self.last_event_type: Optional[str] = None

    # ------------------------------------------------------------------ public API
    def load_events(self, events: List[Dict[str, object]]) -> None:
        self.events = list(events)
        self.total_events = len(self.events)
        self.cursor = 0
        self.autoplay_accumulator = 0.0
        self.scene.reset()
        self.scene_initialized = False
        self.algorithm_name = "unknown"
        self.algorithm_case = None
        self.current_tour_id = None
        self.current_phase = None
        self.current_progress = 0.0
        self.drone_position = None
        self.dp_contexts.clear()
        self.dp_context_key = None
        self.dp_highlights.clear()
        self.bridge_highlights.clear()
        self.completed = False
        self.last_event_type = None
        if self.events:
            self._bootstrap_scene()

    def run(self, autoplay: bool = True) -> None:
        if not self.events:
            raise RuntimeError("No events loaded. Call load_events() first.")

        self.autoplay = autoplay
        if not self.scene_initialized:
            self._bootstrap_scene()

        pygame.display.init()
        pygame.display.set_caption("Coverage Planning Visualization")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running = True

        while self.running:
            dt = self.clock.tick(self.fps) / 1000.0
            self._handle_input()
            self._update_highlights(dt)

            if self.autoplay and not self.completed:
                self._autoplay_advance(dt)

            self._draw_frame()

        pygame.display.quit()

    def process_all_events(self) -> None:
        """Advance through all events without opening a window (testing helper)."""
        while self.cursor < self.total_events:
            self._advance_event()

    def step_once(self) -> None:
        """Advance a single event."""
        self._advance_event()

    # ------------------------------------------------------------------ internals
    def _bootstrap_scene(self) -> None:
        """Consume initial scene setup events (scene + segments + algo info)."""
        while self.cursor < self.total_events:
            event_type = self.events[self.cursor].get("type")
            if event_type in {"set_scene", "add_segment", "algo_info"}:
                self._advance_event()
                self.scene_initialized = True
                continue
            break
        if not self.scene_initialized and self.cursor < self.total_events:
            # ensure at least set_scene processed
            self._advance_event()
            self.scene_initialized = True

    def _handle_input(self) -> None:
        for py_event in pygame.event.get():
            if py_event.type == pygame.QUIT:
                self.running = False
                return
            if py_event.type == pygame.KEYDOWN:
                if py_event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                    return
                if py_event.key == pygame.K_SPACE:
                    self.autoplay = not self.autoplay
                elif py_event.key == pygame.K_RIGHT:
                    self.autoplay = False
                    self._advance_event()
                elif py_event.key == pygame.K_LEFT:
                    self.autoplay = False
                    self._rewind_event()
                elif py_event.key == pygame.K_UP:
                    self.speed_index = min(len(self.SPEED_LEVELS) - 1, self.speed_index + 1)
                elif py_event.key == pygame.K_DOWN:
                    self.speed_index = max(0, self.speed_index - 1)
                elif py_event.key == pygame.K_r:
                    self._reset_playback()

    def _reset_playback(self) -> None:
        current_autoplay = self.autoplay
        self.load_events(self.events)
        self.autoplay = current_autoplay

    def _autoplay_advance(self, dt: float) -> None:
        interval = 1.0 / self.fps
        interval /= max(0.1, self.SPEED_LEVELS[self.speed_index])
        self.autoplay_accumulator += dt
        while self.autoplay_accumulator >= interval and self.cursor < self.total_events:
            self.autoplay_accumulator -= interval
            self._advance_event()

    def _advance_event(self) -> None:
        if self.cursor >= self.total_events:
            self.completed = True
            return
        event = self.events[self.cursor]
        self.cursor += 1
        self.last_event_type = event.get("type")
        self._apply_event(event)
        if self.cursor >= self.total_events:
            self.completed = True

    def _rewind_event(self) -> None:
        if self.cursor == 0:
            return
        target = self.cursor - 1
        events_copy = list(self.events)
        self.load_events(events_copy)
        while self.cursor < target:
            self._advance_event()

    # ------------------------------------------------------------------ event application
    def _apply_event(self, event: Dict[str, object]) -> None:
        event_type = event.get("type")
        if event_type == "set_scene":
            self.h_value = float(event.get("h", 0.0))
            self.L_value = float(event.get("L", 0.0))
            x_min = float(event.get("x_min", -10.0))
            x_max = float(event.get("x_max", 10.0))
            self.scene.set_scene(h=self.h_value, L=self.L_value, x_min=x_min, x_max=x_max)
            self.scene_initialized = True
        elif event_type == "add_segment":
            seg_id = str(event.get("id"))
            x1 = float(event.get("x1", 0.0))
            x2 = float(event.get("x2", 0.0))
            self.scene.add_segment(seg_id, x1, x2)
        elif event_type == "algo_info":
            self.algorithm_name = str(event.get("name", "unknown"))
            case = event.get("case")
            self.algorithm_case = str(case) if case is not None else None
        elif event_type == "start_tour":
            self.current_tour_id = int(event.get("tour_id", 0))
            self.current_p = float(event.get("p", 0.0))
            self.current_q = float(event.get("q", 0.0))
            self.current_phase = "up"
            self.current_progress = 0.0
            self.drone_position = self.scene.interpolate_path(self.current_p, self.current_q, "up", 0.0)
        elif event_type == "animate_tour":
            if self.current_tour_id is None:
                return
            phase = str(event.get("phase", "up"))
            progress = float(event.get("progress", 0.0))
            self.current_phase = phase
            self.current_progress = progress
            self.drone_position = self.scene.interpolate_path(self.current_p, self.current_q, phase, progress)
        elif event_type == "mark_covered":
            x1 = float(event.get("x1", 0.0))
            x2 = float(event.get("x2", 0.0))
            self.scene.mark_covered(x1, x2)
        elif event_type == "end_tour":
            self.drone_position = None
            self.current_phase = None
            self.current_progress = 0.0
            self.current_tour_id = None
        elif event_type == "dp_start":
            key = self._make_dp_key(event)
            self.dp_context_key = key
            self.dp_contexts[key] = {
                "mode": event.get("mode"),
                "side": event.get("side"),
                "stage": event.get("stage"),
                "candidates": {},
            }
        elif event_type == "dp_add_candidate":
            if self.dp_context_key is None:
                return
            idx = int(event.get("idx", 0))
            x = float(event.get("x", 0.0))
            ctx = self.dp_contexts[self.dp_context_key]
            ctx["candidates"][idx] = x
        elif event_type == "dp_try_transition":
            self._add_dp_highlight(event, DP_TRY_COLOR, ttl=0.4, width=2)
        elif event_type == "dp_pick_transition":
            self._add_dp_highlight(event, DP_PICK_COLOR, ttl=0.8, width=3)
        elif event_type == "dp_done":
            self.dp_context_key = None
        elif event_type == "bridge_try":
            self._add_bridge_highlight(event)
        elif event_type == "replay_start":
            self.drone_position = None
        elif event_type == "done":
            self.completed = True
        else:
            print(f"[Renderer] Unhandled event type: {event_type}")

    def _make_dp_key(self, event: Dict[str, object]) -> str:
        stage = event.get("stage")
        side = event.get("side")
        mode = event.get("mode")
        if stage:
            return f"{mode}:{stage}"
        if side:
            return f"{mode}:{side}"
        return str(mode)

    def _add_dp_highlight(self, event: Dict[str, object], color: Tuple[int, int, int], ttl: float, width: int) -> None:
        if self.dp_context_key is None:
            return
        ctx = self.dp_contexts.get(self.dp_context_key)
        if not ctx:
            return
        candidates = ctx["candidates"]
        from_idx = event.get("from")
        to_idx = event.get("to")
        if to_idx is None:
            return
        to_idx = int(to_idx)
        to_x = candidates.get(to_idx)
        if to_x is None:
            return
        start_x = to_x
        if from_idx is not None:
            from_idx = int(from_idx)
            start_x = candidates.get(from_idx, to_x)
        self.dp_highlights.append(
            {
                "start": float(start_x),
                "end": float(to_x),
                "color": color,
                "ttl": ttl,
                "width": width,
            }
        )

    def _add_bridge_highlight(self, event: Dict[str, object]) -> None:
        P = float(event.get("P", 0.0))
        q = float(event.get("q", 0.0))
        accepted = bool(event.get("accepted", False))
        color = BRIDGE_COLOR if accepted else (160, 120, 0)
        ttl = 1.0 if accepted else 0.5
        self.bridge_highlights.append(
            {
                "P": P,
                "q": q,
                "color": color,
                "ttl": ttl,
            }
        )

    def _update_highlights(self, dt: float) -> None:
        def _decay(container: List[Dict[str, object]]) -> None:
            remove: List[Dict[str, object]] = []
            for item in container:
                item["ttl"] = float(item.get("ttl", 0.0)) - dt
                if item["ttl"] <= 0.0:
                    remove.append(item)
            for item in remove:
                container.remove(item)

        _decay(self.dp_highlights)
        _decay(self.bridge_highlights)

    # ------------------------------------------------------------------ drawing
    def _collect_candidate_markers(self) -> List[CandidateMarker]:
        markers: List[CandidateMarker] = []
        for ctx in self.dp_contexts.values():
            for x in ctx["candidates"].values():
                markers.append(CandidateMarker(x=float(x), color=DP_CANDIDATE_COLOR))
        return markers

    def _collect_transition_highlights(self) -> List[TransitionHighlight]:
        highlights: List[TransitionHighlight] = []
        for item in self.dp_highlights:
            highlights.append(
                TransitionHighlight(
                    start_x=float(item["start"]),
                    end_x=float(item["end"]),
                    color=item["color"],
                    width=int(item["width"]),
                )
            )
        return highlights

    def _draw_frame(self) -> None:
        assert self.screen is not None
        self.scene.draw_background(self.screen)
        self.scene.draw_axes(self.screen)
        self.scene.draw_segments(self.screen)

        for bridge in self.bridge_highlights:
            self.scene.draw_bridge(self.screen, bridge["P"], bridge["q"], color=bridge["color"])

        self.scene.draw_candidates(self.screen, self._collect_candidate_markers())
        self.scene.draw_highlights(self.screen, self._collect_transition_highlights())

        if self.current_tour_id is not None:
            self.scene.draw_path_preview(self.screen, self.current_p, self.current_q)

        self.scene.draw_drone(self.screen, self.drone_position)
        self._draw_hud(self.screen)
        pygame.display.flip()

    def _draw_hud(self, surface: "pygame.Surface") -> None:
        lines = [
            f"Algo: {self.algorithm_name}",
            f"Case: {self.algorithm_case or 'n/a'}",
            f"Event: {self.cursor}/{self.total_events}",
            f"Autoplay: {'on' if self.autoplay else 'off'} x{self.SPEED_LEVELS[self.speed_index]:.1f}",
            f"h={self.h_value:.2f} L={self.L_value:.2f}",
        ]
        if self.current_tour_id is not None and self.current_phase is not None:
            lines.append(
                f"Tour {self.current_tour_id} phase={self.current_phase} prog={self.current_progress:.2f}"
            )
        if self.last_event_type:
            lines.append(f"Last: {self.last_event_type}")

        x = 10
        y = 10
        for line in lines:
            text_surface = self.font.render(line, True, (230, 230, 230))
            surface.blit(text_surface, (x, y))
            y += text_surface.get_height() + 2


__all__ = ["PygameRenderer"]

