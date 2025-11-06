import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # type: ignore  # noqa: E402

from coverage_planning.visualization.adapters import build_gs_events
from coverage_planning.visualization.render import PygameRenderer


def test_renderer_process_all_events():
    segments = [(-3.0, -1.0), (1.0, 3.0)]
    events = build_gs_events(segments, h=2.0, L=30.0)

    renderer = PygameRenderer(width=640, height=360, fps=30)
    renderer.load_events(events)
    renderer.process_all_events()

    assert renderer.cursor == renderer.total_events


def test_renderer_perf_budget():
    events = [
        {"type": "set_scene", "h": 2.0, "L": 40.0, "x_min": -1.0, "x_max": 6.0},
        {"type": "add_segment", "id": "seg0", "x1": 0.0, "x2": 5.0},
        {"type": "algo_info", "name": "perf_test", "case": None},
    ]

    tour_id = 0
    for _ in range(3):
        events.append({"type": "start_tour", "tour_id": tour_id, "p": 0.5, "q": 4.5})
        for phase in ("up", "across", "down"):
            for step in range(24):
                progress = (step + 1) / 24
                events.append(
                    {
                        "type": "animate_tour",
                        "tour_id": tour_id,
                        "phase": phase,
                        "progress": progress,
                    }
                )
        events.append({"type": "mark_covered", "x1": 0.5, "x2": 4.5})
        events.append({"type": "end_tour", "tour_id": tour_id})
        tour_id += 1
    events.append({"type": "done"})

    assert len(events) >= 200

    renderer = PygameRenderer(width=640, height=360, fps=30)
    start = time.perf_counter()
    renderer.load_events(events)
    renderer.process_all_events()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5

    pygame.quit()

