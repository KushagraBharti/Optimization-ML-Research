from __future__ import annotations

import argparse
import json
from typing import Iterable, List, Tuple

from coverage_planning.visualization.adapters import build_dp_full_events
from coverage_planning.visualization.render import PygameRenderer

Segment = Tuple[float, float]

PRESETS = {
    "balanced": {
        "segments": [(-5.0, -3.0), (-1.0, -0.4), (1.0, 2.0), (4.0, 6.0)],
        "h": 2.5,
        "L": 45.0,
    },
    "bridge_heavy": {
        "segments": [(-6.0, -4.5), (-2.0, -1.3), (1.2, 2.4), (3.5, 6.0)],
        "h": 3.0,
        "L": 55.0,
    },
    "asymmetric": {
        "segments": [(-7.0, -5.5), (-2.5, -1.5), (2.0, 3.5), (6.0, 8.0)],
        "h": 2.8,
        "L": 60.0,
    },
}


def parse_segments(value: str) -> List[Segment]:
    data = json.loads(value)
    segments = []
    for entry in data:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("segments must be [[x1, x2], ...]")
        x1, x2 = float(entry[0]), float(entry[1])
        segments.append((x1, x2))
    return segments


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Full-line DP visualization demo")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="bridge_heavy")
    parser.add_argument("--segments", type=str, help="JSON list of [x1, x2]")
    parser.add_argument("--h", type=float)
    parser.add_argument("--L", type=float)
    parser.add_argument("--manual", action="store_true", help="Start with autoplay disabled")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.segments:
        segments = parse_segments(args.segments)
        h = args.h or 2.5
        L = args.L or 50.0
    else:
        preset = dict(PRESETS[args.preset])
        segments = preset["segments"]
        h = args.h if args.h is not None else preset["h"]
        L = args.L if args.L is not None else preset["L"]

    events = build_dp_full_events(segments, h, L)
    renderer = PygameRenderer()
    renderer.load_events(events)
    renderer.run(autoplay=not args.manual)


if __name__ == "__main__":
    main()

