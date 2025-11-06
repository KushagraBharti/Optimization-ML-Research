from __future__ import annotations

import argparse
import json
from typing import Iterable, List, Tuple

from coverage_planning.visualization.adapters import build_gs_events
from coverage_planning.visualization.render import PygameRenderer

Segment = Tuple[float, float]

PRESETS = {
    "simple": {
        "segments": [(1.0, 3.0)],
        "h": 2.5,
        "L": 40.0,
    },
    "two_segments": {
        "segments": [(-4.0, -1.5), (2.0, 4.5)],
        "h": 3.0,
        "L": 50.0,
    },
    "long_tail": {
        "segments": [(1.0, 2.0), (5.5, 9.0), (12.0, 13.0)],
        "h": 2.0,
        "L": 60.0,
    },
}


def parse_segments(seg_string: str) -> List[Segment]:
    data = json.loads(seg_string)
    segments = []
    for entry in data:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("segments must be [[x1, x2], ...]")
        x1, x2 = float(entry[0]), float(entry[1])
        segments.append((x1, x2))
    return segments


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Greedy MinTours visualization demo")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="simple")
    parser.add_argument("--segments", type=str, help="JSON list of [x1, x2] pairs")
    parser.add_argument("--h", type=float, help="Override height h")
    parser.add_argument("--L", type=float, help="Override budget L")
    parser.add_argument("--manual", action="store_true", help="Start with autoplay disabled")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.segments:
        segments = parse_segments(args.segments)
        preset = {"segments": segments, "h": args.h or 2.0, "L": args.L or 40.0}
    else:
        preset = dict(PRESETS[args.preset])
        segments = preset["segments"]
        if args.h is not None:
            preset["h"] = args.h
        if args.L is not None:
            preset["L"] = args.L

    events = build_gs_events(segments, preset["h"], preset["L"])
    renderer = PygameRenderer()
    renderer.load_events(events)
    renderer.run(autoplay=not args.manual)


if __name__ == "__main__":
    main()

