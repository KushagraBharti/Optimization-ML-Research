from __future__ import annotations

import argparse
import json
from typing import Iterable, Tuple

from coverage_planning.visualization.adapters import build_gsp_events
from coverage_planning.visualization.render import PygameRenderer

Segment = Tuple[float, float]

PRESETS = {
    "one_sided": {
        "segment": (2.0, 8.0),
        "h": 2.5,
        "L": 45.0,
    },
    "central_finish": {
        "segment": (-6.0, 6.0),
        "h": 3.0,
        "L": 48.0,
    },
    "asymmetric": {
        "segment": (-3.0, 9.0),
        "h": 2.5,
        "L": 40.0,
    },
}


def parse_segment(value: str) -> Segment:
    data = json.loads(value)
    if not isinstance(data, (list, tuple)) or len(data) != 2:
        raise ValueError("segment must be [x1, x2]")
    return float(data[0]), float(data[1])


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GSP single-segment visualization demo")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="one_sided")
    parser.add_argument("--segment", type=str, help="JSON [x1, x2]")
    parser.add_argument("--h", type=float)
    parser.add_argument("--L", type=float)
    parser.add_argument("--manual", action="store_true", help="Start with autoplay disabled")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.segment:
        segment = parse_segment(args.segment)
        h = args.h or 2.5
        L = args.L or 40.0
    else:
        preset = dict(PRESETS[args.preset])
        segment = preset["segment"]
        h = args.h if args.h is not None else preset["h"]
        L = args.L if args.L is not None else preset["L"]

    events = build_gsp_events([segment], h, L)
    renderer = PygameRenderer()
    renderer.load_events(events)
    renderer.run(autoplay=not args.manual)


if __name__ == "__main__":
    main()

