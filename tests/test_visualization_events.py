import json
import math

import pytest

from coverage_planning.visualization.adapters import (
    build_dp_full_events,
    build_dpos_events,
    build_gsp_events,
    build_gs_events,
)


def _assert_common_structure(events, expected_segments):
    assert events, "event list should not be empty"
    assert events[0]["type"] == "set_scene"
    seg_events = [ev for ev in events if ev["type"] == "add_segment"]
    assert len(seg_events) == expected_segments
    done_indices = [idx for idx, ev in enumerate(events) if ev["type"] == "done"]
    assert done_indices == [len(events) - 1]


def _assert_animate_progress(events):
    last_progress = {}
    for ev in events:
        if ev["type"] != "animate_tour":
            continue
        key = (ev["tour_id"], ev["phase"])
        progress = ev["progress"]
        assert 0.0 < progress <= 1.0
        prev = last_progress.get(key)
        if prev is not None:
            assert progress >= prev
        last_progress[key] = progress


def _assert_json_and_finite(events):
    def _check(value):
        if isinstance(value, dict):
            for item in value.values():
                _check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _check(item)
        elif isinstance(value, float):
            assert math.isfinite(value)

    for ev in events:
        json.dumps(ev)  # ensure serialisable
        _check(ev)


def test_gs_events_structure():
    segments = [(-4.0, -2.0), (1.0, 2.5)]
    events = build_gs_events(segments, h=2.0, L=30.0)
    _assert_common_structure(events, expected_segments=len(segments))
    algo = next(ev for ev in events if ev["type"] == "algo_info")
    assert algo["name"] == "gs"
    _assert_animate_progress(events)
    _assert_json_and_finite(events)


def test_gsp_events_structure():
    segments = [(-3.0, 3.0)]
    events = build_gsp_events(segments, h=3.0, L=60.0)
    _assert_common_structure(events, expected_segments=len(segments))
    algo = next(ev for ev in events if ev["type"] == "algo_info")
    assert algo["name"] == "gsp"
    assert algo["case"] in {"single", "central", "multi"}
    _assert_animate_progress(events)
    _assert_json_and_finite(events)


def test_dpos_events_structure():
    segments = [(0.0, 2.5), (4.0, 6.0)]
    events = build_dpos_events(segments, h=2.0, L=35.0)
    _assert_common_structure(events, expected_segments=len(segments))

    dp_starts = [ev for ev in events if ev["type"] == "dp_start"]
    assert dp_starts and any(ev["mode"] == "one_side" for ev in dp_starts)

    dp_add = [ev for ev in events if ev["type"] == "dp_add_candidate"]
    assert dp_add, "DP build should emit candidate events"

    _assert_animate_progress(events)
    _assert_json_and_finite(events)


def test_dp_full_events_structure():
    segments = [(-5.0, -3.5), (-1.0, -0.5), (1.5, 2.5), (4.0, 5.5)]
    events = build_dp_full_events(segments, h=2.5, L=45.0)
    _assert_common_structure(events, expected_segments=len(segments))

    dp_starts = [ev for ev in events if ev["type"] == "dp_start"]
    assert dp_starts
    assert any(ev.get("side") == "left" for ev in dp_starts)
    assert any(ev.get("side") == "right" for ev in dp_starts)
    assert any(ev.get("stage") == "tail" for ev in dp_starts if "stage" in ev)

    bridge_events = [ev for ev in events if ev["type"] == "bridge_try"]
    assert bridge_events, "full DP should record bridge attempts"

    _assert_animate_progress(events)
    _assert_json_and_finite(events)

