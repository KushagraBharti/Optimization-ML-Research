# Visualization Subsystem

This package provides an instrumented visualization stack for the one-dimensional coverage planning algorithms.  Algorithms emit structured event streams (via the adapters in `coverage_planning.visualization.adapters`) and a reusable pygame renderer consumes those events to produce interactive animations.

## Installation

Install the repository in editable mode with the `dev` extras (see the root README).  The renderer relies on [`pygame`](https://www.pygame.org/); the dependency is included in the project extras.

```bash
pip install -e .[dev]
```

## Controls

| Key           | Action                                |
| ------------- | ------------------------------------- |
| `SPACE`       | Toggle autoplay on/off                |
| `RIGHT`       | Step forward one event                |
| `LEFT`        | Step backward (rebuilds state)        |
| `UP` / `DOWN` | Increase / decrease playback speed    |
| `R`           | Reset to the first event              |
| `ESC`, `Q`    | Quit                                  |

HUD text shows the algorithm, variant/case (when supplied by the adapter), current event index, playback speed, and the current tour phase/progress while the drone is active.

## Running the demos

Each algorithm ships with a small CLI wrapper that prepares events and hands them to the pygame renderer.  They are runnable with `python -m …`:

```bash
# Greedy MinTours (GS)
python -m coverage_planning.visualization.demo_gs --preset simple

# GSP (single segment minimum-length)
python -m coverage_planning.visualization.demo_gsp --preset central_finish

# DPOS (one-sided DP)
python -m coverage_planning.visualization.demo_dpos --preset gap_case

# Full-line DP
python -m coverage_planning.visualization.demo_dp_full --preset bridge_heavy
```

Each demo accepts `--manual` to start with autoplay disabled, plus optional overrides for geometry (`--segments`, `--segment`, `--h`, `--L`).  Segment overrides use JSON literals, e.g.:

```bash
python -m coverage_planning.visualization.demo_gs \
  --segments "[[-4, -2], [1, 3.5]]" --h 2.5 --L 55
```

## Architecture Overview

1. **Instrumented solvers** (under `coverage_planning.visualization.algs`) run the paper-faithful algorithms and emit structured traces.
2. **Adapters** convert solver traces into an event stream with a small shared schema (`coverage_planning.visualization.events`).
3. **Renderer** (`coverage_planning.visualization.render.PygameRenderer`) consumes the events and animates the scene.  It does not call algorithm code directly.

The renderer is stateless between runs.  Load a fresh event list with `load_events(events)` and call `run()`.  For testing or scripting, `process_all_events()` advances the state without opening a window.

## Known Limitations

- The renderer focuses on clarity over photorealistic assets.  DP annotations are drawn as simple ticks and transient highlights.
- Reverse stepping (`LEFT`) rebuilds state from the start for simplicity; large event streams may incur a short pause.
- Running the renderer in a headless environment requires setting `SDL_VIDEODRIVER=dummy`.  The automated tests demonstrate this configuration.

