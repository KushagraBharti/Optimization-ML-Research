from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple, cast

import numpy as np

from coverage_planning.algs.geometry import EPS as GEOM_EPSILON, tour_length
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
from coverage_planning.data.schemas import Instance

FamilyName = Literal["uniform", "clustered", "step_gap", "straddlers"]
SideChoice = Literal["one_sided", "two_sided", "straddling"]

DEFAULT_MAX_ATTEMPTS = 256


@dataclass(frozen=True)
class FamilyConfig:
    """Configuration bundle for ``draw_family``.

    Attributes mirror the dataset design requirements. Values are intentionally
    conservative so that callers can override only the pieces they care about.
    """

    # Segment counts ---------------------------------------------------------
    n: int | None = None
    n_range: Tuple[int, int] = (2, 200)
    n_extrap_range: Tuple[int, int] = (300, 500)
    use_extrapolation: bool = False

    # Geometry ---------------------------------------------------------------
    min_gap: float = 0.5
    min_len: float = 1.0
    max_len: float = 40.0
    h_range: Tuple[float, float] = (5.0, 40.0)
    x_extent: Tuple[float, float] = (50.0, 200.0)

    # Family-specific knobs --------------------------------------------------
    straddle_fraction: Tuple[float, float] = (0.15, 0.35)
    cluster_count_range: Tuple[int, int] = (2, 4)
    cluster_spread_ratio: Tuple[float, float] = (0.05, 0.2)
    step_gap_small_factor: Tuple[float, float] = (1.0, 2.5)
    step_gap_large_factor: Tuple[float, float] = (4.0, 8.0)
    step_length_short_factor: Tuple[float, float] = (1.0, 2.5)
    step_length_long_factor: Tuple[float, float] = (4.0, 8.0)

    # Battery regimes --------------------------------------------------------
    L_mode: Literal["tight", "roomy", "mixed"] = "mixed"
    tight_delta_fraction: Tuple[float, float] = (0.05, 0.15)
    roomy_multiplier_range: Tuple[float, float] = (1.3, 3.0)
    tight_probability: float = 0.5
    margin_multiple: float = 12.0  # multiplied by TOL_NUM

    # Side regimes -----------------------------------------------------------
    side_mix: Tuple[float, float, float] = (0.4, 0.4, 0.2)  # (one, two, straddling)

    # Misc -------------------------------------------------------------------
    max_attempts: int = DEFAULT_MAX_ATTEMPTS


def validate_instance(segments: Sequence[Tuple[float, float]]) -> None:
    """Validate that ``segments`` are strictly increasing and disjoint.

    Parameters
    ----------
    segments:
        Sequence of (a, b) pairs representing closed intervals.

    Raises
    ------
    ValueError
        If any segment has non-positive measure or two segments overlap.
    """

    if not segments:
        raise ValueError("instance must contain at least one segment")

    prev_b = None
    for idx, (a, b) in enumerate(sorted(segments, key=lambda seg: seg[0])):
        if not a < b - EPS_GEOM:
            raise ValueError(f"segment #{idx} must satisfy a < b; received ({a}, {b})")
        if prev_b is not None and not (prev_b + EPS_GEOM <= a):
            raise ValueError(
                f"segments #{idx-1} and #{idx} overlap or touch within tolerance"
            )
        prev_b = b


def draw_instance_one_sided(
    n: int,
    h: float,
    L: float,
    *,
    min_gap: float,
    min_len: float,
    max_len: float,
    rng: np.random.Generator,
    margin: float | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> List[Tuple[float, float]]:
    """Generate a right-side instance with strictly positive gaps.

    The generator samples log-uniform lengths, places them sequentially with
    gaps sampled from a mild range, and then recenters the block near the
    origin. ``L`` is used only to enforce feasibility when finite.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> segs = draw_instance_one_sided(
    ...     5, h=10.0, L=120.0, min_gap=0.5, min_len=1.0, max_len=10.0, rng=rng
    ... )
    >>> len(segs)
    5
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if min_len <= 0 or max_len <= 0 or max_len < min_len:
        raise ValueError("segment length bounds must satisfy 0 < min_len ≤ max_len")
    if min_gap <= 0:
        raise ValueError("min_gap must be positive")

    attempt_margin = margin if margin is not None else 10.0 * TOL_NUM
    attempt_margin = max(attempt_margin, 5.0 * TOL_NUM)

    len_log_low = math.log(min_len)
    len_log_high = math.log(max_len)

    for _ in range(max_attempts):
        lengths = np.exp(rng.uniform(len_log_low, len_log_high, size=n))
        lengths = np.clip(lengths, min_len, max_len)
        segments: List[Tuple[float, float]] = []
        cursor = 0.0
        for length in lengths:
            gap = rng.uniform(min_gap, 4.0 * min_gap)
            start = cursor + gap
            end = start + float(length)
            segments.append((float(start), float(end)))
            cursor = end
        # Recenter near the origin so that the first segment starts close to 0.
        shift = rng.uniform(-segments[0][0], min_gap)
        segments = [(a + shift, b + shift) for a, b in segments]

        if not math.isfinite(L) or _per_segment_feasible(segments, h, L, attempt_margin):
            try:
                validate_instance(segments)
            except ValueError:
                continue
            if not math.isfinite(L) or _gs_reach_feasible(segments, h, L, attempt_margin):
                return segments
    raise RuntimeError("failed to generate one-sided instance within attempts budget")


def draw_instance_two_sided(
    n_left: int,
    n_right: int,
    h: float,
    L: float,
    *,
    min_gap: float,
    min_len: float,
    max_len: float,
    rng: np.random.Generator,
    margin: float | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> List[Tuple[float, float]]:
    """Generate a two-sided instance free of straddling segments."""

    if n_left <= 0 or n_right <= 0:
        raise ValueError("both n_left and n_right must be positive")

    attempt_margin = margin if margin is not None else 10.0 * TOL_NUM

    for _ in range(max_attempts):
        pos = draw_instance_one_sided(
            n_right,
            h,
            L,
            min_gap=min_gap,
            min_len=min_len,
            max_len=max_len,
            rng=rng,
            margin=attempt_margin,
            max_attempts=max_attempts,
        )
        left_pos = draw_instance_one_sided(
            n_left,
            h,
            L,
            min_gap=min_gap,
            min_len=min_len,
            max_len=max_len,
            rng=rng,
            margin=attempt_margin,
            max_attempts=max_attempts,
        )
        # Reflect left-hand segments and ensure a padding around zero.
        left = [(-b, -a) for a, b in left_pos][::-1]
        max_left_end = max(b for _, b in left)
        min_right_start = min(a for a, _ in pos)
        if max_left_end > -min_gap:
            shift = -(max_left_end + min_gap)
            left = [(a + shift, b + shift) for a, b in left]
        if min_right_start < min_gap:
            shift = min_gap - min_right_start
            pos = [(a + shift, b + shift) for a, b in pos]

        combined = sorted(left + pos, key=lambda seg: seg[0])
        try:
            validate_instance(combined)
        except ValueError:
            continue
        if not math.isfinite(L) or (
            _per_segment_feasible(combined, h, L, attempt_margin)
            and _gs_reach_feasible(combined, h, L, attempt_margin)
        ):
            return combined

    raise RuntimeError("failed to generate two-sided instance within attempts budget")


def draw_instance_straddling(
    n: int,
    h: float,
    L: float,
    *,
    min_gap: float,
    min_len: float,
    max_len: float,
    rng: np.random.Generator,
    margin: float | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> List[Tuple[float, float]]:
    """Generate segments that include a bridge (straddling) component."""

    if n < 2:
        raise ValueError("straddling instances require at least two segments")

    attempt_margin = margin if margin is not None else 10.0 * TOL_NUM

    for _ in range(max_attempts):
        straddle_len = float(
            np.clip(
                math.exp(rng.uniform(math.log(min_len), math.log(max_len))),
                min_len,
                max_len,
            )
        )
        # Guarantee the straddler crosses the origin with a little wiggle room.
        left_extent = rng.uniform(min_gap, 3.0 * min_gap)
        right_extent = rng.uniform(min_gap, 3.0 * min_gap)
        straddle = (-left_extent - 0.5 * straddle_len, right_extent + 0.5 * straddle_len)

        remaining = n - 1
        n_left = int(rng.integers(0, remaining + 1))
        n_right = remaining - n_left

        segments: List[Tuple[float, float]] = [straddle]
        if n_left:
            left = draw_instance_one_sided(
                n_left,
                h,
                L,
                min_gap=min_gap,
                min_len=min_len,
                max_len=max_len,
                rng=rng,
                margin=attempt_margin,
                max_attempts=max_attempts,
            )
            left = [(-b, -a) for a, b in left][::-1]
            max_left_end = max(b for _, b in left)
            if max_left_end > straddle[0] - min_gap:
                shift = (straddle[0] - min_gap) - max_left_end
                left = [(a + shift, b + shift) for a, b in left]
            segments.extend(left)
        if n_right:
            right = draw_instance_one_sided(
                n_right,
                h,
                L,
                min_gap=min_gap,
                min_len=min_len,
                max_len=max_len,
                rng=rng,
                margin=attempt_margin,
                max_attempts=max_attempts,
            )
            min_right_start = min(a for a, _ in right)
            if min_right_start < straddle[1] + min_gap:
                shift = (straddle[1] + min_gap) - min_right_start
                right = [(a + shift, b + shift) for a, b in right]
            segments.extend(right)

        combined = sorted(segments, key=lambda seg: seg[0])
        try:
            validate_instance(combined)
        except ValueError:
            continue
        if not math.isfinite(L) or (
            _per_segment_feasible(combined, h, L, attempt_margin)
            and _gs_reach_feasible(combined, h, L, attempt_margin)
        ):
            return combined

    raise RuntimeError("failed to generate straddling instance within attempts budget")


def draw_family(family_name: FamilyName, cfg: FamilyConfig, rng: np.random.Generator) -> Instance:
    """Sample a fully parameterised instance for the requested family.

    The function is deterministic with respect to ``rng`` and returns an
    :class:`Instance` object ready for labelling.
    """

    attempts = max(1, cfg.max_attempts)
    margin = max(cfg.margin_multiple * TOL_NUM, 5.0 * TOL_NUM)

    for attempt in range(attempts):
        side_choice = _choose_side(family_name, cfg.side_mix, rng)
        n_segments = _sample_segment_count(cfg, rng, side_choice)
        if family_name == "straddlers" and cfg.n is None:
            target_frac = float(
                rng.uniform(cfg.straddle_fraction[0], cfg.straddle_fraction[1])
            )
            if target_frac <= 0.0:
                target_frac = 0.25
            n_segments = max(3, int(round(1.0 / target_frac)))

        h = float(rng.uniform(cfg.h_range[0], cfg.h_range[1]))
        try:
            segments = _generate_segments_for_family(
                family_name=family_name,
                side_choice=side_choice,
                n_segments=n_segments,
                cfg=cfg,
                rng=rng,
            )
        except ValueError:
            if attempt >= attempts - 1:
                break
            continue

        baseline_L = baseline_length_requirement(segments, h)
        tight_delta = _sample_tight_delta(cfg, segments, rng, margin)
        tight_L = baseline_L + tight_delta

        if cfg.L_mode == "tight":
            L = tight_L
        elif cfg.L_mode == "roomy":
            multiplier = float(
                rng.uniform(cfg.roomy_multiplier_range[0], cfg.roomy_multiplier_range[1])
            )
            L = tight_L * multiplier
        else:
            if rng.random() < cfg.tight_probability:
                L = tight_L
            else:
                multiplier = float(
                    rng.uniform(
                        cfg.roomy_multiplier_range[0], cfg.roomy_multiplier_range[1]
                    )
                )
                L = tight_L * multiplier

        if not _per_segment_feasible(segments, h, L, margin):
            if attempt >= attempts - 1:
                break
            continue
        if not _gs_reach_feasible(segments, h, L, margin):
            if attempt >= attempts - 1:
                break
            continue

        try:
            validate_instance(segments)
        except ValueError:
            if attempt >= attempts - 1:
                break
            continue

        return Instance(tuple(segments), h, L)

    raise RuntimeError(f"failed to draw {family_name} family instance after {attempts} attempts")


# ---------------------------------------------------------------------------
# Helper predicates for curriculum buckets
# ---------------------------------------------------------------------------

def baseline_length_requirement(segments: Sequence[Tuple[float, float]], h: float) -> float:
    """Return the tight feasibility baseline for ``L``."""

    max_seg = max(tour_length(a, b, h) for a, b in segments)
    farthest = max(max(abs(a), abs(b)) for a, b in segments)
    reach = 2.0 * math.hypot(farthest, h)
    return max(max_seg, reach)


def is_tight_L(L: float, baseline: float, margin: float) -> bool:
    """Return ``True`` when ``L`` is within a small margin of the baseline."""

    return L <= baseline + 5.0 * margin


def is_roomy_L(L: float, baseline: float) -> bool:
    """Return ``True`` when ``L`` is comfortably above the baseline."""

    return L >= 1.3 * baseline


def classify_side(segments: Sequence[Tuple[float, float]]) -> str:
    """Classify an instance into one of the geometric side regimes."""

    has_left = any(b < -GEOM_EPSILON for _, b in segments)
    has_right = any(a > GEOM_EPSILON for a, _ in segments)
    has_straddle = any(a < -GEOM_EPSILON and b > GEOM_EPSILON for a, b in segments)
    if has_straddle:
        return "straddling"
    if has_left and has_right:
        return "two_sided"
    return "one_sided"


def estimate_difficulty(instance: Instance) -> str:
    """Heuristic curriculum difficulty classification.

    The estimate is based on how many origin-to-endpoint tours the budget
    can support under coarse approximations. Accurate labelling happens after
    gold inference, but the heuristic is useful for early stratification.
    """

    segments = instance.segments
    h = instance.h
    L = instance.L

    min_x = min(a for a, _ in segments)
    max_x = max(b for _, b in segments)
    span_cost = tour_length(min_x, max_x, h)
    if span_cost <= L + TOL_NUM:
        return "easy"

    left_cost = tour_length(min_x, 0.0, h)
    right_cost = tour_length(0.0, max_x, h)
    if left_cost <= L + TOL_NUM and right_cost <= L + TOL_NUM:
        return "easy"

    approx_tours = max(1.0, span_cost / max(L, 1e-9))
    if approx_tours <= 6.0:
        return "medium"
    return "hard"


def bridge_usefulness(cost_full: float, cost_no_bridge: float, threshold: float = 0.02) -> str:
    """Return ``bridge-useful`` or ``bridge-useless`` depending on cost gap."""

    improvement = cost_no_bridge - cost_full
    if improvement > threshold * max(cost_full, 1e-9):
        return "bridge-useful"
    if abs(improvement) <= threshold * max(cost_full, 1e-9):
        return "bridge-useless"
    return "bridge-ambiguous"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _per_segment_feasible(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
    margin: float,
) -> bool:
    if not math.isfinite(L):
        return True
    max_allowed = L - margin
    if max_allowed <= 0:
        return False
    for a, b in segments:
        if tour_length(a, b, h) > max_allowed:
            return False
    return True


def _gs_reach_feasible(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
    margin: float,
) -> bool:
    if not math.isfinite(L):
        return True
    max_abs = max(max(abs(a), abs(b)) for a, b in segments)
    return 2.0 * math.hypot(max_abs, h) <= L - margin + TOL_NUM


def _choose_side(
    family_name: FamilyName, weights: Tuple[float, float, float], rng: np.random.Generator
) -> SideChoice:
    if family_name == "straddlers":
        return "straddling"
    totals = np.array(weights, dtype=float)
    if totals.sum() <= 0:
        totals = np.array([1.0, 0.0, 0.0])
    probs = totals / totals.sum()
    choice = rng.choice(["one_sided", "two_sided", "straddling"], p=probs)
    return cast(SideChoice, choice)


def _sample_segment_count(cfg: FamilyConfig, rng: np.random.Generator, side: SideChoice) -> int:
    if cfg.n is not None:
        return cfg.n

    n_low, n_high = cfg.n_extrap_range if cfg.use_extrapolation else cfg.n_range
    if side == "straddling" and n_high < 3:
        n_high = max(n_high, 3)
        n_low = min(n_low, n_high)
    return int(rng.integers(n_low, n_high + 1))


def _sample_tight_delta(
    cfg: FamilyConfig,
    segments: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
    margin: float,
) -> float:
    lengths = [b - a for a, b in segments]
    median_length = float(np.median(lengths))
    scale = max(median_length, cfg.min_len, margin)
    low, high = cfg.tight_delta_fraction
    delta = float(rng.uniform(low, high) * scale)
    return max(delta, 1.5 * margin)


def _sample_log_lengths(
    rng: np.random.Generator,
    n: int,
    min_len: float,
    max_len: float,
) -> np.ndarray:
    log_low = math.log(min_len)
    log_high = math.log(max_len)
    lengths = np.exp(rng.uniform(log_low, log_high, size=n))
    return np.clip(lengths, min_len, max_len)


def _realize_segments_from_centers(
    centers: Sequence[float],
    lengths: Sequence[float],
    min_gap: float,
    domain: Tuple[float, float],
) -> List[Tuple[float, float]]:
    if len(centers) != len(lengths):
        raise ValueError("centers and lengths must have the same cardinality")

    low, high = domain
    if high <= low:
        raise ValueError("domain upper bound must exceed lower bound")

    pairs = sorted(zip(centers, lengths), key=lambda item: item[0])
    cursor = low - min_gap
    segments: List[Tuple[float, float]] = []
    for center, length in pairs:
        length = float(length)
        half = 0.5 * length
        start = center - half
        end = center + half

        if start < low:
            end -= start - low
            start = low
        if end > high:
            start -= end - high
            end = high

        if start < low - EPS_GEOM or end > high + EPS_GEOM:
            raise ValueError("segment cannot be placed inside the domain bounds")

        if start < cursor + min_gap:
            start = cursor + min_gap
            end = start + length

        if end > high + EPS_GEOM:
            raise ValueError("insufficient space for segments under constraints")

        segments.append((float(start), float(end)))
        cursor = end

    return segments


def _combine_two_sided(
    left_positive: Sequence[Tuple[float, float]],
    right_positive: Sequence[Tuple[float, float]],
    min_gap: float,
) -> List[Tuple[float, float]]:
    left = [(-b, -a) for a, b in left_positive][::-1]
    right = list(right_positive)

    if left:
        max_left_end = max(b for _, b in left)
        if max_left_end > -min_gap:
            shift = -(max_left_end + min_gap)
            left = [(a + shift, b + shift) for a, b in left]

    if right:
        min_right_start = min(a for a, _ in right)
        if min_right_start < min_gap:
            shift = min_gap - min_right_start
            right = [(a + shift, b + shift) for a, b in right]

    combined = sorted([*left, *right], key=lambda seg: seg[0])
    validate_instance(combined)
    return combined


def _generate_uniform_family(
    side_choice: SideChoice,
    n_segments: int,
    cfg: FamilyConfig,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    Xmax = float(rng.uniform(cfg.x_extent[0], cfg.x_extent[1]))
    domain = (0.0, Xmax)
    lengths = _sample_log_lengths(rng, n_segments, cfg.min_len, cfg.max_len)

    if side_choice == "one_sided":
        centers = np.sort(rng.uniform(domain[0], domain[1], size=n_segments))
        return _realize_segments_from_centers(centers, lengths, cfg.min_gap, domain)

    if side_choice == "two_sided":
        n_left = max(1, int(rng.integers(1, n_segments)))
        n_right = max(n_segments - n_left, 1)
        order = rng.permutation(n_segments)
        lengths_perm = lengths[order]
        centers_left = np.sort(rng.uniform(domain[0], domain[1], size=n_left))
        centers_right = np.sort(rng.uniform(domain[0], domain[1], size=n_right))
        left = _realize_segments_from_centers(
            centers_left, lengths_perm[:n_left], cfg.min_gap, domain
        )
        right = _realize_segments_from_centers(
            centers_right, lengths_perm[n_left:], cfg.min_gap, domain
        )
        return _combine_two_sided(left, right, cfg.min_gap)

    # straddling: reuse dedicated helper
    return draw_instance_straddling(
        max(3, n_segments),
        h=cfg.h_range[0],  # placeholder (ignored by generator)
        L=math.inf,
        min_gap=cfg.min_gap,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
        rng=rng,
        margin=cfg.margin_multiple * TOL_NUM,
    )


def _generate_clustered_family(
    side_choice: SideChoice,
    n_segments: int,
    cfg: FamilyConfig,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    Xmax = float(rng.uniform(cfg.x_extent[0], cfg.x_extent[1]))
    domain = (0.0, Xmax)
    lengths = _sample_log_lengths(rng, n_segments, cfg.min_len, cfg.max_len)
    cluster_count = int(rng.integers(cfg.cluster_count_range[0], cfg.cluster_count_range[1] + 1))
    centers_cluster = rng.uniform(domain[0], domain[1], size=cluster_count)
    sigma = float(rng.uniform(cfg.cluster_spread_ratio[0], cfg.cluster_spread_ratio[1]) * Xmax)

    def sample_centers(count: int) -> np.ndarray:
        assignments = rng.integers(0, cluster_count, size=count)
        centers = centers_cluster[assignments] + rng.normal(0.0, sigma, size=count)
        return np.clip(centers, domain[0], domain[1])

    if side_choice == "one_sided":
        centers = np.sort(sample_centers(n_segments))
        return _realize_segments_from_centers(centers, lengths, cfg.min_gap, domain)

    if side_choice == "two_sided":
        n_left = max(1, int(rng.integers(1, n_segments)))
        n_right = max(n_segments - n_left, 1)
        lengths_perm = lengths[rng.permutation(n_segments)]
        centers_left = np.sort(sample_centers(n_left))
        centers_right = np.sort(sample_centers(n_right))
        left = _realize_segments_from_centers(
            centers_left, lengths_perm[:n_left], cfg.min_gap, domain
        )
        right = _realize_segments_from_centers(
            centers_right, lengths_perm[n_left:], cfg.min_gap, domain
        )
        return _combine_two_sided(left, right, cfg.min_gap)

    # straddling configuration: anchor around zero by composing one straddler.
    return _generate_straddler_family(n_segments, cfg, rng)


def _generate_step_gap_family(
    side_choice: SideChoice,
    n_segments: int,
    cfg: FamilyConfig,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    small_gap = rng.uniform(
        cfg.step_gap_small_factor[0] * cfg.min_gap,
        cfg.step_gap_small_factor[1] * cfg.min_gap,
    )
    large_gap = rng.uniform(
        cfg.step_gap_large_factor[0] * cfg.min_gap,
        cfg.step_gap_large_factor[1] * cfg.min_gap,
    )
    short_len_low = cfg.min_len
    short_len_high = min(cfg.max_len, cfg.step_length_short_factor[1] * cfg.min_len)
    long_len_low = max(cfg.min_len, cfg.step_length_long_factor[0] * cfg.min_len)
    long_len_high = cfg.max_len

    def build_sequence(count: int) -> List[Tuple[float, float]]:
        segments: List[Tuple[float, float]] = []
        cursor = rng.uniform(0.0, cfg.min_gap)
        for idx in range(count):
            gap = small_gap if idx % 2 == 0 else large_gap
            length_range = (short_len_low, short_len_high) if idx % 2 == 0 else (long_len_low, long_len_high)
            length = float(rng.uniform(*length_range))
            cursor += gap
            start = cursor
            end = start + length
            segments.append((start, end))
            cursor = end
        shift = rng.uniform(-segments[0][0], cfg.min_gap)
        return [(a + shift, b + shift) for a, b in segments]

    if side_choice == "one_sided":
        return build_sequence(n_segments)

    if side_choice == "two_sided":
        n_left = max(1, int(rng.integers(1, n_segments)))
        n_right = max(n_segments - n_left, 1)
        left = build_sequence(n_left)
        right = build_sequence(n_right)
        return _combine_two_sided(left, right, cfg.min_gap)

    return _generate_straddler_family(n_segments, cfg, rng)


def _generate_straddler_family(
    n_segments: int,
    cfg: FamilyConfig,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    n_total = max(2, n_segments)
    straddle = draw_instance_straddling(
        n_total,
        h=cfg.h_range[0],
        L=math.inf,
        min_gap=cfg.min_gap,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
        rng=rng,
        margin=cfg.margin_multiple * TOL_NUM,
    )
    return straddle


def _generate_segments_for_family(
    family_name: FamilyName,
    side_choice: SideChoice,
    n_segments: int,
    cfg: FamilyConfig,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    if family_name == "uniform":
        return _generate_uniform_family(side_choice, n_segments, cfg, rng)
    if family_name == "clustered":
        return _generate_clustered_family(side_choice, n_segments, cfg, rng)
    if family_name == "step_gap":
        return _generate_step_gap_family(side_choice, n_segments, cfg, rng)
    if family_name == "straddlers":
        return _generate_straddler_family(n_segments, cfg, rng)

    raise ValueError(f"unknown family_name={family_name}")


__all__ = [
    "FamilyConfig",
    "FamilyName",
    "draw_family",
    "draw_instance_one_sided",
    "draw_instance_two_sided",
    "draw_instance_straddling",
    "validate_instance",
    "baseline_length_requirement",
    "is_tight_L",
    "is_roomy_L",
    "classify_side",
    "estimate_difficulty",
    "bridge_usefulness",
]
