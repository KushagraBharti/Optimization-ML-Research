from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

EPS = 1e-9


@dataclass(frozen=True, order=True)
class Segment:
    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a > self.b + EPS:
            raise ValueError(f"Invalid segment [{self.a}, {self.b}]: expected a <= b.")

    @property
    def length(self) -> float:
        return self.b - self.a

    def clip(self, left: float | None = None, right: float | None = None) -> Segment | None:
        l = self.a if left is None else max(self.a, left)
        r = self.b if right is None else min(self.b, right)
        if r < l - EPS:
            return None
        return Segment(l, r)

    def contains(self, x: float) -> bool:
        return self.a - EPS <= x <= self.b + EPS


@dataclass(frozen=True)
class Instance:
    h: float
    L: float
    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        if self.h <= 0:
            raise ValueError("h must be strictly positive.")
        if self.L <= 0:
            raise ValueError("L must be strictly positive.")
        normalized = tuple(sorted(self.segments, key=lambda s: (s.a, s.b)))
        if normalized != self.segments:
            object.__setattr__(self, "segments", normalized)
        for i in range(1, len(self.segments)):
            prev = self.segments[i - 1]
            curr = self.segments[i]
            if prev.b >= curr.a - EPS:
                raise ValueError(
                    "Segments must be disjoint and sorted with strict gap: "
                    f"[{prev.a}, {prev.b}] then [{curr.a}, {curr.b}]."
                )

    @classmethod
    def from_iterable(cls, h: float, L: float, segments: Iterable[tuple[float, float]]) -> Instance:
        return cls(h=h, L=L, segments=tuple(Segment(a, b) for a, b in segments))

    @property
    def is_empty(self) -> bool:
        return len(self.segments) == 0

    @property
    def a1(self) -> float:
        if self.is_empty:
            raise ValueError("Empty instance has no a1.")
        return self.segments[0].a

    @property
    def bn(self) -> float:
        if self.is_empty:
            raise ValueError("Empty instance has no bn.")
        return self.segments[-1].b

    @property
    def min_x(self) -> float:
        return self.a1

    @property
    def max_x(self) -> float:
        return self.bn

    def one_side(self) -> bool:
        return self.bn <= EPS or self.a1 >= -EPS

    def side_sign(self) -> int:
        if self.is_empty:
            return 0
        if self.a1 >= -EPS:
            return 1
        if self.bn <= EPS:
            return -1
        return 0

    def clipped(self, left: float | None = None, right: float | None = None) -> Instance:
        clipped: list[Segment] = []
        for s in self.segments:
            s2 = s.clip(left=left, right=right)
            if s2 is not None and s2.length >= -EPS:
                clipped.append(s2)
        return Instance(h=self.h, L=self.L, segments=tuple(clipped))

    def mirrored_x(self) -> Instance:
        mirrored = [Segment(-s.b, -s.a) for s in self.segments]
        return Instance(h=self.h, L=self.L, segments=tuple(mirrored))

    def normalize_to_right_side(self) -> tuple[Instance, int]:
        sign = self.side_sign()
        if sign == 0:
            raise ValueError("Instance is not one-sided.")
        if sign == 1:
            return self, 1
        transformed = [Segment(-s.b, -s.a) for s in self.segments]
        return Instance(h=self.h, L=self.L, segments=tuple(transformed)), -1


@dataclass(frozen=True)
class Tour:
    left: float
    right: float
    length: float
    maximal: bool
    tag: str = ""

    def __post_init__(self) -> None:
        if self.left > self.right + EPS:
            raise ValueError("Tour left endpoint must satisfy left <= right.")
        if self.length < -EPS:
            raise ValueError("Tour length must be non-negative.")


@dataclass(frozen=True)
class Solution:
    tours: tuple[Tour, ...]
    total_length: float
    tour_count: int
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_tours(
        cls,
        tours: Iterable[Tour],
        metadata: dict[str, object] | None = None,
    ) -> Solution:
        t = tuple(tours)
        total = sum(x.length for x in t)
        return cls(
            tours=t,
            total_length=total,
            tour_count=len(t),
            metadata={} if metadata is None else dict(metadata),
        )


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    uncovered_segments: tuple[Segment, ...]
    infeasible_tours: tuple[int, ...]
    notes: tuple[str, ...] = ()
