from __future__ import annotations

"""Profiling helpers for optimized multistep implementations."""

from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass
class OptimizationProfile:
    """Optional low-overhead counters for one optimized evaluation."""

    counters: Counter[str] = field(default_factory=Counter)
    seconds_by_operation: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            self.seconds_by_operation[name] += perf_counter() - started_at


_ACTIVE_PROFILE: ContextVar[OptimizationProfile | None] = ContextVar(
    "multistep_optimization_profile",
    default=None,
)


@contextmanager
def collect_optimization_profile() -> Iterator[OptimizationProfile]:
    """Collect counters for optimized calls made inside this context."""

    profile = OptimizationProfile()
    token = _ACTIVE_PROFILE.set(profile)
    try:
        yield profile
    finally:
        _ACTIVE_PROFILE.reset(token)


def increment_profile_counter(name: str, amount: int = 1) -> None:
    profile = _ACTIVE_PROFILE.get()
    if profile is not None:
        profile.increment(name, amount)


@contextmanager
def profile_operation(name: str) -> Iterator[None]:
    profile = _ACTIVE_PROFILE.get()
    if profile is None:
        yield
        return

    with profile.timed(name):
        yield
