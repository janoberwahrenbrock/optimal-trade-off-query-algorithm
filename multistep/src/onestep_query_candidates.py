from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .models import Query
from .ratio_intervals import GoalPairRatioIntervals, RatioInterval


QUERY_EPSILON = 1e-3
RAW_QUERY_TOLERANCE = 1e-12
QUERY_DEDUP_ABS_TOLERANCE = 1e-12
QUERY_DEDUP_REL_TOLERANCE = 1e-9

RawQueryCandidateSide = Literal["left", "right"]


@dataclass(frozen=True)
class _RawQueryCandidate:
    goal_index_a: int
    goal_index_b: int
    breakpoint_value: float
    side: RawQueryCandidateSide


def compute_onestep_query_candidates(
    goal_pair_ratio_intervals: list[GoalPairRatioIntervals],
    epsilon: float = QUERY_EPSILON,
    dedup_abs_tol: float = QUERY_DEDUP_ABS_TOLERANCE,
    dedup_rel_tol: float = QUERY_DEDUP_REL_TOLERANCE,
) -> list[Query]:
    _validate_epsilon(epsilon)
    _validate_dedup_tolerances(dedup_abs_tol, dedup_rel_tol)

    raw_query_candidates: list[_RawQueryCandidate] = []
    for goal_pair_intervals in goal_pair_ratio_intervals:
        raw_query_candidates.extend(
            _extract_raw_query_candidates_for_pair(
                goal_pair_intervals=goal_pair_intervals,
                epsilon=epsilon,
            )
        )

    deduplicated_raw_query_candidates = _deduplicate_raw_query_candidates(
        raw_query_candidates=raw_query_candidates,
        epsilon=epsilon,
        abs_tol=dedup_abs_tol,
        rel_tol=dedup_rel_tol,
    )
    query_candidates = [
        _build_query_from_raw_candidate(raw_query_candidate, epsilon)
        for raw_query_candidate in deduplicated_raw_query_candidates
    ]
    return _sort_and_deduplicate_queries(
        queries=query_candidates,
        abs_tol=dedup_abs_tol,
        rel_tol=dedup_rel_tol,
    )


def _extract_raw_query_candidates_for_pair(
    goal_pair_intervals: GoalPairRatioIntervals,
    epsilon: float,
) -> list[_RawQueryCandidate]:
    raw_query_candidates: list[_RawQueryCandidate] = []

    for ratio_interval in goal_pair_intervals.intervals_by_candidate.values():
        _extend_raw_query_candidates_from_ratio_interval(
            raw_query_candidates=raw_query_candidates,
            ratio_interval=ratio_interval,
            goal_index_a=int(goal_pair_intervals.goal_index_a),
            goal_index_b=int(goal_pair_intervals.goal_index_b),
            epsilon=epsilon,
        )

    raw_query_candidates.sort(
        key=lambda candidate: (
            candidate.side,
            candidate.breakpoint_value,
        )
    )
    return _deduplicate_neighboring_raw_query_candidates(
        raw_query_candidates=raw_query_candidates,
        abs_tol=RAW_QUERY_TOLERANCE,
        rel_tol=0.0,
    )


def _extend_raw_query_candidates_from_ratio_interval(
    raw_query_candidates: list[_RawQueryCandidate],
    ratio_interval: RatioInterval,
    goal_index_a: int,
    goal_index_b: int,
    epsilon: float,
) -> None:
    lower = ratio_interval.lower
    upper = ratio_interval.upper

    if lower.status == "unbounded":
        raise ValueError("lower ratio bound must not be unbounded")

    if lower.status == "optimal":
        if lower.optimal_value is None:
            raise RuntimeError("optimal lower ratio bound has no optimal_value")

        if lower.optimal_value > epsilon:
            raw_query_candidates.append(
                _RawQueryCandidate(
                    goal_index_a=goal_index_a,
                    goal_index_b=goal_index_b,
                    breakpoint_value=float(lower.optimal_value),
                    side="left",
                )
            )

    if upper.status == "optimal":
        if upper.optimal_value is None:
            raise RuntimeError("optimal upper ratio bound has no optimal_value")

        raw_query_candidates.append(
            _RawQueryCandidate(
                goal_index_a=goal_index_a,
                goal_index_b=goal_index_b,
                breakpoint_value=float(upper.optimal_value),
                side="right",
            )
        )


def _deduplicate_raw_query_candidates(
    raw_query_candidates: list[_RawQueryCandidate],
    epsilon: float,
    abs_tol: float,
    rel_tol: float,
) -> list[_RawQueryCandidate]:
    canonical_raw_query_candidates = [
        _canonicalize_raw_query_candidate(
            raw_query_candidate=raw_query_candidate,
            epsilon=epsilon,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        for raw_query_candidate in raw_query_candidates
    ]
    canonical_raw_query_candidates.sort(
        key=lambda candidate: (
            candidate.goal_index_a,
            candidate.goal_index_b,
            candidate.side,
            candidate.breakpoint_value,
        )
    )
    return _deduplicate_neighboring_raw_query_candidates(
        raw_query_candidates=canonical_raw_query_candidates,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )


def _canonicalize_raw_query_candidate(
    raw_query_candidate: _RawQueryCandidate,
    epsilon: float,
    abs_tol: float,
    rel_tol: float,
) -> _RawQueryCandidate:
    if raw_query_candidate.breakpoint_value <= 0.0:
        return raw_query_candidate

    mirrored_raw_query_candidate = _build_mirrored_raw_query_candidate(
        raw_query_candidate
    )
    query_value = _compute_query_value_from_raw_candidate(
        raw_query_candidate=raw_query_candidate,
        epsilon=epsilon,
    )
    mirrored_query_value = _compute_query_value_from_raw_candidate(
        raw_query_candidate=mirrored_raw_query_candidate,
        epsilon=epsilon,
    )

    if mirrored_query_value < 0.0:
        return raw_query_candidate

    if query_value < 0.0:
        return mirrored_raw_query_candidate

    if query_value >= 1.0 and mirrored_query_value < 1.0:
        return raw_query_candidate

    if mirrored_query_value >= 1.0 and query_value < 1.0:
        return mirrored_raw_query_candidate

    if math.isclose(query_value, mirrored_query_value, abs_tol=abs_tol, rel_tol=rel_tol):
        return min(
            raw_query_candidate,
            mirrored_raw_query_candidate,
            key=_raw_query_candidate_sort_key,
        )

    return max(
        raw_query_candidate,
        mirrored_raw_query_candidate,
        key=lambda candidate: float(
            _build_query_from_raw_candidate(candidate, epsilon).value
        ),
    )


def _build_mirrored_raw_query_candidate(
    raw_query_candidate: _RawQueryCandidate,
) -> _RawQueryCandidate:
    mirrored_side: RawQueryCandidateSide = "right"
    if raw_query_candidate.side == "right":
        mirrored_side = "left"

    return _RawQueryCandidate(
        goal_index_a=raw_query_candidate.goal_index_b,
        goal_index_b=raw_query_candidate.goal_index_a,
        breakpoint_value=1.0 / raw_query_candidate.breakpoint_value,
        side=mirrored_side,
    )


def _build_query_from_raw_candidate(
    raw_query_candidate: _RawQueryCandidate,
    epsilon: float,
) -> Query:
    query_value = _compute_query_value_from_raw_candidate(
        raw_query_candidate=raw_query_candidate,
        epsilon=epsilon,
    )
    if query_value < 0.0 and math.isclose(
        query_value,
        0.0,
        abs_tol=RAW_QUERY_TOLERANCE,
        rel_tol=0.0,
    ):
        query_value = 0.0

    return Query(
        ziel_index_a=raw_query_candidate.goal_index_a,
        ziel_index_b=raw_query_candidate.goal_index_b,
        value=query_value,
    )


def _compute_query_value_from_raw_candidate(
    raw_query_candidate: _RawQueryCandidate,
    epsilon: float,
) -> float:
    if raw_query_candidate.side == "left":
        return raw_query_candidate.breakpoint_value - epsilon

    return raw_query_candidate.breakpoint_value + epsilon


def _deduplicate_neighboring_raw_query_candidates(
    raw_query_candidates: list[_RawQueryCandidate],
    abs_tol: float,
    rel_tol: float,
) -> list[_RawQueryCandidate]:
    deduplicated_raw_query_candidates: list[_RawQueryCandidate] = []
    for raw_query_candidate in raw_query_candidates:
        if not deduplicated_raw_query_candidates or not _have_same_raw_candidate_form(
            raw_query_candidate_a=deduplicated_raw_query_candidates[-1],
            raw_query_candidate_b=raw_query_candidate,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        ):
            deduplicated_raw_query_candidates.append(raw_query_candidate)

    return deduplicated_raw_query_candidates


def _have_same_raw_candidate_form(
    raw_query_candidate_a: _RawQueryCandidate,
    raw_query_candidate_b: _RawQueryCandidate,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    return (
        raw_query_candidate_a.goal_index_a == raw_query_candidate_b.goal_index_a
        and raw_query_candidate_a.goal_index_b == raw_query_candidate_b.goal_index_b
        and raw_query_candidate_a.side == raw_query_candidate_b.side
        and math.isclose(
            raw_query_candidate_a.breakpoint_value,
            raw_query_candidate_b.breakpoint_value,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    )


def _sort_and_deduplicate_queries(
    queries: list[Query],
    abs_tol: float,
    rel_tol: float,
) -> list[Query]:
    queries.sort(
        key=lambda query: (
            int(query.ziel_index_a),
            int(query.ziel_index_b),
            float(query.value),
        )
    )

    deduplicated_queries: list[Query] = []
    for query in queries:
        if not deduplicated_queries or not _have_same_query_form(
            query_a=deduplicated_queries[-1],
            query_b=query,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        ):
            deduplicated_queries.append(query)

    return deduplicated_queries


def _have_same_query_form(
    query_a: Query,
    query_b: Query,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    return (
        query_a.ziel_index_a == query_b.ziel_index_a
        and query_a.ziel_index_b == query_b.ziel_index_b
        and math.isclose(
            float(query_a.value),
            float(query_b.value),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    )


def _raw_query_candidate_sort_key(
    raw_query_candidate: _RawQueryCandidate,
) -> tuple[int, int, str, float]:
    return (
        raw_query_candidate.goal_index_a,
        raw_query_candidate.goal_index_b,
        raw_query_candidate.side,
        raw_query_candidate.breakpoint_value,
    )


def _validate_epsilon(epsilon: float) -> None:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")


def _validate_dedup_tolerances(abs_tol: float, rel_tol: float) -> None:
    if abs_tol < 0.0:
        raise ValueError("abs_tol must not be negative")

    if rel_tol < 0.0:
        raise ValueError("rel_tol must not be negative")
