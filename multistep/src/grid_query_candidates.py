from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .linear_constraints import LinearConstraintSystem
from .models import Query
from .models.linear_optimization_result import LinearOptimizationResult
from .ratio_intervals import (
    RatioInterval,
    compute_ratio_bounds_for_weight_space,
    get_ordered_goal_pairs,
)


GridSpacing = Literal["linear", "log"]
DEFAULT_GRID_SIZE = 21
DEFAULT_MIN_QUERY_VALUE = 1e-3
DEFAULT_MAX_QUERY_VALUE = 100.0
DEFAULT_GRID_SPACING: GridSpacing = "log"
QUERY_DEDUP_ABS_TOLERANCE = 1e-12
QUERY_DEDUP_REL_TOLERANCE = 1e-9


def compute_grid_query_candidates(
    weight_space: LinearConstraintSystem,
    grid_size: int = DEFAULT_GRID_SIZE,
    min_query_value: float = DEFAULT_MIN_QUERY_VALUE,
    max_query_value: float = DEFAULT_MAX_QUERY_VALUE,
    spacing: GridSpacing = DEFAULT_GRID_SPACING,
    dedup_abs_tol: float = QUERY_DEDUP_ABS_TOLERANCE,
    dedup_rel_tol: float = QUERY_DEDUP_REL_TOLERANCE,
) -> list[Query]:
    if weight_space.variable_count <= 1:
        raise ValueError("weight_space must contain at least two goals")

    query_candidates: list[Query] = []
    for goal_index_a, goal_index_b in get_ordered_goal_pairs(
        weight_space.variable_count
    ):
        ratio_interval = compute_ratio_bounds_for_weight_space(
            weight_space=weight_space,
            goal_index_a=goal_index_a,
            goal_index_b=goal_index_b,
        )
        query_values = build_grid_query_values_from_ratio_interval(
            ratio_interval=ratio_interval,
            grid_size=grid_size,
            min_query_value=min_query_value,
            max_query_value=max_query_value,
            spacing=spacing,
        )
        query_candidates.extend(
            Query(
                ziel_index_a=goal_index_a,
                ziel_index_b=goal_index_b,
                value=query_value,
            )
            for query_value in query_values
        )

    return deduplicate_mirrored_query_candidates(
        queries=query_candidates,
        abs_tol=dedup_abs_tol,
        rel_tol=dedup_rel_tol,
    )


def build_grid_query_values_from_ratio_interval(
    ratio_interval: RatioInterval,
    grid_size: int,
    min_query_value: float = DEFAULT_MIN_QUERY_VALUE,
    max_query_value: float = DEFAULT_MAX_QUERY_VALUE,
    spacing: GridSpacing = DEFAULT_GRID_SPACING,
) -> list[float]:
    return build_grid_query_values(
        lower=ratio_interval.lower,
        upper=ratio_interval.upper,
        grid_size=grid_size,
        min_query_value=min_query_value,
        max_query_value=max_query_value,
        spacing=spacing,
    )


def build_grid_query_values(
    lower: LinearOptimizationResult,
    upper: LinearOptimizationResult,
    grid_size: int,
    min_query_value: float = DEFAULT_MIN_QUERY_VALUE,
    max_query_value: float = DEFAULT_MAX_QUERY_VALUE,
    spacing: GridSpacing = DEFAULT_GRID_SPACING,
) -> list[float]:
    _validate_grid_parameters(
        grid_size=grid_size,
        min_query_value=min_query_value,
        max_query_value=max_query_value,
        spacing=spacing,
    )

    if lower.status == "unbounded":
        raise ValueError("lower ratio bound must not be unbounded")

    if lower.status != "optimal":
        return []

    if lower.optimal_value is None:
        raise RuntimeError("optimal lower ratio bound has no optimal_value")

    if upper.status == "infeasible":
        return []

    if upper.status == "optimal" and upper.optimal_value is None:
        raise RuntimeError("optimal upper ratio bound has no optimal_value")

    effective_lower = max(float(lower.optimal_value), min_query_value)
    if upper.status == "unbounded":
        effective_upper = max_query_value
    elif upper.status == "optimal":
        effective_upper = min(float(upper.optimal_value), max_query_value)
    else:
        raise ValueError(f"unknown upper ratio bound status: {upper.status}")

    if effective_upper < min_query_value:
        return []

    if effective_lower > effective_upper:
        return []

    if math.isclose(effective_lower, effective_upper, abs_tol=1e-15, rel_tol=1e-12):
        return [effective_lower]

    if spacing == "linear":
        return np.linspace(
            effective_lower,
            effective_upper,
            grid_size,
            dtype=float,
        ).tolist()

    return np.geomspace(
        effective_lower,
        effective_upper,
        grid_size,
        dtype=float,
    ).tolist()


def deduplicate_mirrored_query_candidates(
    queries: list[Query],
    abs_tol: float = QUERY_DEDUP_ABS_TOLERANCE,
    rel_tol: float = QUERY_DEDUP_REL_TOLERANCE,
) -> list[Query]:
    _validate_dedup_tolerances(abs_tol=abs_tol, rel_tol=rel_tol)

    query_pairs = [
        (query, _canonicalize_query(query=query))
        for query in queries
        if float(query.value) > 0.0
    ]
    query_pairs.sort(
        key=lambda query_pair: _query_sort_key(query_pair[1])
    )

    deduplicated_queries: list[Query] = []
    previous_canonical_query: Query | None = None
    for query, canonical_query in query_pairs:
        if previous_canonical_query is None or not _have_same_query_form(
            query_a=previous_canonical_query,
            query_b=canonical_query,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        ):
            deduplicated_queries.append(query)
            previous_canonical_query = canonical_query

    deduplicated_queries.sort(key=_query_sort_key)
    return deduplicated_queries


def _canonicalize_query(query: Query) -> Query:
    value = float(query.value)
    if value <= 0.0:
        return query

    mirrored_query = Query(
        ziel_index_a=int(query.ziel_index_b),
        ziel_index_b=int(query.ziel_index_a),
        value=1.0 / value,
    )

    return min(
        query,
        mirrored_query,
        key=lambda candidate: (
            int(candidate.ziel_index_a),
            int(candidate.ziel_index_b),
            float(candidate.value),
        ),
    )


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


def _query_sort_key(query: Query) -> tuple[int, int, float]:
    return (
        int(query.ziel_index_a),
        int(query.ziel_index_b),
        float(query.value),
    )


def _validate_grid_parameters(
    grid_size: int,
    min_query_value: float,
    max_query_value: float,
    spacing: str,
) -> None:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    if min_query_value <= 0.0:
        raise ValueError("min_query_value must be positive")

    if max_query_value <= 0.0:
        raise ValueError("max_query_value must be positive")

    if min_query_value > max_query_value:
        raise ValueError("min_query_value must not be greater than max_query_value")

    if spacing not in {"linear", "log"}:
        raise ValueError("spacing must be 'linear' or 'log'")


def _validate_dedup_tolerances(abs_tol: float, rel_tol: float) -> None:
    if abs_tol < 0.0:
        raise ValueError("abs_tol must not be negative")

    if rel_tol < 0.0:
        raise ValueError("rel_tol must not be negative")
