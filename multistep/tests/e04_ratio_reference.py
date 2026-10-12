from __future__ import annotations

"""Frozen e04 candidate helper, retained as an independent equivalence oracle."""

import numpy as np

from multistep.src.models.linear_optimization_result import LinearOptimizationResult
from multistep.src.ratio_intervals import (
    RatioInterval,
    _build_infeasible_ratio_interval,
    _compute_ratio_interval_from_vertices,
    get_ordered_goal_pairs,
)


def compute_e04_reference_intervals(
    vertices: np.ndarray,
    goal_count: int,
    tolerance: float,
) -> dict[tuple[int, int], RatioInterval]:
    """Reuse goal-column masks while retaining each direct ratio calculation."""

    goal_pairs = get_ordered_goal_pairs(goal_count)
    if not (
        type(vertices) is np.ndarray
        and vertices.ndim == 2
        and vertices.shape == (len(vertices), goal_count)
        and len(vertices) > 0
        and vertices.dtype == np.dtype(np.float64)
        and np.all(np.isfinite(vertices))
    ):
        return {
            (goal_index_a, goal_index_b): _compute_ratio_interval_from_vertices(
                vertices=vertices,
                goal_index_a=goal_index_a,
                goal_index_b=goal_index_b,
                tolerance=tolerance,
            )
            for goal_index_a, goal_index_b in goal_pairs
        }

    columns = [vertices[:, goal_index] for goal_index in range(goal_count)]
    positive_masks = vertices > tolerance
    nonpositive_masks = vertices <= tolerance
    has_positive = np.any(positive_masks, axis=0)
    positive_denominators = [
        column[positive_masks[:, goal_index]]
        for goal_index, column in enumerate(columns)
    ]
    intervals: dict[tuple[int, int], RatioInterval] = {}
    for goal_index_a, goal_index_b in goal_pairs:
        if not has_positive[goal_index_b]:
            intervals[(goal_index_a, goal_index_b)] = _build_infeasible_ratio_interval(
                "ratio denominator is zero throughout the candidate region"
            )
            continue

        ratios = (
            columns[goal_index_a][positive_masks[:, goal_index_b]]
            / positive_denominators[goal_index_b]
        )
        lower_value = max(0.0, float(np.min(ratios)))
        lower = LinearOptimizationResult(
            status="optimal",
            objective_sense="min",
            optimal_value=lower_value,
        )
        has_unbounded_vertex = bool(
            np.any(nonpositive_masks[:, goal_index_b] & positive_masks[:, goal_index_a])
        )
        if has_unbounded_vertex:
            upper = LinearOptimizationResult(
                status="unbounded",
                objective_sense="max",
            )
        else:
            upper = LinearOptimizationResult(
                status="optimal",
                objective_sense="max",
                optimal_value=max(0.0, float(np.max(ratios))),
            )
        intervals[(goal_index_a, goal_index_b)] = RatioInterval(lower=lower, upper=upper)
    return intervals
