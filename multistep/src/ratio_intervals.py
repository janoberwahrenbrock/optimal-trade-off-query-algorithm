from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from .linear_constraints import LinearConstraintSystem
from .models import AlternativenMatrix
from .models.linear_optimization_result import LinearOptimizationResult
from .optimality_region import build_optimality_region
from .polytope_geometry import enumerate_polytope_vertices
from .weight_space import build_ratio_normalized_weight_space


RatioIntervalEngine = Literal["geometry", "lp"]


class RatioInterval(BaseModel):
    model_config = ConfigDict(frozen=True)

    lower: LinearOptimizationResult
    upper: LinearOptimizationResult


class GoalPairRatioIntervals(BaseModel):
    model_config = ConfigDict(frozen=True)

    goal_index_a: NonNegativeInt
    goal_index_b: NonNegativeInt
    intervals_by_candidate: dict[NonNegativeInt, RatioInterval]

    @model_validator(mode="after")
    def validate_goal_indices(self) -> GoalPairRatioIntervals:
        if self.goal_index_a == self.goal_index_b:
            raise ValueError("goal_index_a and goal_index_b must be different")

        return self


def get_ordered_goal_pairs(goal_count: int) -> list[tuple[int, int]]:
    if goal_count < 0:
        raise ValueError("goal_count must not be negative")

    return [
        (goal_index_a, goal_index_b)
        for goal_index_a in range(goal_count)
        for goal_index_b in range(goal_count)
        if goal_index_a != goal_index_b
    ]


def get_canonical_goal_pairs(goal_count: int) -> list[tuple[int, int]]:
    if goal_count < 0:
        raise ValueError("goal_count must not be negative")

    return [
        (goal_index_a, goal_index_b)
        for goal_index_a in range(goal_count)
        for goal_index_b in range(goal_index_a + 1, goal_count)
    ]


def compute_ratio_bounds_for_weight_space(
    weight_space: LinearConstraintSystem,
    goal_index_a: int,
    goal_index_b: int,
) -> RatioInterval:
    goal_count = weight_space.variable_count
    _validate_goal_pair(
        goal_count=goal_count,
        goal_index_a=goal_index_a,
        goal_index_b=goal_index_b,
    )

    ratio_normalized_weight_space = build_ratio_normalized_weight_space(
        weight_space=weight_space,
        normalization_goal_index=goal_index_b,
    )
    return _compute_goal_index_interval(
        system=ratio_normalized_weight_space,
        goal_count=goal_count,
        goal_index=goal_index_a,
    )


def compute_ratio_interval_for_candidate(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    alternative_index: int,
    goal_index_a: int,
    goal_index_b: int,
) -> RatioInterval:
    goal_count = alternatives.get_anzahl_spalten()
    if weight_space.variable_count != goal_count:
        raise ValueError(
            "weight_space must have the same number of variables as the number of goals"
        )

    _validate_goal_pair(
        goal_count=goal_count,
        goal_index_a=goal_index_a,
        goal_index_b=goal_index_b,
    )
    _validate_candidate_index(
        alternatives=alternatives,
        alternative_index=alternative_index,
    )

    ratio_normalized_weight_space = build_ratio_normalized_weight_space(
        weight_space=weight_space,
        normalization_goal_index=goal_index_b,
    )
    return _compute_ratio_interval_for_candidate_in_normalized_weight_space(
        alternatives=alternatives,
        normalized_weight_space=ratio_normalized_weight_space,
        alternative_index=alternative_index,
        goal_index_a=goal_index_a,
    )


def compute_ratio_intervals_for_pair(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidates: list[int],
    goal_index_a: int,
    goal_index_b: int,
) -> GoalPairRatioIntervals:
    goal_count = alternatives.get_anzahl_spalten()
    if weight_space.variable_count != goal_count:
        raise ValueError(
            "weight_space must have the same number of variables as the number of goals"
        )

    _validate_goal_pair(
        goal_count=goal_count,
        goal_index_a=goal_index_a,
        goal_index_b=goal_index_b,
    )

    ratio_normalized_weight_space = build_ratio_normalized_weight_space(
        weight_space=weight_space,
        normalization_goal_index=goal_index_b,
    )
    intervals_by_candidate: dict[int, RatioInterval] = {}

    for candidate_index in candidates:
        intervals_by_candidate[candidate_index] = (
            _compute_ratio_interval_for_candidate_in_normalized_weight_space(
                alternatives=alternatives,
                normalized_weight_space=ratio_normalized_weight_space,
                alternative_index=candidate_index,
                goal_index_a=goal_index_a,
            )
        )

    return GoalPairRatioIntervals(
        goal_index_a=goal_index_a,
        goal_index_b=goal_index_b,
        intervals_by_candidate=intervals_by_candidate,
    )


def compute_all_ratio_intervals(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidates: list[int],
    engine: RatioIntervalEngine = "geometry",
    geometry_tolerance: float = 1e-10,
) -> list[GoalPairRatioIntervals]:
    if weight_space.variable_count != alternatives.get_anzahl_spalten():
        raise ValueError(
            "weight_space must have the same number of variables as the number of goals"
        )

    if engine == "geometry":
        return _compute_all_ratio_intervals_from_vertices(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=candidates,
            tolerance=geometry_tolerance,
        )
    if engine != "lp":
        raise ValueError("engine must be 'geometry' or 'lp'")

    return _compute_all_ratio_intervals_with_lp(
        alternatives=alternatives,
        weight_space=weight_space,
        candidates=candidates,
    )


def _compute_all_ratio_intervals_with_lp(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidates: list[int],
) -> list[GoalPairRatioIntervals]:
    intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals] = {}
    for goal_index_a, goal_index_b in get_canonical_goal_pairs(
        alternatives.get_anzahl_spalten()
    ):
        direct_intervals = compute_ratio_intervals_for_pair(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=candidates,
            goal_index_a=goal_index_a,
            goal_index_b=goal_index_b,
        )
        intervals_by_goal_pair[(goal_index_a, goal_index_b)] = direct_intervals

        mirrored_intervals = invert_goal_pair_ratio_intervals(direct_intervals)
        if mirrored_intervals is None:
            # A pair can be infeasible only because its denominator is forced to
            # zero.  Its reverse may still be feasible, so retain an exact LP
            # fallback for this boundary case.
            mirrored_intervals = compute_ratio_intervals_for_pair(
                alternatives=alternatives,
                weight_space=weight_space,
                candidates=candidates,
                goal_index_a=goal_index_b,
                goal_index_b=goal_index_a,
            )
        intervals_by_goal_pair[(goal_index_b, goal_index_a)] = mirrored_intervals

    return [
        intervals_by_goal_pair[goal_pair]
        for goal_pair in get_ordered_goal_pairs(
            alternatives.get_anzahl_spalten()
        )
    ]


def _compute_all_ratio_intervals_from_vertices(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidates: list[int],
    tolerance: float,
) -> list[GoalPairRatioIntervals]:
    if tolerance <= 0.0:
        raise ValueError("geometry_tolerance must be positive")

    goal_count = alternatives.get_anzahl_spalten()
    intervals_by_pair_and_candidate: dict[
        tuple[int, int], dict[int, RatioInterval]
    ] = {
        pair: {} for pair in get_ordered_goal_pairs(goal_count)
    }
    for candidate_index in candidates:
        _validate_candidate_index(
            alternatives=alternatives,
            alternative_index=candidate_index,
        )
        optimality_region = build_optimality_region(
            alternatives=alternatives,
            weight_space=weight_space,
            alternative_index=candidate_index,
        )
        polytope = enumerate_polytope_vertices(
            system=optimality_region,
            tolerance=tolerance,
        )
        for goal_index_a, goal_index_b in get_ordered_goal_pairs(goal_count):
            if polytope.status == "infeasible":
                interval = _build_infeasible_ratio_interval(polytope.message)
            elif polytope.status in {"full_dimensional", "point"}:
                interval = _compute_ratio_interval_from_vertices(
                    vertices=polytope.vertices,
                    goal_index_a=goal_index_a,
                    goal_index_b=goal_index_b,
                    tolerance=tolerance,
                )
            else:
                # Exact fallback for degenerate or numerically difficult
                # regions.  The fast path is therefore an optimization only;
                # it does not weaken interval correctness.
                interval = compute_ratio_interval_for_candidate(
                    alternatives=alternatives,
                    weight_space=weight_space,
                    alternative_index=candidate_index,
                    goal_index_a=goal_index_a,
                    goal_index_b=goal_index_b,
                )
            intervals_by_pair_and_candidate[(goal_index_a, goal_index_b)][
                candidate_index
            ] = interval

    return [
        GoalPairRatioIntervals(
            goal_index_a=goal_index_a,
            goal_index_b=goal_index_b,
            intervals_by_candidate=intervals_by_pair_and_candidate[
                (goal_index_a, goal_index_b)
            ],
        )
        for goal_index_a, goal_index_b in get_ordered_goal_pairs(goal_count)
    ]


def _compute_ratio_interval_from_vertices(
    vertices: np.ndarray,
    goal_index_a: int,
    goal_index_b: int,
    tolerance: float,
) -> RatioInterval:
    numerators = vertices[:, goal_index_a]
    denominators = vertices[:, goal_index_b]
    positive_denominator = denominators > tolerance
    if not np.any(positive_denominator):
        return _build_infeasible_ratio_interval(
            "ratio denominator is zero throughout the candidate region"
        )

    ratios = numerators[positive_denominator] / denominators[positive_denominator]
    lower_value = max(0.0, float(np.min(ratios)))
    lower = LinearOptimizationResult(
        status="optimal",
        objective_sense="min",
        optimal_value=lower_value,
    )
    has_unbounded_vertex = bool(
        np.any((denominators <= tolerance) & (numerators > tolerance))
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
    return RatioInterval(lower=lower, upper=upper)


def _build_infeasible_ratio_interval(message: str | None) -> RatioInterval:
    return RatioInterval(
        lower=LinearOptimizationResult(
            status="infeasible",
            objective_sense="min",
            solver_message=message,
        ),
        upper=LinearOptimizationResult(
            status="infeasible",
            objective_sense="max",
            solver_message=message,
        ),
    )


def invert_goal_pair_ratio_intervals(
    goal_pair_intervals: GoalPairRatioIntervals,
) -> GoalPairRatioIntervals | None:
    mirrored_by_candidate: dict[int, RatioInterval] = {}
    for candidate_index, ratio_interval in (
        goal_pair_intervals.intervals_by_candidate.items()
    ):
        mirrored_interval = invert_ratio_interval(ratio_interval)
        if mirrored_interval is None:
            return None
        mirrored_by_candidate[int(candidate_index)] = mirrored_interval

    return GoalPairRatioIntervals(
        goal_index_a=goal_pair_intervals.goal_index_b,
        goal_index_b=goal_pair_intervals.goal_index_a,
        intervals_by_candidate=mirrored_by_candidate,
    )


def invert_ratio_interval(ratio_interval: RatioInterval) -> RatioInterval | None:
    """Return the interval for the reciprocal ratio when it is well-defined.

    ``None`` means that the reverse orientation must be solved directly.  This
    occurs when the original denominator cannot be normalized or when the
    complete interval is exactly zero.
    """

    if ratio_interval.lower.status != "optimal":
        return None

    lower_value = ratio_interval.lower.optimal_value
    if lower_value is None or lower_value < 0.0:
        return None

    if ratio_interval.upper.status == "unbounded":
        mirrored_lower_value = 0.0
    elif ratio_interval.upper.status == "optimal":
        upper_value = ratio_interval.upper.optimal_value
        if upper_value is None or upper_value <= 0.0:
            return None
        mirrored_lower_value = 1.0 / float(upper_value)
    else:
        return None

    mirrored_lower = LinearOptimizationResult(
        status="optimal",
        objective_sense="min",
        optimal_value=mirrored_lower_value,
    )
    if float(lower_value) == 0.0:
        mirrored_upper = LinearOptimizationResult(
            status="unbounded",
            objective_sense="max",
        )
    else:
        mirrored_upper = LinearOptimizationResult(
            status="optimal",
            objective_sense="max",
            optimal_value=1.0 / float(lower_value),
        )

    return RatioInterval(lower=mirrored_lower, upper=mirrored_upper)


def _compute_ratio_interval_for_candidate_in_normalized_weight_space(
    alternatives: AlternativenMatrix,
    normalized_weight_space: LinearConstraintSystem,
    alternative_index: int,
    goal_index_a: int,
) -> RatioInterval:
    goal_count = alternatives.get_anzahl_spalten()
    if normalized_weight_space.variable_count != goal_count:
        raise ValueError(
            "normalized_weight_space must have the same number of variables "
            "as the number of goals"
        )

    if not 0 <= goal_index_a < goal_count:
        raise IndexError("goal_index_a is out of range")

    _validate_candidate_index(
        alternatives=alternatives,
        alternative_index=alternative_index,
    )
    optimality_region = build_optimality_region(
        alternatives=alternatives,
        weight_space=normalized_weight_space,
        alternative_index=alternative_index,
    )
    return _compute_goal_index_interval(
        system=optimality_region,
        goal_count=goal_count,
        goal_index=goal_index_a,
    )


def _compute_goal_index_interval(
    system: LinearConstraintSystem,
    goal_count: int,
    goal_index: int,
) -> RatioInterval:
    objective = [0.0] * goal_count
    objective[goal_index] = 1.0

    return RatioInterval(
        lower=system.minimize(objective),
        upper=system.maximize(objective),
    )


def _validate_goal_pair(
    goal_count: int,
    goal_index_a: int,
    goal_index_b: int,
) -> None:
    if goal_count <= 0:
        raise ValueError("goal_count must be positive")

    if not 0 <= goal_index_a < goal_count:
        raise IndexError("goal_index_a is out of range")

    if not 0 <= goal_index_b < goal_count:
        raise IndexError("goal_index_b is out of range")

    if goal_index_a == goal_index_b:
        raise ValueError("goal_index_a and goal_index_b must be different")


def _validate_candidate_index(
    alternatives: AlternativenMatrix,
    alternative_index: int,
) -> None:
    if not 0 <= alternative_index < alternatives.get_anzahl_zeilen():
        raise IndexError("alternative_index is out of range")
