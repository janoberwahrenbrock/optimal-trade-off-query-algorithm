from __future__ import annotations

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from .linear_constraints import LinearConstraintSystem
from .models import AlternativenMatrix
from .models.linear_optimization_result import LinearOptimizationResult
from .optimality_region import build_optimality_region
from .weight_space import build_ratio_normalized_weight_space


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
) -> list[GoalPairRatioIntervals]:
    if weight_space.variable_count != alternatives.get_anzahl_spalten():
        raise ValueError(
            "weight_space must have the same number of variables as the number of goals"
        )

    return [
        compute_ratio_intervals_for_pair(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=candidates,
            goal_index_a=goal_index_a,
            goal_index_b=goal_index_b,
        )
        for goal_index_a, goal_index_b in get_ordered_goal_pairs(
            alternatives.get_anzahl_spalten()
        )
    ]


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
