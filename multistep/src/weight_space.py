from __future__ import annotations

from .linear_constraints import LinearConstraintSystem, Vector
from .models import AnsweredQuery


def build_weight_space(
    goal_count: int,
    answered_queries: list[AnsweredQuery],
) -> LinearConstraintSystem:
    if goal_count <= 0:
        raise ValueError("goal_count must be positive")

    system = LinearConstraintSystem()
    _add_nonnegativity_constraints(system, goal_count)
    system.add_equality([1.0] * goal_count, 1.0)
    _add_answered_query_constraints(system, answered_queries, goal_count)
    return system


def build_normalized_weight_space(
    goal_count: int,
    answered_queries: list[AnsweredQuery],
    normalization_goal_index: int,
) -> LinearConstraintSystem:
    if goal_count <= 0:
        raise ValueError("goal_count must be positive")

    if not 0 <= normalization_goal_index < goal_count:
        raise IndexError("normalization_goal_index is out of range")

    system = LinearConstraintSystem()
    _add_nonnegativity_constraints(system, goal_count)

    normalization_row = [0.0] * goal_count
    normalization_row[normalization_goal_index] = 1.0
    system.add_equality(normalization_row, 1.0)

    _add_answered_query_constraints(system, answered_queries, goal_count)
    return system


def build_ratio_normalized_weight_space(
    weight_space: LinearConstraintSystem,
    normalization_goal_index: int,
    tolerance: float = 1e-12,
) -> LinearConstraintSystem:
    if weight_space.variable_count <= 0:
        raise ValueError("weight_space must have at least one variable")

    if not 0 <= normalization_goal_index < weight_space.variable_count:
        raise IndexError("normalization_goal_index is out of range")

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    normalized_system = LinearConstraintSystem()

    for left_side, right_side in zip(
        weight_space.inequalities_left_side,
        weight_space.inequalities_right_side,
    ):
        if abs(right_side) > tolerance:
            raise ValueError("ratio normalization requires homogeneous inequalities")

        normalized_system.add_inequality(left_side, right_side)

    for left_side, right_side in zip(
        weight_space.equalities_left_side,
        weight_space.equalities_right_side,
    ):
        if _is_simplex_normalization_equality(
            left_side=left_side,
            right_side=right_side,
            tolerance=tolerance,
        ):
            continue

        if abs(right_side) > tolerance:
            raise ValueError("ratio normalization requires homogeneous equalities")

        normalized_system.add_equality(left_side, right_side)

    normalization_row = [0.0] * weight_space.variable_count
    normalization_row[normalization_goal_index] = 1.0
    normalized_system.add_equality(normalization_row, 1.0)
    return normalized_system


def build_answered_query_constraint(
    answered_query: AnsweredQuery,
    goal_count: int,
) -> tuple[Vector, float, bool]:
    _validate_answered_query(answered_query, goal_count)

    left_side = [0.0] * goal_count
    goal_index_a = int(answered_query.ziel_index_a)
    goal_index_b = int(answered_query.ziel_index_b)
    value = float(answered_query.value)

    if answered_query.operator == ">":
        left_side[goal_index_a] = -1.0
        left_side[goal_index_b] = value
        return _scale_constraint(left_side, 0.0, False)

    if answered_query.operator == "<":
        left_side[goal_index_a] = 1.0
        left_side[goal_index_b] = -value
        return _scale_constraint(left_side, 0.0, False)

    if answered_query.operator == "=":
        left_side[goal_index_a] = 1.0
        left_side[goal_index_b] = -value
        return _scale_constraint(left_side, 0.0, True)

    raise ValueError(f"unknown operator: {answered_query.operator}")


def _add_nonnegativity_constraints(
    system: LinearConstraintSystem,
    goal_count: int,
) -> None:
    for goal_index in range(goal_count):
        left_side = [0.0] * goal_count
        left_side[goal_index] = -1.0
        system.add_inequality(left_side, 0.0)


def _add_answered_query_constraints(
    system: LinearConstraintSystem,
    answered_queries: list[AnsweredQuery],
    goal_count: int,
) -> None:
    for answered_query in answered_queries:
        left_side, right_side, is_equality = build_answered_query_constraint(
            answered_query=answered_query,
            goal_count=goal_count,
        )

        if is_equality:
            system.add_equality(left_side, right_side)
        else:
            system.add_inequality(left_side, right_side)


def _validate_answered_query(
    answered_query: AnsweredQuery,
    goal_count: int,
) -> None:
    if goal_count <= 0:
        raise ValueError("goal_count must be positive")

    if answered_query.ziel_index_a >= goal_count:
        raise IndexError("answered_query.ziel_index_a is out of range")

    if answered_query.ziel_index_b >= goal_count:
        raise IndexError("answered_query.ziel_index_b is out of range")


def _scale_constraint(
    left_side: Vector,
    right_side: float,
    is_equality: bool,
) -> tuple[Vector, float, bool]:
    scale = max(abs(value) for value in left_side)
    if scale == 0.0:
        return left_side, right_side, is_equality

    return (
        [value / scale for value in left_side],
        right_side / scale,
        is_equality,
    )


def _is_simplex_normalization_equality(
    left_side: Vector,
    right_side: float,
    tolerance: float,
) -> bool:
    return abs(right_side - 1.0) <= tolerance and all(
        abs(value - 1.0) <= tolerance for value in left_side
    )
