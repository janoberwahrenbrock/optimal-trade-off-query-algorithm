from __future__ import annotations

from .alternative_utility import build_utility_difference_coefficients
from .linear_constraints import LinearConstraintSystem
from .models import AlternativenMatrix


def build_optimality_region(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    alternative_index: int,
) -> LinearConstraintSystem:
    if not 0 <= alternative_index < alternatives.get_anzahl_zeilen():
        raise IndexError("alternative_index is out of range")

    goal_count = alternatives.get_anzahl_spalten()
    if weight_space.variable_count not in {0, goal_count}:
        raise ValueError(
            "weight_space must have the same number of variables as the number of goals"
        )

    optimality_region = LinearConstraintSystem()
    optimality_region.add_constraint_system(weight_space)
    _add_optimality_constraints(
        system=optimality_region,
        alternatives=alternatives,
        alternative_index=alternative_index,
    )
    return optimality_region


def _add_optimality_constraints(
    system: LinearConstraintSystem,
    alternatives: AlternativenMatrix,
    alternative_index: int,
) -> None:
    for other_index in range(alternatives.get_anzahl_zeilen()):
        if other_index == alternative_index:
            continue

        left_side = build_utility_difference_coefficients(
            alternatives=alternatives,
            minuend_index=other_index,
            subtrahend_index=alternative_index,
        )
        system.add_inequality(left_side, 0.0)
