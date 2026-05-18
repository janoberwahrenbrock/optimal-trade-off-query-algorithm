from __future__ import annotations

from .alternative_utility import compute_utility_value
from .linear_constraints import LinearConstraintSystem
from .models import AlternativenMatrix
from .optimality_region import build_optimality_region


def compute_candidate_set(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
) -> list[int]:
    candidates: list[int] = []

    for alternative_index in range(alternatives.get_anzahl_zeilen()):
        optimality_region = build_optimality_region(
            alternatives=alternatives,
            weight_space=weight_space,
            alternative_index=alternative_index,
        )
        if optimality_region.is_feasible():
            candidates.append(alternative_index)

    return candidates


def estimate_candidate_set_from_samples(
    alternatives: AlternativenMatrix,
    samples: list[list[float]],
    utility_tol: float = 1e-9,
) -> list[int]:
    if not samples:
        raise ValueError("samples must not be empty")

    if utility_tol < 0:
        raise ValueError("utility_tol must not be negative")

    goal_count = alternatives.get_anzahl_spalten()
    candidate_indices: set[int] = set()

    for sample in samples:
        if len(sample) != goal_count:
            raise ValueError("all samples must match the number of goals")

        utility_values = [
            compute_utility_value(
                alternatives=alternatives,
                alternative_index=alternative_index,
                weights=sample,
            )
            for alternative_index in range(alternatives.get_anzahl_zeilen())
        ]
        max_utility_value = max(utility_values)

        for alternative_index, utility_value in enumerate(utility_values):
            if utility_value >= max_utility_value - utility_tol:
                candidate_indices.add(alternative_index)

    return sorted(candidate_indices)

