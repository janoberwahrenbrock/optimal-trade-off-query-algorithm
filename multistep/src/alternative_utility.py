from __future__ import annotations

from .linear_constraints import Vector
from .models import AlternativenMatrix


def compute_utility_value(
    alternatives: AlternativenMatrix,
    alternative_index: int,
    weights: list[float],
) -> float:
    if not 0 <= alternative_index < alternatives.get_anzahl_zeilen():
        raise IndexError("alternative_index is out of range")

    if len(weights) != alternatives.get_anzahl_spalten():
        raise ValueError("weights must match the number of goals")

    alternative = alternatives.get_alternative(alternative_index)
    return sum(
        utility_value * weight
        for utility_value, weight in zip(alternative, weights)
    )


def build_utility_difference_coefficients(
    alternatives: AlternativenMatrix,
    minuend_index: int,
    subtrahend_index: int,
) -> Vector:
    if not 0 <= minuend_index < alternatives.get_anzahl_zeilen():
        raise IndexError("minuend_index is out of range")

    if not 0 <= subtrahend_index < alternatives.get_anzahl_zeilen():
        raise IndexError("subtrahend_index is out of range")

    minuend = alternatives.get_alternative(minuend_index)
    subtrahend = alternatives.get_alternative(subtrahend_index)
    return [
        minuend_value - subtrahend_value
        for minuend_value, subtrahend_value in zip(minuend, subtrahend)
    ]
