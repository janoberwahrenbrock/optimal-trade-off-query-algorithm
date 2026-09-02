from __future__ import annotations

"""Numerically exact intrinsic volumes for bounded linear polytopes."""

from functools import lru_cache

import numpy as np
from scipy.linalg import null_space
from scipy.spatial import ConvexHull, QhullError

from .linear_constraints import LinearConstraintSystem
from .models import Query, QueryOperator
from .polytope_geometry import PolytopeVertices, enumerate_polytope_vertices


ConstraintSystemKey = tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
]


def compute_polytope_intrinsic_volume(
    system: LinearConstraintSystem,
    tolerance: float = 1e-10,
) -> float:
    """Return volume in the affine hull defined by explicit equalities.

    Empty and inequality-degenerate regions have zero volume relative to that
    affine hull.  Results are cached by the immutable constraint signature so
    recursive value-function branches can reuse identical state volumes.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    return _compute_polytope_intrinsic_volume_cached(
        _constraint_system_key(system),
        float(tolerance),
    )


def compute_polytope_intrinsic_volume_from_vertices(
    system: LinearConstraintSystem,
    polytope: PolytopeVertices,
) -> float:
    """Return intrinsic volume while reusing already enumerated vertices."""

    if polytope.status in {"infeasible", "lower_dimensional", "point"}:
        return 0.0
    if polytope.status != "full_dimensional":
        raise RuntimeError(polytope.message or f"polytope status {polytope.status}")

    vertices = np.asarray(polytope.vertices, dtype=float)
    _, _, equality_matrix, equality_right_side = system.get_solver_matrices()
    if equality_matrix is None:
        particular_point = np.zeros(system.variable_count, dtype=float)
        basis = np.eye(system.variable_count, dtype=float)
    else:
        assert equality_right_side is not None
        particular_point, *_ = np.linalg.lstsq(
            equality_matrix,
            equality_right_side,
            rcond=None,
        )
        basis = null_space(equality_matrix)
    coordinates = (vertices - particular_point) @ basis
    dimension = coordinates.shape[1]
    if dimension == 0:
        return 0.0
    if dimension == 1:
        return float(np.ptp(coordinates[:, 0]))
    try:
        return float(ConvexHull(coordinates).volume)
    except QhullError:
        return float(ConvexHull(coordinates, qhull_options="QJ").volume)


def compute_exact_query_answer_probabilities(
    weight_space: LinearConstraintSystem,
    query: Query,
    tolerance: float = 1e-10,
) -> dict[QueryOperator, float]:
    """Compute exact query-branch probabilities from intrinsic volumes."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if query.ziel_index_a >= weight_space.variable_count:
        raise IndexError("query.ziel_index_a is out of range")
    if query.ziel_index_b >= weight_space.variable_count:
        raise IndexError("query.ziel_index_b is out of range")

    objective = [0.0] * weight_space.variable_count
    objective[int(query.ziel_index_a)] = 1.0
    objective[int(query.ziel_index_b)] = -float(query.value)
    lower = weight_space.minimize(objective)
    upper = weight_space.maximize(objective)
    if lower.status != "optimal" or upper.status != "optimal":
        raise RuntimeError("cannot determine exact query support")
    if lower.optimal_value is None or upper.optimal_value is None:
        raise RuntimeError("query support optimization has no value")

    lower_value = float(lower.optimal_value)
    upper_value = float(upper.optimal_value)
    if lower_value >= -tolerance and upper_value <= tolerance:
        return {"<": 0.0, "=": 1.0, ">": 0.0}
    if lower_value >= -tolerance:
        return {"<": 0.0, "=": 0.0, ">": 1.0}
    if upper_value <= tolerance:
        return {"<": 1.0, "=": 0.0, ">": 0.0}

    total_volume = compute_polytope_intrinsic_volume(
        weight_space,
        tolerance=tolerance,
    )
    if total_volume <= 0.0:
        raise RuntimeError("weight space has no positive intrinsic volume")

    less_system = _copy_constraint_system(weight_space)
    less_system.add_inequality(objective, 0.0)
    greater_system = _copy_constraint_system(weight_space)
    greater_system.add_inequality([-value for value in objective], 0.0)
    less_volume = compute_polytope_intrinsic_volume(
        less_system,
        tolerance=tolerance,
    )
    greater_volume = compute_polytope_intrinsic_volume(
        greater_system,
        tolerance=tolerance,
    )
    raw = np.asarray([less_volume, greater_volume], dtype=float) / total_volume
    raw = np.maximum(raw, 0.0)
    probability_sum = float(np.sum(raw))
    if probability_sum <= 0.0:
        raise RuntimeError("exact query branches have zero total probability")
    # The shared cutting hyperplane has volume zero.  Renormalization only
    # removes floating-point accumulation from two independent hull volumes.
    raw /= probability_sum
    return {
        "<": float(raw[0]),
        "=": 0.0,
        ">": float(raw[1]),
    }


def clear_polytope_volume_cache() -> None:
    _compute_polytope_intrinsic_volume_cached.cache_clear()


def polytope_volume_cache_info():
    return _compute_polytope_intrinsic_volume_cached.cache_info()


@lru_cache(maxsize=32_768)
def _compute_polytope_intrinsic_volume_cached(
    key: ConstraintSystemKey,
    tolerance: float,
) -> float:
    system = _constraint_system_from_key(key)
    polytope = enumerate_polytope_vertices(system, tolerance=tolerance)
    return compute_polytope_intrinsic_volume_from_vertices(system, polytope)


def _constraint_system_key(system: LinearConstraintSystem) -> ConstraintSystemKey:
    return (
        tuple(tuple(float(value) for value in row) for row in system.inequalities_left_side),
        tuple(float(value) for value in system.inequalities_right_side),
        tuple(tuple(float(value) for value in row) for row in system.equalities_left_side),
        tuple(float(value) for value in system.equalities_right_side),
    )


def _constraint_system_from_key(key: ConstraintSystemKey) -> LinearConstraintSystem:
    inequalities, inequality_right_side, equalities, equality_right_side = key
    return LinearConstraintSystem(
        inequalities_left_side=[list(row) for row in inequalities],
        inequalities_right_side=list(inequality_right_side),
        equalities_left_side=[list(row) for row in equalities],
        equalities_right_side=list(equality_right_side),
    )


def _copy_constraint_system(system: LinearConstraintSystem) -> LinearConstraintSystem:
    copied = LinearConstraintSystem()
    copied.add_constraint_system(system)
    return copied
