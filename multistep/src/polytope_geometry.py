from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import null_space
from scipy.spatial import HalfspaceIntersection, QhullError

from .linear_constraints import LinearConstraintSystem
from .linear_programming import (
    LINPROG_OPTIONS,
    classify_linprog_failure,
    run_linprog_with_retries,
)


PolytopeStatus = Literal[
    "full_dimensional",
    "point",
    "lower_dimensional",
    "infeasible",
    "error",
]


@dataclass(frozen=True)
class PolytopeVertices:
    """Vertices of a bounded polytope in its equality-defined affine hull."""

    status: PolytopeStatus
    vertices: np.ndarray
    affine_dimension: int
    interior_radius: float | None = None
    message: str | None = None


def enumerate_polytope_vertices(
    system: LinearConstraintSystem,
    tolerance: float = 1e-10,
) -> PolytopeVertices:
    """Enumerate vertices after eliminating all explicit equalities.

    ``HalfspaceIntersection`` needs a strict interior point.  We obtain it from
    a relative Chebyshev-center LP.  Degenerate regions are reported so callers
    can retain an exact LP fallback instead of trusting numerically incomplete
    geometry.
    """

    if system.variable_count <= 0:
        return _empty_result("infeasible", 0, "system has no variables")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    inequality_matrix, inequality_right_side, equality_matrix, equality_right_side = (
        system.get_solver_matrices()
    )
    variable_count = system.variable_count
    if equality_matrix is None:
        particular_point = np.zeros(variable_count, dtype=float)
        affine_basis = np.eye(variable_count, dtype=float)
    else:
        assert equality_right_side is not None
        particular_point, *_ = np.linalg.lstsq(
            equality_matrix,
            equality_right_side,
            rcond=None,
        )
        equality_residual = equality_matrix @ particular_point - equality_right_side
        if np.max(np.abs(equality_residual), initial=0.0) > 10.0 * tolerance:
            return _empty_result("infeasible", 0, "inconsistent equalities")
        affine_basis = null_space(equality_matrix)

    affine_dimension = int(affine_basis.shape[1])
    if inequality_matrix is None:
        return _empty_result(
            "error",
            affine_dimension,
            "vertex enumeration requires a bounded inequality system",
        )
    assert inequality_right_side is not None

    reduced_matrix = inequality_matrix @ affine_basis
    reduced_right_side = inequality_right_side - inequality_matrix @ particular_point
    projected_norms = np.linalg.norm(reduced_matrix, axis=1)
    constant_rows = projected_norms <= tolerance
    if np.any(reduced_right_side[constant_rows] < -tolerance):
        return _empty_result(
            "infeasible",
            affine_dimension,
            "an affine-hull constant inequality is violated",
        )
    reduced_matrix = reduced_matrix[~constant_rows]
    reduced_right_side = reduced_right_side[~constant_rows]
    projected_norms = projected_norms[~constant_rows]

    if affine_dimension == 0:
        return PolytopeVertices(
            status="point",
            vertices=particular_point.reshape(1, variable_count),
            affine_dimension=0,
            interior_radius=0.0,
        )
    if len(reduced_matrix) == 0:
        return _empty_result(
            "error",
            affine_dimension,
            "polytope is unbounded in its affine hull",
        )

    center_result = run_linprog_with_retries(
        c=np.concatenate((np.zeros(affine_dimension), [-1.0])),
        A_ub=np.column_stack((reduced_matrix, projected_norms)),
        b_ub=reduced_right_side,
        bounds=[(None, None)] * affine_dimension + [(0.0, None)],
        method="highs",
        options=LINPROG_OPTIONS,
    )
    if not center_result.success or center_result.x is None:
        status = classify_linprog_failure(
            solver_status_code=int(center_result.status),
            solver_message=str(center_result.message),
        )
        if status == "infeasible":
            return _empty_result("infeasible", affine_dimension, str(center_result.message))
        return _empty_result("error", affine_dimension, str(center_result.message))

    center = np.asarray(center_result.x[:-1], dtype=float)
    radius = max(0.0, float(center_result.x[-1]))
    if radius <= tolerance:
        return PolytopeVertices(
            status="lower_dimensional",
            vertices=np.empty((0, variable_count), dtype=float),
            affine_dimension=affine_dimension,
            interior_radius=radius,
            message="inequalities reduce the dimension beyond explicit equalities",
        )

    try:
        if affine_dimension == 1:
            reduced_vertices = _enumerate_interval_vertices(
                reduced_matrix=reduced_matrix,
                reduced_right_side=reduced_right_side,
                tolerance=tolerance,
            )
        else:
            halfspaces = np.column_stack((reduced_matrix, -reduced_right_side))
            intersection = HalfspaceIntersection(halfspaces, center)
            reduced_vertices = np.asarray(intersection.intersections, dtype=float)
    except (QhullError, ValueError, RuntimeError) as exc:
        return _empty_result("error", affine_dimension, str(exc), radius)

    if len(reduced_vertices) == 0:
        return _empty_result(
            "error",
            affine_dimension,
            "vertex enumeration returned no vertices",
            radius,
        )
    vertices = particular_point + reduced_vertices @ affine_basis.T
    vertices = _deduplicate_rows(vertices, tolerance=tolerance)
    if not _vertices_satisfy_system(
        vertices=vertices,
        inequality_matrix=inequality_matrix,
        inequality_right_side=inequality_right_side,
        equality_matrix=equality_matrix,
        equality_right_side=equality_right_side,
        tolerance=tolerance,
    ):
        return _empty_result(
            "error",
            affine_dimension,
            "enumerated vertices violate the constraint system",
            radius,
        )
    return PolytopeVertices(
        status="full_dimensional",
        vertices=vertices,
        affine_dimension=affine_dimension,
        interior_radius=radius,
    )


def _enumerate_interval_vertices(
    reduced_matrix: np.ndarray,
    reduced_right_side: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    lower = -np.inf
    upper = np.inf
    for coefficient, right_side in zip(
        reduced_matrix[:, 0],
        reduced_right_side,
    ):
        if coefficient > tolerance:
            upper = min(upper, float(right_side / coefficient))
        elif coefficient < -tolerance:
            lower = max(lower, float(right_side / coefficient))
        elif right_side < -tolerance:
            raise ValueError("infeasible constant inequality")
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("polytope is unbounded")
    if lower > upper + tolerance:
        raise ValueError("polytope is infeasible")
    if abs(upper - lower) <= tolerance:
        raise ValueError("polytope is lower-dimensional")
    return np.asarray([[lower], [upper]], dtype=float)


def _deduplicate_rows(values: np.ndarray, tolerance: float) -> np.ndarray:
    decimals = max(0, int(-np.log10(tolerance)))
    _, indices = np.unique(np.round(values, decimals=decimals), axis=0, return_index=True)
    return values[np.sort(indices)]


def _vertices_satisfy_system(
    vertices: np.ndarray,
    inequality_matrix: np.ndarray,
    inequality_right_side: np.ndarray,
    equality_matrix: np.ndarray | None,
    equality_right_side: np.ndarray | None,
    tolerance: float,
) -> bool:
    validation_tolerance = max(1e-8, 100.0 * tolerance)
    inequality_residuals = (
        inequality_matrix @ vertices.T - inequality_right_side[:, None]
    )
    if np.any(inequality_residuals > validation_tolerance):
        return False
    if equality_matrix is not None and equality_right_side is not None:
        if np.any(
            np.abs(equality_matrix @ vertices.T - equality_right_side[:, None])
            > validation_tolerance
        ):
            return False
    return True


def _empty_result(
    status: PolytopeStatus,
    affine_dimension: int,
    message: str,
    interior_radius: float | None = None,
) -> PolytopeVertices:
    return PolytopeVertices(
        status=status,
        vertices=np.empty((0, 0), dtype=float),
        affine_dimension=affine_dimension,
        interior_radius=interior_radius,
        message=message,
    )
