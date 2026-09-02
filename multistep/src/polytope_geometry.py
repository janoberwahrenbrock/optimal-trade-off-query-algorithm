from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
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
            reduced_vertices = _compute_halfspace_intersections_with_retries(
                halfspaces=halfspaces,
                center=center,
                affine_dimension=affine_dimension,
            )
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


def _compute_halfspace_intersections_with_retries(
    halfspaces: np.ndarray,
    center: np.ndarray,
    affine_dimension: int,
) -> np.ndarray:
    """Retry Qhull with increasingly permissive precision recovery.

    High-dimensional posterior polytopes can become nearly degenerate after
    many answered queries.  Q12 accepts wide facets caused by merge roundoff;
    QJ additionally joggles the dual input as a last resort.  The caller still
    validates every returned vertex against the original constraint system.
    """

    coefficients = np.asarray(halfspaces[:, :-1], dtype=float)
    offsets = np.asarray(halfspaces[:, -1], dtype=float)
    coefficient_norms = np.linalg.norm(coefficients, axis=1)
    if np.any(coefficient_norms <= 0.0):
        raise ValueError("halfspace coefficients must be non-zero")

    # Qhull constructs a dual hull around the supplied interior point.  Deep
    # posterior states can have coefficient scales and slacks that differ by
    # many orders of magnitude.  Expressing y = center + radius * z makes the
    # interior point the origin, the nearest supporting plane unit distance
    # away, and every normal unit length.  This is an affine coordinate change
    # only; intersections are transformed back and validated below.
    normalized_coefficients = coefficients / coefficient_norms[:, None]
    normalized_offsets = offsets / coefficient_norms
    center_offsets = normalized_coefficients @ center + normalized_offsets
    slacks = -center_offsets
    conditioning_radius = float(np.min(slacks, initial=np.inf))
    if not np.isfinite(conditioning_radius) or conditioning_radius <= 0.0:
        raise ValueError("halfspace center is not strictly interior")
    conditioned_halfspaces = np.column_stack(
        (
            normalized_coefficients,
            center_offsets / conditioning_radius,
        )
    )
    conditioned_center = np.zeros_like(center)

    exact_merge = "Qx " if affine_dimension > 4 else ""
    options = (None, f"{exact_merge}Q12".strip(), f"{exact_merge}Q12 QJ".strip())
    errors: list[str] = []
    for qhull_options in options:
        try:
            intersection = HalfspaceIntersection(
                conditioned_halfspaces,
                conditioned_center,
                qhull_options=qhull_options,
            )
            conditioned_intersections = np.asarray(
                intersection.intersections,
                dtype=float,
            )
            intersections = (
                center + conditioning_radius * conditioned_intersections
            )
            if not _points_satisfy_halfspaces(
                points=intersections,
                halfspaces=halfspaces,
            ):
                intersections = _snap_intersections_to_halfspace_vertices(
                    points=intersections,
                    halfspaces=halfspaces,
                    affine_dimension=affine_dimension,
                )
                if len(intersections) == 0 or not _points_satisfy_halfspaces(
                    points=intersections,
                    halfspaces=halfspaces,
                ):
                    errors.append(
                        f"Qhull options {qhull_options!r} returned invalid intersections"
                    )
                    continue
            return intersections
        except (QhullError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors))


def _points_satisfy_halfspaces(
    points: np.ndarray,
    halfspaces: np.ndarray,
) -> bool:
    """Validate ``a @ x + b <= 0`` with scale-aware roundoff bounds."""

    if points.ndim != 2 or halfspaces.ndim != 2:
        return False
    if len(points) == 0 or halfspaces.shape[1] != points.shape[1] + 1:
        return False
    coefficients = halfspaces[:, :-1]
    offsets = halfspaces[:, -1]
    residuals = coefficients @ points.T + offsets[:, None]
    row_norms = np.linalg.norm(coefficients, axis=1)[:, None]
    point_norms = np.linalg.norm(points, axis=1)[None, :]
    scales = np.maximum(
        1.0,
        row_norms * point_norms + np.abs(offsets)[:, None],
    )
    return not np.any(residuals > 1e-8 * scales)


def _snap_intersections_to_halfspace_vertices(
    points: np.ndarray,
    halfspaces: np.ndarray,
    affine_dimension: int,
) -> np.ndarray:
    """Recover slightly perturbed Qhull vertices from active constraints.

    Qhull may return a topologically correct intersection whose coordinates
    are just outside a nearly degenerate facet.  For each invalid point, solve
    the closest independent supporting planes and accept the reconstruction
    only if it satisfies every original halfspace and moves by at most a small
    relative amount.  This does not turn arbitrary invalid points into valid
    ones and therefore preserves the retry/failure behavior for real errors.
    """

    coefficients = np.asarray(halfspaces[:, :-1], dtype=float)
    right_side = -np.asarray(halfspaces[:, -1], dtype=float)
    coefficient_norms = np.linalg.norm(coefficients, axis=1)
    recovered: list[np.ndarray] = []
    for point in np.asarray(points, dtype=float):
        if _points_satisfy_halfspaces(point.reshape(1, -1), halfspaces):
            recovered.append(point)
            continue

        distances = np.abs(coefficients @ point - right_side) / coefficient_norms
        ordered_indices = np.argsort(distances)
        candidate = _snap_point_to_nearby_facets(
            point=point,
            coefficients=coefficients,
            right_side=right_side,
            ordered_indices=ordered_indices,
            halfspaces=halfspaces,
            affine_dimension=affine_dimension,
        )
        if candidate is None:
            return np.empty((0, affine_dimension), dtype=float)
        recovered.append(candidate)

    return _deduplicate_rows(np.asarray(recovered), tolerance=1e-10)


def _snap_point_to_nearby_facets(
    point: np.ndarray,
    coefficients: np.ndarray,
    right_side: np.ndarray,
    ordered_indices: np.ndarray,
    halfspaces: np.ndarray,
    affine_dimension: int,
) -> np.ndarray | None:
    selected: list[int] = []
    current_rank = 0
    for raw_index in ordered_indices:
        index = int(raw_index)
        trial = selected + [index]
        trial_rank = int(np.linalg.matrix_rank(coefficients[trial]))
        if trial_rank > current_rank:
            selected.append(index)
            current_rank = trial_rank
        if current_rank == affine_dimension:
            candidate = _solve_and_validate_snapped_vertex(
                point=point,
                active_indices=selected,
                coefficients=coefficients,
                right_side=right_side,
                halfspaces=halfspaces,
            )
            if candidate is not None:
                return candidate
            break

    nearby_count = min(len(ordered_indices), affine_dimension + 5)
    nearby = [int(index) for index in ordered_indices[:nearby_count]]
    for active_indices in combinations(nearby, affine_dimension):
        if np.linalg.matrix_rank(coefficients[list(active_indices)]) < affine_dimension:
            continue
        candidate = _solve_and_validate_snapped_vertex(
            point=point,
            active_indices=list(active_indices),
            coefficients=coefficients,
            right_side=right_side,
            halfspaces=halfspaces,
        )
        if candidate is not None:
            return candidate
    return None


def _solve_and_validate_snapped_vertex(
    point: np.ndarray,
    active_indices: list[int],
    coefficients: np.ndarray,
    right_side: np.ndarray,
    halfspaces: np.ndarray,
) -> np.ndarray | None:
    candidate, *_ = np.linalg.lstsq(
        coefficients[active_indices],
        right_side[active_indices],
        rcond=None,
    )
    if not _points_satisfy_halfspaces(candidate.reshape(1, -1), halfspaces):
        return None
    displacement = float(np.linalg.norm(candidate - point))
    scale = max(1.0, float(np.linalg.norm(point)), float(np.linalg.norm(candidate)))
    if displacement > 1e-5 * scale:
        return None
    return np.asarray(candidate, dtype=float)


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
    vertex_norms = np.linalg.norm(vertices, axis=1)[None, :]
    inequality_scales = np.maximum(
        1.0,
        np.linalg.norm(inequality_matrix, axis=1)[:, None] * vertex_norms
        + np.abs(inequality_right_side)[:, None],
    )
    if np.any(inequality_residuals > validation_tolerance * inequality_scales):
        return False
    if equality_matrix is not None and equality_right_side is not None:
        equality_residuals = np.abs(
            equality_matrix @ vertices.T - equality_right_side[:, None]
        )
        equality_scales = np.maximum(
            1.0,
            np.linalg.norm(equality_matrix, axis=1)[:, None] * vertex_norms
            + np.abs(equality_right_side)[:, None],
        )
        if np.any(equality_residuals > validation_tolerance * equality_scales):
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
