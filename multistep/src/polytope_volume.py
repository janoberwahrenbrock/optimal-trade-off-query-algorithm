from __future__ import annotations

"""Numerically exact intrinsic volumes for bounded linear polytopes."""

from collections import OrderedDict
from functools import lru_cache
from threading import RLock
from typing import NamedTuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .affine_geometry import (
    AffineCacheInfo,
    affine_geometry_cache_info,
    clear_affine_geometry_cache,
    get_affine_decomposition,
)
from .linear_constraints import LinearConstraintSystem
from .models import Query, QueryOperator
from .polytope_geometry import PolytopeVertices, enumerate_polytope_vertices


ConstraintSystemKey = tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
]


_VERTEX_CACHE_MAX_SIZE = 32_768
_VERTEX_CACHE_MAX_BYTES = 64 * 1024 * 1024
_vertex_cache: OrderedDict[tuple[ConstraintSystemKey, float], PolytopeVertices] = (
    OrderedDict()
)
_vertex_cache_lock = RLock()
_vertex_cache_bytes = 0
_vertex_cache_hits = 0
_vertex_cache_misses = 0
_vertex_support_hits = 0
_vertex_support_fallbacks = 0


class PolytopeVertexCacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int
    currsize: int
    maxbytes: int
    currbytes: int
    support_hits: int
    support_fallbacks: int


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
    affine = get_affine_decomposition(
        system.variable_count, equality_matrix, equality_right_side
    )
    coordinates = (vertices - affine.particular_point) @ affine.basis
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
    support = _query_support_from_cached_vertices(weight_space, objective, tolerance)
    if support is None:
        lower = weight_space.minimize(objective)
        upper = weight_space.maximize(objective)
        if lower.status != "optimal" or upper.status != "optimal":
            raise RuntimeError("cannot determine exact query support")
        if lower.optimal_value is None or upper.optimal_value is None:
            raise RuntimeError("query support optimization has no value")
        lower_value = float(lower.optimal_value)
        upper_value = float(upper.optimal_value)
    else:
        lower_value, upper_value = support
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
    global _vertex_cache_bytes, _vertex_cache_hits, _vertex_cache_misses
    global _vertex_support_hits, _vertex_support_fallbacks
    _compute_polytope_intrinsic_volume_cached.cache_clear()
    clear_affine_geometry_cache()
    with _vertex_cache_lock:
        _vertex_cache.clear()
        _vertex_cache_bytes = 0
        _vertex_cache_hits = 0
        _vertex_cache_misses = 0
        _vertex_support_hits = 0
        _vertex_support_fallbacks = 0


def polytope_volume_cache_info():
    return _compute_polytope_intrinsic_volume_cached.cache_info()


def polytope_affine_cache_info() -> AffineCacheInfo:
    return affine_geometry_cache_info()


def polytope_vertex_cache_info() -> PolytopeVertexCacheInfo:
    """Report process-local geometry reuse, including uncertain-support fallbacks.

    ``currbytes`` and ``maxbytes`` count vertex-array storage; entry count also
    bounds the constraint signatures and bookkeeping.  ``support_hits`` counts
    queries for which cached geometry replaced both support LPs.
    """

    with _vertex_cache_lock:
        return PolytopeVertexCacheInfo(
            hits=_vertex_cache_hits,
            misses=_vertex_cache_misses,
            maxsize=_VERTEX_CACHE_MAX_SIZE,
            currsize=len(_vertex_cache),
            maxbytes=_VERTEX_CACHE_MAX_BYTES,
            currbytes=_vertex_cache_bytes,
            support_hits=_vertex_support_hits,
            support_fallbacks=_vertex_support_fallbacks,
        )


def _query_support_from_cached_vertices(
    weight_space: LinearConstraintSystem,
    objective: list[float],
    tolerance: float,
) -> tuple[float, float] | None:
    """Reuse existing complete geometry, without enumerating on a cache miss.

    Linear objectives attain their extrema at vertices of a bounded polytope.
    Geometry is less precise than the support decision's default tolerance,
    however: the enumerator validates vertices with a scale-aware 1e-8 bound.
    Keep the original LP path when either extremum is near the cutting plane,
    including forced equality and degenerate states.
    """

    global _vertex_support_hits, _vertex_support_fallbacks
    polytope = _cached_polytope_vertices(_constraint_system_key(weight_space), tolerance)
    if polytope is None:
        return None
    values = polytope.vertices @ np.asarray(objective, dtype=float)
    lower_value = float(np.min(values))
    upper_value = float(np.max(values))
    coordinate_scale = max(1.0, float(np.max(np.abs(polytope.vertices))))
    objective_scale = max(1.0, float(np.sum(np.abs(objective))))
    margin = max(100.0 * tolerance, 1e-8) * coordinate_scale * objective_scale
    if (
        not np.isfinite(lower_value)
        or not np.isfinite(upper_value)
        or abs(lower_value) <= margin
        or abs(upper_value) <= margin
    ):
        with _vertex_cache_lock:
            _vertex_support_fallbacks += 1
        return None
    with _vertex_cache_lock:
        _vertex_support_hits += 1
    return lower_value, upper_value


def _cached_polytope_vertices(
    key: ConstraintSystemKey,
    tolerance: float,
) -> PolytopeVertices | None:
    global _vertex_cache_hits, _vertex_cache_misses
    with _vertex_cache_lock:
        cache_key = (key, float(tolerance))
        polytope = _vertex_cache.get(cache_key)
        if polytope is None:
            _vertex_cache_misses += 1
            return None
        _vertex_cache.move_to_end(cache_key)
        _vertex_cache_hits += 1
        return polytope


def _remember_polytope_vertices(
    key: ConstraintSystemKey,
    tolerance: float,
    polytope: PolytopeVertices,
) -> None:
    """Retain only complete, validated geometry from an existing volume solve."""

    global _vertex_cache_bytes
    vertices = np.asarray(polytope.vertices, dtype=float)
    if (
        polytope.status != "full_dimensional"
        or vertices.ndim != 2
        or len(vertices) == 0
        or not np.all(np.isfinite(vertices))
        or vertices.nbytes > _VERTEX_CACHE_MAX_BYTES
    ):
        return
    # A bytes-backed array cannot be made writable through setflags, unlike a
    # read-only view of a mutable enumerator result.
    immutable_vertices = np.frombuffer(vertices.tobytes(), dtype=float).reshape(vertices.shape)
    cached = PolytopeVertices(
        status=polytope.status,
        vertices=immutable_vertices,
        affine_dimension=polytope.affine_dimension,
        interior_radius=polytope.interior_radius,
        message=polytope.message,
    )
    cache_key = (key, float(tolerance))
    with _vertex_cache_lock:
        previous = _vertex_cache.pop(cache_key, None)
        if previous is not None:
            _vertex_cache_bytes -= previous.vertices.nbytes
        _vertex_cache[cache_key] = cached
        _vertex_cache_bytes += immutable_vertices.nbytes
        while (
            len(_vertex_cache) > _VERTEX_CACHE_MAX_SIZE
            or _vertex_cache_bytes > _VERTEX_CACHE_MAX_BYTES
        ):
            _, evicted = _vertex_cache.popitem(last=False)
            _vertex_cache_bytes -= evicted.vertices.nbytes


@lru_cache(maxsize=32_768)
def _compute_polytope_intrinsic_volume_cached(
    key: ConstraintSystemKey,
    tolerance: float,
) -> float:
    system = _constraint_system_from_key(key)
    polytope = enumerate_polytope_vertices(system, tolerance=tolerance)
    volume = compute_polytope_intrinsic_volume_from_vertices(system, polytope)
    _remember_polytope_vertices(key, tolerance, polytope)
    return volume


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
