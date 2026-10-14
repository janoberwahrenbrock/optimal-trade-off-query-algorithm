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
_vertex_partial_support_hits = 0
_vertex_partial_lower_hits = 0
_vertex_partial_upper_hits = 0
_vertex_partial_avoided_lps = 0


class PolytopeVertexCacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int
    currsize: int
    maxbytes: int
    currbytes: int
    support_hits: int
    support_fallbacks: int


class PolytopePartialSupportCacheInfo(NamedTuple):
    partial_hits: int
    lower_hits: int
    upper_hits: int
    avoided_lps: int


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

    if polytope.status == "point":
        return 0.0
    if polytope.status in {"infeasible", "lower_dimensional"}:
        return _recover_ambiguous_native_volume(system, original_zero=True)
    if polytope.status != "full_dimensional":
        original_error = _tag_native_geometry_volume_error(
            RuntimeError(polytope.message or f"polytope status {polytope.status}")
        )
        return _recover_ambiguous_native_volume(
            system, original_error=original_error,
        )

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
        native_volume = float(np.ptp(coordinates[:, 0]))
    else:
        try:
            native_volume = float(ConvexHull(coordinates).volume)
        except QhullError:
            native_volume = float(ConvexHull(coordinates, qhull_options="QJ").volume)
    if native_volume == 0.0:
        return _recover_ambiguous_native_volume(system, original_zero=True)
    return native_volume


class _ExactRecoveryVolume(float):
    """Private cache provenance for native zero/error corrected to positive."""

    _exact_lowdimensional_volume_correction = True


class _ExactRecoveredZeroVolume(float):
    """A rescued native exception with exact zero in the explicit measure."""

    _exact_lowdimensional_volume_exception_recovery = True


def _tag_native_geometry_volume_error(error):
    # Preserve original exception class/message/identity; callback exceptions
    # raised before a native geometry result remain untagged and propagate.
    error._otqa_native_geometry_volume_error = True
    return error


def _recover_ambiguous_native_volume(system, *, original_zero=False, original_error=None):
    from .exact_geometry_recovery import (
        RecoveryGeometryError, RecoveryNotApplicable, recover_intrinsic_volume_from_system,
    )
    try:
        value = recover_intrinsic_volume_from_system(system)
        if value > 0.0:
            return _ExactRecoveryVolume(value)
        return _ExactRecoveredZeroVolume(0.0) if original_error is not None else 0.0
    except RecoveryNotApplicable:
        if original_zero:
            return 0.0
        raise original_error
    except RecoveryGeometryError as error:
        if original_error is not None:
            raise original_error from error
        original_error = _tag_native_geometry_volume_error(
            RuntimeError("ambiguous zero intrinsic volume could not be certified")
        )
        raise original_error from error


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
    constraint_key = _constraint_system_key(weight_space)
    support = _query_support_from_cached_vertices(
        weight_space, objective, tolerance,
        constraint_key=constraint_key, allow_partial=True,
    )
    lower_value, upper_value = (None, None) if support is None else support
    if lower_value is None or upper_value is None:
        lower = weight_space.minimize(objective) if lower_value is None else None
        upper = weight_space.maximize(objective) if upper_value is None else None
        if (lower is not None and lower.status != "optimal") or (
            upper is not None and upper.status != "optimal"
        ):
            raise RuntimeError("cannot determine exact query support")
        if (lower is not None and lower.optimal_value is None) or (
            upper is not None and upper.optimal_value is None
        ):
            raise RuntimeError("query support optimization has no value")
        if lower is not None:
            lower_value = float(lower.optimal_value)
        if upper is not None:
            upper_value = float(upper.optimal_value)
    assert lower_value is not None and upper_value is not None
    if lower_value >= -tolerance and upper_value <= tolerance:
        return {"<": 0.0, "=": 1.0, ">": 0.0}
    if lower_value >= -tolerance:
        return {"<": 0.0, "=": 0.0, ">": 1.0}
    if upper_value <= tolerance:
        return {"<": 1.0, "=": 0.0, ">": 0.0}

    try:
        total_volume = _compute_polytope_intrinsic_volume_cached(
            constraint_key, float(tolerance)
        )
    except RuntimeError as original_error:
        if not getattr(original_error, "_otqa_native_geometry_volume_error", False):
            raise
        return _recover_query_volume_exception(weight_space, query, original_error)
    if total_volume <= 0.0:
        return _recover_query_probability_failure(
            weight_space, query, "weight space has no positive intrinsic volume",
        )

    # These signatures are identical to copying the system, appending the
    # cutting inequality and converting the copy back to a signature. Reuse
    # the immutable prefix while retaining every coefficient and row order.
    inequalities, inequality_right_side, equalities, equality_right_side = constraint_key
    less_key: ConstraintSystemKey = (
        inequalities + (tuple(objective),),
        inequality_right_side + (0.0,),
        equalities,
        equality_right_side,
    )
    greater_key: ConstraintSystemKey = (
        inequalities + (tuple(-value for value in objective),),
        inequality_right_side + (0.0,),
        equalities,
        equality_right_side,
    )
    try:
        less_volume = _compute_polytope_intrinsic_volume_cached(less_key, float(tolerance))
        greater_volume = _compute_polytope_intrinsic_volume_cached(greater_key, float(tolerance))
    except RuntimeError as original_error:
        if not getattr(original_error, "_otqa_native_geometry_volume_error", False):
            raise
        return _recover_query_volume_exception(weight_space, query, original_error)
    if any(getattr(value, "_exact_lowdimensional_volume_correction", False)
           or getattr(value, "_exact_lowdimensional_volume_exception_recovery", False)
           for value in (total_volume, less_volume, greater_volume)):
        return _recover_query_probability_failure(
            weight_space, query, "native volume recovery requires a common exact partition",
        )
    raw = np.asarray([less_volume, greater_volume], dtype=float) / total_volume
    raw = np.maximum(raw, 0.0)
    probability_sum = float(np.sum(raw))
    if probability_sum <= 0.0:
        return _recover_query_probability_failure(
            weight_space, query, "exact query branches have zero total probability",
        )
    # The shared cutting hyperplane has volume zero.  Renormalization only
    # removes floating-point accumulation from two independent hull volumes.
    raw /= probability_sum
    return {
        "<": float(raw[0]),
        "=": 0.0,
        ">": float(raw[1]),
    }


class _ExactRecoveryProbabilities(dict):
    """Internal provenance; the public three-key float mapping is unchanged."""

    _exact_lowdimensional_recovery = True

    def copy(self):
        return type(self)(self)


def _recover_query_volume_exception(weight_space, query, original_error):
    from .exact_geometry_recovery import (
        RecoveryGeometryError, RecoveryNotApplicable, recover_probabilities_from_system,
    )
    try:
        return _ExactRecoveryProbabilities(recover_probabilities_from_system(weight_space, query))
    except (RecoveryGeometryError, RecoveryNotApplicable) as error:
        raise original_error from error


def _recover_query_probability_failure(weight_space, query, original_message):
    # Import and exact RREF/clipping occur only at the two existing failures.
    from .exact_geometry_recovery import (
        RecoveryGeometryError, RecoveryNotApplicable, recover_probabilities_from_system,
    )
    try:
        return _ExactRecoveryProbabilities(recover_probabilities_from_system(weight_space, query))
    except (RecoveryGeometryError, RecoveryNotApplicable) as error:
        raise RuntimeError(original_message) from error


def clear_polytope_volume_cache() -> None:
    global _vertex_cache_bytes, _vertex_cache_hits, _vertex_cache_misses
    global _vertex_support_hits, _vertex_support_fallbacks
    global _vertex_partial_support_hits, _vertex_partial_lower_hits
    global _vertex_partial_upper_hits, _vertex_partial_avoided_lps
    _compute_polytope_intrinsic_volume_cached.cache_clear()
    clear_affine_geometry_cache()
    with _vertex_cache_lock:
        _vertex_cache.clear()
        _vertex_cache_bytes = 0
        _vertex_cache_hits = 0
        _vertex_cache_misses = 0
        _vertex_support_hits = 0
        _vertex_support_fallbacks = 0
        _vertex_partial_support_hits = 0
        _vertex_partial_lower_hits = 0
        _vertex_partial_upper_hits = 0
        _vertex_partial_avoided_lps = 0


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


def polytope_partial_support_cache_info() -> PolytopePartialSupportCacheInfo:
    """Report support attempts reusing exactly one cached geometric bound.

    Legacy ``support_hits`` still requires both bounds and ``support_fallbacks``
    still includes partial attempts. ``lower_hits``/``upper_hits`` count the
    retained geometric bound, even if the remaining LP later fails. Each
    partial attempt avoids one support optimization, not its possible native
    retry attempts. This report preserves the original vertex-cache info schema.
    """

    with _vertex_cache_lock:
        return PolytopePartialSupportCacheInfo(
            partial_hits=_vertex_partial_support_hits,
            lower_hits=_vertex_partial_lower_hits,
            upper_hits=_vertex_partial_upper_hits,
            avoided_lps=_vertex_partial_avoided_lps,
        )


def _query_support_from_cached_vertices(
    weight_space: LinearConstraintSystem,
    objective: list[float],
    tolerance: float,
    *,
    constraint_key: ConstraintSystemKey | None = None,
    allow_partial: bool = False,
) -> tuple[float | None, float | None] | None:
    """Reuse existing complete geometry, without enumerating on a cache miss.

    Linear objectives attain their extrema at vertices of a bounded polytope.
    Geometry is less precise than the support decision's default tolerance,
    however: the enumerator validates vertices with a scale-aware 1e-8 bound.
    The default retains the original all-or-nothing helper behavior. An
    explicit partial request retains a bound only under the same margin;
    uncertain extrema are returned as None for the original LP calculation.
    Nonfinite extrema, forced equality, and degenerate states keep both LPs.
    """

    global _vertex_support_hits, _vertex_support_fallbacks
    global _vertex_partial_support_hits, _vertex_partial_lower_hits
    global _vertex_partial_upper_hits, _vertex_partial_avoided_lps
    if constraint_key is None:
        constraint_key = _constraint_system_key(weight_space)
    polytope = _cached_polytope_vertices(constraint_key, tolerance)
    if polytope is None:
        return None
    values = polytope.vertices @ np.asarray(objective, dtype=float)
    lower_value = float(np.min(values))
    upper_value = float(np.max(values))
    coordinate_scale = max(1.0, float(np.max(np.abs(polytope.vertices))))
    objective_scale = max(1.0, float(np.sum(np.abs(objective))))
    margin = max(100.0 * tolerance, 1e-8) * coordinate_scale * objective_scale
    finite = np.isfinite(lower_value) and np.isfinite(upper_value)
    lower_uncertain = abs(lower_value) <= margin
    upper_uncertain = abs(upper_value) <= margin
    if not finite or lower_uncertain or upper_uncertain:
        with _vertex_cache_lock:
            _vertex_support_fallbacks += 1
            if allow_partial and finite and lower_uncertain != upper_uncertain:
                _vertex_partial_support_hits += 1
                _vertex_partial_avoided_lps += 1
                if lower_uncertain:
                    _vertex_partial_upper_hits += 1
                else:
                    _vertex_partial_lower_hits += 1
                return (
                    None if lower_uncertain else lower_value,
                    None if upper_uncertain else upper_value,
                )
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
    if not getattr(volume, "_exact_lowdimensional_volume_correction", False):
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
