from __future__ import annotations

"""Internal candidate summary for leaves that will not generate another query.

Every candidate region and intrinsic volume uses the ordinary geometry path.
Only interval/result objects that the terminal caller cannot consume are
omitted.  Ambiguous geometries retain all ordered native LP interval fallbacks.
"""

import numpy as np

from multistep.src import ratio_intervals as _ratios
from multistep.src.ratio_intervals import (
    _MAX_BATCHED_RATIO_ELEMENTS, _MAX_VECTOR_VERTICES_PER_PAIR,
    _compute_ratio_intervals_from_vertices,
    get_ordered_goal_pairs,
)

from .profiling import increment_profile_counter, profile_operation
from . import value_function as _values
from .value_function import (
    CandidateAnalysis, OptimizedMultistepConfig,
    ratio_interval_has_positive_width, ratio_interval_is_feasible,
)


_ORIGINAL_ANALYSIS_HOOKS = tuple((name, getattr(_values, name)) for name in (
    "compute_candidate_analysis_for_mode", "compute_ratio_relevant_candidate_analysis",
    "_compute_all_ratio_intervals_profiled", "compute_all_ratio_interval_analysis",
    "ratio_interval_is_feasible", "ratio_interval_has_positive_width",
))
_ORIGINAL_RATIO_HOOKS = tuple((name, getattr(_ratios, name)) for name in (
    "compute_all_ratio_interval_analysis", "_compute_all_ratio_intervals_from_vertices",
    "_compute_ratio_intervals_from_vertices", "_compute_prepared_ratio_intervals_from_vertices",
    "_compute_ratio_interval_from_vertices", "compute_ratio_interval_for_candidate",
    "get_ordered_goal_pairs", "_validate_candidate_index", "RatioInterval", "GoalPairRatioIntervals",
))


def _vertices_membership(
    vertices: np.ndarray, goal_count: int, geometry_tolerance: float,
    width_tolerance: float,
) -> tuple[bool, bool]:
    """Return (any feasible pair, any positive-width pair), with original bits."""

    pairs = get_ordered_goal_pairs(goal_count)
    if not (
        type(vertices) is np.ndarray and vertices.ndim == 2
        and vertices.shape == (len(vertices), goal_count) and len(vertices) > 0
        and vertices.dtype == np.dtype(np.float64) and np.all(np.isfinite(vertices))
    ):
        intervals = _compute_ratio_intervals_from_vertices(
            vertices, goal_count, geometry_tolerance,
        ).values()
        return (
            any(ratio_interval_is_feasible(item) for item in intervals),
            any(ratio_interval_has_positive_width(item, width_tolerance) for item in intervals),
        )

    positive_masks = vertices > geometry_tolerance
    nonpositive_masks = vertices <= geometry_tolerance
    has_positive = np.any(positive_masks, axis=0)
    feasible = any(bool(has_positive[b]) for _, b in pairs)
    if (
        len(vertices) > _MAX_VECTOR_VERTICES_PER_PAIR * len(pairs)
        or len(vertices) * goal_count * goal_count > _MAX_BATCHED_RATIO_ELEMENTS
    ):
        # These are the original prepared-path per-pair divisions/reductions.
        columns = [vertices[:, index] for index in range(goal_count)]
        positive_denominators = [
            column[positive_masks[:, index]] for index, column in enumerate(columns)
        ]
        positive_width = False
        for a, b in pairs:
            if not has_positive[b]:
                continue
            ratios = columns[a][positive_masks[:, b]] / positive_denominators[b]
            lower = max(0.0, float(np.min(ratios)))
            if np.any(nonpositive_masks[:, b] & positive_masks[:, a]):
                positive_width = True
            else:
                upper = max(0.0, float(np.max(ratios)))
                if upper > lower + width_tolerance:
                    positive_width = True
        return feasible, positive_width

    # Preserve the original broadcasted ratio operations and reductions.
    pair_mask = ~np.eye(goal_count, dtype=bool)
    ratio_mask = positive_masks[:, None, :] & pair_mask[None, :, :]
    ratios = np.empty((len(vertices), goal_count, goal_count), dtype=np.float64)
    np.divide(vertices[:, :, None], vertices[:, None, :], out=ratios, where=ratio_mask)
    lowers = np.min(ratios, axis=0, where=ratio_mask, initial=np.inf)
    uppers = np.max(ratios, axis=0, where=ratio_mask, initial=-np.inf)
    unbounded = np.any(positive_masks[:, :, None] & nonpositive_masks[:, None, :], axis=0)
    positive_width = False
    for a, b in pairs:
        if not has_positive[b]:
            continue
        lower = max(0.0, float(lowers[a, b]))
        if unbounded[a, b]:
            positive_width = True
        else:
            upper = max(0.0, float(uppers[a, b]))
            if upper > lower + width_tolerance:
                positive_width = True
    return feasible, positive_width


def compute_terminal_candidate_analysis_for_mode(
    alternatives, weight_space, candidate_subset,
    config: OptimizedMultistepConfig, include_candidate_volumes: bool = False,
) -> CandidateAnalysis:
    if (
        config.candidate_count_mode != "ratio_relevant" or config.ratio_interval_engine != "geometry"
        or any(getattr(_values, name) is not original for name, original in _ORIGINAL_ANALYSIS_HOOKS)
        or any(getattr(_ratios, name) is not original for name, original in _ORIGINAL_RATIO_HOOKS)
    ):
        # Full analysis retains custom interval/membership hooks. Geometry and
        # volume callbacks below resolve the same live aliases as that path,
        # including synchronized profiler wrappers and keyword-only overrides.
        return _values.compute_candidate_analysis_for_mode(
            alternatives, weight_space, candidate_subset, config,
            include_candidate_volumes=include_candidate_volumes,
        )
    if weight_space.variable_count != alternatives.get_anzahl_spalten():
        raise ValueError("weight_space must have the same number of variables as the number of goals")
    if config.geometry_tolerance <= 0.0:
        raise ValueError("geometry_tolerance must be positive")
    checked = list(range(alternatives.get_anzahl_zeilen())) if candidate_subset is None else list(candidate_subset)
    if not checked:
        return CandidateAnalysis([], None, {}, {})
    goal_count = alternatives.get_anzahl_spalten()
    feasible: set[int] = set()
    relevant: set[int] = set()
    volumes = {} if include_candidate_volumes else None
    increment_profile_counter("terminal_candidate_summary_calls")
    increment_profile_counter("ratio_interval_batches")
    with profile_operation("ratio_intervals"):
        for candidate in checked:
            if not 0 <= candidate < alternatives.get_anzahl_zeilen():
                raise IndexError("alternative_index is out of range")
            region = _ratios.build_optimality_region(
                alternatives=alternatives, weight_space=weight_space,
                alternative_index=candidate,
            )
            polytope = _ratios.enumerate_polytope_vertices(
                system=region, tolerance=config.geometry_tolerance,
            )
            correction = False
            if volumes is not None:
                volumes[int(candidate)] = _ratios.compute_polytope_intrinsic_volume_from_vertices(
                    system=region, polytope=polytope,
                )
                correction = bool(getattr(volumes[int(candidate)], "_exact_lowdimensional_volume_correction", False))
            if polytope.status in {"full_dimensional", "point"} and not correction:
                is_feasible, has_width = _vertices_membership(
                    polytope.vertices, goal_count, config.geometry_tolerance,
                    config.ratio_terminal_tolerance,
                )
                if is_feasible:
                    feasible.add(int(candidate))
                if has_width:
                    relevant.add(int(candidate))
            elif polytope.status != "infeasible" or correction:
                # Keep every fallback LP, row order, and potential exception.
                for a, b in get_ordered_goal_pairs(goal_count):
                    interval = _ratios.compute_ratio_interval_for_candidate(
                        alternatives=alternatives, weight_space=weight_space,
                        alternative_index=candidate, goal_index_a=a, goal_index_b=b,
                    )
                    if ratio_interval_is_feasible(interval):
                        feasible.add(int(candidate))
                    if ratio_interval_has_positive_width(interval, config.ratio_terminal_tolerance):
                        relevant.add(int(candidate))
    selected = relevant if relevant else feasible
    candidates = [candidate for candidate in checked if candidate in selected]
    shares = None
    if volumes is not None:
        # Existing normalization includes every checked candidate, before filtering.
        total = float(sum(volumes.values()))
        shares = {candidate: volume / total if total > 0.0 else 0.0 for candidate, volume in volumes.items()}
    return CandidateAnalysis(
        candidates, None,
        None if volumes is None else {candidate: volumes[candidate] for candidate in candidates},
        None if shares is None else {candidate: shares[candidate] for candidate in candidates},
    )
