from __future__ import annotations

"""Optimized value-function implementations.

The reference implementation remains in ``multistep.src.value_function``.
This module reuses the core domain functions but changes the evaluation
strategy to avoid unnecessary recursive work.
"""

from concurrent.futures import Executor, ProcessPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from multistep.src.candidates import compute_candidate_set
from multistep.src.grid_query_candidates import (
    DEFAULT_GRID_SIZE,
    DEFAULT_GRID_SPACING,
    DEFAULT_MAX_QUERY_VALUE,
    DEFAULT_MIN_QUERY_VALUE,
    GridSpacing,
    build_grid_query_values_from_ratio_interval,
    compute_grid_query_candidates,
    deduplicate_mirrored_query_candidates,
)
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query, QueryOperator
from multistep.src.onestep_query_candidates import (
    QUERY_EPSILON,
    compute_onestep_query_candidates,
)
from multistep.src.optimality_region import build_optimality_region
from multistep.src.polytope_volume import compute_exact_query_answer_probabilities
from multistep.src.polytope_geometry import enumerate_polytope_vertices
from multistep.src.query_probability import ANSWER_OPTIONS, classify_query_answer
from multistep.src.ratio_intervals import (
    RatioIntervalEngine,
    compute_all_ratio_intervals,
    compute_ratio_bounds_for_weight_space,
)
from multistep.src.ratio_intervals import GoalPairRatioIntervals, RatioInterval
from multistep.src.sampling import sample_points_from_constraint_system
from multistep.src.value_function import (
    QueryBranchResult,
    QueryEvaluation,
    ValueFunctionResult,
    query_evaluation_lexicographic_sort_key,
    refine_query_evaluations_lexicographically,
)
from multistep.src.weight_space import build_weight_space

from .profiling import increment_profile_counter, profile_operation


CandidateCountMode = Literal["closed_lp", "ratio_relevant"]
GridDepthQuerySourceMode = Literal["grid", "ratio", "both", "central"]
DepthOneQuerySourceMode = Literal["grid", "ratio", "both", "central"]
QuerySource = str
PosteriorQueryObjective = Literal["entropy", "regret"]
AnswerProbabilityMode = Literal["sampling", "exact_volume"]


@dataclass(frozen=True)
class CandidateAnalysis:
    """Candidates and reusable ratio intervals for one weight-space state."""

    candidates: list[int]
    ratio_intervals: list[GoalPairRatioIntervals] | None = None


@dataclass(frozen=True)
class StateAnalysis:
    """Reusable feasibility and candidate analysis for one answered state."""

    weight_space: LinearConstraintSystem
    is_feasible: bool
    candidate_analysis: CandidateAnalysis | None = None


@dataclass(frozen=True)
class QueryPosteriorScore:
    query: Query
    expected_entropy: float
    information_gain: float
    expected_regret: float
    partition_balance: float


@dataclass(frozen=True)
class OptimizedMultistepConfig:
    sample_count: int = 1000
    burn_in: int = 200
    thinning: int = 5
    random_seed: int | None = None
    sampling_chain_count: int = 1
    equality_tol: float = 0.0
    grid_size: int = DEFAULT_GRID_SIZE
    min_query_value: float = DEFAULT_MIN_QUERY_VALUE
    max_query_value: float = DEFAULT_MAX_QUERY_VALUE
    grid_spacing: GridSpacing = DEFAULT_GRID_SPACING
    query_epsilon: float = QUERY_EPSILON
    skip_zero_probability_branches: bool = True
    pass_candidate_subset: bool = True
    reuse_conditioned_samples: bool = False
    min_conditioned_sample_count: int = 50
    use_ratio_terminal_counts: bool = False
    ratio_terminal_tolerance: float = 1e-12
    canonical_grid_goal_pairs_only: bool = False
    filter_answered_query_candidates: bool = True
    answered_query_abs_tolerance: float = 1e-12
    answered_query_rel_tolerance: float = 1e-9
    answer_support_tolerance: float = 1e-9
    answer_probability_smoothing: float = 1.0
    answer_probability_mode: AnswerProbabilityMode = "sampling"
    parallelize_root: bool = False
    max_workers: int = 4
    candidate_count_mode: CandidateCountMode = "ratio_relevant"
    include_ratio_queries_on_grid_depths: bool = True
    grid_depth_query_source_mode: GridDepthQuerySourceMode = "both"
    depth_one_query_source_mode: DepthOneQuerySourceMode = "ratio"
    repair_zero_terminal_counts: bool = True
    validate_ratio_terminal_counts: bool = False
    max_query_candidates_per_state: int | None = None
    adaptive_depth_candidate_threshold: int | None = None
    ratio_interval_engine: RatioIntervalEngine = "geometry"
    geometry_tolerance: float = 1e-10
    posterior_quantile_levels: tuple[float, ...] = ()
    posterior_query_objective: PosteriorQueryObjective | None = None
    posterior_query_shortlist_size: int | None = None

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")

        if self.burn_in < 0:
            raise ValueError("burn_in must not be negative")

        if self.thinning <= 0:
            raise ValueError("thinning must be positive")

        if self.sampling_chain_count <= 0:
            raise ValueError("sampling_chain_count must be positive")

        if self.sampling_chain_count > self.sample_count:
            raise ValueError("sampling_chain_count must not exceed sample_count")

        if self.equality_tol < 0.0:
            raise ValueError("equality_tol must not be negative")

        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")

        if self.min_query_value <= 0.0:
            raise ValueError("min_query_value must be positive")

        if self.max_query_value <= 0.0:
            raise ValueError("max_query_value must be positive")

        if self.min_query_value > self.max_query_value:
            raise ValueError("min_query_value must not be greater than max_query_value")

        if self.grid_spacing not in {"linear", "log"}:
            raise ValueError("grid_spacing must be 'linear' or 'log'")

        if self.query_epsilon <= 0.0:
            raise ValueError("query_epsilon must be positive")

        if self.min_conditioned_sample_count <= 0:
            raise ValueError("min_conditioned_sample_count must be positive")

        if self.ratio_terminal_tolerance < 0.0:
            raise ValueError("ratio_terminal_tolerance must not be negative")

        if self.answered_query_abs_tolerance < 0.0:
            raise ValueError("answered_query_abs_tolerance must not be negative")

        if self.answered_query_rel_tolerance < 0.0:
            raise ValueError("answered_query_rel_tolerance must not be negative")

        if self.answer_support_tolerance < 0.0:
            raise ValueError("answer_support_tolerance must not be negative")

        if self.answer_probability_smoothing < 0.0:
            raise ValueError("answer_probability_smoothing must not be negative")

        if self.answer_probability_mode not in {"sampling", "exact_volume"}:
            raise ValueError(
                "answer_probability_mode must be 'sampling' or 'exact_volume'"
            )

        if self.answer_probability_mode == "exact_volume" and (
            self.reuse_conditioned_samples
            or self.max_query_candidates_per_state is not None
            or bool(self.posterior_quantile_levels)
            or self.posterior_query_objective is not None
        ):
            raise ValueError(
                "exact_volume probabilities cannot be combined with sample-based "
                "conditioning, quantiles, or shortlists"
            )

        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")

        if (
            self.max_query_candidates_per_state is not None
            and self.max_query_candidates_per_state <= 0
        ):
            raise ValueError("max_query_candidates_per_state must be positive")

        if (
            self.adaptive_depth_candidate_threshold is not None
            and self.adaptive_depth_candidate_threshold <= 0
        ):
            raise ValueError("adaptive_depth_candidate_threshold must be positive")

        if self.candidate_count_mode not in {"closed_lp", "ratio_relevant"}:
            raise ValueError("candidate_count_mode must be 'closed_lp' or 'ratio_relevant'")

        if self.grid_depth_query_source_mode not in {
            "grid",
            "ratio",
            "both",
            "central",
        }:
            raise ValueError(
                "grid_depth_query_source_mode must be 'grid', 'ratio', 'both', "
                "or 'central'"
            )

        if self.depth_one_query_source_mode not in {
            "grid",
            "ratio",
            "both",
            "central",
        }:
            raise ValueError(
                "depth_one_query_source_mode must be 'grid', 'ratio', 'both', "
                "or 'central'"
            )

        if self.ratio_interval_engine not in {"geometry", "lp"}:
            raise ValueError("ratio_interval_engine must be 'geometry' or 'lp'")

        if self.geometry_tolerance <= 0.0:
            raise ValueError("geometry_tolerance must be positive")

        if any(not 0.0 < level < 1.0 for level in self.posterior_quantile_levels):
            raise ValueError("posterior_quantile_levels must be between zero and one")

        if len(set(self.posterior_quantile_levels)) != len(
            self.posterior_quantile_levels
        ):
            raise ValueError("posterior_quantile_levels must not contain duplicates")

        if self.posterior_query_objective not in {None, "entropy", "regret"}:
            raise ValueError("posterior_query_objective must be 'entropy' or 'regret'")

        if (
            self.posterior_query_shortlist_size is not None
            and self.posterior_query_shortlist_size <= 0
        ):
            raise ValueError("posterior_query_shortlist_size must be positive")

        if (self.posterior_query_objective is None) != (
            self.posterior_query_shortlist_size is None
        ):
            raise ValueError(
                "posterior_query_objective and posterior_query_shortlist_size "
                "must be configured together"
            )

        if (
            self.max_query_candidates_per_state is not None
            and self.posterior_query_objective is not None
        ):
            raise ValueError(
                "sample-balance and posterior-objective shortlists are mutually exclusive"
            )


class OptimizedValueFunctionSession:
    """Reuse root-level worker processes across consecutive user questions."""

    def __init__(
        self,
        alternatives: AlternativenMatrix,
        config: OptimizedMultistepConfig | None = None,
        max_cached_results: int = 16,
    ) -> None:
        if max_cached_results < 0:
            raise ValueError("max_cached_results must not be negative")

        self.alternatives = alternatives
        self.config = config or OptimizedMultistepConfig()
        self.max_cached_results = max_cached_results
        self._executor: ProcessPoolExecutor | None = None
        self._result_cache: OrderedDict[tuple[object, ...], ValueFunctionResult] = (
            OrderedDict()
        )
        self._state_analysis_cache: OrderedDict[tuple[object, ...], StateAnalysis] = (
            OrderedDict()
        )
        self._is_closed = False

        if self.config.parallelize_root and self.config.max_workers > 1:
            self._executor = ProcessPoolExecutor(max_workers=self.config.max_workers)

    def compute(
        self,
        answered_queries: list[AnsweredQuery],
        remaining_depth: int,
        candidate_subset: list[int] | None = None,
        samples: list[list[float]] | None = None,
    ) -> ValueFunctionResult:
        if self._is_closed:
            raise RuntimeError("optimized value-function session is closed")

        cache_key = self._build_result_cache_key(
            answered_queries=answered_queries,
            remaining_depth=remaining_depth,
            candidate_subset=candidate_subset,
            samples=samples,
        )
        if cache_key is not None and cache_key in self._result_cache:
            increment_profile_counter("session_cache_hits")
            self._result_cache.move_to_end(cache_key)
            return self._result_cache[cache_key]

        state_analysis = self.analyze_state(
            answered_queries=answered_queries,
            candidate_subset=candidate_subset,
        )
        result = compute_value_function_optimized(
            alternatives=self.alternatives,
            answered_queries=answered_queries,
            remaining_depth=remaining_depth,
            config=self.config,
            candidate_subset=candidate_subset,
            samples=samples,
            is_root_call=True,
            executor=self._executor,
            precomputed_state_analysis=state_analysis,
        )
        if cache_key is not None:
            self._result_cache[cache_key] = result
            self._result_cache.move_to_end(cache_key)
            while len(self._result_cache) > self.max_cached_results:
                self._result_cache.popitem(last=False)
        return result

    def analyze_state(
        self,
        answered_queries: list[AnsweredQuery],
        candidate_subset: list[int] | None = None,
    ) -> StateAnalysis:
        """Analyze a state once and reuse it across planning and termination."""

        if self._is_closed:
            raise RuntimeError("optimized value-function session is closed")
        cache_key = self._build_state_cache_key(
            answered_queries=answered_queries,
            candidate_subset=candidate_subset,
        )
        if cache_key is not None and cache_key in self._state_analysis_cache:
            increment_profile_counter("state_analysis_cache_hits")
            self._state_analysis_cache.move_to_end(cache_key)
            return self._state_analysis_cache[cache_key]

        weight_space = build_weight_space(
            goal_count=self.alternatives.get_anzahl_spalten(),
            answered_queries=answered_queries,
        )
        is_feasible = weight_space.is_feasible()
        candidate_analysis = (
            compute_candidate_analysis_for_mode(
                alternatives=self.alternatives,
                weight_space=weight_space,
                candidate_subset=candidate_subset,
                config=self.config,
            )
            if is_feasible
            else None
        )
        analysis = StateAnalysis(
            weight_space=weight_space,
            is_feasible=is_feasible,
            candidate_analysis=candidate_analysis,
        )
        if cache_key is not None:
            self._state_analysis_cache[cache_key] = analysis
            self._state_analysis_cache.move_to_end(cache_key)
            while len(self._state_analysis_cache) > self.max_cached_results:
                self._state_analysis_cache.popitem(last=False)
        return analysis

    def clear_cache(self) -> None:
        self._result_cache.clear()
        self._state_analysis_cache.clear()

    def _build_state_cache_key(
        self,
        answered_queries: list[AnsweredQuery],
        candidate_subset: list[int] | None,
    ) -> tuple[object, ...] | None:
        if self.max_cached_results == 0:
            return None
        return (
            tuple(int(candidate) for candidate in candidate_subset)
            if candidate_subset is not None
            else None,
            _normalized_answered_query_key(answered_queries),
        )

    def _build_result_cache_key(
        self,
        answered_queries: list[AnsweredQuery],
        remaining_depth: int,
        candidate_subset: list[int] | None,
        samples: list[list[float]] | None,
    ) -> tuple[object, ...] | None:
        if self.max_cached_results == 0 or samples is not None:
            return None

        return (
            int(remaining_depth),
            tuple(int(candidate) for candidate in candidate_subset)
            if candidate_subset is not None
            else None,
            _normalized_answered_query_key(answered_queries),
        )

    def close(self) -> None:
        if self._is_closed:
            return

        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self.clear_cache()
        self._is_closed = True

    def __enter__(self) -> OptimizedValueFunctionSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def compute_value_function_optimized(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    remaining_depth: int,
    config: OptimizedMultistepConfig | None = None,
    candidate_subset: list[int] | None = None,
    samples: list[list[float]] | None = None,
    is_root_call: bool = True,
    executor: Executor | None = None,
    precomputed_state_analysis: StateAnalysis | None = None,
) -> ValueFunctionResult:
    if remaining_depth < 0:
        raise ValueError("remaining_depth must not be negative")

    resolved_config = config or OptimizedMultistepConfig()
    increment_profile_counter("state_calls")
    state_analysis = precomputed_state_analysis
    if state_analysis is None:
        weight_space = build_weight_space(
            goal_count=alternatives.get_anzahl_spalten(),
            answered_queries=answered_queries,
        )
        state_analysis = StateAnalysis(
            weight_space=weight_space,
            is_feasible=weight_space.is_feasible(),
        )
    else:
        weight_space = state_analysis.weight_space

    if not state_analysis.is_feasible:
        return ValueFunctionResult(
            remaining_depth=remaining_depth,
            value=0.0,
            best_query=None,
            candidate_count=0,
            query_evaluations=(),
            is_feasible=False,
        )

    candidate_analysis = state_analysis.candidate_analysis
    if candidate_analysis is None:
        candidate_analysis = compute_candidate_analysis_for_mode(
            alternatives=alternatives,
            weight_space=weight_space,
            candidate_subset=candidate_subset,
            config=resolved_config,
        )
    candidates = candidate_analysis.candidates
    candidate_count = len(candidates)

    if remaining_depth == 0 or candidate_count <= 1:
        return ValueFunctionResult(
            remaining_depth=remaining_depth,
            value=float(candidate_count),
            best_query=None,
            candidate_count=candidate_count,
            query_evaluations=(),
            is_feasible=True,
        )

    evaluation_depth = remaining_depth
    if (
        remaining_depth > 1
        and resolved_config.adaptive_depth_candidate_threshold is not None
        and candidate_count > resolved_config.adaptive_depth_candidate_threshold
    ):
        evaluation_depth = 1
        increment_profile_counter("adaptive_depth_reductions")

    state_samples = (
        resolve_state_samples(
            weight_space=weight_space,
            samples=samples,
            config=resolved_config,
        )
        if resolved_config.posterior_quantile_levels
        else None
    )
    query_candidate_data = compute_query_candidates_for_depth_optimized(
        alternatives=alternatives,
        weight_space=weight_space,
        candidates=candidates,
        remaining_depth=evaluation_depth,
        config=resolved_config,
        precomputed_ratio_intervals=candidate_analysis.ratio_intervals,
        samples=state_samples,
    )
    query_candidates = query_candidate_data.query_candidates
    increment_profile_counter("query_candidates", len(query_candidates))
    if resolved_config.filter_answered_query_candidates:
        query_candidates = filter_already_answered_queries(
            queries=query_candidates,
            answered_queries=answered_queries,
            abs_tol=resolved_config.answered_query_abs_tolerance,
            rel_tol=resolved_config.answered_query_rel_tolerance,
        )

    if not query_candidates:
        return ValueFunctionResult(
            remaining_depth=remaining_depth,
            value=float(candidate_count),
            best_query=None,
            candidate_count=candidate_count,
            query_evaluations=(),
            is_feasible=True,
        )

    samples_are_required = (
        resolved_config.answer_probability_mode == "sampling"
        or resolved_config.max_query_candidates_per_state is not None
        or resolved_config.posterior_query_objective is not None
    )
    if state_samples is None and samples_are_required:
        state_samples = resolve_state_samples(
            weight_space=weight_space,
            samples=samples,
            config=resolved_config,
        )
    if resolved_config.posterior_query_objective is not None:
        assert state_samples is not None
        query_candidates = shortlist_query_candidates_by_posterior_objective(
            alternatives=alternatives,
            query_candidates=query_candidates,
            query_sources=query_candidate_data.query_sources,
            samples=state_samples,
            equality_tol=resolved_config.equality_tol,
            objective=resolved_config.posterior_query_objective,
            additional_query_limit=resolved_config.posterior_query_shortlist_size,
        )
    if resolved_config.max_query_candidates_per_state is not None:
        assert state_samples is not None
        query_candidates = shortlist_query_candidates_by_sample_balance(
            query_candidates=query_candidates,
            samples=state_samples,
            equality_tol=resolved_config.equality_tol,
            limit=resolved_config.max_query_candidates_per_state,
        )
    increment_profile_counter("evaluated_query_candidates", len(query_candidates))
    parallelize_query_evaluations = (
        is_root_call
        and resolved_config.parallelize_root
        and resolved_config.max_workers > 1
        and len(query_candidates) > 1
    )
    query_evaluations = evaluate_query_candidates_optimized(
        alternatives=alternatives,
        answered_queries=answered_queries,
        weight_space=weight_space,
        query_candidates=query_candidates,
        samples=state_samples,
        remaining_depth=evaluation_depth,
        config=resolved_config,
        candidate_subset=candidates if resolved_config.pass_candidate_subset else None,
        ratio_intervals_by_goal_pair=query_candidate_data.ratio_intervals_by_goal_pair,
        query_sources=query_candidate_data.query_sources,
        parallelize=parallelize_query_evaluations,
        executor=executor,
    )
    query_evaluations, best_evaluation = refine_query_evaluations_lexicographically(
        query_evaluations=query_evaluations,
        remaining_depth=evaluation_depth,
        evaluate_queries_at_depth=lambda queries, depth: (
            evaluate_query_candidates_optimized(
                alternatives=alternatives,
                answered_queries=answered_queries,
                weight_space=weight_space,
                query_candidates=queries,
                samples=state_samples,
                remaining_depth=depth,
                config=resolved_config,
                candidate_subset=(
                    candidates if resolved_config.pass_candidate_subset else None
                ),
                ratio_intervals_by_goal_pair=(
                    query_candidate_data.ratio_intervals_by_goal_pair
                ),
                query_sources=query_candidate_data.query_sources,
                parallelize=(
                    parallelize_query_evaluations and len(queries) > 1
                ),
                executor=executor,
            )
        ),
    )
    if candidate_count > 0 and best_evaluation.expected_value < 1.0 - 1e-9:
        raise RuntimeError(
            "expected candidate count must not be below one for a feasible state "
            f"with candidates; got {best_evaluation.expected_value:g} "
            f"with candidate_count={candidate_count}"
        )

    return ValueFunctionResult(
        remaining_depth=remaining_depth,
        value=best_evaluation.expected_value,
        best_query=best_evaluation.query,
        candidate_count=candidate_count,
        query_evaluations=query_evaluations,
        is_feasible=True,
    )


def shortlist_query_candidates_by_sample_balance(
    query_candidates: list[Query],
    samples: list[list[float]],
    equality_tol: float,
    limit: int | None,
) -> list[Query]:
    """Keep the most balanced sample partitions as an opt-in approximation."""

    if limit is None or len(query_candidates) <= limit:
        return query_candidates

    ranked_indices = sorted(
        range(len(query_candidates)),
        key=lambda index: (
            compute_sample_partition_balance_score(
                query=query_candidates[index],
                samples=samples,
                equality_tol=equality_tol,
            ),
            canonical_query_key(query_candidates[index]),
        ),
    )
    selected_indices = set(ranked_indices[:limit])
    increment_profile_counter(
        "shortlisted_query_candidates",
        len(query_candidates) - limit,
    )
    return [
        query
        for index, query in enumerate(query_candidates)
        if index in selected_indices
    ]


def compute_sample_partition_balance_score(
    query: Query,
    samples: list[list[float]],
    equality_tol: float,
) -> float:
    partitioned_samples = partition_samples_by_query_answer(
        query=query,
        samples=samples,
        equality_tol=equality_tol,
    )
    sample_count = len(samples)
    if sample_count == 0:
        return math.inf

    return sum(
        (len(partitioned_samples[answer]) / sample_count) ** 2
        for answer in ANSWER_OPTIONS
    )


def shortlist_query_candidates_by_posterior_objective(
    alternatives: AlternativenMatrix,
    query_candidates: list[Query],
    query_sources: dict[tuple[int, int, float], QuerySource],
    samples: list[list[float]],
    equality_tol: float,
    objective: PosteriorQueryObjective | None,
    additional_query_limit: int | None,
) -> list[Query]:
    """Retain all ratio queries plus the best posterior-ranked additions."""

    if objective is None or additional_query_limit is None:
        return query_candidates

    baseline_indices = {
        index
        for index, query in enumerate(query_candidates)
        if "ratio" in query_sources.get(canonical_query_key(query), "").split("+")
    }
    additional_indices = [
        index for index in range(len(query_candidates)) if index not in baseline_indices
    ]
    if len(additional_indices) <= additional_query_limit:
        return query_candidates

    scores = score_query_candidates_by_posterior(
        alternatives=alternatives,
        query_candidates=query_candidates,
        samples=samples,
        equality_tol=equality_tol,
    )
    if objective == "entropy":
        objective_key = lambda index: (
            scores[index].expected_entropy,
            scores[index].partition_balance,
            canonical_query_key(scores[index].query),
        )
    else:
        objective_key = lambda index: (
            scores[index].expected_regret,
            scores[index].partition_balance,
            canonical_query_key(scores[index].query),
        )
    selected_additions = set(
        sorted(additional_indices, key=objective_key)[:additional_query_limit]
    )
    selected_indices = baseline_indices | selected_additions
    increment_profile_counter(
        "posterior_shortlisted_query_candidates",
        len(query_candidates) - len(selected_indices),
    )
    return [
        query
        for index, query in enumerate(query_candidates)
        if index in selected_indices
    ]


def score_query_candidates_by_posterior(
    alternatives: AlternativenMatrix,
    query_candidates: list[Query],
    samples: list[list[float]],
    equality_tol: float = 0.0,
) -> list[QueryPosteriorScore]:
    """Evaluate query partitions by winner entropy and decision regret."""

    if not samples:
        raise ValueError("samples must not be empty")
    weights = np.asarray(samples, dtype=float)
    utility_matrix = np.asarray(alternatives.entries, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != utility_matrix.shape[1]:
        raise ValueError("samples must have one weight per alternative goal")

    utilities = weights @ utility_matrix.T
    winners = np.argmax(utilities, axis=1)
    parent_entropy = _entropy_from_labels(
        labels=winners,
        label_count=utility_matrix.shape[0],
    )
    sample_count = len(weights)
    scores: list[QueryPosteriorScore] = []
    for query in query_candidates:
        differences = (
            weights[:, int(query.ziel_index_a)]
            - float(query.value) * weights[:, int(query.ziel_index_b)]
        )
        answer_masks = (
            differences < -equality_tol,
            np.abs(differences) <= equality_tol,
            differences > equality_tol,
        )
        expected_entropy = 0.0
        expected_regret = 0.0
        squared_probabilities = 0.0
        for mask in answer_masks:
            branch_count = int(np.count_nonzero(mask))
            if branch_count == 0:
                continue
            probability = branch_count / sample_count
            branch_utilities = utilities[mask]
            chosen_alternative = int(np.argmax(np.mean(branch_utilities, axis=0)))
            branch_regret = np.mean(
                np.max(branch_utilities, axis=1)
                - branch_utilities[:, chosen_alternative]
            )
            expected_regret += probability * float(branch_regret)
            expected_entropy += probability * _entropy_from_labels(
                labels=winners[mask],
                label_count=utility_matrix.shape[0],
            )
            squared_probabilities += probability**2
        scores.append(
            QueryPosteriorScore(
                query=query,
                expected_entropy=expected_entropy,
                information_gain=parent_entropy - expected_entropy,
                expected_regret=expected_regret,
                partition_balance=squared_probabilities,
            )
        )
    return scores


def _entropy_from_labels(labels: np.ndarray, label_count: int) -> float:
    counts = np.bincount(labels, minlength=label_count).astype(float)
    probabilities = counts[counts > 0.0] / len(labels)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def compute_candidate_set_for_subset(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidate_subset: list[int] | None = None,
) -> list[int]:
    if candidate_subset is None:
        return compute_candidate_set(
            alternatives=alternatives,
            weight_space=weight_space,
        )

    candidates: list[int] = []
    for alternative_index in candidate_subset:
        if not 0 <= alternative_index < alternatives.get_anzahl_zeilen():
            raise IndexError("candidate_subset contains an out-of-range index")

        optimality_region = build_optimality_region(
            alternatives=alternatives,
            weight_space=weight_space,
            alternative_index=alternative_index,
        )
        if optimality_region.is_feasible():
            candidates.append(alternative_index)

    return candidates


def compute_candidate_set_for_mode(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidate_subset: list[int] | None,
    config: OptimizedMultistepConfig,
) -> list[int]:
    return compute_candidate_analysis_for_mode(
        alternatives=alternatives,
        weight_space=weight_space,
        candidate_subset=candidate_subset,
        config=config,
    ).candidates


def compute_candidate_analysis_for_mode(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidate_subset: list[int] | None,
    config: OptimizedMultistepConfig,
) -> CandidateAnalysis:
    if config.candidate_count_mode == "closed_lp":
        return CandidateAnalysis(
            candidates=compute_candidate_set_for_subset(
                alternatives=alternatives,
                weight_space=weight_space,
                candidate_subset=candidate_subset,
            ),
        )

    return compute_ratio_relevant_candidate_analysis(
        alternatives=alternatives,
        weight_space=weight_space,
        candidate_subset=candidate_subset,
        tolerance=config.ratio_terminal_tolerance,
        ratio_interval_engine=config.ratio_interval_engine,
        geometry_tolerance=config.geometry_tolerance,
    )


def compute_ratio_relevant_candidate_set(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidate_subset: list[int] | None = None,
    tolerance: float = 1e-12,
) -> list[int]:
    return compute_ratio_relevant_candidate_analysis(
        alternatives=alternatives,
        weight_space=weight_space,
        candidate_subset=candidate_subset,
        tolerance=tolerance,
    ).candidates


def compute_ratio_relevant_candidate_analysis(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidate_subset: list[int] | None = None,
    tolerance: float = 1e-12,
    ratio_interval_engine: RatioIntervalEngine = "geometry",
    geometry_tolerance: float = 1e-10,
) -> CandidateAnalysis:
    candidates_to_check = (
        list(range(alternatives.get_anzahl_zeilen()))
        if candidate_subset is None
        else list(candidate_subset)
    )
    if not candidates_to_check:
        return CandidateAnalysis(candidates=[], ratio_intervals=[])

    ratio_intervals = _compute_all_ratio_intervals_profiled(
        alternatives=alternatives,
        weight_space=weight_space,
        candidates=candidates_to_check,
        engine=ratio_interval_engine,
        geometry_tolerance=geometry_tolerance,
    )
    relevant_candidates: set[int] = set()
    feasible_candidates: set[int] = set()

    for goal_pair_intervals in ratio_intervals:
        for candidate_index, ratio_interval in (
            goal_pair_intervals.intervals_by_candidate.items()
        ):
            if ratio_interval_is_feasible(ratio_interval):
                feasible_candidates.add(int(candidate_index))

            if ratio_interval_has_positive_width(
                ratio_interval=ratio_interval,
                tolerance=tolerance,
            ):
                relevant_candidates.add(int(candidate_index))

    if relevant_candidates:
        candidates = [
            candidate_index
            for candidate_index in candidates_to_check
            if candidate_index in relevant_candidates
        ]
    else:
        candidates = [
            candidate_index
            for candidate_index in candidates_to_check
            if candidate_index in feasible_candidates
        ]

    candidate_set = set(candidates)
    filtered_ratio_intervals = [
        GoalPairRatioIntervals(
            goal_index_a=goal_pair_intervals.goal_index_a,
            goal_index_b=goal_pair_intervals.goal_index_b,
            intervals_by_candidate={
                candidate_index: ratio_interval
                for candidate_index, ratio_interval in (
                    goal_pair_intervals.intervals_by_candidate.items()
                )
                if int(candidate_index) in candidate_set
            },
        )
        for goal_pair_intervals in ratio_intervals
    ]
    return CandidateAnalysis(
        candidates=candidates,
        ratio_intervals=filtered_ratio_intervals,
    )


def _compute_all_ratio_intervals_profiled(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidates: list[int],
    engine: RatioIntervalEngine = "geometry",
    geometry_tolerance: float = 1e-10,
) -> list[GoalPairRatioIntervals]:
    increment_profile_counter("ratio_interval_batches")
    with profile_operation("ratio_intervals"):
        return compute_all_ratio_intervals(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=candidates,
            engine=engine,
            geometry_tolerance=geometry_tolerance,
        )


def ratio_interval_is_feasible(ratio_interval: RatioInterval) -> bool:
    return ratio_interval.lower.status == "optimal" and ratio_interval.upper.status in {
        "optimal",
        "unbounded",
    }


def ratio_interval_has_positive_width(
    ratio_interval: RatioInterval,
    tolerance: float,
) -> bool:
    if ratio_interval.lower.status != "optimal":
        return False

    if ratio_interval.upper.status == "unbounded":
        return True

    if ratio_interval.upper.status != "optimal":
        return False

    lower_value = get_lower_ratio_value_or_none(ratio_interval)
    upper_value = get_upper_ratio_value_or_none(ratio_interval)
    if lower_value is None or upper_value is None:
        return False

    return upper_value > lower_value + tolerance


def query_evaluation_sort_key(
    evaluation: QueryEvaluation,
) -> tuple[float | int, ...]:
    return query_evaluation_lexicographic_sort_key(evaluation)


def filter_already_answered_queries(
    queries: list[Query],
    answered_queries: list[AnsweredQuery],
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-9,
) -> list[Query]:
    return [
        query
        for query in queries
        if not is_query_already_answered(
            query=query,
            answered_queries=answered_queries,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    ]


def is_query_already_answered(
    query: Query,
    answered_queries: list[AnsweredQuery],
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-9,
) -> bool:
    for answered_query in answered_queries:
        if (
            query.ziel_index_a == answered_query.ziel_index_a
            and query.ziel_index_b == answered_query.ziel_index_b
            and math.isclose(
                float(query.value),
                float(answered_query.value),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        ):
            return True

        if (
            float(query.value) > 0.0
            and float(answered_query.value) > 0.0
            and query.ziel_index_a == answered_query.ziel_index_b
            and query.ziel_index_b == answered_query.ziel_index_a
            and math.isclose(
                float(query.value),
                1.0 / float(answered_query.value),
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        ):
            return True

    return False


def compute_query_candidates_for_depth_optimized(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidates: list[int],
    remaining_depth: int,
    config: OptimizedMultistepConfig,
    precomputed_ratio_intervals: list[GoalPairRatioIntervals] | None = None,
    samples: list[list[float]] | None = None,
) -> "QueryCandidateData":
    if remaining_depth <= 0:
        return QueryCandidateData(query_candidates=[])

    quantile_queries: list[Query] = []
    if config.posterior_quantile_levels:
        if samples is None:
            raise ValueError("samples are required for posterior quantile queries")
        quantile_queries = compute_posterior_quantile_query_candidates(
            samples=samples,
            quantile_levels=config.posterior_quantile_levels,
            min_query_value=config.min_query_value,
            max_query_value=config.max_query_value,
        )

    if remaining_depth == 1:
        ratio_intervals = _resolve_ratio_intervals(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=candidates,
            precomputed_ratio_intervals=precomputed_ratio_intervals,
            config=config,
        )
        grid_queries: list[Query] = []
        if config.depth_one_query_source_mode in {"grid", "both"}:
            grid_queries = compute_grid_queries_for_config(
                weight_space=weight_space,
                config=config,
            )

        ratio_queries: list[Query] = []
        if config.depth_one_query_source_mode in {"ratio", "both"}:
            ratio_queries = compute_onestep_query_candidates(
                goal_pair_ratio_intervals=ratio_intervals,
                epsilon=config.query_epsilon,
            )

        central_queries = (
            compute_central_query_candidates(weight_space)
            if config.depth_one_query_source_mode == "central"
            else []
        )

        query_candidates, query_sources = merge_query_candidates_by_source(
            grid_queries=grid_queries,
            ratio_queries=ratio_queries,
            quantile_queries=quantile_queries,
            central_queries=central_queries,
        )
        return QueryCandidateData(
            query_candidates=query_candidates,
            query_sources=query_sources,
            ratio_intervals_by_goal_pair={
                (
                    int(goal_pair_intervals.goal_index_a),
                    int(goal_pair_intervals.goal_index_b),
                ): goal_pair_intervals
                for goal_pair_intervals in ratio_intervals
            },
        )

    query_source_mode = resolve_grid_depth_query_source_mode(config)
    grid_queries: list[Query] = []
    if query_source_mode in {"grid", "both"}:
        grid_queries = compute_grid_queries_for_config(
            weight_space=weight_space,
            config=config,
        )
    central_queries = (
        compute_central_query_candidates(weight_space)
        if query_source_mode == "central"
        else []
    )

    ratio_intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals] | None
    ratio_intervals_by_goal_pair = None
    ratio_queries: list[Query] = []
    if query_source_mode in {"ratio", "both"}:
        ratio_intervals = _resolve_ratio_intervals(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=candidates,
            precomputed_ratio_intervals=precomputed_ratio_intervals,
            config=config,
        )
        ratio_queries = compute_onestep_query_candidates(
            goal_pair_ratio_intervals=ratio_intervals,
            epsilon=config.query_epsilon,
        )
        ratio_intervals_by_goal_pair = {
            (
                int(goal_pair_intervals.goal_index_a),
                int(goal_pair_intervals.goal_index_b),
            ): goal_pair_intervals
            for goal_pair_intervals in ratio_intervals
        }
    elif query_source_mode == "central":
        ratio_intervals = _resolve_ratio_intervals(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=candidates,
            precomputed_ratio_intervals=precomputed_ratio_intervals,
            config=config,
        )
        ratio_intervals_by_goal_pair = {
            (
                int(goal_pair_intervals.goal_index_a),
                int(goal_pair_intervals.goal_index_b),
            ): goal_pair_intervals
            for goal_pair_intervals in ratio_intervals
        }

    query_candidates, query_sources = merge_query_candidates_by_source(
        grid_queries=grid_queries,
        ratio_queries=ratio_queries,
        quantile_queries=quantile_queries,
        central_queries=central_queries,
    )
    return QueryCandidateData(
        query_candidates=query_candidates,
        query_sources=query_sources,
        ratio_intervals_by_goal_pair=ratio_intervals_by_goal_pair,
    )


def _resolve_ratio_intervals(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidates: list[int],
    precomputed_ratio_intervals: list[GoalPairRatioIntervals] | None,
    config: OptimizedMultistepConfig,
) -> list[GoalPairRatioIntervals]:
    if precomputed_ratio_intervals is not None:
        return precomputed_ratio_intervals

    return _compute_all_ratio_intervals_profiled(
        alternatives=alternatives,
        weight_space=weight_space,
        candidates=candidates,
        engine=config.ratio_interval_engine,
        geometry_tolerance=config.geometry_tolerance,
    )


def resolve_grid_depth_query_source_mode(
    config: OptimizedMultistepConfig,
) -> GridDepthQuerySourceMode:
    if (
        config.grid_depth_query_source_mode == "both"
        and not config.include_ratio_queries_on_grid_depths
    ):
        return "grid"

    return config.grid_depth_query_source_mode


def compute_grid_queries_for_config(
    weight_space: LinearConstraintSystem,
    config: OptimizedMultistepConfig,
) -> list[Query]:
    if config.canonical_grid_goal_pairs_only:
        return compute_canonical_grid_query_candidates(
            weight_space=weight_space,
            grid_size=config.grid_size,
            min_query_value=config.min_query_value,
            max_query_value=config.max_query_value,
            spacing=config.grid_spacing,
        )

    return compute_grid_query_candidates(
        weight_space=weight_space,
        grid_size=config.grid_size,
        min_query_value=config.min_query_value,
        max_query_value=config.max_query_value,
        spacing=config.grid_spacing,
    )


def merge_query_candidates_by_source(
    grid_queries: list[Query],
    ratio_queries: list[Query],
    quantile_queries: list[Query] | None = None,
    central_queries: list[Query] | None = None,
) -> tuple[list[Query], dict[tuple[int, int, float], QuerySource]]:
    resolved_quantile_queries = quantile_queries or []
    resolved_central_queries = central_queries or []
    sources_by_key: dict[tuple[int, int, float], set[str]] = {}
    for query in grid_queries:
        sources_by_key.setdefault(canonical_query_key(query), set()).add("grid")

    for query in ratio_queries:
        sources_by_key.setdefault(canonical_query_key(query), set()).add("ratio")

    for query in resolved_quantile_queries:
        sources_by_key.setdefault(canonical_query_key(query), set()).add("quantile")

    for query in resolved_central_queries:
        sources_by_key.setdefault(canonical_query_key(query), set()).add("central")

    query_candidates = deduplicate_mirrored_query_candidates(
        grid_queries + ratio_queries + resolved_quantile_queries + resolved_central_queries
    )
    query_sources = {
        canonical_query_key(query): combine_query_sources(
            sources_by_key.get(canonical_query_key(query), set())
        )
        for query in query_candidates
    }
    return query_candidates, query_sources


def canonical_query_key(query: Query) -> tuple[int, int, float]:
    value = float(query.value)
    if value <= 0.0:
        return (
            int(query.ziel_index_a),
            int(query.ziel_index_b),
            value,
        )

    direct_key = (
        int(query.ziel_index_a),
        int(query.ziel_index_b),
        value,
    )
    mirrored_key = (
        int(query.ziel_index_b),
        int(query.ziel_index_a),
        1.0 / value,
    )
    return min(direct_key, mirrored_key)


def _normalized_answered_query_key(
    answered_queries: list[AnsweredQuery],
) -> tuple[tuple[int, int, float, QueryOperator], ...]:
    normalized: list[tuple[int, int, float, QueryOperator]] = []
    for answered_query in answered_queries:
        goal_index_a = int(answered_query.ziel_index_a)
        goal_index_b = int(answered_query.ziel_index_b)
        value = float(answered_query.value)
        operator: QueryOperator = answered_query.operator
        if goal_index_a > goal_index_b and value > 0.0:
            goal_index_a, goal_index_b = goal_index_b, goal_index_a
            value = 1.0 / value
            operator = {"<": ">", "=": "=", ">": "<"}[operator]
        normalized.append((goal_index_a, goal_index_b, value, operator))
    return tuple(sorted(normalized))


def combine_query_sources(sources: set[str]) -> QuerySource:
    ordered_sources = [
        source
        for source in ("grid", "ratio", "quantile", "central")
        if source in sources
    ]
    return "+".join(ordered_sources) if ordered_sources else "unknown"


def compute_posterior_quantile_query_candidates(
    samples: list[list[float]],
    quantile_levels: tuple[float, ...],
    min_query_value: float,
    max_query_value: float,
    denominator_tolerance: float = 1e-12,
) -> list[Query]:
    """Create canonical goal-pair thresholds at posterior ratio quantiles."""

    if not samples:
        raise ValueError("samples must not be empty")
    sample_matrix = np.asarray(samples, dtype=float)
    goal_count = sample_matrix.shape[1]
    queries: list[Query] = []
    for goal_index_a in range(goal_count):
        for goal_index_b in range(goal_index_a + 1, goal_count):
            valid = sample_matrix[:, goal_index_b] > denominator_tolerance
            if not np.any(valid):
                continue
            ratios = (
                sample_matrix[valid, goal_index_a]
                / sample_matrix[valid, goal_index_b]
            )
            for query_value in np.quantile(ratios, quantile_levels):
                clipped_value = min(
                    max_query_value,
                    max(min_query_value, float(query_value)),
                )
                queries.append(
                    Query(
                        ziel_index_a=goal_index_a,
                        ziel_index_b=goal_index_b,
                        value=clipped_value,
                    )
                )
    return deduplicate_mirrored_query_candidates(queries)


def compute_central_query_candidates(
    weight_space: LinearConstraintSystem,
) -> list[Query]:
    """Return one deterministic vertex-centroid ratio per canonical goal pair."""

    polytope = enumerate_polytope_vertices(
        weight_space,
        tolerance=1e-10,
    )
    if polytope.status != "full_dimensional":
        raise RuntimeError(polytope.message or f"polytope status {polytope.status}")
    centroid = np.mean(polytope.vertices, axis=0)
    queries: list[Query] = []
    for goal_index_a in range(weight_space.variable_count):
        for goal_index_b in range(goal_index_a + 1, weight_space.variable_count):
            denominator = float(centroid[goal_index_b])
            if denominator <= 1e-14:
                continue
            queries.append(
                Query(
                    ziel_index_a=goal_index_a,
                    ziel_index_b=goal_index_b,
                    value=max(
                        1e-12,
                        float(centroid[goal_index_a]) / denominator,
                    ),
                )
            )
    return queries


def compute_canonical_grid_query_candidates(
    weight_space: LinearConstraintSystem,
    grid_size: int,
    min_query_value: float,
    max_query_value: float,
    spacing: GridSpacing,
) -> list[Query]:
    if weight_space.variable_count <= 1:
        raise ValueError("weight_space must contain at least two goals")

    query_candidates: list[Query] = []
    for goal_index_a in range(weight_space.variable_count):
        for goal_index_b in range(goal_index_a + 1, weight_space.variable_count):
            ratio_interval = compute_ratio_bounds_for_weight_space(
                weight_space=weight_space,
                goal_index_a=goal_index_a,
                goal_index_b=goal_index_b,
            )
            query_values = build_grid_query_values_from_ratio_interval(
                ratio_interval=ratio_interval,
                grid_size=grid_size,
                min_query_value=min_query_value,
                max_query_value=max_query_value,
                spacing=spacing,
            )
            query_candidates.extend(
                Query(
                    ziel_index_a=goal_index_a,
                    ziel_index_b=goal_index_b,
                    value=query_value,
                )
                for query_value in query_values
            )

    return deduplicate_mirrored_query_candidates(query_candidates)


def resolve_state_samples(
    weight_space: LinearConstraintSystem,
    samples: list[list[float]] | None,
    config: OptimizedMultistepConfig,
) -> list[list[float]]:
    if (
        config.reuse_conditioned_samples
        and samples is not None
        and len(samples) >= config.min_conditioned_sample_count
    ):
        return samples

    increment_profile_counter("sampling_calls")
    with profile_operation("sampling"):
        return sample_points_from_constraint_system(
            system=weight_space,
            num_samples=config.sample_count,
            burn_in=config.burn_in,
            thinning=config.thinning,
            seed=config.random_seed,
            chain_count=config.sampling_chain_count,
        )


def evaluate_query_candidates_optimized(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    weight_space: LinearConstraintSystem,
    query_candidates: list[Query],
    samples: list[list[float]] | None,
    remaining_depth: int,
    config: OptimizedMultistepConfig,
    candidate_subset: list[int] | None,
    ratio_intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals] | None,
    query_sources: dict[tuple[int, int, float], QuerySource],
    parallelize: bool,
    executor: Executor | None = None,
) -> tuple[QueryEvaluation, ...]:
    if not parallelize:
        return tuple(
            evaluate_query_candidate_optimized(
                alternatives=alternatives,
                answered_queries=answered_queries,
                weight_space=weight_space,
                query=query,
                samples=samples,
                remaining_depth=remaining_depth,
                config=config,
                candidate_subset=candidate_subset,
                ratio_intervals_by_goal_pair=ratio_intervals_by_goal_pair,
                query_source=query_sources.get(canonical_query_key(query), "unknown"),
            )
            for query in query_candidates
        )

    payloads = [
        (
            alternatives,
            answered_queries,
            weight_space,
            query,
            samples,
            remaining_depth,
            config,
            candidate_subset,
            ratio_intervals_by_goal_pair,
            query_sources.get(canonical_query_key(query), "unknown"),
        )
        for query in query_candidates
    ]
    if executor is not None:
        return tuple(executor.map(_evaluate_query_candidate_worker, payloads))

    with ProcessPoolExecutor(max_workers=config.max_workers) as local_executor:
        return tuple(
            local_executor.map(_evaluate_query_candidate_worker, payloads)
        )


def _evaluate_query_candidate_worker(
    payload: tuple[
        AlternativenMatrix,
        list[AnsweredQuery],
        LinearConstraintSystem,
        Query,
        list[list[float]] | None,
        int,
        OptimizedMultistepConfig,
        list[int] | None,
        dict[tuple[int, int], GoalPairRatioIntervals] | None,
        QuerySource,
    ],
) -> QueryEvaluation:
    (
        alternatives,
        answered_queries,
        weight_space,
        query,
        samples,
        remaining_depth,
        config,
        candidate_subset,
        ratio_intervals_by_goal_pair,
        query_source,
    ) = payload
    return evaluate_query_candidate_optimized(
        alternatives=alternatives,
        answered_queries=answered_queries,
        weight_space=weight_space,
        query=query,
        samples=samples,
        remaining_depth=remaining_depth,
        config=config,
        candidate_subset=candidate_subset,
        ratio_intervals_by_goal_pair=ratio_intervals_by_goal_pair,
        query_source=query_source,
    )


def compute_supported_query_answers(
    weight_space: LinearConstraintSystem,
    query: Query,
    tolerance: float,
) -> dict[QueryOperator, bool]:
    objective = [0.0] * weight_space.variable_count
    objective[int(query.ziel_index_a)] = 1.0
    objective[int(query.ziel_index_b)] = -float(query.value)

    lower_result = weight_space.minimize(objective)
    upper_result = weight_space.maximize(objective)
    if lower_result.status != "optimal" or upper_result.status != "optimal":
        raise RuntimeError("cannot determine query-answer support for weight space")

    if lower_result.optimal_value is None or upper_result.optimal_value is None:
        raise RuntimeError("query-answer support optimization has no optimal value")

    lower_value = float(lower_result.optimal_value)
    upper_value = float(upper_result.optimal_value)
    equality_is_forced = (
        lower_value >= -tolerance
        and upper_value <= tolerance
    )
    return {
        "<": lower_value < -tolerance,
        "=": equality_is_forced,
        ">": upper_value > tolerance,
    }


def compute_supported_query_answers_with_sample_evidence(
    weight_space: LinearConstraintSystem,
    query: Query,
    samples: list[list[float]],
    tolerance: float,
) -> dict[QueryOperator, bool]:
    """Resolve answer support while avoiding LPs already proven by samples.

    A sampled point whose query difference is outside the support tolerance is
    an exact feasibility witness for that strict answer.  We only optimize the
    opposite bound when the samples do not already prove it.  If both strict
    answers have witnesses, no support LP is necessary.
    """

    if not samples:
        raise ValueError("samples must not be empty")

    goal_index_a = int(query.ziel_index_a)
    goal_index_b = int(query.ziel_index_b)
    query_value = float(query.value)
    has_less_witness = False
    has_greater_witness = False
    for weights in samples:
        difference = (
            float(weights[goal_index_a])
            - query_value * float(weights[goal_index_b])
        )
        has_less_witness = has_less_witness or difference < -tolerance
        has_greater_witness = has_greater_witness or difference > tolerance
        if has_less_witness and has_greater_witness:
            return {"<": True, "=": False, ">": True}

    objective = [0.0] * weight_space.variable_count
    objective[goal_index_a] = 1.0
    objective[goal_index_b] = -query_value

    lower_value: float | None = None
    upper_value: float | None = None
    if not has_less_witness:
        lower_result = weight_space.minimize(objective)
        if lower_result.status != "optimal" or lower_result.optimal_value is None:
            raise RuntimeError("cannot determine lower query-answer support")
        lower_value = float(lower_result.optimal_value)

    if not has_greater_witness:
        upper_result = weight_space.maximize(objective)
        if upper_result.status != "optimal" or upper_result.optimal_value is None:
            raise RuntimeError("cannot determine upper query-answer support")
        upper_value = float(upper_result.optimal_value)

    less_is_supported = has_less_witness or (
        lower_value is not None and lower_value < -tolerance
    )
    greater_is_supported = has_greater_witness or (
        upper_value is not None and upper_value > tolerance
    )
    equality_is_forced = (
        not less_is_supported
        and not greater_is_supported
        and lower_value is not None
        and upper_value is not None
        and lower_value >= -tolerance
        and upper_value <= tolerance
    )
    return {
        "<": less_is_supported,
        "=": equality_is_forced,
        ">": greater_is_supported,
    }


def estimate_supported_answer_probabilities(
    answer_counts: dict[QueryOperator, int],
    supported_answers: dict[QueryOperator, bool],
    smoothing: float,
) -> dict[QueryOperator, float]:
    weights = {
        answer: float(answer_counts[answer]) + smoothing
        if supported_answers[answer]
        else 0.0
        for answer in ANSWER_OPTIONS
    }
    weight_sum = sum(weights.values())
    if weight_sum > 0.0:
        return {
            answer: weights[answer] / weight_sum
            for answer in ANSWER_OPTIONS
        }

    active_answers = [
        answer
        for answer in ANSWER_OPTIONS
        if supported_answers[answer]
    ]
    if not active_answers:
        raise RuntimeError("query has no supported answer in feasible weight space")

    fallback_probability = 1.0 / len(active_answers)
    return {
        answer: fallback_probability if answer in active_answers else 0.0
        for answer in ANSWER_OPTIONS
    }


def evaluate_query_candidate_optimized(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    weight_space: LinearConstraintSystem,
    query: Query,
    samples: list[list[float]] | None,
    remaining_depth: int,
    config: OptimizedMultistepConfig,
    candidate_subset: list[int] | None,
    ratio_intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals] | None = None,
    query_source: QuerySource = "unknown",
) -> QueryEvaluation:
    if remaining_depth <= 0:
        raise ValueError("remaining_depth must be positive")

    increment_profile_counter("query_evaluations")
    partitioned_samples: dict[QueryOperator, list[list[float]]] | None = None
    if config.answer_probability_mode == "exact_volume":
        increment_profile_counter("exact_volume_probability_calls")
        with profile_operation("exact_volume_probabilities"):
            probabilities = compute_exact_query_answer_probabilities(
                weight_space=weight_space,
                query=query,
                tolerance=config.geometry_tolerance,
            )
    else:
        if samples is None:
            raise ValueError("samples are required for sampling probabilities")
        partitioned_samples = partition_samples_by_query_answer(
            query=query,
            samples=samples,
            equality_tol=config.equality_tol,
        )
        answer_counts = {
            answer: len(partitioned_samples[answer])
            for answer in ANSWER_OPTIONS
        }
        increment_profile_counter("query_support_checks")
        with profile_operation("query_support"):
            supported_answers = compute_supported_query_answers_with_sample_evidence(
                weight_space=weight_space,
                query=query,
                samples=samples,
                tolerance=config.answer_support_tolerance,
            )
        probabilities = estimate_supported_answer_probabilities(
            answer_counts=answer_counts,
            supported_answers=supported_answers,
            smoothing=config.answer_probability_smoothing,
        )
    branches: list[QueryBranchResult] = []
    expected_value = 0.0

    for answer in ANSWER_OPTIONS:
        increment_profile_counter("branch_checks")
        probability = probabilities[answer]
        if config.skip_zero_probability_branches and probability == 0.0:
            branches.append(
                QueryBranchResult(
                    answer=answer,
                    probability=probability,
                    child_value=0.0,
                    child_candidate_count=None,
                    is_child_feasible=False,
                )
            )
            continue

        if (
            remaining_depth == 1
            and config.use_ratio_terminal_counts
            and ratio_intervals_by_goal_pair is not None
        ):
            child_candidate_count = compute_terminal_candidate_count_from_ratio_intervals(
                query=query,
                answer=answer,
                ratio_intervals_by_goal_pair=ratio_intervals_by_goal_pair,
                tolerance=config.ratio_terminal_tolerance,
            )
            is_child_feasible = child_candidate_count > 0
            if (
                probability > 0.0
                and (
                    config.validate_ratio_terminal_counts
                    or (
                        config.repair_zero_terminal_counts
                        and child_candidate_count == 0
                    )
                )
            ):
                child_candidate_count, is_child_feasible = (
                    compute_terminal_candidate_count_fallback(
                        alternatives=alternatives,
                        answered_queries=answered_queries,
                        query=query,
                        answer=answer,
                        candidate_subset=candidate_subset,
                        config=config,
                    )
                )
            child_value = float(child_candidate_count)
            expected_value += probability * child_value
            branches.append(
                QueryBranchResult(
                    answer=answer,
                    probability=probability,
                    child_value=child_value,
                    child_candidate_count=child_candidate_count,
                    is_child_feasible=is_child_feasible,
                )
            )
            continue

        child_answered_queries = answered_queries + [query.answer(answer)]
        child_weight_space = build_weight_space(
            goal_count=alternatives.get_anzahl_spalten(),
            answered_queries=child_answered_queries,
        )
        child_is_feasible = child_weight_space.is_feasible()
        if child_is_feasible:
            child_candidate_subset = filter_candidate_subset_for_query_answer(
                candidate_subset=candidate_subset,
                query=query,
                answer=answer,
                ratio_intervals_by_goal_pair=ratio_intervals_by_goal_pair,
                tolerance=config.ratio_terminal_tolerance,
            )
            child_samples = (
                partitioned_samples[answer]
                if config.reuse_conditioned_samples
                and partitioned_samples is not None
                else None
            )
            child_result = compute_value_function_optimized(
                alternatives=alternatives,
                answered_queries=child_answered_queries,
                remaining_depth=remaining_depth - 1,
                config=config,
                candidate_subset=child_candidate_subset,
                samples=child_samples,
                is_root_call=False,
                precomputed_state_analysis=StateAnalysis(
                    weight_space=child_weight_space,
                    is_feasible=True,
                ),
            )
            child_value = child_result.value
            child_candidate_count: int | None = child_result.candidate_count
            is_child_feasible = True
        else:
            child_value = 0.0
            child_candidate_count = 0
            is_child_feasible = False

        expected_value += probability * child_value
        branches.append(
            QueryBranchResult(
                answer=answer,
                probability=probability,
                child_value=child_value,
                child_candidate_count=child_candidate_count,
                is_child_feasible=is_child_feasible,
            )
        )

    return QueryEvaluation(
        query=query,
        expected_value=expected_value,
        branches=tuple(branches),
        query_source=query_source,
    )


def filter_candidate_subset_for_query_answer(
    candidate_subset: list[int] | None,
    query: Query,
    answer: QueryOperator,
    ratio_intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals] | None,
    tolerance: float,
) -> list[int] | None:
    """Remove candidates whose parent interval cannot satisfy an answer."""

    if candidate_subset is None or ratio_intervals_by_goal_pair is None:
        return candidate_subset
    goal_pair_intervals = ratio_intervals_by_goal_pair.get(
        (int(query.ziel_index_a), int(query.ziel_index_b))
    )
    if goal_pair_intervals is None:
        return candidate_subset
    return [
        candidate_index
        for candidate_index in candidate_subset
        if candidate_index in goal_pair_intervals.intervals_by_candidate
        and ratio_interval_may_support_answer(
            ratio_interval=goal_pair_intervals.intervals_by_candidate[candidate_index],
            query_value=float(query.value),
            answer=answer,
            tolerance=tolerance,
        )
    ]


def ratio_interval_may_support_answer(
    ratio_interval: RatioInterval,
    query_value: float,
    answer: QueryOperator,
    tolerance: float,
) -> bool:
    """Conservative compatibility check used only for safe candidate pruning."""

    lower_value = get_lower_ratio_value_or_none(ratio_interval)
    if lower_value is None:
        return False
    upper_value = get_upper_ratio_value_or_none(ratio_interval)
    upper_is_unbounded = ratio_interval.upper.status == "unbounded"
    if answer == "<":
        return lower_value < query_value + tolerance
    if answer == ">":
        return upper_is_unbounded or (
            upper_value is not None and upper_value > query_value - tolerance
        )
    return lower_value <= query_value + tolerance and (
        upper_is_unbounded
        or (upper_value is not None and upper_value >= query_value - tolerance)
    )


@dataclass(frozen=True)
class QueryCandidateData:
    query_candidates: list[Query]
    query_sources: dict[tuple[int, int, float], QuerySource] | None = None
    ratio_intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals] | None = None

    def __post_init__(self) -> None:
        if self.query_sources is None:
            object.__setattr__(self, "query_sources", {})


def compute_terminal_candidate_count_from_ratio_intervals(
    query: Query,
    answer: QueryOperator,
    ratio_intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals],
    tolerance: float,
) -> int:
    goal_pair_intervals = ratio_intervals_by_goal_pair[
        (int(query.ziel_index_a), int(query.ziel_index_b))
    ]
    return sum(
        1
        for ratio_interval in goal_pair_intervals.intervals_by_candidate.values()
        if ratio_interval_is_compatible_with_answer(
            ratio_interval=ratio_interval,
            query_value=float(query.value),
            answer=answer,
            tolerance=tolerance,
        )
    )


def compute_terminal_candidate_count_fallback(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    query: Query,
    answer: QueryOperator,
    candidate_subset: list[int] | None,
    config: OptimizedMultistepConfig,
) -> tuple[int, bool]:
    child_answered_queries = answered_queries + [query.answer(answer)]
    child_weight_space = build_weight_space(
        goal_count=alternatives.get_anzahl_spalten(),
        answered_queries=child_answered_queries,
    )
    if not child_weight_space.is_feasible():
        return 0, False

    child_candidates = compute_candidate_set_for_mode(
        alternatives=alternatives,
        weight_space=child_weight_space,
        candidate_subset=candidate_subset,
        config=config,
    )
    return len(child_candidates), True


def ratio_interval_is_compatible_with_answer(
    ratio_interval: RatioInterval,
    query_value: float,
    answer: QueryOperator,
    tolerance: float,
) -> bool:
    lower_value = get_lower_ratio_value_or_none(ratio_interval)
    if lower_value is None:
        return False

    upper_value = get_upper_ratio_value_or_none(ratio_interval)
    upper_is_unbounded = ratio_interval.upper.status == "unbounded"

    if answer == "<":
        return lower_value < query_value - tolerance

    if answer == ">":
        return upper_is_unbounded or (
            upper_value is not None
            and upper_value > query_value + tolerance
        )

    if upper_is_unbounded:
        return lower_value <= query_value + tolerance

    return (
        upper_value is not None
        and lower_value <= query_value + tolerance
        and upper_value >= query_value - tolerance
    )


def get_lower_ratio_value_or_none(ratio_interval: RatioInterval) -> float | None:
    if ratio_interval.lower.status == "unbounded":
        raise ValueError("lower ratio bound must not be unbounded")

    if ratio_interval.lower.status == "infeasible":
        return None

    if ratio_interval.lower.optimal_value is None:
        raise RuntimeError("optimal lower ratio bound has no optimal_value")

    return float(ratio_interval.lower.optimal_value)


def get_upper_ratio_value_or_none(ratio_interval: RatioInterval) -> float | None:
    if ratio_interval.upper.status == "unbounded":
        return None

    if ratio_interval.upper.status == "infeasible":
        return None

    if ratio_interval.upper.optimal_value is None:
        raise RuntimeError("optimal upper ratio bound has no optimal_value")

    return float(ratio_interval.upper.optimal_value)


def partition_samples_by_query_answer(
    query: Query,
    samples: list[list[float]],
    equality_tol: float,
) -> dict[QueryOperator, list[list[float]]]:
    if not samples:
        raise ValueError("samples must not be empty")

    partitioned_samples: dict[QueryOperator, list[list[float]]] = {
        answer: []
        for answer in ANSWER_OPTIONS
    }
    for weights in samples:
        answer = classify_query_answer(
            weights=weights,
            query=query,
            equality_tol=equality_tol,
        )
        partitioned_samples[answer].append(weights)

    return partitioned_samples
