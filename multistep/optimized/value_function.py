from __future__ import annotations

"""Optimized value-function implementations.

The reference implementation remains in ``multistep.src.value_function``.
This module reuses the core domain functions but changes the evaluation
strategy to avoid unnecessary recursive work.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
from typing import Literal

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
from multistep.src.query_probability import ANSWER_OPTIONS, classify_query_answer
from multistep.src.ratio_intervals import (
    compute_all_ratio_intervals,
    compute_ratio_bounds_for_weight_space,
)
from multistep.src.ratio_intervals import GoalPairRatioIntervals, RatioInterval
from multistep.src.sampling import sample_points_from_constraint_system
from multistep.src.value_function import (
    QueryBranchResult,
    QueryEvaluation,
    ValueFunctionResult,
)
from multistep.src.weight_space import build_weight_space


CandidateCountMode = Literal["closed_lp", "ratio_relevant"]
QuerySource = Literal["grid", "ratio", "grid+ratio", "unknown"]


@dataclass(frozen=True)
class OptimizedMultistepConfig:
    sample_count: int = 1000
    burn_in: int = 200
    thinning: int = 5
    random_seed: int | None = None
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
    parallelize_root: bool = False
    max_workers: int = 4
    candidate_count_mode: CandidateCountMode = "ratio_relevant"
    include_ratio_queries_on_grid_depths: bool = True

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")

        if self.burn_in < 0:
            raise ValueError("burn_in must not be negative")

        if self.thinning <= 0:
            raise ValueError("thinning must be positive")

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

        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")

        if self.candidate_count_mode not in {"closed_lp", "ratio_relevant"}:
            raise ValueError("candidate_count_mode must be 'closed_lp' or 'ratio_relevant'")


def compute_value_function_optimized(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    remaining_depth: int,
    config: OptimizedMultistepConfig | None = None,
    candidate_subset: list[int] | None = None,
    samples: list[list[float]] | None = None,
    is_root_call: bool = True,
) -> ValueFunctionResult:
    if remaining_depth < 0:
        raise ValueError("remaining_depth must not be negative")

    resolved_config = config or OptimizedMultistepConfig()
    weight_space = build_weight_space(
        goal_count=alternatives.get_anzahl_spalten(),
        answered_queries=answered_queries,
    )

    if not weight_space.is_feasible():
        return ValueFunctionResult(
            remaining_depth=remaining_depth,
            value=0.0,
            best_query=None,
            candidate_count=0,
            query_evaluations=(),
            is_feasible=False,
        )

    candidates = compute_candidate_set_for_mode(
        alternatives=alternatives,
        weight_space=weight_space,
        candidate_subset=candidate_subset,
        config=resolved_config,
    )
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

    query_candidate_data = compute_query_candidates_for_depth_optimized(
        alternatives=alternatives,
        weight_space=weight_space,
        candidates=candidates,
        remaining_depth=remaining_depth,
        config=resolved_config,
    )
    query_candidates = query_candidate_data.query_candidates
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

    state_samples = resolve_state_samples(
        weight_space=weight_space,
        samples=samples,
        config=resolved_config,
    )
    query_evaluations = evaluate_query_candidates_optimized(
        alternatives=alternatives,
        answered_queries=answered_queries,
        query_candidates=query_candidates,
        samples=state_samples,
        remaining_depth=remaining_depth,
        config=resolved_config,
        candidate_subset=candidates if resolved_config.pass_candidate_subset else None,
        ratio_intervals_by_goal_pair=query_candidate_data.ratio_intervals_by_goal_pair,
        query_sources=query_candidate_data.query_sources,
        parallelize=(
            is_root_call
            and resolved_config.parallelize_root
            and resolved_config.max_workers > 1
            and len(query_candidates) > 1
        ),
    )
    best_evaluation = min(query_evaluations, key=query_evaluation_sort_key)

    return ValueFunctionResult(
        remaining_depth=remaining_depth,
        value=best_evaluation.expected_value,
        best_query=best_evaluation.query,
        candidate_count=candidate_count,
        query_evaluations=query_evaluations,
        is_feasible=True,
    )


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
    if config.candidate_count_mode == "closed_lp":
        return compute_candidate_set_for_subset(
            alternatives=alternatives,
            weight_space=weight_space,
            candidate_subset=candidate_subset,
        )

    return compute_ratio_relevant_candidate_set(
        alternatives=alternatives,
        weight_space=weight_space,
        candidate_subset=candidate_subset,
        tolerance=config.ratio_terminal_tolerance,
    )


def compute_ratio_relevant_candidate_set(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidate_subset: list[int] | None = None,
    tolerance: float = 1e-12,
) -> list[int]:
    candidates_to_check = (
        list(range(alternatives.get_anzahl_zeilen()))
        if candidate_subset is None
        else list(candidate_subset)
    )
    if not candidates_to_check:
        return []

    ratio_intervals = compute_all_ratio_intervals(
        alternatives=alternatives,
        weight_space=weight_space,
        candidates=candidates_to_check,
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
        return [
            candidate_index
            for candidate_index in candidates_to_check
            if candidate_index in relevant_candidates
        ]

    return [
        candidate_index
        for candidate_index in candidates_to_check
        if candidate_index in feasible_candidates
    ]


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
) -> tuple[float, float, int, int, int, float]:
    return (
        float(evaluation.expected_value),
        compute_expected_immediate_candidate_count(evaluation),
        compute_max_immediate_candidate_count(evaluation),
        int(evaluation.query.ziel_index_a),
        int(evaluation.query.ziel_index_b),
        float(evaluation.query.value),
    )


def compute_expected_immediate_candidate_count(evaluation: QueryEvaluation) -> float:
    expected_count = 0.0
    for branch in evaluation.branches:
        if branch.probability == 0.0:
            continue

        child_candidate_count = branch.child_candidate_count
        if child_candidate_count is None:
            child_candidate_count = int(round(branch.child_value))

        expected_count += branch.probability * child_candidate_count

    return expected_count


def compute_max_immediate_candidate_count(evaluation: QueryEvaluation) -> int:
    max_count = 0
    for branch in evaluation.branches:
        if branch.probability == 0.0:
            continue

        child_candidate_count = branch.child_candidate_count
        if child_candidate_count is None:
            child_candidate_count = int(round(branch.child_value))

        max_count = max(max_count, child_candidate_count)

    return max_count


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
) -> "QueryCandidateData":
    if remaining_depth <= 0:
        return QueryCandidateData(query_candidates=[])

    if remaining_depth == 1:
        ratio_intervals = compute_all_ratio_intervals(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=candidates,
        )
        ratio_queries = compute_onestep_query_candidates(
            goal_pair_ratio_intervals=ratio_intervals,
            epsilon=config.query_epsilon,
        )
        query_candidates, query_sources = merge_query_candidates_by_source(
            grid_queries=[],
            ratio_queries=ratio_queries,
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

    if config.canonical_grid_goal_pairs_only:
        grid_queries = compute_canonical_grid_query_candidates(
            weight_space=weight_space,
            grid_size=config.grid_size,
            min_query_value=config.min_query_value,
            max_query_value=config.max_query_value,
            spacing=config.grid_spacing,
        )
    else:
        grid_queries = compute_grid_query_candidates(
            weight_space=weight_space,
            grid_size=config.grid_size,
            min_query_value=config.min_query_value,
            max_query_value=config.max_query_value,
            spacing=config.grid_spacing,
        )

    ratio_intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals] | None
    ratio_intervals_by_goal_pair = None
    ratio_queries: list[Query] = []
    if config.include_ratio_queries_on_grid_depths:
        ratio_intervals = compute_all_ratio_intervals(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=candidates,
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

    query_candidates, query_sources = merge_query_candidates_by_source(
        grid_queries=grid_queries,
        ratio_queries=ratio_queries,
    )
    return QueryCandidateData(
        query_candidates=query_candidates,
        query_sources=query_sources,
        ratio_intervals_by_goal_pair=ratio_intervals_by_goal_pair,
    )


def merge_query_candidates_by_source(
    grid_queries: list[Query],
    ratio_queries: list[Query],
) -> tuple[list[Query], dict[tuple[int, int, float], QuerySource]]:
    sources_by_key: dict[tuple[int, int, float], set[QuerySource]] = {}
    for query in grid_queries:
        sources_by_key.setdefault(canonical_query_key(query), set()).add("grid")

    for query in ratio_queries:
        sources_by_key.setdefault(canonical_query_key(query), set()).add("ratio")

    query_candidates = deduplicate_mirrored_query_candidates(
        grid_queries + ratio_queries
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


def combine_query_sources(sources: set[QuerySource]) -> QuerySource:
    if "grid" in sources and "ratio" in sources:
        return "grid+ratio"

    if "grid" in sources:
        return "grid"

    if "ratio" in sources:
        return "ratio"

    return "unknown"


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

    return sample_points_from_constraint_system(
        system=weight_space,
        num_samples=config.sample_count,
        burn_in=config.burn_in,
        thinning=config.thinning,
        seed=config.random_seed,
    )


def evaluate_query_candidates_optimized(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    query_candidates: list[Query],
    samples: list[list[float]],
    remaining_depth: int,
    config: OptimizedMultistepConfig,
    candidate_subset: list[int] | None,
    ratio_intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals] | None,
    query_sources: dict[tuple[int, int, float], QuerySource],
    parallelize: bool,
) -> tuple[QueryEvaluation, ...]:
    if not parallelize:
        return tuple(
            evaluate_query_candidate_optimized(
                alternatives=alternatives,
                answered_queries=answered_queries,
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

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        return tuple(
            executor.map(
                _evaluate_query_candidate_worker,
                [
                    (
                        alternatives,
                        answered_queries,
                        query,
                        samples,
                        remaining_depth,
                        config,
                        candidate_subset,
                        ratio_intervals_by_goal_pair,
                        query_sources.get(canonical_query_key(query), "unknown"),
                    )
                    for query in query_candidates
                ],
            )
        )


def _evaluate_query_candidate_worker(
    payload: tuple[
        AlternativenMatrix,
        list[AnsweredQuery],
        Query,
        list[list[float]],
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
        query=query,
        samples=samples,
        remaining_depth=remaining_depth,
        config=config,
        candidate_subset=candidate_subset,
        ratio_intervals_by_goal_pair=ratio_intervals_by_goal_pair,
        query_source=query_source,
    )


def evaluate_query_candidate_optimized(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    query: Query,
    samples: list[list[float]],
    remaining_depth: int,
    config: OptimizedMultistepConfig,
    candidate_subset: list[int] | None,
    ratio_intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals] | None = None,
    query_source: QuerySource = "unknown",
) -> QueryEvaluation:
    if remaining_depth <= 0:
        raise ValueError("remaining_depth must be positive")

    partitioned_samples = partition_samples_by_query_answer(
        query=query,
        samples=samples,
        equality_tol=config.equality_tol,
    )
    sample_count = len(samples)
    probabilities = {
        answer: len(partitioned_samples[answer]) / sample_count
        for answer in ANSWER_OPTIONS
    }
    branches: list[QueryBranchResult] = []
    expected_value = 0.0

    for answer in ANSWER_OPTIONS:
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
            child_value = float(child_candidate_count)
            is_child_feasible = child_candidate_count > 0
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
        if child_weight_space.is_feasible():
            child_samples = (
                partitioned_samples[answer]
                if config.reuse_conditioned_samples
                else None
            )
            child_result = compute_value_function_optimized(
                alternatives=alternatives,
                answered_queries=child_answered_queries,
                remaining_depth=remaining_depth - 1,
                config=config,
                candidate_subset=candidate_subset,
                samples=child_samples,
                is_root_call=False,
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
