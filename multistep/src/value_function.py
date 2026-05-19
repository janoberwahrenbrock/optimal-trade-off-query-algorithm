from __future__ import annotations

from dataclasses import dataclass

from .candidates import compute_candidate_set
from .grid_query_candidates import (
    DEFAULT_GRID_SIZE,
    DEFAULT_GRID_SPACING,
    DEFAULT_MAX_QUERY_VALUE,
    DEFAULT_MIN_QUERY_VALUE,
    GridSpacing,
    compute_grid_query_candidates,
)
from .linear_constraints import LinearConstraintSystem
from .models import AlternativenMatrix, AnsweredQuery, Query, QueryOperator
from .onestep_query_candidates import QUERY_EPSILON, compute_onestep_query_candidates
from .query_probability import ANSWER_OPTIONS, estimate_query_answer_probabilities
from .ratio_intervals import compute_all_ratio_intervals
from .sampling import sample_points_from_constraint_system
from .weight_space import build_weight_space


@dataclass(frozen=True)
class MultistepConfig:
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


@dataclass(frozen=True)
class QueryBranchResult:
    answer: QueryOperator
    probability: float
    child_value: float
    child_candidate_count: int | None
    is_child_feasible: bool


@dataclass(frozen=True)
class QueryEvaluation:
    query: Query
    expected_value: float
    branches: tuple[QueryBranchResult, ...]
    query_source: str = "unknown"


@dataclass(frozen=True)
class ValueFunctionResult:
    remaining_depth: int
    value: float
    best_query: Query | None
    candidate_count: int
    query_evaluations: tuple[QueryEvaluation, ...]
    is_feasible: bool


def compute_value_function(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    remaining_depth: int,
    config: MultistepConfig | None = None,
) -> ValueFunctionResult:
    if remaining_depth < 0:
        raise ValueError("remaining_depth must not be negative")

    resolved_config = config or MultistepConfig()
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

    candidates = compute_candidate_set(
        alternatives=alternatives,
        weight_space=weight_space,
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

    query_candidates = compute_query_candidates_for_depth(
        alternatives=alternatives,
        weight_space=weight_space,
        candidates=candidates,
        remaining_depth=remaining_depth,
        config=resolved_config,
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

    samples = sample_points_from_constraint_system(
        system=weight_space,
        num_samples=resolved_config.sample_count,
        burn_in=resolved_config.burn_in,
        thinning=resolved_config.thinning,
        seed=resolved_config.random_seed,
    )
    query_evaluations = tuple(
        evaluate_query_candidate(
            alternatives=alternatives,
            answered_queries=answered_queries,
            query=query,
            samples=samples,
            remaining_depth=remaining_depth,
            config=resolved_config,
        )
        for query in query_candidates
    )
    best_evaluation = min(
        query_evaluations,
        key=lambda evaluation: evaluation.expected_value,
    )

    return ValueFunctionResult(
        remaining_depth=remaining_depth,
        value=best_evaluation.expected_value,
        best_query=best_evaluation.query,
        candidate_count=candidate_count,
        query_evaluations=query_evaluations,
        is_feasible=True,
    )


def compute_query_candidates_for_depth(
    alternatives: AlternativenMatrix,
    weight_space: LinearConstraintSystem,
    candidates: list[int],
    remaining_depth: int,
    config: MultistepConfig,
) -> list[Query]:
    if remaining_depth <= 0:
        return []

    if remaining_depth == 1:
        ratio_intervals = compute_all_ratio_intervals(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=candidates,
        )
        return compute_onestep_query_candidates(
            goal_pair_ratio_intervals=ratio_intervals,
            epsilon=config.query_epsilon,
        )

    return compute_grid_query_candidates(
        weight_space=weight_space,
        grid_size=config.grid_size,
        min_query_value=config.min_query_value,
        max_query_value=config.max_query_value,
        spacing=config.grid_spacing,
    )


def evaluate_query_candidate(
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    query: Query,
    samples: list[list[float]],
    remaining_depth: int,
    config: MultistepConfig,
) -> QueryEvaluation:
    if remaining_depth <= 0:
        raise ValueError("remaining_depth must be positive")

    probabilities = estimate_query_answer_probabilities(
        query=query,
        samples=samples,
        equality_tol=config.equality_tol,
    )
    branches: list[QueryBranchResult] = []
    expected_value = 0.0

    for answer in ANSWER_OPTIONS:
        probability = probabilities[answer]
        child_answered_queries = answered_queries + [query.answer(answer)]
        child_weight_space = build_weight_space(
            goal_count=alternatives.get_anzahl_spalten(),
            answered_queries=child_answered_queries,
        )
        if child_weight_space.is_feasible():
            child_result = compute_value_function(
                alternatives=alternatives,
                answered_queries=child_answered_queries,
                remaining_depth=remaining_depth - 1,
                config=config,
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
    )
