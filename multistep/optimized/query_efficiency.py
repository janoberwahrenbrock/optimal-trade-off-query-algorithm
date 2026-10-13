from __future__ import annotations

"""Experimental one/two-step candidate reduction per expected query.

The denominator counts questions, not CPU time.  At horizon two, a child
executes one further question exactly when its existing depth-one planner
returns a query.  All such continuations cost one, so minimizing their
terminal candidate count also maximizes the complete root plan's rate.
This equivalence does not extend to longer horizons with varying child costs.
Volume-confidence termination is deliberately outside this planning model.
"""

from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
import math
import time

from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query, QueryOperator
from multistep.src.ratio_intervals import GoalPairRatioIntervals
from multistep.src.value_function import QueryEvaluation, ValueFunctionResult, query_identity_sort_key
from multistep.src.weight_space import build_weight_space

from .value_function import (
    OptimizedMultistepConfig,
    StateAnalysis,
    compute_candidate_analysis_for_mode,
    compute_query_candidates_for_depth_optimized,
    evaluate_query_candidate_optimized,
    filter_already_answered_queries,
    resolve_grid_depth_query_source_mode,
)


@dataclass(frozen=True)
class QueryEfficiencyContinuation:
    answer: QueryOperator
    probability: float
    candidate_count: int
    expected_terminal_candidates: float
    evaluation: QueryEvaluation | None


@dataclass(frozen=True)
class QueryEfficiencyEvaluation:
    evaluation: QueryEvaluation
    horizon: int
    expected_queries: float
    gain: float
    rate: float
    continuations: tuple[QueryEfficiencyContinuation, ...] = ()


@dataclass(frozen=True)
class QueryEfficiencyResult:
    candidate_count: int
    is_feasible: bool
    choice: QueryEfficiencyEvaluation | None
    evaluations_by_horizon: dict[int, tuple[QueryEfficiencyEvaluation, ...]]
    best_by_horizon: dict[int, QueryEfficiencyEvaluation | None]
    evaluation_seconds_by_horizon: dict[int, float]


# Preserve the ordinary root worker's ten-element payload layout for tooling.
RootPayload = tuple[
    AlternativenMatrix, list[AnsweredQuery], LinearConstraintSystem, Query,
    None, int, OptimizedMultistepConfig, list[int] | None,
    dict[tuple[int, int], GoalPairRatioIntervals] | None, str,
]


def query_efficiency_sort_key(
    evaluation: QueryEfficiencyEvaluation,
) -> tuple[float | int, ...]:
    return (
        -evaluation.rate,
        evaluation.expected_queries,
        evaluation.horizon,
        *query_identity_sort_key(evaluation.evaluation.query),
    )


def _validate_query_evaluation(evaluation: QueryEvaluation) -> None:
    """Fail on contradictory positive-probability states in this experiment."""

    for branch in evaluation.branches:
        if not math.isfinite(branch.probability) or not 0.0 <= branch.probability <= 1.0:
            raise RuntimeError("invalid branch probability in query-efficiency plan")
        if branch.probability > 0.0 and (
            not branch.is_child_feasible
            or branch.child_candidate_count is None
            or branch.child_candidate_count <= 0
            or not math.isfinite(branch.child_value)
            or branch.child_value < 1.0 - 1e-9
        ):
            raise RuntimeError("positive-probability branch has no valid candidates in query-efficiency plan")
    if not math.isclose(math.fsum(branch.probability for branch in evaluation.branches), 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise RuntimeError("query-efficiency branch probabilities do not sum to one")
    weighted_value = math.fsum(
        branch.probability * branch.child_value
        for branch in evaluation.branches if branch.probability > 0.0
    )
    if not math.isfinite(evaluation.expected_value) or not math.isclose(
        weighted_value, evaluation.expected_value, abs_tol=1e-12, rel_tol=1e-12,
    ):
        raise RuntimeError("inconsistent query-efficiency terminal expectation")


def _evaluate_efficiency_worker(
    payload: RootPayload,
) -> tuple[QueryEvaluation, float, tuple[QueryEfficiencyContinuation, ...]]:
    alternatives, answers, weight_space, query, samples, horizon, config, subset, intervals, source = payload
    if horizon not in (1, 2):
        raise ValueError("query-efficiency planning supports only horizons 1 and 2")
    continuations: list[QueryEfficiencyContinuation] = []

    def observe_child(answer: QueryOperator, probability: float, child: ValueFunctionResult) -> None:
        chosen = (
            next(item for item in child.query_evaluations if item.query == child.best_query)
            if child.best_query is not None
            else None
        )
        continuations.append(QueryEfficiencyContinuation(
            answer=answer,
            probability=probability,
            candidate_count=child.candidate_count,
            expected_terminal_candidates=child.value,
            evaluation=chosen,
        ))

    evaluation = evaluate_query_candidate_optimized(
        alternatives=alternatives,
        answered_queries=answers,
        weight_space=weight_space,
        query=query,
        samples=samples,
        remaining_depth=horizon,
        config=config,
        candidate_subset=subset,
        ratio_intervals_by_goal_pair=intervals,
        query_source=source,
        _child_result_observer=observe_child if horizon == 2 else None,
    )
    _validate_query_evaluation(evaluation)
    if horizon == 2:
        for branch in evaluation.branches:
            if branch.probability <= 0.0:
                continue
            observed = [child for child in continuations if child.answer == branch.answer]
            if len(observed) != 1:
                raise RuntimeError("query-efficiency child plan was not observed exactly once")
            child = observed[0]
            if child.evaluation is not None:
                _validate_query_evaluation(child.evaluation)
            terminal_value = (
                child.evaluation.expected_value if child.evaluation is not None
                else float(child.candidate_count)
            )
            if (
                child.probability != branch.probability
                or child.candidate_count != branch.child_candidate_count
                or child.expected_terminal_candidates != branch.child_value
                or terminal_value != child.expected_terminal_candidates
                or (child.candidate_count <= 1 and child.evaluation is not None)
            ):
                raise RuntimeError("inconsistent query-efficiency child plan")
    expected_queries = 1.0 + math.fsum(
        continuation.probability
        for continuation in continuations
        if continuation.evaluation is not None
    )
    if not math.isfinite(expected_queries) or not 1.0 <= expected_queries <= horizon + 1e-12:
        raise RuntimeError("invalid expected question count in query-efficiency plan")
    return evaluation, expected_queries, tuple(continuations)


def _evaluate_horizon_candidates(
    *,
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    weight_space: LinearConstraintSystem,
    query_candidates: list[Query],
    horizon: int,
    config: OptimizedMultistepConfig,
    candidates: list[int],
    ratio_intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals] | None,
    query_sources: dict[tuple[int, int, float], str],
    executor: Executor | None,
) -> tuple[QueryEfficiencyEvaluation, ...]:
    from .value_function import canonical_query_key

    payloads: list[RootPayload] = [
        (alternatives, answered_queries, weight_space, query, None, horizon, config,
         candidates if config.pass_candidate_subset else None,
         ratio_intervals_by_goal_pair, query_sources.get(canonical_query_key(query), "unknown"))
        for query in query_candidates
    ]
    raw = (
        executor.map(_evaluate_efficiency_worker, payloads)
        if executor is not None and config.parallelize_root and config.max_workers > 1 and len(payloads) > 1
        else map(_evaluate_efficiency_worker, payloads)
    )
    scored: list[QueryEfficiencyEvaluation] = []
    for evaluation, expected_queries, continuations in raw:
        gain = float(len(candidates)) - evaluation.expected_value
        rate = gain / expected_queries
        if not math.isfinite(gain) or not math.isfinite(rate):
            raise RuntimeError("nonfinite query-efficiency score")
        if evaluation.expected_value < 1.0 - 1e-9 or gain < -1e-9:
            raise RuntimeError("terminal candidate expectation is outside feasible query-efficiency bounds")
        scored.append(QueryEfficiencyEvaluation(evaluation, horizon, expected_queries, gain, rate, continuations))
    return tuple(scored)


def compute_query_efficiency(
    *,
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    config: OptimizedMultistepConfig,
    max_horizon: int = 2,
    candidate_subset: list[int] | None = None,
    executor: Executor | None = None,
    precomputed_state_analysis: StateAnalysis | None = None,
) -> QueryEfficiencyResult:
    """Rank every allowed root query/horizon by a ratio of expectations."""

    if type(max_horizon) is not int or max_horizon not in (1, 2):
        raise ValueError("query-efficiency planning supports only horizons 1 and 2")
    if config.answer_probability_mode != "exact_volume":
        raise ValueError("query-efficiency planning requires exact_volume probabilities")
    if config.depth_one_query_source_mode != "ratio" or resolve_grid_depth_query_source_mode(config) != "both":
        raise ValueError("query-efficiency planning requires ratio at depth one and grid plus ratios above it")
    if config.adaptive_depth_candidate_threshold is not None:
        raise ValueError("query-efficiency planning cannot reduce hypothetical depths adaptively")

    state = precomputed_state_analysis
    if state is None:
        weight_space = build_weight_space(alternatives.get_anzahl_spalten(), answered_queries)
        state = StateAnalysis(weight_space, weight_space.is_feasible())
    if not state.is_feasible:
        return QueryEfficiencyResult(0, False, None, {}, {}, {})
    analysis = state.candidate_analysis or compute_candidate_analysis_for_mode(
        alternatives, state.weight_space, candidate_subset, config,
    )
    candidates = analysis.candidates
    if len(candidates) <= 1:
        return QueryEfficiencyResult(len(candidates), True, None, {}, {}, {})

    def evaluate_with_executor(active_executor: Executor | None) -> QueryEfficiencyResult:
        evaluations: dict[int, tuple[QueryEfficiencyEvaluation, ...]] = {}
        best: dict[int, QueryEfficiencyEvaluation | None] = {}
        seconds: dict[int, float] = {}
        for horizon in range(1, max_horizon + 1):
            started = time.perf_counter()
            data = compute_query_candidates_for_depth_optimized(
                alternatives, state.weight_space, candidates, horizon, config,
                precomputed_ratio_intervals=analysis.ratio_intervals,
            )
            queries = data.query_candidates
            if config.filter_answered_query_candidates:
                queries = filter_already_answered_queries(
                    queries, answered_queries,
                    abs_tol=config.answered_query_abs_tolerance,
                    rel_tol=config.answered_query_rel_tolerance,
                )
            evaluated = _evaluate_horizon_candidates(
                alternatives=alternatives, answered_queries=answered_queries,
                weight_space=state.weight_space, query_candidates=queries,
                horizon=horizon, config=config, candidates=candidates,
                ratio_intervals_by_goal_pair=data.ratio_intervals_by_goal_pair,
                query_sources=data.query_sources, executor=active_executor,
            )
            evaluations[horizon] = evaluated
            best[horizon] = min(evaluated, key=query_efficiency_sort_key) if evaluated else None
            seconds[horizon] = time.perf_counter() - started
        choices = [item for item in best.values() if item is not None]
        return QueryEfficiencyResult(
            len(candidates), True,
            min(choices, key=query_efficiency_sort_key) if choices else None,
            evaluations, best, seconds,
        )

    if executor is None and config.parallelize_root and config.max_workers > 1:
        with ProcessPoolExecutor(max_workers=config.max_workers) as local_executor:
            return evaluate_with_executor(local_executor)
    return evaluate_with_executor(executor)


def query_efficiency_evaluation_to_json(plan: QueryEfficiencyEvaluation) -> dict:
    """Include the complete selected two-question policy, not unchosen trees."""

    def query_json(evaluation: QueryEvaluation | None) -> dict | None:
        if evaluation is None:
            return None
        query = evaluation.query
        return {
            "goal_index_a": int(query.ziel_index_a),
            "goal_index_b": int(query.ziel_index_b), "value": float(query.value),
            "query_source": evaluation.query_source,
            "expected_terminal_candidates": evaluation.expected_value,
            "branches": [
                {"answer": branch.answer, "probability": branch.probability,
                 "child_value": branch.child_value,
                 "child_candidate_count": branch.child_candidate_count,
                 "is_child_feasible": branch.is_child_feasible}
                for branch in evaluation.branches
            ],
        }

    return {
        "horizon": plan.horizon,
        "query": query_json(plan.evaluation),
        "expected_terminal_candidates": plan.evaluation.expected_value,
        "expected_queries": plan.expected_queries,
        "gain": plan.gain, "rate": plan.rate,
        "continuations": [
            {"answer": child.answer, "probability": child.probability,
             "candidate_count": child.candidate_count,
             "expected_terminal_candidates": child.expected_terminal_candidates,
             "expected_queries": 1 if child.evaluation is not None else 0,
             "query": query_json(child.evaluation)}
            for child in plan.continuations
        ],
    }
