from __future__ import annotations

"""Experimental one/two-step candidate reduction per expected query.

The denominator counts questions, not CPU time.  At horizon two, a child
executes one further question exactly when its existing depth-one planner
returns a query.  All such continuations cost one, so minimizing their
terminal candidate count also maximizes the complete root plan's rate.
This equivalence does not extend to longer horizons with varying child costs.
Volume-confidence termination is outside the default planning model.  The
opt-in stop-aware model instead treats the existing termination criterion as
absorbing and scores an effective terminal loss, while retaining the genuine
terminal candidate expectation separately.
"""

from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
import math
import time

from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query, QueryOperator
from multistep.src.ratio_intervals import GoalPairRatioIntervals
from multistep.src.polytope_volume import compute_exact_query_answer_probabilities
from multistep.src.query_probability import ANSWER_OPTIONS
from multistep.src.value_function import QueryBranchResult, QueryEvaluation, ValueFunctionResult, query_identity_sort_key
from multistep.src.weight_space import build_weight_space

from .value_function import (
    OptimizedMultistepConfig,
    CandidateAnalysis,
    StateAnalysis,
    compute_candidate_analysis_for_mode,
    compute_query_candidates_for_depth_optimized,
    evaluate_query_candidate_optimized,
    evaluate_volume_confidence_termination,
    filter_candidate_subset_for_query_answer,
    filter_already_answered_queries,
    resolve_grid_depth_query_source_mode,
)
from .profiling import increment_profile_counter, profile_operation


@dataclass(frozen=True)
class QueryEfficiencyContinuation:
    answer: QueryOperator
    probability: float
    candidate_count: int
    expected_terminal_candidates: float
    evaluation: QueryEvaluation | None
    expected_terminal_loss: float | None = None
    stop_reason: str | None = None
    maximum_volume_share: float | None = None


@dataclass(frozen=True)
class StopAwareBranch:
    """Proof for a hypothetical answer, before any planned continuation."""

    answer: QueryOperator
    probability: float
    candidate_count: int | None
    expected_terminal_candidates: float
    expected_terminal_loss: float
    stop_reason: str | None
    maximum_volume_share: float | None
    expected_followup_queries: int


@dataclass(frozen=True)
class StopAwareQueryEvaluation(QueryEvaluation):
    """``expected_value`` remains E[actual C]; loss determines selection."""

    expected_terminal_loss: float = 0.0
    stop_aware_branches: tuple[StopAwareBranch, ...] = ()


@dataclass(frozen=True)
class QueryEfficiencyEvaluation:
    evaluation: QueryEvaluation
    horizon: int
    expected_queries: float
    gain: float
    rate: float
    continuations: tuple[QueryEfficiencyContinuation, ...] = ()
    expected_terminal_loss: float | None = None

    @property
    def stop_aware(self) -> bool:
        return self.expected_terminal_loss is not None


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


def _stop_aware_state_status(
    analysis: CandidateAnalysis,
    config: OptimizedMultistepConfig,
) -> tuple[str | None, float | None]:
    """Use exactly the same confidence decision as actual termination."""

    shares = analysis.candidate_volume_shares
    maximum_share = (
        max(float(shares[candidate]) for candidate in analysis.candidates)
        if shares is not None and analysis.candidates else None
    )
    if len(analysis.candidates) == 1:
        return "single_candidate", maximum_share
    if not analysis.candidates:
        return None, maximum_share
    increment_profile_counter("stop_aware_volume_termination_checks")
    with profile_operation("stop_aware_volume_termination"):
        decision = evaluate_volume_confidence_termination(
            analysis, config.volume_confidence_threshold,
        )
    return ("volume_confidence" if decision is not None else None), maximum_share


def _validate_stop_aware_evaluation(evaluation: StopAwareQueryEvaluation) -> None:
    _validate_query_evaluation(evaluation)
    if len(evaluation.branches) != len(evaluation.stop_aware_branches):
        raise RuntimeError("missing stop-aware branch metadata")
    for branch, proof in zip(evaluation.branches, evaluation.stop_aware_branches, strict=True):
        if (
            branch.answer != proof.answer
            or branch.probability != proof.probability
            or branch.child_candidate_count != proof.candidate_count
            or branch.child_value != proof.expected_terminal_candidates
            or proof.expected_followup_queries not in (0, 1)
        ):
            raise RuntimeError("inconsistent stop-aware branch metadata")
        if branch.probability <= 0.0:
            continue
        count = proof.candidate_count
        loss = proof.expected_terminal_loss
        if (
            count is None or not math.isfinite(loss)
            or not 1.0 - 1e-9 <= loss <= float(count) + 1e-9
            or loss > proof.expected_terminal_candidates + 1e-9
        ):
            raise RuntimeError("invalid stop-aware terminal loss")
        if proof.maximum_volume_share is not None and (
            not math.isfinite(proof.maximum_volume_share)
            or not 0.0 <= proof.maximum_volume_share <= 1.0 + 1e-12
        ):
            raise RuntimeError("invalid stop-aware maximum candidate volume share")
        if proof.stop_reason is not None and (
            proof.stop_reason not in ("single_candidate", "volume_confidence")
            or proof.expected_followup_queries != 0
            or loss != 1.0
            or proof.expected_terminal_candidates != float(count)
            or (proof.stop_reason == "single_candidate" and count != 1)
        ):
            raise RuntimeError("invalid absorbing stop-aware branch")
    weighted_loss = math.fsum(
        proof.probability * proof.expected_terminal_loss
        for proof in evaluation.stop_aware_branches if proof.probability > 0.0
    )
    if not math.isfinite(evaluation.expected_terminal_loss) or not math.isclose(
        weighted_loss, evaluation.expected_terminal_loss, abs_tol=1e-12, rel_tol=1e-12,
    ):
        raise RuntimeError("inconsistent stop-aware terminal loss expectation")


def _compute_stop_aware_depth_one_plan(
    *,
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    weight_space: LinearConstraintSystem,
    analysis: CandidateAnalysis,
    config: OptimizedMultistepConfig,
) -> StopAwareQueryEvaluation | None:
    """Each valid one-question continuation costs one; minimize E[loss]."""

    from .value_function import canonical_query_key

    data = compute_query_candidates_for_depth_optimized(
        alternatives, weight_space, analysis.candidates, 1, config,
        precomputed_ratio_intervals=analysis.ratio_intervals,
    )
    queries = data.query_candidates
    if config.filter_answered_query_candidates:
        queries = filter_already_answered_queries(
            queries, answered_queries,
            abs_tol=config.answered_query_abs_tolerance,
            rel_tol=config.answered_query_rel_tolerance,
        )
    evaluations = [
        _evaluate_stop_aware_query(
            alternatives=alternatives, answered_queries=answered_queries,
            weight_space=weight_space, query=query, horizon=1, config=config,
            candidate_subset=analysis.candidates if config.pass_candidate_subset else None,
            ratio_intervals_by_goal_pair=data.ratio_intervals_by_goal_pair,
            query_source=data.query_sources.get(canonical_query_key(query), "unknown"),
        )[0]
        for query in queries
    ]
    return min(
        evaluations,
        key=lambda evaluation: (
            evaluation.expected_terminal_loss,
            *query_identity_sort_key(evaluation.query),
        ),
    ) if evaluations else None


def _evaluate_stop_aware_query(
    *,
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    weight_space: LinearConstraintSystem,
    query: Query,
    horizon: int,
    config: OptimizedMultistepConfig,
    candidate_subset: list[int] | None,
    ratio_intervals_by_goal_pair: dict[tuple[int, int], GoalPairRatioIntervals] | None,
    query_source: str,
) -> tuple[StopAwareQueryEvaluation, float, tuple[QueryEfficiencyContinuation, ...]]:
    """Evaluate all positive-volume branches with exact existing geometry.

    Terminal ratio counts alone cannot decide 99% confidence.  Analyze the
    same candidate regions with volumes, retaining the existing safe parent
    interval filter.  No query values or probability regions are approximated.
    """

    increment_profile_counter("query_evaluations")
    increment_profile_counter("exact_volume_probability_calls")
    with profile_operation("exact_volume_probabilities"):
        probabilities = compute_exact_query_answer_probabilities(
            weight_space=weight_space, query=query,
            tolerance=config.geometry_tolerance,
        )
    branches: list[QueryBranchResult] = []
    proofs: list[StopAwareBranch] = []
    continuations: list[QueryEfficiencyContinuation] = []
    for answer in ANSWER_OPTIONS:
        increment_profile_counter("branch_checks")
        probability = probabilities[answer]
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("invalid branch probability in stop-aware plan")
        if probability == 0.0:
            branches.append(QueryBranchResult(answer, probability, 0.0, None, False))
            proofs.append(StopAwareBranch(answer, probability, None, 0.0, 0.0, None, None, 0))
            continue

        child_answers = answered_queries + [query.answer(answer)]
        child_space = build_weight_space(alternatives.get_anzahl_spalten(), child_answers)
        if not child_space.is_feasible():
            raise RuntimeError("positive-probability branch has no valid candidates in stop-aware plan")
        child_subset = filter_candidate_subset_for_query_answer(
            candidate_subset=candidate_subset, query=query, answer=answer,
            ratio_intervals_by_goal_pair=ratio_intervals_by_goal_pair,
            tolerance=config.ratio_terminal_tolerance,
        )
        increment_profile_counter("stop_aware_candidate_analyses")
        with profile_operation("stop_aware_candidate_analysis"):
            analysis = compute_candidate_analysis_for_mode(
                alternatives, child_space, child_subset, config,
                include_candidate_volumes=config.volume_confidence_threshold is not None,
            )
        count = len(analysis.candidates)
        if count <= 0:
            raise RuntimeError("positive-probability branch has no valid candidates in stop-aware plan")
        stop_reason, maximum_share = _stop_aware_state_status(analysis, config)
        continuation = None
        if horizon == 2 and stop_reason is None:
            continuation = _compute_stop_aware_depth_one_plan(
                alternatives=alternatives, answered_queries=child_answers,
                weight_space=child_space, analysis=analysis, config=config,
            )
        actual_expectation = (
            continuation.expected_value if continuation is not None else float(count)
        )
        terminal_loss = (
            continuation.expected_terminal_loss if continuation is not None
            else 1.0 if stop_reason is not None else float(count)
        )
        followup_queries = int(continuation is not None)
        branches.append(QueryBranchResult(answer, probability, actual_expectation, count, True))
        proofs.append(StopAwareBranch(
            answer, probability, count, actual_expectation, terminal_loss,
            stop_reason, maximum_share, followup_queries,
        ))
        if horizon == 2:
            continuations.append(QueryEfficiencyContinuation(
                answer=answer, probability=probability, candidate_count=count,
                expected_terminal_candidates=actual_expectation,
                evaluation=continuation, expected_terminal_loss=terminal_loss,
                stop_reason=stop_reason, maximum_volume_share=maximum_share,
            ))

    evaluation = StopAwareQueryEvaluation(
        query=query,
        expected_value=math.fsum(branch.probability * branch.child_value for branch in branches),
        branches=tuple(branches), query_source=query_source,
        expected_terminal_loss=math.fsum(proof.probability * proof.expected_terminal_loss for proof in proofs),
        stop_aware_branches=tuple(proofs),
    )
    _validate_stop_aware_evaluation(evaluation)
    expected_queries = 1.0 + math.fsum(
        proof.probability * proof.expected_followup_queries for proof in proofs
    )
    if not math.isfinite(expected_queries) or not 1.0 <= expected_queries <= horizon + 1e-12:
        raise RuntimeError("invalid expected question count in stop-aware plan")
    return evaluation, expected_queries, tuple(continuations)


def _evaluate_stop_aware_efficiency_worker(
    payload: RootPayload,
) -> tuple[StopAwareQueryEvaluation, float, tuple[QueryEfficiencyContinuation, ...]]:
    alternatives, answers, weight_space, query, samples, horizon, config, subset, intervals, source = payload
    if horizon not in (1, 2):
        raise ValueError("query-efficiency planning supports only horizons 1 and 2")
    return _evaluate_stop_aware_query(
        alternatives=alternatives, answered_queries=answers,
        weight_space=weight_space, query=query, horizon=horizon, config=config,
        candidate_subset=subset, ratio_intervals_by_goal_pair=intervals,
        query_source=source,
    )


def _evaluate_efficiency_worker(
    payload: RootPayload,
) -> tuple[QueryEvaluation, float, tuple[QueryEfficiencyContinuation, ...]]:
    if payload[6].query_efficiency_stop_aware:
        return _evaluate_stop_aware_efficiency_worker(payload)
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
        terminal_loss = (
            evaluation.expected_terminal_loss
            if isinstance(evaluation, StopAwareQueryEvaluation) else None
        )
        objective_value = terminal_loss if terminal_loss is not None else evaluation.expected_value
        gain = float(len(candidates)) - objective_value
        rate = gain / expected_queries
        if not math.isfinite(gain) or not math.isfinite(rate):
            raise RuntimeError("nonfinite query-efficiency score")
        if objective_value < 1.0 - 1e-9 or gain < -1e-9:
            raise RuntimeError("terminal candidate expectation is outside feasible query-efficiency bounds")
        scored.append(QueryEfficiencyEvaluation(
            evaluation, horizon, expected_queries, gain, rate, continuations,
            expected_terminal_loss=terminal_loss,
        ))
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
    analysis = state.candidate_analysis
    need_volumes = config.query_efficiency_stop_aware and config.volume_confidence_threshold is not None
    if analysis is None or (need_volumes and analysis.candidate_volumes is None):
        analysis = compute_candidate_analysis_for_mode(
            alternatives, state.weight_space, candidate_subset, config,
            include_candidate_volumes=need_volumes,
        )
    candidates = analysis.candidates
    if len(candidates) <= 1:
        return QueryEfficiencyResult(len(candidates), True, None, {}, {}, {})
    if config.query_efficiency_stop_aware and _stop_aware_state_status(analysis, config)[0] is not None:
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
        serialized = {
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
        if isinstance(evaluation, StopAwareQueryEvaluation):
            serialized["expected_terminal_loss"] = evaluation.expected_terminal_loss
            serialized["expected_value_semantics"] = "expected_actual_terminal_candidate_count"
            for branch, proof in zip(serialized["branches"], evaluation.stop_aware_branches, strict=True):
                branch.update({
                    "expected_terminal_candidates": proof.expected_terminal_candidates,
                    "expected_terminal_loss": proof.expected_terminal_loss,
                    "stop_reason": proof.stop_reason,
                    "maximum_volume_share": proof.maximum_volume_share,
                    "expected_followup_queries": proof.expected_followup_queries,
                })
        return serialized

    serialized = {
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
    if plan.stop_aware:
        serialized.update({
            "planning_stop_aware": True,
            "expected_terminal_loss": plan.expected_terminal_loss,
            "expected_value_semantics": "expected_actual_terminal_candidate_count",
        })
        for child, proof in zip(serialized["continuations"], plan.continuations, strict=True):
            child.update({
                "expected_terminal_loss": proof.expected_terminal_loss,
                "stop_reason": proof.stop_reason,
                "maximum_volume_share": proof.maximum_volume_share,
            })
    return serialized
