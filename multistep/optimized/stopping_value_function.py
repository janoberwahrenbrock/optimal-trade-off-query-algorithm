from __future__ import annotations

"""Fixed-horizon planning with the actual stopping rule in hypothetical states.

The objective is E[terminal loss], where an absorbing singleton or confidence
state has loss one.  E[the actual candidate count] is retained independently in
the ordinary value/result fields.  This is a separate, explicit policy; the
legacy fixed planner remains unchanged when planning stopping is disabled.
"""

from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
import math

from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query
from multistep.src.polytope_volume import compute_exact_query_answer_probabilities
from multistep.src.query_probability import ANSWER_OPTIONS
from multistep.src.value_function import (
    QueryBranchResult, ValueFunctionResult, query_identity_sort_key,
)
from multistep.src.weight_space import build_weight_space

from .profiling import increment_profile_counter, profile_operation
from .query_efficiency import (
    RootPayload, StopAwareBranch, StopAwareQueryEvaluation,
    _evaluate_stop_aware_query, _stop_aware_state_status,
    _validate_query_evaluation, _validate_stop_aware_evaluation,
)
from .value_function import (
    OptimizedMultistepConfig, StateAnalysis, canonical_query_key,
    compute_candidate_analysis_for_mode,
    compute_query_candidates_for_depth_optimized,
    compute_value_function_optimized,
    filter_already_answered_queries, filter_candidate_subset_for_query_answer,
)


@dataclass(frozen=True)
class FixedStopAwareResult(ValueFunctionResult):
    """``value`` is genuine E[C]; the separate loss chooses the fixed plan."""

    expected_terminal_loss: float = 0.0
    expected_queries: float = 0.0
    planning_stop_aware: bool = True


def _evaluate_fixed_stop_aware_worker(
    payload: RootPayload,
) -> tuple[StopAwareQueryEvaluation, float]:
    alternatives, answers, space, query, _, depth, config, subset, intervals, source = payload
    if depth in (1, 2):
        evaluation, expected_queries, _ = _evaluate_stop_aware_query(
            alternatives=alternatives, answered_queries=answers,
            weight_space=space, query=query, horizon=depth, config=config,
            candidate_subset=subset, ratio_intervals_by_goal_pair=intervals,
            query_source=source,
        )
        return evaluation, expected_queries
    if depth != 3:
        raise ValueError("fixed stop-aware planning supports depths 1, 2 and 3")

    increment_profile_counter("query_evaluations")
    increment_profile_counter("exact_volume_probability_calls")
    with profile_operation("exact_volume_probabilities"):
        probabilities = compute_exact_query_answer_probabilities(
            weight_space=space, query=query, tolerance=config.geometry_tolerance,
        )
    branches: list[QueryBranchResult] = []
    proofs: list[StopAwareBranch] = []
    for answer in ANSWER_OPTIONS:
        increment_profile_counter("branch_checks")
        probability = probabilities[answer]
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("invalid branch probability in fixed stop-aware plan")
        if probability == 0.0:
            branches.append(QueryBranchResult(answer, probability, 0.0, None, False))
            proofs.append(StopAwareBranch(answer, probability, None, 0.0, 0.0, None, None, 0))
            continue
        child_answers = answers + [query.answer(answer)]
        child_space = build_weight_space(alternatives.get_anzahl_spalten(), child_answers)
        if not child_space.is_feasible():
            raise RuntimeError("positive-probability branch is infeasible in fixed stop-aware plan")
        child_subset = filter_candidate_subset_for_query_answer(
            candidate_subset=subset, query=query, answer=answer,
            ratio_intervals_by_goal_pair=intervals,
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
            raise RuntimeError("positive-probability branch has no candidates in fixed stop-aware plan")
        stop_reason, maximum_share = _stop_aware_state_status(analysis, config)
        if stop_reason is not None:
            actual_expectation, loss, followups = float(count), 1.0, 0.0
        else:
            child = _compute_fixed_stop_aware(
                alternatives=alternatives, answered_queries=child_answers,
                remaining_depth=depth - 1, config=config,
                executor=None,
                precomputed_state_analysis=StateAnalysis(child_space, True, analysis),
            )
            actual_expectation = child.value
            loss = child.expected_terminal_loss
            followups = child.expected_queries
        branches.append(QueryBranchResult(answer, probability, actual_expectation, count, True))
        proofs.append(StopAwareBranch(
            answer, probability, count, actual_expectation, loss,
            stop_reason, maximum_share, followups,
        ))
    evaluation = StopAwareQueryEvaluation(
        query=query,
        expected_value=math.fsum(branch.probability * branch.child_value for branch in branches),
        branches=tuple(branches), query_source=source,
        expected_terminal_loss=math.fsum(proof.probability * proof.expected_terminal_loss for proof in proofs),
        stop_aware_branches=tuple(proofs),
    )
    _validate_stop_aware_evaluation(evaluation, max_followup_queries=2)
    if (
        not math.isfinite(evaluation.expected_terminal_loss)
        or not 1.0 - 1e-9 <= evaluation.expected_terminal_loss <= evaluation.expected_value + 1e-9
    ):
        raise RuntimeError("invalid terminal loss in fixed stop-aware plan")
    expected_queries = 1.0 + math.fsum(
        proof.probability * proof.expected_followup_queries for proof in proofs
    )
    if not math.isfinite(expected_queries) or not 1.0 <= expected_queries <= depth + 1e-12:
        raise RuntimeError("invalid expected question count in fixed stop-aware plan")
    return evaluation, expected_queries


def _compute_fixed_stop_aware(
    *,
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    remaining_depth: int,
    config: OptimizedMultistepConfig,
    executor: Executor | None,
    precomputed_state_analysis: StateAnalysis | None,
) -> FixedStopAwareResult:
    """Evaluate one state; recursion never creates a process pool."""

    increment_profile_counter("state_calls")
    state = precomputed_state_analysis
    if state is None:
        space = build_weight_space(alternatives.get_anzahl_spalten(), answered_queries)
        state = StateAnalysis(space, space.is_feasible())
    if not state.is_feasible:
        return FixedStopAwareResult(remaining_depth, 0.0, None, 0, (), False)
    analysis = state.candidate_analysis
    need_volumes = config.volume_confidence_threshold is not None
    if analysis is None or (need_volumes and analysis.candidate_volumes is None):
        analysis = compute_candidate_analysis_for_mode(
            alternatives, state.weight_space,
            analysis.candidates if analysis is not None and config.pass_candidate_subset else None,
            config, include_candidate_volumes=need_volumes,
        )
    candidates = analysis.candidates
    count = len(candidates)
    stop_reason, _ = _stop_aware_state_status(analysis, config)
    if remaining_depth == 0 or count <= 1 or stop_reason is not None:
        loss = 1.0 if stop_reason is not None else float(count)
        return FixedStopAwareResult(
            remaining_depth, float(count), None, count, (), True,
            expected_terminal_loss=loss,
        )

    data = compute_query_candidates_for_depth_optimized(
        alternatives, state.weight_space, candidates, remaining_depth, config,
        precomputed_ratio_intervals=analysis.ratio_intervals,
    )
    queries = data.query_candidates
    increment_profile_counter("query_candidates", len(queries))
    if config.filter_answered_query_candidates:
        queries = filter_already_answered_queries(
            queries, answered_queries,
            abs_tol=config.answered_query_abs_tolerance,
            rel_tol=config.answered_query_rel_tolerance,
        )
    if not queries:
        return FixedStopAwareResult(
            remaining_depth, float(count), None, count, (), True,
            expected_terminal_loss=float(count),
        )
    increment_profile_counter("evaluated_query_candidates", len(queries))
    payloads: list[RootPayload] = [
        (alternatives, answered_queries, state.weight_space, query, None,
         remaining_depth, config,
         candidates if config.pass_candidate_subset else None,
         data.ratio_intervals_by_goal_pair,
         data.query_sources.get(canonical_query_key(query), "unknown"))
        for query in queries
    ]
    raw = list(
        executor.map(_evaluate_fixed_stop_aware_worker, payloads)
        if executor is not None and config.parallelize_root and config.max_workers > 1 and len(payloads) > 1
        else map(_evaluate_fixed_stop_aware_worker, payloads)
    )
    chosen, expected_queries = min(
        raw,
        key=lambda pair: (
            pair[0].expected_terminal_loss,
            *query_identity_sort_key(pair[0].query),
        ),
    )
    return FixedStopAwareResult(
        remaining_depth, chosen.expected_value, chosen.query, count,
        tuple(evaluation for evaluation, _ in raw), True,
        expected_terminal_loss=chosen.expected_terminal_loss,
        expected_queries=expected_queries,
    )


def compute_fixed_stop_aware(
    *,
    alternatives: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    remaining_depth: int,
    config: OptimizedMultistepConfig,
    executor: Executor | None = None,
    precomputed_state_analysis: StateAnalysis | None = None,
) -> ValueFunctionResult:
    """Minimize fixed-horizon E[loss] using exact volumes and shared query sets.

    Only the root is parallelized.  An existing session executor is reused;
    callers outside a session receive at most one temporary root executor.
    """

    if type(remaining_depth) is not int or remaining_depth not in (0, 1, 2, 3):
        raise ValueError("fixed stop-aware planning supports depths 0, 1, 2 and 3")
    if config.answer_probability_mode != "exact_volume":
        raise ValueError("fixed stop-aware planning requires exact_volume probabilities")
    if config.depth_one_query_source_mode != "ratio":
        raise ValueError("fixed stop-aware planning requires ratio queries at depth one")
    if config.adaptive_depth_candidate_threshold is not None:
        raise ValueError("fixed stop-aware planning cannot reduce hypothetical depths adaptively")
    if config.volume_confidence_threshold is None:
        # The ordinary fixed planner already absorbs singleton states.  Keep
        # its lexicographic E[C] refinement, query ties and fast terminal counts.
        return compute_value_function_optimized(
            alternatives=alternatives, answered_queries=answered_queries,
            remaining_depth=remaining_depth, config=config, executor=executor,
            precomputed_state_analysis=precomputed_state_analysis,
        )
    arguments = dict(
        alternatives=alternatives, answered_queries=answered_queries,
        remaining_depth=remaining_depth, config=config,
        precomputed_state_analysis=precomputed_state_analysis,
    )
    if executor is None and config.parallelize_root and config.max_workers > 1 and remaining_depth > 0:
        with ProcessPoolExecutor(max_workers=config.max_workers) as local_executor:
            return _compute_fixed_stop_aware(executor=local_executor, **arguments)
    return _compute_fixed_stop_aware(executor=executor, **arguments)
