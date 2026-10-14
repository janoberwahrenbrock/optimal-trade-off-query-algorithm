from __future__ import annotations

"""One exact-volume solver for fixed and adaptive query strategies.

The solver sees alternatives and answers, never the answerer's target weights.
The lower-level optimized planners retain their geometric calculations, query
ordering, numerical tolerances, and persistent worker pool.
"""

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable, Literal, Sequence

from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query, QueryOperator
from multistep.src.polytope_volume import clear_polytope_volume_cache, polytope_volume_cache_info

from .value_function import (
    OptimizedMultistepConfig,
    OptimizedValueFunctionSession,
    StateAnalysis,
    evaluate_volume_confidence_termination,
)


AnswerQuery = Callable[[Query], QueryOperator]
StateObserver = Callable[[StateAnalysis, int], None]
QueryObserver = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class AlgorithmConfig:
    """The nine method and execution settings for the public solver.

    ``depth`` is the fixed lookahead or the largest adaptive horizon. Adaptive
    planning compares all horizons from one through ``depth``. At remaining
    depth one the query source is always ratios; exact volumes are always used.
    The confidence threshold is inactive for ``single_candidate`` stopping.
    ``max_queries`` includes any supplied initial answers; ``None`` is uncapped.
    """

    query_strategy: Literal["fixed", "reduction_per_query"] = "fixed"
    depth: int = 2
    higher_depth_query_source: Literal["grid", "both"] = "both"
    grid_size: int = 5
    stop_rule: Literal["single_candidate", "volume_confidence"] = "volume_confidence"
    confidence_threshold: float | None = 0.99
    consider_stop_in_planning: bool = False
    workers: int = 8
    max_queries: int | None = 100

    def __post_init__(self) -> None:
        if self.query_strategy not in {"fixed", "reduction_per_query"}:
            raise ValueError("query_strategy must be 'fixed' or 'reduction_per_query'")
        if type(self.depth) is not int or self.depth not in (1, 2, 3):
            raise ValueError("depth must be an integer in {1, 2, 3}")
        if self.higher_depth_query_source not in {"grid", "both"}:
            raise ValueError("higher_depth_query_source must be 'grid' or 'both'")
        if type(self.grid_size) is not int or self.grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        if self.stop_rule not in {"single_candidate", "volume_confidence"}:
            raise ValueError("stop_rule must be 'single_candidate' or 'volume_confidence'")
        if self.confidence_threshold is not None and (
            isinstance(self.confidence_threshold, bool)
            or not isinstance(self.confidence_threshold, (int, float))
            or not math.isfinite(self.confidence_threshold)
            or not 0.0 < self.confidence_threshold <= 1.0
        ):
            raise ValueError("confidence_threshold must be in (0, 1] or None")
        if self.stop_rule == "volume_confidence" and self.confidence_threshold is None:
            raise ValueError("volume_confidence stopping requires a confidence_threshold")
        if type(self.consider_stop_in_planning) is not bool:
            raise ValueError("consider_stop_in_planning must be a bool")
        if type(self.workers) is not int or self.workers <= 0:
            raise ValueError("workers must be a positive integer")
        if self.max_queries is not None and (
            type(self.max_queries) is not int or self.max_queries <= 0
        ):
            raise ValueError("max_queries must be a positive integer or None")

    def to_optimized_config(self) -> OptimizedMultistepConfig:
        """Resolve the same efficient exact configuration used by benchmarks."""

        return OptimizedMultistepConfig(
            answer_probability_mode="exact_volume",
            skip_zero_probability_branches=True,
            pass_candidate_subset=True,
            use_ratio_terminal_counts=True,
            validate_ratio_terminal_counts=False,
            candidate_count_mode="ratio_relevant",
            ratio_interval_engine="geometry",
            grid_size=self.grid_size,
            grid_depth_query_source_mode=self.higher_depth_query_source,
            depth_one_query_source_mode="ratio",
            parallelize_root=self.workers > 1,
            max_workers=self.workers,
            volume_confidence_threshold=(
                self.confidence_threshold if self.stop_rule == "volume_confidence" else None
            ),
            query_efficiency_stop_aware=self.consider_stop_in_planning,
        )


def solve(
    alternatives: AlternativenMatrix | Sequence[Sequence[float]],
    config: AlgorithmConfig,
    answer_query: AnswerQuery,
    *,
    initial_answered_queries: Sequence[AnsweredQuery] | None = None,
    on_state: StateObserver | None = None,
    on_query: QueryObserver | None = None,
) -> dict[str, Any]:
    """Ask questions until the configured stop, limit, or unavailable query.

    ``answer_query(query)`` must return ``'<'``, ``'>'``, or ``'='``. It may be
    an interactive interface, simulator, or recorded-answer provider. Exceptions
    and infeasible answer histories are reported to the caller, not hidden.
    Observers are optional synchronous hooks; their time belongs to elapsed time.
    """

    if not isinstance(config, AlgorithmConfig):
        raise TypeError("config must be an AlgorithmConfig")
    if not callable(answer_query):
        raise TypeError("answer_query must be callable")
    matrix = alternatives if isinstance(alternatives, AlternativenMatrix) else AlternativenMatrix(
        entries=[list(row) for row in alternatives]
    )
    history = list(initial_answered_queries or ())
    if any(not isinstance(answer, AnsweredQuery) for answer in history):
        raise TypeError("initial_answered_queries must contain AnsweredQuery values")
    goal_count = matrix.get_anzahl_spalten()
    if any(max(answer.ziel_index_a, answer.ziel_index_b) >= goal_count for answer in history):
        raise ValueError("an initial answer references a goal outside the alternatives matrix")
    strategy = config.query_strategy
    if strategy == "reduction_per_query" and config.consider_stop_in_planning:
        strategy = "reduction_per_query_stop_aware"
    result = _solve_with_optimized_config(
        alternatives=matrix,
        optimized_config=config.to_optimized_config(),
        answer_query=answer_query,
        depth=config.depth,
        query_strategy=strategy,
        max_queries=config.max_queries,
        initial_answered_queries=history,
        on_state=on_state,
        on_query=on_query,
    )
    result["config"] = asdict(config)
    result["resolved_config"] = asdict(config.to_optimized_config())
    result["initial_question_count"] = len(history)
    result["new_question_count"] = len(result["queries"])
    result["answered_queries"] = [answer.model_dump() for answer in history] + [
        {
            "ziel_index_a": record["goal_index_a"],
            "ziel_index_b": record["goal_index_b"],
            "value": record["value"],
            "operator": record["answer"],
        }
        for record in result["queries"]
    ]
    return result


def _format_candidate_volume_shares(
    candidates: list[int], volume_shares: dict[int, float] | None,
) -> str:
    if volume_shares is None:
        return "unavailable"
    return ",".join(f"{candidate}:{100.0 * volume_shares[candidate]:.4f}%" for candidate in candidates)


def _solve_with_optimized_config(
    alternatives: AlternativenMatrix,
    optimized_config: OptimizedMultistepConfig,
    answer_query: AnswerQuery,
    *,
    depth: int,
    query_strategy: str = "fixed",
    max_queries: int | None = 100,
    initial_answered_queries: Sequence[AnsweredQuery] | None = None,
    session_factory: Callable[..., OptimizedValueFunctionSession] = OptimizedValueFunctionSession,
    on_state: StateObserver | None = None,
    on_query: QueryObserver | None = None,
    progress_per_query: bool = False,
    raise_on_no_query: bool = False,
) -> dict[str, Any]:
    """Shared loop; explicit lower-level config also supports legacy scripts.

    Benchmark wrappers add their independent winner/correctness observations.
    This compatibility path deliberately retains the old result shape and timers.
    """

    if query_strategy not in {"fixed", "reduction_per_query", "reduction_per_query_stop_aware"}:
        raise ValueError("unsupported query strategy")
    if type(depth) is not int or depth not in (1, 2, 3):
        raise ValueError("depth must be an integer in {1, 2, 3}")
    if max_queries is not None and (type(max_queries) is not int or max_queries <= 0):
        raise ValueError("max_queries must be a positive integer or None")
    if optimized_config.answer_probability_mode != "exact_volume":
        raise ValueError("the unified algorithm requires exact_volume probabilities")
    clear_polytope_volume_cache()
    config = optimized_config
    answered_queries = list(initial_answered_queries or ())
    query_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    initial_candidate_count: int | None = None
    initial_candidates: list[int] = []
    initial_candidate_volumes: dict[int, float] | None = None
    initial_candidate_volume_shares: dict[int, float] | None = None
    final_candidates: list[int] = []
    final_candidate_volumes: dict[int, float] | None = None
    final_candidate_volume_shares: dict[int, float] | None = None
    termination_reason = "max_questions"
    selected_candidate: int | None = None
    selected_candidate_volume: float | None = None
    selected_candidate_volume_share: float | None = None
    residual_volume: float | None = None
    residual_volume_share: float | None = None
    with session_factory(
        alternatives=alternatives,
        config=config,
        max_cached_results=max_queries + 1 if max_queries is not None else 16,
    ) as session:
        while True:
            question_index = len(answered_queries)
            iteration_started = time.perf_counter()
            state = session.analyze_state(answered_queries, include_candidate_volumes=True)
            if not state.is_feasible or state.candidate_analysis is None:
                raise RuntimeError("query trajectory became infeasible")
            candidates = state.candidate_analysis.candidates
            candidate_volumes = state.candidate_analysis.candidate_volumes
            candidate_volume_shares = state.candidate_analysis.candidate_volume_shares
            if initial_candidate_count is None:
                initial_candidate_count = len(candidates)
                initial_candidates = [int(candidate) for candidate in candidates]
                initial_candidate_volumes = candidate_volumes
                initial_candidate_volume_shares = candidate_volume_shares
                if progress_per_query:
                    print(
                        f"query=00 candidates={initial_candidates} "
                        f"volume_shares={_format_candidate_volume_shares(candidates, candidate_volume_shares)} "
                        "initial=True", flush=True,
                    )
            if query_records and "remaining_candidate_count" not in query_records[-1]:
                query_record = query_records[-1]
                query_record["remaining_candidate_count"] = len(candidates)
                query_record["remaining_candidates"] = [int(candidate) for candidate in candidates]
                query_record["candidate_volumes"] = candidate_volumes
                query_record["candidate_volume_shares"] = candidate_volume_shares
                if progress_per_query:
                    print(
                        f"query={len(query_records):02d} "
                        f"candidates={query_record['remaining_candidates']} "
                        f"volume_shares={_format_candidate_volume_shares(candidates, candidate_volume_shares)} "
                        f"expected_value={query_record['expected_value']:.6f} "
                        f"seconds={query_record['seconds']:.3f}", flush=True,
                    )
            if on_state is not None:
                on_state(state, question_index)
            if not candidates:
                raise RuntimeError("feasible query state has no candidate alternatives")
            final_candidates = candidates
            final_candidate_volumes = candidate_volumes
            final_candidate_volume_shares = candidate_volume_shares
            if len(candidates) <= 1:
                termination_reason = "exact_single_candidate"
                if candidates:
                    selected_candidate = int(candidates[0])
                    if candidate_volumes is not None:
                        selected_candidate_volume = float(candidate_volumes[selected_candidate])
                    if candidate_volume_shares is not None:
                        selected_candidate_volume_share = float(candidate_volume_shares[selected_candidate])
                        residual_volume_share = max(0.0, 1.0 - selected_candidate_volume_share)
                    if candidate_volumes is not None:
                        residual_volume = max(
                            0.0, sum(candidate_volumes.values()) - float(candidate_volumes[selected_candidate]),
                        )
                break
            confidence_decision = evaluate_volume_confidence_termination(
                candidate_analysis=state.candidate_analysis,
                threshold=config.volume_confidence_threshold,
            )
            if confidence_decision is not None:
                termination_reason = "volume_confidence"
                selected_candidate = confidence_decision.selected_candidate
                selected_candidate_volume = confidence_decision.selected_candidate_volume
                selected_candidate_volume_share = confidence_decision.selected_candidate_volume_share
                residual_volume = confidence_decision.residual_volume
                residual_volume_share = confidence_decision.residual_volume_share
                if progress_per_query:
                    print(
                        f"termination={termination_reason} selected_candidate={selected_candidate} "
                        f"confidence={selected_candidate_volume_share:.6f} "
                        f"residual_volume_share={residual_volume_share:.6f}", flush=True,
                    )
                break
            if max_queries is not None and question_index >= max_queries:
                break

            efficiency_result = None
            if query_strategy in {"reduction_per_query", "reduction_per_query_stop_aware"}:
                efficiency_result = session.compute_query_efficiency(
                    answered_queries=answered_queries, max_horizon=depth,
                )
                if efficiency_result.choice is None:
                    if raise_on_no_query:
                        raise RuntimeError("query-efficiency planner returned no query")
                    termination_reason = "no_query"
                    break
                best_evaluation = efficiency_result.choice.evaluation
                best_query = best_evaluation.query
                expected_value = best_evaluation.expected_value
            else:
                if config.query_efficiency_stop_aware:
                    from .stopping_value_function import compute_fixed_stop_aware

                    result = compute_fixed_stop_aware(
                        alternatives=alternatives,
                        answered_queries=answered_queries,
                        remaining_depth=depth,
                        config=config,
                        executor=session._executor,
                        precomputed_state_analysis=state,
                    )
                else:
                    result = session.compute(answered_queries=answered_queries, remaining_depth=depth)
                if result.best_query is None:
                    if raise_on_no_query:
                        raise RuntimeError("exact planner returned no query")
                    termination_reason = "no_query"
                    break
                best_query = result.best_query
                best_evaluation = next(
                    evaluation for evaluation in result.query_evaluations if evaluation.query == best_query
                )
                expected_value = (
                    best_evaluation.expected_value if config.query_efficiency_stop_aware else float(result.value)
                )
            expected_values = best_evaluation.lexicographic_expected_values or (float(best_evaluation.expected_value),)
            answer = answer_query(best_query)
            if answer not in ("<", ">", "="):
                raise ValueError("answer_query must return '<', '>', or '='")
            answered_queries.append(best_query.answer(answer))
            query_records.append({
                "goal_index_a": int(best_query.ziel_index_a),
                "goal_index_b": int(best_query.ziel_index_b),
                "value": float(best_query.value),
                "answer": answer,
                "seconds": time.perf_counter() - iteration_started,
                "expected_value": float(expected_value),
                "lexicographic_expected_values": [float(value) for value in expected_values],
            })
            if efficiency_result is not None:
                from .query_efficiency import query_efficiency_evaluation_to_json

                choice = efficiency_result.choice
                assert choice is not None
                query_records[-1].update({
                    "query_strategy": query_strategy,
                    "selected_horizon": choice.horizon,
                    "expected_terminal_candidates": choice.evaluation.expected_value,
                    "expected_queries": choice.expected_queries,
                    "gain": choice.gain,
                    "rate": choice.rate,
                    "horizon_best_plans": [
                        query_efficiency_evaluation_to_json(plan)
                        for _, plan in sorted(efficiency_result.best_by_horizon.items()) if plan is not None
                    ],
                    "horizon_evaluation_seconds": {
                        str(horizon): seconds for horizon, seconds in efficiency_result.evaluation_seconds_by_horizon.items()
                    },
                })
                if config.query_efficiency_stop_aware:
                    if choice.expected_terminal_loss is None:
                        raise RuntimeError("stop-aware decision lacks its terminal loss")
                    _record_planning_loss(query_records[-1], len(candidates), choice.expected_terminal_loss)
            elif config.query_efficiency_stop_aware:
                loss = getattr(best_evaluation, "expected_terminal_loss", None)
                if loss is None and config.volume_confidence_threshold is None:
                    loss = best_evaluation.expected_value
                if loss is None:
                    raise RuntimeError("stop-aware decision lacks its terminal loss")
                _record_planning_loss(query_records[-1], len(candidates), loss)
            if on_query is not None:
                on_query(query_records[-1])

    seconds = time.perf_counter() - started
    cache = polytope_volume_cache_info()
    return {
        "solved": selected_candidate is not None,
        "exactly_solved": len(final_candidates) == 1,
        "termination_reason": termination_reason,
        "question_count": len(answered_queries),
        "seconds": seconds,
        "initial_candidate_count": initial_candidate_count,
        "initial_candidates": initial_candidates,
        "initial_candidate_volumes": initial_candidate_volumes,
        "initial_candidate_volume_shares": initial_candidate_volume_shares,
        "final_candidates": final_candidates,
        "final_candidate_volumes": final_candidate_volumes,
        "final_candidate_volume_shares": final_candidate_volume_shares,
        "final_candidate": final_candidates[0] if len(final_candidates) == 1 else None,
        "selected_candidate": selected_candidate,
        "selected_candidate_volume": selected_candidate_volume,
        "selected_candidate_volume_share": selected_candidate_volume_share,
        "residual_volume": residual_volume,
        "residual_volume_share": residual_volume_share,
        "query_seconds": [record["seconds"] for record in query_records],
        "queries": query_records,
        "volume_cache": {"hits": cache.hits, "misses": cache.misses, "size": cache.currsize},
    }


def _record_planning_loss(record: dict[str, Any], candidate_count: int, loss: float) -> None:
    record.update({
        "planning_stop_aware": True,
        "candidate_count_before_planning": candidate_count,
        "expected_terminal_loss": float(loss),
        "expected_value_semantics": "expected_actual_terminal_candidate_count",
        "planning_terminal_loss_semantics": "one_for_single_candidate_or_volume_confidence_else_actual_candidate_count",
    })
