from __future__ import annotations

"""Stopping-time rollout planning with the depth-two policy as baseline.

The ordinary finite-horizon value function minimizes the expected candidate
count at the horizon.  This experimental planner instead attaches an estimate
of the expected remaining question count of the unmodified depth-two policy to
every rollout leaf.  It follows exact volume branches, but prunes individually
tiny paths because the complete baseline tree is not finite in practice.
"""

from dataclasses import dataclass
import math

from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query, QueryOperator
from multistep.src.polytope_volume import compute_exact_query_answer_probabilities
from multistep.src.query_probability import ANSWER_OPTIONS

from .value_function import (
    OptimizedMultistepConfig,
    OptimizedValueFunctionSession,
    StateAnalysis,
    canonical_query_key,
    compute_query_candidates_for_depth_optimized,
    filter_already_answered_queries,
)


@dataclass(frozen=True)
class StoppingTimeBranchResult:
    answer: QueryOperator
    probability: float
    expected_remaining_questions: float
    child_candidate_count: int | None
    is_child_feasible: bool


@dataclass(frozen=True)
class StoppingTimeQueryEvaluation:
    query: Query
    expected_questions: float
    branches: tuple[StoppingTimeBranchResult, ...]
    query_source: str = "unknown"


@dataclass(frozen=True)
class StoppingTimeRolloutResult:
    rollout_depth: int
    expected_questions: float
    baseline_expected_questions: float
    best_query: Query | None
    baseline_query: Query | None
    candidate_count: int
    query_evaluations: tuple[StoppingTimeQueryEvaluation, ...]
    is_feasible: bool


@dataclass(frozen=True)
class StoppingTimeRolloutStatistics:
    baseline_states_evaluated: int
    baseline_cache_hits: int
    rollout_states_evaluated: int
    rollout_cache_hits: int
    maximum_pruned_baseline_probability: float


StateKey = tuple[tuple[int, int, float, QueryOperator], ...]


class StoppingTimeRolloutSession:
    """Evaluate a receding rollout policy against a depth-two baseline.

    For rollout depth ``h`` the recursion is

    ``W_0(T) = J_pi2(T)`` and
    ``W_h(T) = min_q (1 + sum_a p(a | T, q) W_{h-1}(T_q,a))``.

    ``J_pi2`` estimates the number of questions until termination when the
    existing depth-two policy is followed without modification.  Its policy
    tree uses exact branch volumes and prunes paths below
    ``baseline_path_probability_cutoff``.  The cumulative pruned mass is
    exposed in ``statistics`` because it can be much larger than the cutoff.
    The baseline query is explicitly included at every rollout state, so the
    baseline path remains an available option under this approximation.
    """

    def __init__(
        self,
        alternatives: AlternativenMatrix,
        config: OptimizedMultistepConfig,
        rollout_depth: int = 3,
        baseline_depth: int = 2,
        max_baseline_questions: int = 100,
        baseline_path_probability_cutoff: float = 1e-8,
        numerical_tolerance: float = 1e-10,
    ) -> None:
        if config.answer_probability_mode != "exact_volume":
            raise ValueError(
                "stopping-time rollout currently requires exact_volume probabilities"
            )
        if config.parallelize_root:
            raise ValueError(
                "stopping-time rollout currently requires parallelize_root=False"
            )
        if rollout_depth <= 0:
            raise ValueError("rollout_depth must be positive")
        if baseline_depth <= 0:
            raise ValueError("baseline_depth must be positive")
        if max_baseline_questions <= 0:
            raise ValueError("max_baseline_questions must be positive")
        if not 0.0 < baseline_path_probability_cutoff < 1.0:
            raise ValueError(
                "baseline_path_probability_cutoff must be between zero and one"
            )
        if numerical_tolerance < 0.0:
            raise ValueError("numerical_tolerance must not be negative")

        self.alternatives = alternatives
        self.config = config
        self.rollout_depth = rollout_depth
        self.baseline_depth = baseline_depth
        self.max_baseline_questions = max_baseline_questions
        self.baseline_path_probability_cutoff = baseline_path_probability_cutoff
        self.numerical_tolerance = numerical_tolerance
        self._base_session = OptimizedValueFunctionSession(
            alternatives=alternatives,
            config=config,
            max_cached_results=8192,
        )
        self._baseline_cost_cache: dict[StateKey, float] = {}
        self._baseline_query_cache: dict[StateKey, Query | None] = {}
        self._baseline_pruned_probability_cache: dict[StateKey, float] = {}
        self._rollout_cache: dict[
            tuple[int, StateKey], StoppingTimeRolloutResult
        ] = {}
        self._baseline_states_evaluated = 0
        self._baseline_cache_hits = 0
        self._rollout_states_evaluated = 0
        self._rollout_cache_hits = 0
        self._closed = False

    def compute(
        self,
        answered_queries: list[AnsweredQuery],
    ) -> StoppingTimeRolloutResult:
        if self._closed:
            raise RuntimeError("stopping-time rollout session is closed")
        return self._compute_rollout(answered_queries, self.rollout_depth)

    def baseline_expected_questions(
        self,
        answered_queries: list[AnsweredQuery],
    ) -> float:
        if self._closed:
            raise RuntimeError("stopping-time rollout session is closed")
        return self._compute_baseline_cost(answered_queries, path_length=0)

    def analyze_state(
        self,
        answered_queries: list[AnsweredQuery],
    ) -> StateAnalysis:
        return self._base_session.analyze_state(answered_queries)

    @property
    def statistics(self) -> StoppingTimeRolloutStatistics:
        return StoppingTimeRolloutStatistics(
            baseline_states_evaluated=self._baseline_states_evaluated,
            baseline_cache_hits=self._baseline_cache_hits,
            rollout_states_evaluated=self._rollout_states_evaluated,
            rollout_cache_hits=self._rollout_cache_hits,
            maximum_pruned_baseline_probability=max(
                self._baseline_pruned_probability_cache.values(),
                default=0.0,
            ),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._base_session.close()
        self._closed = True

    def __enter__(self) -> StoppingTimeRolloutSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _compute_baseline_cost(
        self,
        answered_queries: list[AnsweredQuery],
        path_length: int,
    ) -> float:
        del path_length
        state_key = _state_key(answered_queries)
        cached = self._baseline_cost_cache.get(state_key)
        if cached is not None:
            self._baseline_cache_hits += 1
            return cached
        state = self.analyze_state(answered_queries)
        if not state.is_feasible or state.candidate_analysis is None:
            raise RuntimeError("rollout reached an infeasible positive-volume state")
        if len(state.candidate_analysis.candidates) <= 1:
            self._baseline_cost_cache[state_key] = 0.0
            self._baseline_query_cache[state_key] = None
            self._baseline_pruned_probability_cache[state_key] = 0.0
            return 0.0

        self._baseline_states_evaluated += 1
        expected_questions = 0.0
        pruned_probability = 0.0
        frontier: list[tuple[list[AnsweredQuery], float, int]] = [
            (answered_queries, 1.0, 0)
        ]
        while frontier:
            current_queries, path_probability, current_depth = frontier.pop()
            current_state = self.analyze_state(current_queries)
            if (
                not current_state.is_feasible
                or current_state.candidate_analysis is None
            ):
                raise RuntimeError(
                    "baseline tree reached an infeasible positive-volume state"
                )
            if len(current_state.candidate_analysis.candidates) <= 1:
                continue
            if current_depth >= self.max_baseline_questions:
                raise RuntimeError(
                    "depth-two baseline retained relevant probability mass after "
                    f"{self.max_baseline_questions} questions"
                )

            expected_questions += path_probability
            baseline_query = self._compute_baseline_query(current_queries)
            if baseline_query is None:
                raise RuntimeError(
                    "depth-two baseline returned no query before termination"
                )
            probabilities = compute_exact_query_answer_probabilities(
                weight_space=current_state.weight_space,
                query=baseline_query,
                tolerance=self.config.geometry_tolerance,
            )
            for answer in ANSWER_OPTIONS:
                probability = float(probabilities[answer])
                if probability == 0.0:
                    continue
                child_path_probability = path_probability * probability
                if (
                    child_path_probability
                    < self.baseline_path_probability_cutoff
                ):
                    pruned_probability += child_path_probability
                    continue
                frontier.append(
                    (
                        current_queries + [baseline_query.answer(answer)],
                        child_path_probability,
                        current_depth + 1,
                    )
                )

        self._baseline_cost_cache[state_key] = expected_questions
        self._baseline_pruned_probability_cache[state_key] = pruned_probability
        return expected_questions

    def _compute_baseline_query(
        self,
        answered_queries: list[AnsweredQuery],
    ) -> Query | None:
        state_key = _state_key(answered_queries)
        if state_key in self._baseline_query_cache:
            return self._baseline_query_cache[state_key]

        state = self.analyze_state(answered_queries)
        if not state.is_feasible or state.candidate_analysis is None:
            raise RuntimeError("cannot plan a baseline query for an infeasible state")
        if len(state.candidate_analysis.candidates) <= 1:
            self._baseline_query_cache[state_key] = None
            return None
        baseline_result = self._base_session.compute(
            answered_queries=answered_queries,
            remaining_depth=self.baseline_depth,
        )
        baseline_query = baseline_result.best_query
        if baseline_query is None:
            raise RuntimeError("depth-two baseline returned no query before termination")
        self._baseline_query_cache[state_key] = baseline_query
        return baseline_query

    def _baseline_query(
        self,
        answered_queries: list[AnsweredQuery],
    ) -> Query | None:
        state_key = _state_key(answered_queries)
        if state_key in self._baseline_query_cache:
            return self._baseline_query_cache[state_key]
        return self._compute_baseline_query(answered_queries)

    def _compute_rollout(
        self,
        answered_queries: list[AnsweredQuery],
        remaining_depth: int,
    ) -> StoppingTimeRolloutResult:
        state_key = _state_key(answered_queries)
        cache_key = (remaining_depth, state_key)
        cached = self._rollout_cache.get(cache_key)
        if cached is not None:
            self._rollout_cache_hits += 1
            return cached

        state = self.analyze_state(answered_queries)
        if not state.is_feasible or state.candidate_analysis is None:
            return StoppingTimeRolloutResult(
                rollout_depth=remaining_depth,
                expected_questions=0.0,
                baseline_expected_questions=0.0,
                best_query=None,
                baseline_query=None,
                candidate_count=0,
                query_evaluations=(),
                is_feasible=False,
            )

        candidates = state.candidate_analysis.candidates
        candidate_count = len(candidates)
        baseline_value = self._compute_baseline_cost(answered_queries, path_length=0)
        baseline_query = self._baseline_query(answered_queries)
        if remaining_depth == 0 or candidate_count <= 1:
            result = StoppingTimeRolloutResult(
                rollout_depth=remaining_depth,
                expected_questions=baseline_value,
                baseline_expected_questions=baseline_value,
                best_query=None,
                baseline_query=baseline_query,
                candidate_count=candidate_count,
                query_evaluations=(),
                is_feasible=True,
            )
            self._rollout_cache[cache_key] = result
            return result

        self._rollout_states_evaluated += 1
        query_data = compute_query_candidates_for_depth_optimized(
            alternatives=self.alternatives,
            weight_space=state.weight_space,
            candidates=candidates,
            remaining_depth=remaining_depth,
            config=self.config,
            precomputed_ratio_intervals=state.candidate_analysis.ratio_intervals,
        )
        query_candidates = list(query_data.query_candidates)
        if baseline_query is not None and all(
            canonical_query_key(query) != canonical_query_key(baseline_query)
            for query in query_candidates
        ):
            query_candidates.append(baseline_query)
        query_candidates = filter_already_answered_queries(
            queries=query_candidates,
            answered_queries=answered_queries,
            abs_tol=self.config.answered_query_abs_tolerance,
            rel_tol=self.config.answered_query_rel_tolerance,
        )
        if not query_candidates:
            raise RuntimeError("rollout returned no query before termination")

        evaluations = tuple(
            self._evaluate_rollout_query(
                answered_queries=answered_queries,
                state=state,
                query=query,
                remaining_depth=remaining_depth,
                query_source=query_data.query_sources.get(
                    canonical_query_key(query),
                    "baseline" if query == baseline_query else "unknown",
                ),
            )
            for query in query_candidates
        )
        minimum_value = min(
            evaluation.expected_questions for evaluation in evaluations
        )
        tied = [
            evaluation
            for evaluation in evaluations
            if math.isclose(
                evaluation.expected_questions,
                minimum_value,
                rel_tol=0.0,
                abs_tol=self.numerical_tolerance,
            )
        ]
        best_evaluation = next(
            (
                evaluation
                for evaluation in tied
                if baseline_query is not None
                and canonical_query_key(evaluation.query)
                == canonical_query_key(baseline_query)
            ),
            min(tied, key=lambda evaluation: canonical_query_key(evaluation.query)),
        )
        result = StoppingTimeRolloutResult(
            rollout_depth=remaining_depth,
            expected_questions=best_evaluation.expected_questions,
            baseline_expected_questions=baseline_value,
            best_query=best_evaluation.query,
            baseline_query=baseline_query,
            candidate_count=candidate_count,
            query_evaluations=evaluations,
            is_feasible=True,
        )
        self._rollout_cache[cache_key] = result
        return result

    def _evaluate_rollout_query(
        self,
        answered_queries: list[AnsweredQuery],
        state: StateAnalysis,
        query: Query,
        remaining_depth: int,
        query_source: str,
    ) -> StoppingTimeQueryEvaluation:
        probabilities = compute_exact_query_answer_probabilities(
            weight_space=state.weight_space,
            query=query,
            tolerance=self.config.geometry_tolerance,
        )
        expected_child_cost = 0.0
        branches: list[StoppingTimeBranchResult] = []
        for answer in ANSWER_OPTIONS:
            probability = float(probabilities[answer])
            if probability == 0.0:
                branches.append(
                    StoppingTimeBranchResult(
                        answer=answer,
                        probability=0.0,
                        expected_remaining_questions=0.0,
                        child_candidate_count=None,
                        is_child_feasible=False,
                    )
                )
                continue

            child_queries = answered_queries + [query.answer(answer)]
            child_result = self._compute_rollout(
                child_queries,
                remaining_depth=remaining_depth - 1,
            )
            if not child_result.is_feasible:
                raise RuntimeError("positive-volume rollout branch became infeasible")
            expected_child_cost += probability * child_result.expected_questions
            branches.append(
                StoppingTimeBranchResult(
                    answer=answer,
                    probability=probability,
                    expected_remaining_questions=child_result.expected_questions,
                    child_candidate_count=child_result.candidate_count,
                    is_child_feasible=True,
                )
            )

        return StoppingTimeQueryEvaluation(
            query=query,
            expected_questions=1.0 + expected_child_cost,
            branches=tuple(branches),
            query_source=query_source,
        )


def _state_key(answered_queries: list[AnsweredQuery]) -> StateKey:
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
