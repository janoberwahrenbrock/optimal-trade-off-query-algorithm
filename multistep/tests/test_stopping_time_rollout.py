from __future__ import annotations

import unittest
from unittest.mock import patch

from multistep.optimized.stopping_time_rollout import (
    StoppingTimeRolloutSession,
    _state_key,
)
from multistep.optimized.value_function import (
    CandidateAnalysis,
    OptimizedMultistepConfig,
    QueryCandidateData,
    StateAnalysis,
    canonical_query_key,
)
from multistep.src.models import AlternativenMatrix, Query
from multistep.src.weight_space import build_weight_space


class StoppingTimeRolloutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.config = OptimizedMultistepConfig(
            answer_probability_mode="exact_volume",
            parallelize_root=False,
            grid_depth_query_source_mode="central",
            depth_one_query_source_mode="central",
        )

    def test_rejects_non_exact_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact_volume"):
            StoppingTimeRolloutSession(
                alternatives=self.alternatives,
                config=OptimizedMultistepConfig(
                    answer_probability_mode="sampling",
                    parallelize_root=False,
                ),
            )

    def test_state_key_normalizes_mirrored_queries(self) -> None:
        direct = Query(ziel_index_a=0, ziel_index_b=1, value=2.0).answer("<")
        mirrored = Query(ziel_index_a=1, ziel_index_b=0, value=0.5).answer(">")

        self.assertEqual(_state_key([direct]), _state_key([mirrored]))

    def test_rollout_uses_expected_questions_instead_of_baseline_query(self) -> None:
        baseline_query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
        better_query = Query(ziel_index_a=0, ziel_index_b=2, value=1.0)
        state = StateAnalysis(
            weight_space=build_weight_space(goal_count=3, answered_queries=[]),
            is_feasible=True,
            candidate_analysis=CandidateAnalysis(candidates=[0, 1]),
        )

        def baseline_cost(answered_queries, path_length):
            del path_length
            if not answered_queries:
                return 6.0
            answered = answered_queries[-1]
            if int(answered.ziel_index_b) == 1:
                return 5.0
            return 1.0 if answered.operator == "<" else 3.0

        query_data = QueryCandidateData(
            query_candidates=[baseline_query, better_query],
            query_sources={
                canonical_query_key(baseline_query): "central",
                canonical_query_key(better_query): "central",
            },
        )
        with StoppingTimeRolloutSession(
            alternatives=self.alternatives,
            config=self.config,
            rollout_depth=1,
        ) as session:
            with (
                patch.object(session, "analyze_state", return_value=state),
                patch.object(
                    session,
                    "_compute_baseline_cost",
                    side_effect=baseline_cost,
                ),
                patch.object(
                    session,
                    "_baseline_query",
                    return_value=baseline_query,
                ),
                patch(
                    "multistep.optimized.stopping_time_rollout."
                    "compute_query_candidates_for_depth_optimized",
                    return_value=query_data,
                ),
                patch(
                    "multistep.optimized.stopping_time_rollout."
                    "compute_exact_query_answer_probabilities",
                    return_value={"<": 0.5, "=": 0.0, ">": 0.5},
                ),
            ):
                result = session.compute([])

        self.assertEqual(result.best_query, better_query)
        self.assertAlmostEqual(result.expected_questions, 3.0)
        self.assertEqual(result.baseline_query, baseline_query)

    def test_rollout_prefers_baseline_query_on_numerical_tie(self) -> None:
        baseline_query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
        other_query = Query(ziel_index_a=0, ziel_index_b=2, value=1.0)
        state = StateAnalysis(
            weight_space=build_weight_space(goal_count=3, answered_queries=[]),
            is_feasible=True,
            candidate_analysis=CandidateAnalysis(candidates=[0, 1]),
        )
        query_data = QueryCandidateData(
            query_candidates=[other_query, baseline_query],
            query_sources={},
        )

        with StoppingTimeRolloutSession(
            alternatives=self.alternatives,
            config=self.config,
            rollout_depth=1,
        ) as session:
            with (
                patch.object(session, "analyze_state", return_value=state),
                patch.object(session, "_compute_baseline_cost", return_value=2.0),
                patch.object(
                    session,
                    "_baseline_query",
                    return_value=baseline_query,
                ),
                patch(
                    "multistep.optimized.stopping_time_rollout."
                    "compute_query_candidates_for_depth_optimized",
                    return_value=query_data,
                ),
                patch(
                    "multistep.optimized.stopping_time_rollout."
                    "compute_exact_query_answer_probabilities",
                    return_value={"<": 0.5, "=": 0.0, ">": 0.5},
                ),
            ):
                result = session.compute([])

        self.assertEqual(result.best_query, baseline_query)


if __name__ == "__main__":
    unittest.main()
