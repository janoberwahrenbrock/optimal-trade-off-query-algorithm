from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import math
import pickle
import unittest
from unittest.mock import Mock, patch

from multistep.optimized import stopping_value_function as fixed
from multistep.optimized.query_efficiency import (
    StopAwareQueryEvaluation, _evaluate_stop_aware_query,
)
from multistep.optimized.value_function import (
    CandidateAnalysis, OptimizedMultistepConfig, QueryCandidateData, StateAnalysis,
    compute_value_function_optimized,
)
from multistep.src.models import AlternativenMatrix, Query
from multistep.src.value_function import QueryBranchResult
from multistep.src.weight_space import build_weight_space


class FixedStopAwareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.alternatives = AlternativenMatrix(entries=[[1., 0.], [48. / 49., 1.]])
        self.space = build_weight_space(2, [])
        self.config = OptimizedMultistepConfig(
            answer_probability_mode="exact_volume", use_ratio_terminal_counts=True,
            grid_size=2, query_efficiency_stop_aware=True,
            volume_confidence_threshold=.99,
        )
        self.query = Query(ziel_index_a=0, ziel_index_b=1, value=197. / 3.)

    def payload(self, depth=1, config=None):
        return (self.alternatives, [], self.space, self.query, None, depth,
                config or self.config, [0, 1], None, "ratio")

    def state(self, share=.98):
        analysis = CandidateAnalysis(
            [0, 1], candidate_volumes={0: 1. - share, 1: share},
            candidate_volume_shares={0: 1. - share, 1: share},
        )
        return StateAnalysis(self.space, True, analysis)

    def test_horizons_one_two_reuse_existing_exact_stop_aware_semantics(self):
        # A cut at .985 leaves two possible candidates in the '<' branch,
        # but its dominant candidate already owns 196/197 > .99 volume.
        for depth in (1, 2):
            with self.subTest(depth=depth):
                actual, cost = fixed._evaluate_fixed_stop_aware_worker(self.payload(depth))
                expected, old_cost, _ = _evaluate_stop_aware_query(
                    alternatives=self.alternatives, answered_queries=[],
                    weight_space=self.space, query=self.query, horizon=depth,
                    config=self.config, candidate_subset=[0, 1],
                    ratio_intervals_by_goal_pair=None, query_source="ratio",
                )
                self.assertEqual(actual, expected)
                self.assertEqual(cost, old_cost)
                self.assertAlmostEqual(actual.expected_value, 1.985, places=10)
                self.assertEqual(actual.expected_terminal_loss, 1.)
                self.assertEqual(cost, 1.)

    def test_horizon_three_absorbs_confidence_before_continuation(self):
        with patch.object(fixed, "_compute_fixed_stop_aware") as recurse:
            evaluation, cost = fixed._evaluate_fixed_stop_aware_worker(self.payload(3))
        recurse.assert_not_called()
        self.assertAlmostEqual(evaluation.expected_value, 1.985, places=10)
        self.assertEqual(evaluation.expected_terminal_loss, 1.)
        self.assertEqual(cost, 1.)
        self.assertEqual({p.stop_reason for p in evaluation.stop_aware_branches
                          if p.probability > 0.}, {"single_candidate", "volume_confidence"})

    def test_precomputed_confidence_rule_is_inclusive_and_threshold_specific(self):
        for threshold in (.95, .99):
            for share, stopped in ((threshold, True),
                                   (math.nextafter(threshold, 0.), False),
                                   (math.nextafter(threshold, 1.), True)):
                with self.subTest(threshold=threshold, share=share):
                    with patch.object(fixed, "compute_query_candidates_for_depth_optimized",
                                      return_value=QueryCandidateData(query_candidates=[])) as queries:
                        result = fixed.compute_fixed_stop_aware(
                            alternatives=self.alternatives, answered_queries=[], remaining_depth=3,
                            config=replace(self.config, volume_confidence_threshold=threshold),
                            precomputed_state_analysis=self.state(share),
                        )
                    self.assertEqual(result.value, 2.)
                    self.assertEqual(result.expected_terminal_loss, 1. if stopped else 2.)
                    self.assertIsNone(result.best_query)
                    self.assertEqual(queries.call_count, int(not stopped))

    def test_singleton_threshold_uses_unchanged_legacy_fixed_planner(self):
        config = replace(self.config, volume_confidence_threshold=None)
        for depth in (1, 2, 3):
            with self.subTest(depth=depth):
                old = compute_value_function_optimized(self.alternatives, [], depth, config)
                new = fixed.compute_fixed_stop_aware(
                    alternatives=self.alternatives, answered_queries=[], remaining_depth=depth,
                    config=config,
                )
                self.assertEqual(old, new)

    def test_no_query_keeps_nonterminal_loss_and_executes_no_questions(self):
        with patch.object(fixed, "compute_query_candidates_for_depth_optimized",
                          return_value=QueryCandidateData(query_candidates=[])):
            result = fixed.compute_fixed_stop_aware(
                alternatives=self.alternatives, answered_queries=[], remaining_depth=2,
                config=self.config, precomputed_state_analysis=self.state(),
            )
        self.assertTrue(result.is_feasible)
        self.assertIsNone(result.best_query)
        self.assertEqual(result.value, 2.)
        self.assertEqual(result.expected_terminal_loss, 2.)
        self.assertEqual(result.expected_queries, 0.)

    def test_fixed_selection_minimizes_loss_without_replacing_actual_candidate_value(self):
        other = Query(ziel_index_a=0, ziel_index_b=1, value=2.)
        low_loss = StopAwareQueryEvaluation(
            self.query, 2., (), expected_terminal_loss=1.,
        )
        low_actual_count = StopAwareQueryEvaluation(
            other, 1.5, (), expected_terminal_loss=1.5,
        )
        with patch.object(fixed, "compute_query_candidates_for_depth_optimized",
                          return_value=QueryCandidateData(query_candidates=[other, self.query])), \
             patch.object(fixed, "_evaluate_fixed_stop_aware_worker",
                          side_effect=lambda payload: (low_loss if payload[3] == self.query
                                                       else low_actual_count, 1.)):
            result = fixed.compute_fixed_stop_aware(
                alternatives=self.alternatives, answered_queries=[], remaining_depth=2,
                config=self.config, precomputed_state_analysis=self.state(),
            )
        self.assertEqual(result.best_query, self.query)
        self.assertEqual(result.value, 2.)
        self.assertEqual(result.expected_terminal_loss, 1.)

    def test_equal_terminal_losses_use_stable_query_identity(self):
        other = Query(ziel_index_a=0, ziel_index_b=1, value=1.)
        plans = {
            query: StopAwareQueryEvaluation(query, 2., (), expected_terminal_loss=1.)
            for query in (self.query, other)
        }
        for queries in ([self.query, other], [other, self.query]):
            with self.subTest(order=queries), \
                 patch.object(fixed, "compute_query_candidates_for_depth_optimized",
                              return_value=QueryCandidateData(query_candidates=queries)), \
                 patch.object(fixed, "_evaluate_fixed_stop_aware_worker",
                              side_effect=lambda payload: (plans[payload[3]], 1.)):
                result = fixed.compute_fixed_stop_aware(
                    alternatives=self.alternatives, answered_queries=[], remaining_depth=2,
                    config=self.config, precomputed_state_analysis=self.state(),
                )
            self.assertEqual(result.best_query, other)

    def test_three_step_weighted_cost_excludes_absorbed_and_zero_volume_paths(self):
        alternatives = AlternativenMatrix(entries=[[1., 0.], [0., 1.], [.5, .5]])
        stopped = CandidateAnalysis([0, 1], candidate_volumes={0: .96, 1: .04},
                                    candidate_volume_shares={0: .96, 1: .04})
        unfinished = CandidateAnalysis([0, 1, 2], candidate_volumes={0: .4, 1: .3, 2: .3},
                                       candidate_volume_shares={0: .4, 1: .3, 2: .3})
        child = fixed.FixedStopAwareResult(
            2, 2., self.query, 3, (), True,
            expected_terminal_loss=1.5, expected_queries=2.,
        )
        space = Mock()
        space.is_feasible.return_value = True
        payload = (alternatives, [], space, self.query, None, 3,
                   replace(self.config, volume_confidence_threshold=.95), [0, 1, 2], None, "grid")
        with patch.object(fixed, "compute_exact_query_answer_probabilities",
                          return_value={"<": .05, "=": 0., ">": .95}), \
             patch.object(fixed, "build_weight_space", return_value=space), \
             patch.object(fixed, "compute_candidate_analysis_for_mode",
                          side_effect=[stopped, unfinished]) as analyze, \
             patch.object(fixed, "_compute_fixed_stop_aware", return_value=child) as recurse:
            evaluation, cost = fixed._evaluate_fixed_stop_aware_worker(payload)
        self.assertEqual(analyze.call_count, 2)  # '=' has zero volume.
        recurse.assert_called_once()
        self.assertEqual(recurse.call_args.kwargs["remaining_depth"], 2)
        self.assertIsNone(recurse.call_args.kwargs["executor"])
        self.assertIs(recurse.call_args.kwargs["precomputed_state_analysis"].candidate_analysis,
                      unfinished)
        self.assertAlmostEqual(cost, .05 * 1. + .95 * 3., places=12)
        self.assertAlmostEqual(evaluation.expected_value, 2., places=12)
        self.assertAlmostEqual(evaluation.expected_terminal_loss, .05 + .95 * 1.5, places=12)

    def test_serial_and_reused_root_executor_results_match(self):
        serial_config = replace(self.config, parallelize_root=True, max_workers=2)
        serial = fixed.compute_fixed_stop_aware(
            alternatives=self.alternatives, answered_queries=[], remaining_depth=2,
            config=replace(serial_config, parallelize_root=False),
        )
        with ProcessPoolExecutor(max_workers=2) as executor:
            parallel = fixed.compute_fixed_stop_aware(
                alternatives=self.alternatives, answered_queries=[], remaining_depth=2,
                config=serial_config, executor=executor,
            )
        self.assertEqual(asdict(serial), asdict(parallel))
        self.assertEqual(parallel, pickle.loads(pickle.dumps(parallel)))

    def test_rejects_unsupported_depth_and_nonexact_probability_method(self):
        for depth in (-1, 4, True, 1.):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                fixed.compute_fixed_stop_aware(
                    alternatives=self.alternatives, answered_queries=[], remaining_depth=depth,
                    config=self.config,
                )
        with self.assertRaisesRegex(ValueError, "exact_volume"):
            fixed.compute_fixed_stop_aware(
                alternatives=self.alternatives, answered_queries=[], remaining_depth=2,
                config=replace(self.config, answer_probability_mode="sampling",
                               query_efficiency_stop_aware=False),
            )


if __name__ == "__main__":
    unittest.main()
