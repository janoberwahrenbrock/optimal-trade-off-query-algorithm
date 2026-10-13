from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import math
import multiprocessing
import pickle
import unittest
from unittest.mock import Mock, patch

from multistep.optimized import query_efficiency as efficiency
from multistep.optimized.value_function import (
    CandidateAnalysis, OptimizedMultistepConfig, QueryCandidateData, StateAnalysis,
    compute_value_function_optimized,
)
from multistep.src.models import AlternativenMatrix, Query
from multistep.src.value_function import QueryBranchResult
from multistep.src.weight_space import build_weight_space


class StopAwareQueryEfficiencyTest(unittest.TestCase):
    def setUp(self) -> None:
        # In the two-objective simplex, w0 is uniform on [0, 1].  These
        # utilities intersect at w0 = 49/50, so both candidates are possible
        # and the larger optimality region owns 98% of the initial interval.
        self.alternatives = AlternativenMatrix(entries=[[1., 0.], [48. / 49., 1.]])
        self.space = build_weight_space(2, [])
        self.config = OptimizedMultistepConfig(
            answer_probability_mode="exact_volume", use_ratio_terminal_counts=True,
            grid_size=2, query_efficiency_stop_aware=True,
            volume_confidence_threshold=.99,
        )
        # w0/w1 = 197/3 cuts at w0 = 197/200 = .985.  The '<' answer has
        # probability .985 and still two candidates; its larger region has
        # share (49/50)/(197/200) = 196/197, above 99%.  The other answer has
        # probability .015 and a single candidate.
        self.query = Query(ziel_index_a=0, ziel_index_b=1, value=197. / 3.)

    def payload(self, horizon: int = 1, config=None):
        return (self.alternatives, [], self.space, self.query, None, horizon,
                config or self.config, [0, 1], None, "ratio")

    def evaluate(self, horizon: int = 1, config=None):
        return efficiency._evaluate_horizon_candidates(
            alternatives=self.alternatives, answered_queries=[], weight_space=self.space,
            query_candidates=[self.query], horizon=horizon,
            config=config or self.config, candidates=[0, 1],
            ratio_intervals_by_goal_pair=None, query_sources={}, executor=None,
        )[0]

    def test_volume_stop_earns_gain_without_removing_the_second_candidate(self):
        plan = self.evaluate()
        self.assertAlmostEqual(plan.evaluation.expected_value, 1.985, places=10)
        self.assertEqual(plan.expected_terminal_loss, 1.)
        self.assertEqual(plan.expected_queries, 1.)
        self.assertEqual(plan.gain, 1.)
        self.assertEqual(plan.rate, 1.)
        less = next(branch for branch in plan.evaluation.branches if branch.answer == "<")
        self.assertEqual(less.child_candidate_count, 2)
        self.assertEqual(less.child_value, 2.)
        self.assertAlmostEqual(less.probability, .985, places=10)

    def test_horizon_two_absorbs_volume_stop_and_needs_no_second_question(self):
        plan = self.evaluate(2)
        self.assertAlmostEqual(plan.evaluation.expected_value, 1.985, places=10)
        self.assertEqual(plan.expected_terminal_loss, 1.)
        self.assertEqual(plan.expected_queries, 1.)
        self.assertEqual(plan.gain, 1.)
        self.assertTrue(plan.continuations)
        self.assertTrue(all(child.evaluation is None for child in plan.continuations))
        less = next(child for child in plan.continuations if child.answer == "<")
        self.assertEqual(less.candidate_count, 2)

    def test_stricter_threshold_requires_followup_on_only_the_unfinished_branch(self):
        plan = self.evaluate(2, replace(self.config, volume_confidence_threshold=.999))
        self.assertAlmostEqual(plan.expected_queries, 1.985, places=10)
        self.assertAlmostEqual(plan.expected_terminal_loss, 1., places=10)
        # Ratio queries retain their existing epsilon offset from the true
        # utility crossover.  The small remaining two-candidate region is
        # still present in E[C], even though its volume is sufficiently
        # concentrated to count as finished in the planning loss.
        self.assertGreater(plan.evaluation.expected_value, 1.)
        self.assertLess(plan.evaluation.expected_value, 1.01)
        continuations = {child.answer: child for child in plan.continuations}
        self.assertIsNotNone(continuations["<"].evaluation)
        self.assertIsNone(continuations[">"].evaluation)

    def test_disabled_volume_threshold_preserves_existing_query_cost_and_count(self):
        old_config = replace(self.config, query_efficiency_stop_aware=False,
                             volume_confidence_threshold=None)
        new_config = replace(old_config, query_efficiency_stop_aware=True)
        old = self.evaluate(2, old_config)
        new = self.evaluate(2, new_config)
        self.assertAlmostEqual(new.expected_queries, old.expected_queries, places=12)
        self.assertAlmostEqual(new.evaluation.expected_value, old.evaluation.expected_value, places=12)
        self.assertAlmostEqual(new.gain, old.gain, places=12)
        self.assertAlmostEqual(new.rate, old.rate, places=12)
        self.assertEqual(new.expected_terminal_loss, new.evaluation.expected_value)

    def test_flag_does_not_change_probabilities_or_actual_one_step_candidates(self):
        stop_aware = self.evaluate()
        existing = self.evaluate(config=replace(self.config, query_efficiency_stop_aware=False))
        self.assertEqual(asdict(stop_aware.evaluation)["branches"], asdict(existing.evaluation)["branches"])
        self.assertEqual(stop_aware.evaluation.expected_value, existing.evaluation.expected_value)
        self.assertAlmostEqual(existing.gain, .015, places=10)
        old_fields = asdict(replace(self.config, query_efficiency_stop_aware=False))
        new_fields = asdict(self.config)
        self.assertEqual({key for key in new_fields if old_fields[key] != new_fields[key]},
                         {"query_efficiency_stop_aware"})

    def test_precomputed_volume_stop_threshold_is_inclusive_without_geometry_epsilon(self):
        for share, stopped in [(.99, True), (math.nextafter(.99, 0.), False),
                               (math.nextafter(.99, 1.), True)]:
            with self.subTest(share=share):
                analysis = CandidateAnalysis([0, 1], [], {0: 1. - share, 1: share},
                                             {0: 1. - share, 1: share})
                state = StateAnalysis(self.space, True, analysis)
                with patch.object(efficiency, "compute_query_candidates_for_depth_optimized",
                                  return_value=QueryCandidateData([])) as generation:
                    result = efficiency.compute_query_efficiency(
                        alternatives=self.alternatives, answered_queries=[],
                        config=self.config, precomputed_state_analysis=state,
                    )
                self.assertEqual(result.candidate_count, 2)
                self.assertIsNone(result.choice)
                self.assertEqual(generation.call_count, 0 if stopped else 2)

    def test_singleton_root_is_finished_even_with_volume_threshold_disabled(self):
        state = StateAnalysis(self.space, True, CandidateAnalysis([1], []))
        with patch.object(efficiency, "compute_query_candidates_for_depth_optimized") as generation:
            result = efficiency.compute_query_efficiency(
                alternatives=self.alternatives, answered_queries=[],
                config=replace(self.config, volume_confidence_threshold=None),
                precomputed_state_analysis=state,
            )
        self.assertEqual(result.candidate_count, 1)
        self.assertIsNone(result.choice)
        generation.assert_not_called()

    def test_fixed_depth_value_function_is_unchanged_when_flag_is_enabled(self):
        kwargs = dict(alternatives=self.alternatives, answered_queries=[], remaining_depth=2)
        old = compute_value_function_optimized(**kwargs,
                    config=replace(self.config, query_efficiency_stop_aware=False))
        new = compute_value_function_optimized(**kwargs, config=self.config)
        self.assertEqual(asdict(old), asdict(new))

    def test_child_policy_prefers_finished_volume_over_smaller_actual_candidate_count(self):
        # Query A retains three candidates for every answer, but finishes
        # with probability .6: E[L] = .6*1 + .4*3 = 1.8.  Query B retains two
        # candidates and never finishes: E[L] = 2.  The intended policy picks
        # A even though B has the lower genuine E[C].
        query_a = Query(ziel_index_a=0, ziel_index_b=1, value=1.)
        query_b = Query(ziel_index_a=0, ziel_index_b=1, value=2.)
        evaluation_a = efficiency.StopAwareQueryEvaluation(
            query=query_a, expected_value=3.,
            branches=(QueryBranchResult("<", .6, 3., 3, True),
                      QueryBranchResult(">", .4, 3., 3, True)),
            expected_terminal_loss=1.8,
            stop_aware_branches=(
                efficiency.StopAwareBranch("<", .6, 3, 3., 1., "volume_confidence", .995, 0),
                efficiency.StopAwareBranch(">", .4, 3, 3., 3., None, .7, 0),
            ),
        )
        evaluation_b = efficiency.StopAwareQueryEvaluation(
            query=query_b, expected_value=2.,
            branches=(QueryBranchResult("<", 1., 2., 2, True),),
            expected_terminal_loss=2.,
            stop_aware_branches=(efficiency.StopAwareBranch("<", 1., 2, 2., 2., None, .7, 0),),
        )
        for evaluation in (evaluation_a, evaluation_b):
            efficiency._validate_stop_aware_evaluation(evaluation)
        def evaluate(**kwargs):
            return (evaluation_a if kwargs["query"] == query_a else evaluation_b), 1., ()
        with patch.object(efficiency, "compute_query_candidates_for_depth_optimized",
                          return_value=QueryCandidateData([query_a, query_b])), \
             patch.object(efficiency, "_evaluate_stop_aware_query", side_effect=evaluate):
            chosen = efficiency._compute_stop_aware_depth_one_plan(
                alternatives=self.alternatives, answered_queries=[], weight_space=self.space,
                analysis=CandidateAnalysis([0, 1, 2], []), config=self.config,
            )
        self.assertEqual(chosen.query, query_a)
        self.assertEqual(chosen.expected_terminal_loss, 1.8)
        self.assertEqual(chosen.expected_value, 3.)

    def test_root_ranks_every_horizon_pair_by_effective_loss_rate(self):
        # At depth two, A has smaller E[C] and smaller E[L], but needs two
        # questions.  B wins by gain / expected questions: 1.6/1.5 > 1.9/2.
        query_a = Query(ziel_index_a=0, ziel_index_b=1, value=1.)
        query_b = Query(ziel_index_a=0, ziel_index_b=1, value=2.)
        def worker(payload):
            query, horizon = payload[3], payload[5]
            actual, loss, cost = ((2., 2., 1.) if horizon == 1 else
                (1.1, 1.1, 2.) if query == query_a else (2.5, 1.4, 1.5))
            return efficiency.StopAwareQueryEvaluation(
                query=query, expected_value=actual, branches=(),
                expected_terminal_loss=loss), cost, ()
        analysis = CandidateAnalysis([0, 1, 2], [], {0: 1., 1: 1., 2: 1.},
                                     {0: 1./3., 1: 1./3., 2: 1./3.})
        with patch.object(efficiency, "compute_query_candidates_for_depth_optimized",
                          return_value=QueryCandidateData([query_a, query_b])), \
             patch.object(efficiency, "_evaluate_efficiency_worker", side_effect=worker):
            result = efficiency.compute_query_efficiency(
                alternatives=self.alternatives, answered_queries=[], config=self.config,
                precomputed_state_analysis=StateAnalysis(self.space, True, analysis),
            )
        self.assertEqual(result.choice.horizon, 2)
        self.assertEqual(result.choice.evaluation.query, query_b)
        self.assertEqual(result.choice.expected_terminal_loss, 1.4)
        self.assertEqual(result.choice.evaluation.expected_value, 2.5)
        self.assertAlmostEqual(result.choice.rate, 1.6 / 1.5)

    def test_zero_probability_branch_does_not_build_an_infeasible_child(self):
        with patch.object(efficiency, "build_weight_space", wraps=build_weight_space) as build:
            evaluation, cost, _ = efficiency._evaluate_efficiency_worker(self.payload())
        self.assertEqual(cost, 1.)
        self.assertEqual(build.call_count, 2)
        self.assertEqual({call.args[1][-1].operator for call in build.call_args_list}, {"<", ">"})
        proof = next(item for item in evaluation.stop_aware_branches if item.answer == "=")
        self.assertEqual(proof.probability, 0.)
        self.assertEqual(proof.expected_followup_queries, 0)
        self.assertIsNone(proof.candidate_count)

    def test_positive_probability_infeasible_or_empty_candidate_child_fails(self):
        with patch.object(efficiency, "build_weight_space",
                          return_value=Mock(is_feasible=Mock(return_value=False))):
            with self.assertRaisesRegex(RuntimeError, "positive-probability"):
                efficiency._evaluate_efficiency_worker(self.payload())
        with patch.object(efficiency, "compute_candidate_analysis_for_mode",
                          return_value=CandidateAnalysis([], [], {}, {})):
            with self.assertRaisesRegex(RuntimeError, "positive-probability"):
                efficiency._evaluate_efficiency_worker(self.payload())

    def test_invalid_probabilities_are_not_silently_normalized(self):
        for less, greater in [(math.nan, 1.), (math.inf, 0.), (-1e-12, 1.), (.6, .6)]:
            with self.subTest(less=less, greater=greater):
                with patch.object(efficiency, "compute_exact_query_answer_probabilities",
                                  return_value={"<": less, "=": 0., ">": greater}):
                    with self.assertRaises(RuntimeError):
                        efficiency._evaluate_efficiency_worker(self.payload())

    def test_json_preserves_actual_candidate_expectation_separately_from_loss(self):
        serialized = efficiency.query_efficiency_evaluation_to_json(self.evaluate(2))
        self.assertTrue(serialized["planning_stop_aware"])
        self.assertEqual(serialized["expected_terminal_loss"], 1.)
        self.assertAlmostEqual(serialized["expected_terminal_candidates"], 1.985, places=10)
        child = next(item for item in serialized["continuations"] if item["answer"] == "<")
        self.assertEqual(child["candidate_count"], 2)
        self.assertEqual(child["expected_terminal_candidates"], 2.)
        self.assertEqual(child["expected_terminal_loss"], 1.)
        self.assertEqual(child["expected_queries"], 0)
        self.assertEqual(child["stop_reason"], "volume_confidence")
        self.assertAlmostEqual(child["maximum_volume_share"], 196. / 197., places=10)
        json.dumps(serialized, allow_nan=False)

    def test_enabled_worker_runs_after_spawn_and_keeps_proofs_pickleable(self):
        payload = pickle.loads(pickle.dumps(self.payload(2)))
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            output = pool.submit(efficiency._evaluate_efficiency_worker, payload).result(timeout=30)
        evaluation, cost, children = output
        self.assertIsInstance(evaluation, efficiency.StopAwareQueryEvaluation)
        self.assertEqual(evaluation.expected_terminal_loss, 1.)
        self.assertEqual(cost, 1.)
        self.assertEqual(len(children), 2)
        self.assertEqual(pickle.loads(pickle.dumps(output)), output)


if __name__ == "__main__":
    unittest.main()
