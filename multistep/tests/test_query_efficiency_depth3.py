from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import itertools
import json
import math
import multiprocessing
import random
import unittest
from unittest.mock import patch

from multistep.optimized import query_efficiency as efficiency
from multistep.optimized.value_function import CandidateAnalysis, OptimizedMultistepConfig, QueryCandidateData
from multistep.src.models import AlternativenMatrix, Query
from multistep.src.value_function import QueryBranchResult, QueryEvaluation
from multistep.src.weight_space import build_weight_space


class QueryEfficiencyDepthThreeTest(unittest.TestCase):
    def setUp(self):
        self.alternatives = AlternativenMatrix(entries=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.space = build_weight_space(3, [])
        self.query = Query(ziel_index_a=0, ziel_index_b=1, value=1.)
        self.config = OptimizedMultistepConfig(
            answer_probability_mode="exact_volume", use_ratio_terminal_counts=True,
            grid_size=2, volume_confidence_threshold=None,
        )

    def option(self, loss, cost, identity=1.):
        query = Query(ziel_index_a=0, ziel_index_b=2, value=identity)
        evaluation = QueryEvaluation(query, float(loss), (
            QueryBranchResult("<", 1., float(loss), max(1, math.ceil(loss)), True),
        ))
        return efficiency._ContinuationOption(evaluation, float(loss), float(loss), float(cost))

    def test_global_rate_does_not_choose_child_standalone_rate_or_minimum_loss(self):
        # Root has C=5.  Its .5 branch leaves C=2; its other .5 branch is
        # already a singleton.  In the unfinished branch, B wins locally:
        # (2-1)/2=.5 > (2-1.6)/1=.4.  The complete root prefers A instead:
        # (5-(.5*1.6+.5*1))/(1+.5*1)=2.466... > (5-1)/(1+.5*2)=2.
        a = self.option(1.6, 1., 1.)
        b = self.option(1., 2., 2.)
        singleton = efficiency._ContinuationOption(None, 1., 1., 0., stop_reason="single_candidate")
        self.assertGreater((2. - b.expected_terminal_loss) / b.expected_queries,
                           (2. - a.expected_terminal_loss) / a.expected_queries)
        selected = efficiency._select_global_continuations(5, (.5, .5), ((a, b), (singleton,)))
        self.assertEqual(selected, (a, singleton))

    def test_weighted_early_termination_uses_executed_questions(self):
        unfinished = self.option(1., 2.)
        singleton = efficiency._ContinuationOption(None, 1., 1., 0., stop_reason="single_candidate")
        with patch.object(efficiency, "compute_exact_query_answer_probabilities", return_value={"<": .95, "=": 0., ">": .05}), \
             patch.object(efficiency, "compute_candidate_analysis_for_mode", side_effect=[CandidateAnalysis([0, 1], []), CandidateAnalysis([2], [])]), \
             patch.object(efficiency, "_depth_two_child_frontier", side_effect=[(unfinished,), (singleton,)]):
            evaluation, cost, children = efficiency._evaluate_efficiency_worker((
                self.alternatives, [], self.space, self.query, None, 3,
                self.config, [0, 1, 2], None, "grid",
            ))
        self.assertEqual(cost, 2.9)
        self.assertEqual(evaluation.expected_value, 1.)
        self.assertEqual([child.expected_queries for child in children], [2., 0.])
        self.assertEqual(children[1].stop_reason, "single_candidate")

    def test_frontier_result_matches_full_cartesian_oracle(self):
        rng = random.Random(173)
        for branch_count in (1, 2, 3):
            probabilities = tuple([1. / branch_count] * branch_count)
            for _ in range(30):
                options = tuple(tuple(self.option(rng.randrange(8, 33) / 8.,
                                                  rng.randrange(8, 17) / 8.,
                                                  float(index + 1))
                                      for index in range(7))
                                for _ in range(branch_count))
                def key(chosen):
                    loss = math.fsum(p * item.expected_terminal_loss for p, item in zip(probabilities, chosen))
                    cost = 1. + math.fsum(p * item.expected_queries for p, item in zip(probabilities, chosen))
                    return (-(5. - loss) / cost, cost,
                            tuple(efficiency._option_identity(item) for item in chosen))
                expected = min(itertools.product(*options), key=key)
                frontiers = tuple(efficiency._pareto_continuation_options(list(items)) for items in options)
                actual = efficiency._select_global_continuations(5, probabilities, frontiers)
                self.assertEqual(key(actual), key(expected))

    def test_pareto_pruning_does_not_use_an_approximation_tolerance(self):
        cheaper = self.option(2., 1., 1.)
        less_loss = self.option(math.nextafter(2., 0.), math.nextafter(1., 2.), 2.)
        dominated = self.option(3., 2., 3.)
        self.assertEqual(efficiency._pareto_continuation_options([dominated, less_loss, cheaper]),
                         (cheaper, less_loss))

    def test_zero_probability_answers_do_not_create_child_states(self):
        singleton = efficiency._ContinuationOption(None, 1., 1., 0., stop_reason="single_candidate")
        with patch.object(efficiency, "compute_exact_query_answer_probabilities", return_value={"<": 1., "=": 0., ">": 0.}), \
             patch.object(efficiency, "compute_candidate_analysis_for_mode", return_value=CandidateAnalysis([0], [])) as analysis, \
             patch.object(efficiency, "_depth_two_child_frontier", return_value=(singleton,)):
            evaluation, cost, children = efficiency._evaluate_efficiency_worker((
                self.alternatives, [], self.space, self.query, None, 3,
                self.config, [0, 1, 2], None, "ratio",
            ))
        analysis.assert_called_once()
        self.assertEqual(cost, 1.)
        self.assertEqual(len(children), 1)
        self.assertEqual(len(evaluation.branches), 3)
        self.assertIsNone(evaluation.branches[1].child_candidate_count)

    def test_unfinished_child_without_queries_has_zero_question_cost(self):
        with patch.object(efficiency, "compute_query_candidates_for_depth_optimized", return_value=QueryCandidateData([])), \
             patch.object(efficiency, "_evaluate_efficiency_worker") as evaluate:
            frontier = efficiency._depth_two_child_frontier(
                alternatives=self.alternatives, answered_queries=[], weight_space=self.space,
                analysis=CandidateAnalysis([0, 1, 2], []), config=self.config,
            )
        evaluate.assert_not_called()
        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0].expected_queries, 0.)
        self.assertEqual(frontier[0].expected_terminal_loss, 3.)
        self.assertIsNone(frontier[0].stop_reason)

    def test_volume_stop_is_absorbing_and_actual_count_remains_separate(self):
        alternatives = AlternativenMatrix(entries=[[1., 0.], [48. / 49., 1.]])
        query = Query(ziel_index_a=0, ziel_index_b=1, value=197. / 3.)
        config = replace(self.config, query_efficiency_stop_aware=True, volume_confidence_threshold=.99)
        plans = efficiency._evaluate_horizon_candidates(
            alternatives=alternatives, answered_queries=[], weight_space=build_weight_space(2, []),
            query_candidates=[query], horizon=3, config=config, candidates=[0, 1],
            ratio_intervals_by_goal_pair=None, query_sources={}, executor=None,
        )
        plan = plans[0]
        self.assertEqual(plan.expected_queries, 1.)
        self.assertEqual(plan.expected_terminal_loss, 1.)
        self.assertAlmostEqual(plan.evaluation.expected_value, 1.985, places=10)
        self.assertTrue(all(child.evaluation is None for child in plan.continuations))
        less = next(child for child in plan.continuations if child.answer == "<")
        self.assertEqual(less.stop_reason, "volume_confidence")
        self.assertEqual(less.candidate_count, 2)
        self.assertEqual(less.expected_queries, 0.)

    def test_recursive_json_preserves_all_selected_continuations(self):
        leaf = efficiency.QueryEfficiencyContinuation("<", .5, 2, 1., self.option(1., 1.).evaluation)
        child = efficiency.QueryEfficiencyContinuation(
            "<", 1., 3, 1., self.option(1., 1.).evaluation,
            expected_queries=1.5, continuations=(leaf,),
        )
        plan = efficiency.QueryEfficiencyEvaluation(
            self.option(1., 1.).evaluation, 3, 2.5, 2., .8, (child,),
        )
        serialized = efficiency.query_efficiency_evaluation_to_json(plan)
        self.assertEqual(serialized["continuations"][0]["expected_queries"], 1.5)
        self.assertEqual(serialized["continuations"][0]["continuations"][0]["expected_queries"], 1)
        self.assertEqual(serialized["continuations"][0]["continuations"][0]["query"]["branches"][0]["child_value"], 1.)
        json.dumps(serialized, allow_nan=False)

    def test_serial_parallel_horizon_three_and_grid_only_are_equal(self):
        # This checks real exact-volume geometry and spawned root workers;
        # child planners stay serial even when root parallelization is active.
        alternatives = AlternativenMatrix(entries=[[1., 0.], [0., 1.]])
        config = replace(self.config, grid_depth_query_source_mode="grid")
        serial = efficiency.compute_query_efficiency(
            alternatives=alternatives, answered_queries=[], config=config, max_horizon=3,
        )
        parallel_config = replace(config, parallelize_root=True, max_workers=2)
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            parallel = efficiency.compute_query_efficiency(
                alternatives=alternatives, answered_queries=[], config=parallel_config,
                max_horizon=3, executor=pool,
            )
        self.assertEqual(serial.choice, parallel.choice)
        self.assertEqual(serial.evaluations_by_horizon, parallel.evaluations_by_horizon)
        self.assertEqual(set(serial.best_by_horizon), {1, 2, 3})
        existing = efficiency.compute_query_efficiency(
            alternatives=alternatives, answered_queries=[], config=config, max_horizon=2,
        )
        self.assertEqual(existing.evaluations_by_horizon[1], serial.evaluations_by_horizon[1])
        self.assertEqual(existing.evaluations_by_horizon[2], serial.evaluations_by_horizon[2])


if __name__ == "__main__":
    unittest.main()
