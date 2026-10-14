from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import math
import multiprocessing
import pickle
import unittest
from unittest.mock import patch

from multistep.optimized import query_efficiency as efficiency
from multistep.optimized.value_function import (
    CandidateAnalysis, OptimizedMultistepConfig, OptimizedValueFunctionSession,
    QueryCandidateData, StateAnalysis, evaluate_query_candidate_optimized,
)
from multistep.scripts import benchmark_exact_end_to_end as benchmark
from multistep.src.models import AlternativenMatrix, Query
from multistep.src.value_function import QueryBranchResult, QueryEvaluation, ValueFunctionResult
from multistep.src.weight_space import build_weight_space


class QueryEfficiencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.alternatives = AlternativenMatrix(entries=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.config = OptimizedMultistepConfig(answer_probability_mode="exact_volume", use_ratio_terminal_counts=True, grid_size=2)
        self.space = build_weight_space(3, [])
        self.query = Query(ziel_index_a=0, ziel_index_b=1, value=1.)
        self.leaf_query = Query(ziel_index_a=0, ziel_index_b=2, value=1.)

    def payload(self, horizon: int = 2) -> efficiency.RootPayload:
        return (self.alternatives, [], self.space, self.query, None, horizon, self.config, [0, 1, 2], None, "ratio")

    def child(self, count: int, query: Query | None = None, terminal_count: int = 1) -> ValueFunctionResult:
        evaluation = QueryEvaluation(query, float(terminal_count), (QueryBranchResult("<", 1., float(terminal_count), terminal_count, True),)) if query else None
        return ValueFunctionResult(1, float(terminal_count if query else count), query, count, (evaluation,) if evaluation else (), True)

    def weighted_core(self, **kwargs) -> QueryEvaluation:
        first = self.child(2, self.leaf_query)
        second = self.child(1)
        observer = kwargs.get("_child_result_observer")
        if observer is not None:
            observer("<", .95, first)
            observer(">", .05, second)
        return QueryEvaluation(self.query, 1., (
            QueryBranchResult("<", .95, 1., 2, True),
            QueryBranchResult(">", .05, 1., 1, True),
        ))

    def test_branch_weighted_question_cost_and_ratio_of_expectations(self) -> None:
        with patch.object(efficiency, "evaluate_query_candidate_optimized", side_effect=self.weighted_core):
            plans = efficiency._evaluate_horizon_candidates(
                alternatives=self.alternatives, answered_queries=[], weight_space=self.space,
                query_candidates=[self.query], horizon=2, config=self.config, candidates=[0, 1, 2],
                ratio_intervals_by_goal_pair=None, query_sources={}, executor=None,
            )
        plan = plans[0]
        self.assertEqual(plan.expected_queries, 1.95)
        self.assertEqual(plan.gain, 2.)
        self.assertEqual(plan.rate, 2. / 1.95)
        self.assertNotAlmostEqual(plan.rate, .95 * (2. / 2.) + .05 * (2. / 1.))
        self.assertEqual([child.candidate_count for child in plan.continuations], [2, 1])
        serialized = efficiency.query_efficiency_evaluation_to_json(plan)
        self.assertEqual([child["expected_queries"] for child in serialized["continuations"]], [1, 0])
        self.assertEqual(serialized["continuations"][0]["query"]["branches"][0]["child_candidate_count"], 1)
        json.dumps(serialized, allow_nan=False)

    def test_nonterminal_no_query_child_does_not_execute_a_question(self) -> None:
        def core(**kwargs):
            observer = kwargs["_child_result_observer"]
            observer("<", .7, self.child(2))
            observer(">", .3, self.child(2, self.leaf_query))
            return QueryEvaluation(self.query, 1.7, (
                QueryBranchResult("<", .7, 2., 2, True), QueryBranchResult(">", .3, 1., 2, True),
            ))
        with patch.object(efficiency, "evaluate_query_candidate_optimized", side_effect=core):
            _, cost, children = efficiency._evaluate_efficiency_worker(self.payload())
        self.assertEqual(cost, 1.3)
        self.assertIsNone(children[0].evaluation)
        self.assertEqual(children[0].candidate_count, 2)

    def test_hand_calculated_root_and_leaf_probabilities(self) -> None:
        # Joint leaf probabilities are .8*.25=.2, .8*.75=.6, and .2.
        # E[C] = .2*1 + .6*3 + .2*1 = 2.2; E[Q] = 1 + .8 = 1.8.
        leaf = QueryEvaluation(self.leaf_query, 2.5, (
            QueryBranchResult("<", .25, 1., 1, True),
            QueryBranchResult(">", .75, 3., 3, True),
        ))
        def core(**kwargs):
            observer = kwargs["_child_result_observer"]
            observer("<", .8, ValueFunctionResult(1, 2.5, self.leaf_query, 3, (leaf,), True))
            observer(">", .2, self.child(1))
            return QueryEvaluation(self.query, 2.2, (
                QueryBranchResult("<", .8, 2.5, 3, True),
                QueryBranchResult(">", .2, 1., 1, True),
            ))
        with patch.object(efficiency, "evaluate_query_candidate_optimized", side_effect=core):
            plans = efficiency._evaluate_horizon_candidates(
                alternatives=self.alternatives, answered_queries=[], weight_space=self.space,
                query_candidates=[self.query], horizon=2, config=self.config, candidates=[0, 1, 2],
                ratio_intervals_by_goal_pair=None, query_sources={}, executor=None,
            )
        plan = plans[0]
        self.assertEqual(plan.evaluation.expected_value, 2.2)
        self.assertEqual(plan.expected_queries, 1.8)
        self.assertAlmostEqual(plan.gain, .8)
        self.assertAlmostEqual(plan.rate, .8 / 1.8)
        self.assertAlmostEqual(
            plan.evaluation.expected_value,
            .8 * (.25 * 1. + .75 * 3.) + .2 * 1.,
        )
        self.assertEqual(plan.continuations[0].evaluation.branches[1].probability, .75)

    def test_horizon_one_cost_is_one(self) -> None:
        with patch.object(efficiency, "evaluate_query_candidate_optimized", side_effect=self.weighted_core) as mocked:
            _, cost, children = efficiency._evaluate_efficiency_worker(self.payload(1))
        self.assertEqual(cost, 1.)
        self.assertEqual(children, ())
        self.assertIsNone(mocked.call_args.kwargs["_child_result_observer"])

    def test_zero_probability_infeasible_branch_has_no_cost(self) -> None:
        def core(**kwargs):
            kwargs["_child_result_observer"]("<", 1., self.child(2, self.leaf_query))
            return QueryEvaluation(self.query, 1., (
                QueryBranchResult("<", 1., 1., 2, True),
                QueryBranchResult(">", 0., 0., None, False),
            ))
        with patch.object(efficiency, "evaluate_query_candidate_optimized", side_effect=core):
            _, cost, _ = efficiency._evaluate_efficiency_worker(self.payload())
        self.assertEqual(cost, 2.)

    def test_rejects_positive_probability_infeasible_branch_even_above_one(self) -> None:
        malformed = QueryEvaluation(self.query, 1.8, (
            QueryBranchResult("<", .1, 0., 0, False),
            QueryBranchResult(">", .9, 2., 2, True),
        ))
        with patch.object(efficiency, "evaluate_query_candidate_optimized", return_value=malformed):
            with self.assertRaisesRegex(RuntimeError, "positive-probability"):
                efficiency._evaluate_efficiency_worker(self.payload(1))

    def test_rejects_corrupt_selected_leaf_policy(self) -> None:
        def core(**kwargs):
            malformed = QueryEvaluation(self.leaf_query, 1.8, (
                QueryBranchResult("<", .1, 0., 0, False), QueryBranchResult(">", .9, 2., 2, True),
            ))
            kwargs["_child_result_observer"]("<", 1., ValueFunctionResult(1, 1.8, self.leaf_query, 2, (malformed,), True))
            return QueryEvaluation(self.query, 1.8, (QueryBranchResult("<", 1., 1.8, 2, True),))
        with patch.object(efficiency, "evaluate_query_candidate_optimized", side_effect=core):
            with self.assertRaisesRegex(RuntimeError, "positive-probability"):
                efficiency._evaluate_efficiency_worker(self.payload())

    def test_rejects_missing_child_observation(self) -> None:
        valid = QueryEvaluation(self.query, 1., (QueryBranchResult("<", 1., 1., 1, True),))
        with patch.object(efficiency, "evaluate_query_candidate_optimized", return_value=valid):
            with self.assertRaisesRegex(RuntimeError, "observed exactly once"):
                efficiency._evaluate_efficiency_worker(self.payload())

    def test_rejects_nonfinite_probabilities_or_expectations(self) -> None:
        for value in [math.nan, math.inf, -math.inf]:
            with self.subTest(value=value):
                malformed = QueryEvaluation(self.query, 1., (QueryBranchResult("<", value, 1., 1, True),))
                with patch.object(efficiency, "evaluate_query_candidate_optimized", return_value=malformed):
                    with self.assertRaises(RuntimeError):
                        efficiency._evaluate_efficiency_worker(self.payload(1))

    def test_ranks_every_root_plan_not_only_minimum_terminal_expectation(self) -> None:
        # The lowest terminal count at horizon two is query A (2.2), but query B
        # has a better gain/question rate: 2.6/1.5 > 2.8/2 > 1.2/1.
        def raw(payload):
            horizon, query = payload[5], payload[3]
            terminal, cost = (3.8, 1.) if horizon == 1 else ((2.2, 2.) if query == self.query else (2.4, 1.5))
            return QueryEvaluation(query, terminal, ()), cost, ()
        state = StateAnalysis(self.space, True, CandidateAnalysis(list(range(5)), []))
        data = QueryCandidateData([self.query, self.leaf_query])
        with patch.object(efficiency, "compute_query_candidates_for_depth_optimized", return_value=data), patch.object(efficiency, "_evaluate_efficiency_worker", side_effect=raw):
            result = efficiency.compute_query_efficiency(alternatives=self.alternatives, answered_queries=[], config=self.config, precomputed_state_analysis=state)
        self.assertEqual(result.choice.horizon, 2)
        self.assertEqual(result.choice.evaluation.query, self.leaf_query)
        self.assertAlmostEqual(result.choice.rate, 2.6 / 1.5)
        self.assertEqual(len(result.evaluations_by_horizon[2]), 2)
        self.assertEqual(result.best_by_horizon[1].expected_queries, 1.)
        self.assertEqual(set(result.evaluation_seconds_by_horizon), {1, 2})

    def test_deterministic_ties_prefer_cost_then_horizon_then_identity(self) -> None:
        base = efficiency.QueryEfficiencyEvaluation(QueryEvaluation(self.query, 1., ()), 2, 2., 2., 1.)
        cheaper = replace(base, expected_queries=1., gain=1.)
        shallower = replace(cheaper, horizon=1)
        other_query = replace(shallower, evaluation=QueryEvaluation(self.leaf_query, 1., ()))
        ordered = sorted([other_query, base, cheaper, shallower], key=efficiency.query_efficiency_sort_key)
        self.assertEqual(ordered, [shallower, other_query, cheaper, base])

    def test_root_sources_ratio_at_one_both_at_two(self) -> None:
        state = StateAnalysis(self.space, True, CandidateAnalysis([0, 1, 2], []))
        observed = {}
        def evaluate(**kwargs):
            observed[kwargs["horizon"]] = kwargs["query_candidates"]
            return ()
        with patch("multistep.optimized.value_function.compute_onestep_query_candidates", return_value=[self.leaf_query]), patch("multistep.optimized.value_function.compute_grid_queries_for_config", return_value=[self.query]), patch.object(efficiency, "_evaluate_horizon_candidates", side_effect=evaluate):
            efficiency.compute_query_efficiency(alternatives=self.alternatives, answered_queries=[], config=self.config, precomputed_state_analysis=state)
        self.assertEqual(observed[1], [self.leaf_query])
        self.assertEqual(observed[2], [self.query, self.leaf_query])

    def test_single_candidate_has_no_plan_or_question(self) -> None:
        state = StateAnalysis(self.space, True, CandidateAnalysis([1], []))
        with patch.object(efficiency, "_evaluate_horizon_candidates") as evaluate:
            result = efficiency.compute_query_efficiency(alternatives=self.alternatives, answered_queries=[], config=self.config, precomputed_state_analysis=state)
        self.assertIsNone(result.choice)
        self.assertEqual(result.evaluations_by_horizon, {})
        evaluate.assert_not_called()

    def test_no_available_root_queries_returns_no_plan(self) -> None:
        state = StateAnalysis(self.space, True, CandidateAnalysis([0, 1, 2], []))
        with patch.object(efficiency, "compute_query_candidates_for_depth_optimized", return_value=QueryCandidateData([])):
            result = efficiency.compute_query_efficiency(alternatives=self.alternatives, answered_queries=[], config=self.config, precomputed_state_analysis=state)
        self.assertIsNone(result.choice)
        self.assertEqual(result.best_by_horizon, {1: None, 2: None})

    def test_rejects_horizon_four_and_unsupported_sources_before_any_solver(self) -> None:
        for horizon, config in [(4, self.config), (2, replace(self.config, grid_depth_query_source_mode="central")), (2, replace(self.config, adaptive_depth_candidate_threshold=2)), (2, replace(self.config, answer_probability_mode="sampling"))]:
            with self.subTest(horizon=horizon, config=config):
                with patch.object(efficiency, "build_weight_space") as build:
                    with self.assertRaises(ValueError):
                        efficiency.compute_query_efficiency(alternatives=self.alternatives, answered_queries=[], config=config, max_horizon=horizon)
                build.assert_not_called()
        with self.assertRaises(ValueError):
            benchmark.solve_problem(self.alternatives, [.8, .1, .1], 10, 4, root_query_source="both", depth_one_query_source="ratio", query_strategy="reduction_per_query")

    def test_child_observer_preserves_existing_query_evaluation(self) -> None:
        kwargs = dict(alternatives=self.alternatives, answered_queries=[], weight_space=self.space, query=self.query, samples=None, remaining_depth=2, config=self.config, candidate_subset=[0, 1, 2])
        existing = evaluate_query_candidate_optimized(**kwargs)
        children = []
        observed = evaluate_query_candidate_optimized(**kwargs, _child_result_observer=lambda *args: children.append(args))
        self.assertEqual(asdict(existing), asdict(observed))
        self.assertTrue(children)

    def test_spawn_worker_payload_and_result_serialization(self) -> None:
        payload = pickle.loads(pickle.dumps(self.payload()))
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            evaluation, cost, children = pool.submit(efficiency._evaluate_efficiency_worker, payload).result(timeout=30)
        self.assertEqual(evaluation.query, self.query)
        self.assertTrue(1. <= cost <= 2.)
        self.assertTrue(children)
        self.assertEqual(pickle.loads(pickle.dumps((evaluation, cost, children))), (evaluation, cost, children))

    def test_explicit_fixed_strategy_preserves_default_trajectory_and_lexicography(self) -> None:
        kwargs = dict(alternatives=self.alternatives, target_weights=[.8, .1, .1], max_questions=10, depth=2, grid_size=2, root_query_source="both", depth_one_query_source="ratio")
        default = benchmark.solve_problem(**kwargs)
        explicit = benchmark.solve_problem(**kwargs, query_strategy="fixed")
        def semantics(result):
            return {key: value for key, value in result.items() if key not in {"seconds", "query_seconds", "queries", "volume_cache"}}, [{key: value for key, value in query.items() if key != "seconds"} for query in result["queries"]]
        self.assertEqual(semantics(default), semantics(explicit))
        self.assertTrue(default["queries"])
        self.assertNotIn("selected_horizon", default["queries"][0])
        self.assertTrue(default["queries"][0]["lexicographic_expected_values"])

    def test_experimental_benchmark_records_both_horizons_and_query_based_rate(self) -> None:
        result = benchmark.solve_problem(
            self.alternatives, [.8, .1, .1], 10, 2, grid_size=2,
            root_query_source="both", depth_one_query_source="ratio", query_strategy="reduction_per_query",
        )
        self.assertTrue(result["selection_is_correct"])
        self.assertTrue(result["queries"])
        count_before = result["initial_candidate_count"]
        for record in result["queries"]:
            self.assertIn(record["selected_horizon"], (1, 2))
            self.assertTrue(1. <= record["expected_queries"] <= record["selected_horizon"])
            self.assertEqual(record["gain"], count_before - record["expected_terminal_candidates"])
            self.assertEqual(record["rate"], record["gain"] / record["expected_queries"])
            self.assertEqual({plan["horizon"] for plan in record["horizon_best_plans"]}, {1, 2})
            self.assertEqual(set(record["horizon_evaluation_seconds"]), {"1", "2"})
            self.assertTrue(all(value >= 0. for value in record["horizon_evaluation_seconds"].values()))
            count_before = record["remaining_candidate_count"]

    def test_hypothetical_planner_ignores_volume_confidence_threshold(self) -> None:
        low = efficiency.compute_query_efficiency(
            alternatives=self.alternatives, answered_queries=[], config=replace(self.config, volume_confidence_threshold=.01),
        )
        disabled = efficiency.compute_query_efficiency(
            alternatives=self.alternatives, answered_queries=[], config=replace(self.config, volume_confidence_threshold=None),
        )
        self.assertEqual(low.choice, disabled.choice)
        self.assertEqual(low.evaluations_by_horizon, disabled.evaluations_by_horizon)

    def test_closed_session_rejects_experimental_planning(self) -> None:
        session = OptimizedValueFunctionSession(self.alternatives, self.config)
        session.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.compute_query_efficiency([])


if __name__ == "__main__":
    unittest.main()
