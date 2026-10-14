from __future__ import annotations

from dataclasses import asdict, replace
import unittest

from multistep.optimized import algorithm
from multistep.optimized.algorithm import AlgorithmConfig, solve
from multistep.optimized.value_function import CandidateAnalysis, StateAnalysis
from multistep.scripts.benchmark_exact_end_to_end import build_config
from multistep.src.models import AlternativenMatrix, Query
from multistep.src.query_probability import classify_query_answer
from multistep.src.value_function import ValueFunctionResult
from multistep.src.weight_space import build_weight_space


class AlgorithmConfigTest(unittest.TestCase):
    def test_resolves_the_existing_exact_benchmark_configuration(self) -> None:
        for source in ("grid", "both"):
            for confidence in (0.95, 0.99):
                for stop_aware in (False, True):
                    config = AlgorithmConfig(
                        higher_depth_query_source=source,
                        confidence_threshold=confidence,
                        consider_stop_in_planning=stop_aware,
                    )
                    old = build_config(
                        parallelize_root=True, max_workers=8, grid_size=5,
                        root_query_source=source, depth_one_query_source="ratio",
                        volume_confidence_threshold=confidence,
                        query_efficiency_stop_aware=stop_aware,
                    )
                    self.assertEqual(asdict(config.to_optimized_config()), asdict(old))

    def test_single_candidate_disables_confidence_in_actual_and_planned_stop(self) -> None:
        config = AlgorithmConfig(stop_rule="single_candidate", consider_stop_in_planning=True)
        self.assertIsNone(config.to_optimized_config().volume_confidence_threshold)
        self.assertTrue(config.to_optimized_config().query_efficiency_stop_aware)

    def test_all_three_fixed_and_adaptive_depths_are_valid(self) -> None:
        for strategy in ("fixed", "reduction_per_query"):
            for depth in (1, 2, 3):
                self.assertEqual(AlgorithmConfig(query_strategy=strategy, depth=depth).depth, depth)

    def test_invalid_settings_are_rejected_before_any_solver(self) -> None:
        settings = (
            {"query_strategy": "other"}, {"depth": 0}, {"depth": 4}, {"depth": True},
            {"higher_depth_query_source": "ratio"}, {"grid_size": 0},
            {"stop_rule": "other"}, {"confidence_threshold": 0.0},
            {"confidence_threshold": 1.1}, {"confidence_threshold": float("nan")},
            {"confidence_threshold": True}, {"confidence_threshold": None},
            {"confidence_threshold": "0.99"},
            {"consider_stop_in_planning": 1}, {"workers": 0},
            {"workers": True}, {"max_queries": 0}, {"max_queries": False},
        )
        for kwargs in settings:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AlgorithmConfig(**kwargs)


class UnifiedAlgorithmTest(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = AlternativenMatrix(entries=[[1., 0., 0.], [0., 1., 0.]])
        self.config = AlgorithmConfig(workers=1, grid_size=2, max_queries=10)

    def answerer(self, weights):
        return lambda query: classify_query_answer(weights, query, equality_tol=0.0)

    def test_real_callback_solves_and_does_not_receive_target_weights(self) -> None:
        seen = []

        def answer(query):
            self.assertIsInstance(query, Query)
            seen.append(query)
            return self.answerer([.8, .1, .1])(query)

        result = solve(self.matrix, self.config, answer)
        self.assertTrue(result["solved"])
        self.assertEqual(result["selected_candidate"], 0)
        self.assertIn(result["termination_reason"], {"exact_single_candidate", "volume_confidence"})
        self.assertEqual(result["question_count"], len(seen))
        self.assertGreater(len(seen), 0)
        self.assertNotIn("target_winner", result)
        self.assertNotIn("selection_is_correct", result)
        self.assertEqual(result["resolved_config"]["answer_probability_mode"], "exact_volume")
        self.assertEqual(result["resolved_config"]["depth_one_query_source_mode"], "ratio")

    def test_95_and_99_apply_different_actual_stops(self) -> None:
        matrix = [[1., 0., 0.], [.8, 1., 1.]]
        result95 = solve(matrix, replace(self.config, confidence_threshold=.95), self.answerer([.2, .4, .4]))
        result99 = solve(matrix, self.config, self.answerer([.2, .4, .4]))
        self.assertEqual(result95["termination_reason"], "volume_confidence")
        self.assertEqual(result95["question_count"], 0)
        self.assertGreaterEqual(result95["selected_candidate_volume_share"], .95)
        self.assertLess(result95["selected_candidate_volume_share"], .99)
        self.assertGreater(result99["question_count"], 0)
        self.assertEqual(result95["selected_candidate"], result99["selected_candidate"])

    def test_initial_history_is_preserved_and_included_in_total_count(self) -> None:
        initial = Query(ziel_index_a=0, ziel_index_b=1, value=1.).answer(">")
        result = solve(
            self.matrix, replace(self.config, max_queries=None),
            lambda _: self.fail("already solved history must not ask another query"),
            initial_answered_queries=[initial],
        )
        self.assertEqual(result["question_count"], 1)
        self.assertEqual(result["initial_question_count"], 1)
        self.assertEqual(result["new_question_count"], 0)
        self.assertEqual(result["answered_queries"], [initial.model_dump()])
        self.assertTrue(result["solved"])
        self.assertEqual(result["selected_candidate"], 0)

    def test_invalid_initial_goal_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            solve(self.matrix, self.config, lambda _: ">", initial_answered_queries=[
                Query(ziel_index_a=0, ziel_index_b=3, value=1.).answer(">"),
            ])

    def test_invalid_callback_answer_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "answer_query must return"):
            solve(self.matrix, self.config, lambda _: "yes")

    def test_no_query_is_structured_unsolved_without_fabricated_selection(self) -> None:
        class Session:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def analyze_state(self, *args, **kwargs):
                return StateAnalysis(build_weight_space(3, []), True, CandidateAnalysis([0, 1]))

            def compute(self, **kwargs):
                return ValueFunctionResult(2, 2., None, 2, (), True)

        # The private factory hook avoids modifying any actual planner.
        result = algorithm._solve_with_optimized_config(
            self.matrix, replace(self.config, stop_rule="single_candidate").to_optimized_config(),
            lambda _: ">", depth=2, session_factory=Session,
        )
        self.assertFalse(result["solved"])
        self.assertEqual(result["termination_reason"], "no_query")
        self.assertIsNone(result["selected_candidate"])
        self.assertEqual(result["question_count"], 0)

    def test_query_cap_counts_existing_answers(self) -> None:
        class Session:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def analyze_state(self, *args, **kwargs):
                return StateAnalysis(build_weight_space(3, []), True, CandidateAnalysis([0, 1]))

            def compute(self, **kwargs):
                raise AssertionError("total answer cap must be checked before planning")

        initial = Query(ziel_index_a=0, ziel_index_b=2, value=1.).answer(">")
        result = algorithm._solve_with_optimized_config(
            self.matrix, replace(self.config, stop_rule="single_candidate").to_optimized_config(),
            lambda _: ">", depth=2,
            max_queries=1, initial_answered_queries=[initial], session_factory=Session,
        )
        self.assertFalse(result["solved"])
        self.assertEqual(result["termination_reason"], "max_questions")
        self.assertEqual(result["question_count"], 1)

    def test_callbacks_run_on_every_state_and_actual_query(self) -> None:
        states, queries = [], []
        result = solve(
            self.matrix, self.config, self.answerer([.8, .1, .1]),
            on_state=lambda state, index: states.append((index, list(state.candidate_analysis.candidates))),
            on_query=lambda record: queries.append(dict(record)),
        )
        self.assertEqual(len(states), result["question_count"] + 1)
        self.assertEqual(len(queries), result["question_count"])
        self.assertEqual(states[-1][1], result["final_candidates"])
        self.assertEqual(queries[0]["answer"], result["queries"][0]["answer"])

    def test_fixed_singleton_planning_stop_retains_the_ordinary_candidate_loss(self) -> None:
        matrix = [[.9, .1, .3], [.2, .8, .4], [.4, .3, .9]]
        config = replace(self.config, stop_rule="single_candidate", consider_stop_in_planning=True)
        result = solve(matrix, config, self.answerer([.2, .3, .5]))
        self.assertTrue(result["exactly_solved"])
        self.assertEqual(result["selected_candidate"], 2)
        self.assertGreater(result["question_count"], 0)
        self.assertTrue(all(
            query["expected_terminal_loss"] == query["expected_value"]
            for query in result["queries"]
        ))

    def test_empty_candidate_state_raises_after_the_state_observer(self) -> None:
        observations = []

        class Session:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def analyze_state(self, *args, **kwargs):
                return StateAnalysis(build_weight_space(3, []), True, CandidateAnalysis([]))

        with self.assertRaisesRegex(RuntimeError, "no candidate alternatives"):
            algorithm._solve_with_optimized_config(
                self.matrix, self.config.to_optimized_config(), lambda _: ">", depth=2,
                session_factory=Session, on_state=lambda state, index: observations.append(index),
            )
        self.assertEqual(observations, [0])


if __name__ == "__main__":
    unittest.main()
