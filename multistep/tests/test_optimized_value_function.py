from __future__ import annotations

import unittest
from unittest.mock import patch

from multistep.optimized.profiling import collect_optimization_profile
from multistep.optimized.value_function import (
    OptimizedMultistepConfig,
    OptimizedValueFunctionSession,
    compute_candidate_set_for_subset,
    compute_central_query_candidates,
    compute_query_candidates_for_depth_optimized,
    compute_ratio_relevant_candidate_set,
    compute_posterior_quantile_query_candidates,
    compute_supported_query_answers,
    compute_supported_query_answers_with_sample_evidence,
    compute_terminal_candidate_count_fallback,
    compute_terminal_candidate_count_from_ratio_intervals,
    compute_value_function_optimized,
    estimate_supported_answer_probabilities,
    filter_already_answered_queries,
    is_query_already_answered,
    shortlist_query_candidates_by_sample_balance,
    score_query_candidates_by_posterior,
)
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.ratio_intervals import compute_all_ratio_intervals
from multistep.src.models import Query
from multistep.src.candidates import compute_candidate_set
from multistep.src.models import AlternativenMatrix
from multistep.src.value_function import (
    MultistepConfig,
    QueryEvaluation,
    compute_value_function,
    refine_query_evaluations_lexicographically,
)
from multistep.src.weight_space import build_weight_space


class OptimizedValueFunctionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.alternatives = AlternativenMatrix(
            entries=[
                [0.9, 0.2, 0.1],
                [0.2, 0.9, 0.1],
                [0.2, 0.1, 0.9],
                [0.6, 0.6, 0.6],
            ]
        )

    def test_central_query_candidates_have_one_query_per_goal_pair(self) -> None:
        queries = compute_central_query_candidates(
            build_weight_space(goal_count=3, answered_queries=[])
        )

        self.assertEqual(len(queries), 3)
        self.assertEqual(
            {
                (int(query.ziel_index_a), int(query.ziel_index_b))
                for query in queries
            },
            {(0, 1), (0, 2), (1, 2)},
        )
        for query in queries:
            self.assertAlmostEqual(float(query.value), 1.0, places=10)

    def test_lexicographic_key_uses_e_two_to_break_e_three_tie(self) -> None:
        first = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
        second = Query(ziel_index_a=0, ziel_index_b=2, value=1.0)
        initial_evaluations = (
            QueryEvaluation(query=first, expected_value=1.0, branches=()),
            QueryEvaluation(query=second, expected_value=1.0, branches=()),
        )
        evaluated_depths: list[int] = []

        def evaluate_tied_queries(
            queries: list[Query],
            depth: int,
        ) -> tuple[QueryEvaluation, ...]:
            evaluated_depths.append(depth)
            values = {first: 1.0, second: 2.0}
            return tuple(
                QueryEvaluation(
                    query=query,
                    expected_value=values[query],
                    branches=(),
                )
                for query in queries
            )

        refined, best = refine_query_evaluations_lexicographically(
            query_evaluations=initial_evaluations,
            remaining_depth=3,
            evaluate_queries_at_depth=evaluate_tied_queries,
        )

        self.assertEqual(best.query, first)
        self.assertEqual(evaluated_depths, [2])
        self.assertEqual(refined[0].lexicographic_expected_values, (1.0, 1.0))
        self.assertEqual(refined[1].lexicographic_expected_values, (1.0, 2.0))

    def test_lexicographic_key_continues_to_e_one_if_e_two_is_tied(self) -> None:
        first = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
        second = Query(ziel_index_a=0, ziel_index_b=2, value=1.0)
        initial_evaluations = (
            QueryEvaluation(query=first, expected_value=1.0, branches=()),
            QueryEvaluation(query=second, expected_value=1.0, branches=()),
        )

        def evaluate_tied_queries(
            queries: list[Query],
            depth: int,
        ) -> tuple[QueryEvaluation, ...]:
            values = {
                2: {first: 1.0, second: 1.0},
                1: {first: 2.0, second: 1.0},
            }
            return tuple(
                QueryEvaluation(
                    query=query,
                    expected_value=values[depth][query],
                    branches=(),
                )
                for query in queries
            )

        refined, best = refine_query_evaluations_lexicographically(
            query_evaluations=initial_evaluations,
            remaining_depth=3,
            evaluate_queries_at_depth=evaluate_tied_queries,
        )

        self.assertEqual(best.query, second)
        self.assertEqual(
            refined[0].lexicographic_expected_values,
            (1.0, 1.0, 2.0),
        )
        self.assertEqual(
            refined[1].lexicographic_expected_values,
            (1.0, 1.0, 1.0),
        )

    def test_candidate_subset_matches_full_candidate_set_when_subset_is_all(self) -> None:
        weight_space = build_weight_space(
            goal_count=self.alternatives.get_anzahl_spalten(),
            answered_queries=[],
        )
        full_candidates = compute_candidate_set(
            alternatives=self.alternatives,
            weight_space=weight_space,
        )

        subset_candidates = compute_candidate_set_for_subset(
            alternatives=self.alternatives,
            weight_space=weight_space,
            candidate_subset=list(range(self.alternatives.get_anzahl_zeilen())),
        )

        self.assertEqual(subset_candidates, full_candidates)

    def test_exact_optimized_depth_two_matches_reference_value(self) -> None:
        reference_config = MultistepConfig(
            sample_count=80,
            burn_in=20,
            thinning=2,
            random_seed=7,
            grid_size=5,
            max_query_value=10.0,
        )
        optimized_config = OptimizedMultistepConfig(
            sample_count=80,
            burn_in=20,
            thinning=2,
            random_seed=7,
            grid_size=5,
            max_query_value=10.0,
            reuse_conditioned_samples=False,
            parallelize_root=False,
            candidate_count_mode="closed_lp",
            include_ratio_queries_on_grid_depths=False,
        )

        reference_result = compute_value_function(
            alternatives=self.alternatives,
            answered_queries=[],
            remaining_depth=2,
            config=reference_config,
        )
        optimized_result = compute_value_function_optimized(
            alternatives=self.alternatives,
            answered_queries=[],
            remaining_depth=2,
            config=optimized_config,
        )

        self.assertAlmostEqual(optimized_result.value, reference_result.value)
        self.assertEqual(optimized_result.best_query, reference_result.best_query)

    def test_is_query_already_answered_detects_same_query(self) -> None:
        query = Query(ziel_index_a=0, ziel_index_b=1, value=2.0)

        self.assertTrue(
            is_query_already_answered(
                query=query,
                answered_queries=[query.answer(">")],
            )
        )

    def test_is_query_already_answered_detects_mirrored_query(self) -> None:
        query = Query(ziel_index_a=0, ziel_index_b=1, value=2.0)
        mirrored_answer = Query(ziel_index_a=1, ziel_index_b=0, value=0.5).answer("<")

        self.assertTrue(
            is_query_already_answered(
                query=query,
                answered_queries=[mirrored_answer],
            )
        )

    def test_filter_already_answered_queries_removes_answered_queries(self) -> None:
        answered_query = Query(ziel_index_a=0, ziel_index_b=1, value=2.0)
        new_query = Query(ziel_index_a=0, ziel_index_b=2, value=1.5)

        filtered_queries = filter_already_answered_queries(
            queries=[answered_query, new_query],
            answered_queries=[answered_query.answer(">")],
        )

        self.assertEqual(filtered_queries, [new_query])

    def test_ratio_relevant_candidate_set_excludes_point_only_candidate(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        weight_space = build_weight_space(
            goal_count=alternatives.get_anzahl_spalten(),
            answered_queries=[],
        )

        self.assertEqual(
            compute_ratio_relevant_candidate_set(
                alternatives=alternatives,
                weight_space=weight_space,
            ),
            [0],
        )

    def test_grid_depth_query_candidates_include_ratio_sources(self) -> None:
        weight_space = build_weight_space(
            goal_count=self.alternatives.get_anzahl_spalten(),
            answered_queries=[],
        )
        query_candidate_data = compute_query_candidates_for_depth_optimized(
            alternatives=self.alternatives,
            weight_space=weight_space,
            candidates=list(range(self.alternatives.get_anzahl_zeilen())),
            remaining_depth=2,
            config=OptimizedMultistepConfig(
                grid_size=3,
                max_query_value=10.0,
                include_ratio_queries_on_grid_depths=True,
            ),
        )

        self.assertTrue(query_candidate_data.query_candidates)
        self.assertIn("ratio", set(query_candidate_data.query_sources.values()))

    def test_posterior_quantiles_create_balanced_ratio_query(self) -> None:
        queries = compute_posterior_quantile_query_candidates(
            samples=[
                [0.9, 0.1],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.1, 0.9],
            ],
            quantile_levels=(0.5,),
            min_query_value=1e-3,
            max_query_value=100.0,
        )

        self.assertEqual(len(queries), 1)
        self.assertAlmostEqual(float(queries[0].value), 5.0 / 3.0)

    def test_entropy_and_regret_scores_prefer_informative_query(self) -> None:
        alternatives = AlternativenMatrix(entries=[[1.0, 0.0], [0.0, 1.0]])
        informative = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
        uninformative = Query(ziel_index_a=0, ziel_index_b=1, value=10.0)

        scores = score_query_candidates_by_posterior(
            alternatives=alternatives,
            query_candidates=[informative, uninformative],
            samples=[
                [0.9, 0.1],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.1, 0.9],
            ],
        )

        self.assertEqual(scores[0].expected_entropy, 0.0)
        self.assertEqual(scores[0].expected_regret, 0.0)
        self.assertGreater(scores[1].expected_entropy, scores[0].expected_entropy)
        self.assertGreater(scores[1].expected_regret, scores[0].expected_regret)

    def test_session_reuses_canonical_state_analysis(self) -> None:
        direct = Query(ziel_index_a=0, ziel_index_b=1, value=2.0).answer("<")
        mirrored = Query(ziel_index_a=1, ziel_index_b=0, value=0.5).answer(">")
        with OptimizedValueFunctionSession(
            alternatives=self.alternatives,
            config=OptimizedMultistepConfig(parallelize_root=False),
        ) as session:
            with collect_optimization_profile() as profile:
                first = session.analyze_state([direct])
                second = session.analyze_state([mirrored])

        self.assertIs(first, second)
        self.assertEqual(profile.counters["state_analysis_cache_hits"], 1)

    def test_session_compute_reuses_prior_candidate_analysis(self) -> None:
        config = OptimizedMultistepConfig(
            sample_count=40,
            burn_in=10,
            thinning=1,
            random_seed=3,
            parallelize_root=False,
            use_ratio_terminal_counts=True,
            repair_zero_terminal_counts=False,
        )
        with OptimizedValueFunctionSession(
            alternatives=self.alternatives,
            config=config,
        ) as session:
            with collect_optimization_profile() as profile:
                session.analyze_state([])
                session.compute(answered_queries=[], remaining_depth=1)

        self.assertEqual(profile.counters["ratio_interval_batches"], 1)
        self.assertEqual(profile.counters["state_analysis_cache_hits"], 1)

    def test_supported_answer_probabilities_remove_boundary_equality_answer(self) -> None:
        probabilities = estimate_supported_answer_probabilities(
            answer_counts={"<": 0, "=": 70, ">": 30},
            supported_answers={"<": False, "=": False, ">": True},
            smoothing=1.0,
        )

        self.assertEqual(probabilities, {"<": 0.0, "=": 0.0, ">": 1.0})

    def test_supported_answer_probabilities_smooth_missing_supported_branch(self) -> None:
        probabilities = estimate_supported_answer_probabilities(
            answer_counts={"<": 400, "=": 0, ">": 0},
            supported_answers={"<": True, "=": False, ">": True},
            smoothing=1.0,
        )

        self.assertGreater(probabilities[">"], 0.0)
        self.assertEqual(probabilities["="], 0.0)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_supported_answers_detect_forced_equality(self) -> None:
        weight_space = build_weight_space(
            goal_count=3,
            answered_queries=[
                Query(ziel_index_a=0, ziel_index_b=1, value=2.0).answer("=")
            ],
        )

        self.assertEqual(
            compute_supported_query_answers(
                weight_space=weight_space,
                query=Query(ziel_index_a=0, ziel_index_b=1, value=2.0),
                tolerance=1e-9,
            ),
            {"<": False, "=": True, ">": False},
        )

    def test_supported_answers_treat_boundary_equality_as_zero_probability(self) -> None:
        weight_space = build_weight_space(
            goal_count=3,
            answered_queries=[
                Query(ziel_index_a=0, ziel_index_b=1, value=2.0).answer(">")
            ],
        )

        self.assertEqual(
            compute_supported_query_answers(
                weight_space=weight_space,
                query=Query(ziel_index_a=0, ziel_index_b=1, value=2.0),
                tolerance=1e-9,
            ),
            {"<": False, "=": False, ">": True},
        )

    def test_sample_witnesses_avoid_query_support_optimizations(self) -> None:
        weight_space = build_weight_space(goal_count=2, answered_queries=[])
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)

        with (
            patch.object(
                LinearConstraintSystem,
                "minimize",
                side_effect=AssertionError("unexpected lower-bound LP"),
            ),
            patch.object(
                LinearConstraintSystem,
                "maximize",
                side_effect=AssertionError("unexpected upper-bound LP"),
            ),
        ):
            supported_answers = compute_supported_query_answers_with_sample_evidence(
                weight_space=weight_space,
                query=query,
                samples=[[0.2, 0.8], [0.8, 0.2]],
                tolerance=1e-9,
            )

        self.assertEqual(
            supported_answers,
            {"<": True, "=": False, ">": True},
        )

    def test_one_sample_witness_requires_only_the_opposite_support_bound(self) -> None:
        weight_space = build_weight_space(goal_count=2, answered_queries=[])
        query = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
        original_minimize = LinearConstraintSystem.minimize
        original_maximize = LinearConstraintSystem.maximize

        with (
            patch.object(
                LinearConstraintSystem,
                "minimize",
                autospec=True,
                side_effect=original_minimize,
            ) as minimize,
            patch.object(
                LinearConstraintSystem,
                "maximize",
                autospec=True,
                side_effect=original_maximize,
            ) as maximize,
        ):
            supported_answers = compute_supported_query_answers_with_sample_evidence(
                weight_space=weight_space,
                query=query,
                samples=[[0.2, 0.8]],
                tolerance=1e-9,
            )

        self.assertEqual(minimize.call_count, 1)
        self.assertEqual(maximize.call_count, 1)
        self.assertEqual(
            supported_answers,
            {"<": True, "=": False, ">": True},
        )

    def test_sample_evidence_preserves_forced_equality_support(self) -> None:
        query = Query(ziel_index_a=0, ziel_index_b=1, value=2.0)
        weight_space = build_weight_space(
            goal_count=3,
            answered_queries=[query.answer("=")],
        )

        supported_answers = compute_supported_query_answers_with_sample_evidence(
            weight_space=weight_space,
            query=query,
            samples=[[2.0 / 3.0, 1.0 / 3.0, 0.0]],
            tolerance=1e-9,
        )

        self.assertEqual(
            supported_answers,
            compute_supported_query_answers(
                weight_space=weight_space,
                query=query,
                tolerance=1e-9,
            ),
        )

    def test_ratio_candidate_analysis_reuses_one_interval_batch(self) -> None:
        config = OptimizedMultistepConfig(
            sample_count=30,
            burn_in=5,
            thinning=1,
            random_seed=13,
            grid_size=3,
            parallelize_root=False,
            candidate_count_mode="ratio_relevant",
            use_ratio_terminal_counts=True,
            repair_zero_terminal_counts=False,
        )

        with collect_optimization_profile() as profile:
            result = compute_value_function_optimized(
                alternatives=self.alternatives,
                answered_queries=[],
                remaining_depth=1,
                config=config,
            )

        self.assertTrue(result.is_feasible)
        self.assertTrue(result.query_evaluations)
        self.assertEqual(profile.counters["state_calls"], 1)
        self.assertEqual(profile.counters["ratio_interval_batches"], 1)

    def test_session_reuses_one_executor_across_calls(self) -> None:
        alternatives = AlternativenMatrix(entries=[[1.0, 0.0], [0.0, 1.0]])
        created_executors: list[InlineExecutor] = []

        class InlineExecutor:
            def __init__(self, max_workers: int) -> None:
                self.max_workers = max_workers
                self.map_calls = 0
                self.shutdown_calls = 0
                created_executors.append(self)

            def map(self, function: object, payloads: object) -> object:
                self.map_calls += 1
                return map(function, payloads)  # type: ignore[arg-type]

            def shutdown(self, wait: bool = True) -> None:
                self.shutdown_calls += 1

        config = OptimizedMultistepConfig(
            sample_count=10,
            burn_in=2,
            thinning=1,
            random_seed=3,
            grid_size=2,
            parallelize_root=True,
            max_workers=2,
            use_ratio_terminal_counts=True,
        )
        with patch(
            "multistep.optimized.value_function.ProcessPoolExecutor",
            InlineExecutor,
        ):
            with OptimizedValueFunctionSession(
                alternatives=alternatives,
                config=config,
            ) as session:
                with collect_optimization_profile() as profile:
                    first = session.compute(answered_queries=[], remaining_depth=1)
                    second = session.compute(answered_queries=[], remaining_depth=1)

        self.assertEqual(first, second)
        self.assertEqual(len(created_executors), 1)
        self.assertEqual(created_executors[0].map_calls, 1)
        self.assertEqual(created_executors[0].shutdown_calls, 1)
        self.assertEqual(profile.counters["session_cache_hits"], 1)
        with self.assertRaisesRegex(RuntimeError, "session is closed"):
            session.compute(answered_queries=[], remaining_depth=1)

    def test_query_shortlist_keeps_most_balanced_sample_partition(self) -> None:
        queries = [
            Query(ziel_index_a=0, ziel_index_b=1, value=0.1),
            Query(ziel_index_a=0, ziel_index_b=1, value=1.0),
            Query(ziel_index_a=0, ziel_index_b=1, value=10.0),
        ]
        samples = [
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.8, 0.2],
        ]

        shortlisted = shortlist_query_candidates_by_sample_balance(
            query_candidates=queries,
            samples=samples,
            equality_tol=0.0,
            limit=1,
        )

        self.assertEqual(shortlisted, [queries[1]])

    def test_adaptive_depth_matches_explicit_depth_one_evaluation(self) -> None:
        common_config = dict(
            sample_count=30,
            burn_in=5,
            thinning=1,
            random_seed=17,
            grid_size=3,
            parallelize_root=False,
            candidate_count_mode="ratio_relevant",
            use_ratio_terminal_counts=True,
        )
        depth_one_result = compute_value_function_optimized(
            alternatives=self.alternatives,
            answered_queries=[],
            remaining_depth=1,
            config=OptimizedMultistepConfig(**common_config),
        )
        adaptive_result = compute_value_function_optimized(
            alternatives=self.alternatives,
            answered_queries=[],
            remaining_depth=2,
            config=OptimizedMultistepConfig(
                **common_config,
                adaptive_depth_candidate_threshold=1,
            ),
        )

        self.assertEqual(adaptive_result.best_query, depth_one_result.best_query)
        self.assertAlmostEqual(adaptive_result.value, depth_one_result.value)
        self.assertEqual(
            adaptive_result.query_evaluations,
            depth_one_result.query_evaluations,
        )

    def test_terminal_ratio_counts_match_exact_child_candidate_counts(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [
                    0.13552100374068055,
                    0.7918000796455894,
                    0.6753462694980112,
                    0.4216035573870036,
                    0.02556709896246312,
                ],
                [
                    0.16798147095798632,
                    0.7494608262320578,
                    0.08400476777677468,
                    0.3126499249880641,
                    0.25521243827660445,
                ],
                [
                    0.7461008968986407,
                    0.3590598291763175,
                    0.08718824098875189,
                    0.3704421018161166,
                    0.3268298767186818,
                ],
                [
                    0.7170213673193974,
                    0.32012222019868797,
                    0.6936156490920399,
                    0.538594453433759,
                    0.8877328287244716,
                ],
                [
                    0.7333651729435877,
                    0.4079187425363865,
                    0.4849095181984443,
                    0.4713857598653367,
                    0.871702397518525,
                ],
                [
                    0.1380361717125459,
                    0.4242300124285143,
                    0.5349120574812903,
                    0.43622283223869607,
                    0.5980284679987804,
                ],
                [
                    0.49884309541984917,
                    0.4136957426619674,
                    0.6868129726159958,
                    0.329322465708467,
                    0.6069010136690235,
                ],
                [
                    0.7285583473338744,
                    0.13064663579388291,
                    0.326452101504932,
                    0.9447045555986052,
                    0.9660212875626245,
                ],
                [
                    0.9923887631356604,
                    0.042525833888270914,
                    0.8265087689118491,
                    0.9352758069942098,
                    0.9019444400552439,
                ],
                [
                    0.7149413034806142,
                    0.6756193428878896,
                    0.7196066164883864,
                    0.5748858181470116,
                    0.7884200798433164,
                ],
            ]
        )
        answered_queries = [
            Query(ziel_index_a=1, ziel_index_b=3, value=1.3183431633996698).answer("<"),
            Query(ziel_index_a=0, ziel_index_b=3, value=1.7103128591798624).answer("<"),
            Query(ziel_index_a=4, ziel_index_b=1, value=2.668450154891162).answer(">"),
            Query(ziel_index_a=3, ziel_index_b=4, value=0.5994842503189409).answer(">"),
            Query(ziel_index_a=1, ziel_index_b=4, value=0.001).answer(">"),
        ]
        terminal_query = Query(
            ziel_index_a=3,
            ziel_index_b=2,
            value=2.9164594015565445,
        )
        weight_space = build_weight_space(
            goal_count=alternatives.get_anzahl_spalten(),
            answered_queries=answered_queries,
        )
        candidates = compute_ratio_relevant_candidate_set(
            alternatives=alternatives,
            weight_space=weight_space,
        )
        ratio_intervals = compute_all_ratio_intervals(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=candidates,
        )
        ratio_intervals_by_goal_pair = {
            (intervals.goal_index_a, intervals.goal_index_b): intervals
            for intervals in ratio_intervals
        }
        supported_answers = compute_supported_query_answers(
            weight_space=weight_space,
            query=terminal_query,
            tolerance=1e-9,
        )
        config = OptimizedMultistepConfig(candidate_count_mode="ratio_relevant")

        for answer, is_supported in supported_answers.items():
            if not is_supported:
                continue

            shortcut_count = compute_terminal_candidate_count_from_ratio_intervals(
                query=terminal_query,
                answer=answer,
                ratio_intervals_by_goal_pair=ratio_intervals_by_goal_pair,
                tolerance=config.ratio_terminal_tolerance,
            )
            exact_count, is_feasible = compute_terminal_candidate_count_fallback(
                alternatives=alternatives,
                answered_queries=answered_queries,
                query=terminal_query,
                answer=answer,
                candidate_subset=candidates,
                config=config,
            )

            self.assertTrue(is_feasible)
            self.assertEqual(shortcut_count, exact_count, answer)


if __name__ == "__main__":
    unittest.main()
