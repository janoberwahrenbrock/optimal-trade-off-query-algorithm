from __future__ import annotations

import unittest

from multistep.optimized.value_function import (
    OptimizedMultistepConfig,
    compute_candidate_set_for_subset,
    compute_query_candidates_for_depth_optimized,
    compute_ratio_relevant_candidate_set,
    compute_supported_query_answers,
    compute_terminal_candidate_count_fallback,
    compute_terminal_candidate_count_from_ratio_intervals,
    compute_value_function_optimized,
    estimate_supported_answer_probabilities,
    filter_already_answered_queries,
    is_query_already_answered,
)
from multistep.src.ratio_intervals import compute_all_ratio_intervals
from multistep.src.models import Query
from multistep.src.candidates import compute_candidate_set
from multistep.src.models import AlternativenMatrix
from multistep.src.value_function import MultistepConfig, compute_value_function
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
