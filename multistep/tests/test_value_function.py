from __future__ import annotations

import unittest

from multistep.src.models import AlternativenMatrix, Query
from multistep.src.sampling import sample_points_from_constraint_system
from multistep.src.value_function import (
    MultistepConfig,
    compute_query_candidates_for_depth,
    compute_value_function,
    evaluate_query_candidate,
)
from multistep.src.weight_space import build_weight_space


class ValueFunctionTests(unittest.TestCase):
    def test_config_rejects_invalid_sample_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_count must be positive"):
            MultistepConfig(sample_count=0)

    def test_config_rejects_invalid_grid_spacing(self) -> None:
        with self.assertRaisesRegex(ValueError, "grid_spacing"):
            MultistepConfig(grid_spacing="quadratic")

    def test_compute_value_function_depth_zero_returns_candidate_count(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        result = compute_value_function(
            alternatives=alternatives,
            answered_queries=[],
            remaining_depth=0,
            config=MultistepConfig(sample_count=20, burn_in=0, thinning=1),
        )

        self.assertEqual(result.value, 2.0)
        self.assertEqual(result.candidate_count, 2)
        self.assertIsNone(result.best_query)
        self.assertEqual(result.query_evaluations, ())

    def test_compute_query_candidates_for_depth_one_uses_terminal_candidates(
        self,
    ) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        weight_space = build_weight_space(goal_count=2, answered_queries=[])

        queries = compute_query_candidates_for_depth(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=[0, 1],
            remaining_depth=1,
            config=MultistepConfig(query_epsilon=0.1),
        )

        self.assertTrue(queries)

    def test_compute_query_candidates_for_depth_two_uses_grid_candidates(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        weight_space = build_weight_space(goal_count=2, answered_queries=[])

        queries = compute_query_candidates_for_depth(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=[0, 1],
            remaining_depth=2,
            config=MultistepConfig(
                grid_size=3,
                min_query_value=0.1,
                max_query_value=10.0,
                grid_spacing="log",
            ),
        )

        self.assertEqual(
            [(query.ziel_index_a, query.ziel_index_b, query.value) for query in queries],
            [
                (0, 1, 0.1),
                (0, 1, 1.0),
                (0, 1, 10.0),
            ],
        )

    def test_compute_value_function_depth_one_returns_best_query(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        result = compute_value_function(
            alternatives=alternatives,
            answered_queries=[],
            remaining_depth=1,
            config=MultistepConfig(
                sample_count=20,
                burn_in=0,
                thinning=1,
                random_seed=1,
                query_epsilon=0.1,
            ),
        )

        self.assertIsNotNone(result.best_query)
        self.assertTrue(result.query_evaluations)
        self.assertLessEqual(result.value, result.candidate_count)

    def test_compute_value_function_depth_two_runs_with_grid_first_step(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        result = compute_value_function(
            alternatives=alternatives,
            answered_queries=[],
            remaining_depth=2,
            config=MultistepConfig(
                sample_count=12,
                burn_in=0,
                thinning=1,
                random_seed=2,
                grid_size=3,
                min_query_value=0.1,
                max_query_value=10.0,
                grid_spacing="log",
                query_epsilon=0.1,
            ),
        )

        self.assertIsNotNone(result.best_query)
        self.assertTrue(result.query_evaluations)
        self.assertLessEqual(result.value, result.candidate_count)

    def test_evaluate_query_candidate_marks_infeasible_child_branch(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        answered_queries = [Query(ziel_index_a=0, ziel_index_b=1, value=2.0).answer(">")]
        weight_space = build_weight_space(
            goal_count=2,
            answered_queries=answered_queries,
        )
        samples = sample_points_from_constraint_system(
            system=weight_space,
            num_samples=10,
            burn_in=0,
            thinning=1,
            seed=3,
        )

        evaluation = evaluate_query_candidate(
            alternatives=alternatives,
            answered_queries=answered_queries,
            query=Query(ziel_index_a=0, ziel_index_b=1, value=1.0),
            samples=samples,
            remaining_depth=1,
            config=MultistepConfig(sample_count=10, burn_in=0, thinning=1),
        )
        less_branch = next(
            branch for branch in evaluation.branches if branch.answer == "<"
        )

        self.assertFalse(less_branch.is_child_feasible)
        self.assertEqual(less_branch.child_value, 0.0)
        self.assertEqual(less_branch.child_candidate_count, 0)

    def test_compute_value_function_rejects_negative_depth(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        with self.assertRaisesRegex(ValueError, "remaining_depth"):
            compute_value_function(
                alternatives=alternatives,
                answered_queries=[],
                remaining_depth=-1,
            )


if __name__ == "__main__":
    unittest.main()
