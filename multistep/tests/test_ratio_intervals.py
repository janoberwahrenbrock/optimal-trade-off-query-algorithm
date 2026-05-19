from __future__ import annotations

import unittest

from multistep.src.models import AlternativenMatrix, AnsweredQuery
from multistep.src.ratio_intervals import (
    compute_all_ratio_intervals,
    compute_ratio_bounds_for_weight_space,
    compute_ratio_interval_for_candidate,
    compute_ratio_intervals_for_pair,
    get_ordered_goal_pairs,
)
from multistep.src.weight_space import build_weight_space


class RatioIntervalsTests(unittest.TestCase):
    def test_get_ordered_goal_pairs(self) -> None:
        self.assertEqual(
            get_ordered_goal_pairs(3),
            [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)],
        )

    def test_compute_ratio_bounds_for_weight_space_without_answers(self) -> None:
        interval = compute_ratio_bounds_for_weight_space(
            weight_space=build_weight_space(goal_count=2, answered_queries=[]),
            goal_index_a=0,
            goal_index_b=1,
        )

        self.assertEqual(interval.lower.status, "optimal")
        self.assertEqual(interval.lower.optimal_value, 0.0)
        self.assertEqual(interval.upper.status, "unbounded")

    def test_compute_ratio_bounds_for_weight_space_with_upper_answer(self) -> None:
        answered_queries = [
            AnsweredQuery(
                ziel_index_a=0,
                ziel_index_b=1,
                value=2.0,
                operator="<",
            )
        ]

        interval = compute_ratio_bounds_for_weight_space(
            weight_space=build_weight_space(
                goal_count=2,
                answered_queries=answered_queries,
            ),
            goal_index_a=0,
            goal_index_b=1,
        )

        self.assertEqual(interval.lower.status, "optimal")
        self.assertIsNotNone(interval.lower.optimal_value)
        self.assertAlmostEqual(interval.lower.optimal_value, 0.0)
        self.assertEqual(interval.upper.status, "optimal")
        self.assertIsNotNone(interval.upper.optimal_value)
        self.assertAlmostEqual(interval.upper.optimal_value, 2.0)

    def test_compute_ratio_interval_for_candidate(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        first_interval = compute_ratio_interval_for_candidate(
            alternatives=alternatives,
            weight_space=build_weight_space(goal_count=2, answered_queries=[]),
            alternative_index=0,
            goal_index_a=0,
            goal_index_b=1,
        )
        second_interval = compute_ratio_interval_for_candidate(
            alternatives=alternatives,
            weight_space=build_weight_space(goal_count=2, answered_queries=[]),
            alternative_index=1,
            goal_index_a=0,
            goal_index_b=1,
        )

        self.assertEqual(first_interval.lower.status, "optimal")
        self.assertIsNotNone(first_interval.lower.optimal_value)
        self.assertAlmostEqual(first_interval.lower.optimal_value, 1.0)
        self.assertEqual(first_interval.upper.status, "unbounded")

        self.assertEqual(second_interval.lower.status, "optimal")
        self.assertIsNotNone(second_interval.lower.optimal_value)
        self.assertAlmostEqual(second_interval.lower.optimal_value, 0.0)
        self.assertEqual(second_interval.upper.status, "optimal")
        self.assertIsNotNone(second_interval.upper.optimal_value)
        self.assertAlmostEqual(second_interval.upper.optimal_value, 1.0)

    def test_compute_ratio_intervals_for_pair_reuses_goal_pair(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        goal_pair_intervals = compute_ratio_intervals_for_pair(
            alternatives=alternatives,
            weight_space=build_weight_space(goal_count=2, answered_queries=[]),
            candidates=[0, 1],
            goal_index_a=0,
            goal_index_b=1,
        )

        self.assertEqual(goal_pair_intervals.goal_index_a, 0)
        self.assertEqual(goal_pair_intervals.goal_index_b, 1)
        self.assertEqual(set(goal_pair_intervals.intervals_by_candidate), {0, 1})

    def test_compute_all_ratio_intervals_returns_every_ordered_pair(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        all_intervals = compute_all_ratio_intervals(
            alternatives=alternatives,
            weight_space=build_weight_space(goal_count=2, answered_queries=[]),
            candidates=[0, 1],
        )

        self.assertEqual(
            [(item.goal_index_a, item.goal_index_b) for item in all_intervals],
            [(0, 1), (1, 0)],
        )

    def test_compute_ratio_interval_rejects_invalid_candidate_index(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        with self.assertRaisesRegex(IndexError, "alternative_index is out of range"):
            compute_ratio_interval_for_candidate(
                alternatives=alternatives,
                weight_space=build_weight_space(goal_count=2, answered_queries=[]),
                alternative_index=2,
                goal_index_a=0,
                goal_index_b=1,
            )


if __name__ == "__main__":
    unittest.main()
