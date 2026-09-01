from __future__ import annotations

import unittest
from unittest.mock import patch

from multistep.src.models import AlternativenMatrix, AnsweredQuery
import multistep.src.ratio_intervals as ratio_intervals_module
from multistep.src.ratio_intervals import (
    compute_all_ratio_intervals,
    compute_ratio_bounds_for_weight_space,
    compute_ratio_interval_for_candidate,
    compute_ratio_intervals_for_pair,
    get_canonical_goal_pairs,
    get_ordered_goal_pairs,
    invert_goal_pair_ratio_intervals,
)
from multistep.src.weight_space import build_weight_space


class RatioIntervalsTests(unittest.TestCase):
    def test_get_canonical_goal_pairs(self) -> None:
        self.assertEqual(
            get_canonical_goal_pairs(3),
            [(0, 1), (0, 2), (1, 2)],
        )

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

    def test_geometric_engine_matches_lp_engine(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [0.9, 0.2, 0.1],
                [0.2, 0.9, 0.1],
                [0.1, 0.2, 0.9],
                [0.6, 0.6, 0.6],
            ]
        )
        weight_space = build_weight_space(goal_count=3, answered_queries=[])

        geometric = compute_all_ratio_intervals(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=[0, 1, 2, 3],
            engine="geometry",
        )
        exact_lp = compute_all_ratio_intervals(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=[0, 1, 2, 3],
            engine="lp",
        )

        for geometric_pair, lp_pair in zip(geometric, exact_lp):
            for candidate_index in [0, 1, 2, 3]:
                actual = geometric_pair.intervals_by_candidate[candidate_index]
                expected = lp_pair.intervals_by_candidate[candidate_index]
                self.assertEqual(actual.lower.status, expected.lower.status)
                self.assertEqual(actual.upper.status, expected.upper.status)
                if actual.lower.optimal_value is not None:
                    self.assertAlmostEqual(
                        actual.lower.optimal_value,
                        expected.lower.optimal_value,
                    )
                if actual.upper.optimal_value is not None:
                    self.assertAlmostEqual(
                        actual.upper.optimal_value,
                        expected.upper.optimal_value,
                    )

    def test_inverted_intervals_match_direct_reverse_solve(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        weight_space = build_weight_space(goal_count=2, answered_queries=[])
        direct = compute_ratio_intervals_for_pair(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=[0, 1],
            goal_index_a=0,
            goal_index_b=1,
        )
        inverted = invert_goal_pair_ratio_intervals(direct)
        reverse = compute_ratio_intervals_for_pair(
            alternatives=alternatives,
            weight_space=weight_space,
            candidates=[0, 1],
            goal_index_a=1,
            goal_index_b=0,
        )

        self.assertIsNotNone(inverted)
        assert inverted is not None
        for candidate_index in [0, 1]:
            actual = inverted.intervals_by_candidate[candidate_index]
            expected = reverse.intervals_by_candidate[candidate_index]
            self.assertEqual(actual.lower.status, expected.lower.status)
            self.assertEqual(actual.upper.status, expected.upper.status)
            if actual.lower.optimal_value is not None:
                self.assertAlmostEqual(
                    actual.lower.optimal_value,
                    expected.lower.optimal_value,
                )
            if actual.upper.optimal_value is not None:
                self.assertAlmostEqual(
                    actual.upper.optimal_value,
                    expected.upper.optimal_value,
                )

    def test_compute_all_ratio_intervals_solves_only_canonical_pairs(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        original = ratio_intervals_module.compute_ratio_intervals_for_pair

        with patch.object(
            ratio_intervals_module,
            "compute_ratio_intervals_for_pair",
            autospec=True,
            side_effect=original,
        ) as compute_pair:
            compute_all_ratio_intervals(
                alternatives=alternatives,
                weight_space=build_weight_space(goal_count=2, answered_queries=[]),
                candidates=[0, 1],
                engine="lp",
            )

        self.assertEqual(compute_pair.call_count, 1)

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
