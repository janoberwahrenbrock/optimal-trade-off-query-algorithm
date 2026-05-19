from __future__ import annotations

import unittest

from multistep.src.models.linear_optimization_result import LinearOptimizationResult
from multistep.src.onestep_query_candidates import compute_onestep_query_candidates
from multistep.src.ratio_intervals import GoalPairRatioIntervals, RatioInterval


def optimal_min(value: float) -> LinearOptimizationResult:
    return LinearOptimizationResult(
        status="optimal",
        objective_sense="min",
        optimal_value=value,
    )


def optimal_max(value: float) -> LinearOptimizationResult:
    return LinearOptimizationResult(
        status="optimal",
        objective_sense="max",
        optimal_value=value,
    )


def unbounded_max() -> LinearOptimizationResult:
    return LinearOptimizationResult(
        status="unbounded",
        objective_sense="max",
    )


def unbounded_min() -> LinearOptimizationResult:
    return LinearOptimizationResult(
        status="unbounded",
        objective_sense="min",
    )


class OnestepQueryCandidatesTests(unittest.TestCase):
    def test_compute_onestep_query_candidates_builds_left_and_right_queries(
        self,
    ) -> None:
        goal_pair_intervals = GoalPairRatioIntervals(
            goal_index_a=0,
            goal_index_b=1,
            intervals_by_candidate={
                0: RatioInterval(
                    lower=optimal_min(0.5),
                    upper=optimal_max(2.0),
                ),
            },
        )

        queries = compute_onestep_query_candidates(
            goal_pair_ratio_intervals=[goal_pair_intervals],
            epsilon=0.1,
        )

        self.assertEqual(
            [(query.ziel_index_a, query.ziel_index_b, query.value) for query in queries],
            [
                (0, 1, 0.4),
                (0, 1, 2.1),
            ],
        )

    def test_compute_onestep_query_candidates_skips_lower_at_epsilon_boundary(
        self,
    ) -> None:
        goal_pair_intervals = GoalPairRatioIntervals(
            goal_index_a=0,
            goal_index_b=1,
            intervals_by_candidate={
                0: RatioInterval(
                    lower=optimal_min(0.1),
                    upper=optimal_max(2.0),
                ),
            },
        )

        queries = compute_onestep_query_candidates(
            goal_pair_ratio_intervals=[goal_pair_intervals],
            epsilon=0.1,
        )

        self.assertEqual(
            [(query.ziel_index_a, query.ziel_index_b, query.value) for query in queries],
            [(0, 1, 2.1)],
        )

    def test_compute_onestep_query_candidates_skips_unbounded_upper(self) -> None:
        goal_pair_intervals = GoalPairRatioIntervals(
            goal_index_a=0,
            goal_index_b=1,
            intervals_by_candidate={
                0: RatioInterval(
                    lower=optimal_min(0.5),
                    upper=unbounded_max(),
                ),
            },
        )

        queries = compute_onestep_query_candidates(
            goal_pair_ratio_intervals=[goal_pair_intervals],
            epsilon=0.1,
        )

        self.assertEqual(
            [(query.ziel_index_a, query.ziel_index_b, query.value) for query in queries],
            [(0, 1, 0.4)],
        )

    def test_compute_onestep_query_candidates_deduplicates_identical_breakpoints(
        self,
    ) -> None:
        goal_pair_intervals = GoalPairRatioIntervals(
            goal_index_a=0,
            goal_index_b=1,
            intervals_by_candidate={
                0: RatioInterval(
                    lower=optimal_min(0.5),
                    upper=unbounded_max(),
                ),
                1: RatioInterval(
                    lower=optimal_min(0.5),
                    upper=unbounded_max(),
                ),
            },
        )

        queries = compute_onestep_query_candidates(
            goal_pair_ratio_intervals=[goal_pair_intervals],
            epsilon=0.1,
        )

        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].value, 0.4)

    def test_compute_onestep_query_candidates_deduplicates_mirrored_candidates(
        self,
    ) -> None:
        first_goal_pair_intervals = GoalPairRatioIntervals(
            goal_index_a=0,
            goal_index_b=1,
            intervals_by_candidate={
                0: RatioInterval(
                    lower=optimal_min(2.0),
                    upper=unbounded_max(),
                ),
            },
        )
        mirrored_goal_pair_intervals = GoalPairRatioIntervals(
            goal_index_a=1,
            goal_index_b=0,
            intervals_by_candidate={
                0: RatioInterval(
                    lower=optimal_min(0.0),
                    upper=optimal_max(0.5),
                ),
            },
        )

        queries = compute_onestep_query_candidates(
            goal_pair_ratio_intervals=[
                first_goal_pair_intervals,
                mirrored_goal_pair_intervals,
            ],
            epsilon=0.1,
        )

        self.assertEqual(len(queries), 1)
        self.assertEqual(
            (queries[0].ziel_index_a, queries[0].ziel_index_b),
            (0, 1),
        )

    def test_compute_onestep_query_candidates_rejects_unbounded_lower(self) -> None:
        goal_pair_intervals = GoalPairRatioIntervals(
            goal_index_a=0,
            goal_index_b=1,
            intervals_by_candidate={
                0: RatioInterval(
                    lower=unbounded_min(),
                    upper=unbounded_max(),
                ),
            },
        )

        with self.assertRaisesRegex(ValueError, "lower ratio bound"):
            compute_onestep_query_candidates([goal_pair_intervals])

    def test_compute_onestep_query_candidates_rejects_nonpositive_epsilon(self) -> None:
        with self.assertRaisesRegex(ValueError, "epsilon must be positive"):
            compute_onestep_query_candidates([], epsilon=0.0)

    def test_compute_onestep_query_candidates_rejects_negative_dedup_tolerance(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "abs_tol must not be negative"):
            compute_onestep_query_candidates([], dedup_abs_tol=-1.0)


if __name__ == "__main__":
    unittest.main()
