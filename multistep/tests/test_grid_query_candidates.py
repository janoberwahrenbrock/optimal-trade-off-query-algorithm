from __future__ import annotations

import unittest

from multistep.src.grid_query_candidates import (
    DEFAULT_MAX_QUERY_VALUE,
    build_grid_query_values,
    compute_grid_query_candidates,
    deduplicate_mirrored_query_candidates,
)
from multistep.src.models import Query
from multistep.src.models.linear_optimization_result import LinearOptimizationResult
from multistep.src.weight_space import build_weight_space


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


class GridQueryCandidatesTests(unittest.TestCase):
    def test_default_max_query_value_is_100(self) -> None:
        self.assertEqual(DEFAULT_MAX_QUERY_VALUE, 100.0)

    def test_build_grid_query_values_linear(self) -> None:
        values = build_grid_query_values(
            lower=optimal_min(0.0),
            upper=optimal_max(1.0),
            grid_size=3,
            min_query_value=0.1,
            max_query_value=10.0,
            spacing="linear",
        )

        self.assertEqual(values, [0.1, 0.55, 1.0])

    def test_build_grid_query_values_log(self) -> None:
        values = build_grid_query_values(
            lower=optimal_min(0.1),
            upper=optimal_max(10.0),
            grid_size=3,
            min_query_value=0.1,
            max_query_value=10.0,
            spacing="log",
        )

        self.assertEqual(len(values), 3)
        self.assertAlmostEqual(values[0], 0.1)
        self.assertAlmostEqual(values[1], 1.0)
        self.assertAlmostEqual(values[2], 10.0)

    def test_build_grid_query_values_uses_max_for_unbounded_upper(self) -> None:
        values = build_grid_query_values(
            lower=optimal_min(0.0),
            upper=unbounded_max(),
            grid_size=2,
            min_query_value=0.5,
            max_query_value=10.0,
            spacing="linear",
        )

        self.assertEqual(values, [0.5, 10.0])

    def test_build_grid_query_values_returns_empty_for_infeasible_lower(self) -> None:
        lower = LinearOptimizationResult(status="infeasible", objective_sense="min")

        self.assertEqual(
            build_grid_query_values(
                lower=lower,
                upper=optimal_max(1.0),
                grid_size=3,
            ),
            [],
        )

    def test_build_grid_query_values_rejects_unbounded_lower(self) -> None:
        with self.assertRaisesRegex(ValueError, "lower ratio bound"):
            build_grid_query_values(
                lower=unbounded_min(),
                upper=optimal_max(1.0),
                grid_size=3,
            )

    def test_build_grid_query_values_rejects_invalid_spacing(self) -> None:
        with self.assertRaisesRegex(ValueError, "spacing"):
            build_grid_query_values(
                lower=optimal_min(0.0),
                upper=optimal_max(1.0),
                grid_size=3,
                spacing="quadratic",
            )

    def test_deduplicate_mirrored_query_candidates_removes_equivalent_query(
        self,
    ) -> None:
        queries = deduplicate_mirrored_query_candidates(
            queries=[
                Query(ziel_index_a=0, ziel_index_b=1, value=2.0),
                Query(ziel_index_a=1, ziel_index_b=0, value=0.5),
            ]
        )

        self.assertEqual(len(queries), 1)
        self.assertEqual(
            (queries[0].ziel_index_a, queries[0].ziel_index_b, queries[0].value),
            (0, 1, 2.0),
        )

    def test_compute_grid_query_candidates_uses_all_ordered_pairs_then_deduplicates(
        self,
    ) -> None:
        queries = compute_grid_query_candidates(
            weight_space=build_weight_space(goal_count=2, answered_queries=[]),
            grid_size=3,
            min_query_value=0.1,
            max_query_value=10.0,
            spacing="log",
        )

        self.assertEqual(
            [(query.ziel_index_a, query.ziel_index_b, query.value) for query in queries],
            [
                (0, 1, 0.1),
                (0, 1, 1.0),
                (0, 1, 10.0),
            ],
        )

    def test_compute_grid_query_candidates_rejects_weight_space_with_one_goal(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "at least two goals"):
            compute_grid_query_candidates(
                weight_space=build_weight_space(goal_count=1, answered_queries=[]),
            )


if __name__ == "__main__":
    unittest.main()
