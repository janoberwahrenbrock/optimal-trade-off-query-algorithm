from __future__ import annotations

import unittest

from multistep.scripts.benchmark_depth3_e2_guard import (
    select_depth_three_query_with_depth_two_guard,
)
from multistep.src.models import Query
from multistep.src.value_function import QueryEvaluation, ValueFunctionResult


class _FakeSession:
    def __init__(
        self,
        depth_three: ValueFunctionResult,
        depth_two: ValueFunctionResult,
    ) -> None:
        self.depth_three = depth_three
        self.depth_two = depth_two

    def compute(self, answered_queries, remaining_depth):
        del answered_queries
        return self.depth_three if remaining_depth == 3 else self.depth_two


def _result(
    depth: int,
    evaluations: tuple[QueryEvaluation, ...],
) -> ValueFunctionResult:
    best = min(evaluations, key=lambda evaluation: evaluation.expected_value)
    return ValueFunctionResult(
        remaining_depth=depth,
        value=best.expected_value,
        best_query=best.query,
        candidate_count=3,
        query_evaluations=evaluations,
        is_feasible=True,
    )


class DepthThreeDepthTwoGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first = Query(ziel_index_a=0, ziel_index_b=1, value=1.0)
        self.second = Query(ziel_index_a=0, ziel_index_b=2, value=1.0)
        self.third = Query(ziel_index_a=1, ziel_index_b=2, value=1.0)
        self.session = _FakeSession(
            depth_three=_result(
                3,
                (
                    QueryEvaluation(self.first, 1.2, ()),
                    QueryEvaluation(self.second, 1.1, ()),
                    QueryEvaluation(self.third, 1.0, ()),
                ),
            ),
            depth_two=_result(
                2,
                (
                    QueryEvaluation(self.first, 1.0, ()),
                    QueryEvaluation(self.second, 1.04, ()),
                    QueryEvaluation(self.third, 1.2, ()),
                ),
            ),
        )

    def test_selects_best_e_three_inside_depth_two_band(self) -> None:
        selection = select_depth_three_query_with_depth_two_guard(
            session=self.session,
            answered_queries=[],
            delta=0.05,
        )

        self.assertEqual(selection.query, self.second)
        self.assertAlmostEqual(selection.expected_candidates_depth_three, 1.1)
        self.assertAlmostEqual(selection.expected_candidates_depth_two, 1.04)
        self.assertAlmostEqual(selection.best_expected_candidates_depth_two, 1.0)
        self.assertEqual(selection.admissible_query_count, 2)

    def test_zero_delta_reproduces_best_depth_two_query(self) -> None:
        selection = select_depth_three_query_with_depth_two_guard(
            session=self.session,
            answered_queries=[],
            delta=0.0,
        )

        self.assertEqual(selection.query, self.first)

    def test_rejects_negative_delta(self) -> None:
        with self.assertRaisesRegex(ValueError, "delta"):
            select_depth_three_query_with_depth_two_guard(
                session=self.session,
                answered_queries=[],
                delta=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
