from __future__ import annotations

import unittest

from multistep.src.models import AnsweredQuery
from multistep.src.weight_space import (
    build_answered_query_constraint,
    build_weight_space,
)


class WeightSpaceTests(unittest.TestCase):
    def test_build_weight_space_adds_nonnegativity_and_normalization(self) -> None:
        weight_space = build_weight_space(goal_count=3, answered_queries=[])

        self.assertEqual(weight_space.variable_count, 3)
        self.assertEqual(
            weight_space.inequalities_left_side,
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
        )
        self.assertEqual(weight_space.inequalities_right_side, [0.0, 0.0, 0.0])
        self.assertEqual(weight_space.equalities_left_side, [[1.0, 1.0, 1.0]])
        self.assertEqual(weight_space.equalities_right_side, [1.0])

    def test_build_weight_space_adds_answered_query_constraints(self) -> None:
        answered_queries = [
            AnsweredQuery(ziel_index_a=0, ziel_index_b=1, value=2.0, operator="<"),
            AnsweredQuery(ziel_index_a=1, ziel_index_b=2, value=3.0, operator=">"),
            AnsweredQuery(ziel_index_a=0, ziel_index_b=2, value=4.0, operator="="),
        ]

        weight_space = build_weight_space(
            goal_count=3,
            answered_queries=answered_queries,
        )

        self.assertEqual(
            weight_space.inequalities_left_side,
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.5, -1.0, 0.0],
                [0.0, -1.0 / 3.0, 1.0],
            ],
        )
        self.assertEqual(
            weight_space.equalities_left_side,
            [
                [1.0, 1.0, 1.0],
                [0.25, 0.0, -1.0],
            ],
        )

    def test_build_weight_space_rejects_nonpositive_goal_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "goal_count must be positive"):
            build_weight_space(goal_count=0, answered_queries=[])

    def test_answered_query_goal_indices_must_be_in_range(self) -> None:
        answered_query = AnsweredQuery(
            ziel_index_a=0,
            ziel_index_b=2,
            value=1.0,
            operator="<",
        )

        with self.assertRaisesRegex(IndexError, "ziel_index_b is out of range"):
            build_weight_space(goal_count=2, answered_queries=[answered_query])

    def test_answered_query_constraint_for_less_operator(self) -> None:
        answered_query = AnsweredQuery(
            ziel_index_a=0,
            ziel_index_b=1,
            value=2.0,
            operator="<",
        )

        self.assertEqual(
            build_answered_query_constraint(answered_query, goal_count=2),
            ([0.5, -1.0], 0.0, False),
        )

    def test_answered_query_constraint_for_greater_operator(self) -> None:
        answered_query = AnsweredQuery(
            ziel_index_a=0,
            ziel_index_b=1,
            value=2.0,
            operator=">",
        )

        self.assertEqual(
            build_answered_query_constraint(answered_query, goal_count=2),
            ([-0.5, 1.0], 0.0, False),
        )

    def test_answered_query_constraint_for_equal_operator(self) -> None:
        answered_query = AnsweredQuery(
            ziel_index_a=0,
            ziel_index_b=1,
            value=2.0,
            operator="=",
        )

        self.assertEqual(
            build_answered_query_constraint(answered_query, goal_count=2),
            ([0.5, -1.0], 0.0, True),
        )


if __name__ == "__main__":
    unittest.main()
