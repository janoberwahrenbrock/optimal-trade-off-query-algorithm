from __future__ import annotations

import unittest

from multistep.src.models import AlternativenMatrix
from multistep.src.optimality_region import build_optimality_region
from multistep.src.weight_space import build_weight_space


class OptimalityRegionTests(unittest.TestCase):
    def test_build_optimality_region_adds_weight_space_and_optimality_constraints(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )
        weight_space = build_weight_space(goal_count=2, answered_queries=[])

        region = build_optimality_region(
            alternatives=alternatives,
            weight_space=weight_space,
            alternative_index=0,
        )

        self.assertEqual(
            region.inequalities_left_side,
            [
                [-1.0, 0.0],
                [0.0, -1.0],
                [-1.0, 1.0],
            ],
        )
        self.assertEqual(region.inequalities_right_side, [0.0, 0.0, 0.0])
        self.assertEqual(region.equalities_left_side, [[1.0, 1.0]])
        self.assertTrue(region.is_feasible())

    def test_build_optimality_region_rejects_wrong_weight_space_dimension(self) -> None:
        alternatives = AlternativenMatrix(entries=[[1.0, 0.0]])
        weight_space = build_weight_space(goal_count=3, answered_queries=[])

        with self.assertRaisesRegex(ValueError, "same number of variables"):
            build_optimality_region(
                alternatives=alternatives,
                weight_space=weight_space,
                alternative_index=0,
            )


if __name__ == "__main__":
    unittest.main()
