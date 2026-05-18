from __future__ import annotations

import unittest

from multistep.src.alternative_utility import (
    build_utility_difference_coefficients,
    compute_utility_value,
)
from multistep.src.models import AlternativenMatrix


class AlternativeUtilityTests(unittest.TestCase):
    def test_compute_utility_value(self) -> None:
        alternatives = AlternativenMatrix(entries=[[0.2, 0.8]])

        utility_value = compute_utility_value(
            alternatives=alternatives,
            alternative_index=0,
            weights=[0.25, 0.75],
        )

        self.assertAlmostEqual(utility_value, 0.65)

    def test_compute_utility_value_rejects_wrong_weight_dimension(self) -> None:
        alternatives = AlternativenMatrix(entries=[[1.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "weights must match"):
            compute_utility_value(
                alternatives=alternatives,
                alternative_index=0,
                weights=[1.0, 0.0, 0.0],
            )

    def test_build_utility_difference_coefficients(self) -> None:
        alternatives = AlternativenMatrix(
            entries=[
                [1.0, 0.2],
                [0.4, 0.8],
            ],
        )

        coefficients = build_utility_difference_coefficients(
            alternatives=alternatives,
            minuend_index=1,
            subtrahend_index=0,
        )

        self.assertEqual(coefficients, [-0.6, 0.6000000000000001])


if __name__ == "__main__":
    unittest.main()
